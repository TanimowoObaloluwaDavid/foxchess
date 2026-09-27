"""Position state: bitboards, make/unmake, FEN and the rules of chess.

Design notes
------------
*Hybrid representation.* Attacks are computed on bitboards (whole-board set
operations), while piece lookups go through a 64-entry mailbox — in CPython the
array index beats a dict lookup on every touched square. Both stay in sync
because :meth:`Board._remove_at`, :meth:`Board._move_piece` and
:meth:`Board._place` are the only mutators.

*Legality by verification.* Generation produces pseudo-legal moves and
:meth:`Board.legal_moves` confirms each by making it and asking "is my own king
attacked?". That is slower than a pin-aware generator, but it is the only
approach that handles en-passant discovered checks without a thicket of special
cases — and perft can prove it is right.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from . import zobrist
from .attacks import (
    KING_ATTACKS,
    KNIGHT_ATTACKS,
    PAWN_ATTACKERS,
    PAWN_ATTACKS,
    PAWN_DOUBLE_PUSH,
    PAWN_DOUBLE_PUSH_FROM,
    PAWN_PROMOTION_PUSH,
    PAWN_PUSH,
    bishop_attacks,
    queen_attacks,
    rook_attacks,
)
from .move import (
    CAPTURE,
    CASTLE_KINGSIDE,
    CASTLE_QUEENSIDE,
    DOUBLE_PUSH,
    EN_PASSANT,
    MOVE_NONE,
    PROMOTION_BISHOP,
    PROMOTION_BISHOP_CAPTURE,
    PROMOTION_KNIGHT,
    PROMOTION_KNIGHT_CAPTURE,
    PROMOTION_QUEEN,
    PROMOTION_QUEEN_CAPTURE,
    PROMOTION_ROOK,
    PROMOTION_ROOK_CAPTURE,
    QUIET,
    decode_flag,
    decode_from,
    decode_to,
    encode_int,
    from_uci,
    to_uci,
)
from .types import (
    BLACK_KING_SQUARE,
    BLACK_ROOK_KING_SQUARE,
    BLACK_ROOK_QUEEN_SQUARE,
    BLACK_ROOK_TO_KING_SQUARE,
    BLACK_ROOK_TO_QUEEN_SQUARE,
    CASTLING_EMPTY,
    CASTLING_KING_PATH,
    CASTLING_SQUARES,
    DARK_SQUARES,
    EMPTY,
    FILE_COUNT,
    LIGHT_SQUARES,
    PIECE_BY_LETTER,
    SQUARE_COUNT,
    WHITE_KING_SQUARE,
    WHITE_ROOK_KING_SQUARE,
    WHITE_ROOK_QUEEN_SQUARE,
    WHITE_ROOK_TO_KING_SQUARE,
    WHITE_ROOK_TO_QUEEN_SQUARE,
    CastlingRight,
    Color,
    PieceType,
    color_of,
    make_piece,
    name_of,
    piece_letter,
    piece_type_of,
    popcount,
    square_from_name,
)

STARTING_FEN: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

PROMOTION_FLAGS: frozenset[int] = frozenset(
    {
        PROMOTION_KNIGHT,
        PROMOTION_BISHOP,
        PROMOTION_ROOK,
        PROMOTION_QUEEN,
        PROMOTION_KNIGHT_CAPTURE,
        PROMOTION_BISHOP_CAPTURE,
        PROMOTION_ROOK_CAPTURE,
        PROMOTION_QUEEN_CAPTURE,
    }
)

PROMOTION_BY_FLAG: dict[int, PieceType] = {
    PROMOTION_KNIGHT: PieceType.KNIGHT,
    PROMOTION_BISHOP: PieceType.BISHOP,
    PROMOTION_ROOK: PieceType.ROOK,
    PROMOTION_QUEEN: PieceType.QUEEN,
    PROMOTION_KNIGHT_CAPTURE: PieceType.KNIGHT,
    PROMOTION_BISHOP_CAPTURE: PieceType.BISHOP,
    PROMOTION_ROOK_CAPTURE: PieceType.ROOK,
    PROMOTION_QUEEN_CAPTURE: PieceType.QUEEN,
}

#: Promotion choices in the order generators emit them: queen first because it is by
#: far the most common, under-promotions last.
PROMOTION_CHOICES: tuple[int, ...] = (
    PROMOTION_QUEEN,
    PROMOTION_ROOK,
    PROMOTION_BISHOP,
    PROMOTION_KNIGHT,
)
PROMOTION_CAPTURE_CHOICES: tuple[int, ...] = (
    PROMOTION_QUEEN_CAPTURE,
    PROMOTION_ROOK_CAPTURE,
    PROMOTION_BISHOP_CAPTURE,
    PROMOTION_KNIGHT_CAPTURE,
)

#: Flags that make a move "noisy" — the only moves quiescence search looks at.
NOISY_FLAGS: frozenset[int] = frozenset(
    {CAPTURE, EN_PASSANT, PROMOTION_QUEEN, PROMOTION_QUEEN_CAPTURE}
)

#: Castling bits that survive a piece leaving or landing on each square. Applying
#: this to the ``from`` and ``to`` squares of a move covers "the king moved", "the
#: rook moved" and "a rook was captured" in a single lookup.
_CASTLE_RIGHTS_MASK: dict[int, int] = {
    0: ~int(CastlingRight.WHITE_QUEENSIDE),  # a1
    7: ~int(CastlingRight.WHITE_KINGSIDE),  # h1
    4: ~(int(CastlingRight.WHITE_KINGSIDE) | int(CastlingRight.WHITE_QUEENSIDE)),  # e1
    56: ~int(CastlingRight.BLACK_QUEENSIDE),  # a8
    63: ~int(CastlingRight.BLACK_KINGSIDE),  # h8
    60: ~(int(CastlingRight.BLACK_KINGSIDE) | int(CastlingRight.BLACK_QUEENSIDE)),  # e8
}

_KING_SQUARE = (WHITE_KING_SQUARE, BLACK_KING_SQUARE)
_ROOK_FROM_KING = (WHITE_ROOK_KING_SQUARE, BLACK_ROOK_KING_SQUARE)
_ROOK_FROM_QUEEN = (WHITE_ROOK_QUEEN_SQUARE, BLACK_ROOK_QUEEN_SQUARE)
_ROOK_TO_KING = (WHITE_ROOK_TO_KING_SQUARE, BLACK_ROOK_TO_KING_SQUARE)
_ROOK_TO_QUEEN = (WHITE_ROOK_TO_QUEEN_SQUARE, BLACK_ROOK_TO_QUEEN_SQUARE)


class IllegalMoveError(ValueError):
    """Raised when a move is not legal in the current position."""


class InvalidFenError(ValueError):
    """Raised when a FEN string cannot be parsed."""


@dataclass(frozen=True, slots=True)
class Undo:
    """Everything needed to restore a position after a single ply."""

    move: int
    captured: int
    castling: int
    ep_square: int
    halfmove_clock: int
    fullmove_number: int
    key: int
    king_squares: tuple[int, int]


class Board:
    """A mutable chess position with O(1) make/unmake and incremental Zobrist."""

    __slots__ = (
        "_undo",
        "bitboards",
        "castling",
        "ep_square",
        "fullmove_number",
        "halfmove_clock",
        "key",
        "king_squares",
        "mailbox",
        "occupancy",
        "side",
    )

    def __init__(self, fen: str = STARTING_FEN) -> None:
        self.bitboards: list[int] = [0] * 16
        self.mailbox: list[int] = [EMPTY] * SQUARE_COUNT
        self.occupancy: list[int] = [0, 0, 0]  # [all, white, black]
        self.side: Color = Color.WHITE
        self.castling: int = 0
        self.ep_square: int = -1
        self.halfmove_clock: int = 0
        self.fullmove_number: int = 1
        self.key: int = 0
        self.king_squares: list[int] = [-1, -1]
        self._undo: list[Undo] = []
        self.load(fen)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def load(self, fen: str) -> None:
        """Reset the position from a FEN string."""
        self._set_from_fen(fen)

    def copy(self) -> Board:
        """A detached copy, including the undo stack."""
        clone = Board.__new__(Board)
        clone.bitboards = self.bitboards[:]
        clone.mailbox = self.mailbox[:]
        clone.occupancy = self.occupancy[:]
        clone.side = self.side
        clone.castling = self.castling
        clone.ep_square = self.ep_square
        clone.halfmove_clock = self.halfmove_clock
        clone.fullmove_number = self.fullmove_number
        clone.key = self.key
        clone.king_squares = self.king_squares[:]
        clone._undo = list(self._undo)
        return clone

    def push_uci(self, text: str) -> None:
        """Apply a move given in long algebraic notation, validating legality."""
        raw = from_uci(text, self.legal_moves())
        self.make_move(raw)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def occupancy_all(self) -> int:
        return self.occupancy[0]

    @property
    def side_bb(self) -> int:
        """Occupancy of the side to move. ``occupancy`` is indexed ``[all, white, black]``."""
        return self.occupancy[1 + int(self.side)]

    @property
    def ply(self) -> int:
        return len(self._undo)

    def piece_at(self, sq: int) -> int:
        return self.mailbox[sq]

    def piece_bb(self, piece_type: PieceType, color: Color) -> int:
        return self.bitboards[make_piece(piece_type, color)]

    def count(self, piece_type: PieceType, color: Color) -> int:
        return popcount(self.piece_bb(piece_type, color))

    def has_castling_right(self, right: CastlingRight) -> bool:
        return bool(self.castling & int(right))

    def is_attacked(self, sq: int, by: Color) -> bool:
        """Is ``sq`` attacked by any piece of colour ``by``?"""
        by_side = int(by)
        pawns = self.bitboards[make_piece(PieceType.PAWN, by)]
        if pawns and PAWN_ATTACKERS[by_side][sq] & pawns:
            return True

        knights = self.bitboards[make_piece(PieceType.KNIGHT, by)]
        if knights and KNIGHT_ATTACKS[sq] & knights:
            return True

        kings = self.bitboards[make_piece(PieceType.KING, by)]
        if kings and KING_ATTACKS[sq] & kings:
            return True

        occupancy = self.occupancy[0]
        if not occupancy:
            return False

        bishops = self.bitboards[make_piece(PieceType.BISHOP, by)]
        queens = self.bitboards[make_piece(PieceType.QUEEN, by)]
        if bishops or queens:
            diagonal = bishops | queens
            if bishop_attacks(sq, occupancy) & diagonal:
                return True

        rooks = self.bitboards[make_piece(PieceType.ROOK, by)]
        if rooks or queens:
            straight = rooks | queens
            if rook_attacks(sq, occupancy) & straight:
                return True

        return False

    def in_check(self, color: Color | None = None) -> bool:
        who = self.side if color is None else color
        king = self.king_squares[int(who)]
        return king >= 0 and self.is_attacked(king, who.opposite)

    # ------------------------------------------------------------------
    # Mutators — the only places bitboards and the mailbox are touched
    # ------------------------------------------------------------------

    def _place(self, piece: int, sq: int) -> None:
        self.mailbox[sq] = piece
        self.bitboards[piece] |= 1 << sq
        self.occupancy[0] |= 1 << sq
        self.occupancy[1 + int(color_of(piece))] |= 1 << sq
        if piece_type_of(piece) is PieceType.KING:
            self.king_squares[int(color_of(piece))] = sq

    def _remove_at(self, sq: int) -> int:
        """Clear whatever stands on ``sq`` and return it (``EMPTY`` if nothing)."""
        piece = self.mailbox[sq]
        if piece == EMPTY:
            return EMPTY
        self.mailbox[sq] = EMPTY
        self.bitboards[piece] &= ~(1 << sq)
        self.occupancy[0] &= ~(1 << sq)
        self.occupancy[1 + int(color_of(piece))] &= ~(1 << sq)
        if piece_type_of(piece) is PieceType.KING:
            self.king_squares[int(color_of(piece))] = -1
        return piece

    def _move_piece(self, piece: int, from_sq: int, to_sq: int) -> None:
        bit_from, bit_to = 1 << from_sq, 1 << to_sq
        self.mailbox[from_sq] = EMPTY
        self.mailbox[to_sq] = piece
        self.bitboards[piece] = (self.bitboards[piece] & ~bit_from) | bit_to
        self.occupancy[0] = (self.occupancy[0] & ~bit_from) | bit_to
        color = 1 + int(color_of(piece))
        self.occupancy[color] = (self.occupancy[color] & ~bit_from) | bit_to
        if piece_type_of(piece) is PieceType.KING:
            self.king_squares[color - 1] = to_sq

    # ------------------------------------------------------------------
    # Make / unmake
    # ------------------------------------------------------------------

    def make_move(self, move: int, *, validate: bool = False) -> None:
        """Apply ``move``, which is assumed legal.

        ``validate=True`` re-checks legality first, at the cost of generating and
        trying the whole move list. It exists for the API and tests, not the search.
        """
        if validate and move not in self.legal_moves():
            raise IllegalMoveError(f"{to_uci(move)} is not legal in {self.fen()}")

        from_sq = decode_from(move)
        to_sq = decode_to(move)
        flag = decode_flag(move)
        piece = self.mailbox[from_sq]
        color = int(self.side)
        is_pawn = piece_type_of(piece) is PieceType.PAWN
        victim_sq = to_sq - 8 if color == 0 else to_sq + 8
        had_capture = flag == EN_PASSANT or self.mailbox[to_sq] != EMPTY
        # For en passant the victim is not on the destination square, so the undo
        # record has to name it explicitly rather than reading the mailbox.
        captured = (
            self.mailbox[victim_sq] if flag == EN_PASSANT else self.mailbox[to_sq]
        )

        self._undo.append(
            Undo(
                move=move,
                captured=captured,
                castling=self.castling,
                ep_square=self.ep_square,
                halfmove_clock=self.halfmove_clock,
                fullmove_number=self.fullmove_number,
                key=self.key,
                king_squares=(self.king_squares[0], self.king_squares[1]),
            )
        )

        # Retire the outgoing en-passant contribution before anything moves.
        self.key ^= zobrist.ep_key(self._ep_hash_target())

        # 1. Remove the captured piece, if any.
        if flag == EN_PASSANT:
            victim = self._remove_at(victim_sq)
            self.key ^= zobrist.piece_key(victim, victim_sq)
        elif captured != EMPTY:
            victim = self._remove_at(to_sq)
            self.key ^= zobrist.piece_key(victim, to_sq)

        # 2. Place the moving piece (and rook, when castling).
        if flag in (CASTLE_KINGSIDE, CASTLE_QUEENSIDE):
            kingside = flag == CASTLE_KINGSIDE
            self._move_piece(piece, from_sq, to_sq)
            rook = make_piece(PieceType.ROOK, color)
            rook_from = _ROOK_FROM_KING[color] if kingside else _ROOK_FROM_QUEEN[color]
            rook_to = _ROOK_TO_KING[color] if kingside else _ROOK_TO_QUEEN[color]
            self._move_piece(rook, rook_from, rook_to)
            self.key ^= zobrist.piece_key(piece, from_sq) ^ zobrist.piece_key(piece, to_sq)
            self.key ^= zobrist.piece_key(rook, rook_from) ^ zobrist.piece_key(rook, rook_to)
        elif flag in PROMOTION_FLAGS:
            # The pawn never reaches the destination as a pawn, so it must not be
            # hashed out of it. Step 1 already cleared the destination, if there
            # was a capture, so this is just "lift the pawn, drop the new piece".
            promoted = make_piece(PROMOTION_BY_FLAG[flag], Color(color))
            self._remove_at(from_sq)
            self._place(promoted, to_sq)
            self.key ^= zobrist.piece_key(piece, from_sq) ^ zobrist.piece_key(promoted, to_sq)
        else:
            self._move_piece(piece, from_sq, to_sq)
            self.key ^= zobrist.piece_key(piece, from_sq) ^ zobrist.piece_key(piece, to_sq)

        # 3. Castling rights: the mask handles king moves, rook moves and rook captures.
        if self.castling:
            updated = self.castling
            updated &= _CASTLE_RIGHTS_MASK.get(from_sq, 0xFFFF)
            updated &= _CASTLE_RIGHTS_MASK.get(to_sq, 0xFFFF)
            updated &= 0xF
            if updated != self.castling:
                self.key ^= zobrist.castling_key(self.castling) ^ zobrist.castling_key(updated)
                self.castling = updated

        # 4. Clocks, side to move, and the incoming en-passant contribution.
        self.ep_square = (from_sq + to_sq) // 2 if flag == DOUBLE_PUSH else -1
        self.halfmove_clock = 0 if (is_pawn or had_capture) else self.halfmove_clock + 1

        self.side = Color(1 - color)
        if color == 1:
            self.fullmove_number += 1

        self.key ^= zobrist.ep_key(self._ep_hash_target())
        self.key ^= zobrist.SIDE_KEY

    def unmake_move(self) -> int:
        """Undo the last ply and return the move that was applied."""
        undo = self._undo.pop()
        move = undo.move
        from_sq = decode_from(move)
        to_sq = decode_to(move)
        flag = decode_flag(move)

        self.side = Color(1 - int(self.side))
        color = int(self.side)
        self.fullmove_number = undo.fullmove_number

        if flag in PROMOTION_FLAGS:
            # Un-promote: the promoted piece leaves, a pawn takes its place, and
            # that pawn walks back to where it came from.
            self._remove_at(to_sq)
            pawn = make_piece(PieceType.PAWN, Color(color))
            self._place(pawn, to_sq)
            self._move_piece(pawn, to_sq, from_sq)
        else:
            self._move_piece(self.mailbox[to_sq], to_sq, from_sq)

        if flag == CASTLE_KINGSIDE:
            self._move_piece(
                make_piece(PieceType.ROOK, Color(color)),
                _ROOK_TO_KING[color],
                _ROOK_FROM_KING[color],
            )
        elif flag == CASTLE_QUEENSIDE:
            self._move_piece(
                make_piece(PieceType.ROOK, Color(color)),
                _ROOK_TO_QUEEN[color],
                _ROOK_FROM_QUEEN[color],
            )

        if flag == EN_PASSANT:
            victim_sq = to_sq - 8 if color == 0 else to_sq + 8
            self._place(undo.captured, victim_sq)
        elif undo.captured != EMPTY:
            self._place(undo.captured, to_sq)
        self.castling = undo.castling
        self.ep_square = undo.ep_square
        self.halfmove_clock = undo.halfmove_clock
        self.key = undo.key
        self.king_squares[0], self.king_squares[1] = undo.king_squares
        return move

    def make_null_move(self) -> None:
        """Pass the turn. Used by null-move pruning; never a legal move."""
        self._undo.append(
            Undo(
                move=MOVE_NONE,
                captured=EMPTY,
                castling=self.castling,
                ep_square=self.ep_square,
                halfmove_clock=self.halfmove_clock,
                fullmove_number=self.fullmove_number,
                key=self.key,
                king_squares=(self.king_squares[0], self.king_squares[1]),
            )
        )
        self.key ^= zobrist.ep_key(self._ep_hash_target())
        self.ep_square = -1
        self.halfmove_clock += 1
        self.side = Color(1 - int(self.side))
        if self.side is Color.BLACK:
            self.fullmove_number += 1
        self.key ^= zobrist.SIDE_KEY

    def unmake_null_move(self) -> None:
        undo = self._undo.pop()
        self.side = Color(1 - int(self.side))
        self.fullmove_number = undo.fullmove_number
        self.castling = undo.castling
        self.ep_square = undo.ep_square
        self.halfmove_clock = undo.halfmove_clock
        self.key = undo.key
        self.king_squares[0], self.king_squares[1] = undo.king_squares

    # ------------------------------------------------------------------
    # Move generation
    # ------------------------------------------------------------------

    def generate_pseudo_legal(self) -> list[int]:
        """Every move that keeps the piece's own movement rules, ignoring pins."""
        color = int(self.side)
        occupancy = self.occupancy[0]
        side_bb = self.occupancy[1 + color]
        enemies = self.occupancy[1 + (1 - color)]
        quiet_targets = ~occupancy
        moves: list[int] = []

        pawns = self.bitboards[make_piece(PieceType.PAWN, color)]
        if pawns:
            moves.extend(self._pawn_moves(pawns, color, quiet_targets, enemies))

        leapers: tuple[tuple[PieceType, list[int] | None], ...] = (
            (PieceType.KNIGHT, KNIGHT_ATTACKS),
            (PieceType.KING, KING_ATTACKS),
            (PieceType.BISHOP, None),
            (PieceType.ROOK, None),
            (PieceType.QUEEN, None),
        )
        for piece_type, table in leapers:
            pieces = self.bitboards[make_piece(piece_type, color)]
            if not pieces:
                continue
            while pieces:
                sq = (pieces & -pieces).bit_length() - 1
                pieces &= pieces - 1
                if table is not None:
                    attacks = table[sq]
                elif piece_type is PieceType.BISHOP:
                    attacks = bishop_attacks(sq, occupancy)
                elif piece_type is PieceType.ROOK:
                    attacks = rook_attacks(sq, occupancy)
                else:
                    attacks = queen_attacks(sq, occupancy)
                attacks &= ~side_bb
                while attacks:
                    target = (attacks & -attacks).bit_length() - 1
                    attacks &= attacks - 1
                    moves.append(
                        encode_int(sq, target, CAPTURE if enemies >> target & 1 else QUIET)
                    )

        castle = self._castling_moves(color)
        if castle:
            moves.extend(castle)
        return moves

    def _pawn_moves(
        self, pawns: int, color: int, quiet_targets: int, enemies: int
    ) -> Iterator[int]:
        push = PAWN_PUSH[color]
        double_push = PAWN_DOUBLE_PUSH[color]
        double_from = PAWN_DOUBLE_PUSH_FROM[color]
        promotion = PAWN_PROMOTION_PUSH[color]
        attacks_table = PAWN_ATTACKS[color]
        ep = self.ep_square

        bb = pawns
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1

            one = push[sq]
            if one >= 0 and quiet_targets >> one & 1:
                if promotion >> one & 1:
                    for flag in PROMOTION_CHOICES:
                        yield encode_int(sq, one, flag)
                else:
                    yield encode_int(sq, one, QUIET)
                    if double_from >> sq & 1:
                        two = double_push[sq]
                        if two >= 0 and quiet_targets >> two & 1:
                            yield encode_int(sq, two, DOUBLE_PUSH)

            attacks = attacks_table[sq]
            while attacks:
                target = (attacks & -attacks).bit_length() - 1
                attacks &= attacks - 1
                if enemies >> target & 1:
                    if promotion >> target & 1:
                        for flag in PROMOTION_CAPTURE_CHOICES:
                            yield encode_int(sq, target, flag)
                    else:
                        yield encode_int(sq, target, CAPTURE)
                elif target == ep and ep >= 0:
                    yield encode_int(sq, target, EN_PASSANT)

    def _castling_moves(self, color: int) -> list[int]:
        """Castling is only legal out of check, through empty, unattacked squares."""
        side = Color(color)
        king_square = _KING_SQUARE[color]
        if self.mailbox[king_square] != make_piece(PieceType.KING, side):
            return []
        if self.is_attacked(king_square, side.opposite):
            return []

        moves: list[int] = []
        for _color, right, king_from, king_to, rook_from, _rook_to in CASTLING_SQUARES:
            if not self.castling & int(right):
                continue
            if self.mailbox[rook_from] != make_piece(PieceType.ROOK, side):
                continue
            if self.occupancy[0] & CASTLING_EMPTY[side][right]:
                continue
            path = CASTLING_KING_PATH[side][right]
            safe = True
            while path:
                sq = (path & -path).bit_length() - 1
                path &= path - 1
                if self.is_attacked(sq, side.opposite):
                    safe = False
                    break
            if not safe:
                continue
            moves.append(
                encode_int(
                    king_from,
                    king_to,
                    CASTLE_KINGSIDE if king_to > king_from else CASTLE_QUEENSIDE,
                )
            )
        return moves

    def legal_moves(self) -> list[int]:
        """Fully legal moves in the current position."""
        legal: list[int] = []
        append = legal.append
        for move in self.generate_pseudo_legal():
            self.make_move(move)
            if not self.in_check(self.side.opposite):
                append(move)
            self.unmake_move()
        return legal

    def legal_moves_iter(self) -> Iterator[int]:
        """Generator form of :meth:`legal_moves`, for callers that break early."""
        for move in self.generate_pseudo_legal():
            self.make_move(move)
            if not self.in_check(self.side.opposite):
                yield move
            self.unmake_move()

    def is_legal(self, move: int) -> bool:
        return move in self.legal_moves()

    def captures(self) -> list[int]:
        """Captures plus queen promotions — the moves quiescence search cares about."""
        return [m for m in self.legal_moves() if decode_flag(m) in NOISY_FLAGS]

    def is_capture(self, move: int) -> bool:
        return decode_flag(move) in NOISY_FLAGS

    # ------------------------------------------------------------------
    # Terminal conditions
    # ------------------------------------------------------------------

    def is_checkmate(self) -> bool:
        return self.in_check() and not self.legal_moves()

    def is_stalemate(self) -> bool:
        return not self.in_check() and not self.legal_moves()

    def is_fifty_move(self) -> bool:
        """A player may claim a draw after 50 moves by each side (100 plies)."""
        return self.halfmove_clock >= 100

    def is_seventy_five_move(self) -> bool:
        """Automatic draw at 75 moves by each side (150 plies)."""
        return self.halfmove_clock >= 150

    def is_insufficient_material(self) -> bool:
        """Dead positions, where no legal sequence can end in checkmate."""
        return (
            self.has_insufficient_material(Color.WHITE)
            and self.has_insufficient_material(Color.BLACK)
        )

    def has_insufficient_material(self, color: Color) -> bool:
        """Whether ``color`` cannot win, even with the opponent's pieces helping.

        "With help" is the whole subtlety, and it is what makes a lone knight
        interesting: K+N against a lone king is dead, because the bare king has
        nothing to hold a flight square against, but K+N against K+N is not -- the
        enemy knight can block one and the mate goes through. The same applies to a
        lone bishop, which needs either a second colour complex or some other
        piece to work with. Two knights, or a knight and anything else, are always
        enough, since those combinations can mate unaided.

        This is the same rule python-chess applies, which keeps the differential
        tests and the published perft counts on the same page.
        """
        if (
            self.piece_bb(PieceType.PAWN, color)
            or self.piece_bb(PieceType.ROOK, color)
            or self.piece_bb(PieceType.QUEEN, color)
        ):
            return False

        opponent = Color.BLACK if color is Color.WHITE else Color.WHITE
        knights = self.piece_bb(PieceType.KNIGHT, color)

        if knights:
            # Anything beyond a lone knight beside the king can mate unaided.
            if popcount(self.occupancy[1 + int(color)]) > 2:
                return False
            # Otherwise the enemy needs a piece that is neither king nor queen to
            # hold a square for us.
            return not (
                self.piece_bb(PieceType.PAWN, opponent)
                | self.piece_bb(PieceType.KNIGHT, opponent)
                | self.piece_bb(PieceType.BISHOP, opponent)
                | self.piece_bb(PieceType.ROOK, opponent)
            )

        if self.piece_bb(PieceType.BISHOP, color):
            bishops = self.piece_bb(PieceType.BISHOP, Color.WHITE) | self.piece_bb(
                PieceType.BISHOP, Color.BLACK
            )
            # Dead only when every bishop on the board is on the same complex: a
            # mate needs both, and the other has to come from a pawn, a knight, or
            # an enemy bishop.
            one_complex = not (bishops & DARK_SQUARES) or not (bishops & LIGHT_SQUARES)
            if not one_complex:
                return False
            return not (
                self.piece_bb(PieceType.PAWN, Color.WHITE)
                | self.piece_bb(PieceType.PAWN, Color.BLACK)
                | self.piece_bb(PieceType.KNIGHT, Color.WHITE)
                | self.piece_bb(PieceType.KNIGHT, Color.BLACK)
            )

        return True

    def is_insufficient_material_relaxed(self) -> bool:
        """Also treats K+two minors against a lone king as drawn.

        Engines use this to avoid grinding out a fifty-move draw in a position
        that cannot be won anyway.
        """
        if self.is_insufficient_material():
            return True
        for color in (Color.WHITE, Color.BLACK):
            if self.piece_bb(PieceType.PAWN, color):
                return False
            heavy = self.piece_bb(PieceType.ROOK, color) | self.piece_bb(
                PieceType.QUEEN, color
            )
            if heavy:
                return False
            if popcount(self.piece_bb(PieceType.KING, color)) != 1:
                return True
        return True

    def is_draw(self) -> bool:
        """Is the game drawn?

        Checkmate counts: under the laws of chess a mated game is over and drawn,
        because neither side can go on. Leaving it out here made ``is_draw()``
        contradict ``result()``, which reports ``1-0`` or ``0-1`` for a mate.
        """
        return (
            self.is_checkmate()
            or self.is_stalemate()
            or self.is_fifty_move()
            or self.is_insufficient_material()
        )

    def is_game_over(self) -> bool:
        return self.is_draw()

    def result(self) -> str:
        """Result from White's point of view, ``'*'`` while the game runs on."""
        if self.is_checkmate():
            return "0-1" if self.side is Color.WHITE else "1-0"
        if self.is_draw():
            return "1/2-1/2"
        return "*"

    # ------------------------------------------------------------------
    # Zobrist
    # ------------------------------------------------------------------

    def _ep_hash_target(self) -> int:
        """The en-passant square, but only when a capture is actually available.

        Without this filter, ``1. e4`` and a position reached without a double
        pawn push would hash differently despite being the same position, and the
        transposition table would miss hits it should have.

        ``PAWN_ATTACKERS[color][sq]`` holds the squares *from which* a pawn of
        that colour attacks ``sq``, so the pawn that could make the capture is the
        one belonging to the side to move, and the pawn that would be taken is the
        opponent's. Intersecting the wrong one of the two makes this return -1
        always, which silently disables the en-passant half of the key.
        """
        if self.ep_square < 0:
            return -1
        capturer = make_piece(PieceType.PAWN, self.side)
        if PAWN_ATTACKERS[int(self.side)][self.ep_square] & self.bitboards[capturer]:
            return self.ep_square
        return -1

    def compute_key(self) -> int:
        """Hash from scratch, ignoring the incrementally maintained ``key``."""
        key = 0
        for piece in range(1, 16):
            bb = self.bitboards[piece]
            table = zobrist.PIECE_KEYS[piece]
            while bb:
                sq = (bb & -bb).bit_length() - 1
                key ^= table[sq]
                bb &= bb - 1
        key ^= zobrist.castling_key(self.castling)
        key ^= zobrist.ep_key(self._ep_hash_target())
        if self.side is Color.BLACK:
            key ^= zobrist.SIDE_KEY
        return key

    def verify_key(self) -> bool:
        """Invariant: the incremental hash still matches a fresh computation."""
        return self.key == self.compute_key()

    # ------------------------------------------------------------------
    # FEN
    # ------------------------------------------------------------------

    def fen(self) -> str:
        rows: list[str] = []
        for rank in range(7, -1, -1):
            row: list[str] = []
            gap = 0
            base = rank * FILE_COUNT
            for file_index in range(FILE_COUNT):
                piece = self.mailbox[base + file_index]
                if piece == EMPTY:
                    gap += 1
                    continue
                if gap:
                    row.append(str(gap))
                    gap = 0
                row.append(piece_letter(piece))
            if gap:
                row.append(str(gap))
            rows.append("".join(row))
        return (
            f"{'/'.join(rows)} {self.side.fen_letter} {self.castling_fen()} "
            f"{self.ep_fen()} {self.halfmove_clock} {self.fullmove_number}"
        )

    def castling_fen(self) -> str:
        if not self.castling:
            return "-"
        return "".join(
            char
            for right, char in (
                (CastlingRight.WHITE_KINGSIDE, "K"),
                (CastlingRight.WHITE_QUEENSIDE, "Q"),
                (CastlingRight.BLACK_KINGSIDE, "k"),
                (CastlingRight.BLACK_QUEENSIDE, "q"),
            )
            if self.castling & int(right)
        )

    def ep_fen(self) -> str:
        """The en-passant field, or ``-`` when no capture is actually available.

        A double pawn push does not by itself make a position different: after
        ``1. e4`` with no black pawn on d4 or f4, the ep square records nothing
        that a player can act on. Writing it anyway would make two FENs differ for
        one position, so transposition and repetition checks built on FEN text would
        disagree with :attr:`key`, which already ignores an unusable ep square.
        """
        target = self._ep_hash_target()
        return name_of(target) if target >= 0 else "-"

    def _set_from_fen(self, fen: str) -> None:
        fields = fen.split()
        if len(fields) < 2:
            raise InvalidFenError(f"FEN needs at least 2 fields: {fen!r}")

        self._set_placement(fields[0])

        side_field = fields[1].lower()
        if side_field not in ("w", "b"):
            raise InvalidFenError(f"invalid side to move: {fields[1]!r}")
        self.side = Color.WHITE if side_field == "w" else Color.BLACK

        self.castling = 0
        castling_field = fields[2] if len(fields) > 2 else "-"
        for char in castling_field:
            if char == "-":
                continue
            right = _CASTLE_CHARS.get(char)
            if right is None:
                raise InvalidFenError(f"invalid castling rights: {castling_field!r}")
            self.castling |= int(right)

        ep_field = fields[3] if len(fields) > 3 else "-"
        if ep_field == "-":
            self.ep_square = -1
        else:
            try:
                self.ep_square = square_from_name(ep_field)
            except ValueError as exc:
                raise InvalidFenError(str(exc)) from exc

        self.halfmove_clock = _int_field(fields, 4, "halfmove clock", default=0)
        self.fullmove_number = _int_field(fields, 5, "fullmove number", default=1)
        self._undo.clear()
        self.key = self.compute_key()

    def _set_placement(self, placement: str) -> None:
        self.bitboards = [0] * 16
        self.mailbox = [EMPTY] * SQUARE_COUNT
        self.occupancy = [0, 0, 0]
        self.king_squares = [-1, -1]

        rows = placement.split("/")
        if len(rows) != 8:
            raise InvalidFenError(f"FEN placement needs 8 ranks, got {len(rows)}")
        for rank_index, row in enumerate(rows):
            rank = 7 - rank_index
            file_index = 0
            for char in row:
                if char.isdigit():
                    if char == "0":
                        raise InvalidFenError("FEN run lengths cannot be zero")
                    file_index += int(char)
                    continue
                piece_type = PIECE_BY_LETTER.get(char)
                if piece_type is None:
                    raise InvalidFenError(f"unknown piece letter: {char!r}")
                if not 0 <= file_index < 8:
                    raise InvalidFenError(f"rank {rank + 1} has more than 8 files: {row!r}")
                color = Color.WHITE if char.isupper() else Color.BLACK
                self._place(make_piece(piece_type, color), rank * 8 + file_index)
                file_index += 1
            if file_index != 8:
                raise InvalidFenError(f"rank {rank + 1} covers {file_index} files: {row!r}")

    # ------------------------------------------------------------------
    # Debug output
    # ------------------------------------------------------------------

    def ascii(self) -> str:
        lines = ["  +------------------------+", "  |  a  b  c  d  e  f  g  h  |"]
        for rank in range(7, -1, -1):
            cells = (
                piece_letter(self.mailbox[rank * 8 + f]) if self.mailbox[rank * 8 + f] else "."
                for f in range(8)
            )
            lines.append(f"{rank + 1} | " + "  ".join(cells) + " |")
        lines.append("  +------------------------+")
        status = f"{self.side.name_lower} to move   castling: {self.castling_fen()}"
        lines.append(f"    {status}")
        if self.in_check():
            lines.append("    king in check")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.fen()

    def __repr__(self) -> str:
        return f"Board({self.fen()!r})"


_CASTLE_CHARS: dict[str, CastlingRight] = {
    "K": CastlingRight.WHITE_KINGSIDE,
    "Q": CastlingRight.WHITE_QUEENSIDE,
    "k": CastlingRight.BLACK_KINGSIDE,
    "q": CastlingRight.BLACK_QUEENSIDE,
}


def _int_field(fields: list[str], index: int, label: str, *, default: int) -> int:
    if index >= len(fields):
        return default
    try:
        value = int(fields[index])
    except ValueError as exc:
        raise InvalidFenError(f"invalid {label}: {fields[index]!r}") from exc
    if value < 0:
        raise InvalidFenError(f"{label} cannot be negative: {value}")
    return value

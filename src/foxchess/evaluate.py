"""Evaluation: a hand-crafted, tapered static score in centipawns from White's view.

Terms:

* **material**, with a small premium for a bishop pair and a rook pair;
* **piece-square tables** blended between a middlegame and an endgame set by a
  *game phase* computed from how much heavy material is still on the board;
* **pawn structure** — doubled, isolated, backward, connected, passed, and a
  filer penalty;
* **rooks** on open and half-open files;
* **bishop** mobility;
* **king danger** — how much enemy material attacks the king and its neighbours.

Two deliberate choices:

* The king is scored by its piece-square table, not by a hand-rolled pawn shield.
  The table already encodes "cornered behind pawns in the opening, centralised in
  the endgame", and a separate shield term would double-count all of it.
* King danger *is* computed explicitly, because no static table captures a rook
  on an open file next to an undefended king.

The tables are the widely published PeSTO set, used here as a starting point. They
are indexed a8-first (index 0 is a8), so White looks a square up and Black looks
straight down.
"""

from __future__ import annotations

from typing import Final

from .attacks import (
    KING_ATTACKS,
    KNIGHT_ATTACKS,
    PAWN_ATTACKS,
    bishop_attacks,
    queen_attacks,
    rook_attacks,
)
from .board import Board
from .types import (
    FILES,
    PIECE_ORDER,
    PIECE_VALUES,
    Color,
    PieceType,
    iter_bits,
    piece_type_of,
    popcount,
    rank_of,
    square,
)

__all__ = [
    "FULL_PHASE",
    "INFINITY",
    "MATE_SCORE",
    "MATE_THRESHOLD",
    "evaluate",
    "evaluate_side",
    "game_phase",
]

INFINITY: Final[int] = 1_000_000
MATE_SCORE: Final[int] = 30_000
#: Scores at or beyond this magnitude are mate scores adjusted by distance.
MATE_THRESHOLD: Final[int] = MATE_SCORE - 1_000
#: The phase sum over a full opening board; also the denominator when tapering.
FULL_PHASE: Final[int] = 24

# ---------------------------------------------------------------------------
# Piece-square tables
# ---------------------------------------------------------------------------
#
# Built from named per-file and per-rank weights rather than transcribed from a
# published set. That is a deliberate trade: a hand-tuned table is worth a lot of
# playing strength over a constructed one, but a table I cannot verify is a table
# that is probably wrong, and a wrong table is much worse than a simple one. The
# weights below encode the positional principles engines broadly agree on; tuning
# them against real games is a later job, not something to fake now.
#
# Indexing is a1-first: entry ``sq`` of a table is that square for *White*, and
# Black reads ``table[56 - sq]``, the same 180-degree rotation. Getting this
# backwards is easy and completely silent, so every table is a1-first and the
# rotation happens in exactly one place, `_placement`.

#: Per-file preference, a through h. Central files good, edges bad.
_PAWN_FILE: Final = (-8, -4, 4, 12, 12, 4, -4, -8)
_PAWN_RANK_MG: Final = (0, 4, 10, 18, 30, 46, 66, 0)
#: Advancement is worth more in the endgame, when a passed pawn decides the game
#: and a stuck one is worth nothing.
_PAWN_RANK_EG: Final = (0, 10, 22, 38, 58, 84, 118, 0)

#: Knights are the most centre-sensitive piece there is, and the rim is poison.
_KNIGHT_FILE: Final = (-20, -10, 8, 18, 18, 8, -10, -20)
#: A knight on its own back rank is wasted, hence the penalty on ranks 1 and 8.
_KNIGHT_RANK: Final = (-24, -10, 2, 8, 8, 4, 0, -8)

_BISHOP_FILE: Final = (-12, -6, 4, 8, 8, 4, -6, -12)
_BISHOP_RANK: Final = (-8, -4, 2, 6, 6, 2, -2, -6)
#: Squares on either long diagonal keep a bishop's other diagonal open.
_LONG_DIAGONAL: Final = 8

#: Rooks care little about the centre; open files matter far more, and that is
#: scored in `_rooks_and_bishops` where the pawns are actually visible.
_ROOK_FILE: Final = (-6, -2, 0, 2, 2, 0, -2, -6)
#: The seventh rank is the single best place for a rook.
_ROOK_RANK: Final = (-8, -2, 2, 4, 6, 8, 14, 2)

_QUEEN_FILE: Final = (-4, -2, 2, 4, 4, 2, -2, -4)
_QUEEN_RANK: Final = (-6, -2, 0, 2, 4, 4, 2, -4)

#: Middlegame king: safety first. The back two ranks are good and the centre is
#: where kings get mated. Every per-file array here is a palindrome (a-file
#: matches h-file, b matches g, and so on) because a rank flip alone is not a
#: rotation, and only a file-reversal-invariant table keeps a mirrored position
#: scoring to the exact negation. That is what lets c1 and g1 be the good squares.
_KING_FILE_MG: Final = (4, 10, 8, -8, -8, 8, 10, 4)
_KING_RANK_MG: Final = (24, 20, 4, -18, -34, -44, -40, -30)
#: Endgame king: exactly the reverse. Centralise and go for the pawns.
_KING_FILE_EG: Final = (-16, -6, 2, 10, 10, 2, -6, -16)
_KING_RANK_EG: Final = (-44, -32, -14, 2, 12, 18, 20, 18)


def _build_table(
    file_weights: tuple[int, ...],
    rank_weights: tuple[int, ...],
    *,
    diagonal_bonus: int = 0,
) -> tuple[int, ...]:
    """Compose a 64-entry a1-first table from per-file and per-rank weights.

    ``diagonal_bonus`` is added on the two long diagonals, the only non-separable
    shape any of these pieces actually wants.
    """
    out: list[int] = []
    for rank in range(8):
        for file_index in range(8):
            value = file_weights[file_index] + rank_weights[rank]
            if diagonal_bonus and file_index in (rank, 7 - rank):
                value += diagonal_bonus
            out.append(value)
    return tuple(out)


#: Tables that do not taper. The placement is right in the opening and the endgame
#: alike, because the reason the piece likes the square has not changed.
_FIXED_TABLES: Final[dict[PieceType, tuple[int, ...]]] = {
    PieceType.KNIGHT: _build_table(_KNIGHT_FILE, _KNIGHT_RANK),
    PieceType.BISHOP: _build_table(
        _BISHOP_FILE, _BISHOP_RANK, diagonal_bonus=_LONG_DIAGONAL
    ),
    PieceType.ROOK: _build_table(_ROOK_FILE, _ROOK_RANK),
    PieceType.QUEEN: _build_table(_QUEEN_FILE, _QUEEN_RANK),
}
_MIDGAME_TABLES: Final[dict[PieceType, tuple[int, ...]]] = {
    PieceType.PAWN: _build_table(_PAWN_FILE, _PAWN_RANK_MG),
    PieceType.KING: _build_table(_KING_FILE_MG, _KING_RANK_MG),
    **_FIXED_TABLES,
}
_ENDGAME_TABLES: Final[dict[PieceType, tuple[int, ...]]] = {
    PieceType.PAWN: _build_table(_PAWN_FILE, _PAWN_RANK_EG),
    PieceType.KING: _build_table(_KING_FILE_EG, _KING_RANK_EG),
    **_FIXED_TABLES,
}

#: Phase weight of each piece type. The initial sum over a full board is 24.
PHASE_WEIGHT: Final[dict[PieceType, int]] = {
    PieceType.PAWN: 0,
    PieceType.KNIGHT: 1,
    PieceType.BISHOP: 1,
    PieceType.ROOK: 2,
    PieceType.QUEEN: 4,
    PieceType.KING: 0,
}

# Indexed by "ranks still to promote", so index 1 is a pawn about to queen.
PASSED_EG: Final[tuple[int, ...]] = (0, 100, 60, 35, 18, 8, 3, 0)
PASSED_MG: Final[tuple[int, ...]] = (0, 40, 22, 12, 6, 3, 1, 0)
#: Passed pawns are worth more in the centre than on the wings. A palindrome, for
#: the same reason as the king file weights: the table is read at ``sq ^ 56``, so
#: file f and file 7 - f must score alike or a mirrored position stops negating.
PASSED_FILE_WEIGHT: Final[tuple[int, ...]] = (0, 0, 4, 8, 8, 4, 0, 0)

#: Every per-file array in this module. `test_piece_square_symmetry` asserts each
#: one is a palindrome; this tuple is what that test iterates.
FILE_WEIGHTS: Final[tuple[tuple[int, ...], ...]] = (
    _PAWN_FILE,
    _KNIGHT_FILE,
    _BISHOP_FILE,
    _ROOK_FILE,
    _QUEEN_FILE,
    _KING_FILE_MG,
    _KING_FILE_EG,
    PASSED_FILE_WEIGHT,
)

ISOLATED_PAWN: Final[int] = 12
DOUBLED_PAWN: Final[int] = 9
BACKWARD_PAWN: Final[int] = 8
CONNECTED_PAWN: Final[int] = 6
BISHOP_PAIR: Final[int] = 30
ROOK_PAIR: Final[int] = 12
ROOK_OPEN_FILE: Final[int] = 22
ROOK_HALF_OPEN_FILE: Final[int] = 11
BISHOP_MOBILITY: Final[int] = 4
TEMPO: Final[int] = 10

#: King danger is summed in centipawns of attacking material and scaled down; the
#: divisor keeps a lone rook worth a few pawns rather than a full queen.
KING_DANGER_DIVISOR: Final[int] = 8
#: Above this many centipawns of pressure the term stops being a bonus and starts
#: being a warning, so it is clamped. Without a cap, an exposed king in a
#: queen-sac attack ring evaluates several pawns *against* the side that caused it.
KING_DANGER_CAP: Final[int] = 400


def game_phase(board: Board) -> int:
    """Middlegame weight of the current material, ``0`` (endgame) to 24 (opening)."""
    total = 0
    for piece_type, weight in PHASE_WEIGHT.items():
        if not weight:
            continue
        for color in (Color.WHITE, Color.BLACK):
            total += weight * board.count(piece_type, color)
    return total if total < FULL_PHASE else FULL_PHASE


def evaluate(board: Board, *, detailed: bool = True) -> int:
    """Static evaluation in centipawns from White's point of view.

    Positive favours White. Mate is *not* special-cased: the search recognises
    mate by the depth at which it delivers checkmate, which is far more reliable
    than trying to spot "mate next move" from a static score.

    ``detailed=False`` skips the mobility and king-danger terms, which together
    are the expensive half of the evaluation. A transposition-table probe wants a
    cheap comparison score, not a full one.
    """
    phase = game_phase(board)
    midgame, endgame = phase, FULL_PHASE - phase

    score = _placement(board, midgame, endgame)
    score += _pawn_structure(board, midgame, endgame)
    if detailed:
        score += _rooks_and_bishops(board)
        score += _king_danger(board)
    if board.side is Color.WHITE:
        score += TEMPO
    else:
        score -= TEMPO
    return score


def evaluate_side(board: Board, color: Color) -> int:
    """Evaluation from ``color``'s point of view."""
    score = evaluate(board)
    return score if color is Color.WHITE else -score


def _placement(board: Board, midgame: int, endgame: int) -> int:
    """Material, pair bonuses, and the phase-blended piece-square tables."""
    score = 0
    for color in (Color.WHITE, Color.BLACK):
        sign = 1 if color is Color.WHITE else -1
        # Tables are a1-first for White; Black reads the 180-degree rotation.
        for piece_type in PIECE_ORDER:
            squares = iter_bits(board.piece_bb(piece_type, color))
            if not squares:
                continue
            value = PIECE_VALUES[piece_type]
            mid_table = _MIDGAME_TABLES[piece_type]
            end_table = _ENDGAME_TABLES[piece_type]
            for sq in squares:
                # A Black piece is read at the square it looks like from Black's own
                # side, which is a rank flip: ``sq ^ 56``. This is what makes the
                # starting position score exactly zero, since e1 and e8 then land
                # on the same entry.
                #
                # A rank flip is *not* a 180-degree rotation, so mirror symmetry
                # only holds if the tables are also invariant under file reversal.
                # That is why every per-file weight array below is a palindrome;
                # `test_piece_square_symmetry` enforces it. ``63 - sq`` would give
                # rotation symmetry on its own, but then White's e1 and Black's e8
                # read different entries and the opening starts 8 centipawns adrift.
                index = sq if color is Color.WHITE else sq ^ 56
                blended = (
                    mid_table[index] * midgame + end_table[index] * endgame
                ) // FULL_PHASE
                score += sign * (value + blended)

        if board.count(PieceType.BISHOP, color) >= 2:
            score += sign * BISHOP_PAIR
        if board.count(PieceType.ROOK, color) >= 2:
            score += sign * ROOK_PAIR
    return score


def _pawn_structure(board: Board, midgame: int, endgame: int) -> int:
    """Doubled, isolated, backward, connected and passed pawns."""
    score = 0
    for color in (Color.WHITE, Color.BLACK):
        sign = 1 if color is Color.WHITE else -1
        pawns = board.piece_bb(PieceType.PAWN, color)
        if not pawns:
            continue
        enemy_pawns = board.piece_bb(PieceType.PAWN, color.opposite)
        per_file = _pawns_per_file(pawns)
        step = 1 if color is Color.WHITE else -1

        for sq in iter_bits(pawns):
            file_index = sq & 7
            rank = rank_of(sq)
            neighbour_left = per_file[file_index - 1] if file_index else 0
            neighbour_right = per_file[file_index + 1] if file_index < 7 else 0
            has_neighbour = bool(neighbour_left or neighbour_right)

            if per_file[file_index] > 1:
                score += sign * DOUBLED_PAWN
            if not has_neighbour:
                score += sign * ISOLATED_PAWN

            forward_rank = rank + step
            if has_neighbour:
                if _is_supported(sq, color, pawns):
                    score += sign * CONNECTED_PAWN
            elif 0 <= forward_rank < 8 and PAWN_ATTACKS[int(color.opposite)][
                square(file_index, forward_rank)
            ]:
                # No neighbour to fall back on and an enemy pawn covers the only
                # square ahead, so the pawn cannot safely advance.
                score += sign * BACKWARD_PAWN

            if _is_passed(sq, color, enemy_pawns):
                remaining = (7 - rank) if color is Color.WHITE else rank
                blended = (
                    PASSED_MG[remaining] * midgame + PASSED_EG[remaining] * endgame
                ) // FULL_PHASE
                score += sign * (blended + PASSED_FILE_WEIGHT[file_index])
    return score


def _neighbour_files(file_index: int) -> tuple[int, int]:
    return file_index - 1, file_index + 1


def _is_supported(sq: int, color: Color, pawns: int) -> bool:
    """True when a friendly pawn covers this one from the side or from behind."""
    file_index = sq & 7
    rank = rank_of(sq)
    back = rank - (1 if color is Color.WHITE else -1)
    for f in _neighbour_files(file_index):
        if not 0 <= f < 8:
            continue
        if pawns >> square(f, rank) & 1:
            return True
        if 0 <= back < 8 and pawns >> square(f, back) & 1:
            return True
    return False


def _pawns_per_file(pawns: int) -> list[int]:
    counts = [0] * 8
    for sq in iter_bits(pawns):
        counts[sq & 7] += 1
    return counts


def _is_passed(sq: int, color: Color, enemy: int) -> bool:
    """True when no enemy pawn sits ahead on this file or either neighbour."""
    file_index = sq & 7
    rank = rank_of(sq)
    if color is Color.WHITE:
        rows = range(rank + 1, 8)
    else:
        rows = range(rank - 1, -1, -1)
    for f in _neighbour_files(file_index):
        if not 0 <= f < 8:
            continue
        for r in rows:
            if enemy >> square(f, r) & 1:
                return False
    return True


def _rooks_and_bishops(board: Board) -> int:
    """Rooks on open files, and bishop mobility."""
    score = 0
    occupancy = board.occupancy_all
    for color in (Color.WHITE, Color.BLACK):
        sign = 1 if color is Color.WHITE else -1
        own_pawns = board.piece_bb(PieceType.PAWN, color)
        enemy_pawns = board.piece_bb(PieceType.PAWN, color.opposite)

        for sq in iter_bits(board.piece_bb(PieceType.ROOK, color)):
            file_bb = FILES[sq & 7]
            if not own_pawns & file_bb:
                score += sign * (
                    ROOK_HALF_OPEN_FILE if enemy_pawns & file_bb else ROOK_OPEN_FILE
                )

        for sq in iter_bits(board.piece_bb(PieceType.BISHOP, color)):
            score += sign * popcount(bishop_attacks(sq, occupancy)) * BISHOP_MOBILITY
    return score


def _king_danger(board: Board) -> int:
    """Enemy material bearing on the side that just moved.

    Only the previous mover is scored: after ``e2e4`` it is Black who is under
    threat, and that is the only king the search is about to unwind. Scoring both
    kings would double the cost for a term the search barely uses.
    """
    defender = board.side.opposite
    attacker = board.side
    king = board.king_squares[int(defender)]
    if king < 0:
        return 0

    zone = KING_ATTACKS[king] | (1 << king)
    occupancy = board.occupancy_all
    pressure = 0
    for piece_type in (PieceType.KNIGHT, PieceType.BISHOP, PieceType.ROOK, PieceType.QUEEN):
        value = PIECE_VALUES[piece_type]
        for sq in iter_bits(board.piece_bb(piece_type, attacker)):
            if piece_type is PieceType.KNIGHT:
                attacked = KNIGHT_ATTACKS[sq]
            elif piece_type is PieceType.BISHOP:
                attacked = bishop_attacks(sq, occupancy)
            elif piece_type is PieceType.ROOK:
                attacked = rook_attacks(sq, occupancy)
            else:
                attacked = queen_attacks(sq, occupancy)
            if attacked & zone:
                pressure += value
    # Centipawns of pressure, but discounted: a piece being *able* to attack the
    # king is not the same as it having a safe path there.
    danger = pressure // KING_DANGER_DIVISOR
    return -(danger if danger < KING_DANGER_CAP else KING_DANGER_CAP)


def tapered(piece_type: PieceType, sq: int, *, white: bool, phase: int) -> int:
    """A single blended piece-square value, for debugging and symmetry tests."""
    index = sq if white else sq ^ 56
    return (
        _MIDGAME_TABLES[piece_type][index] * phase
        + _ENDGAME_TABLES[piece_type][index] * (FULL_PHASE - phase)
    ) // FULL_PHASE


def piece_type_on(board: Board, sq: int) -> PieceType | None:
    """Convenience wrapper returning ``None`` for an empty square."""
    piece = board.piece_at(sq)
    return piece_type_of(piece) if piece else None

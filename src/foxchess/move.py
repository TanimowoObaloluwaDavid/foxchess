"""16-bit move encoding.

A move packs into a single integer: ``from | to << 6 | flag << 12``. Python ints
are arbitrary precision, but staying inside 16 bits keeps move keys cheap to hash
for the transposition table and makes ``Move`` a plain int under the hood, which
is what lets move generation return bare ints in the hot path.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .types import PieceType, name_of, square_from_name

QUIET: int = 0
DOUBLE_PUSH: int = 1
CAPTURE: int = 2
EN_PASSANT: int = 3
CASTLE_KINGSIDE: int = 4
CASTLE_QUEENSIDE: int = 5
PROMOTION_KNIGHT: int = 8
PROMOTION_BISHOP: int = 9
PROMOTION_ROOK: int = 10
PROMOTION_QUEEN: int = 11
PROMOTION_KNIGHT_CAPTURE: int = 12
PROMOTION_BISHOP_CAPTURE: int = 13
PROMOTION_ROOK_CAPTURE: int = 14
PROMOTION_QUEEN_CAPTURE: int = 15

#: Flag -> promoted piece type, for every promotion flag.
PROMOTION_PIECE: dict[int, PieceType] = {
    PROMOTION_KNIGHT: PieceType.KNIGHT,
    PROMOTION_BISHOP: PieceType.BISHOP,
    PROMOTION_ROOK: PieceType.ROOK,
    PROMOTION_QUEEN: PieceType.QUEEN,
    PROMOTION_KNIGHT_CAPTURE: PieceType.KNIGHT,
    PROMOTION_BISHOP_CAPTURE: PieceType.BISHOP,
    PROMOTION_ROOK_CAPTURE: PieceType.ROOK,
    PROMOTION_QUEEN_CAPTURE: PieceType.QUEEN,
}

CASTLE_FLAGS: frozenset[int] = frozenset({CASTLE_KINGSIDE, CASTLE_QUEENSIDE})

#: 32 promotions * 64 origins * 64 destinations, precomputed and verified in tests.
MOVE_COUNT: int = 32 * 64 * 64

#: Sentinel used where a move is required but none exists (no legal moves).
MOVE_NONE: int = 0
#: The null move: "pass". Encoded as a0a0, never appears in a legal move list.
NULL_MOVE: int = 0
#: "There is no move here", as distinct from ``a0a0``. A packed move of 0 is a real
#: move -- the a1a1 encoding -- so the only safe stand-in for *nothing* is a value
#: outside the packed range, which is what -1 is.
NO_MOVE: int = -1


@dataclass(frozen=True, slots=True)
class Move:
    """A decoded, human-friendly move.

    The engine's hot paths pass raw ``int`` moves around; this type exists for the
    notation, API and CLI layers where readability wins.
    """

    from_square: int
    to_square: int
    promotion: PieceType | None = None
    is_capture: bool = False
    is_en_passant: bool = False
    is_castling: bool = False

    @staticmethod
    def encode(move: Move) -> int:
        flag = _encode_flag(move)
        return move.from_square | (move.to_square << 6) | (flag << 12)

    @staticmethod
    def decode(raw: int) -> Move:
        from_square = raw & 0x3F
        to_square = (raw >> 6) & 0x3F
        flag = (raw >> 12) & 0xF
        return Move(
            from_square=from_square,
            to_square=to_square,
            promotion=PROMOTION_PIECE.get(flag),
            is_capture=flag in _CAPTURE_FLAGS,
            is_en_passant=flag == EN_PASSANT,
            is_castling=flag in CASTLE_FLAGS,
        )

    @property
    def is_promotion(self) -> bool:
        return self.promotion is not None

    @property
    def is_quiet(self) -> bool:
        return not self.is_capture and not self.is_promotion

    def uci(self) -> str:
        """Long algebraic, the format UCI and ``position fen ... moves`` use."""
        base = f"{name_of(self.from_square)}{name_of(self.to_square)}"
        return f"{base}{self.promotion.letter.lower()}" if self.promotion else base

    def algebraic(self) -> str:
        """Human-friendly long algebraic, e.g. ``Nf3`` or ``e8=Q``."""
        if self.is_castling and self.to_square > self.from_square:
            return "O-O"
        if self.is_castling:
            return "O-O-O"
        return self.uci()

    def __str__(self) -> str:
        return self.uci()


_CAPTURE_FLAGS: frozenset[int] = frozenset(
    {
        CAPTURE,
        EN_PASSANT,
        PROMOTION_KNIGHT_CAPTURE,
        PROMOTION_BISHOP_CAPTURE,
        PROMOTION_ROOK_CAPTURE,
        PROMOTION_QUEEN_CAPTURE,
    }
)

_CAPTURE_FLAGS_SET = _CAPTURE_FLAGS


def _encode_flag(move: Move) -> int:
    if move.is_en_passant:
        return EN_PASSANT
    if move.is_castling:
        return CASTLE_KINGSIDE if move.to_square > move.from_square else CASTLE_QUEENSIDE
    if move.promotion is not None:
        base = _PROMOTION_BASE[move.promotion]
        return base + 4 if move.is_capture else base
    return CAPTURE if move.is_capture else QUIET


_PROMOTION_BASE: dict[PieceType, int] = {
    PieceType.KNIGHT: PROMOTION_KNIGHT,
    PieceType.BISHOP: PROMOTION_BISHOP,
    PieceType.ROOK: PROMOTION_ROOK,
    PieceType.QUEEN: PROMOTION_QUEEN,
}

_UCI_PROMOTION: dict[str, PieceType] = {
    "q": PieceType.QUEEN,
    "r": PieceType.ROOK,
    "b": PieceType.BISHOP,
    "n": PieceType.KNIGHT,
}


# ---------------------------------------------------------------------------
# Int helpers used by the hot loops
# ---------------------------------------------------------------------------


def encode_int(from_square: int, to_square: int, flag: int = QUIET) -> int:
    return from_square | (to_square << 6) | (flag << 12)


def decode_flag(raw: int) -> int:
    return (raw >> 12) & 0xF


def decode_from(raw: int) -> int:
    return raw & 0x3F


def decode_to(raw: int) -> int:
    return (raw >> 6) & 0x3F


def is_capture_flag(raw: int) -> bool:
    return decode_flag(raw) in _CAPTURE_FLAGS_SET


def is_promotion_flag(raw: int) -> bool:
    return (raw >> 12) & 0xF >= PROMOTION_KNIGHT


def promotion_of(raw: int) -> PieceType | None:
    return PROMOTION_PIECE.get(decode_flag(raw))


def to_uci(raw: int) -> str:
    """Long algebraic straight from the packed integer.

    ``NO_MOVE`` renders as ``0000``, the null move UCI reserves, rather than as
    whatever the packed bits happen to decode to.
    """
    if raw == NO_MOVE:
        return "0000"
    text = f"{name_of(decode_from(raw))}{name_of(decode_to(raw))}"
    promotion = PROMOTION_PIECE.get(decode_flag(raw))
    return f"{text}{promotion.letter.lower()}" if promotion else text


def from_uci(text: str, legal_moves: Iterable[int] | None = None) -> int:
    """Parse a UCI move such as ``e2e4`` or ``e7e8q`` into a packed move.

    A four-character UCI string is genuinely ambiguous: ``e2e4`` is a double pawn
    push, and ``e5d6`` may be an en-passant capture. The flag lives in bits that
    UCI simply does not carry, so a packed move built from the text alone is
    *wrong* half the time and will fail an ``in legal_moves()`` membership test.

    Pass ``legal_moves`` to resolve the ambiguity against the real position. That
    is the only sound way, and every caller that needs a usable move should do it.
    Without it the flag is guessed as ``QUIET``, which is enough for display and
    for sorting, and not enough to make a move.

    Promotions default to a queen when the suffix is omitted, which is what UCI
    clients in the wild actually send.
    """
    cleaned = text.strip().lower().replace(" ", "")
    if len(cleaned) not in (4, 5):
        raise ValueError(f"invalid UCI move: {text!r}")
    from_square = square_from_name(cleaned[:2])
    to_square = square_from_name(cleaned[2:4])

    if len(cleaned) == 5:
        promotion = _UCI_PROMOTION.get(cleaned[4])
        if promotion is None:
            raise ValueError(f"invalid promotion piece in {text!r}")
        wanted_promotion: PieceType | None = promotion
    else:
        wanted_promotion = None

    if legal_moves is not None:
        for candidate in legal_moves:
            if decode_from(candidate) != from_square or decode_to(candidate) != to_square:
                continue
            if promotion_of(candidate) != wanted_promotion:
                continue
            # "e2e4" with no suffix could be a double push or a plain push; the
            # generator already decided, so take its answer.
            if wanted_promotion is None and is_promotion_flag(candidate):
                continue
            return candidate
        raise ValueError(f"no legal move matches {text!r}")

    if wanted_promotion is None:
        return encode_int(from_square, to_square, QUIET)
    return encode_int(from_square, to_square, _PROMOTION_BASE[wanted_promotion])


def move_from_parts(
    from_square: int,
    to_square: int,
    *,
    promotion: PieceType | None = None,
    is_capture: bool = False,
    is_en_passant: bool = False,
    is_castling: bool = False,
) -> int:
    """Build a packed move from explicit parts, deriving the flag."""
    return Move.encode(
        Move(
            from_square=from_square,
            to_square=to_square,
            promotion=promotion,
            is_capture=is_capture,
            is_en_passant=is_en_passant,
            is_castling=is_castling,
        )
    )

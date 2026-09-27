"""Core types, constants and small helpers shared by every other module.

Square indexing is ``rank * 8 + file`` with ``a1 == 0`` and ``h8 == 63``. Bit 0 of
a 64-bit integer therefore always means the same square everywhere in the
codebase, which is what lets every attack table be a plain ``int``.
"""

from __future__ import annotations

from enum import IntEnum

# --------------------------------------------------------------------------------------
# Board geometry
# --------------------------------------------------------------------------------------

FILE_A: int = 0
RANK_1: int = 0

FILE_COUNT: int = 8
RANK_COUNT: int = 8
SQUARE_COUNT: int = 64
SQUARE_NONE: int = -1

EMPTY: int = 0
FULL: int = (1 << SQUARE_COUNT) - 1

RANK_1_BB: int = 0xFF
RANK_2_BB: int = 0xFF << 8
RANK_3_BB: int = 0xFF << 16
RANK_4_BB: int = 0xFF << 24
RANK_5_BB: int = 0xFF << 32
RANK_6_BB: int = 0xFF << 40
RANK_7_BB: int = 0xFF << 48
RANK_8_BB: int = 0xFF << 56

FILE_A_BB: int = 0x0101010101010101
FILE_B_BB: int = FILE_A_BB << 1
FILE_C_BB: int = FILE_A_BB << 2
FILE_D_BB: int = FILE_A_BB << 3
FILE_E_BB: int = FILE_A_BB << 4
FILE_F_BB: int = FILE_A_BB << 5
FILE_G_BB: int = FILE_A_BB << 6
FILE_H_BB: int = FILE_A_BB << 7

FILES: tuple[int, ...] = (
    FILE_A_BB,
    FILE_B_BB,
    FILE_C_BB,
    FILE_D_BB,
    FILE_E_BB,
    FILE_F_BB,
    FILE_G_BB,
    FILE_H_BB,
)
RANKS: tuple[int, ...] = (
    RANK_1_BB,
    RANK_2_BB,
    RANK_3_BB,
    RANK_4_BB,
    RANK_5_BB,
    RANK_6_BB,
    RANK_7_BB,
    RANK_8_BB,
)

SQUARE_NAMES: tuple[str, ...] = tuple(
    f"{file}{rank}" for rank in range(1, 9) for file in "abcdefgh"
)

# North is +8, east is +1. The eight ray directions in canonical order.
NORTH: int = 8
EAST: int = 1
SOUTH: int = -8
WEST: int = -1
NORTH_EAST: int = 9
SOUTH_EAST: int = -7
NORTH_WEST: int = 7
SOUTH_WEST: int = -9

DIRECTIONS: tuple[int, ...] = (
    NORTH,
    EAST,
    SOUTH,
    WEST,
    NORTH_EAST,
    SOUTH_EAST,
    SOUTH_WEST,
    NORTH_WEST,
)

ORTHOGONAL: tuple[int, ...] = (NORTH, EAST, SOUTH, WEST)
DIAGONAL: tuple[int, ...] = (NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST)

# Bitwise-not needs to stay inside 64 bits.
NOT_A_FILE: int = FULL ^ FILE_A_BB
NOT_H_FILE: int = FULL ^ FILE_H_BB

LIGHT_SQUARES: int = 0x55AA55AA55AA55AA
DARK_SQUARES: int = FULL ^ LIGHT_SQUARES


def square(file_index: int, rank_index: int) -> int:
    """Combine a 0-based file and rank into a square index."""
    return rank_index * FILE_COUNT + file_index


def file_of(sq: int) -> int:
    return sq & 7


def rank_of(sq: int) -> int:
    return sq >> 3


def name_of(sq: int) -> str:
    return SQUARE_NAMES[sq]


def square_from_name(text: str) -> int:
    """Parse ``'e4'`` into a square index. Raises ``ValueError`` if malformed."""
    cleaned = text.strip().lower()
    if len(cleaned) != 2 or cleaned[0] not in "abcdefgh" or cleaned[1] not in "12345678":
        raise ValueError(f"invalid square name: {text!r}")
    return square(ord(cleaned[0]) - ord("a"), int(cleaned[1]) - 1)


# --------------------------------------------------------------------------------------
# Colours and pieces
# --------------------------------------------------------------------------------------


class Color(IntEnum):
    """White is 0 so ``~color`` flips sides and ``color & 1`` stays cheap."""

    WHITE = 0
    BLACK = 1

    @property
    def opposite(self) -> Color:
        return Color.BLACK if self is Color.WHITE else Color.WHITE

    @property
    def name_lower(self) -> str:
        return "white" if self is Color.WHITE else "black"

    @property
    def fen_letter(self) -> str:
        """The single-character side-to-move field of a FEN: ``w`` or ``b``.

        Deliberately not ``name_lower[0]``, and deliberately not the same thing as
        ``name_lower``. FEN wants one character, and a position whose FEN says
        ``white`` instead of ``w`` will not parse in half the software on earth.
        """
        return "w" if self is Color.WHITE else "b"

    @property
    def pawn_start_rank(self) -> int:
        """Rank index pawns of this colour begin on: 1 for White, 6 for Black."""
        return 1 if self is Color.WHITE else 6

    @property
    def promotion_rank(self) -> int:
        """Rank index this colour promotes on: 7 for White, 0 for Black."""
        return 7 if self is Color.WHITE else 0

    @property
    def home_rank(self) -> int:
        """Rank index of this colour's back rank: 0 for White, 7 for Black."""
        return 0 if self is Color.WHITE else 7


class PieceType(IntEnum):
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6

    @property
    def name_lower(self) -> str:
        return self.name.lower()

    @property
    def letter(self) -> str:
        return _PIECE_LETTERS[self]

    @property
    def value(self) -> int:
        return PIECE_VALUES[self]


# A piece is ``(piece_type << 1) | color``. 0 stays free to mean "empty square",
# which is why piece types start at 1 rather than 0.
def make_piece(piece_type: PieceType | int, color: Color | int) -> int:
    return (int(piece_type) << 1) | int(color)


def piece_type_of(piece: int) -> PieceType:
    return PieceType(piece >> 1)


def color_of(piece: int) -> Color:
    return Color(piece & 1)


def piece_letter(piece: int) -> str:
    """``'N'`` for a white knight, ``'n'`` for a black one."""
    letter = _PIECE_LETTERS[PieceType(piece >> 1)]
    return letter if color_of(piece) is Color.WHITE else letter.lower()


PIECE_VALUES: dict[PieceType, int] = {
    PieceType.PAWN: 100,
    PieceType.KNIGHT: 320,
    PieceType.BISHOP: 330,
    PieceType.ROOK: 500,
    PieceType.QUEEN: 900,
    PieceType.KING: 0,
}

_PIECE_LETTERS: dict[PieceType, str] = {
    PieceType.PAWN: "P",
    PieceType.KNIGHT: "N",
    PieceType.BISHOP: "B",
    PieceType.ROOK: "R",
    PieceType.QUEEN: "Q",
    PieceType.KING: "K",
}

PIECE_BY_LETTER: dict[str, PieceType] = {
    letter: piece_type for piece_type, letter in _PIECE_LETTERS.items()
}
PIECE_BY_LETTER.update(
    {letter.lower(): piece_type for piece_type, letter in _PIECE_LETTERS.items()}
)

# Concrete piece codes, spelled out because they show up in the hot loops.
WP: int = make_piece(PieceType.PAWN, Color.WHITE)
WN: int = make_piece(PieceType.KNIGHT, Color.WHITE)
WB: int = make_piece(PieceType.BISHOP, Color.WHITE)
WR: int = make_piece(PieceType.ROOK, Color.WHITE)
WQ: int = make_piece(PieceType.QUEEN, Color.WHITE)
WK: int = make_piece(PieceType.KING, Color.WHITE)
BP: int = make_piece(PieceType.PAWN, Color.BLACK)
BN: int = make_piece(PieceType.KNIGHT, Color.BLACK)
BB: int = make_piece(PieceType.BISHOP, Color.BLACK)
BR: int = make_piece(PieceType.ROOK, Color.BLACK)
BQ: int = make_piece(PieceType.QUEEN, Color.BLACK)
BK: int = make_piece(PieceType.KING, Color.BLACK)

PIECE_SETS: dict[Color, tuple[int, int, int, int, int, int]] = {
    Color.WHITE: (WP, WN, WB, WR, WQ, WK),
    Color.BLACK: (BP, BN, BB, BR, BQ, BK),
}

PIECE_ORDER: tuple[PieceType, ...] = (
    PieceType.PAWN,
    PieceType.KNIGHT,
    PieceType.BISHOP,
    PieceType.ROOK,
    PieceType.QUEEN,
    PieceType.KING,
)

# --------------------------------------------------------------------------------------
# Castling
# --------------------------------------------------------------------------------------


class CastlingRight(IntEnum):
    """Bit flags stored in the position's castling field."""

    WHITE_KINGSIDE = 1
    WHITE_QUEENSIDE = 2
    BLACK_KINGSIDE = 4
    BLACK_QUEENSIDE = 8


WHITE_KING_SQUARE: int = square(4, 0)
WHITE_ROOK_KING_SQUARE: int = square(7, 0)
WHITE_ROOK_QUEEN_SQUARE: int = square(0, 0)
BLACK_KING_SQUARE: int = square(4, 7)
BLACK_ROOK_KING_SQUARE: int = square(7, 7)
BLACK_ROOK_QUEEN_SQUARE: int = square(0, 7)

#: Where the rook lands after castling.
WHITE_ROOK_TO_KING_SQUARE: int = square(5, 0)
WHITE_ROOK_TO_QUEEN_SQUARE: int = square(3, 0)
BLACK_ROOK_TO_KING_SQUARE: int = square(5, 7)
BLACK_ROOK_TO_QUEEN_SQUARE: int = square(3, 7)

CASTLING_SQUARES: tuple[tuple[Color, CastlingRight, int, int, int, int], ...] = (
    # color, right, king from, king to, rook from, rook to
    (Color.WHITE, CastlingRight.WHITE_KINGSIDE,
     WHITE_KING_SQUARE, square(6, 0), WHITE_ROOK_KING_SQUARE, square(5, 0)),
    (Color.WHITE, CastlingRight.WHITE_QUEENSIDE,
     WHITE_KING_SQUARE, square(2, 0), WHITE_ROOK_QUEEN_SQUARE, square(3, 0)),
    (Color.BLACK, CastlingRight.BLACK_KINGSIDE,
     BLACK_KING_SQUARE, square(6, 7), BLACK_ROOK_KING_SQUARE, square(5, 7)),
    (Color.BLACK, CastlingRight.BLACK_QUEENSIDE,
     BLACK_KING_SQUARE, square(2, 7), BLACK_ROOK_QUEEN_SQUARE, square(3, 7)),
)

#: Squares that must be *empty* before castling is allowed. The king's own square is
#: excluded because it is of course occupied by the king.
CASTLING_EMPTY: dict[Color, dict[CastlingRight, int]] = {
    Color.WHITE: {
        CastlingRight.WHITE_KINGSIDE: 1 << square(5, 0) | 1 << square(6, 0),
        CastlingRight.WHITE_QUEENSIDE: 1 << square(1, 0) | 1 << square(2, 0) | 1 << square(3, 0),
    },
    Color.BLACK: {
        CastlingRight.BLACK_KINGSIDE: 1 << square(5, 7) | 1 << square(6, 7),
        CastlingRight.BLACK_QUEENSIDE: 1 << square(1, 7) | 1 << square(2, 7) | 1 << square(3, 7),
    },
}

#: Every square the king passes through. None of them may be under attack.
CASTLING_KING_PATH: dict[Color, dict[CastlingRight, int]] = {
    Color.WHITE: {
        CastlingRight.WHITE_KINGSIDE: 1 << square(4, 0) | 1 << square(5, 0) | 1 << square(6, 0),
        CastlingRight.WHITE_QUEENSIDE: 1 << square(4, 0) | 1 << square(3, 0) | 1 << square(2, 0),
    },
    Color.BLACK: {
        CastlingRight.BLACK_KINGSIDE: 1 << square(4, 7) | 1 << square(5, 7) | 1 << square(6, 7),
        CastlingRight.BLACK_QUEENSIDE: 1 << square(4, 7) | 1 << square(3, 7) | 1 << square(2, 7),
    },
}


def square_bb(sq: int) -> int:
    """The single-square bitboard for ``sq``."""
    return 1 << sq


def popcount(bb: int) -> int:
    """Number of set bits. Uses the Kernighan trick, no binning."""
    count = 0
    while bb:
        bb &= bb - 1
        count += 1
    return count


def lsb(bb: int) -> int:
    """Index of the least significant set bit. ``bb`` must be non-zero."""
    return (bb & -bb).bit_length() - 1


def msb(bb: int) -> int:
    """Index of the most significant set bit. ``bb`` must be non-zero."""
    return bb.bit_length() - 1


def pop_lsb(bb: int) -> tuple[int, int]:
    """Split off the lowest set bit: returns ``(square, remaining_bb)``."""
    sq = (bb & -bb).bit_length() - 1
    return sq, bb & (bb - 1)


def iter_bits(bb: int) -> list[int]:
    """All set bits as square indices, lowest first."""
    out: list[int] = []
    while bb:
        sq, bb = pop_lsb(bb)
        out.append(sq)
    return out


def shift(bb: int, delta: int) -> int:
    """Shift a bitboard by a fixed square delta, clipping at the board edges."""
    if delta > 0:
        return (bb << delta) & FULL
    if delta < 0:
        return (bb >> -delta) & FULL
    return bb


def is_set(bb: int, sq: int) -> bool:
    return (bb >> sq) & 1 == 1

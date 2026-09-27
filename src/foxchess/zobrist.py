"""Zobrist hashing.

Random 64-bit keys for every (piece, square) pair, castling right, en-passant
file and the side to move. The keys come from a seeded PRNG, so hashes are
byte-for-byte reproducible across processes and machines — which is what lets
the tests assert on ``transposition_key`` values and lets a transposition table
be shared between runs.
"""

from __future__ import annotations

import random
from typing import Final

from .types import FILE_COUNT, SQUARE_COUNT, Color, PieceType, make_piece

#: 12 concrete pieces x 64 squares, 4 castling rights, 8 en-passant files, 1 side.
PIECE_KEYS: Final[list[list[int]]] = []
CASTLING_KEYS: Final[list[int]] = [0] * 16
EP_KEYS: Final[list[int]] = [0] * FILE_COUNT


def _build_keys() -> int:
    # A fixed seed is deliberate: reproducible hashes are a feature, not an
    # accident. A random seed would make hash values untestable.
    rng = random.Random(0xF0C4_5E55_2024)
    for _ in range(16):
        PIECE_KEYS.append([rng.getrandbits(64) for _ in range(SQUARE_COUNT)])
    for i in range(16):
        CASTLING_KEYS[i] = rng.getrandbits(64)
    for i in range(FILE_COUNT):
        EP_KEYS[i] = rng.getrandbits(64)
    # Row 0 is EMPTY. Callers guard with `captured != EMPTY` today, so a random
    # row 0 would never be read -- but a row of zeros means a future caller that
    # forgets the guard hashes a missing piece as a no-op instead of quietly
    # corrupting every key it touches.
    PIECE_KEYS[0] = [0] * SQUARE_COUNT
    return rng.getrandbits(64)


#: Returned rather than assigned from inside the builder, so the module never
#: needs a `global` statement to publish it.
SIDE_KEY: int = _build_keys()


def piece_key(piece: int, square: int) -> int:
    return PIECE_KEYS[piece][square]


def castling_key(rights: int) -> int:
    return CASTLING_KEYS[rights & 0xF]


def ep_key(square: int) -> int:
    """Key for the en-passant target square, or 0 when there is none.

    The key is indexed by *file* and deliberately ignores the rank, so that an
    en-passant target is only distinguished from "no target" when it is actually
    capturable. A naive implementation that keys on the raw square produces a
    different hash for a1 and a5, breaking transpositions for no reason.
    """
    if square < 0:
        return 0
    return EP_KEYS[square & 7]


def side_key() -> int:
    return SIDE_KEY


def compute(pieces: dict[int, int], castling: int, ep_square: int, side: Color) -> int:
    """Hash a position from scratch. Used by tests and FEN validation."""
    key = 0
    for piece, occupancy in pieces.items():
        if piece == 0:
            continue
        table = PIECE_KEYS[piece]
        # A separate name from the loop variable, so the bit-clearing below
        # cannot be mistaken for reassigning the thing being iterated.
        remaining = occupancy
        while remaining:
            sq = (remaining & -remaining).bit_length() - 1
            key ^= table[sq]
            remaining &= remaining - 1
    key ^= CASTLING_KEYS[castling & 0xF]
    key ^= ep_key(ep_square)
    if int(side) == 1:
        key ^= SIDE_KEY
    return key


# Sanity check: every concrete piece code must have a key row.
assert len(PIECE_KEYS) == 16
assert all(
    len(PIECE_KEYS[make_piece(pt, color)]) == SQUARE_COUNT
    for color in Color
    for pt in (PieceType.PAWN, PieceType.KNIGHT, PieceType.KING)
)

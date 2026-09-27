"""Precomputed attack tables.

Sliding attacks use the classic "whole ray + first blocker" formulation rather
than magic bitboards. Magic lookups are faster in C because a multiply is nearly
free; in CPython the interpreter overhead dominates, so the simpler version
wins — and it is far easier to verify, which matters more than raw nps for a
project whose correctness claims are backed by perft.
"""

from __future__ import annotations

from .types import (
    DIAGONAL,
    DIRECTIONS,
    EAST,
    NORTH,
    NORTH_EAST,
    NORTH_WEST,
    ORTHOGONAL,
    SOUTH,
    SOUTH_EAST,
    SOUTH_WEST,
    SQUARE_COUNT,
    WEST,
    file_of,
    rank_of,
    square,
)

# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------

#: Explicit ``(file_step, rank_step)`` per direction. A square is ``rank * 8 + file``,
#: so a move of ``(df, dr)`` shifts the index by ``dr * 8 + df``; deriving the steps
#: from the sign of the delta alone conflates EAST with NORTH, so they are spelled out.
DIRECTION_STEPS: dict[int, tuple[int, int]] = {
    NORTH: (0, 1),
    EAST: (1, 0),
    SOUTH: (0, -1),
    WEST: (-1, 0),
    NORTH_EAST: (1, 1),
    SOUTH_EAST: (1, -1),
    SOUTH_WEST: (-1, -1),
    NORTH_WEST: (-1, 1),
}


# ---------------------------------------------------------------------------
# Rays
# ---------------------------------------------------------------------------

#: ``RAYS[dir_index][square]`` is the entire ray from ``square`` outwards along
#: that direction, ignoring blockers. Built once at import time.
def _build_rays() -> list[list[int]]:
    rays: list[list[int]] = []
    for delta in DIRECTIONS:
        df, dr = DIRECTION_STEPS[delta]
        table: list[int] = []
        for sq in range(SQUARE_COUNT):
            f, r = file_of(sq), rank_of(sq)
            bb = 0
            while True:
                f += df
                r += dr
                if not (0 <= f < 8 and 0 <= r < 8):
                    break
                bb |= 1 << square(f, r)
            table.append(bb)
        rays.append(table)
    return rays


RAYS: list[list[int]] = _build_rays()

#: Whether a ray's square indices increase as it travels. North, east, north-east
#: and north-west walk towards higher bit indices; south, west and the two
#: southward diagonals walk towards lower ones. This decides whether the *first*
#: blocker along a ray is the lowest or the highest set bit.
RAY_INCREASES: list[bool] = [delta > 0 for delta in DIRECTIONS]

ORTHO_RAY_INDICES: tuple[int, ...] = tuple(DIRECTIONS.index(d) for d in ORTHOGONAL)
DIAG_RAY_INDICES: tuple[int, ...] = tuple(DIRECTIONS.index(d) for d in DIAGONAL)


def _slide(sq: int, occupancy: int, indices: tuple[int, ...]) -> int:
    """Sliding attacks from ``sq``, stopping at the first occupied square.

    The squares strictly between ``sq`` and the first blocker are
    ``ray & ~RAYS[index][blocker]``: the blocker's own ray covers everything from
    the blocker onwards, so masking it off leaves exactly the in-between squares.
    Expressing it that way keeps the tail direction-agnostic.
    """
    attacks = 0
    for index in indices:
        ray = RAYS[index][sq]
        if not ray:
            continue
        blockers = ray & occupancy
        if not blockers:
            attacks |= ray
            continue
        # The first blocker is the lowest set bit on an increasing ray and the
        # highest on a decreasing one.
        if RAY_INCREASES[index]:
            first = blockers & -blockers
        else:
            first = 1 << (blockers.bit_length() - 1)
        first_sq = first.bit_length() - 1
        attacks |= (ray & ~RAYS[index][first_sq]) | first
    return attacks


BISHOP_ATTACKS: list[int] = [0] * SQUARE_COUNT
ROOK_ATTACKS: list[int] = [0] * SQUARE_COUNT
QUEEN_ATTACKS: list[int] = [0] * SQUARE_COUNT

for _sq in range(SQUARE_COUNT):
    BISHOP_ATTACKS[_sq] = _slide(_sq, 0, DIAG_RAY_INDICES)
    ROOK_ATTACKS[_sq] = _slide(_sq, 0, ORTHO_RAY_INDICES)
    QUEEN_ATTACKS[_sq] = BISHOP_ATTACKS[_sq] | ROOK_ATTACKS[_sq]


def bishop_attacks(sq: int, occupancy: int) -> int:
    """Squares a bishop on ``sq`` attacks, given ``occupancy`` (all pieces)."""
    return _slide(sq, occupancy, DIAG_RAY_INDICES)


def rook_attacks(sq: int, occupancy: int) -> int:
    """Squares a rook on ``sq`` attacks, given ``occupancy`` (all pieces)."""
    return _slide(sq, occupancy, ORTHO_RAY_INDICES)


def queen_attacks(sq: int, occupancy: int) -> int:
    """Squares a queen on ``sq`` attacks, given ``occupancy`` (all pieces)."""
    return bishop_attacks(sq, occupancy) | rook_attacks(sq, occupancy)


# ---------------------------------------------------------------------------
# Leapers
# ---------------------------------------------------------------------------

#: ``(file_step, rank_step)`` pairs. Written as coordinate deltas so no board
#: edge is ever crossed by accident.
KNIGHT_STEPS: tuple[tuple[int, int], ...] = (
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
)
KING_STEPS: tuple[tuple[int, int], ...] = (
    (1, 1), (1, -1), (-1, 1), (-1, -1),
    (1, 0), (-1, 0), (0, 1), (0, -1),
)


def _stepper(steps: tuple[tuple[int, int], ...]) -> list[int]:
    table: list[int] = []
    for sq in range(SQUARE_COUNT):
        f, r = file_of(sq), rank_of(sq)
        bb = 0
        for df, dr in steps:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                bb |= 1 << square(nf, nr)
        table.append(bb)
    return table


KNIGHT_ATTACKS: list[int] = _stepper(KNIGHT_STEPS)
KING_ATTACKS: list[int] = _stepper(KING_STEPS)


def knight_attacks(sq: int) -> int:
    return KNIGHT_ATTACKS[sq]


def king_attacks(sq: int) -> int:
    return KING_ATTACKS[sq]


# ---------------------------------------------------------------------------
# Pawns
# ---------------------------------------------------------------------------

WHITE: int = 0
BLACK: int = 1

#: ``PAWN_ATTACKS[color][square]`` — the two squares a pawn of ``color`` standing
#: on ``square`` attacks, with no regard for whether those squares are empty.
PAWN_ATTACKS: dict[int, list[int]] = {WHITE: [0] * SQUARE_COUNT, BLACK: [0] * SQUARE_COUNT}

#: Single push destination, or ``-1`` when the pawn has no square to push to.
#:
#: The off-board sentinel has to be ``-1`` and not ``0``, because a1 *is* square 0
#: and is a perfectly good destination for the black pawn on a2. Using 0 as
#: "nowhere" made ``a2a1`` look like an absent move, which quietly deleted all
#: four a-file promotions.
PAWN_PUSH: dict[int, list[int]] = {WHITE: [-1] * SQUARE_COUNT, BLACK: [-1] * SQUARE_COUNT}

#: Double push destination, or ``-1`` when the pawn is not on its start rank.
PAWN_DOUBLE_PUSH: dict[int, list[int]] = {WHITE: [-1] * SQUARE_COUNT, BLACK: [-1] * SQUARE_COUNT}

#: Bitboards of squares a pawn of ``color`` may double-push from / promote on.
PAWN_DOUBLE_PUSH_FROM: dict[int, int] = {WHITE: 0, BLACK: 0}
PAWN_PROMOTION_PUSH: dict[int, int] = {WHITE: 0, BLACK: 0}

#: ``PAWN_ATTACKERS[color][square]`` — bitboard of pawns of ``color`` attacking
#: ``square``. The workhorse for pinned-pawn and en-passant legality checks.
PAWN_ATTACKERS: dict[int, list[int]] = {WHITE: [0] * SQUARE_COUNT, BLACK: [0] * SQUARE_COUNT}

WHITE_PAWN_ATTACKER_SQUARES = [square(f, 1) for f in range(8)]
BLACK_PAWN_ATTACKER_SQUARES = [square(f, 6) for f in range(8)]

for _color, _step, _start_rank, _promo_rank in (
    (WHITE, 1, 1, 7),
    (BLACK, -1, 6, 0),
):
    for _sq in range(SQUARE_COUNT):
        f, r = file_of(_sq), rank_of(_sq)
        if 0 <= r + _step < 8:
            PAWN_PUSH[_color][_sq] = square(f, r + _step)
        if r == _start_rank:
            PAWN_DOUBLE_PUSH[_color][_sq] = square(f, r + 2 * _step)
            PAWN_DOUBLE_PUSH_FROM[_color] |= 1 << _sq
        if r == _promo_rank:
            PAWN_PROMOTION_PUSH[_color] |= 1 << _sq
        for df in (-1, 1):
            nf = f + df
            if 0 <= nf < 8 and 0 <= r + _step < 8:
                target = square(nf, r + _step)
                PAWN_ATTACKS[_color][_sq] |= 1 << target
                PAWN_ATTACKERS[_color][target] |= 1 << _sq


def pawn_attacks(color: int, sq: int) -> int:
    return PAWN_ATTACKS[color][sq]


def pawn_attackers(color: int, sq: int) -> int:
    return PAWN_ATTACKERS[color][sq]


# ---------------------------------------------------------------------------
# Distance tables (evaluation)
# ---------------------------------------------------------------------------


def _build_distance_tables() -> tuple[list[int], list[int], list[int]]:
    """Edge distances, indexed by square."""
    file_dist = [0] * SQUARE_COUNT
    rank_dist = [0] * SQUARE_COUNT
    for sq in range(SQUARE_COUNT):
        f, r = file_of(sq), rank_of(sq)
        file_dist[sq] = min(f, 7 - f)
        rank_dist[sq] = min(r, 7 - r)
    return file_dist, rank_dist, [max(a, b) for a, b in zip(file_dist, rank_dist, strict=True)]


FILE_DISTANCE, RANK_DISTANCE, KING_DISTANCE = _build_distance_tables()

# Kept so the direction constants stay importable from one place.
__all__ = [
    "BISHOP_ATTACKS",
    "DIAG_RAY_INDICES",
    "KING_ATTACKS",
    "KNIGHT_ATTACKS",
    "ORTHO_RAY_INDICES",
    "PAWN_ATTACKERS",
    "PAWN_ATTACKS",
    "PAWN_DOUBLE_PUSH",
    "PAWN_DOUBLE_PUSH_FROM",
    "PAWN_PROMOTION_PUSH",
    "PAWN_PUSH",
    "QUEEN_ATTACKS",
    "RAYS",
    "ROOK_ATTACKS",
    "bishop_attacks",
    "king_attacks",
    "knight_attacks",
    "pawn_attackers",
    "pawn_attacks",
    "queen_attacks",
    "rook_attacks",
]

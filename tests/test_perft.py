"""Perft against published node counts.

Perft is the load-bearing correctness test in this project. It exercises en
passant, promotions, castling rights and pins by sheer volume, and a bug that
hides from the move-list tests shows up here as a wrong number.

Depth 4 over the whole suite is roughly 10.5 million nodes and takes on the order
of ten minutes in pure Python, so the default run stops at depth 3 and the
heavier depths are marked ``slow``. Run them with ``pytest -m slow``.
"""

from __future__ import annotations

import pytest

from conftest import PERFT_SUITE
from foxchess.board import Board
from foxchess.perft import perft

#: Depth 3 is the default ceiling. The counts below are the published ones.
QUICK_DEPTH = 3
#: The two heaviest positions at depth 4, for ``pytest -m slow``.
DEEP_CASES = [
    ("kiwipete", PERFT_SUITE[1][1], 4_085_603),
    ("position6", PERFT_SUITE[5][1], 3_894_594),
]


def test_perft_counts_at_depth_one(board: Board) -> None:
    assert perft(board, 1) == 20


def test_perft_counts_at_depth_two(board: Board) -> None:
    assert perft(board, 2) == 400


@pytest.mark.parametrize(("name", "fen", "counts"), PERFT_SUITE, ids=[c[0] for c in PERFT_SUITE])
def test_published_perft_counts(name: str, fen: str, counts: tuple[int, ...]) -> None:
    board = Board(fen)
    for depth in range(1, QUICK_DEPTH + 1):
        assert depth <= len(counts), f"{name} has no published count for depth {depth}"
        got = perft(board, depth)
        assert got == counts[depth - 1], f"{name} perft({depth})"


def test_perft_does_not_disturb_the_position() -> None:
    board = Board(PERFT_SUITE[1][1])
    before = board.fen()
    key = board.key
    perft(board, 3)
    assert board.fen() == before
    assert board.key == key
    assert board.verify_key()
    assert not board._undo


def test_perft_divides_by_the_root() -> None:
    """Every perft(n) must equal the sum of perft(n - 1) over the root moves."""
    board = Board(PERFT_SUITE[2][1])
    by_move = perft_divide(board, 3)
    assert sum(by_move.values()) == perft(board, 3)
    assert len(by_move) == perft(board, 1)


def perft_divide(board: Board, depth: int) -> dict[str, int]:
    """A perft variant that returns the node count per root move, as UCI."""
    from foxchess.move import to_uci

    result: dict[str, int] = {}
    if depth == 0:
        return {to_uci(m): 1 for m in board.legal_moves()}
    for move in board.legal_moves():
        board.make_move(move)
        result[to_uci(move)] = perft(board, depth - 1)
        board.unmake_move()
    return result


@pytest.mark.slow
@pytest.mark.parametrize(("name", "fen", "expected"), DEEP_CASES, ids=[c[0] for c in DEEP_CASES])
def test_published_perft_counts_at_depth_four(name: str, fen: str, expected: int) -> None:
    assert perft(Board(fen), 4) == expected

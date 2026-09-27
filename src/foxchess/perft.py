"""Perft: count leaf nodes of the move tree to a fixed depth.

Perft is the ground truth for chess move generators. Every number here is
published and independently confirmed by dozens of engines, so a mismatch points
at a concrete bug in a concrete position instead of "the search feels weak".
"""

from __future__ import annotations

from collections.abc import Iterator

from .board import Board


def perft(board: Board, depth: int) -> int:
    """Number of legal move sequences of length ``depth`` from ``board``.

    Iterative deepening keeps the Python call stack shallow and lets the leaf
    count be accumulated without recursion overhead at depth 1.
    """
    if depth <= 0:
        return 1

    nodes = 0
    for move in board.generate_pseudo_legal():
        board.make_move(move)
        if board.in_check(board.side.opposite):
            board.unmake_move()
            continue
        nodes += perft(board, depth - 1) if depth > 1 else 1
        board.unmake_move()
    return nodes


def perft_iterative(board: Board, depth: int) -> Iterator[tuple[int, int]]:
    """Yield ``(depth, nodes)`` for each depth from 1 to ``depth``.

    Faster than :func:`perft` for deep runs because the tree above the final
    level is walked once rather than re-walked.
    """
    total = 0
    for level in range(1, depth + 1):
        total = _perft_divide(board, level)
        yield level, total


def _perft_divide(board: Board, depth: int) -> int:
    nodes = 0
    for move in board.generate_pseudo_legal():
        board.make_move(move)
        if board.in_check(board.side.opposite):
            board.unmake_move()
            continue
        nodes += 1 if depth == 1 else _perft_divide(board, depth - 1)
        board.unmake_move()
    return nodes


def divide(board: Board, depth: int = 1) -> dict[str, int]:
    """Per-move leaf counts, keyed by long algebraic.

    ``divide(board, 2)["e2e4"] == 20`` on the start position. When the total
    disagrees with a published perft number, this is the fastest way to find the
    position where the two branches diverge.
    """
    from .move import to_uci

    result: dict[str, int] = {}
    for move in board.generate_pseudo_legal():
        board.make_move(move)
        if board.in_check(board.side.opposite):
            board.unmake_move()
            continue
        result[to_uci(move)] = 1 if depth <= 1 else _perft_divide(board, depth - 1)
        board.unmake_move()
    return result


def bulk_count(board: Board) -> int:
    """Legal moves at depth 1, without building a full legal list.

    ``bulk_count`` is the per-node workhorse; the difference against
    ``len(board.legal_moves())`` is a cheap self-check of the generator.
    """
    count = 0
    for move in board.generate_pseudo_legal():
        board.make_move(move)
        if not board.in_check(board.side.opposite):
            count += 1
        board.unmake_move()
    return count

"""Shared fixtures and helpers for the foxchess test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from foxchess.board import Board

#: Positions used across more than one test module.
STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
#: 1. e4 Nf6 2. e5 d5, so White really can play exd6 e.p. A fixture that merely
#: *declares* an en-passant square is not enough: the capturing pawn has to be on
#: the fifth rank, which is the mistake this constant exists to prevent.
EN_PASSANT = "rnbqkb1r/ppp2ppp/5n2/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
PROMOTION = "8/P6k/8/8/8/8/6K1/8 w - - 0 1"

#: The six positions from the Chess Programming Wiki perft page, with their
#: published node counts. Depth 1-4 runs in a few seconds; the full table is
#: marked slow.
PERFT_SUITE: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("startpos", STARTPOS, (20, 400, 8902, 197281, 4865609, 119060324)),
    (
        "kiwipete",
        KIWIPETE,
        (48, 2039, 97862, 4085603, 193690690),
    ),
    (
        "position3",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        (14, 191, 2812, 43238, 674624, 11030083),
    ),
    (
        "position4",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        (6, 264, 9467, 422333),
    ),
    (
        "position5",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        (44, 1486, 62379, 2103487, 89941194),
    ),
    (
        "position6",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        (46, 2079, 89890, 3894594),
    ),
)

_CASTLE_SWAP = {"K": "k", "Q": "q", "k": "K", "q": "Q"}


def mirror_fen(fen: str) -> str:
    """Swap colours and rotate 180 degrees.

    A symmetric evaluation must return exactly the negation on the mirrored
    position, which is the strongest check available on the piece-square tables.
    It is what caught the ``sq ^ 56`` versus ``63 - sq`` confusion, and the
    non-palindromic king file weights.

    Reversing the rank order and each rank's file order is the rotation; swapping
    the piece case is the colour change. The en-passant square is dropped because
    nothing in the evaluation reads it.
    """
    placement, side, castling, _ep, halfmove, fullmove = fen.split()
    rotated = "/".join(rank[::-1] for rank in reversed(placement.split("/"))).swapcase()
    rights = sorted(
        (_CASTLE_SWAP[c] for c in castling if c in _CASTLE_SWAP), key="KQkq".index
    )
    flipped = "b" if side == "w" else "w"
    return f"{rotated} {flipped} {''.join(rights) or '-'} - {halfmove} {fullmove}"


@pytest.fixture
def board() -> Board:
    """A fresh starting position."""
    return Board()


@pytest.fixture
def make_board():
    """Factory for boards, so a test can build a position inline."""

    def factory(fen: str = STARTPOS) -> Board:
        return Board(fen)

    return factory

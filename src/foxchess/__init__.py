"""A chess engine in pure Python, with no third-party dependencies.

The shortest useful program::

    import foxchess

    board = foxchess.Board()
    result = foxchess.search(board, foxchess.SearchLimits(depth=4))
    print(result.pv_uci())

The search limits are an object rather than keyword arguments so that a caller
adding a limit later is not breaking the signature of every existing call.

Everything a caller normally needs is re-exported here. The submodules stay
importable for anything deeper -- :mod:`foxchess.notation` for SAN and PGN,
:mod:`foxchess.perft` for move-generation counting, :mod:`foxchess.types` for the
square and piece encodings.
"""

from __future__ import annotations

from foxchess.board import Board
from foxchess.move import (
    NO_MOVE,
    decode_from,
    decode_to,
    from_uci,
    to_uci,
)
from foxchess.notation import Game, algebraic, parse_pgn, parse_san, san, to_pgn
from foxchess.perft import divide, perft
from foxchess.search import (
    MAX_PLY,
    MAX_QUIESCENCE_PLY,
    Searcher,
    SearchLimits,
    SearchResult,
    SearchStopped,
    search,
)
from foxchess.tt import TranspositionTable
from foxchess.types import (
    EMPTY,
    SQUARE_NONE,
    Color,
    PieceType,
    name_of,
    square,
    square_from_name,
)

__version__ = "1.0.0"

__all__ = [
    "EMPTY",
    "MAX_PLY",
    "MAX_QUIESCENCE_PLY",
    "NO_MOVE",
    "SQUARE_NONE",
    "Board",
    "Color",
    "Game",
    "PieceType",
    "SearchLimits",
    "SearchResult",
    "SearchStopped",
    "Searcher",
    "TranspositionTable",
    "__version__",
    "algebraic",
    "decode_from",
    "decode_to",
    "divide",
    "from_uci",
    "name_of",
    "parse_pgn",
    "parse_san",
    "perft",
    "san",
    "search",
    "square",
    "square_from_name",
    "to_pgn",
    "to_uci",
]

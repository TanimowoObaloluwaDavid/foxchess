"""The ``foxchess`` command: analyse positions, count moves, serve the API.

Deliberately thin. Every subcommand is a few lines over the library, so the
interesting behaviour lives in the modules this calls and stays testable without
spawning a process.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from foxchess import __version__
from foxchess.api import MAX_ANALYSIS_DEPTH
from foxchess.board import STARTING_FEN, Board
from foxchess.move import from_uci
from foxchess.notation import Game, san, to_pgn
from foxchess.perft import divide as perft_divide
from foxchess.perft import perft
from foxchess.search import MAX_PLY, Searcher, SearchLimits
from foxchess.tt import TranspositionTable

__all__ = ["main"]


def _render(result: dict[str, object], *, verbose: bool) -> str:
    """Print a search result the way a person reads it."""
    move = result.get("bestmove") or "(none)"
    score = result.get("score", 0)
    mate = result.get("mate")
    if isinstance(mate, int):
        # `mate` is signed in plies: positive means the side to move is delivering
        # it, negative means it is being delivered against us.
        verdict = f"mate in {mate}" if mate > 0 else f"mated in {-mate}"
    else:
        # Scores are from the side to move's point of view, which is the sign
        # convention chess uses.
        centipawns = score if isinstance(score, int) else 0
        verdict = f"{centipawns / 100:+.2f}"
    lines = [
        f"best move  {move}   ({verdict})",
        f"depth      {result.get('depth')}   nodes {result.get('nodes')}   "
        f"{result.get('time_ms')} ms   {result.get('nps')} nps",
    ]
    pv = result.get("pv")
    if isinstance(pv, list) and pv:
        lines.append("line       " + " ".join(str(move) for move in pv))
    if verbose:
        lines.append(f"terminal   {result.get('terminal')}")
    return "\n".join(lines)


def _resolve(tokens: list[str]) -> tuple[str, list[str]]:
    """Work out a position from loose command-line tokens.

    A FEN always has at least six space-separated fields, and a UCI move never
    has any, so the two are told apart by looking for a space. That lets
    ``analyse e2e4 e7e5`` mean "startpos, then these moves" while
    ``analyse "8/8/8/8/8/8/8/8 w - - 0 1"`` still means what it looks like --
    without a flag, a subparser, or a documented convention for which is which.
    """
    if tokens and " " in tokens[0]:
        return tokens[0], tokens[1:]
    return STARTING_FEN, tokens


def _load(tokens: list[str]) -> Board:
    """Build the board named by ``tokens``."""
    fen, moves = _resolve(tokens)
    board = Board(fen)
    for text in moves:
        move = from_uci(text, board.legal_moves())
        if move is None:
            raise ValueError(f"illegal move: {text}")
        board.make_move(move)
    return board


def _command_analyse(args: argparse.Namespace) -> int:
    board = _load(args.position)
    result = Searcher(board, tt=TranspositionTable()).search(
        SearchLimits(depth=args.depth, movetime_ms=args.movetime)
    )
    print(_render(result.as_dict(), verbose=args.verbose))
    return 0


def _command_perft(args: argparse.Namespace) -> int:
    board = _load(args.position)
    if args.divide:
        for move, count in sorted(perft_divide(board, args.depth).items()):
            print(f"{move}: {count}")
    else:
        print(perft(board, args.depth))
    return 0


def _command_fen(args: argparse.Namespace) -> int:
    print(_load(args.position).fen())
    return 0


def _command_moves(args: argparse.Namespace) -> int:
    board = _load(args.position)
    for move in sorted(board.legal_moves(), key=lambda raw: san(board, raw)):
        print(san(board, move))
    return 0


def _command_pgn(args: argparse.Namespace) -> int:
    """Play a short game against itself and print the result as PGN."""
    board = _load(args.position)
    game = Game(
        headers={"Event": "foxchess self-play", "Site": "local"},
        start_fen=None if board.fen() == STARTING_FEN else board.fen(),
    )
    for _ in range(args.plies):
        moves = board.legal_moves()
        if not moves:
            break
        # Deterministic rather than random: this is a formatting demo, and a
        # reproducible game is a far more useful thing to paste into a bug report.
        move = min(moves)
        game.san_moves.append(san(board, move))
        board.make_move(move)
    game.result = board.result()
    print(to_pgn(game))
    return 0


def _command_serve(args: argparse.Namespace) -> int:
    from foxchess.api import serve

    serve(args.host, args.port)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foxchess",
        description="A chess engine in pure Python.",
    )
    parser.add_argument("--version", action="version", version=f"foxchess {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_position(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "position",
            nargs="*",
            metavar="POSITION",
            help=(
                "a FEN, or UCI moves to apply to the start position "
                "(quote a FEN: it contains spaces)"
            ),
        )

    analyse = sub.add_parser("analyse", aliases=["analyze"], help="search a position")
    add_position(analyse)
    analyse.add_argument(
        "-d", "--depth", type=int, default=4, help="search depth (default: 4)"
    )
    analyse.add_argument("-m", "--movetime", type=int, help="stop after this many milliseconds")
    analyse.add_argument("-v", "--verbose", action="store_true", help="show extra fields")
    analyse.set_defaults(run=_command_analyse)

    count = sub.add_parser("perft", help="count leaf nodes, a move-generation self-test")
    add_position(count)
    count.add_argument("-d", "--depth", type=int, default=4, help="depth to count to")
    count.add_argument("--divide", action="store_true", help="count each root move separately")
    count.set_defaults(run=_command_perft)

    show = sub.add_parser("fen", help="print the FEN after applying moves")
    add_position(show)
    show.set_defaults(run=_command_fen)

    listing = sub.add_parser("moves", help="list legal moves in SAN")
    add_position(listing)
    listing.set_defaults(run=_command_moves)

    pgn = sub.add_parser("pgn", help="play a short self-play game and print PGN")
    add_position(pgn)
    pgn.add_argument("-n", "--plies", type=int, default=8, help="half-moves to play")
    pgn.set_defaults(run=_command_pgn)

    server = sub.add_parser("serve", help="run the JSON analysis API")
    server.add_argument("--host", default="127.0.0.1", help="bind address (default: loopback)")
    server.add_argument("-p", "--port", type=int, default=8000, help="port (default: 8000)")
    server.set_defaults(run=_command_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``foxchess`` script. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in {"analyse", "analyze"} and not (
        1 <= args.depth <= min(MAX_ANALYSIS_DEPTH, MAX_PLY - 2)
    ):
        parser.error(f"depth must be between 1 and {MAX_ANALYSIS_DEPTH}")
    if args.command == "perft" and args.depth < 0:
        parser.error("depth must not be negative")

    try:
        return int(args.run(args))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""UCI protocol, the language chess GUIs speak.

Run it as ``foxchess-uci``, or from Python::

    from foxchess.uci import UciSession
    for line in UciSession().handle("position startpos moves e2e4"):
        print(line)

This module owns the *protocol*, not the chess. A session holds a board and a
transposition table, turns ``position``/``go``/``stop`` into search calls, and
formats the replies. Everything it needs comes from the engine proper, so this is
the only place that has to know about threads and output buffering.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterable, Iterator
from typing import TextIO

from foxchess import __version__
from foxchess.board import STARTING_FEN, Board
from foxchess.move import NO_MOVE, from_uci, to_uci
from foxchess.search import (
    MATE_SCORE,
    MATE_THRESHOLD,
    MAX_PLY,
    Searcher,
    SearchLimits,
)
from foxchess.tt import TranspositionTable

__all__ = ["UciSession", "main"]

_AUTHOR = "Tanimowo Obaloluwa"

#: Replayed positions, keyed by FEN plus move list. A GUI exploring a variation
#: re-sends the same moves constantly, and replaying them is most of what a small
#: engine spends its time on. Bounded, and dropped wholesale when full, because
#: the working set is a GUI's current line rather than a growing history.
_POSITION_CACHE: dict[str, Board] = {}
_POSITION_CACHE_LIMIT = 64

#: What ``go`` searches when it is given no limit at all.
_DEFAULT_DEPTH = 4


class UciSession:
    """One engine instance speaking UCI."""

    __slots__ = ("_board", "_out", "_searcher", "_thread", "_tt")

    def __init__(self, out: TextIO | None = None) -> None:
        self._board = Board()
        self._tt = TranspositionTable()
        self._searcher: Searcher | None = None
        self._thread: threading.Thread | None = None
        self._out = sys.stdout if out is None else out

    # ------------------------------------------------------------------
    # Input

    def lines(self, stream: Iterable[str] | None = None) -> Iterator[str]:
        """Yield the reply to every command on ``stream``.

        This is the whole engine loop: read a line, hand it to :meth:`handle`,
        pass on what comes back. Nothing is printed here, so the caller decides
        where the output goes -- :func:`main` writes it to stdout, and a test can
        just read the generator.
        """
        source = sys.stdin if stream is None else stream
        for raw in source:
            text = raw.strip()
            if text:
                yield from self.handle(text)

    def handle(self, text: str) -> list[str]:
        """Handle one command, returning the lines to send back.

        Bad input gets an ``error`` line rather than an exception. A GUI that
        sends something we do not understand would otherwise see a stack trace
        on stderr, and most of them read that as the engine having died.
        """
        parts = text.split()
        if not parts:
            return []
        command, args = parts[0], parts[1:]

        if command in {"uci", "uci_loaded"}:
            return [
                f"id name foxchess {__version__}",
                f"id author {_AUTHOR}",
                "uciok",
            ]
        if command == "isready":
            return self._handle_isready()
        if command == "ucinewgame":
            self._reset()
            return []
        if command == "setoption":
            return self._handle_setoption(args)
        if command == "position":
            self._stop()
            return self._handle_position(args)
        if command == "go":
            return self._handle_go(args)
        if command == "stop":
            self._stop()
            return []
        if command in {"ponderhit", "debug", "register"}:
            # Pondering is never entered, so `ponderhit` has nothing to hit.
            return []
        if command == "quit":
            self._stop()
            return []
        return [f"error unknown command: {command}"]

    # ------------------------------------------------------------------
    # Handlers

    def _handle_isready(self) -> list[str]:
        # Stop before joining, not just join. A bare join here would block until
        # the search finished on its own, and `go infinite` never finishes on its
        # own -- so a GUI that sent `isready` while thinking (which is a normal
        # thing for a GUI to do when its window regains focus) would hang the
        # engine outright. Stopping first keeps the answer meaningful: we are
        # idle by the time we say so.
        self._stop()
        return ["readyok"]

    def _handle_setoption(self, args: list[str]) -> list[str]:
        # Options are accepted and ignored. Advertising a Hash size we cannot
        # honour would be worse than offering none, so a `setoption` is simply
        # absorbed -- except a nameless one, which is a malformed command.
        if "name" not in args:
            return ["error setoption: expected 'name'"]
        return []

    def _handle_position(self, args: list[str]) -> list[str]:
        try:
            self._board = self._build_position(args)
        except ValueError as error:
            return [f"error position: {error}"]
        return []

    def _handle_go(self, args: list[str]) -> list[str]:
        self._stop()
        limits = self._parse_go(args)
        if limits is None:
            return ["error go: unrecognised argument"]

        searcher = Searcher(self._board, tt=self._tt)
        searcher.on_update = self._emit_info
        self._searcher = searcher

        if limits.infinite:
            # `go infinite` has to stay answerable to `stop`, and the thread
            # that reads `stop` is the one blocked by the search. So this is the
            # one case that needs a worker of its own, and the worker is also
            # the one that has to print the `bestmove` when it finishes.
            self._thread = threading.Thread(
                target=self._search_in_background, args=(searcher, limits), daemon=True
            )
            self._thread.start()
            return []
        return self._search_and_reply(searcher, limits)

    # ------------------------------------------------------------------
    # Search plumbing

    def _search_in_background(self, searcher: Searcher, limits: SearchLimits) -> None:
        """Search off-thread and print the reply ourselves.

        The return value of a thread target goes nowhere, so this cannot be the
        same function the synchronous path uses: dropping the result here would
        leave the GUI waiting for a ``bestmove`` that was computed and thrown
        away.
        """
        for reply in self._search_and_reply(searcher, limits):
            self._out.write(reply + "\n")
        self._out.flush()
        self._thread = None

    def _search_and_reply(self, searcher: Searcher, limits: SearchLimits) -> list[str]:
        result = searcher.search(limits)
        self._searcher = None
        if result.best_move == NO_MOVE:
            return ["bestmove 0000"]
        reply = f"bestmove {to_uci(result.best_move)}"
        if len(result.principal_variation) > 1:
            reply += f" ponder {to_uci(result.principal_variation[1])}"
        return [reply]

    def _emit_info(self, depth: int, score: int, line: list[int]) -> None:
        """Write one ``info`` line as each depth completes."""
        self._out.write(f"info depth {depth} score {self._score_text(score)}")
        self._out.write(f" pv {' '.join(to_uci(move) for move in line)}\n")
        self._out.flush()

    @staticmethod
    def _score_text(score: int) -> str:
        """Render a score the way a GUI expects: ``cp`` or ``mate``.

        Internally a mate is a large number, and a GUI reading that as centipawns
        would draw a bar ten thousand wide.
        """
        if score >= MATE_THRESHOLD:
            return f"mate {MATE_SCORE - score}"
        if score <= -MATE_THRESHOLD:
            return f"mate {-(MATE_SCORE + score)}"
        return f"cp {score}"

    def _stop(self) -> None:
        """Stop any running search and wait for it to unwind."""
        if self._searcher is not None:
            self._searcher.stop()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._searcher = None

    def _reset(self) -> None:
        self._stop()
        self._board = Board()
        self._tt.clear()

    # ------------------------------------------------------------------
    # Parsing

    def _build_position(self, args: list[str]) -> Board:
        if not args or args[0] not in {"startpos", "fen"}:
            raise ValueError("expected 'startpos' or 'fen'")

        if args[0] == "startpos":
            fen = STARTING_FEN
            tail = args[1:]
        else:
            tail = args[1:]

        if "moves" in tail:
            cut = tail.index("moves")
            fields, moves = tail[:cut], tail[cut + 1 :]
        else:
            fields, moves = tail, []

        if args[0] == "fen":
            if len(fields) != 6:
                raise ValueError(f"fen takes six fields, got {len(fields)}")
            fen = " ".join(fields)

        return self._replay(fen, moves)

    def _replay(self, fen: str, moves: list[str]) -> Board:
        key = fen + "|" + " ".join(moves)
        cached = _POSITION_CACHE.get(key)
        if cached is not None:
            return cached.copy()

        board = Board(fen)
        for text in moves:
            move = from_uci(text, board.legal_moves())
            if move is None:
                raise ValueError(f"illegal move: {text}")
            board.make_move(move)

        if len(_POSITION_CACHE) >= _POSITION_CACHE_LIMIT:
            _POSITION_CACHE.clear()
        _POSITION_CACHE[key] = board.copy()
        return board

    def _parse_go(self, args: list[str]) -> SearchLimits | None:
        """Turn ``go`` arguments into limits, or ``None`` if they make no sense.

        ``searchmoves`` is parsed and dropped. Restricting the root to a subset
        changes the score, so honouring it partially would be a quiet lie; a GUI
        that uses it will simply get a full-width search.
        """
        limits = SearchLimits()
        index = 0
        bounded = False

        while index < len(args):
            token = args[index]
            if token == "infinite":
                limits.infinite = True
                index += 1
                continue
            if token == "ponder":
                index += 1
                continue
            if token == "searchmoves":
                # Everything up to the next `key value` pair belongs to the list.
                index += 1
                while index < len(args) and args[index] not in _GO_KEYS:
                    index += 1
                continue
            if index + 1 >= len(args) or token not in _GO_KEYS:
                return None
            try:
                value = int(args[index + 1])
            except ValueError:
                return None
            index += 2

            if token == "depth":
                limits.depth, bounded = max(1, value), True
            elif token == "nodes":
                limits.nodes, bounded = max(1, value), True
            elif token == "movetime":
                limits.movetime_ms, bounded = max(1, value), True
            elif token == "wtime":
                limits.wtime_ms, bounded = max(0, value), True
            elif token == "btime":
                limits.btime_ms, bounded = max(0, value), True
            elif token == "winc":
                limits.winc_ms = max(0, value)
            elif token == "binc":
                limits.binc_ms = max(0, value)
            elif token == "movestogo":
                limits.movestogo = max(1, value)

        if not bounded and not limits.infinite:
            limits.depth = _DEFAULT_DEPTH
        if limits.infinite:
            limits.depth = MAX_PLY - 2
        return limits


#: Every ``go`` token that takes a value. Anything else in that position is a
#: malformed command rather than something to guess at.
_GO_KEYS = frozenset(
    {"depth", "nodes", "movetime", "wtime", "btime", "winc", "binc", "movestogo"}
)


def main() -> None:
    """Console-script entry point for ``foxchess-uci``."""
    session = UciSession()
    for reply in session.lines():
        print(reply, flush=True)


if __name__ == "__main__":
    main()

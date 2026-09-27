"""A chess analysis API over HTTP, with no third-party dependencies.

The engine has nothing to do with HTTP, so the server is built on
:mod:`http.server` from the standard library. That keeps the promise in the
package description honest: ``pip install foxchess`` pulls in nothing, and
``foxchess serve`` still gives you a JSON endpoint.

    GET /health          liveness, version, engine name
    GET /fen?fen=...     parse a position, list legal moves
    GET /perft?fen=...   count leaf nodes
    POST /analyze        search a position

The engine is CPU-bound and single-threaded, and Python holds the GIL, so
concurrent searches would not run faster in parallel -- they would just fight
for the interpreter. Requests are therefore serialised behind a lock, which also
keeps the shared transposition table consistent. One position at a time, but
never two searches interleaving on one table.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from foxchess import __version__
from foxchess.board import STARTING_FEN, Board
from foxchess.move import to_uci
from foxchess.perft import divide as perft_divide
from foxchess.perft import perft
from foxchess.search import Searcher, SearchLimits
from foxchess.tt import TranspositionTable

__all__ = ["AnalysisService", "RequestError", "build_server", "serve"]

#: Refuse to count past this. Perft is a debug aid over HTTP, and it runs under
#: the one lock the whole server has, so a request that walks a few million nodes
#: would block every other caller for minutes. Depth 4 from the start position is
#: ~200k nodes and a few seconds; deeper belongs on the command line, where
#: nothing else is waiting behind it.
MAX_PERFT_DEPTH = 4

#: Ceiling on an analysis search, for the same reason. Callers wanting more can
#: use the library directly.
MAX_ANALYSIS_DEPTH = 8


class RequestError(Exception):
    """A problem with the request itself, as opposed to the server."""


@dataclass(frozen=True, slots=True)
class AnalysisService:
    """Position analysis, independent of how it is reached.

    Holding the logic here rather than in the request handler keeps the chess
    testable without a socket, and the handler down to parsing.
    """

    table: TranspositionTable | None = None
    #: The engine is single-threaded, so requests queue here rather than
    #: interleaving on one shared table.
    lock: threading.Lock = field(default_factory=threading.Lock, compare=False)

    # ------------------------------------------------------------------

    def describe(self, fen: str = STARTING_FEN) -> dict[str, Any]:
        """Legal moves and terminal state, without searching."""
        with self.lock:
            board = Board(self._require_fen(fen))
            return {
                "fen": board.fen(),
                "turn": "white" if board.side.name_lower == "white" else "black",
                "legal_moves": sorted(to_uci(move) for move in board.legal_moves()),
                "in_check": board.in_check(),
                "is_checkmate": board.is_checkmate(),
                "is_stalemate": board.is_stalemate(),
                "is_insufficient_material": board.is_insufficient_material(),
                "is_game_over": board.is_game_over(),
                "result": board.result(),
                "key": board.key,
                "key_ok": board.verify_key(),
            }

    def count(self, fen: str, depth: int, *, divide: bool = False) -> dict[str, Any]:
        """Perft, optionally split by root move.

        The split is opt-in because it is not free: a divide is a whole extra
        search, and adding one to the total would mean walking the tree twice for
        a number the total already knows.
        """
        self._require_fen(fen)
        if not 0 <= depth <= MAX_PERFT_DEPTH:
            raise RequestError(f"depth must be between 0 and {MAX_PERFT_DEPTH}")
        with self.lock:
            board = Board(fen)
            if divide and depth:
                counts = perft_divide(board, depth)
                nodes = sum(counts.values())
            else:
                counts = {}
                nodes = perft(board, depth)
            return {
                "fen": board.fen(),
                "depth": depth,
                "nodes": nodes,
                "divide": dict(sorted(counts.items())),
            }

    def analyze(
        self,
        fen: str,
        depth: int = 4,
        movetime_ms: int | None = None,
        nodes: int | None = None,
    ) -> dict[str, Any]:
        """Search a position and return the result as JSON-ready data."""
        self._require_fen(fen)
        if not 1 <= depth <= MAX_ANALYSIS_DEPTH:
            raise RequestError(f"depth must be between 1 and {MAX_ANALYSIS_DEPTH}")
        if movetime_ms is not None and movetime_ms <= 0:
            raise RequestError("movetime_ms must be positive")
        if movetime_ms is not None and depth < MAX_ANALYSIS_DEPTH:
            # Whichever bound bites first wins, so the smaller one must be the
            # one the search actually sees.
            depth = MAX_ANALYSIS_DEPTH

        limits = SearchLimits(
            depth=depth,
            movetime_ms=movetime_ms,
            nodes=nodes,
        )
        with self.lock:
            board = Board(fen)
            result = Searcher(board, tt=self.table or TranspositionTable()).search(limits)
            payload = result.as_dict()
            payload["fen"] = board.fen()
            return payload

    @staticmethod
    def _require_fen(fen: str) -> str:
        """Reject a bad FEN here, so callers see one error shape."""
        text = (fen or "").strip()
        if not text:
            raise RequestError("a 'fen' parameter is required")
        try:
            Board(text)
        except ValueError as error:
            raise RequestError(f"invalid FEN: {error}") from error
        return text


class _Handler(BaseHTTPRequestHandler):
    """Routes requests to :class:`AnalysisService` and renders the JSON."""

    server_version = f"foxchess/{__version__}"
    protocol_version = "HTTP/1.1"
    service: AnalysisService

    # -- routing ------------------------------------------------------

    def do_GET(self) -> None:
        route = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(route.query).items()}
        # Omitting `fen` means the start position, matching what the service
        # methods assume. Being half-optional is worse than either choice.
        fen = query.get("fen", STARTING_FEN)
        try:
            if route.path == "/health":
                self._send(HTTPStatus.OK, {"status": "ok", "version": __version__})
            elif route.path == "/fen":
                self._send(HTTPStatus.OK, self.service.describe(fen))
            elif route.path == "/perft":
                self._send(
                    HTTPStatus.OK,
                    self.service.count(
                        fen,
                        _as_int(query.get("depth"), 1),
                        divide=query.get("divide", "").lower() in {"1", "true", "yes"},
                    ),
                )
            else:
                self._fail(HTTPStatus.NOT_FOUND, f"no such endpoint: {route.path}")
        except RequestError as error:
            self._fail(HTTPStatus.BAD_REQUEST, str(error))
        except Exception as error:
            self._fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"internal error: {error}")

    def do_POST(self) -> None:
        route = urlparse(self.path)
        try:
            body = self._read_json()
            if route.path == "/analyze":
                body.setdefault("fen", STARTING_FEN)
                self._send(HTTPStatus.OK, self.service.analyze(**body))
            else:
                self._fail(HTTPStatus.NOT_FOUND, f"no such endpoint: {route.path}")
        except RequestError as error:
            self._fail(HTTPStatus.BAD_REQUEST, str(error))
        except (TypeError, ValueError) as error:
            self._fail(HTTPStatus.BAD_REQUEST, f"bad request: {error}")
        except Exception as error:
            self._fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"internal error: {error}")

    def log_message(self, _format: str, *args: Any) -> None:
        """Quieter than the default, which timestamps every line to stderr."""

    # -- helpers ------------------------------------------------------

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as error:
            raise RequestError("bad Content-Length") from error
        if length <= 0:
            raise RequestError("a JSON body is required")
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as error:
            raise RequestError(f"invalid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise RequestError("the body must be a JSON object")
        return payload

    def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, status: HTTPStatus, message: str) -> None:
        # A rejected request may not have had its body read, and with HTTP/1.1
        # keep-alive the next request would then be read from the middle of that
        # leftover body. Closing is the honest signal that this connection is
        # finished.
        self.close_connection = True
        self._send(status, {"error": message, "status": int(status)})


def _as_int(text: str | None, fallback: int) -> int:
    try:
        return int(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def build_server(
    host: str = "127.0.0.1", port: int = 8000, service: AnalysisService | None = None
) -> ThreadingHTTPServer:
    """Build a configured server without starting it.

    Separate from :func:`serve` so a caller -- a test, or an application that
    wants the server inside an existing event loop -- can choose the port and
    the lifetime. Port 0 asks the OS for a free one; read the real port back from
    ``server.server_address``.
    """
    bound = service or AnalysisService(table=TranspositionTable())
    handler = type("_BoundHandler", (_Handler,), {"service": bound})
    return ThreadingHTTPServer((host, port), handler)


def serve(host: str = "127.0.0.1", port: int = 8000, *, block: bool = True) -> None:
    """Run the API. Binds to loopback unless told otherwise.

    There is no authentication and no rate limit, so the default is loopback:
    exposing this on a network hands anyone a CPU burner.
    """
    server = build_server(host, port)
    address_host, address_port = server.server_address[:2]
    # An IPv6 socket can hand the address back as bytes; printing that raw would
    # produce a URL nobody can open.
    if isinstance(address_host, bytes):
        address_host = address_host.decode()
    print(f"foxchess {__version__} serving on http://{address_host}:{address_port}")
    if not block:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


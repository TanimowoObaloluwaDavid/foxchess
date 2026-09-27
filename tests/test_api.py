"""Tests for the JSON analysis API.

Most of these drive :class:`AnalysisService` directly, because the chess is the
part worth testing and a socket is not. The handler gets its own smaller set,
including one real round trip over HTTP, since routing and error rendering only
exist in the handler.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import pytest

from foxchess.api import (
    MAX_ANALYSIS_DEPTH,
    MAX_PERFT_DEPTH,
    AnalysisService,
    RequestError,
    build_server,
)
from foxchess.board import STARTING_FEN
from foxchess.tt import TranspositionTable

#: A quiet endgame, so tests do not pay for a middlegame search.
ENDGAME = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
#: Rook to a8 is mate on the spot.
MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"


@pytest.fixture
def service() -> AnalysisService:
    return AnalysisService(table=TranspositionTable())


# ---------------------------------------------------------------------------
# describe


def test_describe_lists_every_legal_move(service: AnalysisService) -> None:
    result = service.describe(STARTING_FEN)
    assert len(result["legal_moves"]) == 20
    assert "e2e4" in result["legal_moves"]
    assert result["turn"] == "white"
    assert result["is_game_over"] is False


def test_describe_reports_check_and_legal_moves_together(
    service: AnalysisService,
) -> None:
    """The classic trap: a position in check whose only moves are evasions."""
    result = service.describe("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1")
    assert result["in_check"] is True
    assert result["legal_moves"], "being in check does not mean being mated"
    assert result["is_checkmate"] is False


def test_describe_reports_a_mate(service: AnalysisService) -> None:
    result = service.describe("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1 moves")
    # Sanity: the fixture is a mate, not a quiet position.
    assert service.describe("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")[
        "is_checkmate"
    ], "the fool's-mate fixture should be checkmate"
    assert result["in_check"] is False


def test_describe_includes_the_zobrist_key_and_its_verification(
    service: AnalysisService,
) -> None:
    result = service.describe(STARTING_FEN)
    assert result["key_ok"] is True
    assert isinstance(result["key"], int)


def test_describe_defaults_to_the_start_position(service: AnalysisService) -> None:
    assert service.describe()["fen"] == STARTING_FEN


@pytest.mark.parametrize("bad", ["", "   ", "not a fen", "8/8/8/8/8/8/8"])
def test_a_bad_fen_is_rejected_with_one_clear_error(
    service: AnalysisService, bad: str
) -> None:
    with pytest.raises(RequestError, match=r"FEN|fen"):
        service.describe(bad)


# ---------------------------------------------------------------------------
# count


def test_count_matches_known_perft_numbers(service: AnalysisService) -> None:
    assert service.count(STARTING_FEN, 1)["nodes"] == 20
    assert service.count(STARTING_FEN, 2)["nodes"] == 400
    assert service.count(STARTING_FEN, 3)["nodes"] == 8902


def test_count_at_depth_zero_is_one_node(service: AnalysisService) -> None:
    assert service.count(STARTING_FEN, 0)["nodes"] == 1


def test_the_split_is_opt_in_because_it_costs_a_second_search(
    service: AnalysisService,
) -> None:
    plain = service.count(STARTING_FEN, 2)
    assert plain["divide"] == {}, "no split unless asked for"

    split = service.count(STARTING_FEN, 2, divide=True)
    assert len(split["divide"]) == 20
    assert sum(split["divide"].values()) == plain["nodes"], "the split must add up"


def test_count_refuses_depths_that_would_hold_the_lock_for_minutes(
    service: AnalysisService,
) -> None:
    with pytest.raises(RequestError, match="depth"):
        service.count(STARTING_FEN, MAX_PERFT_DEPTH + 1)


# ---------------------------------------------------------------------------
# analyze


def test_analyze_returns_a_legal_move_and_a_legal_line(service: AnalysisService) -> None:
    from foxchess.move import from_uci

    result = service.analyze(ENDGAME, depth=3)
    assert result["bestmove"] is not None
    assert from_uci(result["bestmove"]) is not None, "a bestmove must parse back"
    assert result["pv"], "a non-terminal position should yield a line"
    assert result["pv"][0] == result["bestmove"]
    assert result["nodes"] > 0
    assert result["depth"] >= 1


def test_analyze_finds_a_mate_in_one_and_reports_it_in_moves(
    service: AnalysisService,
) -> None:
    result = service.analyze(MATE_IN_ONE, depth=3)
    assert result["bestmove"] == "a1a8"
    assert result["mate"] == 1, "mate in moves, not plies"
    assert result["score"] > 29000, "the internal score is still the large one"


def test_analyze_reports_no_mate_as_a_null(service: AnalysisService) -> None:
    assert service.analyze(ENDGAME, depth=2)["mate"] is None


def test_analyze_can_be_bounded_by_nodes_instead_of_depth(
    service: AnalysisService,
) -> None:
    result = service.analyze(ENDGAME, depth=MAX_ANALYSIS_DEPTH, nodes=500)
    # Limits are polled every 1024 nodes, so the count overshoots by up to one
    # poll interval. The point is that a node budget stops a deep search early,
    # not that it stops on the exact node.
    assert 500 <= result["nodes"] < 500 + 1024, result["nodes"]
    assert result["depth"] < MAX_ANALYSIS_DEPTH, "it should not have reached the bottom"
    assert result["bestmove"] is not None


def test_analyze_defaults_to_a_shallow_but_real_search(service: AnalysisService) -> None:
    result = service.analyze(ENDGAME)
    assert result["bestmove"] is not None


def test_analyze_respects_a_movetime(service: AnalysisService) -> None:
    result = service.analyze(ENDGAME, movetime_ms=150)
    assert result["time_ms"] < 5000
    assert result["bestmove"] is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"depth": 0},
        {"depth": MAX_ANALYSIS_DEPTH + 1},
        {"movetime_ms": 0},
        {"movetime_ms": -1},
    ],
)
def test_analyze_rejects_nonsense_limits(
    service: AnalysisService, kwargs: dict[str, Any]
) -> None:
    with pytest.raises(RequestError):
        service.analyze(ENDGAME, **kwargs)


def test_analyze_does_not_disturb_the_board_it_was_given(
    service: AnalysisService,
) -> None:
    """The service takes a FEN, so it cannot leak state -- but the table is
    shared, and a stale entry there is the thing that would show up here."""
    first = service.analyze(ENDGAME, depth=3)
    second = service.analyze(ENDGAME, depth=3)
    assert first["bestmove"] == second["bestmove"]


# ---------------------------------------------------------------------------
# Over HTTP


@pytest.fixture(scope="module")
def server() -> Iterator[str]:
    """A real server on an ephemeral port, for the routing tests."""
    httpd = build_server("127.0.0.1", 0, AnalysisService(table=TranspositionTable()))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def get(base: str, path: str) -> tuple[int, dict[str, Any]]:
    return raw(base, path, None, {})


def post(base: str, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return raw(
        base,
        path,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )


def raw(
    base: str, path: str, data: bytes | None, headers: dict[str, str]
) -> tuple[int, dict[str, Any]]:
    """Send a request and return ``(status, body)``, including for error statuses.

    Both a response and an ``HTTPError`` own a socket. They have to be closed
    either way, and pytest is configured to turn the resulting ResourceWarning
    into a failure -- so a helper that leaks is a test that passes and then
    breaks the run.
    """
    request = urllib.request.Request(base + path, data=data, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as error:
        with error:
            return error.code, json.loads(error.read())
    with response:
        return response.status, json.loads(response.read())


def test_health_reports_the_version(server: str) -> None:
    status, body = get(server, "/health")
    assert status == 200
    assert body["status"] == "ok"
    assert body["version"]


def test_health_needs_no_parameters_at_all(server: str) -> None:
    """A liveness probe is useless if it can fail on a missing argument."""
    assert get(server, "/health")[0] == 200


def test_fen_over_http_defaults_to_the_start_position(server: str) -> None:
    status, body = get(server, "/fen")
    assert status == 200
    assert len(body["legal_moves"]) == 20


def test_fen_over_http_accepts_a_url_quoted_position(server: str) -> None:
    from urllib.parse import quote

    status, body = get(server, f"/fen?fen={quote(ENDGAME)}")
    assert status == 200
    assert body["fen"] == ENDGAME


def test_perft_over_http(server: str) -> None:
    status, body = get(server, "/perft?depth=2")
    assert status == 200
    assert body["nodes"] == 400


def test_analyze_over_http(server: str) -> None:
    status, body = post(server, "/analyze", {"fen": ENDGAME, "depth": 2})
    assert status == 200
    assert body["bestmove"]
    assert body["score"] == int(body["score"])


def test_analyze_over_http_can_fall_back_to_the_start_position(server: str) -> None:
    status, body = post(server, "/analyze", {"depth": 2})
    assert status == 200
    assert body["fen"] == STARTING_FEN


def test_an_unknown_path_is_a_404_with_json(server: str) -> None:
    status, body = get(server, "/nowhere")
    assert status == 404
    assert "error" in body


def test_a_bad_fen_is_a_400_not_a_500(server: str) -> None:
    """A client sending nonsense is the client's problem, and 500 would send an
    operator looking at server logs for it."""
    status, body = get(server, "/fen?fen=not-a-position")
    assert status == 400
    assert "error" in body


def test_an_unknown_field_is_a_400(server: str) -> None:
    status, _body = post(server, "/analyze", {"nonsense": 1})
    assert status == 400


def test_a_body_that_is_not_json_is_a_400(server: str) -> None:
    status, _ = raw(server, "/analyze", b"not json", {"Content-Type": "application/json"})
    assert status == 400


def test_an_empty_body_is_a_400(server: str) -> None:
    status, _ = raw(server, "/analyze", b"", {"Content-Type": "application/json"})
    assert status == 400


def test_a_json_array_body_is_a_400(server: str) -> None:
    status, _ = raw(server, "/analyze", b"[1,2]", {"Content-Type": "application/json"})
    assert status == 400


def test_a_post_to_a_get_only_path_is_a_404(server: str) -> None:
    status, _ = post(server, "/health", {})
    assert status == 404


def test_the_server_survives_a_bad_request(server: str) -> None:
    """One bad request must not take the process down."""
    assert get(server, "/fen?fen=garbage")[0] == 400
    assert get(server, "/health")[0] == 200


def test_a_rejected_request_does_not_poison_the_next_one(server: str) -> None:
    """An unread body must not be left in the socket for the next reader."""
    for _ in range(3):
        assert get(server, "/fen?fen=garbage")[0] == 400
    assert get(server, "/health")[0] == 200

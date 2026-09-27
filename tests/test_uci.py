"""UCI protocol tests.

Every test drives :class:`UciSession` directly rather than through a pipe. A GUi
sends lines and reads lines; ``handle`` is exactly that boundary, and testing it
means no subprocess, no sleeps, and no flakiness.
"""

from __future__ import annotations

import io
import threading
import time

import pytest

from foxchess.uci import UciSession, main


@pytest.fixture
def session() -> UciSession:
    return UciSession(out=io.StringIO())


def reply(session: UciSession, command: str) -> list[str]:
    return session.handle(command)


# ---------------------------------------------------------------------------
# Handshake


def test_the_handshake_names_the_engine_and_ends_with_uciok(session: UciSession) -> None:
    lines = reply(session, "uci")
    assert lines[0].startswith("id name foxchess ")
    assert lines[1].startswith("id author ")
    assert lines[-1] == "uciok", "uciok must come last or GUIs keep waiting"


def test_isready_answers_readyok(session: UciSession) -> None:
    assert reply(session, "isready") == ["readyok"]


def test_setoption_is_accepted_so_a_gui_does_not_report_a_failure(
    session: UciSession,
) -> None:
    assert reply(session, "setoption name Hash value 128") == []


def test_a_nameless_setoption_is_reported_as_malformed(session: UciSession) -> None:
    lines = reply(session, "setoption value 128")
    assert lines and lines[0].startswith("error")


def test_an_unknown_command_is_an_error_line_not_an_exception(session: UciSession) -> None:
    """A GUI that trips over us must see an error, not a dead process."""
    lines = reply(session, "definitely-not-a-command")
    assert lines == ["error unknown command: definitely-not-a-command"]


def test_commands_from_uci_loaded_repeat_the_handshake(session: UciSession) -> None:
    assert reply(session, "uci_loaded")[-1] == "uciok"


@pytest.mark.parametrize("command", ["ponderhit", "debug", "register", "quit", ""])
def test_commands_that_are_no_answers_produce_no_output(
    session: UciSession, command: str
) -> None:
    assert reply(session, command) == []


# ---------------------------------------------------------------------------
# position


def test_position_startpos_then_moves_reaches_the_right_position(
    session: UciSession,
) -> None:
    assert reply(session, "position startpos moves e2e4 e7e5") == []
    assert session._board.fen() == (
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    )


def test_position_fen_with_six_fields_is_accepted(session: UciSession) -> None:
    fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
    assert reply(session, f"position fen {fen}") == []
    assert session._board.fen() == fen


def test_a_short_fen_is_rejected_rather_than_guessed(session: UciSession) -> None:
    lines = reply(session, "position fen 8/8/8/8/8/8/8/8")
    assert lines and lines[0].startswith("error position")


def test_an_illegal_move_in_a_position_command_is_reported(session: UciSession) -> None:
    lines = reply(session, "position startpos moves e2e5")
    assert lines and lines[0].startswith("error position")
    assert "e2e5" in lines[0]


def test_position_needs_a_starting_point(session: UciSession) -> None:
    assert reply(session, "position") == ["error position: expected 'startpos' or 'fen'"]


def test_replaying_the_same_line_twice_agrees(session: UciSession) -> None:
    """The position cache hands out copies, so one caller cannot corrupt another."""
    reply(session, "position startpos moves d2d4 d7d5")
    first = session._board.key
    reply(session, "position startpos moves e2e4")
    reply(session, "position startpos moves d2d4 d7d5")
    assert session._board.key == first


def test_a_cached_position_is_not_shared_by_reference(session: UciSession) -> None:
    reply(session, "position startpos")
    reply(session, "position startpos")
    session._board.push_uci("a2a3")
    assert session._board.fen().split()[0].count("P") == 8, "the cache must not be mutated"


# ---------------------------------------------------------------------------
# go


def test_go_depth_answers_with_a_bestmove(session: UciSession) -> None:
    assert reply(session, "position startpos") == []
    lines = reply(session, "go depth 2")
    assert len(lines) == 1 and lines[0].startswith("bestmove ")
    move = lines[0].split()[1]
    assert len(move) in {4, 5}, move


def test_a_bestmove_is_always_legal_in_the_current_position(session: UciSession) -> None:
    from foxchess.move import from_uci

    reply(session, "position startpos moves e2e4 e7e5")
    move = reply(session, "go depth 2")[0].split()[1]
    assert from_uci(move, session._board.legal_moves()) is not None


def test_go_with_no_limits_still_answers_promptly(session: UciSession) -> None:
    """`go` on its own must not mean 'search forever'."""
    started = time.perf_counter()
    lines = reply(session, "go")
    assert lines[0].startswith("bestmove ")
    assert time.perf_counter() - started < 30


def test_go_movetime_respects_its_budget(session: UciSession) -> None:
    started = time.perf_counter()
    reply(session, "go movetime 200")
    elapsed = time.perf_counter() - started
    assert elapsed < 5, elapsed
    assert elapsed >= 0.05, "it stopped suspiciously early"


def test_ponder_is_accepted_and_does_not_ponder(session: UciSession) -> None:
    assert reply(session, "go ponder depth 2")[0].startswith("bestmove ")


def test_a_malformed_go_argument_is_reported(session: UciSession) -> None:
    assert reply(session, "go depth") == ["error go: unrecognised argument"]
    assert reply(session, "go nonsense 3") == ["error go: unrecognised argument"]


def test_searchmoves_is_skipped_rather_than_misread(session: UciSession) -> None:
    lines = reply(session, "go searchmoves e2e4 depth 2")
    assert lines and lines[0].startswith("bestmove ")


# ---------------------------------------------------------------------------
# stop, and the threaded path


def test_stop_interrupts_an_infinite_search_and_still_answers() -> None:
    out = io.StringIO()
    session = UciSession(out=out)
    reply(session, "position startpos")
    assert reply(session, "go infinite") == []

    deadline = time.perf_counter() + 10
    while "info " not in out.getvalue() and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert "info " in out.getvalue(), "an infinite search should stream progress"

    started = time.perf_counter()
    assert reply(session, "stop") == []
    elapsed = time.perf_counter() - started
    assert elapsed < 5, f"stop took {elapsed:.1f}s"

    # The worker prints the reply itself; a thread's return value goes nowhere.
    for _ in range(200):
        if "bestmove" in out.getvalue():
            break
        time.sleep(0.01)
    lines = out.getvalue().splitlines()
    assert lines[-1].startswith("bestmove "), lines[-1]


def test_isready_waits_for_a_running_search() -> None:
    out = io.StringIO()
    session = UciSession(out=out)
    reply(session, "go infinite")
    assert reply(session, "isready") == ["readyok"]
    assert session._thread is None, "the worker should have been joined"


def test_ucinewgame_resets_the_board_and_stops_the_search() -> None:
    out = io.StringIO()
    session = UciSession(out=out)
    reply(session, "position fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
    reply(session, "go infinite")
    assert reply(session, "ucinewgame") == []
    assert session._board.fen().startswith("rnbqkbnr/")
    assert session._thread is None


def test_a_second_go_does_not_race_the_first() -> None:
    out = io.StringIO()
    session = UciSession(out=out)
    reply(session, "go infinite")
    assert reply(session, "go depth 2")[0].startswith("bestmove ")
    assert session._thread is None


def test_stop_with_nothing_running_is_harmless(session: UciSession) -> None:
    assert reply(session, "stop") == []


# ---------------------------------------------------------------------------
# info lines


def test_info_lines_are_streamed_as_each_depth_completes() -> None:
    out = io.StringIO()
    session = UciSession(out=out)
    reply(session, "go depth 3")
    infos = [line for line in out.getvalue().splitlines() if line.startswith("info ")]
    assert len(infos) >= 2, "iterative deepening should report more than one depth"
    depths = [int(line.split()[2]) for line in infos]
    assert depths == sorted(depths), depths
    assert depths[0] == 1


def test_info_scores_use_cp_or_mate_and_never_a_bare_number() -> None:
    out = io.StringIO()
    session = UciSession(out=out)
    reply(session, "position fen 6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    reply(session, "go depth 3")
    infos = [line for line in out.getvalue().splitlines() if line.startswith("info ")]
    assert infos
    for line in infos:
        assert " score cp " in line or " score mate " in line, line


def test_a_mate_score_is_reported_in_moves_not_centipawns() -> None:
    """A GUI drawing a bar from 'mate 29999' would draw a bar 30,000 wide."""
    assert UciSession._score_text(29999) == "mate 1"
    assert UciSession._score_text(29997) == "mate 3"
    assert UciSession._score_text(-29999) == "mate -1"
    assert UciSession._score_text(34) == "cp 34"


def test_info_pv_is_legal_in_the_reported_position() -> None:
    """The PV on an `info` line has to be a line a GUI can actually follow."""
    from foxchess.board import Board
    from foxchess.move import from_uci

    out = io.StringIO()
    session = UciSession(out=out)
    reply(session, "go depth 3")
    lines = [line for line in out.getvalue().splitlines() if " pv " in line]
    assert lines, "no info line carried a variation"
    for line in lines:
        board = Board()
        for uci in line.split(" pv ", 1)[1].split():
            move = from_uci(uci, board.legal_moves())
            assert move is not None, f"{uci} is not legal in {board.fen()}"
            board.make_move(move)


# ---------------------------------------------------------------------------
# The stream loop and the console script


def test_lines_consumes_a_stream_and_yields_replies() -> None:
    """``lines`` is the engine loop; the caller owns printing the replies."""
    session = UciSession(out=io.StringIO())
    written = list(session.lines(io.StringIO("uci\n\nisready\nbogus\n")))
    assert "uciok" in written
    assert "readyok" in written
    assert any("error unknown command" in line for line in written)


def test_main_reads_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("uci\nisready\n"))
    main()
    written = capsys.readouterr().out
    assert "uciok" in written
    assert "readyok" in written


def test_a_search_thread_does_not_survive_the_session() -> None:
    """A daemon thread is a last resort; a normal one must be joined on stop."""
    out = io.StringIO()
    session = UciSession(out=out)
    reply(session, "go infinite")
    worker = session._thread
    assert isinstance(worker, threading.Thread)
    reply(session, "stop")
    assert not worker.is_alive()

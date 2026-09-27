"""Tests for the ``foxchess`` command.

Called through :func:`foxchess.cli.main` rather than as a subprocess: the exit
code and the printed text are the contract, and both are reachable this way
without paying for interpreter start-up on every case.
"""

from __future__ import annotations

import pytest

from foxchess.board import STARTING_FEN
from foxchess.cli import _resolve, main

#: Quiet endgame: no promotions, no checks, cheap to search.
ENDGAME = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"


def run(*argv: str) -> int:
    return main(list(argv))


# ---------------------------------------------------------------------------
# Positional handling


def test_a_fen_is_recognised_because_it_contains_spaces() -> None:
    assert _resolve([ENDGAME]) == (ENDGAME, [])


def test_a_bare_token_is_a_move_not_a_fen() -> None:
    """`foxchess fen e2e4` has to mean the move, not a FEN called 'e2e4'."""
    assert _resolve(["e2e4", "e7e5"]) == (STARTING_FEN, ["e2e4", "e7e5"])


def test_nothing_given_means_the_start_position() -> None:
    assert _resolve([]) == (STARTING_FEN, [])


def test_a_fen_followed_by_moves() -> None:
    assert _resolve([ENDGAME, "b4b5"]) == (ENDGAME, ["b4b5"])


# ---------------------------------------------------------------------------
# fen / moves


def test_fen_prints_the_start_position(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("fen") == 0
    assert capsys.readouterr().out.strip() == STARTING_FEN


def test_fen_applies_moves_in_uci(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("fen", "e2e4", "e7e5") == 0
    printed = capsys.readouterr().out.strip()
    assert printed.startswith("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w")


def test_fen_accepts_a_quoted_fen_then_moves(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("fen", ENDGAME) == 0
    assert capsys.readouterr().out.strip() == ENDGAME


def test_fen_rejects_an_illegal_move(
    capsys: pytest.CaptureFixture[str], capsyserr: pytest.CaptureFixture[str] | None = None
) -> None:
    assert run("fen", "e2e5") == 2
    assert "e2e5" in capsys.readouterr().err


def test_moves_lists_san_and_the_count_is_right(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("moves", ENDGAME) == 0
    lines = capsys.readouterr().out.split("\n")
    assert len([line for line in lines if line.strip()]) == 14


def test_moves_uses_san_not_uci(capsys: pytest.CaptureFixture[str]) -> None:
    """SAN is what makes the output readable; a column of 'b4b5' is not."""
    import re

    uci_shape = re.compile(r"^[a-h][1-8][a-h][1-8]")

    run("moves", ENDGAME)
    printed = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert printed
    offenders = [line for line in printed if uci_shape.match(line)]
    assert not offenders, f"these look like UCI moves: {offenders}"


def test_moves_includes_castling_as_san(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run("moves", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    printed = capsys.readouterr().out
    assert "O-O" in printed
    assert "O-O-O" in printed


# ---------------------------------------------------------------------------
# perft


def test_perft_matches_the_published_start_position_numbers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run("perft", "-d", "3") == 0
    assert capsys.readouterr().out.strip() == "8902"


def test_perft_divide_sums_to_the_total(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("perft", "-d", "2", "--divide") == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 20
    total = sum(int(line.rsplit(": ", 1)[1]) for line in lines)
    assert total == 400


def test_perft_rejects_a_negative_depth() -> None:
    with pytest.raises(SystemExit):
        run("perft", "-d", "-1")


# ---------------------------------------------------------------------------
# analyse


def test_analyse_reports_a_move_a_score_and_a_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run("analyse", ENDGAME, "-d", "3") == 0
    printed = capsys.readouterr().out
    assert "best move" in printed
    assert "depth" in printed
    assert "line" in printed


def test_analyse_finds_a_mate_in_one(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("analyse", MATE_IN_ONE, "-d", "3") == 0
    assert "a1a8" in capsys.readouterr().out


def test_a_mate_is_reported_in_moves_not_a_negative_number(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`mate in -1` reads as a mate *for* the opponent, or as no mate at all."""
    run("analyse", MATE_IN_ONE, "-d", "3")
    printed = capsys.readouterr().out
    assert "mate in 1" in printed
    assert "mate in -" not in printed


def test_analyse_accepts_moves_after_the_start_position(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run("analyse", "e2e4", "e7e5", "g1f3", "-d", "2") == 0
    assert "best move" in capsys.readouterr().out


def test_analyse_can_be_bounded_by_time(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("analyse", ENDGAME, "-d", "8", "-m", "150") == 0
    assert "best move" in capsys.readouterr().out


def test_analyse_refuses_a_depth_the_api_would_also_refuse() -> None:
    with pytest.raises(SystemExit):
        run("analyse", "-d", "99")


def test_analyse_verbose_adds_the_terminal_field(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run("analyse", ENDGAME, "-d", "2", "--verbose")
    assert "terminal" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# pgn


def test_pgn_prints_a_parseable_game(capsys: pytest.CaptureFixture[str]) -> None:
    from foxchess.notation import parse_pgn

    assert run("pgn", "-n", "8") == 0
    printed = capsys.readouterr().out
    assert printed.startswith("[Event ")
    game = parse_pgn(printed)
    assert len(game.san_moves) == 8


def test_pgn_records_the_result_when_the_game_is_already_over(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Checkmate on the board: there is nothing to play, and the PGN must say so."""
    from foxchess.notation import parse_pgn

    run("pgn", "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", "-n", "10")
    game = parse_pgn(capsys.readouterr().out)
    assert game.san_moves == []
    assert game.result == "0-1", "White is the one who got mated"


def test_pgn_stops_at_a_stalemate(capsys: pytest.CaptureFixture[str]) -> None:
    from foxchess.notation import parse_pgn

    run("pgn", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "-n", "10")
    game = parse_pgn(capsys.readouterr().out)
    assert game.san_moves == []
    assert game.result == "1/2-1/2"


def test_pgn_leaves_an_unfinished_game_as_a_asterisk(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from foxchess.notation import parse_pgn

    run("pgn", "-n", "6")
    assert parse_pgn(capsys.readouterr().out).result == "*"


def test_pgn_is_reproducible(capsys: pytest.CaptureFixture[str]) -> None:
    run("pgn", "-n", "6")
    first = capsys.readouterr().out
    run("pgn", "-n", "6")
    assert capsys.readouterr().out == first


# ---------------------------------------------------------------------------
# Plumbing


def test_version_is_reported() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0


def test_help_lists_every_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    printed = capsys.readouterr().out
    for command in ("analyse", "perft", "fen", "moves", "pgn", "serve"):
        assert command in printed


def test_no_subcommand_is_a_usage_error() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_analyze_is_spelled_the_american_way_too(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("analyze", ENDGAME, "-d", "2") == 0
    assert "best move" in capsys.readouterr().out


def test_a_malformed_fen_exits_with_a_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run("fen", "this is not a fen at all") == 2
    assert "error" in capsys.readouterr().err

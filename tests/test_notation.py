"""SAN, long algebraic notation, PGN and repetition detection.

Every SAN expectation in this file was cross-checked against python-chess by
`tools/_reference_san.py`, which also confirmed that foxchess produces
byte-identical SAN for *every* legal move in fifteen positions. The disambiguation
rules are about which rival moves are actually legal, not about how many pieces
of that type sit on the board: a pinned knight is not a candidate, and so does not
force a file or rank hint.
"""

from __future__ import annotations

import pytest

from conftest import EN_PASSANT, KIWIPETE, PROMOTION, STARTPOS
from foxchess import move as mv
from foxchess import notation
from foxchess.board import Board
from foxchess.types import Color

BLACK_TO_MOVE = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
# The f1 bishop is boxed in by the e2 pawn at the start, so a bishop that can
# actually move needs an opening first. Castling rights alone are not enough
# either: the squares between king and rook must also be empty and unattacked.
BISHOP_OUT = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 4"
CASTLE_BOTH = "r3k2r/pppppppp/8/8/8/8/8/R3K2R w KQkq - 0 1"
TWO_ROOKS_SAME_FILE = "4k3/8/8/R7/8/8/8/R3K3 w - - 0 1"
TWO_ROOKS_OPEN = "4k3/8/8/8/8/4K3/8/R6R w - - 0 1"
# Two queens on a1 and e1 attack a1-h8, so the black king must sit off that
# diagonal (g8) or the position would be illegal, with Black already in check.
TWO_QUEENS = "6k1/8/8/8/8/8/8/Q3QK2 w - - 0 1"
PINNED_KNIGHT = "4k3/8/8/8/2N5/8/8/2N1K3 w - - 0 1"
KNIGHTS_BLOCKED = "3r2k1/8/8/N7/8/8/8/3N2K1 w - - 0 1"
STALEMATE_ZONE = "7k/5K2/6Q1/8/8/8/8/8 w - - 0 1"
BACK_RANK_MATE = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
BARE_CHECK = "4k3/8/8/8/8/8/8/4K2R w K - 0 1"
# King on h7, not h8: a queen landing on b8 would attack along the eighth rank
# and add a "+" that has nothing to do with the promotion this fixture tests.
# King on h7, not h8: a queen landing on b8 would attack along the eighth rank
# and add a "+" that has nothing to do with the promotion this fixture tests.
PROMOTION_CAPTURE = "1n6/P6k/8/8/8/8/6K1/8 w - - 0 1"


def san_map(board: Board) -> dict[str, str]:
    return {mv.to_uci(m): notation.san(board, m) for m in board.legal_moves()}


def san_of(fen: str, uci: str) -> str:
    board = Board(fen)
    return notation.san(board, mv.from_uci(uci, board.legal_moves()))


# ---------------------------------------------------------------------------
# SAN generation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fen", "uci", "expected"),
    [
        (STARTPOS, "e2e4", "e4"),
        (STARTPOS, "a2a4", "a4"),
        (STARTPOS, "g1f3", "Nf3"),
        (STARTPOS, "b1c3", "Nc3"),
        (STARTPOS, "g1h3", "Nh3"),
        (BISHOP_OUT, "b5c4", "Bc4"),
        (BISHOP_OUT, "b5c6", "Bxc6"),
        (BISHOP_OUT, "b5a4", "Ba4"),
        (BLACK_TO_MOVE, "e7e5", "e5"),
        (BLACK_TO_MOVE, "b8c6", "Nc6"),
        (BLACK_TO_MOVE, "g8f6", "Nf6"),
        (CASTLE_BOTH, "e1g1", "O-O"),
        (CASTLE_BOTH, "e1c1", "O-O-O"),
        (PROMOTION, "a7a8q", "a8=Q"),
        (PROMOTION, "a7a8r", "a8=R"),
        (PROMOTION, "a7a8b", "a8=B"),
        (PROMOTION, "a7a8n", "a8=N"),
        (PROMOTION_CAPTURE, "a7b8q", "axb8=Q"),
        (EN_PASSANT, "e5d6", "exd6"),
    ],
)
def test_san_generation(fen: str, uci: str, expected: str) -> None:
    assert san_of(fen, uci) == expected


def test_san_marks_a_capture() -> None:
    fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    rendered = san_map(Board(fen))
    assert rendered["e4d5"] == "exd5"
    assert rendered["e4e5"] == "e5"


def test_san_marks_a_bare_check() -> None:
    rendered = san_map(Board(BARE_CHECK))
    assert rendered["h1h8"] == "Rh8+"
    assert rendered["h1h7"] == "Rh7"


def test_san_marks_checkmate() -> None:
    rendered = san_map(Board(BACK_RANK_MATE))
    assert rendered["a1a8"] == "Ra8#"
    assert rendered["a1a7"] == "Ra7"


def test_san_does_not_mark_a_stalemate() -> None:
    # Qg7 and Qh6 are mate; Qg5 is neither check nor stalemate-inducing here, and
    # in particular must not pick up a spurious "+" or "#".
    rendered = san_map(Board(STALEMATE_ZONE))
    assert rendered["g6g7"] == "Qg7#"
    assert rendered["g6h6"] == "Qh6#"
    assert rendered["g6g5"] == "Qg5"


def test_rook_disambiguates_by_file() -> None:
    rendered = san_map(Board(TWO_ROOKS_OPEN))
    assert rendered["a1d1"] == "Rad1"
    assert rendered["h1d1"] == "Rhd1"


def test_rook_disambiguates_by_rank_when_both_share_a_file() -> None:
    # Both rooks are on the a-file, so a file hint would still be ambiguous.
    rendered = san_map(Board(TWO_ROOKS_SAME_FILE))
    assert rendered["a1a4"] == "R1a4"
    assert rendered["a5a4"] == "R5a4"


def test_queen_disambiguates_by_file() -> None:
    rendered = san_map(Board(TWO_QUEENS))
    # a1 and e1 share no rank, and both reach e5 and d4.
    assert rendered["a1e5"] == "Qae5"
    assert rendered["e1e5"] == "Qee5"
    # A move only one queen can make needs no hint at all.
    assert rendered["a1a3"] == "Qa3"
    assert rendered["e1d2"] == "Qd2"
    # Both queens reach d1, so the file hint reappears.
    assert rendered["a1d1"] == "Qad1"


def test_a_pinned_rival_does_not_force_disambiguation() -> None:
    # The c4 knight cannot move at all: it is pinned to nothing useful and its
    # only pseudo-moves are illegal. So the c1 knight needs no hint, even though
    # a knight of the same type sits on c4.
    rendered = san_map(Board(PINNED_KNIGHT))
    assert rendered["c1e2"] == "Ne2"
    assert rendered["c4e3"] == "Ne3"
    assert "c4e2" not in rendered, "the pinned knight has no legal move to e2"


def test_a_knight_pinned_to_its_king_is_not_a_candidate() -> None:
    # The d1 knight is pinned by the d8 rook, so it cannot reach d2, and the a5
    # knight therefore needs no hint for a move the pinned knight would also allow.
    rendered = san_map(Board(KNIGHTS_BLOCKED))
    assert rendered["a5c4"] == "Nc4"
    assert "a5d2" not in rendered and "d1d2" not in rendered


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fen", "san", "uci"),
    [
        (STARTPOS, "e4", "e2e4"),
        (STARTPOS, "d4", "d2d4"),
        (STARTPOS, "Nf3", "g1f3"),
        (STARTPOS, "Nc3", "b1c3"),
        (BLACK_TO_MOVE, "e5", "e7e5"),
        (BLACK_TO_MOVE, "Nc6", "b8c6"),
        (EN_PASSANT, "exd6", "e5d6"),
        (CASTLE_BOTH, "O-O", "e1g1"),
        (CASTLE_BOTH, "O-O-O", "e1c1"),
        (PROMOTION, "a8=Q", "a7a8q"),
        (PROMOTION, "a8=N", "a7a8n"),
        (PROMOTION_CAPTURE, "axb8=Q", "a7b8q"),
        (BARE_CHECK, "Rh8+", "h1h8"),
        (BACK_RANK_MATE, "Ra8#", "a1a8"),
        (BISHOP_OUT, "Bxc6", "b5c6"),
    ],
)
def test_san_parsing(fen: str, san: str, uci: str) -> None:
    board = Board(fen)
    assert mv.to_uci(notation.parse_san(board, san)) == uci


def test_san_parsing_requires_an_ambiguity_hint() -> None:
    board = Board(TWO_ROOKS_OPEN)
    # "Rd1" is ambiguous between the a1 and h1 rooks and must be rejected.
    with pytest.raises(notation.NotationError):
        notation.parse_san(board, "Rd1")
    assert mv.to_uci(notation.parse_san(board, "Rad1")) == "a1d1"
    assert mv.to_uci(notation.parse_san(board, "Rhd1")) == "h1d1"


def test_san_parsing_accepts_a_hint_when_one_is_legal_but_unnecessary() -> None:
    # Nc3 is the only legal knight move to c3, so "N1c3" must also be accepted.
    board = Board(STARTPOS)
    assert mv.to_uci(notation.parse_san(board, "N1c3")) == "b1c3"


@pytest.mark.parametrize("bad", ["e5", "Qd4", "e8e9", "Ke2", "zz9", "e4e5", ""])
def test_san_parsing_rejects_illegal_moves(bad: str) -> None:
    with pytest.raises(notation.NotationError):
        notation.parse_san(Board(STARTPOS), bad)


def test_san_parsing_rejects_the_wrong_piece_type() -> None:
    # Bf3 asks a bishop to make a knight's move.
    with pytest.raises(notation.NotationError):
        notation.parse_san(Board(STARTPOS), "Bf3")


def test_san_parsing_rejects_illegal_castling() -> None:
    with pytest.raises(notation.NotationError):
        notation.parse_san(Board(STARTPOS), "O-O-O")


def test_long_algebraic_accepts_san_and_uci() -> None:
    board = Board(STARTPOS)
    for text, uci in (
        ("e4", "e2e4"),
        ("e2e4", "e2e4"),
        ("Nf3", "g1f3"),
        (" e2e4 ", "e2e4"),
    ):
        assert mv.to_uci(notation.parse_long_algebraic(board, text)) == uci


def test_algebraic_is_uci_and_long_algebraic_is_explicit() -> None:
    """`algebraic` is the machine form; `to_long_algebraic` names both squares."""
    board = Board(STARTPOS)
    quiet = mv.from_uci("g1f3", board.legal_moves())
    assert notation.algebraic(board, quiet) == "g1f3"
    assert notation.to_long_algebraic(board, quiet) == "Ng1-f3"
    assert notation.to_long_algebraic(board, mv.from_uci("e2e4", board.legal_moves())) == "e2-e4"

    capturing = Board(EN_PASSANT)
    en_passant = mv.from_uci("e5d6", capturing.legal_moves())
    assert notation.algebraic(capturing, en_passant) == "e5d6"
    assert notation.to_long_algebraic(capturing, en_passant) == "e5xd6"

    promoting = Board(PROMOTION)
    promote = mv.from_uci("a7a8q", promoting.legal_moves())
    assert notation.to_long_algebraic(promoting, promote) == "a7-a8=Q"
    assert notation.to_long_algebraic(promoting, mv.from_uci("a7a8n", promoting.legal_moves())) == (
        "a7-a8=N"
    )

    castles = Board(CASTLE_BOTH)
    assert notation.to_long_algebraic(castles, mv.from_uci("e1g1", castles.legal_moves())) == "O-O"
    assert notation.to_long_algebraic(castles, mv.from_uci("e1c1", castles.legal_moves())) == (
        "O-O-O"
    )


@pytest.mark.parametrize(
    ("fen", "uci", "text"),
    [
        (STARTPOS, "g1f3", "Ng1-f3"),
        (STARTPOS, "e2e4", "e2-e4"),
        (BARE_CHECK, "e1e2", "Ke1-e2"),
        (BARE_CHECK, "h1h7", "Rh1-h7"),
        (EN_PASSANT, "e5d6", "e5xd6"),
        (PROMOTION, "a7a8q", "a7-a8=Q"),
        (PROMOTION_CAPTURE, "a7b8q", "a7xb8=Q"),
    ],
)
def test_long_algebraic_round_trips_through_the_parser(fen: str, uci: str, text: str) -> None:
    """Whatever `to_long_algebraic` prints, the parser must read back."""
    board = Board(fen)
    move = mv.from_uci(uci, board.legal_moves())
    assert notation.to_long_algebraic(board, move) == text
    assert mv.to_uci(notation.parse_long_algebraic(Board(fen), text)) == uci


def test_long_algebraic_parser_is_unambiguous_where_san_is_not() -> None:
    """Two rooks on a1 and h1: SAN needs a hint, long algebraic never does."""
    with pytest.raises(notation.NotationError):
        notation.parse_san(Board(TWO_ROOKS_OPEN), "Rd1")
    for uci, text in (("a1d1", "Ra1-d1"), ("h1d1", "Rh1-d1")):
        assert mv.to_uci(notation.parse_long_algebraic(Board(TWO_ROOKS_OPEN), text)) == uci


@pytest.mark.parametrize(
    "fen", [STARTPOS, KIWIPETE, EN_PASSANT, PROMOTION, PROMOTION_CAPTURE, BACK_RANK_MATE]
)
def test_san_round_trips_for_every_legal_move(fen: str) -> None:
    """Generate SAN, parse it back, and require the identical move."""
    board = Board(fen)
    for move in board.legal_moves():
        text = notation.san(board, move)
        assert mv.to_uci(notation.parse_san(Board(fen), text)) == mv.to_uci(move), text


def test_square_helpers() -> None:
    assert notation.square_index("a1") == 0
    assert notation.square_index("h8") == 63
    assert notation.file_rank(0) == "a1"
    assert notation.file_rank(63) == "h8"
    assert notation.side_name(Color.WHITE) == "white"
    assert notation.side_name(Color.BLACK) == "black"


# ---------------------------------------------------------------------------
# Games and PGN
# ---------------------------------------------------------------------------

OPENING = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]


def test_game_numbers_only_the_white_moves() -> None:
    # PGN writes "1. e4 e5", so the number appears on White's move and is elided
    # on Black's. `numbered_moves` yields None for the elided half.
    game = notation.Game(san_moves=OPENING[:4], result="1-0")
    assert list(game.numbered_moves()) == [
        (1, "e4"),
        (None, "e5"),
        (2, "Nf3"),
        (None, "Nc6"),
    ]
    assert game.ply() == 4
    assert game.result == "1-0"
    assert game.start_move_number() == 1


def test_game_replays_to_the_same_position() -> None:
    game = notation.Game(san_moves=OPENING)
    board = game.replay()
    assert board.ply == len(OPENING)
    assert board.verify_key()
    # 1.e4 e5 2.Nf3 Nc6 3.Bb5 a6 -- Black has just moved, so it is White's turn
    # on move 4 with a fresh halfmove clock.
    assert board.fen() == "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
    assert [game.side_at(i) for i in range(len(OPENING))] == [
        Color.WHITE, Color.BLACK, Color.WHITE, Color.BLACK, Color.WHITE, Color.BLACK,
    ]


def test_pgn_round_trip() -> None:
    game = notation.Game(
        san_moves=OPENING,
        result="1-0",
        headers={"White": "foxchess", "Black": "Stockfish"},
    )
    text = notation.to_pgn(game)
    assert '[White "foxchess"]' in text
    assert '[Black "Stockfish"]' in text
    assert text.rstrip().endswith("1-0")

    parsed = notation.parse_pgn(text)
    assert parsed.san_moves == OPENING
    assert parsed.result == "1-0"
    assert parsed.headers["White"] == "foxchess"
    assert parsed.replay().fen() == game.replay().fen()


def test_pgn_preserves_a_start_fen() -> None:
    """A game that did not start from the opening must carry its position.

    Without the SetUp/FEN tag pair, a parsed game silently restarts from the
    standard position and `replay()` then raises on the first move.
    """
    game = notation.Game(san_moves=["O-O"], start_fen=KIWIPETE)
    text = notation.to_pgn(game)
    assert '[SetUp "1"]' in text
    assert f'[FEN "{KIWIPETE}"]' in text

    parsed = notation.parse_pgn(text)
    assert parsed.start_fen == KIWIPETE
    assert parsed.san_moves == ["O-O"]
    assert parsed.replay().fen() == game.replay().fen()


def test_pgn_omits_the_fen_tag_for_a_standard_game() -> None:
    text = notation.to_pgn(notation.Game(san_moves=OPENING, result="1-0"))
    assert "[SetUp" not in text
    assert "[FEN" not in text
    assert notation.parse_pgn(text).start_fen is None


@pytest.mark.parametrize("result", ["1-0", "0-1", "1/2-1/2", "*"])
def test_every_pgn_result_survives(result: str) -> None:
    """The result must come from the movetext when the tag is absent.

    `_tokenise` used to drop result tokens before the parser could see them, so
    every game parsed back as "*" unless a Result tag happened to be present.
    """
    text = notation.to_pgn(notation.Game(san_moves=OPENING, result=result))
    body = text[text.rindex("]") + 1 :]  # drop the tags entirely
    assert notation.parse_pgn(body).result == result


def test_pgn_ignores_comments_variations_and_annotations() -> None:
    text = (
        '[Event "Test"]\n\n1. e4 {best by test} e5 ; a line comment\n'
        "2. Nf3 (2. f4 exf4) Nc6 $13 3. Bb5 a6 1-0\n"
    )
    game = notation.parse_pgn(text)
    assert game.san_moves == ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]
    assert game.result == "1-0"


def test_pgn_wraps_long_movetext() -> None:
    game = notation.Game(san_moves=OPENING * 12, result="*")
    text = notation.to_pgn(game, width=40)
    body = [line for line in text.splitlines() if line and not line.startswith("[")]
    assert all(len(line) <= 40 for line in body)


def test_pgn_of_an_empty_game() -> None:
    parsed = notation.parse_pgn(notation.to_pgn(notation.Game(san_moves=[], result="*")))
    assert parsed.san_moves == []
    assert parsed.result == "*"


# ---------------------------------------------------------------------------
# Repetition
# ---------------------------------------------------------------------------


def test_repetition_tracker_counts_occurrences_of_the_current_position() -> None:
    board = Board()
    tracker = notation.RepetitionTracker()
    tracker.record(board)
    assert tracker.count() == 1
    assert not tracker.is_threefold()

    # Nf3 Nf6 Ng1 Ng8 returns every piece home, so the start position has now
    # occurred twice. `count()` reports the *current* position, so ask there.
    start_key = board.key
    cycle = ("g1f3", "g8f6", "f3g1", "f6g8")
    for uci in cycle:
        board.push_uci(uci)
        tracker.record(board)
    # The knights are home again, so this is a repetition -- even though the FEN's
    # move counters have moved on, which is exactly why repetition is keyed on the
    # position rather than on the whole FEN.
    assert board.key == start_key
    assert board.fen().split()[4:] != STARTPOS.split()[4:]
    assert tracker.count() == 2
    assert not tracker.is_threefold()

    # One more time round and the start position is a genuine threefold.
    for uci in cycle:
        board.push_uci(uci)
        tracker.record(board)
    assert board.key == start_key
    assert tracker.count() == 3
    assert tracker.is_threefold()
    assert not tracker.is_fivefold()


def test_repetition_tracker_undo_and_reset() -> None:
    board = Board()
    tracker = notation.RepetitionTracker()
    tracker.record(board)
    board.push_uci("g1f3")
    tracker.record(board)
    tracker.undo()
    assert tracker.count() == 1
    tracker.reset()
    assert tracker.count() == 0


def test_repetition_window_is_bounded() -> None:
    tracker = notation.RepetitionTracker(window=8)
    for _ in range(20):
        tracker.record(Board())
    assert len(tracker.keys) == 8

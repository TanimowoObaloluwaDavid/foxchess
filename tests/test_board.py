"""Board state, move application and rule edge cases.

Every expected move count in this file was cross-checked against python-chess by
`tools/_reference.py`. Perft already pins move generation to depth 4 on the six
reference positions, so if a test here fails on a count, the count is what is
wrong, not the engine.
"""

from __future__ import annotations

import pytest

from conftest import EN_PASSANT, KIWIPETE, PROMOTION, STARTPOS
from foxchess import move as mv
from foxchess.board import Board
from foxchess.types import (
    SQUARE_NAMES,
    Color,
    PieceType,
    make_piece,
    square_from_name,
)

CASTLING_THROUGH_CHECK = "4k1r1/8/8/8/8/8/8/R3K2R w KQ - 0 1"
IN_CHECK_NO_CASTLE = "4k3/8/8/8/8/8/4r3/R3K2R w KQ - 0 1"
PINNED_ROOK = "4r2k/8/8/8/8/8/4R3/4K3 w - - 0 1"
DEFENDED_PIECE = "4k3/8/8/8/8/4r3/4r3/4K3 w - - 0 1"
BLACK_MATED = "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1"
BLACK_STALEMATED = "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"
WHITE_MATED = "8/8/8/8/8/5k2/6q1/7K w - - 0 1"


def test_starting_position_layout(board: Board) -> None:
    assert board.fen() == STARTPOS
    assert board.count(PieceType.PAWN, Color.WHITE) == 8
    assert board.count(PieceType.KING, Color.BLACK) == 1
    assert board.in_check() is False
    assert len(board.legal_moves()) == 20


@pytest.mark.parametrize("fen", [STARTPOS, KIWIPETE, EN_PASSANT, PROMOTION, BLACK_MATED])
def test_fen_round_trip(fen: str) -> None:
    assert Board(fen).fen() == fen


def test_ascii_shows_every_rank(board: Board) -> None:
    rendered = board.ascii()
    assert rendered.count("\n") >= 8
    for rank in "87654321":
        assert f"{rank} |" in rendered
    assert "white to move" in rendered


@pytest.mark.parametrize(
    ("fen", "expected"),
    [
        (STARTPOS, 20),
        ("4k3/8/8/8/8/8/8/4K2R w K - 0 1", 15),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", 26),
        (CASTLING_THROUGH_CHECK, 25),
        (IN_CHECK_NO_CASTLE, 3),
        (PINNED_ROOK, 10),
        (DEFENDED_PIECE, 2),
        (EN_PASSANT, 32),
        (PROMOTION, 12),
        (WHITE_MATED, 0),
        (BLACK_MATED, 0),
        (BLACK_STALEMATED, 0),
    ],
)
def test_legal_move_counts(fen: str, expected: int) -> None:
    assert len(Board(fen).legal_moves()) == expected


def test_pawn_double_push_is_flagged() -> None:
    board = Board()
    move = mv.from_uci("e2e4", board.legal_moves())
    assert mv.decode_flag(move) == mv.DOUBLE_PUSH
    board.make_move(move)
    # The double push happened, but with no Black pawn beside the e4 pawn there is
    # no capture to make, so the FEN does not advertise an ep square.
    assert board.ep_square == square_from_name("e3")
    assert board.ep_fen() == "-"
    assert board.fen() == ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")


def test_double_push_records_an_ep_square_when_a_capture_exists() -> None:
    # 1. e4 Nf6 2. exd5 d5 -- White's pawn now stands on e5, so the d6 target is
    # genuinely capturable and belongs in the FEN.
    board = Board()
    for uci in ("e2e4", "g8f6", "e4e5", "d7d5"):
        board.push_uci(uci)
    assert board.ep_square == square_from_name("d6")
    assert board.ep_fen() == "d6"
    assert board.fen().split()[3] == "d6"
    assert board.is_legal(mv.from_uci("e5d6", board.legal_moves()))


def test_double_push_omits_an_uncapturable_ep_square() -> None:
    # 1. e4 d5 -- the same double push, but White's pawn is still on e4. An
    # en-passant capture needs the capturing pawn on the fifth rank, so nothing
    # can be taken and the FEN says so rather than advertising a capture that does
    # not exist.
    board = Board()
    board.push_uci("e2e4")
    board.push_uci("d7d5")
    assert board.ep_square == square_from_name("d6"), "the square is still tracked internally"
    assert board.ep_fen() == "-"
    assert board.fen().split()[3] == "-"


def test_en_passant_captures_the_skipped_pawn_and_lands_on_the_target() -> None:
    board = Board(EN_PASSANT)
    move = mv.from_uci("e5d6", board.legal_moves())
    assert mv.decode_flag(move) == mv.EN_PASSANT
    assert mv.is_capture_flag(move)
    board.make_move(move)
    assert board.piece_at(square_from_name("d5")) == 0, "the skipped pawn left d5"
    assert board.piece_at(square_from_name("d6")) == make_piece(
        PieceType.PAWN, Color.WHITE
    ), "our pawn landed on d6"
    assert board.ep_fen() == "-"
    assert board.fen() == "rnbqkb1r/ppp2ppp/3P1n2/8/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3"


def test_en_passant_right_appears_only_immediately() -> None:
    board = Board(EN_PASSANT)
    assert board.ep_fen() == "d6"
    board.push_uci("g1f3")
    assert board.ep_fen() == "-"
    board.unmake_move()
    assert board.ep_fen() == "d6"


def test_all_four_promotions_are_generated() -> None:
    board = Board(PROMOTION)
    promotions = sorted(
        mv.promotion_of(m) for m in board.legal_moves() if mv.is_promotion_flag(m)
    )
    assert promotions == [
        PieceType.KNIGHT,
        PieceType.BISHOP,
        PieceType.ROOK,
        PieceType.QUEEN,
    ]


@pytest.mark.parametrize("suffix", ["q", "r", "b", "n"])
def test_promotion_round_trips_through_unmake(suffix: str) -> None:
    board = Board(PROMOTION)
    before = board.fen()
    board.make_move(mv.from_uci(f"a7a8{suffix}"))
    assert board.fen().split()[0].split("/")[0][0] == suffix.upper()
    assert board.count(PieceType.PAWN, Color.WHITE) == 0
    board.unmake_move()
    assert board.fen() == before
    assert board.verify_key()


def test_castling_both_sides() -> None:
    board = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    castles = {
        mv.to_uci(m) for m in board.legal_moves() if mv.decode_flag(m) in mv.CASTLE_FLAGS
    }
    assert castles == {"e1g1", "e1c1"}


def test_castling_is_illegal_through_an_attacked_square() -> None:
    # The Black rook on g8 covers g1, so the king may not step through it.
    board = Board(CASTLING_THROUGH_CHECK)
    castles = {
        mv.to_uci(m) for m in board.legal_moves() if mv.decode_flag(m) in mv.CASTLE_FLAGS
    }
    assert "e1g1" not in castles
    assert "e1c1" in castles


def test_king_cannot_castle_out_of_check() -> None:
    board = Board(IN_CHECK_NO_CASTLE)
    assert board.in_check()
    castles = {
        mv.to_uci(m) for m in board.legal_moves() if mv.decode_flag(m) in mv.CASTLE_FLAGS
    }
    assert castles == set()


def test_a_pinned_piece_may_only_stay_on_the_pin() -> None:
    board = Board(PINNED_ROOK)
    rook = make_piece(PieceType.ROOK, Color.WHITE)
    rook_moves = [m for m in board.legal_moves() if board.piece_at(mv.decode_from(m)) == rook]
    # The rook on e2 is pinned to the king on e1 by the rook on e8, so every move
    # it is allowed keeps it on the e-file.
    assert {SQUARE_NAMES[mv.decode_to(m)] for m in rook_moves} == {
        "e3", "e4", "e5", "e6", "e7", "e8",
    }


def test_king_cannot_capture_a_defended_piece() -> None:
    # The rook on e2 is defended by the rook on e3, so Kxe2 loses the king. The
    # rook on e2 also covers the whole of rank 2, leaving only d1 and f1.
    board = Board(DEFENDED_PIECE)
    assert not board.is_legal(mv.from_uci("e1e2"))
    assert {mv.to_uci(m) for m in board.legal_moves()} == {"e1d1", "e1f1"}


def test_checkmate_and_stalemate_are_distinguished() -> None:
    mated = Board(BLACK_MATED)
    assert mated.is_checkmate()
    assert not mated.is_stalemate()
    assert mated.is_game_over()
    # The side to move is the side that lost.
    assert mated.result() == "1-0"

    stalemated = Board(BLACK_STALEMATED)
    assert stalemated.is_stalemate()
    assert not stalemated.is_checkmate()
    assert stalemated.result() == "1/2-1/2"


@pytest.mark.parametrize(
    ("fen", "insufficient"),
    [
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", True),
        ("4k3/8/8/8/8/8/8/3BK3 w - - 0 1", True),
        ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", False),
        ("4k3/8/8/8/8/8/4P3/3RK3 w - - 0 1", False),
    ],
)
def test_insufficient_material(fen: str, insufficient: bool) -> None:
    assert Board(fen).is_insufficient_material() is insufficient


def test_fifty_and_seventy_five_move_clocks() -> None:
    board = Board("4k3/8/8/8/8/8/4P3/4K3 w - - 99 120")
    assert not board.is_fifty_move()
    board.make_null_move()
    assert board.halfmove_clock == 100
    assert board.is_fifty_move()
    assert not board.is_seventy_five_move()
    for _ in range(50):
        board.make_null_move()
    assert board.is_seventy_five_move()


def test_null_move_passes_the_turn_and_undoes_cleanly() -> None:
    board = Board()
    before = board.fen()
    board.make_null_move()
    assert board.side is Color.BLACK
    assert board.ply == 1
    assert board.ep_fen() == "-"
    board.unmake_null_move()
    assert board.fen() == before
    assert board.verify_key()


def test_push_uci_applies_a_move() -> None:
    board = Board()
    board.push_uci("e2e4")
    assert board.fen() == ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    board.push_uci("e7e5")
    assert board.fen() == ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")


def test_push_uci_rejects_an_illegal_move() -> None:
    with pytest.raises(ValueError):
        Board().push_uci("e2e5")


def test_push_uci_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        Board().push_uci("not a move")


def test_a_full_move_sequence_unmakes_exactly() -> None:
    board = Board(KIWIPETE)
    history: list[str] = []
    while (moves := board.legal_moves()) and len(history) < 11:
        # Record where we are *before* each move, so undoing move N lands exactly
        # on history[N]. Recording the post-move FEN instead is off by one ply and
        # fails on a position that is in fact perfect.
        history.append(board.fen())
        board.make_move(moves[0])
        assert board.verify_key(), "the incremental key drifted mid-search-tree"
    assert len(history) == 11
    while history:
        board.unmake_move()
        assert board.fen() == history.pop(), "unmake did not restore the position"
    assert board.fen() == KIWIPETE
    assert board.verify_key()
    assert not board._undo


def test_copy_is_independent() -> None:
    board = Board()
    clone = board.copy()
    clone.push_uci("e2e4")
    assert board.fen() == STARTPOS
    assert clone.fen() != STARTPOS


def test_a_repeated_sequence_restores_the_original_key() -> None:
    board = Board()
    key = board.compute_key()
    for _ in range(4):
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
            board.push_uci(uci)
            assert board.verify_key()
    assert board.compute_key() == key

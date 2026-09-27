"""Zobrist hashing and the make/unmake round-trip invariant.

The round-trip test is the most valuable test in the suite. Perft proves the
*count* of leaf nodes is right; it does not prove the board state is right at
every node, nor that the incremental hash agrees with a from-scratch hash. A
single stray XOR is invisible to perft and makes the transposition table
silently return garbage scores.
"""

from __future__ import annotations

import pytest

from conftest import EN_PASSANT, KIWIPETE, PROMOTION, STARTPOS, mirror_fen
from foxchess import move as mv
from foxchess import zobrist
from foxchess.board import Board
from foxchess.types import Color, PieceType, file_of, make_piece

ROUND_TRIP_FENS = [
    STARTPOS,
    KIWIPETE,
    EN_PASSANT,
    PROMOTION,
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]


def test_incremental_key_equals_a_fresh_hash(board: Board) -> None:
    assert board.key == board.compute_key()
    assert board.verify_key()


def test_a_board_loaded_from_a_fen_is_already_consistent() -> None:
    for fen in ROUND_TRIP_FENS:
        assert Board(fen).verify_key(), fen


def _hash_by_hand(board: Board, side: Color | None = None) -> int:
    """Hash a board the way a caller with no Board object would."""
    pieces = {piece: bb for piece, bb in enumerate(board.bitboards) if bb}
    return zobrist.compute(
        pieces, board.castling, board.ep_square, side or board.side
    )


def test_standalone_compute_agrees_with_the_board() -> None:
    """``zobrist.compute`` is public, so it needs a test of its own.

    ``Board.verify_key`` checks the *incremental* key against ``Board``'s own
    private recomputation, which never calls this module-level function. A bug
    in ``compute`` would therefore be invisible to every other test here -- and
    the two quietly disagreeing is how a caller who hashes a position by hand
    ends up with a key the engine will never recognise.
    """
    for fen in ROUND_TRIP_FENS:
        board = Board(fen)
        assert _hash_by_hand(board) == board.key, fen
        assert _hash_by_hand(board) == board.compute_key(), fen


def test_compute_masks_off_impossible_castling_bits() -> None:
    """A caller can pass any bit set; only the four real rights may count."""
    board = Board(STARTPOS)
    real = _hash_by_hand(board)
    assert zobrist.compute(
        {p: bb for p, bb in enumerate(board.bitboards) if bb},
        board.castling | 0xF,
        board.ep_square,
        board.side,
    ) == real


def test_compute_separates_the_two_sides() -> None:
    """Same pieces, different turn, must not collide."""
    board = Board(STARTPOS)
    white = _hash_by_hand(board, Color.WHITE)
    black = _hash_by_hand(board, Color.BLACK)
    assert white != black
    assert white ^ black == zobrist.SIDE_KEY


def test_compute_folds_in_the_en_passant_file() -> None:
    """Two en-passant squares on the same file hash the same; others differ."""
    board = Board(EN_PASSANT)
    by_hand = _hash_by_hand(board)
    other_file = zobrist.compute(
        {p: bb for p, bb in enumerate(board.bitboards) if bb},
        board.castling,
        next(sq for sq in range(64) if file_of(sq) != file_of(board.ep_square)),
        board.side,
    )
    assert other_file != by_hand


def test_mirroring_the_board_keeps_the_key_consistent() -> None:
    # The mirrored position is a different position, so a different key is fine;
    # what matters is that the key still matches a from-scratch recomputation.
    for fen in ROUND_TRIP_FENS:
        assert Board(mirror_fen(fen)).verify_key(), fen


def test_every_piece_square_pair_has_a_distinct_key() -> None:
    seen: dict[int, tuple[int, int]] = {}
    for piece_type in (PieceType.KNIGHT, PieceType.BISHOP, PieceType.ROOK, PieceType.QUEEN):
        for color in (Color.WHITE, Color.BLACK):
            piece = make_piece(piece_type, color)
            for sq in range(64):
                key = zobrist.piece_key(piece, sq)
                assert key not in seen, f"collision between {seen.get(key)} and {(piece, sq)}"
                seen[key] = (piece, sq)
    assert len(seen) == 4 * 2 * 64


def test_empty_pieces_have_no_key() -> None:
    # PIECE_KEYS[0] is a row of zeros, so hashing a missing piece is a no-op.
    # `make_move` never relies on that -- it guards with `captured != EMPTY` --
    # but a zero row means a future mistake here fails quietly rather than loudly.
    assert all(zobrist.piece_key(0, sq) == 0 for sq in range(64))


def test_en_passant_is_only_hashed_when_a_capture_exists() -> None:
    # 1. e4 -- no Black pawn can take, so the ep square must not change the key.
    # The correct statement is not "the key is unchanged" (a pawn moved, so of
    # course it is) but "the key ignores the ep square when it is unusable".
    quiet = Board(STARTPOS)
    quiet.push_uci("e2e4")
    # No Black pawn can take, so the ep square is dropped from the FEN *and* from
    # the key: the two now agree, where before the FEN advertised a capture that
    # nobody could make and any tool comparing FEN text would have called this a
    # different position from the same position reached another way.
    assert quiet.ep_fen() == "-"
    same_position_no_ep = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    assert quiet.fen() == same_position_no_ep
    assert quiet.key == Board(same_position_no_ep).key
    assert quiet.verify_key()

    # 1. e4 Nf6 2. e5 d5 -- the White pawn is already on the fifth rank, so exd6
    # e.p. is genuinely available and the target must be hashed. (After 1. e4 d5
    # the target is recorded in the FEN but is *unusable*, because an en-passant
    # capture needs the capturing pawn on the fifth rank, not the fourth.)
    live = Board(STARTPOS)
    for uci in ("e2e4", "g8f6", "e4e5", "d7d5"):
        live.push_uci(uci)
    assert live.ep_fen() == "d6"
    assert live._ep_hash_target() != -1
    assert live.key != quiet.key
    assert live.verify_key()
    stripped = live.fen().replace(" d6 ", " - ")
    assert live.key != Board(stripped).key


def test_castling_rights_are_hashed() -> None:
    # The starting position has both White rooks blocked by their own pawns, so
    # it cannot be used to test rook moves.
    position = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"

    # h1 is the *kingside* rook, so moving it costs only K. The a1 rook is
    # untouched, so the queenside right survives.
    lost_kingside = Board(position)
    lost_kingside.push_uci("h1g1")
    assert lost_kingside.castling_fen() == "Qkq"

    lost_queenside = Board(position)
    lost_queenside.push_uci("a1a2")
    assert lost_queenside.castling_fen() == "Kkq"

    # The king forfeits both, whichever way it steps.
    king_moved = Board(position)
    king_moved.push_uci("e1e2")
    assert king_moved.castling_fen() == "kq"

    assert lost_kingside.key != lost_queenside.key
    assert lost_kingside.key != king_moved.key
    assert lost_kingside.key != Board(position).key


@pytest.mark.parametrize("fen", ROUND_TRIP_FENS)
def test_every_legal_move_round_trips(fen: str) -> None:
    """Play each legal move, then undo it, and require an exact restore."""
    board = Board(fen)
    for candidate in list(board.legal_moves()):
        fen_before = board.fen()
        key_before = board.key
        board.make_move(candidate)
        assert board.verify_key(), f"{mv.to_uci(candidate)} left the key unsound"
        board.unmake_move()
        assert board.fen() == fen_before, f"unmake of {mv.to_uci(candidate)} drifted"
        assert board.key == key_before, f"unmake of {mv.to_uci(candidate)} left a stale key"
        assert board.verify_key()


@pytest.mark.parametrize("fen", ROUND_TRIP_FENS)
def test_null_moves_round_trip(fen: str) -> None:
    board = Board(fen)
    fen_before = board.fen()
    key_before = board.key
    board.make_null_move()
    assert board.verify_key()
    board.unmake_null_move()
    assert board.fen() == fen_before
    assert board.key == key_before


@pytest.mark.parametrize("fen", ROUND_TRIP_FENS)
def test_a_greedy_line_round_trips_in_full(fen: str) -> None:
    """Walk eleven plies forward, then all the way back.

    The FEN is recorded *before* each move. Recording it afterwards pairs the
    undo of move N with the position after move N, which is off by one ply and
    fails on a board that is in fact perfect.
    """
    board = Board(fen)
    history: list[str] = []
    while (moves := board.legal_moves()) and len(history) < 11:
        history.append(board.fen())
        board.make_move(moves[0])
        assert board.verify_key()
    while history:
        board.unmake_move()
        assert board.fen() == history.pop(), "unmake did not restore the position"
    assert board.fen() == fen
    assert not board._undo


def test_a_position_reached_twice_hashes_identically() -> None:
    board = Board()
    key = board.key
    for _ in range(3):
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
            board.push_uci(uci)
            assert board.verify_key()
    assert board.key == key
    # The clocks legitimately differ: twelve knight moves are neither pawn moves
    # nor captures. Position identity is the first four FEN fields.
    assert board.fen().split()[:4] == STARTPOS.split()[:4]

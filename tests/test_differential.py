"""Differential tests against python-chess.

python-chess is a mature, heavily fuzzed implementation, so it makes a far better
oracle than any expectation written by hand. These tests compare the *sets* of
legal moves and the SAN text of every one of them, which is what actually matters:
two implementations can disagree about how a position is scored and still both
play perfectly legal chess.

python-chess is a development dependency only. If it is not installed the module
is skipped rather than failing, so `pip install foxchess` users are unaffected.
"""

from __future__ import annotations

import random

import pytest

from conftest import EN_PASSANT, KIWIPETE, PROMOTION, STARTPOS
from foxchess import move as mv
from foxchess import notation
from foxchess.board import Board

chess = pytest.importorskip("chess", reason="python-chess is a dev dependency")

# Positions chosen for the rules they stress rather than for being famous.
POSITIONS: dict[str, str] = {
    "startpos": STARTPOS,
    "kiwipete": KIWIPETE,
    "endgame": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "position3": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
    "position4": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "position5": "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "position6": "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "en_passant": EN_PASSANT,
    "promotion": PROMOTION,
    "promotion_capture": "1n6/P6k/8/8/8/8/6K1/8 w - - 0 1",
    "rook_take_own_pawn": "4k3/8/8/R7/8/8/8/R3K2r w Q - 0 1",
    "stalemate": "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
    "checkmate": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "insufficient_material": "8/8/4k3/8/8/4K3/8/8 w - - 0 1",
    "king_and_rook": "8/8/8/8/8/4k3/4r3/4K3 w - - 0 1",
    "double_check": "4k3/8/8/8/8/8/3q4/2b1K3 w - - 0 1",
    "pinned_piece": "4k3/8/8/8/8/8/4R3/4K2r w - - 0 1",
    "many_queens": "qqqqkqqq/qqqqqqqq/8/8/8/8/8/K6k w - - 0 1",
}

#: The minor-piece combinations, paired against each other, that decide whether a
#: position is drawn. Generating them beats hand-picking: the subtle ones are
#: exactly the cases where a side cannot mate *unaided*, yet the opponent's pieces
#: can still help deliver it. Counting minors gets K+N+N against a lone king
#: wrong, because the bare king can be driven into a corner where both knights
#: mate. Only an oracle agrees on all of these, which is why the sweep lives here
#: rather than in a unit test.
_MATERIAL_SQUARES = (
    "a1", "c1", "e1", "g1", "a3", "c3", "e3", "g3",
    "a5", "c5", "e5", "g5", "a7", "c7", "e7", "g7",
)  # fmt: skip
_MATERIAL_INDEX = {name: i for i, name in enumerate(_MATERIAL_SQUARES)}

#: a1/a3 are dark and c1/c3 are light, so these pairs cover both colour complexes.
_MATERIAL_COMBOS: tuple[tuple[tuple[str, int], ...], ...] = (
    (),
    (("a1", chess.KNIGHT),),
    (("a1", chess.BISHOP),),
    (("a1", chess.KNIGHT), ("c1", chess.KNIGHT)),
    (("a1", chess.BISHOP), ("e1", chess.BISHOP)),
    (("a1", chess.BISHOP), ("c1", chess.BISHOP)),
    (("a1", chess.BISHOP), ("e1", chess.BISHOP), ("a5", chess.KNIGHT)),
    (("a1", chess.KNIGHT), ("c1", chess.KNIGHT), ("e1", chess.KNIGHT)),
    (("a1", chess.BISHOP), ("c1", chess.BISHOP), ("a5", chess.KNIGHT)),
    (("a1", chess.BISHOP), ("e1", chess.BISHOP), ("g1", chess.BISHOP)),
    (("a1", chess.KNIGHT), ("c1", chess.BISHOP)),
)


def _material_fen(white: tuple, black: tuple) -> str:
    board = chess.Board.empty()
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    for square, kind in white:
        board.set_piece_at(_MATERIAL_INDEX[square], chess.Piece(kind, chess.WHITE))
    for square, kind in black:
        board.set_piece_at(_MATERIAL_INDEX[square], chess.Piece(kind, chess.BLACK))
    return board.fen()


def our_moves(board: Board) -> dict[str, int]:
    return {mv.to_uci(move): move for move in board.legal_moves()}


def their_moves(fen: str) -> dict[str, chess.Move]:
    theirs = chess.Board(fen)
    return {move.uci(): move for move in theirs.legal_moves}


@pytest.mark.parametrize("label", sorted(POSITIONS))
def test_legal_move_sets_match_python_chess(label: str) -> None:
    fen = POSITIONS[label]
    mine = our_moves(Board(fen))
    theirs = their_moves(fen)
    assert sorted(mine) == sorted(theirs), "move sets differ"


@pytest.mark.parametrize("label", sorted(POSITIONS))
def test_san_matches_python_chess_for_every_legal_move(label: str) -> None:
    """Not just that the move sets agree, but that each move is *named* the same.

    A shared move set with divergent SAN is still a bug: the SAN is what gets
    written to a PGN and read by other programs.
    """
    fen = POSITIONS[label]
    board = Board(fen)
    theirs = {move.uci(): move for move in chess.Board(fen).legal_moves}
    for uci, move in our_moves(board).items():
        assert notation.san(board, move) == chess.Board(fen).san(theirs[uci]), uci


@pytest.mark.parametrize("label", sorted(POSITIONS))
def test_san_round_trips_through_our_parser(label: str) -> None:
    """Everything we print, we must also be able to read back."""
    fen = POSITIONS[label]
    board = Board(fen)
    for uci, move in our_moves(board).items():
        text = notation.san(board, move)
        assert mv.to_uci(notation.parse_san(Board(fen), text)) == uci, text


@pytest.mark.parametrize("label", sorted(POSITIONS))
def test_fen_matches_python_chess_both_ways(label: str) -> None:
    """Our FEN must be the FEN python-chess writes, and must parse back to it."""
    fen = POSITIONS[label]
    assert Board(fen).fen() == chess.Board(fen).fen()
    assert Board(Board(fen).fen()).fen() == fen


@pytest.mark.parametrize("label", sorted(POSITIONS))
def test_terminal_states_match_python_chess(label: str) -> None:
    fen = POSITIONS[label]
    mine = Board(fen)
    theirs = chess.Board(fen)
    assert mine.in_check() == theirs.is_check(), label
    assert mine.is_checkmate() == theirs.is_checkmate(), label
    assert mine.is_stalemate() == theirs.is_stalemate(), label
    assert mine.is_insufficient_material() == theirs.is_insufficient_material(), label
    # A fresh position has no repetition history on either side, so "over" should
    # mean the same thing: mated, stalemated, out of material, or a 50-move claim.
    their_over = (
        theirs.is_checkmate()
        or theirs.is_stalemate()
        or theirs.is_insufficient_material()
        or theirs.is_fifty_moves()
        or theirs.is_seventyfive_moves()
    )
    assert mine.is_game_over() == their_over, label
    assert mine.is_draw() == theirs.is_stalemate() or theirs.is_insufficient_material() or (
        theirs.is_checkmate()
    ), label


@pytest.mark.parametrize("label", sorted(POSITIONS))
def test_uci_text_agrees_with_python_chess(label: str) -> None:
    """A move we emit as text must name the same move in python-chess."""
    fen = POSITIONS[label]
    board = Board(fen)
    for uci in our_moves(board):
        parsed = notation.parse_long_algebraic(Board(fen), uci)
        assert mv.to_uci(parsed) == uci
        assert chess.Board(fen).parse_uci(uci) is not None


@pytest.mark.parametrize(
    ("white", "black"),
    [
        (white, black)
        for white in _MATERIAL_COMBOS
        for black in _MATERIAL_COMBOS
        if white != black or not white
    ],
)
def test_insufficient_material_matches_python_chess(white: tuple, black: tuple) -> None:
    """Every minor-only pairing, judged the same way by both implementations."""
    fen = _material_fen(white, black)
    assert Board(fen).is_insufficient_material() == chess.Board(fen).is_insufficient_material(), fen


def test_random_games_never_diverge_from_python_chess() -> None:
    """Play random games side by side and compare after every single ply.

    This is the test that catches state bugs a fixed position cannot: a wrong
    unmake, a stale Zobrist key, a castling right that should have been cleared.
    The seed is fixed so a failure is reproducible.
    """
    random.seed(20260927)
    starts = [STARTPOS, KIWIPETE, POSITIONS["position4"], POSITIONS["pinned_piece"]]
    for start in starts:
        mine = Board(start)
        theirs = chess.Board(start)
        for _ply in range(120):
            mine_uci = sorted(our_moves(mine))
            their_uci = sorted(m.uci() for m in theirs.legal_moves)
            assert mine_uci == their_uci, f"diverged in {start} after {_ply} plies"
            if not their_uci:
                break
            choice = random.choice(their_uci)
            mine.push_uci(choice)
            theirs.push_uci(choice)
            assert mine.fen() == theirs.fen(), f"FEN diverged in {start} at ply {_ply}"
            assert mine.verify_key(), f"incremental key drifted in {start}"
            assert mine.in_check() == theirs.is_check()
            assert mine.is_checkmate() == theirs.is_checkmate()


def _python_chess_perft(fen: str, depth: int) -> int:
    if depth == 0:
        return 1
    board = chess.Board(fen)
    if depth == 1:
        return board.legal_moves.count()
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += _python_chess_perft(board.fen(), depth - 1)
        board.pop()
    return total


def test_perft_counts_agree_at_depth_three() -> None:
    """Perft is move generation plus make/unmake, so this covers both at once."""
    from foxchess.perft import perft

    suite = {
        STARTPOS: (20, 400, 8902),
        KIWIPETE: (48, 2039, 97862),
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1": (14, 191, 2812),
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1": (6, 264, 9467),
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8": (44, 1486, 62379),
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10": (
            46, 2079, 89890,
        ),
    }
    for fen, counts in suite.items():
        mine = Board(fen)
        for depth, expected in enumerate(counts, start=1):
            assert perft(mine, depth) == expected, f"foxchess {fen} at depth {depth}"
        assert _python_chess_perft(fen, 3) == counts[2], f"python-chess disagrees on {fen}"

"""Search: mate detection, terminal positions, PV validity, and board hygiene.

Most of these assertions are about *correctness under search*, which is a
different class of bug from wrong move generation. Move generation is checked
against python-chess elsewhere; here the question is whether the search reports
what it found, whether its principal variation is a real game, and whether it
leaves the board exactly as it found it.

On cost: this engine searches at roughly 1000 nodes per second under CPython, and
a dense middlegame such as Kiwipete needs about 150k nodes at depth 4 -- several
minutes. Anything that searches a dense position deeply is marked ``slow`` so the
default run stays fast; the deep checks still run under ``pytest -m slow``. Sparse
positions are used wherever the property under test does not depend on the position
being realistic.
"""

from __future__ import annotations

import pytest

from conftest import KIWIPETE, STARTPOS
from foxchess import move as mv
from foxchess.board import Board
from foxchess.evaluate import INFINITY, MATE_SCORE, MATE_THRESHOLD
from foxchess.search import (
    MAX_PLY,
    MAX_QUIESCENCE_PLY,
    NO_MOVE,
    Searcher,
    SearchLimits,
    SearchResult,
    _from_tt_score,
    _to_tt_score,
)
from foxchess.tt import TranspositionTable

#: White mates in one with Ra8#.
MATE_IN_ONE = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
#: The same idea from the other end: White is already mated and it is White's
#: turn, so there is nothing to play at all. Fool's mate.
ALREADY_MATED = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
#: White is in check from Qb2 and Kxb2 is the only legal answer.
ONE_MOVE_ONLY = "7k/8/8/8/8/8/1q6/K7 w - - 0 1"
#: K+B against a lone king: no sequence of moves ends in mate.
DEAD_DRAW = "4k3/8/8/8/8/8/8/4KB2 w - - 0 1"
#: K+N+N against a lone king: unwinnable in practice, but *not* a dead position,
#: because a mate is constructible even though it cannot be forced.
UNWINNABLE = "4k3/8/8/8/8/8/8/NN2K3 w - - 0 1"
#: Kings and rooks only, so any repetition is a genuine draw claim.
REPETITION = "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1"
#: K+R against a lone king: mate is impossible, so no mate score may ever appear.
KR_VS_K = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
STALEMATE = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
#: A pawn about to promote with a rook behind it: a cheap, dense enough middlegame.
SIMPLE_MIDGAME = "4k3/1p6/8/8/8/8/6PP/R3K3 w Q - 0 1"


def search(fen: str, depth: int = 4, *, tt: TranspositionTable | None = None) -> SearchResult:
    return Searcher(Board(fen), tt=tt).search(SearchLimits(depth=depth))


# ---------------------------------------------------------------------------
# Mate
# ---------------------------------------------------------------------------


def test_it_finds_a_mate_in_one() -> None:
    result = search(MATE_IN_ONE, depth=3)
    assert mv.to_uci(result.best_move) == "a1a8"
    assert result.is_mate
    assert result.mate_in == 1
    assert result.score > 0, "the mating side is White here"


def test_a_mated_side_reports_no_move() -> None:
    """Nothing to play, so no move may be invented."""
    result = search(ALREADY_MATED, depth=3)
    assert result.best_move == NO_MOVE
    assert result.terminal == "checkmate"
    assert result.score == -MATE_SCORE
    assert result.pv_uci() == []


def test_mate_distance_is_graded_in_plies() -> None:
    """A mate already delivered is zero plies away; one that still has to be
    played is one. Each side's score follows as MATE_SCORE minus that distance."""
    delivered = search(ALREADY_MATED, depth=2)
    played = search(MATE_IN_ONE, depth=3)
    assert delivered.mate_in == 0, "the game is over: no plies remain"
    assert played.mate_in == 1
    assert abs(delivered.score) == MATE_SCORE - delivered.mate_in
    assert abs(played.score) == MATE_SCORE - played.mate_in
    # Closer and winning beats further and winning.
    assert played.score > delivered.score


def test_the_mate_score_is_signed_by_the_side_that_gets_mated() -> None:
    """MATE_IN_ONE and ALREADY_MATED are the two signs of the same scale: a
    position where the side to move wins by mate, and one where the side to move
    is already the one that was mated."""
    winning = search(MATE_IN_ONE, depth=3)
    losing = search(ALREADY_MATED, depth=2)
    assert winning.score > MATE_THRESHOLD
    assert losing.score == -MATE_SCORE
    assert winning.mate_in == 1 and losing.mate_in == 0


def test_mate_scores_survive_a_transposition_table() -> None:
    """Mate scores are stored ply-relative, so a TT hit must be re-based."""
    board = Board(MATE_IN_ONE)
    warm = TranspositionTable(1)
    first = Searcher(board, tt=warm).search(SearchLimits(depth=4))
    second = Searcher(board, tt=warm).search(SearchLimits(depth=4))
    assert mv.to_uci(first.best_move) == mv.to_uci(second.best_move)
    assert first.score == second.score
    assert first.mate_in == second.mate_in


def test_the_search_stops_at_a_mate_rather_than_searching_on() -> None:
    """Once a mate is proven, deeper search cannot improve on it."""
    result = search(MATE_IN_ONE, depth=8)
    assert result.is_mate
    assert result.depth <= 3, "a mate in one is known long before depth 8"


# ---------------------------------------------------------------------------
# Terminal positions
# ---------------------------------------------------------------------------


def test_a_stalemate_is_reported_as_a_draw() -> None:
    result = search(STALEMATE, depth=3)
    assert result.best_move == NO_MOVE
    assert result.terminal == "stalemate"
    assert result.score == 0


def test_a_dead_position_scores_zero_not_a_bishop_up() -> None:
    """K+B against a lone king cannot be mated by any line, so the score must be
    a draw. Leaving the raw material in the evaluation made the engine treat a dead
    draw as a won one and decline the draw it was entitled to."""
    result = search(DEAD_DRAW, depth=4)
    assert result.score == 0
    assert not result.is_mate


def test_two_knights_against_a_lone_king_are_not_called_dead() -> None:
    """Unwinnable in practice, yet mate is constructible, so the search must keep
    looking rather than score it 0 by rule."""
    result = search(UNWINNABLE, depth=4)
    assert result.score != 0, "a dead-position shortcut would have zeroed this"


def test_a_forced_move_is_still_reported() -> None:
    """One legal move needs no search, but it must not be reported as no move."""
    board = Board(ONE_MOVE_ONLY)
    legal = board.legal_moves()
    assert len(legal) == 1
    result = search(ONE_MOVE_ONLY, depth=4)
    assert result.best_move == legal[0]
    assert result.terminal is None
    assert result.depth == 1


# ---------------------------------------------------------------------------
# Principal variation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fen", [STARTPOS, MATE_IN_ONE, SIMPLE_MIDGAME, ALREADY_MATED, "4k3/8/8/8/8/8/8/4K3 w - - 0 1"]
)
def test_the_principal_variation_is_a_legal_game(fen: str) -> None:
    """The PV is replayed move by move: a PV that cannot be played is a lie."""
    board = Board(fen)
    result = Searcher(board).search(SearchLimits(depth=4))
    pv = result.principal_variation
    if not pv:
        assert result.terminal is not None, "a terminal position has no PV"
        return
    for move in pv:
        assert board.is_legal(move), mv.to_uci(move)
        board.make_move(move)
    # The first PV move is the move the engine is actually going to play.
    assert pv[0] == result.best_move


@pytest.mark.slow
def test_the_principal_variation_is_a_legal_game_in_a_real_middlegame() -> None:
    board = Board(KIWIPETE)
    result = Searcher(board).search(SearchLimits(depth=4))
    for move in result.principal_variation:
        assert board.is_legal(move), mv.to_uci(move)
        board.make_move(move)


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_the_principal_variation_stays_a_legal_line_at_every_depth(depth: int) -> None:
    """The PV may be longer than the nominal depth, because check extensions add
    plies the search genuinely proved. What must always hold is that it is a line
    that can be played, and that it starts with the move being returned."""
    board = Board(SIMPLE_MIDGAME)
    result = Searcher(board).search(SearchLimits(depth=depth))
    pv = result.principal_variation
    assert pv, "a non-terminal position must produce a principal variation"
    assert pv[0] == result.best_move
    assert len(pv) <= MAX_PLY
    for move in pv:
        assert board.is_legal(move), mv.to_uci(move)
        board.make_move(move)


def test_pv_uci_round_trips_through_the_board() -> None:
    board = Board(MATE_IN_ONE)
    result = Searcher(board).search(SearchLimits(depth=3))
    for text in result.pv_uci():
        assert mv.from_uci(text, board.legal_moves()) is not None
        board.push_uci(text)


def test_the_best_move_is_always_playable() -> None:
    for fen in (STARTPOS, SIMPLE_MIDGAME, MATE_IN_ONE, ONE_MOVE_ONLY, DEAD_DRAW):
        board = Board(fen)
        result = Searcher(board).search(SearchLimits(depth=3))
        if result.best_move == NO_MOVE:
            assert not board.legal_moves()
        else:
            assert board.is_legal(result.best_move), (fen, mv.to_uci(result.best_move))


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


def test_a_node_limit_stops_the_search_early() -> None:
    result = Searcher(Board(KIWIPETE)).search(SearchLimits(depth=6, nodes=500))
    assert result.nodes <= 500 * 4, "overshoot should be a handful of nodes"
    assert result.depth < 6, "the search must report the depth it completed"


def test_a_time_limit_stops_the_search_early() -> None:
    result = Searcher(Board(KIWIPETE)).search(SearchLimits(depth=20, movetime_ms=250))
    assert result.time_ms < 5000, result.time_ms
    assert result.depth < 20


def test_a_search_always_completes_at_least_one_depth() -> None:
    """An impossibly small limit still has to return a usable move."""
    result = Searcher(Board(STARTPOS)).search(SearchLimits(depth=4, nodes=1))
    assert result.best_move != NO_MOVE
    assert Board(STARTPOS).is_legal(result.best_move)


def test_depth_one_returns_a_legal_move() -> None:
    result = search(SIMPLE_MIDGAME, depth=1)
    assert Board(SIMPLE_MIDGAME).is_legal(result.best_move)
    assert result.depth == 1


def test_reported_counters_are_self_consistent() -> None:
    result = search(SIMPLE_MIDGAME, depth=4)
    assert result.nodes > 0
    assert result.nps > 0
    assert result.time_ms > 0
    # nps is nodes per second derived from the elapsed time, so it cannot run away
    # from the node count unless the clock is being read in the wrong unit.
    assert result.nps <= result.nodes * 1000


def test_as_dict_is_serialisable_and_complete() -> None:
    payload = search(MATE_IN_ONE, depth=3).as_dict()
    assert payload["bestmove"] == "a1a8"
    assert payload["mate"] == 1
    assert payload["pv"][0] == "a1a8"
    assert set(payload) == {
        "bestmove", "score", "mate", "depth", "nodes", "nps", "time_ms", "pv", "terminal"
    }


def test_a_terminal_result_serialises_with_no_move() -> None:
    payload = search(ALREADY_MATED, depth=2).as_dict()
    assert payload["bestmove"] is None
    assert payload["pv"] == []
    assert payload["terminal"] == "checkmate"


# ---------------------------------------------------------------------------
# Repetition and draws inside the tree
# ---------------------------------------------------------------------------


def test_a_king_and_rook_against_a_lone_king_is_never_a_mate() -> None:
    """K+R cannot mate, so the search must never report a mate score here --
    whatever it shuffles the rook around doing."""
    result = Searcher(Board(KR_VS_K)).search(SearchLimits(depth=4))
    assert not result.is_mate
    assert abs(result.score) < MATE_THRESHOLD, result.score


def test_searching_twice_gives_the_same_answer() -> None:
    """The search must be deterministic: no hidden state between runs."""
    first = search(SIMPLE_MIDGAME, depth=4)
    second = search(SIMPLE_MIDGAME, depth=4)
    assert mv.to_uci(first.best_move) == mv.to_uci(second.best_move)
    assert first.score == second.score
    assert first.pv_uci() == second.pv_uci()


# ---------------------------------------------------------------------------
# Board hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fen", [STARTPOS, SIMPLE_MIDGAME, MATE_IN_ONE, STALEMATE, REPETITION, ALREADY_MATED]
)
@pytest.mark.parametrize("depth", [1, 3])
def test_search_leaves_the_board_untouched(fen: str, depth: int) -> None:
    """The single most important invariant: a search must not move the position."""
    board = Board(fen)
    before_fen = board.fen()
    before_key = board.key
    before_halfmove = board.halfmove_clock
    before_fullmove = board.fullmove_number
    Searcher(board).search(SearchLimits(depth=depth))
    assert board.fen() == before_fen
    assert board.key == before_key
    assert board.halfmove_clock == before_halfmove
    assert board.fullmove_number == before_fullmove
    assert board.verify_key(), "an unmake went wrong somewhere in the tree"
    assert board.ply == 0, "a move was left on the undo stack"


def test_the_transposition_table_does_not_corrupt_the_board() -> None:
    board = Board(SIMPLE_MIDGAME)
    before = board.key
    Searcher(board, tt=TranspositionTable(1)).search(SearchLimits(depth=4))
    assert board.key == before
    assert board.verify_key()


# ---------------------------------------------------------------------------
# Quiescence bounds
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_quiescence_stops_at_its_own_limit() -> None:
    """The cap exists because a dense position can otherwise chain check evasions
    for tens of plies. ``Searcher`` uses ``__slots__``, so this subclasses rather
    than patching the bound method."""
    assert 0 < MAX_QUIESCENCE_PLY < MAX_PLY

    class Recorded(Searcher):
        reached: list[int]

        def _quiescence(self, alpha: int, beta: int, ply: int) -> int:
            self.reached.append(ply)
            return super()._quiescence(alpha, beta, ply)

    searcher = Recorded(Board(KIWIPETE))
    searcher.reached = []
    searcher.search(SearchLimits(depth=1))
    assert searcher.reached, "quiescence never ran"
    assert max(searcher.reached) <= MAX_QUIESCENCE_PLY, max(searcher.reached)


@pytest.mark.slow
def test_the_quiescence_cap_keeps_a_middlegame_search_small() -> None:
    """The observable consequence: one ply of a dense position stays in the
    thousands of nodes rather than the tens of thousands."""
    result = Searcher(Board(KIWIPETE)).search(SearchLimits(depth=1))
    assert 0 < result.nodes < 20_000, result.nodes


@pytest.mark.slow
def test_quiescence_keeps_the_board_consistent() -> None:
    """The capped branch in-check still makes and unmakes its evasions."""
    board = Board(KIWIPETE)
    before = board.fen()
    Searcher(board).search(SearchLimits(depth=2))
    assert board.fen() == before
    assert board.verify_key()
    assert board.ply == 0


# ---------------------------------------------------------------------------
# Mate score re-basing
# ---------------------------------------------------------------------------


def test_to_tt_and_from_tt_scores_are_inverse() -> None:
    for score in (0, 42, -42, MATE_THRESHOLD + 1, MATE_SCORE, -MATE_THRESHOLD - 1, -MATE_SCORE):
        for ply in (0, 1, 7, 30):
            assert _from_tt_score(_to_tt_score(score, ply), ply) == score, (score, ply)


def test_a_stored_mate_score_survives_a_deeper_node() -> None:
    """A mate found at ply 1 must not read back as a mate one ply further on."""
    for ply in (1, 5, 20, 90):
        stored = _to_tt_score(MATE_SCORE - 3, ply)
        assert _from_tt_score(stored, ply) == MATE_SCORE - 3
        assert _from_tt_score(stored, ply + 1) == MATE_SCORE - 4


def test_non_mate_scores_are_left_alone() -> None:
    for score in (0, 100, -100, MATE_THRESHOLD - 1, -(MATE_THRESHOLD - 1)):
        assert _to_tt_score(score, 7) == score
        assert _from_tt_score(score, 7) == score


# ---------------------------------------------------------------------------
# Helpers and constants
# ---------------------------------------------------------------------------


def test_no_move_is_distinct_from_a_valid_move() -> None:
    """A packed move of 0 is ``a1a1``, so -1 is the only safe 'no move'."""
    assert NO_MOVE == -1
    assert mv.to_uci(NO_MOVE) == "0000", "UCI reserves 0000 for the null move"
    assert mv.to_uci(0) == "a1a1", "0 really is the a1a1 encoding"
    assert mv.encode_int(0, 0) == 0


def test_a_pv_never_exceeds_the_ply_limit() -> None:
    assert len(search(SIMPLE_MIDGAME, depth=4).principal_variation) <= MAX_PLY


def test_search_limits_default_to_a_sane_depth() -> None:
    assert SearchLimits().depth >= 8
    assert SearchLimits().budget_ms() is None, "no clock means no time limit"


def test_movetime_takes_priority_over_the_clock() -> None:
    limits = SearchLimits(movetime_ms=123, wtime_ms=60_000, btime_ms=60_000)
    assert limits.budget_ms() == 123


def test_clock_budget_scales_with_the_increment() -> None:
    without = SearchLimits(wtime_ms=10_000, btime_ms=10_000).budget_ms()
    with_increment = SearchLimits(wtime_ms=10_000, btime_ms=10_000, winc_ms=500).budget_ms()
    assert with_increment is not None and without is not None
    assert with_increment > without


def test_the_side_to_move_selects_the_clock() -> None:
    limits = SearchLimits(wtime_ms=1000, btime_ms=30_000, movestogo=10)
    limits.white_to_move = True
    assert limits.budget_ms() == 100
    limits.white_to_move = False
    assert limits.budget_ms() == 3000


def test_a_zero_clock_still_yields_a_positive_budget() -> None:
    """Zero would mean 'never check the clock' and search forever."""
    assert SearchLimits(wtime_ms=0, btime_ms=0).budget_ms() == 1
    assert SearchLimits(movetime_ms=0).budget_ms() == 1


def test_a_quiet_score_stays_inside_the_evaluation_range() -> None:
    """A non-mate score must be small; a runaway score means an arithmetic slip."""
    result = search(SIMPLE_MIDGAME, depth=4)
    assert abs(result.score) < MATE_THRESHOLD
    assert abs(result.score) < INFINITY


def test_on_update_is_called_once_per_completed_depth() -> None:
    seen: list[tuple[int, int]] = []
    searcher = Searcher(Board(STARTPOS))
    searcher.on_update = lambda depth, score, _pv: seen.append((depth, score))
    result = searcher.search(SearchLimits(depth=3))
    assert [depth for depth, _ in seen] == list(range(1, result.depth + 1))

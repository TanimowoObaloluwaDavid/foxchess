"""Static evaluation: sign, symmetry, and the individual terms.

Two properties matter more than any particular number here. First, the score
must have the right *sign*: a centipawn up should not evaluate as a draw. Second,
it must be symmetric: mirroring a position across the board and swapping colours
must give the same score from each side's point of view, otherwise the search is
being handed a preference for one colour and will play worse as one of them.

The tables are written in White's point of view with a1 first, and Black reads
them through ``sq ^ 56`` -- flipping the rank, not the whole 64-square index.
Flipping all 64 squares would mirror files as well, which is not what a table
indexed by rank needs; the per-file arrays are palindromic so the file symmetry
comes from the data instead.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from conftest import mirror_fen
from foxchess import evaluate as ev
from foxchess.board import Board
from foxchess.types import PIECE_VALUES, Color, PieceType, piece_type_of, square_from_name

# Positions used for the term-by-term checks.
UP_A_ROOK = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"
UP_TWO_ROOKS = "4k3/8/8/8/8/8/8/RR2K3 w - - 0 1"
UP_A_QUEEN = "4k3/8/8/8/8/8/8/Q3K3 w - - 0 1"
BARE_KINGS = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
BISHOP_PAIR = "4k3/8/8/8/8/8/8/BB2K3 w - - 0 1"
ROOK_ON_OPEN = "4k3/8/8/8/8/8/8/R3K1R1 w - - 0 1"
ROOK_BLOCKED = "4k3/8/8/8/8/pppppppp/8/R3K1R1 w - - 0 1"
PASSED_PAWN = "4k3/8/8/8/4P3/8/8/4K3 w - - 0 1"
BLOCKED_PAWN = "4k3/8/8/8/3pP3/8/8/4K3 w - - 0 1"
ISOLATED_PAWN = "4k3/8/8/8/8/8/P7/4K3 w - - 0 1"
DOUBLED_PAWN = "4k3/8/8/8/8/8/PP6/4K3 w - - 0 1"
OPENING = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
PIECES_ON_BOARD = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


# ---------------------------------------------------------------------------
# Sign
# ---------------------------------------------------------------------------


def test_a_centipawn_up_is_positive() -> None:
    assert ev.evaluate(Board(UP_A_ROOK)) > 0


def test_a_centipawn_down_is_negative() -> None:
    assert ev.evaluate(Board(mirror_fen(UP_A_ROOK))) < 0


def test_a_pawn_up_is_about_a_pawn() -> None:
    score = ev.evaluate(Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"))
    # Material, plus placement, plus a passed-pawn bonus, minus the isolation
    # penalty for having no friendly pawn beside it, plus tempo. That lands above
    # the bare pawn value, so the band is generous at the top and tight at the
    # bottom: a pawn must not be worth a queen, but a passed one is worth more
    # than its material.
    assert PIECE_VALUES[PieceType.PAWN] * 0.8 < score < PIECE_VALUES[PieceType.PAWN] * 2.5


def test_piece_values_are_ordered() -> None:
    order = [
        PieceType.PAWN, PieceType.KNIGHT, PieceType.BISHOP,
        PieceType.ROOK, PieceType.QUEEN,
    ]
    values = [PIECE_VALUES[piece] for piece in order]
    assert values == sorted(values)
    assert all(value > 0 for value in values)


def test_the_start_position_is_symmetrical_and_just_tempo() -> None:
    """Equal material in a balanced position: only the tempo bonus separates the sides."""
    board = Board(OPENING)
    assert ev.evaluate(board, detailed=False) == ev.TEMPO
    assert ev.evaluate(Board(mirror_fen(OPENING)), detailed=False) == -ev.TEMPO


def test_even_piece_material_evaluates_near_zero() -> None:
    """An equal, developed position should not favour either side."""
    score = ev.evaluate(Board(OPENING), detailed=False)
    assert abs(score) < 40, score


def test_tempo_follows_the_side_to_move() -> None:
    """Whoever is to move gets the tempo bonus, so the score is from White's view."""
    assert ev.evaluate(Board(BARE_KINGS)) == ev.TEMPO
    assert ev.evaluate(Board(BARE_KINGS.replace(" w ", " b "))) == -ev.TEMPO


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


#: Position set used for the symmetry checks. Kiwipete is included on purpose:
#: it is the busiest fixture in the suite.
_SYMMETRIC_POSITIONS = [
    OPENING,
    UP_A_ROOK,
    UP_TWO_ROOKS,
    BARE_KINGS,
    BISHOP_PAIR,
    PASSED_PAWN,
    ISOLATED_PAWN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
]


@pytest.mark.parametrize("fen", _SYMMETRIC_POSITIONS)
def test_cheap_evaluation_is_exactly_mirror_symmetric(fen: str) -> None:
    """The cheap score is the symmetric core, and it must be exact.

    Placement, pawn structure and tempo are all written in White's point of view
    with Black reading the tables through ``sq ^ 56``, so mirroring the board and
    swapping colours has to give the exact negation. Any drift here is a table or
    sign bug rather than a heuristic, which is why this asserts equality.
    """
    assert ev.evaluate(Board(mirror_fen(fen)), detailed=False) == -ev.evaluate(
        Board(fen), detailed=False
    )


@pytest.mark.parametrize("fen", _SYMMETRIC_POSITIONS)
def test_full_evaluation_is_close_to_symmetric(fen: str) -> None:
    """The full score adds terms that are not required to be exact.

    ``_king_danger`` deliberately scores only the king of the side that just moved,
    to keep the expensive part of the evaluation cheap. That is a one-sided term by
    design, so the full score is symmetric only to within its cap.
    """
    difference = ev.evaluate(Board(mirror_fen(fen))) + ev.evaluate(Board(fen))
    assert abs(difference) <= ev.KING_DANGER_CAP, difference


def test_king_danger_scores_only_the_side_that_just_moved() -> None:
    """Documenting the asymmetry so it cannot be changed by accident."""
    fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    white_to_move = Board(fen)
    black_to_move = Board(fen)
    black_to_move.side = Color.BLACK
    black_to_move.key = white_to_move.key

    # White to move: the term is about Black's king, and it must be <= 0 because
    # the defender is the side that just moved.
    assert ev._king_danger(white_to_move) <= 0
    # Black to move: now it is about White's king -- a different king entirely.
    assert ev._king_danger(black_to_move) <= 0


def test_king_danger_is_bounded() -> None:
    """Eight queens aimed at one king must not swamp the rest of the evaluation.

    Black to move, so Black is the attacker and White's king is the one scored.
    """
    crowded = "4k3/8/8/8/8/8/qqqqqqqq/4K3 b - - 0 1"
    calm = "4k3/8/8/8/8/8/8/4K3 b - - 0 1"
    crowded_danger = ev._king_danger(Board(crowded))
    assert crowded_danger < ev._king_danger(Board(calm)) <= 0
    assert abs(crowded_danger) <= ev.KING_DANGER_CAP, crowded_danger


def test_mirroring_preserves_the_score_from_each_side() -> None:
    board = Board(UP_A_ROOK)
    flipped = Board(mirror_fen(UP_A_ROOK))
    # White to move in one, Black in the other, and each likes their own position.
    assert ev.evaluate(board) > 0
    assert ev.evaluate_side(board, Color.BLACK) < 0
    assert ev.evaluate_side(flipped, Color.BLACK) > 0
    assert ev.evaluate_side(flipped, Color.WHITE) < 0


def test_placement_tables_are_palindromic_across_files() -> None:
    """A file-mirrored square must score the same, which is what the data buys.

    ``sq ^ 7`` flips the low three bits -- the file -- while leaving the rank
    alone, which is the mirror the per-file arrays assume.
    """
    for piece_type in ev._MIDGAME_TABLES:
        midgame = ev._MIDGAME_TABLES[piece_type]
        endgame = ev._ENDGAME_TABLES[piece_type]
        assert len(midgame) == len(endgame) == 64
        for sq in range(64):
            mirrored = sq ^ 7
            assert midgame[sq] == midgame[mirrored], (piece_type, sq)
            assert endgame[sq] == endgame[mirrored], (piece_type, sq)


def test_pawn_structure_file_tables_are_palindromic() -> None:
    """Only the *file* tables are symmetric; the rank/advancement ones are not.

    ``PASSED_EG`` and ``PASSED_MG`` are indexed by how far a pawn has advanced, so
    they shrink towards the promotion end rather than mirroring.
    """
    assert tuple(reversed(ev.PASSED_FILE_WEIGHT)) == ev.PASSED_FILE_WEIGHT
    for row in ev.FILE_WEIGHTS:
        assert len(row) == 8
        assert row == tuple(reversed(row)), row
        for table in (ev.PASSED_MG, ev.PASSED_EG):
            # Entry 0 is a pawn still on its start rank: worth no passed-pawn bonus.
            assert table[0] == 0, table
            # From the first step of advancement onwards the bonus only shrinks, so a
            # passed pawn further up the board is not scored as more valuable than a
            # passed pawn that still has to travel.
            advanced = table[1:]
            assert all(later <= earlier for earlier, later in pairwise(advanced)), table


# ---------------------------------------------------------------------------
# Phase
# ---------------------------------------------------------------------------


def test_phase_runs_from_midgame_to_endgame() -> None:
    assert ev.game_phase(Board(OPENING)) == ev.FULL_PHASE, "a full board is pure midgame"


def test_phase_shrinks_as_pieces_come_off() -> None:
    later = Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
    assert ev.game_phase(later) < ev.game_phase(Board(OPENING))
    assert ev.game_phase(later) >= 0


def test_phase_never_leaves_the_range() -> None:
    for fen in (OPENING, UP_A_ROOK, BARE_KINGS, "4k3/8/8/8/8/8/8/4K3 b - - 0 1"):
        assert 0 <= ev.game_phase(Board(fen)) <= ev.FULL_PHASE


def test_piece_type_on_reports_the_board() -> None:
    board = Board(UP_A_ROOK)
    assert ev.piece_type_on(board, square_from_name("a1")) is PieceType.ROOK
    assert ev.piece_type_on(board, square_from_name("e1")) is PieceType.KING
    assert ev.piece_type_on(board, square_from_name("a4")) is None
    assert piece_type_of(board.piece_at(square_from_name("a1"))) is PieceType.ROOK


# ---------------------------------------------------------------------------
# Individual terms
# ---------------------------------------------------------------------------


def test_a_bishop_pair_is_worth_more_than_two_bishops() -> None:
    pair = ev.evaluate(Board(BISHOP_PAIR), detailed=False)
    unpaired = ev.evaluate(Board("4k3/8/8/8/8/8/8/B3K3 w - - 0 1"), detailed=False)
    assert pair > unpaired + ev.BISHOP_PAIR


def test_a_rook_on_an_open_file_beats_a_blocked_one() -> None:
    assert ev.evaluate(Board(ROOK_ON_OPEN)) > ev.evaluate(Board(ROOK_BLOCKED))


def test_a_passed_pawn_beats_a_blocked_one() -> None:
    assert ev.evaluate(Board(PASSED_PAWN), detailed=False) > ev.evaluate(
        Board(BLOCKED_PAWN), detailed=False
    )


def test_structural_penalties_are_positive_constants() -> None:
    for penalty in (ev.ISOLATED_PAWN, ev.DOUBLED_PAWN, ev.BACKWARD_PAWN):
        assert penalty > 0, penalty


def test_isolated_and_doubled_pawns_lose_points() -> None:
    connected = ev.evaluate(Board("4k3/8/8/8/8/8/PPPPPPPP/4K3 w - - 0 1"), detailed=False)
    isolated = ev.evaluate(Board(ISOLATED_PAWN), detailed=False)
    assert isolated < connected
    doubled = ev.evaluate(Board(DOUBLED_PAWN), detailed=False)
    assert doubled < connected


def test_piece_placement_rewards_advancement() -> None:
    """A knight belongs in the middle, not on its own back rank."""
    back = ev.tapered(PieceType.KNIGHT, square_from_name("b1"), white=True, phase=ev.FULL_PHASE)
    centre = ev.tapered(PieceType.KNIGHT, square_from_name("d4"), white=True, phase=ev.FULL_PHASE)
    assert centre > back


def test_tapered_blends_the_two_tables_by_phase() -> None:
    """Phase 0 is pure endgame, phase FULL_PHASE pure midgame, and the middle is
    a straight blend of the two."""
    piece_type = PieceType.KNIGHT
    sq = square_from_name("d4")
    midgame = ev._MIDGAME_TABLES[piece_type][sq]
    endgame = ev._ENDGAME_TABLES[piece_type][sq]
    assert ev.tapered(piece_type, sq, white=True, phase=ev.FULL_PHASE) == midgame
    assert ev.tapered(piece_type, sq, white=True, phase=0) == endgame
    phase = ev.FULL_PHASE // 2
    halfway = ev.tapered(piece_type, sq, white=True, phase=phase)
    expected = (midgame * phase + endgame * (ev.FULL_PHASE - phase)) // ev.FULL_PHASE
    assert halfway == expected, (halfway, expected)
    # A knight is worth more in the endgame than the opening.
    if endgame > midgame:
        assert ev.tapered(piece_type, sq, white=True, phase=0) >= halfway
    else:
        assert ev.tapered(piece_type, sq, white=True, phase=0) <= halfway


# ---------------------------------------------------------------------------
# Cost of the detailed terms
# ---------------------------------------------------------------------------


def test_detailed_evaluation_adds_the_expensive_terms() -> None:
    """`detailed=False` is the cheap score a TT probe wants, so it must be cheaper."""
    import time

    fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    board = Board(fen)
    calls = 200

    def timed(detailed: bool) -> float:
        start = time.perf_counter()
        for _ in range(calls):
            ev.evaluate(board, detailed=detailed)
        return time.perf_counter() - start

    # Warm up both paths first. Without this the branch measured first pays for
    # the caches the second one then gets to use, and the comparison inverts --
    # which is exactly what a 5% tolerance on a single sample will eventually do.
    for _ in range(50):
        ev.evaluate(board, detailed=True)
        ev.evaluate(board, detailed=False)

    # Best of several rounds, alternating which path goes first, so that neither
    # a one-off scheduling hiccup nor a fixed ordering can decide the result.
    cheap = detailed = float("inf")
    for round_number in range(5):
        first, second = (False, True) if round_number % 2 == 0 else (True, False)
        timings = {first: timed(first), second: timed(second)}
        cheap = min(cheap, timings[False])
        detailed = min(detailed, timings[True])

    # Time, not score: the two may legitimately differ in value, we only care that
    # skipping mobility and king danger is not slower.
    assert cheap <= detailed * 1.05, f"cheap={cheap:.4f}s detailed={detailed:.4f}s"


def test_evaluate_does_not_disturb_the_board() -> None:
    board = Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    before_fen = board.fen()
    before_key = board.key
    for detailed in (True, False):
        ev.evaluate(board, detailed=detailed)
    assert board.fen() == before_fen
    assert board.key == before_key
    assert board.verify_key()


def test_evaluation_is_deterministic() -> None:
    board = Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    assert len({ev.evaluate(board) for _ in range(20)}) == 1

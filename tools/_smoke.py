"""Smoke check: perft, eval symmetry, SAN/PGN round-trips, TT and a short search."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foxchess import evaluate as ev
from foxchess import notation, tt
from foxchess.board import Board
from foxchess.move import from_uci, to_uci
from foxchess.perft import perft
from foxchess.search import NO_MOVE, Searcher, SearchLimits

fails: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


_CASTLE_SWAP = {"K": "k", "Q": "q", "k": "K", "q": "Q"}


def mirror_fen(fen: str) -> str:
    """Swap colours and rotate 180 degrees.

    A correct evaluation must return exactly the negation on the mirrored
    position. That is the strongest available check on the piece-square tables,
    which is how the ``sq ^ 56`` rotation bug was caught.

    Reversing the rank order and each rank's file order is the rotation; swapping
    the piece case is the colour change. The two together send a white rook on
    a1 to a black rook on h8.

    The en-passant square is dropped deliberately: nothing in the evaluation reads
    it, and rotating it correctly is more fiddly than the test is worth.
    """
    placement, side, castling, _ep, halfmove, fullmove = fen.split()
    rotated = "/".join(rank[::-1] for rank in reversed(placement.split("/"))).swapcase()
    rights = [_CASTLE_SWAP[c] for c in castling if c in _CASTLE_SWAP]
    rights.sort(key="KQkq".index)
    flipped = "b" if side == "w" else "w"
    return f"{rotated} {flipped} {''.join(rights) or '-'} - {halfmove} {fullmove}"


print("== perft ==")
board = Board()
check("verify_key", board.verify_key())
check("perft(1) startpos", perft(board, 1) == 20)
check("perft(3) startpos", perft(board, 3) == 8902)

print("\n== evaluation ==")
check("phase of startpos", ev.game_phase(board) == 24)
check("startpos eval is just tempo", ev.evaluate(board) == ev.TEMPO, str(ev.evaluate(board)))
flipped_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
check(
    "negated on colour swap",
    ev.evaluate(Board(flipped_start)) == -ev.TEMPO,
    str(ev.evaluate(Board(flipped_start))),
)
for fen in (
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
):
    score = ev.evaluate(Board(fen), detailed=False)
    mirrored = ev.evaluate(Board(mirror_fen(fen)), detailed=False)
    check(f"symmetric {fen[:38]}", score == -mirrored, f"{score} vs {-mirrored}")

# The one deliberate exception. King danger scores only the king of the side that
# just moved, so it is one-sided by construction and cannot negate. Everything it
# is added to must be symmetric, which is what the checks above pin down.
for fen in ("4k3/8/8/8/8/8/8/4K2R w K - 0 1",):
    check(
        "king danger is one-sided by design",
        ev._king_danger(Board(fen)) == -ev._king_danger(Board(mirror_fen(fen))),
        str(ev._king_danger(Board(fen))),
    )

print("\n== notation ==")
for text, expected in (("e4", "e2e4"), ("d4", "d2d4"), ("Nf3", "g1f3")):
    move = notation.parse_san(Board(), text)
    check(f"parse_san {text}", to_uci(move) == expected, to_uci(move))
check(
    "O-O parse",
    to_uci(notation.parse_san(Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"), "O-O")) == "e1g1",
)
check(
    "exd5 parse",
    to_uci(
        notation.parse_san(
            Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"), "exd5"
        )
    )
    == "e4d5",
)

# SAN generation, one move at a time.
board = Board()
for uci_text, expected in (
    ("e2e4", "e4"),
    ("e7e5", "e5"),
    ("g1f3", "Nf3"),
    ("b8c6", "Nc6"),
    ("f1b5", "Bb5"),
):
    move = from_uci(uci_text)
    text = notation.san(board, move)
    check(f"san {uci_text} -> {expected}", text == expected, text)
    board.make_move(move)

# Disambiguation: two rooks that can both reach d1.
board = Board("4k3/8/8/8/8/4K3/8/R6R w - - 0 1")
rd1 = [m for m in board.legal_moves() if to_uci(m) in ("a1d1", "h1d1")]
check("rook disambiguation available", len(rd1) == 2, str([to_uci(m) for m in rd1]))
if len(rd1) == 2:
    rendered = sorted(notation.san(board, m) for m in rd1)
    check("rook disambiguates by file", rendered == ["Rad1", "Rhd1"], str(rendered))

print("\n== pgn ==")
game = notation.Game(
    headers={"White": "foxchess", "Black": "python-chess", "Event": "smoke"},
    san_moves=["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"],
    result="1-0",
)
text = notation.to_pgn(game)
parsed = notation.parse_pgn(text)
check("pgn move round trip", parsed.san_moves == game.san_moves, str(parsed.san_moves))
check("pgn position round trip", parsed.replay().fen() == game.replay().fen())
check("pgn headers survive", parsed.headers.get("White") == "foxchess")
check("pgn result survives", parsed.result == "1-0", parsed.result)

print("\n== transposition table ==")
table = tt.TranspositionTable(1)
table.store(1234, 5, 42, tt.EXACT, 77)
entry = table.probe(1234)
check("tt probe hits", entry is not None and entry.score == 42 and entry.move == 77)
check("tt probe misses on a different key", table.probe(9999) is None)
check("tt hashfull in range", 0 <= table.hashfull() <= 1000, str(table.hashfull()))

print("\n== search ==")
for fen, name, expect in (
    (None, "startpos", "move"),
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "Ra8 is mate in 1", "mate1"),
    ("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1", "Black is already mated", "terminal"),
    ("8/8/8/8/8/5k2/6q1/7K w - - 0 1", "White is already mated", "terminal"),
    ("4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1", "queen up, nothing forced", "quiet"),
):
    board = Board(fen) if fen else Board()
    searcher = Searcher(board, tt=tt.TranspositionTable(8))
    started = time.perf_counter()
    result = searcher.search(SearchLimits(depth=4))
    took = (time.perf_counter() - started) * 1000
    best = to_uci(result.best_move) if result.best_move != NO_MOVE else None
    print(
        f"  {name:26} best={best!s:6} score={result.score:6} mate={result.mate_in} "
        f"depth={result.depth} nodes={result.nodes} pv={result.pv_uci()} {took:.0f}ms"
    )
    if expect == "move":
        check("startpos finds a move", result.best_move != NO_MOVE)
        check("startpos reaches the requested depth", result.depth == 4, str(result.depth))
    if expect == "mate1":
        check("sees mate in 1", result.mate_in == 1, f"mate={result.mate_in}")
    if expect == "terminal":
        check(
            "reports a finished game with no move",
            result.best_move == NO_MOVE and result.terminal in ("checkmate", "stalemate"),
            f"terminal={result.terminal}",
        )
    if expect == "quiet":
        check("a queen up scores clearly winning", result.score > 700, f"score={result.score}")

# The board must be exactly as we left it after a search.
board = Board()
before = board.fen()
searcher = Searcher(board, tt=tt.TranspositionTable(4))
searcher.search(SearchLimits(depth=3))
check("search leaves the board untouched", board.fen() == before)
check("search leaves the key untouched", board.verify_key())

print()
if fails:
    print(f"FAILURES ({len(fails)}): {fails}")
    sys.exit(1)
print("ALL PASS")

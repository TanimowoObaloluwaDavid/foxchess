"""Perft cross-check against published node counts (CPW perftsuite)."""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")

from foxchess.board import Board
from foxchess.perft import perft

SUITE = {
    "startpos": (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        {1: 20, 2: 400, 3: 8902, 4: 197281, 5: 4865609},
    ),
    "kiwipete": (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        {1: 48, 2: 2039, 3: 97862, 4: 4085603},
    ),
    "pos3": (
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        {1: 14, 2: 191, 3: 2812, 4: 43238, 5: 674624, 6: 11030083},
    ),
    "pos4": (
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        {1: 6, 2: 264, 3: 9467, 4: 422333, 5: 15833292},
    ),
    "pos5": (
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        {1: 44, 2: 1486, 3: 62379, 4: 2103487, 5: 89941194},
    ),
    "pos6": (
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        {1: 46, 2: 2079, 3: 89890, 4: 3894594, 5: 164075551},
    ),
}

MAX_DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 4

failures = 0
total_nodes = 0
started = time.perf_counter()
for name, (fen, counts) in SUITE.items():
    board = Board(fen)
    for depth, expected in counts.items():
        if depth > MAX_DEPTH:
            continue
        start = time.perf_counter()
        got = perft(board, depth)
        elapsed = time.perf_counter() - start
        total_nodes += got
        ok = got == expected
        failures += not ok
        # flush, because a full run at depth 5 takes long enough that buffered
        # output is simply lost when the process is killed.
        print(
            f"{'OK  ' if ok else 'FAIL'} {name:9s} d{depth} "
            f"got={got:>10d} exp={expected:>10d} {elapsed:8.2f}s",
            flush=True,
        )

elapsed = time.perf_counter() - started
print(
    f"\n{total_nodes:,} nodes in {elapsed:.1f}s "
    f"({total_nodes / elapsed:,.0f} nps) at depth {MAX_DEPTH}",
    flush=True,
)
print("ALL PASS" if not failures else f"{failures} FAILURES", flush=True)
sys.exit(1 if failures else 0)

# foxchess

A chess engine in pure Python. Bitboard move generation, alpha-beta search with
quiescence, a UCI implementation for desktop chess GUIs, and a JSON HTTP
analysis API — with **no third-party runtime dependencies**.

```python
import foxchess

board = foxchess.Board()
result = foxchess.search(board, foxchess.SearchLimits(depth=4))
print(result.pv_uci())          # ['g1f3', 'b8c6', 'b1c3', 'g8f6']
print(result.score)              # 10
```

## Why it exists

Most chess libraries are either a binding to a fast engine written in C
(`python-chess` wrapping Stockfish) or a teaching implementation that skips
search. This is neither: the move generation, the search, the evaluation and the
network protocols are all in this repository, in readable Python, and the whole
thing installs from one wheel with nothing else to compile.

The obvious cost is speed. A Python engine is roughly two orders of magnitude
slower than a C one, so this is a library to learn from, to test against, or to
put behind a service — not a replacement for Stockfish in a tournament.

## Install

```console
$ pip install foxchess
```

Python 3.11 or newer. There is nothing else to install.

## The four ways to use it

### As a library

```python
import foxchess

board = foxchess.Board()
board.make_move(foxchess.from_uci("e2e4", board.legal_moves()))
board.fen()        # 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'

# SAN needs the position, because "Nbd2" and "Nfd2" are told apart by it.
board = foxchess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4")
for move in board.legal_moves():
    print(foxchess.to_uci(move), foxchess.san(board, move))
```

A four-character UCI string is ambiguous — `e2e4` is a double pawn push, and
`e5d6` may be an en-passant capture — and the flag that distinguishes them is
not in the string. `from_uci` therefore takes the legal moves to resolve it
against; without them it guesses, which is enough for display and not enough to
make a move.
`Board` keeps a Zobrist key that is updated incrementally on every move, so
`board.key` is safe to use as a transposition key. `board.verify_key()` checks
the incremental key against a fresh computation, which the test suite does at
every node of every perft.

Moves are plain `int`s, packed to keep move generation allocation-free. The
decoders are in the public API: `to_uci`, `from_uci`, `decode_from`, `decode_to`.

### As a UCI engine

```console
$ foxchess-uci
uci
position startpos moves e2e4 e7e5
go depth 8
```

Implements the parts of the protocol a GUI actually uses: `uci`, `isready`,
`ucinewgame`, `position`, `go` (including `infinite`/`ponder`/`movetime`/
`searchmoves`), `stop`, and `setoption` for the hash size. `go infinite` runs on
a background thread so `stop` is answered immediately; `bestmove` is always
emitted exactly once, including after `stop` and after the search is killed at
depth 0 in a finished position (as `0000`).

### As an HTTP service

```console
$ foxchess serve --port 8000
```

| Method | Path      | Purpose |
| ------ | --------- | ------- |
| `GET`  | `/health` | Version and a liveness check |
| `GET`  | `/fen`    | FEN, side to move, legal moves, terminal flags, and a hash self-check |
| `GET`  | `/perft`  | `?fen=…&depth=…`, optional `&divide=1` to split by root move |
| `POST` | `/analyze`| `{"fen": …, "depth": 4}` → score, best move, PV, node counts |

```console
$ curl -s localhost:8000/analyze -d '{"depth": 6}' | python -m json.tool
```

There is no authentication and no rate limit, so the default bind is loopback.
Exposing it on a network hands anyone a CPU burner. Depth is capped at 8 and
perft at 4 — see [Limits](#limits).

### As a CLI

```console
$ foxchess analyse --depth 5 e2e4 e7e5
$ foxchess perft --depth 4 "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
$ foxchess moves e2e4
$ foxchess fen e2e4
$ foxchess pgn --plies 12
$ foxchess serve --host 127.0.0.1 --port 8000
```

A quoted argument containing a space is treated as a FEN; anything else is
parsed as a sequence of UCI moves from the starting position. A FEN must be
quoted because it contains spaces — that is the only rule.

## Search and evaluation

Alpha-beta with:

- iterative deepening, so an interrupted search still returns the best move
  found so far;
- a transposition table that is a fixed-size list of buckets, so a lookup can
  never block on a resize or a GC pass;
- principal-variation search and null-window pruning;
- quiescence search over captures, promotions and en-passant up to 10 plies,
  which is what stops the classic horizon effect of trading a queen for a pawn
  and thinking it has won material;
- check extensions and null-move pruning;
- MVV/LVA move ordering with killers and a history heuristic;
- repetition, fifty-move and insufficient-material draws.

Evaluation is tapered between a middlegame and an endgame score, with material,
piece-square tables, pawn structure (passed, doubled, isolated, backward
pawns), mobility, king safety and rook-on-open-file terms.

```python
result = foxchess.search(board, foxchess.SearchLimits(depth=6, movetime_ms=2000))
print(result.pv_uci())       # the principal variation, as UCI strings
print(result.score)          # centipawns, from the side to move's point of view
print(result.mate_in)        # moves to mate, sign-corrected, or None
print(result.terminal)       # "checkmate" / "stalemate" / "draw", else None
print(result.nodes, result.nps)
```

A search never mutates the board it is given, and the transposition table is
optional and owned by the caller, so a long-lived service can reuse one:

```python
table = foxchess.TranspositionTable(64)        # megabytes
result = foxchess.search(board, limits, tt=table)
```

## Correctness

The engine is verified rather than asserted to be right:

- **Perft** against the published counts for six standard positions, to depth 4
  (~10.5 million nodes) in the slow test set.
- **Differential testing** against `python-chess` — every legal move, castling
  right, en-passant square and FEN round trip is compared move for move. This
  is a development-only dependency; the engine itself needs nothing.
- **Zobrist invariance** at every node of every perft, so an incremental hash
  bug cannot hide behind a correct move count.
- **PV legality** — the reported line is replayed and asserted to be legal, at
  every depth, because a search that returns an illegal move is worse than one
  that returns a bad move.

## Limits

These are deliberate, to keep a network-facing or untrusted-input path from
turning into a denial of service:

| Surface | Limit |
| ------- | ----- |
| API analysis depth | 1–8 |
| API perft depth | 0–4 |
| CLI analysis depth | 1–8 |
| Maximum PV / recursion depth | `MAX_PLY` |

Search extensions mean a legal PV can be longer than the nominal depth; the
invariant is that it is legal, that `pv[0]` is the returned best move, and that
it is never longer than `MAX_PLY`.

## Development

```console
$ pip install -e ".[dev]"
$ pytest                 # fast suite; the slow set is deselected
$ pytest -m slow         # deep perft and long searches (~16 min)
$ ruff check .
$ mypy
```

`pytest` runs with `filterwarnings = ["error"]`, so a new warning fails the build
rather than scrolling past.

The `differential` marker selects the `python-chess` comparison tests:
`pytest -m differential`.

`tools/perft_check.py` and `tools/_smoke.py` are the same checks in a form you
can point at a position by hand.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together and
[CHANGELOG.md](CHANGELOG.md) for what changed.

## License

MIT. See [LICENSE](LICENSE).

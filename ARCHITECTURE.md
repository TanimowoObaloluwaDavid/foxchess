# Architecture

A map of the codebase, for someone who wants to change it rather than just call
it. Module sizes, roughly:

| Module | Lines | Responsibility |
| ------ | ----- | -------------- |
| `board.py` | ~880 | Position state, legality, make/unmake, FEN, draw detection |
| `search.py` | ~690 | Iterative deepening, alpha-beta, quiescence, ordering, limits |
| `notation.py` | ~460 | SAN, long algebraic, PGN read and write |
| `evaluate.py` | ~390 | Tapered evaluation and its term tables |
| `types.py` | ~300 | Square and piece encodings, enums, direction and distance tables |
| `uci.py` | ~290 | The UCI protocol and the threaded search loop |
| `api.py` | ~250 | The JSON analysis service and its HTTP handler |
| `cli.py` | ~175 | The `foxchess` console command |
| `attacks.py` | ~230 | Precomputed sliding, knight, king and pawn attack tables |
| `move.py` | ~215 | Packed moves and their codecs |
| `tt.py` | ~115 | The fixed-size transposition table |
| `perft.py` | ~70 | Node counting |
| `zobrist.py` | ~75 | Hash keys and key construction |

Nothing imports anything above it except `search` and the entry points, so the
dependency order is: `types` → `move` → `zobrist` → `attacks` → `board` →
`notation`/`evaluate`/`perft`/`tt` → `search` → `uci`/`api`/`cli`.

## The representation

Squares are `0..63` with **a1 = 0** and h8 = 63, so `(sq >> 3)` is the rank and
`sq & 7` the file. This is a deliberate departure from the `a8 = 0` layout that
most chess code uses: reading the rank and file out of a square is done
constantly in this engine, and having them match what a human writes on a board
is worth more here than the habit of a FEN parse.

A piece is one `int` combining a `PieceType` and a `Color`, via
`make_piece(type, color)`. Index 0 is reserved as "no piece", which is what lets
a 16-entry list hold every piece on the board — see `PIECE_KEYS` in
`zobrist.py`, which is indexed by exactly that code.

Pieces live in `Board.bitboards[piece_code]`, with `occupancy[0]` for everything
and `occupancy[1]`/`occupancy[2]` for one side. `Board.mailbox[sq]` is a plain
list mirroring the bitboards, so a `Board` is deliberately *not* a single
bitboard structure: the mailbox exists because some rules (promotion, castling
rights, en-passant legality) are clearer against a per-square array than against
bit twiddling, and the two are kept in step by `make_move`/`unmake_move`.

### Moves are integers

A move is a packed `int`: from-square, to-square, and a 3-bit move flag (quiet,
double push, capture, en passant, promotion, with promotion piece as a
sub-index). Keeping moves as `int` is what makes move generation allocation-free
— the hot loop yields `int`s and allocates nothing per node.

The cost is that a caller cannot read a move without the codecs in `move.py`:
`to_uci`, `from_uci`, `decode_from`, `decode_to`. `from_uci` takes the legal
moves alongside the text, because a four-character UCI string genuinely does not
determine the flag — `e2e4` is a double push and `e5d6` may be an en passant
capture — and only the position can settle it.

`NO_MOVE = -1` is the single sentinel, re-exported from both `move` and
`search` so a caller comparing against it cannot end up with two different `-1`s
that happen to be equal. It formats as `0000`, which is what UCI's `bestmove`
expects for a position with no move to make.

## Move generation

`attacks.py` precomputes rays and blocker distances, and builds the classic
"whole ray minus the first blocker" attack set at import time. It is **not**
magic bitboards.

That is a considered choice, not an oversight. Magic lookups win in C because a
multiply is nearly free; in CPython the interpreter overhead around the lookup
dominates whatever the arithmetic saves. The whole-ray version is also far easier
to verify by eye, which matters more than raw nps for a project whose
correctness claims rest on perft.

`board.py` generates pseudo-legal moves and then filters them: `is_attacked` on
the king after the move, with pinned pieces handled by the same check. Making
everything legal up front would mean a second, slower generator; making
everything pseudo-legal and filtering at the boundary means one generator and one
filter, and `generate_legal()` is the only entry point most code uses.

### Legality by verification, not by construction

Castling, en passant and the two rare check-evasion cases are checked rather
than filtered in a way that has to be proved correct. Each rule has a test that
tries to break it specifically, because these are the rules that are wrong in
most implementations.

## The hash

`zobrist.py` generates 12 piece tables of 64 keys, 16 castling keys, 8 en-passant
file keys, and one side key, from a fixed seed. The fixed seed is a feature:
reproducible hashes mean a key from a previous run can be reasoned about, and
reproducible test failures.

`Board` maintains the key **incrementally** — every make and unmake xors the
difference in piece, castling, en-passant and side. `Board.compute_key()`
recomputes it from scratch, and `Board.verify_key()` compares. The test suite
calls `verify_key()` at every node of every perft, which is what makes the
incrementally-maintained hash trustworthy: perft proves the move *count* is
right, and only the hash check proves the *state* is right at every node.

The en-passant key is indexed by **file**, not by square, and is only mixed in
when an en-passant capture is actually available. This matches what python-chess
and Stockfish emit in a FEN, and it means two positions that are the same
position get the same key instead of splitting the transposition table for a
distinction nobody can act on.

`zobrist.compute()` is also public, for hashing a position a caller holds as raw
bitboards. It is tested directly against `Board.compute_key()`, because
`verify_key()` never calls it and a bug there would otherwise be invisible.

## Search

`search.py` is iterative deepening over negamax with alpha-beta.

- **Mate is scored by ply**, as `±(MATE_SCORE - ply)`, so a mate in 3 beats a
  mate in 5 and both beat a large material edge. The stored TT score is
  normalised back out of that ply offset before it is written and re-applied on
  read, because a score that was correct at ply 4 is wrong at ply 20.
- **Quiescence** (`_quiescence`) runs past the depth limit over captures,
  promotions and en passant, to `MAX_QUIESCENCE_PLY` (10). This is the horizon
  defence: without it the engine happily trades a queen for a pawn and calls the
  position won.
- **Extensions**: checks extend, which is why a PV can be legal and longer than
  the nominal depth. The invariants the tests assert are legality, that
  `pv[0] == best_move`, and `len(pv) <= MAX_PLY` — never `len(pv) <= depth`.
- **Null-move pruning** with a verification search, disabled in likely-zugzwang
  positions.
- **PV handling**: `state.pv[ply]` is claimed at the top of every `_negamax`
  entry, before any early return. It is not appended to, because a node reached
  twice through different parents would otherwise concatenate two unrelated
  lines.
- **TT cutoffs are disabled on full-window nodes.** A cutoff returns immediately
  and has no PV to offer, so cutting on the PV nodes produced truncated — and
  occasionally illegal — reported lines. Cutoffs still happen at null-window
  nodes, where the line is discarded anyway.

### Ordering

Captures by MVV-LVA, then killers (two per ply), then the history heuristic.
Quiescence captures are ordered by victim value, because a capture that fails
high should be tried before one that is merely plausible.

### Limits

`Searcher.check_limits()` is called every `CHECK_INTERVAL` nodes and raises
`SearchStopped`, which the iterative-deepening loop catches. It is caught rather
than propagated because a stopped search still has an answer: the best move from
the last completed iteration, which is the entire reason for iterative deepening.
`Searcher.stop()` sets a flag, which is how `go infinite` + `stop` returns
immediately from another thread.

## Evaluation

`evaluate.py` is tapered: a middlegame score and an endgame score, blended by a
phase computed from the remaining material, so the same weights stop counting a
knight as eight pawns once the queens are off.

Terms: material, piece-square tables, pawn structure (passed, doubled,
isolated, backward), mobility, king safety (attacked squares near the king, pawn
shield, passed-pawn proximity to a king), and rooks on open and semi-open files.

One asymmetry is deliberate and worth knowing about: **king danger is scored for
the side that just moved**, not from the mover's point of view. A full evaluation
is therefore not colour-mirrored, and `evaluate(board, detailed=False)` — the
cheap score a TT probe wants — is the one that is guaranteed symmetric. The
tests check the symmetry on the cheap path only, and say why.

## The entry points

Three thin layers over the same core, and each is importable without dragging in
the others:

- `uci.py` — the protocol. `go infinite` runs the search on a background thread
  so `stop` is answered at once; `bestmove` is emitted exactly once on every
  path, including a position that is already finished (`0000`).
- `api.py` — `ThreadingHTTPServer` plus a small `AnalysisService` that owns the
  table and serialises access to it with a lock. `build_server()` is the seam
  the tests use to get an ephemeral port and a real socket round trip.
- `cli.py` — `argparse`. A quoted argument containing a space is a FEN,
  otherwise the tokens are UCI moves from the start position. That rule is the
  whole convention.

`api`, `uci` and `cli` are **not** imported by `__init__`, so a plain
`import foxchess` does not pay for a socket module and a thread module.

### A naming sharp edge

The package re-exports functions under the same names as the submodules that
define them: `foxchess.search` is the *function*, `foxchess.move` and
`foxchess.notation` are the *modules*. `from foxchess import search` gives the
function. To reach a shadowed module, use
`importlib.import_module("foxchess.search")`. This is pinned by a test, because
it has surprised twice.

## Testing

- **Perft** against published counts, six positions, to depth 4 in the slow set.
- **Differential** against `python-chess` (development-only): legal moves,
  castling rights, en-passant squares, SAN agreement both directions, and FEN
  round trips. A shared move set with a divergent SAN is still a bug, because
  SAN is what ends up in a PGN.
- **Invariants at every node**: hash consistency, board equality after search.
- **Pinned public surface**: every name in `__all__` exists, submodules import
  cleanly, the version matches the packaging metadata.

`pytest` runs with `filterwarnings = ["error"]`, so a new warning is a test
failure.

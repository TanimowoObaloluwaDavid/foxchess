# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project uses [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-28

First release.

### Added

**Core engine**

- Bitboard position representation: pieces as integers, 12 piece codes in a
  16-entry table, a per-square mailbox alongside the bitboards, a1 = 0.
- Packed integer moves, with `to_uci`, `from_uci`, `decode_from` and `decode_to`
  as the public codecs, and `NO_MOVE = -1` as the single shared sentinel.
- Move generation producing pseudo-legal moves, filtered by verifying that the
  king is not left attacked — one generator, one filter, rather than two paths
  to keep in agreement.
- Precomputed sliding, knight, king and pawn attack tables using the whole-ray /
  first-blocker formulation rather than magic bitboards, which is faster in
  CPython and far easier to verify.
- Full rule set: castling rights and paths, en passant, promotion, the
  fifty-move rule, threefold repetition and insufficient material.
- FEN in and out, matching python-chess on the en-passant square: the target
  appears only when a capture is actually available.
- Zobrist hashing from a fixed seed, maintained incrementally on every move, with
  `verify_key()` to check the incremental value against a fresh computation.
- Long algebraic notation and SAN in both directions, and PGN read and write.
- `perft` and `perft divide`.

**Search**

- Iterative deepening negamax with alpha-beta.
- Quiescence search over captures, promotions and en passant, to 10 plies.
- Check extensions and null-move pruning with verification.
- Mate scores by ply, normalised against the ply offset before and after a
  transposition-table round trip.
- Principal-variation search and null-window pruning.
- MVV/LVA ordering with killers and a history heuristic.
- A fixed-size transposition table of buckets, so a probe can never block on a
  resize or a garbage-collection pass.
- `SearchLimits` covering depth, movetime, clock and increment, node count and
  infinite search; `Searcher.stop()` for interrupting from another thread.
- `SearchStopped`, caught inside the search so an interrupted search still
  returns the best move from its last completed iteration.

**Evaluation**

- Tapered between middlegame and endgame by a material-derived phase.
- Material, piece-square tables, pawn structure, mobility, king safety and rooks
  on open and semi-open files.

**Interfaces**

- `foxchess` console command: `analyse`/`analyze`, `perft`, `fen`, `moves`,
  `pgn` and `serve`.
- `foxchess-uci`: the parts of the UCI protocol a GUI uses, including threaded
  `go infinite` and an immediately-answered `stop`.
- A dependency-free JSON HTTP service: `/health`, `/fen`, `/perft` and
  `/analyze`, on `http.server`.
- Inline type hints throughout, with `py.typed` and a `mypy --strict` clean
  configuration.
- No runtime dependencies. Python 3.11 or newer.

**Verification**

- Perft against the published counts for six standard positions, to depth 4.
- Differential testing against `python-chess`, a development-only dependency,
  over legal moves, castling rights, en-passant squares, SAN in both directions
  and FEN round trips.
- Zobrist and board-state invariants asserted at every node of every perft.
- Reported principal variations replayed and asserted legal at every depth.

[1.0.0]: https://github.com/TanimowoObaloluwaDavid/foxchess/releases/tag/v1.0.0

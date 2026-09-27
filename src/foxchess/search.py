"""Alpha-beta search: negamax, iterative deepening, quiescence, transposition table.

The classic set of techniques, written for a Python interpreter rather than
ported from C. The decisions that actually matter here:

* **Ordering is the speed.** A recursive call costs orders of magnitude more than
  a sort, so moves are ordered by a cheap key and only then iterated.
* **Mate is scored by ply**, as ``±(MATE_SCORE - ply)``. A mate in 3 must beat a
  mate in 5, and mate scores must be recognisable as such.
* **TT scores are stored ply-relative** and re-based on probe. Storing a mate as
  "mate in 5" from a node at ply 3 and "mate in 5" from ply 7 fills the table with
  near-duplicates that all look like long mates.
* **The principal variation uses a triangular table**, indexed by ply, not a
  dictionary keyed by position. A dict keyed by position is subtly wrong: the same
  position reached by transposition has one PV slot, so whichever path finished
  first wins and the reported line can be a move the search never verified.
* **Only threefold repetition scores as a draw.** Twofold is claimable, not
  forced, and returning 0 for it throws away real winning chances. The search
  needs no termination guarantee from this: it is depth-bounded.

Check extensions are taken only for moves that give check *out of* check, not for
every checking move in a forcing sequence — the latter makes the tree explode and
foxchess has no node budget to pay for it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

from .board import Board
from .evaluate import INFINITY, MATE_SCORE, MATE_THRESHOLD, evaluate
from .move import (
    CAPTURE,
    CASTLE_FLAGS,
    DOUBLE_PUSH,
    EN_PASSANT,
    NO_MOVE,
    PROMOTION_PIECE,
    decode_flag,
    decode_from,
    decode_to,
    promotion_of,
    to_uci,
)
from .tt import EXACT, LOWER, UPPER, TranspositionTable
from .types import PIECE_VALUES, Color, PieceType, piece_type_of

__all__ = [
    "MAX_PLY",
    "MAX_QUIESCENCE_PLY",
    "NO_MOVE",
    "SearchLimits",
    "SearchResult",
    "SearchStopped",
    "Searcher",
    "search",
]

#: Deeper than any sane engine setting. Bounds the PV table and the ply arrays.
MAX_PLY: Final[int] = 128
DEFAULT_MAX_DEPTH: Final[int] = 64
#: Past this, one extra ply costs far more than it can possibly be worth.
MAX_USEFUL_DEPTH: Final[int] = 32
#: How far past the nominal depth the capture search may run.
#:
#: Quiescence is meant to stop reading a score in the middle of an exchange, which
#: takes two or three plies. Without a cap it also chases chains the exchange search
#: was never meant to cover: in a dense middlegame, a discovered check is easy to
#: come by, and every evasion is a fresh node. Kiwipete reached ply 28 and ~47k
#: nodes from a *depth 1* search before this bound existed, which in pure Python is
#: the difference between half a second and a minute. Exchanges that run longer than
#: this are a static-exchange question, and the search answers it with the eval.
MAX_QUIESCENCE_PLY: Final[int] = 10

# ``NO_MOVE`` is imported from the move layer, where it is defined, and re-exported
# from here because a caller inspecting a :class:`SearchResult` needs the same
# sentinel the search used to produce it.

_CAPTURE_FLAGS: Final[frozenset[int]] = frozenset(
    {CAPTURE, EN_PASSANT}
    | {flag for flag, _ in PROMOTION_PIECE.items() if flag >= 12}
)
_FIRST_PAWN_RANK_BB: Final[int] = 0xFFFF


class SearchStopped(Exception):
    """Raised internally to unwind the tree when a limit is reached."""


@dataclass(slots=True)
class SearchLimits:
    """UCI ``go`` limits. ``None`` means unbounded."""

    depth: int = DEFAULT_MAX_DEPTH
    movetime_ms: int | None = None
    wtime_ms: int | None = None
    btime_ms: int | None = None
    winc_ms: int = 0
    binc_ms: int = 0
    movestogo: int | None = None
    nodes: int | None = None
    infinite: bool = False
    #: Whose clock the search is running on. Set by :meth:`Searcher.search`.
    white_to_move: bool = True

    def budget_ms(self) -> int | None:
        """Milliseconds allowed for this search, or ``None`` for no time limit."""
        if self.movetime_ms is not None:
            return max(1, self.movetime_ms)
        if self.wtime_ms is None or self.btime_ms is None:
            return None
        clock = self.wtime_ms if self.white_to_move else self.btime_ms
        increment = self.winc_ms if self.white_to_move else self.binc_ms
        to_go = min(self.movestogo, 30) if self.movestogo else 30
        # A fraction of the clock plus the increment. Engines that spend the whole
        # remaining time lose to timeouts; this is the conventional split.
        return max(1, (clock + increment * 3) // to_go)


@dataclass(slots=True)
class SearchResult:
    """The outcome of a completed search."""

    best_move: int = NO_MOVE
    score: int = 0
    depth: int = 0
    nodes: int = 0
    nps: int = 0
    time_ms: int = 0
    principal_variation: list[int] = field(default_factory=list)
    #: ``True`` when a limit cut the search short rather than the tree ending.
    stopped: bool = False
    terminal: str | None = None

    @property
    def mate_in(self) -> int | None:
        """Moves to mate, sign-corrected, or ``None`` if the score is not a mate."""
        if abs(self.score) < MATE_THRESHOLD:
            return None
        plies = MATE_SCORE - abs(self.score)
        return plies if self.score > 0 else -plies

    @property
    def is_mate(self) -> bool:
        return abs(self.score) >= MATE_THRESHOLD

    def pv_uci(self) -> list[str]:
        return [to_uci(move) for move in self.principal_variation]

    def as_dict(self) -> dict[str, object]:
        return {
            "bestmove": None if self.best_move == NO_MOVE else to_uci(self.best_move),
            "score": self.score,
            "mate": self.mate_in,
            "depth": self.depth,
            "nodes": self.nodes,
            "nps": self.nps,
            "time_ms": self.time_ms,
            "pv": self.pv_uci(),
            "terminal": self.terminal,
        }


@dataclass(slots=True)
class _State:
    """Per-search mutable state, kept off the call stack."""

    limits: SearchLimits = field(default_factory=SearchLimits)
    nodes: int = 0
    #: Absolute ``time.perf_counter()`` value past which the search must stop.
    deadline: float | None = None
    stopped: bool = False
    tt: TranspositionTable | None = None
    killers: list[list[int]] = field(default_factory=lambda: [[] for _ in range(MAX_PLY)])
    history: dict[int, int] = field(default_factory=dict)
    #: Zobrist keys of every ancestor position, used for threefold detection.
    path: list[int] = field(default_factory=list)
    #: ``pv[ply][ply:]`` is the principal variation from the node at ``ply``.
    pv: list[list[int]] = field(default_factory=lambda: [[] for _ in range(MAX_PLY)])

    def reset(self, limits: SearchLimits) -> None:
        self.limits = limits
        self.nodes = 0
        self.stopped = False
        self.killers = [[] for _ in range(MAX_PLY)]
        self.history.clear()
        self.path.clear()
        self.pv = [[] for _ in range(MAX_PLY)]
        budget = limits.budget_ms()
        self.deadline = None if budget is None else time.perf_counter() + budget / 1000

    def check_limits(self) -> None:
        """Cheap poll, run every 1024 nodes and at each root iteration."""
        if self.stopped:
            return
        out_of_nodes = self.limits.nodes is not None and self.nodes >= self.limits.nodes
        out_of_time = self.deadline is not None and time.perf_counter() >= self.deadline
        if out_of_nodes or out_of_time:
            self.stopped = True


class Searcher:
    """A negamax searcher over one :class:`~foxchess.board.Board`."""

    __slots__ = ("board", "on_update", "state")

    def __init__(self, board: Board, *, tt: TranspositionTable | None = None) -> None:
        self.board = board
        self.state = _State(tt=tt)
        #: Progress callback ``(depth, score, pv)``; the UCI loop streams ``info``.
        self.on_update: Callable[[int, int, list[int]], None] | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def search(self, limits: SearchLimits | None = None) -> SearchResult:
        """Search iteratively deeper, returning the best move from the last
        *completed* depth.

        An interrupted iteration is discarded outright: at the root a cut-off
        yields a bound, not a best move, so reporting it would be a lie.
        """
        board = self.board
        self.state.reset(limits or self.state.limits)
        self.state.limits.white_to_move = board.side is Color.WHITE

        started = time.perf_counter()
        legal = board.legal_moves()
        if not legal:
            return self._terminal_result(started)

        if len(legal) == 1:
            return self._forced_result(legal[0], started)

        best = SearchResult(best_move=legal[0], principal_variation=[legal[0]], stopped=True)
        limit = min(MAX_USEFUL_DEPTH, self.state.limits.depth, MAX_PLY - 2)
        for depth in range(1, limit + 1):
            self.state.pv = [[] for _ in range(MAX_PLY)]
            self.state.check_limits()
            if self.state.stopped:
                break
            try:
                score, move, line = self._root(depth)
            except SearchStopped:
                break
            elapsed = max(1, int((time.perf_counter() - started) * 1000))
            best = SearchResult(
                best_move=move,
                score=score,
                depth=depth,
                nodes=self.state.nodes,
                nps=int(self.state.nodes * 1000 / elapsed),
                time_ms=elapsed,
                principal_variation=line,
                stopped=depth >= limit or self.state.limits.infinite,
            )
            if self.on_update is not None:
                self.on_update(depth, score, line)
            if abs(score) >= MATE_THRESHOLD:
                break
        return best

    def stop(self) -> None:
        """Ask a running search to finish at the next poll.

        Safe to call from another thread, which is how UCI ``stop`` works: the
        search notices within 1024 nodes and returns the best move from the last
        completed depth. Calling this when no search is running does nothing, so
        a stray ``stop`` is not an error.
        """
        self.state.stopped = True

    def best_move(self, depth: int = 4) -> SearchResult:
        """Search to a fixed depth, ignoring every clock and node budget.

        A convenience for scripts and tests. Anything playing a real game wants
        :meth:`search` with a :class:`SearchLimits` instead.
        """
        return self.search(SearchLimits(depth=depth))

    # ------------------------------------------------------------------
    # Root
    # ------------------------------------------------------------------

    def _root(self, depth: int) -> tuple[int, int, list[int]]:
        """Full-window root search. No aspiration window on purpose.

        An aspiration window needs a re-search on failure, and at this node rate
        in Python the re-search costs more than the window ever saves. A
        transposition table recovers the shared subtree anyway.
        """
        board = self.board
        state = self.state
        moves = self._order(state, 0, self._tt_hint())
        alpha, beta = -INFINITY, INFINITY
        best_score, best_move = -INFINITY, NO_MOVE

        for index, move in enumerate(moves):
            board.make_move(move)
            state.path.append(board.key)
            if index == 0:
                score = -self._negamax(depth - 1, -beta, -alpha, 1)
            else:
                # Null window first; widen only if the move turns out to be good.
                score = -self._negamax(depth - 1, -alpha - 1, -alpha, 1)
                if score > alpha:
                    score = -self._negamax(depth - 1, -beta, -alpha, 1)
            state.path.pop()
            board.unmake_move()

            if score > best_score:
                best_score, best_move = score, move
                state.pv[0] = [move, *state.pv[1][: MAX_PLY - 1]]
                alpha = max(alpha, score)
            if state.stopped:
                raise SearchStopped

        if best_move == NO_MOVE:
            raise SearchStopped
        line = state.pv[0]
        return best_score, best_move, line if line else [best_move]

    def _terminal_result(self, started: float) -> SearchResult:
        elapsed = max(1, int((time.perf_counter() - started) * 1000))
        board = self.board
        if board.in_check():
            return SearchResult(
                score=-MATE_SCORE, depth=0, time_ms=elapsed, stopped=True, terminal="checkmate"
            )
        if board.is_stalemate():
            reason = "stalemate"
        elif board.is_insufficient_material():
            reason = "material"
        else:
            reason = "draw"
        return SearchResult(score=0, depth=0, time_ms=elapsed, stopped=True, terminal=reason)

    def _forced_result(self, move: int, started: float) -> SearchResult:
        board = self.board
        board.make_move(move)
        score = -self._negamax(3, -INFINITY, INFINITY, 1)
        board.unmake_move()
        elapsed = max(1, int((time.perf_counter() - started) * 1000))
        return SearchResult(
            best_move=move,
            score=score,
            depth=1,
            nodes=self.state.nodes,
            nps=int(self.state.nodes * 1000 / elapsed),
            time_ms=elapsed,
            principal_variation=[move],
            stopped=True,
        )

    # ------------------------------------------------------------------
    # Negamax
    # ------------------------------------------------------------------

    def _negamax(self, depth: int, alpha: int, beta: int, ply: int) -> int:
        state = self.state
        state.nodes += 1
        if not state.nodes & 1023:
            state.check_limits()
            if state.stopped:
                raise SearchStopped
        # Claim this ply's line before anything can return early.
        #
        # Plenty of exits below never choose a move -- a draw, a transposition cut,
        # a null move, the leaf, a mate on the spot -- and each of them leaves
        # ``pv[ply]`` holding whatever the previous visit wrote. The parent then
        # builds its own line as ``[move] + pv[ply + 1]`` and inherits that stale
        # tail, which is how a principal variation ends up advertising a move from
        # a square nothing stands on. Clearing on entry makes every path either
        # write a real line or write nothing at all.
        state.pv[ply] = []
        if ply >= MAX_PLY - 2:
            return self._stand_pat()

        board = self.board
        in_check = board.in_check()

        if ply and self._is_draw_position():
            return 0
        # Mate-distance pruning is deliberately absent. The usual formulation
        # narrows the window to the already-proven mate distance, and getting
        # that clamp wrong is invisible: alpha and beta cross on the first
        # ply, every move scores "mate in one", and the engine plays nonsense
        # while looking perfectly healthy. Mate scores already terminate the
        # search at the root, so the correctness win from MDP is not worth a
        # whole class of silent bugs. It can come back with a test.

        if depth <= 0:
            return self._quiescence(alpha, beta, ply)

        # No transposition cut on a full window.
        #
        # A cutoff returns a score and nothing else, so the principal variation
        # stops dead at that node. That is harmless for correctness -- the line
        # returned is still a real prefix of a real line -- but it is a poor thing
        # to show a caller, and it happens constantly: with a table in play, the
        # root's own child is usually a hit, so `analyse --depth 5` reports a
        # one-move line. A null-window node is a scout whose answer is a bound, and
        # there is no variation to preserve, so the table still saves all the work
        # it can there.
        if beta - alpha > 1:
            tt_cutoff = None
        else:
            tt_cutoff = self._tt_cutoff(ply, depth, alpha, beta)
        if tt_cutoff is not None:
            return tt_cutoff

        tt_move = self._tt_hint()
        moves = self._order(state, ply, tt_move)

        if (
            not in_check
            and depth >= 3
            and ply > 0
            and not self._in_endgame(board)
        ):
            # Null move: hand the opponent a free move. If the position is still
            # good enough to fail high, the real move ordering is not going to
            # help. Never at the root, and never in pawn endgames, where zugzwang
            # makes passing the initiative actively bad.
            reduction = 3 if depth > 6 else 2
            board.make_null_move()
            score = -self._negamax(depth - 1 - reduction, -beta, -beta + 1, ply + 1)
            board.unmake_null_move()
            if score >= beta:
                return MATE_SCORE if score >= MATE_THRESHOLD else beta

        best_score, best_move = -INFINITY, NO_MOVE
        for index, move in enumerate(moves):
            quiet = decode_flag(move) in (0, DOUBLE_PUSH)
            board.make_move(move)
            state.path.append(board.key)

            # Check extension, decided after the move is made so it costs nothing:
            # only escapes and checks that come from a quiet position, so a forced
            # sequence cannot extend forever.
            extension = 1 if (not in_check and board.in_check()) else 0
            reduction = 0
            if (
                index >= 3
                and depth >= 3
                and quiet
                and not in_check
                and not extension
            ):
                # Late move reductions: a quiet move this far down a well-ordered
                # list is unlikely to be best, so try it shallow and re-search only
                # if the shallow pass exceeds alpha.
                reduction = 1 if index < 8 else 2

            if index == 0:
                score = -self._negamax(depth - 1 + extension, -beta, -alpha, ply + 1)
            else:
                score = -self._negamax(
                    depth - 1 - reduction + extension, -alpha - 1, -alpha, ply + 1
                )
                if score > alpha and (reduction or extension):
                    score = -self._negamax(
                        depth - 1 + extension, -alpha - 1, -alpha, ply + 1
                    )
                if score > alpha and score < beta:
                    score = -self._negamax(depth - 1 + extension, -beta, -alpha, ply + 1)

            state.path.pop()
            board.unmake_move()

            if score > best_score:
                best_score, best_move = score, move
                state.pv[ply] = [move, *state.pv[ply + 1][: MAX_PLY - ply - 1]]
                if score > alpha:
                    alpha = score
                    if score >= beta:
                        if quiet:
                            self._record_killer(ply, move)
                            self._record_history(move, depth)
                        break

        if best_move == NO_MOVE:
            # No legal move: mate if in check, otherwise stalemate.
            return -MATE_SCORE + ply if in_check else 0

        self._tt_store(ply, best_score, best_move, depth, alpha, beta)
        return best_score

    # ------------------------------------------------------------------
    # Quiescence
    # ------------------------------------------------------------------

    def _quiescence(self, alpha: int, beta: int, ply: int) -> int:
        """Search only captures and promotions, so the score is never read
        part-way through an exchange.

        In check, stand-pat is forbidden: every move is an evasion, not a choice
        about whether to grab material.
        """
        state = self.state
        state.nodes += 1
        if not state.nodes & 1023:
            state.check_limits()
            if state.stopped:
                raise SearchStopped
        if ply >= MAX_PLY - 2:
            return self._stand_pat()

        board = self.board
        if ply >= MAX_QUIESCENCE_PLY:
            # Out of capture-search budget. Standing pat is still the right answer
            # unless we are in check, where it is not a choice we get to make.
            if not board.in_check():
                return self._stand_pat()
            best = -MATE_SCORE + ply
            for move in board.legal_moves():
                board.make_move(move)
                score = -self._stand_pat()
                board.unmake_move()
                best = max(best, score)
            return best

        if self._is_draw_position():
            return 0

        in_check = board.in_check()
        if in_check:
            stand_pat, moves = -MATE_SCORE + ply, board.legal_moves()
            moves.sort(key=self._evasion_key, reverse=True)
        else:
            stand_pat = self._stand_pat()
            if stand_pat >= beta:
                return stand_pat
            moves = board.captures()
            moves.sort(key=self._capture_key, reverse=True)

        best = stand_pat
        for move in moves:
            # Delta pruning: if even winning this outright falls short of alpha,
            # the deeper search cannot rescue it. Standing pat already covers
            # being in check, where there is no stand-pat score to be cautious
            # about -- every evasion has to be searched.
            if not in_check and stand_pat + self._capture_gain(move) + 200 < alpha:
                continue
            board.make_move(move)
            state.path.append(board.key)
            score = -self._quiescence(-beta, -alpha, ply + 1)
            state.path.pop()
            board.unmake_move()
            if score > best:
                best = score
                if score >= beta:
                    return score
                alpha = max(alpha, score)
        return best

    def _stand_pat(self) -> int:
        """Static score, from the side to move's point of view."""
        score = evaluate(self.board)
        return -score if self.board.side is Color.BLACK else score

    # ------------------------------------------------------------------
    # Move ordering
    # ------------------------------------------------------------------

    def _order(self, state: _State, ply: int, tt_move: int | None) -> list[int]:
        """Legal moves, best first.

        Tiers, in order: transposition-table move, captures, then castling and
        promotions, then in-check evasions, then killer and history for quiet
        moves. The sort key is pure arithmetic on purpose — a recursive call costs
        so much more than a comparison that ordering deserves real effort, but not
        at the price of a Python-level call per move.
        """
        board = self.board
        moves = board.legal_moves()
        in_check = board.in_check()
        killers = state.killers[ply] if ply < MAX_PLY else ()
        history = state.history

        def key(move: int) -> tuple[int, int]:
            if move == tt_move:
                return 1 << 24, decode_to(move)
            flag = decode_flag(move)
            if flag in _CAPTURE_FLAGS:
                tier = 1 << 20
            elif flag in CASTLE_FLAGS or flag >= 8:
                tier = 1 << 19
            elif in_check:
                tier = 1 << 18
            else:
                tier = 1 << 17
                for killer in killers:
                    if move == killer:
                        tier = 1 << 18
                        break
                tier += history.get(move, 0)
            if not in_check:
                # Prefer the centre. It matters for captures (a piece in the centre
                # is worth more) and mildly for quiet moves, so one nudge serves both.
                tier -= abs((decode_to(move) & 7) - 3) + abs((decode_to(move) >> 3) - 3)
            return tier, decode_to(move)

        moves.sort(key=key, reverse=True)
        return moves

    def _capture_key(self, move: int) -> int:
        """Most Valuable Victim: take the most valuable piece with the least able."""
        gain = self._capture_gain(move)
        promotion = promotion_of(move)
        if promotion is not None:
            gain += PIECE_VALUES[promotion]
        return gain

    def _evasion_key(self, move: int) -> int:
        """When in check, take the checker first, then centralise."""
        tier = 1 << 20 if decode_flag(move) in _CAPTURE_FLAGS else 0
        return tier - abs((decode_to(move) & 7) - 3)

    def _capture_gain(self, move: int) -> int:
        """MVV-LVA: the victim's value, less a nudge for the attacker giving it up.

        This is asked about *every* move quiescence considers, evasions and quiet
        checks included, so the target square is usually empty. ``piece_at`` returns
        0 for an empty square and ``piece_type_of(0)`` is not a valid enum member,
        so the emptiness check has to happen on the raw piece code.
        """
        board = self.board
        if decode_flag(move) == EN_PASSANT:
            return PIECE_VALUES[PieceType.PAWN]
        victim_code = board.piece_at(decode_to(move))
        victim = PIECE_VALUES[piece_type_of(victim_code)] if victim_code else 0
        attacker = piece_type_of(board.piece_at(decode_from(move)))
        return victim - PIECE_VALUES[attacker] // 16

    # ------------------------------------------------------------------
    # Heuristic memory
    # ------------------------------------------------------------------

    def _record_killer(self, ply: int, move: int) -> None:
        if ply >= MAX_PLY:
            return
        slot = self.state.killers[ply]
        if move in slot:
            return
        slot.insert(0, move)
        del slot[2:]

    def _record_history(self, move: int, depth: int) -> None:
        state = self.state
        updated = state.history.get(move, 0) + depth * depth
        if updated > 1 << 20:
            # Decay in place: rebuilding the dict every time is not worth it.
            state.history = {key: value >> 1 for key, value in state.history.items()}
            updated >>= 1
        state.history[move] = updated

    # ------------------------------------------------------------------
    # Transposition table
    # ------------------------------------------------------------------

    def _tt_cutoff(self, ply: int, depth: int, alpha: int, beta: int) -> int | None:
        """Return a score that proves this node can be cut, or ``None``."""
        table = self.state.tt
        if table is None or ply == 0:
            return None
        entry = table.probe(self.board.key)
        if entry is None or entry.depth < depth:
            return None
        score = _from_tt_score(entry.score, ply)
        if entry.flag == EXACT:
            return score
        if entry.flag == LOWER and score > alpha:
            return score
        if entry.flag == UPPER and score < beta:
            return score
        return None

    def _tt_hint(self) -> int | None:
        table = self.state.tt
        if table is None:
            return None
        entry = table.probe(self.board.key)
        if entry is None or entry.move == NO_MOVE or entry.move < 0:
            return None
        return entry.move

    def _tt_store(
        self, ply: int, score: int, move: int, depth: int, alpha: int, beta: int
    ) -> None:
        table = self.state.tt
        if table is None or ply == 0:
            return
        if score <= alpha:
            flag = UPPER
        elif score >= beta:
            flag = LOWER
        else:
            flag = EXACT
        table.store(self.board.key, depth, _to_tt_score(score, ply), flag, move)

    # ------------------------------------------------------------------
    # Draws
    # ------------------------------------------------------------------

    def _is_draw_position(self) -> bool:
        """True when the game is drawn by rule rather than by a claimable score.

        Three things end a game without a mating move: a threefold repetition, the
        fifty-move rule, and a position with no legal sequence to checkmate. The
        last one is the reason this helper exists -- leaving it out let a bare
        bishop score as a bishop up in a dead position, and the search would then
        decline a draw it is entitled to and grind towards a fifty-move claim
        instead. The check costs two bitboard tests in any position that still has
        pawns, rooks or queens, which is every position worth searching.
        """
        board = self.board
        return (
            self._is_threefold(board.key)
            or board.halfmove_clock >= 100
            or board.is_insufficient_material()
        )

    def _is_threefold(self, key: int) -> bool:
        """True when this is the third time the position has occurred.

        ``state.path`` holds ancestor keys and does not yet include the current
        node, so the current occurrence is the ``+ 1``. Twofold is not a draw —
        it is claimable, and scoring it 0 throws away won positions.
        """
        if len(self.state.path) < 4:
            return False
        occurrences = 1
        for ancestor in self.state.path:
            if ancestor == key:
                occurrences += 1
                if occurrences >= 3:
                    return True
        return False

    @staticmethod
    def _in_endgame(board: Board) -> bool:
        """True when neither side has anything left to manoeuvre with.

        Used only to switch off null-move pruning, which is unreliable once the
        position is decided by pawn structure alone.
        """
        for piece_type in (PieceType.KNIGHT, PieceType.BISHOP, PieceType.ROOK, PieceType.QUEEN):
            if board.count(piece_type, Color.WHITE) or board.count(piece_type, Color.BLACK):
                return False
        return True


def _to_tt_score(score: int, ply: int) -> int:
    """Re-base a mate score so it is relative to the root, not the node."""
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def _from_tt_score(score: int, ply: int) -> int:
    """Undo :func:`_to_tt_score` for the node now being searched."""
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


def search(
    board: Board | str,
    limits: SearchLimits | None = None,
    *,
    tt: TranspositionTable | None = None,
) -> SearchResult:
    """Search a position and return the result.

    The one-shot form, for callers that do not want to own a searcher::

        result = search(Board(), SearchLimits(depth=6))

    Accepts a FEN as well as a board. The table is per-call unless one is passed
    in, so repeated calls share nothing -- which is fine for a single question and
    wasteful for a game. A :class:`Searcher` holding one table is the right shape
    for a loop. The board is left exactly as it was found.
    """
    return Searcher(Board(board) if isinstance(board, str) else board, tt=tt).search(limits)

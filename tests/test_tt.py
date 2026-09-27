"""Transposition table: storage, probing, and the statistics UCI reports.

The table is a flat list of buckets indexed by the low bits of the key, so two
different positions can share a slot. That is the point of the design rather than a
flaw: a bounded table can be replaced without allocating, and a stale slot costs a
recomputation, not a wrong move. The invariant that matters is therefore that
``probe`` never returns an entry belonging to a *different* key.
"""

from __future__ import annotations

import typing as tt

import pytest

from foxchess import tt as tt_module
from foxchess.tt import (
    BOUND_MASK,
    DEFAULT_SIZE_MB,
    EXACT,
    LOWER,
    UPPER,
    Entry,
    TranspositionTable,
)


def test_bounds_are_distinct_and_fit_the_mask() -> None:
    assert len({LOWER, UPPER, EXACT}) == 3
    for flag in (LOWER, UPPER, EXACT):
        assert flag & ~BOUND_MASK == 0, f"{flag} does not fit in the score's low bits"
        assert flag | BOUND_MASK == BOUND_MASK


def test_bound_alias_matches_the_flag_constants() -> None:
    """``Bound`` spells its members as raw numbers; keep it from drifting.

    ``Literal`` refuses a ``Final`` variable, so the alias cannot be written as
    ``Literal[LOWER, UPPER, EXACT]`` and the duplication is unavoidable. This
    test is the only thing tying the two spellings together.
    """
    assert set(tt.get_args(tt_module.Bound)) == {LOWER, UPPER, EXACT}


def test_capacity_is_a_power_of_two() -> None:
    """The index is a mask, so the bucket count has to be a power of two."""
    for size_mb in (1, 2, 4, 8, 16, 64, 256):
        capacity = TranspositionTable._capacity(size_mb)
        assert capacity & (capacity - 1) == 0, (size_mb, capacity)


def test_capacity_never_drops_below_the_floor() -> None:
    """A silly or negative size still yields a usable table."""
    for size_mb in (0, -5, 1):
        assert len(TranspositionTable(size_mb)) >= 1024


def test_capacity_tracks_the_requested_size() -> None:
    small = TranspositionTable(1)
    large = TranspositionTable(64)
    assert len(small) < len(large)
    assert len(large) == TranspositionTable._capacity(DEFAULT_SIZE_MB)


# ---------------------------------------------------------------------------
# Store and probe
# ---------------------------------------------------------------------------


def test_probing_an_empty_table_returns_nothing() -> None:
    assert TranspositionTable(1).probe(12345) is None


def test_a_stored_entry_comes_back_unchanged() -> None:
    table = TranspositionTable(1)
    table.store(key=0xABCD, depth=7, score=123, flag=EXACT, move=0x1234)
    entry = table.probe(0xABCD)
    assert entry is not None
    assert (entry.key, entry.depth, entry.score, entry.flag, entry.move) == (
        0xABCD, 7, 123, EXACT, 0x1234
    )


def test_probe_never_returns_another_keys_entry() -> None:
    """The whole safety argument: a wrong key is a miss, never a wrong entry."""
    table = TranspositionTable(1)
    table.store(key=1, depth=3, score=50, flag=EXACT)
    assert table.probe(2) is None
    assert table.probe(1 << 40) is None
    # Even a key that lands on the same bucket must miss: the key is compared in
    # full, not just the index bits.
    same_bucket = 1 + len(table)
    assert table.probe(same_bucket) is None
    assert table.probe(1) is not None


def test_storing_the_same_key_twice_replaces_it() -> None:
    table = TranspositionTable(1)
    table.store(key=7, depth=1, score=10, flag=UPPER)
    table.store(key=7, depth=9, score=99, flag=EXACT)
    entry = table.probe(7)
    assert entry is not None
    assert (entry.depth, entry.score, entry.flag) == (9, 99, EXACT)


def test_a_replaced_slot_does_not_inflate_the_fill_count() -> None:
    table = TranspositionTable(1)
    table.store(key=1, depth=1, score=0, flag=EXACT)
    table.store(key=1, depth=2, score=0, flag=EXACT)
    assert table.stats()["entries"] == 1


def test_negative_scores_round_trip() -> None:
    """A score with the bound packed into its low bits must not come back mangled."""
    table = TranspositionTable(1)
    for score in (-1, -30000, -1, 0, 1, 30000):
        table.store(key=42, depth=2, score=score, flag=EXACT)
        entry = table.probe(42)
        assert entry is not None and entry.score == score, score


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


def test_hits_are_counted_and_misses_are_not() -> None:
    table = TranspositionTable(1)
    table.store(key=5, depth=1, score=1, flag=EXACT)
    table.probe(5)
    table.probe(5)
    table.probe(6)
    assert table.stats()["hits"] == 2


def test_stores_and_collisions_are_counted() -> None:
    table = TranspositionTable(1)
    table.store(key=1, depth=1, score=0, flag=EXACT)
    assert table.stats()["collisions"] == 0
    table.store(key=1 + len(table), depth=1, score=0, flag=EXACT)
    assert table.stats()["collisions"] == 1
    assert table.stats()["entries"] == 1


def test_clear_empties_the_table_and_ages_the_generation() -> None:
    table = TranspositionTable(1)
    table.store(key=3, depth=4, score=8, flag=LOWER)
    table.probe(3)
    generation = table.generation
    table.clear()
    assert table.probe(3) is None
    assert table.generation == generation + 1
    # Occupancy must go back to zero as well: a table reused across games that
    # kept reporting the old game's fill would make `hashfull` meaningless.
    assert table.hashfull() == 0
    assert table.stats() == {
        "buckets": len(table),
        "entries": 0,
        "hits": 0,
        "collisions": 0,
        "clears": generation + 1,
    }


def test_clear_keeps_the_capacity() -> None:
    """A new game must not have to reallocate the table."""
    table = TranspositionTable(2)
    before = len(table)
    table.clear()
    assert len(table) == before


# ---------------------------------------------------------------------------
# hashfull
# ---------------------------------------------------------------------------


def test_hashfull_is_zero_when_empty_and_rises_as_it_fills() -> None:
    table = TranspositionTable(1)
    assert table.hashfull() == 0
    # Keys 0, 1, 2 ... land in successive buckets, so each store occupies a new
    # slot and the occupancy genuinely climbs.
    previous = 0
    for key in range(5000):
        table.store(key=key, depth=1, score=0, flag=EXACT)
        current = table.hashfull()
        assert current >= previous
        previous = current
    assert table.hashfull() == 5000 * 1000 // len(table)
    assert 0 < table.hashfull() < 1000


def test_hashfull_never_exceeds_the_permille_scale() -> None:
    table = TranspositionTable(1)
    for key in range(1, 20000):
        table.store(key=key, depth=1, score=0, flag=EXACT)
    assert 0 <= table.hashfull() <= 1000
    assert 0 <= table.hashfull(permille=100) <= 100


def test_hashfull_reflects_replacement_not_stores() -> None:
    """Overwriting the same slot must not report the table as fuller than it is."""
    table = TranspositionTable(1)
    for key in range(1, 1000):
        table.store(key=key, depth=1, score=0, flag=EXACT)
    assert table.hashfull() == table.stats()["entries"] * 1000 // len(table)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def test_entry_defaults_are_the_empty_state() -> None:
    entry = Entry()
    assert entry.key == 0
    assert entry.depth == -127, "a negative depth is below any real search depth"
    assert entry.move == -1


def test_entry_matches_only_its_own_key() -> None:
    entry = Entry(key=99)
    assert entry.matches(99)
    assert not entry.matches(98)
    assert not entry.matches(0)


def test_entry_uses_slots_so_instances_stay_small() -> None:
    """A per-entry ``__dict__`` would dominate the memory the size argument claims."""
    with pytest.raises(AttributeError):
        Entry().not_a_field = 1  # type: ignore[attr-defined]


def test_module_exports_are_stable() -> None:
    assert set(tt_module.__all__) == {"Entry", "TranspositionTable", "Bound"}
    assert tt_module.Bound is not None

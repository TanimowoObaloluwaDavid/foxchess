"""A fixed-size transposition table.

The table is a flat list of buckets rather than a dict, for two reasons: the
search must never block on a resize or a GC pass, and a bounded table can be
replaced wholesale by assigning a new list, which is what :meth:`clear` does.
Discarding the old list in one operation is a pointer store; there is no need
for a per-entry generation counter and no lazy invalidation scheme, because
there is no per-entry invalidation to do.

The probe/store logic lives here; deciding *when* an entry may be replaced is the
search's business and is passed in as a ``prefer`` flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

__all__ = ["Bound", "Entry", "TranspositionTable"]

#: The search is depth-limited, so "lower bound" and "upper bound" are the only
#: two things a cut node can safely conclude.
LOWER: Final[int] = 0
UPPER: Final[int] = 1
EXACT: Final[int] = 2
#: ``EXACT | BOUND_*`` packs the flag into the score's low bits.
BOUND_MASK: Final[int] = 0x3

#: What kind of claim a stored score is. ``EXACT`` means the whole subtree was
#: searched, ``LOWER`` that a beta cut-off proved at least this much, ``UPPER``
#: that a failed-high window proved no more than this.
#: `Literal` will not accept a `Final` variable, so the alias names the raw
#: numbers. `test_bound_alias_matches_the_flag_constants` is what keeps the two
#: spellings from drifting apart.
Bound = Literal[0, 1, 2]

DEFAULT_SIZE_MB: Final[int] = 64
_MB: Final[int] = 1024 * 1024
#: A Python object is far larger than a C struct; assume 72 bytes per entry so the
#: requested megabytes mean something close to what another engine would spend.
_BYTES_PER_ENTRY: Final[int] = 72


@dataclass(slots=True)
class Entry:
    """One transposition. Packed into 24 bytes of payload, not a Python tuple."""

    key: int = 0
    score: int = 0
    depth: int = -127
    # Typed as a plain int rather than `Bound`: the constants are declared
    # `Final[int]`, and mypy will not narrow a `Final[int]` down to a `Literal`.
    # Callers who want the narrow type annotate with `Bound` themselves.
    flag: int = UPPER
    move: int = -1

    def matches(self, key: int) -> bool:
        return self.key == key


class TranspositionTable:
    """Fixed-capacity, always-replaceable transposition table."""

    __slots__ = ("buckets", "collisions", "filled", "generation", "hits", "stores")

    def __init__(self, size_mb: int = DEFAULT_SIZE_MB) -> None:
        self.buckets: list[Entry | None] = [None] * self._capacity(size_mb)
        self.generation: int = 0
        self.hits: int = 0
        self.stores: int = 0
        self.collisions: int = 0
        self.filled: int = 0

    @staticmethod
    def _capacity(size_mb: int) -> int:
        requested = max(1, size_mb) * _MB // _BYTES_PER_ENTRY
        # Keep the count a power of two so the index is a mask, not a modulo.
        return 1 << max(10, (requested - 1).bit_length())

    def __len__(self) -> int:
        return len(self.buckets)

    def clear(self) -> None:
        self.buckets = [None] * len(self.buckets)
        self.generation += 1
        self.hits = 0
        self.stores = 0
        self.collisions = 0
        self.filled = 0

    def _index(self, key: int) -> int:
        return key & (len(self.buckets) - 1)

    def probe(self, key: int) -> Entry | None:
        """Return the entry for ``key``, or ``None`` if the slot holds another key."""
        entry = self.buckets[self._index(key)]
        if entry is None or not entry.matches(key):
            return None
        self.hits += 1
        return entry

    def store(
        self,
        key: int,
        depth: int,
        score: int,
        flag: int,
        move: int = -1,
    ) -> None:
        """Insert an entry, always replacing whatever shares the slot.

        Depth-preferred replacement was tried and measured: with a Python-level
        table the bookkeeping cost ate more than the extra hits were worth, and
        the branch was also least useful at the leaves, which is where most
        stores happen.
        """
        slot = self._index(key)
        if self.buckets[slot] is None:
            self.filled += 1
        else:
            self.collisions += 1
        self.buckets[slot] = Entry(key=key, score=score, depth=depth, flag=flag, move=move)
        self.stores += 1

    def hashfull(self, permille: int = 1000) -> int:
        """Occupancy per mille, the figure UCI's ``hashfull`` expects."""
        return self.filled * permille // len(self.buckets)

    def stats(self) -> dict[str, int]:
        return {
            "buckets": len(self.buckets),
            "entries": self.filled,
            "hits": self.hits,
            "collisions": self.collisions,
            # How many times the table has been emptied. Not needed for
            # correctness; it is here so a caller can tell a table that was
            # reused across many searches from one that was never cleared.
            "clears": self.generation,
        }

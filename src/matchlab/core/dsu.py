"""Functions to transform data between tabular and graph structures."""

from collections import defaultdict
from collections.abc import Hashable
from typing import Generic, TypeVar

T = TypeVar("T", bound=Hashable)


class DisjointSet(Generic[T]):
    """Disjoint set forest with "path compression" and "union by rank" heuristics.

    This follows implementation from Cormen, Thomas H., et al. Introduction to
    algorithms. MIT press, 2022
    """

    def __init__(self) -> None:
        """Initialise the disjoint set."""
        self.parent: dict[T, T] = {}
        self.rank: dict[T, int] = {}

    def _make_set(self, x: T) -> None:
        """Create a new set with a single element x."""
        self.parent[x] = x
        self.rank[x] = 0

    def add(self, x: T) -> None:
        """Add a new element to the disjoint set."""
        if x not in self.parent:
            self._make_set(x)

    def union(self, x: T, y: T) -> None:
        """Merge the sets containing elements x and y."""
        self._link(self._find(x), self._find(y))

    def _link(self, x: T, y: T) -> None:
        """Merge the sets containing elements x and y."""
        if self.rank[x] > self.rank[y]:
            self.parent[y] = x
        else:
            self.parent[x] = y
            if self.rank[x] == self.rank[y]:
                self.rank[y] += 1

    def _find(self, x: T) -> T:
        """Return the representative element of the set containing x."""
        if x not in self.parent:
            self._make_set(x)
            return x

        if x != self.parent[x]:
            self.parent[x] = self._find(self.parent[x])

        return self.parent[x]

    def get_components(self) -> list[set[T]]:
        """Return the connected components of the disjoint set."""
        components = defaultdict(set)
        for x in self.parent:
            root = self._find(x)
            components[root].add(x)
        return list(components.values())

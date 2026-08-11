"""The disjoint-set forest a resolver's connected components are built on.

Union-find is small. A handful of behaviours pin it down completely: an empty forest,
`union` closing transitively, an idempotent singleton `add`, and the two composing
together. Every finer case (union after add, redundant union, and so on) follows from
those, so it would only cost maintenance without being able to fail on its own.
"""

from collections.abc import Hashable

import pytest

from matchlab.core.dsu import DisjointSet


def _components(dsu: DisjointSet) -> set[frozenset[Hashable]]:
    """The components as an order-independent set, so comparisons don't mind order."""
    return {frozenset(group) for group in dsu.get_components()}


def test_empty_forest() -> None:
    """A fresh forest reports no components."""
    assert DisjointSet().get_components() == []


@pytest.mark.parametrize(
    ("unions", "expected"),
    [
        pytest.param([(1, 1)], {frozenset({1})}, id="self-union-is-a-singleton"),
        pytest.param(
            [(1, 2), (2, 1)], {frozenset({1, 2})}, id="redundant-union-is-idempotent"
        ),
        pytest.param(
            [(1, 2), (3, 4), (5, 6)],
            {frozenset({1, 2}), frozenset({3, 4}), frozenset({5, 6})},
            id="disjoint-pairs-stay-apart",
        ),
        pytest.param(
            [(1, 2), (3, 4), (5, 6), (2, 3), (4, 5)],
            {frozenset({1, 2, 3, 4, 5, 6})},
            id="a-chain-of-unions-closes-transitively",
        ),
    ],
)
def test_union(unions: list[tuple[int, int]], expected: set[frozenset[int]]) -> None:
    """Union-find exists to guarantee that unioned elements share a component."""
    dsu: DisjointSet[int] = DisjointSet()
    for left, right in unions:
        dsu.union(left, right)
    assert _components(dsu) == expected


@pytest.mark.parametrize(
    ("adds", "expected"),
    [
        pytest.param([1], {frozenset({1})}, id="one"),
        pytest.param([1, 1], {frozenset({1})}, id="the-same-one-twice"),
        pytest.param(
            [1, 2, 3],
            {frozenset({1}), frozenset({2}), frozenset({3})},
            id="several-distinct",
        ),
    ],
)
def test_add(adds: list[int], expected: set[frozenset[int]]) -> None:
    """A new element is its own component. Adding it again changes nothing."""
    dsu: DisjointSet[int] = DisjointSet()
    for element in adds:
        dsu.add(element)
    assert _components(dsu) == expected


def test_add_and_union() -> None:
    """An element no `union` touched keeps its own component beside the merged ones."""
    dsu: DisjointSet[int] = DisjointSet()
    dsu.add(1)
    dsu.add(2)
    dsu.union(3, 4)
    dsu.add(5)

    assert _components(dsu) == {
        frozenset({1}),
        frozenset({2}),
        frozenset({3, 4}),
        frozenset({5}),
    }

"""The vocabulary: what a planted answer is, and what a claimed answer is.

Two types exist, and the difference between them matters:

* `TrueEntity` is the *answer*: one planted real-world thing, identified by the values
  it was generated from, spanning every source it landed in.
* `Cluster` is *an answer*: a set of records claimed to be one entity, identified by
  that membership alone.

`TrueEntity.cluster()` projects the first onto the second, so the two become
comparable. Only membership can be compared this way. A resolver output's IDs are
minted at collect time and have no counterpart in generated data.
"""

from random import getrandbits
from typing import Any, Self

from frozendict import frozendict
from pydantic import BaseModel, ConfigDict, Field


class EntityReference(frozendict):
    """Reference to an entity's presence in specific sources.

    Maps source names to sets of primary keys.
    """

    def __init__(
        self,
        mapping: dict[str, frozenset[str]] | None = None,
    ) -> None:
        """Initialise the EntityReference."""
        super().__init__({} if mapping is None else mapping)

    def __add__(self, other: "EntityReference") -> "EntityReference":
        """Merge two EntityReferences by unioning keys for each source."""
        if not isinstance(other, EntityReference):
            return NotImplemented

        return EntityReference(
            {
                k: self.get(k, frozenset()) | other.get(k, frozenset())
                for k in self.keys() | other.keys()
            }
        )

    def __le__(self, other: Self) -> bool:
        """Test if self is a subset of other."""
        if not isinstance(other, EntityReference):
            return NotImplemented

        return all(name in other and self[name] <= other[name] for name in self)


class EntityIDMixin:
    """Shared ID behaviour for entity classes.

    Provides integer conversion and comparison operators, so entities sort by `id`.
    """

    id: int

    def __int__(self) -> int:
        """Allow converting an entity to an integer by returning its ID."""
        return self.id

    def __lt__(self, other: Self | int) -> bool:
        """Compare based on ID for sorting operations."""
        if hasattr(other, "id"):
            return self.id < other.id
        if isinstance(other, int):
            return self.id < other
        return NotImplemented

    def __gt__(self, other: Self | int) -> bool:
        """Compare based on ID for sorting operations."""
        if hasattr(other, "id"):
            return self.id > other.id
        if isinstance(other, int):
            return self.id > other
        return NotImplemented

    def __le__(self, other: Self | int) -> bool:
        """Compare based on ID for sorting operations."""
        if hasattr(other, "id"):
            return self.id <= other.id
        if isinstance(other, int):
            return self.id <= other
        return NotImplemented

    def __ge__(self, other: Self | int) -> bool:
        """Compare based on ID for sorting operations."""
        if hasattr(other, "id"):
            return self.id >= other.id
        if isinstance(other, int):
            return self.id >= other
        return NotImplemented


class SourceKeyMixin:
    """Shared source-key behaviour for entity classes.

    Provides `get_keys()`, for reading back the keys held for one source.
    """

    keys: EntityReference

    def get_keys(self, name: str) -> set[str]:
        """Get keys for a specific source.

        Args:
            name: Name of the source.

        Returns:
            Set of keys, empty if the source is not found.
        """
        return set(self.keys.get(name, frozenset()))


class Cluster(BaseModel, EntityIDMixin, SourceKeyMixin):
    """A set of records claimed to be one entity, an *answer*.

    This is the unit of comparison. Both sides of `diff_entities` are clusters: the
    expected side projected from `TrueEntity.cluster()`, and the actual side read back
    off a model or a resolver's output. It carries membership and nothing else, because
    membership is the only thing the two sides can agree on. A resolver output's IDs
    are minted at collect time and have no counterpart in generated data.

    Equality and hashing are over `keys` alone. `id` is carried for bookkeeping and is
    deliberately ignored when comparing. Contrast `TrueEntity`, which is the answer key
    rather than an answer.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    id: int = Field(default_factory=lambda: getrandbits(63))  # 64 gives OverflowError
    keys: EntityReference

    def __add__(self, other: "Cluster") -> "Cluster":
        """Combine two Cluster objects by combining the keys."""
        if other is None:
            return self
        if not isinstance(other, Cluster):
            return NotImplemented
        return Cluster(keys=self.keys + other.keys)

    def __radd__(self, other: object) -> "Cluster":
        """Handle sum() by treating 0 as an empty Cluster."""
        if other == 0:  # sum() starts with 0
            return self
        return NotImplemented

    def __sub__(self, other: "Cluster") -> dict[str, frozenset[str]]:
        """Return keys in self that aren't in other, by source.

        Used to diff two Cluster objects.
        """
        if not isinstance(other, Cluster):
            return NotImplemented

        diff = {}
        for name, our_keys in self.keys.items():
            their_keys = other.keys.get(name, frozenset())
            if remaining := our_keys - their_keys:
                diff[name] = remaining

        return diff

    def __rsub__(self, other: "Cluster") -> dict[str, frozenset[str]]:
        """Support reverse subtraction."""
        if not isinstance(other, Cluster):
            return NotImplemented
        return other - self

    def __eq__(self, other: object) -> bool:
        """Compare based on keys."""
        if not isinstance(other, Cluster):
            return NotImplemented
        return self.keys == other.keys

    def __contains__(self, other: "Cluster") -> bool:
        """Check if this entity contains all keys from other entity."""
        return other.keys <= self.keys

    def __hash__(self) -> int:
        """Hash based on EntityReference which is itself hashable."""
        return hash(self.keys)

    def is_subset_of_true_entity(self, source_entity: "TrueEntity") -> bool:
        """Check if this Cluster's references are a subset of a TrueEntity's."""
        return self.keys <= source_entity.keys

    def similarity_ratio(self, other: "Cluster") -> float:
        """Return ratio of shared keys to total keys across all sources."""
        total_keys = 0
        shared_keys = 0

        # Get all source names
        all_sources = set(self.keys.keys()) | set(other.keys.keys())

        for name in all_sources:
            our_keys = self.keys.get(name, frozenset())
            their_keys = other.keys.get(name, frozenset())

            total_keys += len(our_keys | their_keys)
            shared_keys += len(our_keys & their_keys)

        return shared_keys / total_keys if total_keys > 0 else 0.0


class TrueEntity(BaseModel, EntityIDMixin, SourceKeyMixin):
    """One planted real-world thing, the *answer key*.

    This is what the generator started from. It holds `base_values`, the feature values
    its rows were derived from, and accumulates the keys it landed under in every source
    it appears in. Equality is over `base_values`. Two entities are the same thing if
    they were generated from the same values.

    It spans every source, so it is not directly comparable with a result. Project it
    onto the sources under test with `cluster()` to get something that is.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int = Field(default_factory=lambda: getrandbits(63))  # 64 gives OverflowError
    base_values: dict[str, Any] = Field(description="Feature name -> base value")
    keys: EntityReference = Field(
        description="Source to keys mapping",
        default=EntityReference(mapping=frozenset()),
    )
    total_unique_variations: int = Field(default=0)

    def __eq__(self, other: object) -> bool:
        """Equal if base values are shared, or integer ID matches."""
        if isinstance(other, TrueEntity):
            return self.base_values == other.base_values
        if isinstance(other, int):
            return self.id == other
        return NotImplemented

    def __hash__(self) -> int:
        """Hash based on sorted base values."""
        return hash(tuple(sorted(self.base_values.items())))

    def add_source_reference(self, name: str, keys: list[str]) -> None:
        """Add or update a source reference.

        Args:
            name: Source name.
            keys: List of primary keys for this source.
        """
        mapping = dict(self.keys)
        mapping[name] = frozenset(keys)
        self.keys = EntityReference(mapping)

    def cluster(self, *names: str) -> Cluster | None:
        """Project this true entity onto the given sources, making it comparable.

        Comparing equality of `Cluster` sets is a simpler, more reliable test than
        checking whether `Cluster` objects are subsets of `TrueEntity` objects. This
        method is what makes that comparison possible:

        ```python
        actual: set[Cluster] = ...
        expected: set[Cluster] = {
            s.cluster("source1", "source2") for s in true_entities
        }

        is_identical = expected == actual
        missing = expected - actual
        extra = actual - expected
        ```

        Args:
            *names: Names of sources to include in the Cluster.

        Returns:
            A `Cluster` with only the named sources' keys, or `None` if this entity has
            no keys in any of them.
        """
        filtered = {
            name: self.keys.get(name)
            for name in names
            if self.keys.get(name) is not None
        }

        if len(filtered) == 0:
            return None

        return Cluster(keys=EntityReference(filtered))

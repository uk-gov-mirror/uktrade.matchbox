"""Factory functions for generating model testkits and data for testing."""

import warnings
from collections import Counter
from collections.abc import Hashable, Iterable
from functools import cache
from itertools import combinations
from textwrap import dedent
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import polars as pl
import pyarrow as pa
from faker import Faker
from pyarrow import compute as pc
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import create_engine

from matchlab.core.arrow import SCHEMA_MODEL_EDGES
from matchlab.core.config import (
    ModelType,
)
from matchlab.core.dsu import DisjointSet
from matchlab.core.factories.entities import (
    ClusterEntity,
    FeatureConfig,
    SourceEntity,
    SuffixRule,
    scores_to_results_entities,
)
from matchlab.core.factories.sources import (
    SourceTestkit,
    SourceTestkitParameters,
    linked_sources_factory,
)
from matchlab.core.hash import HASH_FUNC
from matchlab.models import add_model_class
from matchlab.models.dedupers.base import Deduper, DeduperSettings
from matchlab.models.linkers.base import Linker, LinkerSettings
from matchlab.models.models import Model
from matchlab.views import View

if TYPE_CHECKING:
    from matchlab.core.factories.resolvers import ResolverTestkit
    from matchlab.resolvers import Resolver
else:
    Resolver = Any
    ResolverTestkit = Any

T = TypeVar("T", bound=Hashable)


# Scripted methodologies — ground truth expressed over *values*, not IDs.
#
# A scripted model is a perfect matcher: it knows which records belong to the same
# true entity and emits exactly those edges. The subtlety is *how* it identifies a
# record.
#
# The obvious approach — pre-generate edges between record IDs — does not work.
# Matchlab derives cluster IDs by content-hashing rows at collect time, so the IDs a
# model actually receives are not known when the testkit builds its fixtures. (Under
# the old server the testkit both generated and uploaded the ID column, so it
# controlled both sides; that assumption died with the server.)
#
# So the truth is keyed by the row's *values*. At match time the model reads those
# columns out of the frame it was handed, looks up each row's true entity, and emits
# edges between whatever IDs are actually present. That makes the testkit independent
# of how identity is assigned, so it survives any future change to ID minting.
_TRUTH: dict[str, "ValueTruth"] = {}


class ValueTruth(BaseModel):
    """Ground truth mapping row values to the true entity they belong to.

    `groups` maps a tuple of column values to a true-entity ID. `columns` names the
    columns to read, in the same order, as they appear in the frame the model is given
    (i.e. source-qualified). Linkers carry a second set for the right-hand frame; both
    sides map into the same entity-ID space, which is what lets them be joined.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    left_columns: tuple[str, ...]
    left_groups: dict[tuple, int]
    right_columns: tuple[str, ...] | None = None
    right_groups: dict[tuple, int] | None = None
    score: float = 1.0

    def _assign(
        self, data: pl.DataFrame, columns: tuple[str, ...], groups: dict[tuple, int]
    ) -> dict[int, list[int]]:
        """Bucket the frame's row IDs by the true entity their values belong to."""
        missing = [column for column in columns if column not in data.columns]
        if missing:
            raise KeyError(
                f"Ground-truth columns absent from the model input: {missing}. "
                f"Available: {sorted(data.columns)}"
            )

        buckets: dict[int, list[int]] = {}
        for row in data.select("id", *columns).iter_rows(named=True):
            entity = groups.get(tuple(row[column] for column in columns))
            if entity is not None:
                buckets.setdefault(entity, []).append(row["id"])
        return buckets

    def _edges(self, pairs: list[tuple[int, int]]) -> pl.DataFrame:
        if not pairs:
            return pl.from_arrow(SCHEMA_MODEL_EDGES.empty_table())
        left, right = zip(*pairs, strict=True)
        return pl.DataFrame(
            {
                "left_id": list(left),
                "right_id": list(right),
                "score": [self.score] * len(pairs),
            },
            schema=pl.Schema(SCHEMA_MODEL_EDGES),
        )

    def dedupe_edges(self, data: pl.DataFrame) -> pl.DataFrame:
        """All within-entity pairs among the frame's records."""
        pairs: list[tuple[int, int]] = []
        for ids in self._assign(data, self.left_columns, self.left_groups).values():
            unique = sorted(set(ids))
            pairs.extend(combinations(unique, 2))
        return self._edges(pairs)

    def link_edges(self, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
        """All cross-side pairs whose records share a true entity."""
        left_buckets = self._assign(left, self.left_columns, self.left_groups)
        right_buckets = self._assign(
            right, self.right_columns or (), self.right_groups or {}
        )
        pairs = [
            (left_id, right_id)
            for entity, left_ids in left_buckets.items()
            for left_id in sorted(set(left_ids))
            for right_id in sorted(set(right_buckets.get(entity, [])))
        ]
        return self._edges(pairs)


def register_truth(truth: ValueTruth) -> str:
    """Register ground truth, returning a content-addressed ID for it.

    Settings are JSON-serialised into the step fingerprint, so the truth itself cannot
    live there. Keying by content hash keeps the fingerprint honest: different truth
    produces a different ID, and therefore a different model artifact.
    """
    payload = repr(
        (
            truth.left_columns,
            sorted(map(repr, truth.left_groups.items())),
            truth.right_columns,
            sorted(map(repr, (truth.right_groups or {}).items())),
            truth.score,
        )
    ).encode()
    truth_id = HASH_FUNC(payload).hexdigest()
    _TRUTH[truth_id] = truth
    return truth_id


class ScriptedDeduperSettings(DeduperSettings):
    """Settings for a deduper that matches from registered ground truth."""

    truth_id: str


class ScriptedDeduper(Deduper):
    """A perfect deduper: emits every within-entity pair it is given."""

    settings: ScriptedDeduperSettings

    def prepare(self, data: pl.DataFrame) -> None:
        """No preparation required."""

    def dedupe(self, data: pl.DataFrame) -> pl.DataFrame:
        """Emit edges between records sharing a true entity."""
        return _TRUTH[self.settings.truth_id].dedupe_edges(data)


class ScriptedLinkerSettings(LinkerSettings):
    """Settings for a linker that matches from registered ground truth."""

    truth_id: str


class ScriptedLinker(Linker):
    """A perfect linker: emits every cross-side pair sharing a true entity."""

    settings: ScriptedLinkerSettings

    def prepare(self, left: pl.DataFrame, right: pl.DataFrame) -> None:
        """No preparation required."""

    def link(self, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
        """Emit edges between records sharing a true entity."""
        return _TRUTH[self.settings.truth_id].link_edges(left, right)


def truth_from_testkits(
    left_testkit: SourceTestkit,
    source_entities: Iterable[SourceEntity],
    right_testkit: SourceTestkit | None = None,
    score: float = 1.0,
) -> ValueTruth:
    """Derive ground truth over row values from generated source testkits.

    Each generated row is mapped to the true entity that owns its source key, then
    keyed by its feature values under the names the model will see them by
    (source-qualified).
    """
    entities = tuple(source_entities)

    def side(testkit: SourceTestkit) -> tuple[tuple[str, ...], dict[tuple, int]]:
        key_to_entity: dict[str, int] = {
            str(key): entity.id
            for entity in entities
            for key in entity.get_keys(testkit.name)
        }
        # From the testkit, not the source: the source would have to read the
        # warehouse to list its columns, and this runs before any collect.
        features = [feature.name for feature in testkit.features]
        groups: dict[tuple, int] = {}
        for row in pl.from_arrow(testkit.data).iter_rows(named=True):
            entity_id = key_to_entity.get(str(row["key"]))
            if entity_id is not None:
                groups[tuple(row[feature] for feature in features)] = entity_id
        columns = tuple(f"{testkit.name}_{feature}" for feature in features)
        return columns, groups

    left_columns, left_groups = side(left_testkit)
    right_columns, right_groups = (
        side(right_testkit) if right_testkit is not None else (None, None)
    )
    return ValueTruth(
        left_columns=left_columns,
        left_groups=left_groups,
        right_columns=right_columns,
        right_groups=right_groups,
        score=score,
    )


add_model_class(ScriptedDeduper)
add_model_class(ScriptedLinker)


def component_report(all_nodes: list[int], table: pl.DataFrame) -> dict:
    """Fast reporting on connected components.

    Args:
        all_nodes: list of identities of inputs being matched
        table: Polars dataframe matching SCHEMA_MODEL_EDGES

    Returns:
        dictionary containing basic component statistics
    """
    ds = DisjointSet[int]()
    for node in all_nodes:
        ds.add(node)
    for left_id, right_id in table.select(["left_id", "right_id"]).rows():
        ds.union(left_id, right_id)

    components = ds.get_components()
    component_sizes = Counter(len(component) for component in components)
    total_nodes = sum(component_sizes.values())

    return {
        "num_components": len(components),
        "total_nodes": total_nodes,
        "total_edges": len(table),
        "component_sizes": component_sizes,
        "min_component_size": min(component_sizes.keys()) if component_sizes else 0,
        "max_component_size": max(component_sizes.keys()) if component_sizes else 0,
    }


def validate_components(
    edges: list[tuple[int, int]],
    entities: set[ClusterEntity],
    source_entities: set[SourceEntity],
) -> bool:
    """Validate that score edges create valid components.

    Each component should be a subset of exactly one source entity.

    Args:
        edges: List of (left_id, right_id) edges
        entities: Set of ClusterEntity objects
        source_entities: Set of SourceEntities
    """
    # Create disjoint set of ClusterEntity objects
    ds = DisjointSet[ClusterEntity]()

    # Map IDs to ClusterEntity objects for lookup
    id_to_entity = {entity.id: entity for entity in entities}

    # Union entities based on score edges
    for left_id, right_id in edges:
        left = id_to_entity[left_id]
        right = id_to_entity[right_id]
        ds.union(left, right)

    # Get components
    components = ds.get_components()

    # For each component, check if it's a subset of a single source entity
    for component in components:
        if len(component) <= 1:
            continue

        # Merge all ClusterEntity objects in component
        merged = None
        for entity in component:
            if merged is None:
                merged = entity
            else:
                merged += entity

        # Find which source entity this component belongs to
        found_source = None
        for source in source_entities:
            if merged.is_subset_of_source_entity(source):
                if found_source is not None:
                    # Component matches multiple source entities - invalid!
                    return False
                found_source = source

        if found_source is None:
            # Component doesn't match any source entity
            return False

    return True


def _min_edges_component(left: int, right: int, deduplicate: bool) -> int:
    """Calculate min edges for component to be connected.

    Does so by assuming a spanning tree.

    Args:
        left: number of nodes of component on the left
        right: number of nodes of component on the right (for linking)
        deduplicate: whether edges are for deduplication

    Returns:
        Minimum number of edges
    """
    if not deduplicate:
        return left + right - 1

    return left - 1


def _max_edges_component(left: int, right: int, deduplicate: bool) -> int:
    """Calculate max edges for component to be avoid duplication.

    Considers complete graph for deduping, and complete bipartite graph for linking.

    Args:
        left: number of nodes of component on the left
        right: number of nodes of component on the right (for linking)
        deduplicate: whether edges are for deduplication

    Returns:
        Maximum number of edges
    """
    if not deduplicate:
        return left * right
    # n*(n-1) is always divisible by 2
    return left * (left - 1) // 2


def calculate_min_max_edges(
    left_nodes: int, right_nodes: int, num_components: int, deduplicate: bool
) -> tuple[int, int]:
    """Calculate min and max edges for a graph.

    Args:
        left_nodes: number of nodes in left source
        right_nodes: number of nodes in right source
        num_components: number of requested components
        deduplicate: whether edges are for deduplication

    Returns:
        Two-tuple representing min and max edges
    """
    left_mod, right_mod = left_nodes % num_components, right_nodes % num_components
    left_div, right_div = left_nodes // num_components, right_nodes // num_components

    min_mod, max_mod = sorted([left_mod, right_mod])

    min_edges, max_edges = 0, 0
    # Components where both sides have maximum nodes
    min_edges += (
        _min_edges_component(left_div + 1, right_div + 1, deduplicate) * min_mod
    )
    max_edges += (
        _max_edges_component(left_div + 1, right_div + 1, deduplicate) * min_mod
    )
    # Components where one side has maximum nodes
    left_after_min_mod, right_after_min_mod = left_div + 1, right_div
    if left_mod == min_mod:
        left_after_min_mod, right_after_min_mod = left_div, right_div + 1
    min_edges += _min_edges_component(
        left_after_min_mod, right_after_min_mod, deduplicate
    ) * (max_mod - min_mod)
    max_edges += _max_edges_component(
        left_after_min_mod, right_after_min_mod, deduplicate
    ) * (max_mod - min_mod)
    # Components where both side have minimum nodes
    min_edges += _min_edges_component(left_div, right_div, deduplicate) * (
        num_components - max_mod
    )
    max_edges += _max_edges_component(left_div, right_div, deduplicate) * (
        num_components - max_mod
    )

    return min_edges, max_edges


@cache
def generate_dummy_scores(
    left_values: tuple[int],
    right_values: tuple[int] | None,
    score_range: tuple[float, float],
    num_components: int,
    total_rows: int | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate dummy Arrow scores data with guaranteed isolated components.

    While much of the factory system uses generate_entity_scores, this function
    is still in use in PostgreSQL benchmarking, and has been designed to be performant
    at scale.

    Args:
        left_values: Tuple of integers to use for left column
        right_values: Tuple of integers to use for right column. If None, assume we
            are generating scores for deduplication
        score_range: Tuple of (min_score, max_score) to constrain scores
        num_components: Number of distinct connected components to generate
        total_rows: Total number of rows to generate
        seed: Random seed for reproducibility

    Returns:
        Polars dataframe with 'left_id', 'right_id', and 'score' columns
    """
    # Validate inputs
    deduplicate = False
    if right_values is None:
        left_values = tuple(set(left_values))  # Remove duplicates
        right_values = left_values
        deduplicate = True

    if len(left_values) < 2 or len(right_values) < 2:
        raise ValueError("Need at least 2 possible values for both left and right")
    if num_components > min(len(left_values), len(right_values)):
        raise ValueError(
            "Cannot have more components than minimum of left/right values"
        )

    left_nodes, right_nodes = len(left_values), len(right_values)
    min_possible_edges, max_possible_edges = calculate_min_max_edges(
        left_nodes, right_nodes, num_components, deduplicate
    )

    mode = "dedupe" if deduplicate else "link"

    if total_rows is None:
        total_rows = min_possible_edges

    if total_rows == 0:
        raise ValueError("At least one edge must be generated")
    elif total_rows < min_possible_edges:
        raise ValueError(
            dedent(f"""
            Cannot generate {total_rows:,} {mode} edges with {num_components:,}
            components.
            Min edges is {min_possible_edges:,} for nodes given.
            Either decrease the number of nodes, increase the number of components, 
            or increase the total edges requested.
            """)
        )
    elif total_rows > max_possible_edges:
        raise ValueError(
            dedent(f"""
            Cannot generate {total_rows:,} {mode} edges with {num_components:,}
            components. 
            Max edges is {max_possible_edges:,} for nodes given.
            Either increase the number of nodes, decrease the number of components, 
            or decrease the total edges requested.
            """)
        )

    n_extra_edges = total_rows - min_possible_edges

    # Create seeded random number generator
    rng = np.random.default_rng(seed=seed)

    # Split values into completely separate groups for each component
    left_components = np.array_split(np.array(left_values), num_components)
    right_components = np.array_split(np.array(right_values), num_components)
    # For each left-right component pair, the right equals the left rotated by one
    right_components = [np.roll(c, -1) for c in right_components]

    all_edges = []

    # Generate edges for each component
    for comp_idx in range(num_components):
        comp_left_values = left_components[comp_idx]
        comp_right_values = right_components[comp_idx]

        min_comp_nodes, max_comp_nodes = sorted(
            [len(comp_left_values), len(comp_right_values)]
        )

        # Ensure basic connectivity within the component by creating a spanning-tree
        base_edges = set()
        # For deduping (A B C) you just need (A - B) (B - C) (C - A)
        # which just needs matching pairwise the data and its rotated version.
        # For deduping, `min_comp_nodes` == `max_comp_nodes`
        if deduplicate:
            for i in range(min_comp_nodes - 1):
                small_n, large_n = sorted([comp_left_values[i], comp_right_values[i]])
                base_edges.add((small_n, large_n))
        else:
            # For linking (A B) and (C D E), we begin by adding (A - C) and (B - D)
            for i in range(min_comp_nodes):
                base_edges.add((comp_left_values[i], comp_right_values[i]))
            # We now add (C - B)
            for i in range(min_comp_nodes - 1):
                base_edges.add((comp_left_values[i + 1], comp_right_values[i]))
            # We now add (A - D)
            left_right_diff = max_comp_nodes - min_comp_nodes
            for i in range(left_right_diff):
                left_i, right_i = 0, min_comp_nodes + i
                if len(comp_right_values) < len(comp_left_values):
                    left_i, right_i = min_comp_nodes + i, 0

                base_edges.add((comp_left_values[left_i], comp_right_values[right_i]))

        component_edges = list(base_edges)

        if n_extra_edges > 0:
            # Generate remaining random edges strictly within this component
            # Note: this can certainly be optimised, but doesn't matter much for now
            if deduplicate:
                all_possible_edges = list(
                    {
                        tuple(sorted([x, y]))
                        for x in comp_left_values
                        for y in comp_right_values
                        if x != y and tuple(sorted([x, y])) not in base_edges
                    }
                )
            else:
                all_possible_edges = list(
                    {
                        (x, y)
                        for x in comp_left_values
                        for y in comp_right_values
                        if x != y and (x, y) not in base_edges
                    }
                )
            max_new_edges = len(all_possible_edges)
            if max_new_edges >= n_extra_edges:
                edges_required = n_extra_edges
                n_extra_edges = 0
            else:
                edges_required = max_new_edges
                n_extra_edges -= max_new_edges

            extra_edges_idx = rng.choice(
                len(all_possible_edges), size=edges_required, replace=False
            )
            extra_edges = [
                e for i, e in enumerate(all_possible_edges) if i in extra_edges_idx
            ]
            component_edges += extra_edges
        random_scores = rng.uniform(
            score_range[0], score_range[1], size=len(component_edges)
        ).astype(np.float32)

        component_edges = [
            (le, ri, pr)
            for (le, ri), pr in zip(component_edges, random_scores, strict=True)
        ]

        all_edges.extend(component_edges)

    # Convert to arrays
    lefts, rights, score_values = zip(*all_edges, strict=True)

    return pl.DataFrame(
        {"left_id": lefts, "right_id": rights, "score": score_values},
        schema={
            "left_id": pl.UInt64,
            "right_id": pl.UInt64,
            "score": pl.Float32,
        },
    )


def generate_entity_scores(
    left_entities: frozenset[ClusterEntity],
    right_entities: frozenset[ClusterEntity] | None,
    source_entities: frozenset[SourceEntity],
    score_range: tuple[float, float] = (0.8, 1.0),
    seed: int = 42,
) -> pl.DataFrame:
    """Generate scores that will recover entity relationships.

    Compares ClusterEntity objects against ground truth SourceEntities by checking
    whether their EntityReferences are subsets of the source entities. Initially
    focused on generating fully connected, correct scores only.

    Args:
        left_entities: Set of ClusterEntity objects from left input
        right_entities: Set of ClusterEntity objects from right input. If None, assume
            we are deduplicating left_entities.
        source_entities: Ground truth set of SourceEntities
        score_range: Range of scores to assign to matches. All matches will
            be assigned a random score in this range.
        seed: Random seed for reproducibility

    Returns:
        PyArrow Table with 'left_id', 'right_id', and 'score' columns
    """
    # Validate inputs
    if not (0 <= score_range[0] <= score_range[1] <= 1):
        raise ValueError("Scores must be increasing values between 0 and 1")

    # Handle deduplication case
    if right_entities is None:
        right_entities = left_entities

    # Create mapping of ClusterEntity -> SourceEntity
    entity_mapping: dict[ClusterEntity, SourceEntity] = {}

    def _map_entity(entity: ClusterEntity) -> None:
        matching_sources = [
            source
            for source in source_entities
            if entity.is_subset_of_source_entity(source)
        ]
        if len(matching_sources) > 1:
            raise ValueError(
                f"ClusterEntity with ID {entity.id} is a subset of multiple "
                f"SourceEntities. This violates the uniqueness constraint."
            )
        if matching_sources:
            entity_mapping[entity] = matching_sources[0]

    # Map all entities, checking constraints
    for entity in left_entities:
        _map_entity(entity)
    if right_entities is not left_entities:
        for entity in right_entities:
            _map_entity(entity)

    # Group by SourceEntity
    source_groups: dict[
        SourceEntity, tuple[set[ClusterEntity], set[ClusterEntity]]
    ] = {}
    for entity, source in entity_mapping.items():
        if source not in source_groups:
            source_groups[source] = (set(), set())
        # Add to left or right group based on which input set it came from
        if entity in left_entities:
            source_groups[source][0].add(entity)
        if entity in right_entities:  # Note: could be in both for deduplication
            source_groups[source][1].add(entity)

    # Generate score edges for each group
    edges = []
    rng = np.random.default_rng(seed=seed)

    for left_group, right_group in source_groups.values():
        # Skip empty groups
        if not left_group or not right_group:
            continue

        # Generate all pairs within this group
        for left_entity in left_group:
            for right_entity in right_group:
                # For deduplication, only include each pair once
                # and ensure left_id < right_id
                if (
                    right_entities == left_entities
                    and left_entity.id >= right_entity.id
                ):
                    continue

                # Generate random score in range
                score = np.float32(rng.uniform(score_range[0], score_range[1]))
                edges.append((left_entity.id, right_entity.id, score))

    # If no edges were generated, return empty table with correct schema
    if not edges:
        return pl.DataFrame(schema=pl.Schema(SCHEMA_MODEL_EDGES))

    return pl.DataFrame(edges, orient="row", schema=pl.Schema(SCHEMA_MODEL_EDGES))


class ModelTestkit(BaseModel):
    """A testkit of data and metadata for a Model."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    model: Model
    left_data: pa.Table
    left: View
    left_clusters: dict[int, ClusterEntity]
    right_data: pa.Table | None
    right: View | None
    right_clusters: dict[int, ClusterEntity] | None
    scores: pl.DataFrame = Field(frozen=True)

    _entities: tuple[ClusterEntity, ...]
    _query_lookup: pa.Table

    @property
    def name(self) -> str:
        """Return the full name of the Model."""
        return self.model.name

    @property
    def data(self) -> pa.Table:
        """Return a PyArrow table in the same format as matchlab queries."""
        if self.model.model_type == ModelType.DEDUPER:
            data = self.left_data
        else:
            data = pa.concat_tables(
                [self.left_data, self.right_data], promote_options="default"
            )

        indices = pc.index_in(data["id"], self._query_lookup["id"])
        indices = indices.combine_chunks()
        replacements = pc.take(self._query_lookup["new_id"], indices).combine_chunks()

        new_ids = pc.replace_with_mask(
            data["id"],
            pc.is_valid(indices),
            replacements,
        )

        id_index = data.schema.get_field_index("id")
        return data.set_column(id_index, "id", new_ids)

    @property
    def entities(self) -> tuple[ClusterEntity, ...]:
        """ClusterEntities that were generated by the model."""
        return self._entities

    @model_validator(mode="after")
    def init_query_lookup(self) -> "ModelTestkit":
        """Initialise query lookup and derived entities using all model edges."""
        right_clusters = self.right_clusters.values() if self.right_clusters else []
        input_results = set(self.left_clusters.values()) | set(right_clusters)

        entities: tuple[ClusterEntity, ...] = scores_to_results_entities(
            scores=self.scores,
            left_clusters=tuple(self.left_clusters.values()),
            right_clusters=tuple(right_clusters)
            if self.right_clusters is not None
            else None,
            threshold=0,
        )

        id_mapping = {
            original.id: entity.id
            for original in input_results
            for entity in set(entities)
            if original in entity
        }

        ids, new_ids = zip(*id_mapping.items(), strict=True) if id_mapping else ([], [])

        self._query_lookup = pa.table(
            {
                "id": pa.array(ids, type=pa.int64()),
                "new_id": pa.array(new_ids, type=pa.int64()),
            }
        )
        self._entities = entities

        return self

    def resolve(self, *args: Any, **kwargs: Any) -> "Resolver":
        """Thin wrapper to resolve this testkit's Model into clusters."""
        return self.model.resolve(*args, **kwargs)


def model_factory(
    name: str | None = None,
    left_testkit: SourceTestkit | ResolverTestkit | None = None,
    right_testkit: SourceTestkit | ResolverTestkit | None = None,
    true_entities: tuple[SourceEntity, ...] | None = None,
    model_type: ModelType | None = None,
    n_true_entities: int | None = None,
    score_range: tuple[float, float] = (0.8, 1.0),
    seed: int = 42,
) -> ModelTestkit:
    """Generate a complete model testkit.

    Allows autoconfiguration with minimal settings, or more nuanced control.

    Can either be used to generate a model in a pipeline, interconnected with existing
    testkit objects, or generate a standalone model with random data.

    Args:
        name: Name of the model. Defaults to a randomly generated word suffixed
            with '_model'.
        left_testkit: A SourceTestkit or ResolverTestkit for the left source
        right_testkit: If creating a linker, a SourceTestkit or ResolverTestkit for the
            right source
        true_entities: Ground truth SourceEntity objects to use for
            generating scores. Must be supplied if sources are given
        model_type: Type of the model, one of 'deduper' or 'linker'
            Defaults to deduper. Ignored if left_testkit or right_testkit are provided.
        n_true_entities: Base number of entities to generate when using default configs.
            Defaults to 10. Ignored if left_testkit or right_testkit are provided.
        score_range: Range of scores to generate
        seed: Random seed for reproducibility

    Returns:
        ModelTestkit: A model testkit with generated data

    Raises:
        ValueError:
            * If scores are not in increasing order and between 0 and 1
            * If sources are provided without true entities
        UserWarning: If some arguments are ignored due to sources or true entities
    """
    # ==== Input validation ====
    if not (0 <= score_range[0] <= score_range[1] <= 1):
        raise ValueError("Scores must be increasing values between 0 and 1")

    if left_testkit is not None and true_entities is None:
        raise ValueError("Must provide true entities when sources are given")

    if any([left_testkit, true_entities]) and any(
        [model_type is not None, n_true_entities is not None]
    ):
        warnings.warn(
            "Some arguments will be ignored as sources or true entities are provided",
            UserWarning,
            stacklevel=2,
        )

    # ==== Setup ====
    generator = Faker()
    generator.seed_instance(seed)
    n_true_entities = n_true_entities or 10
    dummy_true_entities = None

    # ==== SourceConfig configuration ====
    if left_testkit is not None:
        # Using provided sources
        left_data = left_testkit.data
        left = left_testkit.view()
        left_entities = left_testkit.entities
        left_source_testkit = (
            left_testkit if isinstance(left_testkit, SourceTestkit) else None
        )

        right_source_testkit = None
        right_data = None
        right = None
        right_entities = None

        if right_testkit is not None:
            model_type = ModelType.LINKER
            right_data = right_testkit.data
            right = right_testkit.view()
            right_entities = right_testkit.entities
            right_source_testkit = (
                right_testkit if isinstance(right_testkit, SourceTestkit) else None
            )
        else:
            model_type = ModelType.DEDUPER
            right_entities = None
    else:
        # Create default sources
        engine = create_engine("sqlite:///:memory:")
        resolved_model_type = model_type or ModelType.DEDUPER

        # Define common features
        features = {
            "company_name": FeatureConfig(
                name="company_name", base_generator="company"
            ),
            "crn": FeatureConfig(
                name="crn",
                base_generator="bothify",
                parameters=(("text", "???-###-???-###"),),
            ),
            "cdms": FeatureConfig(
                name="cdms",
                base_generator="numerify",
                parameters=(("text", "ORG-########"),),
            ),
        }

        # Configure left source
        left_parameters = SourceTestkitParameters(
            name="crn",
            engine=engine,
            features=(
                features["company_name"].add_variations(
                    SuffixRule(suffix=" Limited"),
                    SuffixRule(suffix=" UK"),
                    SuffixRule(suffix=" Company"),
                ),
                features["crn"],
            ),
            drop_base=True,
            repetition=0,
        )

        # Configure sources based on model type
        source_parameters = [left_parameters]
        if resolved_model_type == ModelType.LINKER:
            source_parameters.append(
                SourceTestkitParameters(
                    name="cdms",
                    features=(features["crn"], features["cdms"]),
                    repetition=1,
                )
            )

        # Generate linked sources
        linked = linked_sources_factory(
            source_parameters=tuple(source_parameters),
            n_true_entities=n_true_entities,
            seed=seed,
        )

        # Extract source data
        left_source_testkit = linked.sources["crn"]
        left_data = left_source_testkit.data
        left = left_source_testkit.source.view()
        left_entities = left_source_testkit.entities

        right_source_testkit = None
        right_data = None
        right = None
        right_entities = None
        if resolved_model_type == ModelType.LINKER:
            right_source_testkit = linked.sources["cdms"]
            right_data = right_source_testkit.data
            right = right_source_testkit.source.view()
            right_entities = right_source_testkit.entities

        dummy_true_entities = tuple(linked.true_entities)
        model_type = resolved_model_type

    # ==== Entity and score generation ====
    # We need to generate true entities when either:
    # * No true entities are provided (true_entities is None)
    # * We're using default sources (left_testkit is None)
    if true_entities is None or left_testkit is None:
        final_true_entities = generator.random_elements(
            elements=dummy_true_entities,
            unique=True,
            length=n_true_entities,
        )
    else:
        final_true_entities = true_entities

    scores = generate_entity_scores(
        left_entities=frozenset(left_entities),
        right_entities=frozenset(right_entities) if right_entities else None,
        source_entities=frozenset(final_true_entities),
        score_range=score_range,
        seed=seed,
    )

    # ==== Model creation ====
    # `scores` above expresses the expectation in the testkit's own entity space, which
    # is what `entities` / `diff_model_edges` assert against without running a plan.
    # The model itself matches on *values*, so that when the plan is collected it emits
    # the same truth over the content-derived IDs the frame actually carries.
    if left_source_testkit is None:
        raise ValueError(
            "model_factory needs SourceTestkit inputs to derive value-based ground "
            "truth. Pass source testkits, or build the model with "
            "clean_to_model_factory and an explicit ValueTruth."
        )

    truth_id = register_truth(
        truth_from_testkits(
            left_testkit=left_source_testkit,
            source_entities=final_true_entities,
            right_testkit=right_source_testkit,
            score=score_range[1],
        )
    )
    if model_type == ModelType.LINKER:
        model_class = ScriptedLinker
        model_settings = ScriptedLinkerSettings(truth_id=truth_id)
    else:
        model_class = ScriptedDeduper
        model_settings = ScriptedDeduperSettings(truth_id=truth_id)

    model = Model(
        name=name or f"{generator.unique.word()}_model",
        model_class=model_class,
        model_settings=model_settings,
        left=left,
        right=right,
    )

    return ModelTestkit(
        model=model,
        left_data=left_data,
        left=left,
        left_clusters={entity.id: entity for entity in left_entities},
        right_data=right_data,
        right=right,
        right_clusters={entity.id: entity for entity in right_entities}
        if right_entities
        else None,
        scores=scores,
    )

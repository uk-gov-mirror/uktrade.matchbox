"""The generated model, and the expectation it is asserted against.

`GeneratedModel.scores` and `.predicted_clusters` express the expected result in the
testkit's *own* entity ID space, which is what `LinkedSources.diff_model_edges` compares
a methodology against without running a plan. The model itself matches on row values
(see `matchers`), so a collected plan reproduces the same answer over the
content-derived IDs the record step actually carries.

Build one with `LinkedSources.dedupe()` or `.link()`. They know the answer, so nothing
has to be threaded through by hand.
"""

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from matchlab.core.schemas import SCHEMA_MODEL_EDGES
from matchlab.models.models import Model
from matchlab.testkit.compare import scores_to_clusters
from matchlab.testkit.entities import Cluster, TrueEntity
from matchlab.testkit.sources import GeneratedSource


def generate_entity_scores(
    left_entities: frozenset[Cluster],
    right_entities: frozenset[Cluster] | None,
    true_entities: frozenset[TrueEntity],
    score_range: tuple[float, float] = (0.8, 1.0),
    seed: int = 42,
) -> pl.DataFrame:
    """Generate scores that recover the planted entity relationships.

    Maps each `Cluster` to the `TrueEntity` it is a subset of, then emits a score for
    every same-entity pair. The result is a fully connected, perfectly correct score
    graph. This function does not generate partial, wrong, or missing matches.

    Args:
        left_entities: Cluster objects from the left input.
        right_entities: Cluster objects from the right input. `None` to deduplicate
            `left_entities` instead.
        true_entities: The planted true entities to score against.
        score_range: Assign each matching pair a random score in this range.
        seed: Random seed for reproducibility.

    Returns:
        A Polars DataFrame with `left_id`, `right_id`, and `score` columns.

    Raises:
        ValueError: If `score_range` is not increasing within [0, 1], or a `Cluster` is
            a subset of more than one true entity.
    """
    # Validate inputs
    if not (0 <= score_range[0] <= score_range[1] <= 1):
        raise ValueError("Scores must be increasing values between 0 and 1")

    # Handle deduplication case
    if right_entities is None:
        right_entities = left_entities

    # Create mapping of Cluster -> TrueEntity
    entity_mapping: dict[Cluster, TrueEntity] = {}

    def _map_entity(entity: Cluster) -> None:
        matching_sources = [
            source
            for source in true_entities
            if entity.is_subset_of_true_entity(source)
        ]
        if len(matching_sources) > 1:
            raise ValueError(
                f"Cluster with ID {entity.id} is a subset of multiple "
                f"true entities. This violates the uniqueness constraint."
            )
        if matching_sources:
            entity_mapping[entity] = matching_sources[0]

    # Map all entities, checking constraints
    for entity in left_entities:
        _map_entity(entity)
    if right_entities is not left_entities:
        for entity in right_entities:
            _map_entity(entity)

    # Group by TrueEntity
    source_groups: dict[TrueEntity, tuple[set[Cluster], set[Cluster]]] = {}
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
                # Deduplication counts each pair once, so keep left_id < right_id.
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


class GeneratedModel(BaseModel):
    """A generated model: the plan node, its inputs, and the answer expected of it.

    As with `GeneratedSource`, this exposes what it knows about the fixture and nothing
    else. To run the plan, go through `.model`, e.g. `model.model.resolve()`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    # The plan node. Run it with `model.resolve()`.
    model: Model
    # The generated source it reads as its left input. Its rows and starting partition
    # are reachable from here rather than copied onto this object.
    left_source: GeneratedSource
    # Its right input, for a linker. `None` for a deduper.
    right_source: GeneratedSource | None = None
    # The edges a perfect matcher should emit, in the sources' own entity ID space.
    scores: pl.DataFrame = Field(frozen=True)

    _predicted_clusters: tuple[Cluster, ...]

    @property
    def predicted_clusters(self) -> tuple[Cluster, ...]:
        """What the model's scores imply, to compare against the planted answer."""
        return self._predicted_clusters

    @model_validator(mode="after")
    def _derive_predicted_clusters(self) -> "GeneratedModel":
        """Resolve the expected scores into the clusters they imply."""
        self._predicted_clusters = scores_to_clusters(
            scores=self.scores,
            left_clusters=self.left_source.input_clusters,
            right_clusters=(
                self.right_source.input_clusters
                if self.right_source is not None
                else None
            ),
            threshold=0,
        )

        return self

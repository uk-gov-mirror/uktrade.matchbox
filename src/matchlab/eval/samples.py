"""Client-side helpers for retrieving and preparing evaluation samples.

Everything here takes a **resolution**: a `Resolver` you are holding, the label one was
published under, or a sequence of either. The sequence form is what makes two
methodologies comparable — sampling across several resolutions unions their components,
so one round of judging covers all of them and the scores are answering the same
question.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import polars as pl
from pydantic import BaseModel

from matchlab.core.dataframes import qualify
from matchlab.core.dsu import DisjointSet
from matchlab.core.exceptions import SourceTableError
from matchlab.core.resolution import root_id
from matchlab.eval.judgements import Judgement
from matchlab.eval.metrics import PrecisionRecall, precision_recall

if TYPE_CHECKING:
    from matchlab.adapters import Adapter, Fingerprint
    from matchlab.resolvers import Resolver
else:
    Adapter = Any
    Fingerprint = Any
    Resolver = Any

Resolution: TypeAlias = "Resolver | str"
"""One resolution to read: a resolver, or the label one was published under."""

Reading: TypeAlias = "tuple[Adapter, Fingerprint]"
"""A located resolution: the store holding it, and its fingerprint."""


class EvaluationFieldMetadata(BaseModel):
    """Metadata for a field in evaluation."""

    display_name: str
    source_columns: list[str]


class EvaluationItem(BaseModel):
    """A cluster ready for evaluation.

    The records dataframe contains the leaf IDs and the qualified index fields
    associated with it. For example:

    | leaf_id | src_a_first | src_a_last | src_b_first | src_b_last |
    |---------|-------------|------------|-------------|------------|
    | 1       | Thomas      | Bayes      |             |            |
    | 2       | Tommy       | B          |             |            |
    | 12      |             |            | Tom         | Bayes      |

    The fields attribute allows any evaluation system to map between a display
    version of the source columns, and the actual columns contained in the
    records dataframe. For example:

    ```text
    {
        "display_name": "first",
        "source_columns": "src_a_first", "src_b_first"
    }
    ```
    """

    model_config = {"arbitrary_types_allowed": True}

    leaves: list[int]
    records: pl.DataFrame
    fields: list[EvaluationFieldMetadata]

    def get_unique_record_groups(self) -> list[list[int]]:
        """Group identical records by leaf ID.

        Returns:
            List of groups, where each group is a list of leaf IDs
            that have identical values across all data fields.
            Example: [[1, 3], [2], [4, 5, 6]] means records 1 & 3 are identical.
        """
        # Get all data column names (not "leaf")
        # Flatten the source_columns lists from all fields
        data_cols = [col for field in self.fields for col in field.source_columns]

        # Group by all data columns to find duplicates
        grouped = self.records.group_by(data_cols, maintain_order=True).agg(
            pl.col("leaf")
        )

        # Extract list of leaf ID lists
        return [group for group in grouped["leaf"]]


def create_judgement(
    item: EvaluationItem,
    assignments: dict[int, str],
    tag: str | None = None,
) -> Judgement:
    """Convert item assignments to Judgement - no default group assignment.

    Args:
        item: evaluation item
        assignments: column assignments (group_idx -> group_letter)
        tag: string by which to tag the judgement

    Returns:
        Judgement with endorsed groups based on assignments
    """
    groups: dict[str, list[int]] = {}
    unique_record_groups = item.get_unique_record_groups()

    for col_idx, group in assignments.items():
        leaf_ids = unique_record_groups[col_idx]
        groups.setdefault(group, []).extend(leaf_ids)

    endorsed = [sorted(set(leaf_ids)) for leaf_ids in groups.values()]
    return Judgement(shown=item.leaves, endorsed=endorsed, tag=tag)


def create_evaluation_item(
    df: pl.DataFrame,
    source_fields: list[tuple[str, list[str]]],
    leaves: list[int],
) -> EvaluationItem:
    """Create EvaluationItem with structured metadata.

    Args:
        df: The cluster's rows, with source-qualified data columns.
        source_fields: `(prefix, qualified columns)` per source. The columns come
            from the fetched data rather than from the sources, which would have to
            re-read the warehouse just to list their names.
        leaves: The leaf IDs in this cluster.
    """
    # Get all data columns (exclude metadata columns)
    data_cols = [c for c in df.columns if c not in ["root", "leaf", "key"]]

    # Build mapping of field_name -> list of qualified column names. The same field
    # in two sources shares a display name, which is what lines them up for review.
    field_to_columns: dict[str, list[str]] = {}

    for prefix, columns in source_fields:
        for column in columns:
            if column in data_cols:
                field = column.removeprefix(prefix)
                field_to_columns.setdefault(field, []).append(column)

    # Create EvaluationFieldMetadata objects (one per unique field name)
    fields: list[EvaluationFieldMetadata] = []
    for field_name, source_columns in field_to_columns.items():
        fields.append(
            EvaluationFieldMetadata(
                display_name=field_name, source_columns=source_columns
            )
        )

    # Keep ALL data columns in records
    records = df.select(["leaf"] + data_cols)

    return EvaluationItem(leaves=leaves, records=records, fields=fields)


def _many(resolution: "Resolution | Sequence[Resolution]") -> bool:
    """Whether several resolutions were asked for.

    A `str` is itself a `Sequence`, and a label is one resolution, not a pile of
    one-character ones — so it is excluded before the sequence check, not after.
    """
    return not isinstance(resolution, str) and isinstance(resolution, Sequence)


def _locate(
    resolution: "Resolution", adapter: "Adapter | None"
) -> tuple["Adapter", bytes]:
    """Turn a resolver or a label into the store and fingerprint to read.

    The two forms differ only in how the fingerprint is found — a live resolver knows
    its own; a label is looked up in a store. Everything downstream wants the same pair,
    so this is the only place the distinction exists.

    Raises:
        SourceTableError: If nothing is published under that label.
    """
    from matchlab.steps import default_adapter  # noqa: PLC0415 - avoids a cycle

    if isinstance(resolution, str):
        adapter = adapter or default_adapter()
        fingerprint = adapter.find(resolution)
        if fingerprint is None:
            known = ", ".join(adapter.labels()) or "none"
            raise SourceTableError(
                f"No resolution is published under the label '{resolution}'. "
                f"Known labels: {known}."
            )
        return adapter, fingerprint

    if not resolution.is_collected:
        resolution.collect(adapter)
    collected_in, fingerprint = resolution._collected()
    return adapter or collected_in, fingerprint


def _readings(
    resolution: "Resolution | Sequence[Resolution]", adapter: "Adapter | None"
) -> list[Reading]:
    """Locate every resolution asked for, in the order given."""
    if not _many(resolution):
        return [_locate(resolution, adapter)]
    if not resolution:
        raise ValueError("At least one resolution must be given.")
    return [_locate(one, adapter) for one in resolution]


def _sources_of(readings: list[Reading]) -> dict[str, Reading]:
    """Source name to the store and fingerprint its rows come from.

    Raises:
        SourceTableError: If two resolutions cover the same source name with different
            artifacts. Names repeat across generations of a source, so agreeing on the
            name is not agreeing on the data, and comparing methodologies over different
            data is not a comparison.
    """
    located: dict[str, Reading] = {}
    for store, fp in readings:
        for name, source_fp in store.resolution_sources(fp).items():
            seen = located.setdefault(name, (store, source_fp))
            if seen[1] != source_fp:
                raise SourceTableError(
                    f"These resolutions disagree about source '{name}': one covers "
                    f"{seen[1].hex()[:8]}, another {source_fp.hex()[:8]}. They are "
                    "built over different data, so their clusters cannot be compared. "
                    "Re-collect them over the same sources."
                )
    return located


def _merged_resolution(readings: list[Reading]) -> pl.DataFrame:
    """Union several resolutions' components into one `(root, leaf, key, source)`.

    Two records land in the same merged component when *either* resolution put them
    together. That is the right sample for a bake-off: every cluster where the
    methodologies could disagree is on screen, so one judgement settles it for both,
    and neither gets to pick the clusters it is scored on.

    Merged roots are minted with `root_id`, the same content-addressed function a
    resolver mints its own with, so two people running the same comparison key on the
    same IDs. Nothing persists them — `store_judgement` re-mints from the leaves it is
    given — so a merged root only ever lives as far as the reviewer.
    """
    frames = [store.read_resolver(fp) for store, fp in readings]

    components = DisjointSet[int]()
    for frame in frames:
        for leaves in frame.group_by("root").agg("leaf")["leaf"].to_list():
            components.add(leaves[0])
            for leaf in leaves[1:]:
                components.union(leaves[0], leaf)

    # `root_id` is invariant to leaf order but the caller does the sorting, and it is
    # vectorised because there are as many clusters here as there are entities.
    merged = (
        pl.DataFrame(
            {"leaf": [sorted(component) for component in components.get_components()]},
            schema={"leaf": pl.List(pl.UInt64)},
        )
        .with_columns(root_id(pl.col("leaf")).alias("root"))
        # A component always holds at least one leaf, so the empty-list case this
        # settles cannot arise; pinning it keeps the polars 2.0 default change quiet.
        .explode("leaf", empty_as_null=False)
    )

    records = pl.concat(
        [frame.select("leaf", "key", "source") for frame in frames]
    ).unique()
    return merged.join(records, on="leaf").select("root", "leaf", "key", "source")


def _sample_clusters(
    resolution: pl.DataFrame, n: int, seed: int | None
) -> pl.DataFrame:
    """Take up to `n` whole clusters from a resolution held in memory.

    The in-memory twin of `Adapter.sample`, for the merged resolution of several
    readings — which no store holds, because it exists only for the comparison.
    """
    roots = resolution["root"].unique()
    if n < roots.len():
        roots = roots.sample(n=n, seed=seed, shuffle=True)
    return resolution.filter(pl.col("root").is_in(roots.to_list()))


def get_samples(
    n: int,
    resolution: "Resolution | Sequence[Resolution]",
    adapter: "Adapter | None" = None,
    seed: int | None = None,
) -> dict[int, EvaluationItem]:
    """Retrieve samples enriched with source data as EvaluationItems.

    Record values come from the extract stored when each source was collected, not
    from a fresh warehouse read — so this works offline, and shows the data the
    matching actually saw.

    Args:
        n: Number of clusters to sample.
        resolution: The resolver to sample from — collected first if it isn't
            already — or the label one was published under, which needs no plan: a
            stored resolution records which source artifacts it covers. Pass several
            and the sample is drawn from their merged components, so one round of
            judging scores all of them against the same clusters.
        adapter: Where to read from. Defaults to the resolver's, else the module
            default.
        seed: Fixes which clusters come back. The same store, `n` and seed give the
            same sample, which is how two people review the same clusters.

    Returns:
        Dictionary of cluster ID to EvaluationItems describing the cluster.

    Raises:
        SourceTableError: If nothing is published under `resolution`, if a source the
            resolution covers isn't in the store, or if several resolutions disagree
            about a source.
        ValueError: If `resolution` is an empty sequence.
    """
    readings = _readings(resolution, adapter)

    if len(readings) == 1:
        store, resolver_fp = readings[0]
        samples = store.sample(resolver_fp, n, seed)
    else:
        samples = _sample_clusters(_merged_resolution(readings), n, seed)

    if not len(samples):
        return {}

    sources = _sources_of(readings)
    results_by_source: list[pl.DataFrame] = []
    source_fields: list[tuple[str, list[str]]] = []

    for source_step in samples["source"].unique():
        located = sources.get(source_step)
        if located is None:
            raise SourceTableError(
                f"This resolution references source '{source_step}', which is not "
                "in the store. Re-collect the plan to repopulate it."
            )
        source_store, source_fp = located

        samples_by_source = samples.filter(pl.col("source") == source_step)
        rows, qualified_key = source_store.read_source_records(
            source_fp, source_step, samples_by_source["key"].to_list()
        )
        values = [column for column in rows.columns if column != qualified_key]

        samples_and_source = samples_by_source.join(
            rows, left_on="key", right_on=qualified_key
        )
        source_fields.append((qualify(source_step), values))
        results_by_source.append(samples_and_source[["root", "leaf", "key"] + values])

    if not results_by_source:
        return {}

    all_results: pl.DataFrame = pl.concat(results_by_source, how="diagonal")

    results_by_root: dict[int, EvaluationItem] = {}
    for root in all_results["root"].unique():
        cluster_df = all_results.filter(pl.col("root") == root).drop("root")
        leaves = cluster_df.select("leaf").to_series().unique().to_list()
        evaluation_item = create_evaluation_item(cluster_df, source_fields, leaves)
        results_by_root[root] = evaluation_item

    return results_by_root


class EvalData:
    """Object which caches evaluation data to measure model performance."""

    def __init__(self, adapter: Adapter, tag: str | None = None) -> None:
        """Load judgement and expansion data used to compute evaluation metrics.

        Args:
            adapter: The storage adapter holding judgements (e.g. `dag.adapter`).
            tag: Optional tag to filter judgements by.
        """
        self.adapter = adapter
        self.tag = tag
        self.judgements, self.expansion = adapter.read_eval_data(tag)

    def precision_recall(
        self, resolution: "Resolution | Sequence[Resolution]"
    ) -> PrecisionRecall | list[PrecisionRecall]:
        """Score one or more resolutions against these judgements.

        Only pairs present in every resolution *and* in the judgements are compared, so
        scoring several at once is the fair way to rank them: each is measured over the
        same records, and none is flattered by clusters the others never saw. Scoring
        them one at a time gives each its own comparison set, and those numbers do not
        line up.

        Args:
            resolution: A resolver, the label one was published under, or a sequence of
                either.

        Returns:
            One `(precision, recall)` pair, or a list of them in the order given if a
            sequence was passed.
        """
        readings = _readings(resolution, self.adapter)
        scores = precision_recall(
            [
                store.read_resolver(fp).select("root", "leaf").unique()
                for store, fp in readings
            ],
            self.judgements,
            self.expansion,
        )
        return scores if _many(resolution) else scores[0]

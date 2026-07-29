"""Client-side helpers for retrieving and preparing evaluation samples."""

from typing import TYPE_CHECKING, Any

import polars as pl
from pydantic import BaseModel

from matchlab.core.dataframes import qualify
from matchlab.core.exceptions import SourceTableError
from matchlab.eval.judgements import Judgement
from matchlab.eval.metrics import precision_recall

if TYPE_CHECKING:
    from matchlab.adapters import Adapter
    from matchlab.resolvers import Resolver
else:
    Adapter = Any
    Resolver = Any


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


def _read_sample_file(sample_file: str, n: int) -> pl.DataFrame:
    resolver_matches_dump = pl.read_parquet(sample_file)
    clusters = resolver_matches_dump["id"].unique()
    select_clusters = clusters.sample(min(n, len(clusters)), shuffle=True).to_list()
    select_rows = resolver_matches_dump.filter(pl.col("id").is_in(select_clusters))
    return select_rows.rename({"id": "root", "leaf_id": "leaf"})


def _stored_records(
    adapter: "Adapter", source_fp: "bytes", name: str, keys: list[str]
) -> tuple[pl.DataFrame, str, list[str]]:
    """Return a source's rows for `keys`, from the store rather than the warehouse.

    The extract cached when the source was collected *is* the data the matching saw,
    which is what you want to judge a resolution against — and it means review needs
    no warehouse connection.

    Returns:
        `(rows, qualified key column, qualified value columns)`.
    """
    key_field = adapter.source_key_field(source_fp)
    extract = adapter.read_source_extract(source_fp)
    qualified = extract.select(pl.all().name.prefix(qualify(name)))
    qualified_key = qualify(name, key_field)
    rows = qualified.filter(pl.col(qualified_key).cast(pl.Utf8).is_in(keys))
    values = [column for column in qualified.columns if column != qualified_key]
    return rows.with_columns(pl.col(qualified_key).cast(pl.Utf8)), qualified_key, values


def _locate(
    resolution: "Resolver | str", adapter: "Adapter | None"
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


def get_samples(
    n: int,
    resolution: "Resolver | str",
    sample_file: str | None = None,
    adapter: "Adapter | None" = None,
) -> dict[int, EvaluationItem]:
    """Retrieve samples enriched with source data as EvaluationItems.

    Record values come from the extract stored when each source was collected, not
    from a fresh warehouse read — so this works offline, and shows the data the
    matching actually saw.

    Args:
        n: Number of clusters to sample.
        resolution: The resolver to sample from — collected first if it isn't
            already — or the label one was published under, which needs no plan: a
            stored resolution records which source artifacts it covers.
        sample_file: A parquet file written by `ResolverMatches.as_dump()`. Samples
            come from it rather than from the stored resolution.
        adapter: Where to read from. Defaults to the resolver's, else the module
            default.

    Returns:
        Dictionary of cluster ID to EvaluationItems describing the cluster.

    Raises:
        SourceTableError: If nothing is published under `resolution`, or if a source
            the resolution covers isn't in the store.
    """
    adapter, resolver_fp = _locate(resolution, adapter)

    if sample_file:
        samples = _read_sample_file(sample_file=sample_file, n=n)
    else:
        samples = adapter.sample(resolver_fp, n)

    if not len(samples):
        return {}

    sources = adapter.resolution_sources(resolver_fp)
    results_by_source: list[pl.DataFrame] = []
    source_fields: list[tuple[str, list[str]]] = []

    for source_step in samples["source"].unique():
        source_fp = sources.get(source_step)
        if source_fp is None:
            raise SourceTableError(
                f"This resolution references source '{source_step}', which is not "
                "in the store. Re-collect the plan to repopulate it."
            )

        samples_by_source = samples.filter(pl.col("source") == source_step)
        rows, qualified_key, values = _stored_records(
            adapter, source_fp, source_step, samples_by_source["key"].to_list()
        )

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
        self.tag = tag
        self.judgements, self.expansion = adapter.read_eval_data(tag)

    def precision_recall(self, results_eval: pl.DataFrame) -> tuple[float, float]:
        """Compute precision and recall for cluster data.

        Args:
            results_eval: a dataframe with id and leaf_id columns, where leaf_id must
                correspond to server leaf IDs.

        Returns:
            Precision and recall values as a tuple
        """
        values = precision_recall([results_eval], self.judgements, self.expansion)[0]
        return values[0], values[1]

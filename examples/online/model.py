"""The shared linkage model, trained once and reused.

The blog keeps estimation out of the repeatable run: train rarely, persist a
``model.json``, reuse it for prediction. Both pipelines here share one such artefact,
and the same standardisation (cleaning to a common ``name`` and ``postcode``).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import polars as pl
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

MODEL_JSON = Path(__file__).parent / "model.json"

# Standardisation: upper-case, strip whitespace, punctuation and company suffixes off
# the name, strip whitespace off the postcode. ``{0}`` is the raw (qualified) column.
CLEAN_NAME = (
    "regexp_replace(upper({0}), '\\s+|\\.|\\bLIMITED\\b|\\bLTD\\b|\\bPLC\\b', '', 'g')"
)
CLEAN_POSTCODE = "regexp_replace(upper({0}), '\\s+', '', 'g')"
THRESHOLD = 0.9


def _settings(link_type: str, trained: dict | None = None) -> SettingsCreator:
    """A Splink settings object, loading a trained artefact's m/u if given.

    Blocks on name only: this data's postcodes are low-cardinality (derived from the
    entity id), so blocking on them explodes the candidate set.
    """
    if trained is not None:
        settings = SettingsCreator.from_path_or_dict(trained)
        settings.link_type = link_type
        return settings
    return SettingsCreator(
        link_type=link_type,
        unique_id_column_name="id",
        retain_matching_columns=False,
        retain_intermediate_calculation_columns=False,
        blocking_rules_to_generate_predictions=[block_on("name")],
        comparisons=[
            cl.JaroWinklerAtThresholds("name", [0.9, 0.7]),
            cl.ExactMatch("postcode"),
        ],
    )


def load_or_train(nodes: pl.DataFrame) -> dict:
    """Return the trained model dict, training and persisting it on a miss."""
    if MODEL_JSON.exists():
        return json.loads(MODEL_JSON.read_text())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        linker = Linker(nodes.to_pandas(), _settings("dedupe_only"), db_api=DuckDBAPI())
        linker.training.estimate_probability_two_random_records_match(
            [block_on("name")], recall=0.8
        )
        linker.training.estimate_u_using_random_sampling(max_pairs=1e6, seed=1)
        linker.training.estimate_parameters_using_expectation_maximisation(
            block_on("name")
        )
    trained = linker.misc.save_model_to_json()
    MODEL_JSON.write_text(json.dumps(trained))
    return trained

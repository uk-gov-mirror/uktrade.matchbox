"""Model — a deduper or linker producing scored candidate edges."""

import inspect
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from matchlab.adapters import Adapter, Fingerprint
from matchlab.core.kinds import StepKind
from matchlab.core.logging import logger
from matchlab.core.schemas import SCHEMA_MODEL_EDGES
from matchlab.models import dedupers, linkers
from matchlab.models.dedupers.base import Deduper, DeduperSettings
from matchlab.models.linkers.base import Linker, LinkerSettings
from matchlab.specs import ModelSpec, ModelType
from matchlab.steps import Step
from matchlab.views import View

if TYPE_CHECKING:
    # `matchlab.resolvers` imports this module
    from matchlab.resolvers import Resolver
    from matchlab.resolvers.base import ResolverMethod, ResolverSettings

_MODEL_CLASSES: dict[str, type[Linker] | type[Deduper]] = {
    **dict(inspect.getmembers(dedupers, inspect.isclass)),
    **dict(inspect.getmembers(linkers, inspect.isclass)),
}


def add_model_class(model_class: type[Linker] | type[Deduper]) -> None:
    """Register a custom deduper or linker so it can be named in a plan."""
    if not issubclass(model_class, Linker | Deduper):
        raise ValueError("The argument is not a subclass of Deduper or Linker.")
    _MODEL_CLASSES[model_class.__name__] = model_class


def normalise_model_scores(scores: pl.DataFrame) -> pl.DataFrame:
    """Validate and normalise model output scores."""
    if not isinstance(scores, pl.DataFrame):
        raise ValueError(f"Expected a polars DataFrame, got {type(scores)}.")

    expected_fields = set(SCHEMA_MODEL_EDGES.names)
    if set(scores.columns) != expected_fields:
        raise ValueError(
            f"Expected {expected_fields}.\nFound {set(scores.column_names)}."
        )

    if scores.height == 0:
        scores = pl.DataFrame(schema=pl.Schema(SCHEMA_MODEL_EDGES))

    if not scores["score"].dtype.is_numeric():
        raise ValueError(
            "Score column must contain numeric values in the range [0.0, 1.0]."
        )

    normalised_scores = scores.with_columns(pl.col("score").cast(pl.Float32))
    invalid_scores = normalised_scores.filter(
        pl.col("score").is_null()
        | pl.col("score").is_nan()
        | (pl.col("score") < 0.0)
        | (pl.col("score") > 1.0)
    )
    if invalid_scores.height:
        min_score = normalised_scores["score"].min()
        max_score = normalised_scores["score"].max()
        raise ValueError(f"Score range misconfigured: [{min_score}, {max_score}]")

    unique_scores = (
        normalised_scores.with_columns(
            pl.concat_list(
                [pl.col("left_id").cast(pl.Utf8), pl.col("right_id").cast(pl.Utf8)]
            )
            .list.sort()
            .list.join("_")
            .alias("sorted_ids")
        )
        .sort("score", descending=True)  # sort so largest score comes first
        .unique(
            subset=["sorted_ids"], keep="first"
        )  # keep first occurrence after sorting
    ).drop("sorted_ids")
    if len(scores) != len(unique_scores):
        logger.warning("Duplicate pairs! Keeping only pairs with highest score.")

    return unique_scores.cast(pl.Schema(SCHEMA_MODEL_EDGES))


class Model(Step):
    """A deduper or linker over one or two cleaned views."""

    kind: ClassVar[StepKind] = StepKind.MODEL

    def __init__(
        self,
        left: View,
        model_class: type[Deduper] | type[Linker] | str,
        model_settings: DeduperSettings | LinkerSettings | dict,
        right: View | None = None,
    ) -> None:
        """Define a model.

        Args:
            left: The view to deduplicate, or the left side of a link.
            model_class: A `Deduper`/`Linker` subclass, or its registered name.
            model_settings: The settings object for that class, or a dict.
            right: The right side of a link. Omit for a deduper.
        """
        self.left = left
        self.right = right

        self.model_class = (
            _MODEL_CLASSES[model_class] if isinstance(model_class, str) else model_class
        )
        self.model_instance = self.model_class(settings=model_settings)
        self.model_type = (
            ModelType.LINKER
            if issubclass(self.model_class, Linker)
            else ModelType.DEDUPER
        )

        if isinstance(model_settings, dict):
            settings_class = self.model_class.model_fields["settings"].annotation
            self.model_settings = settings_class(**model_settings)
        else:
            self.model_settings = model_settings

        if (self.model_type == ModelType.LINKER) != (right is not None):
            raise ValueError(
                "A linker requires a right input; a deduper must not have one."
            )

        upstream: tuple[Step, ...] = (left,) if right is None else (left, right)
        super().__init__(upstream=upstream)

    # -- Step contract ----------------------------------------------------------------

    def __str__(self) -> str:
        """A model is drawn with the class implementing it."""
        return f"{self.kind}({self.model_class.__name__})"

    @property
    def spec(self) -> ModelSpec:
        """The serialisable spec for this model."""
        return ModelSpec(
            model_type=self.model_type,
            model_class=self.model_class.__name__,
            model_settings=self.model_settings.model_dump(mode="json"),
        )

    def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:
        left = self.left._read_cache(adapter)
        right = self.right._read_cache(adapter) if self.right else None

        if self.model_type == ModelType.LINKER:
            self.model_instance.prepare(left, right)
            scores = self.model_instance.link(left=left, right=right)
        else:
            self.model_instance.prepare(left)
            scores = self.model_instance.dedupe(data=left)

        adapter.store_model(fp, normalise_model_scores(scores))

    # -- data -------------------------------------------------------------------------

    def edges(self) -> pl.DataFrame:
        """Return this model's scored edges. Collects the plan first if needed."""
        if not self.is_collected:
            self.collect()
        adapter, fp = self._collected()
        return adapter.read_model(fp)

    @property
    def inputs(self) -> tuple[View, ...]:
        """The cleaned views this model reads."""
        return (self.left,) if self.right is None else (self.left, self.right)

    # -- verbs ------------------------------------------------------------------------

    def resolve(
        self,
        *other_models: "Model",
        resolver_class: "type[ResolverMethod] | str" = "Components",
        resolver_settings: "ResolverSettings | dict[str, Any] | None" = None,
    ) -> "Resolver":
        """Resolve this model (and any others) into clusters."""
        from matchlab.resolvers import Resolver  # noqa: PLC0415 - avoids a cycle

        return Resolver(
            self,
            *other_models,
            resolver_class=resolver_class,
            resolver_settings=resolver_settings,
        )

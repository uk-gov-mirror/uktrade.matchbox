"""Model — a deduper or linker producing scored candidate edges."""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from matchlab.adapters import Adapter, Fingerprint
from matchlab.cleaning import Clean
from matchlab.core.config import ModelType
from matchlab.core.logging import logger, profile_time
from matchlab.models import dedupers, linkers
from matchlab.models.dedupers.base import Deduper, DeduperSettings
from matchlab.models.linkers.base import Linker, LinkerSettings
from matchlab.results import normalise_model_scores
from matchlab.steps import Step

if TYPE_CHECKING:
    from matchlab.resolvers import Resolve
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


class Model(Step):
    """A deduper or linker over one or two cleaned views."""

    kind: ClassVar[str] = "model"

    def __init__(
        self,
        left: Clean,
        model_class: type[Deduper] | type[Linker] | str,
        model_settings: DeduperSettings | LinkerSettings | dict,
        right: Clean | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Define a model.

        Args:
            left: The view to deduplicate, or the left side of a link.
            model_class: A `Deduper`/`Linker` subclass, or its registered name.
            model_settings: The settings object for that class, or a dict.
            right: The right side of a link. Omit for a deduper.
            name: Optional plan name; derived from the inputs when omitted.
            description: Optional human description.
        """
        self.left = left
        self.right = right
        self.description = description

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
            settings_class = self.model_instance.__annotations__["settings"]
            self.model_settings = settings_class(**model_settings)
        else:
            self.model_settings = model_settings

        if (self.model_type == ModelType.LINKER) != (right is not None):
            raise ValueError(
                "A linker requires a right input; a deduper must not have one."
            )

        verb = "link" if right is not None else "dedupe"
        sources = "_".join(
            source.name for view in (left, right) if view for source in view.sources
        )
        upstream: tuple[Step, ...] = (left,) if right is None else (left, right)
        super().__init__(name=name or f"{verb}_{sources}", upstream=upstream)

    # -- Step contract ----------------------------------------------------------------

    def _config_key(self) -> bytes:
        return json.dumps(
            {
                "type": str(self.model_type),
                "model_class": self.model_class.__name__,
                "model_settings": self.model_settings.model_dump(mode="json"),
                "left": self.left.name,
                "right": self.right.name if self.right else None,
            },
            sort_keys=True,
        ).encode()

    @profile_time(attr="name")
    def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:
        logger.info("Building inputs", prefix=f"Run {self.name}")
        left = self.left._frame(adapter)
        right = self.right._frame(adapter) if self.right else None

        logger.info("Running model logic", prefix=f"Run {self.name}")
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
        return self._require_adapter().read_model(self._fp)

    @property
    def inputs(self) -> tuple[Clean, ...]:
        """The cleaned views this model reads."""
        return (self.left,) if self.right is None else (self.left, self.right)

    # -- verbs ------------------------------------------------------------------------

    def resolve(
        self,
        *other_models: Model,
        resolver_class: type[ResolverMethod] | str = "Components",
        resolver_settings: ResolverSettings | dict[str, Any] | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> Resolve:
        """Resolve this model (and any others) into clusters."""
        from matchlab.resolvers import Resolve  # noqa: PLC0415 - avoids a cycle

        return Resolve(
            self,
            *other_models,
            resolver_class=resolver_class,
            resolver_settings=resolver_settings,
            name=name,
            description=description,
        )

"""Transform, a plan node that reshapes one frame with a `Transformer`.

`Transform` is to `Transformer` what `Model` is to a `Deduper`/`Linker`, the lazy plan
node that wraps a serialisable methodology, folds its configuration into a cache key,
and runs it on collect. Its single input is a `Frame`, so transforms chain, each its
own cached artifact.
"""

from typing import ClassVar

import polars as pl

from matchlab.adapters import Adapter, Fingerprint
from matchlab.core.kinds import StepKind
from matchlab.frames import Frame, IdentifierRead
from matchlab.specs import TransformSpec
from matchlab.transformers.base import Transformer
from matchlab.transformers.clean import Clean
from matchlab.transformers.group import Group
from matchlab.transformers.select import Select

_TRANSFORMER_CLASSES: dict[str, type[Transformer]] = {}


def add_transformer_class(transformer_class: type[Transformer]) -> None:
    """Register a custom transformer so it can be named in a plan (and a document)."""
    if not issubclass(transformer_class, Transformer):
        raise ValueError("The argument is not a subclass of Transformer.")
    _TRANSFORMER_CLASSES[transformer_class.__name__] = transformer_class


for _builtin in (Select, Clean, Group):
    add_transformer_class(_builtin)


class Transform(Frame):
    """A frame reshaped by one transformer."""

    kind: ClassVar[StepKind] = StepKind.TRANSFORM

    def __init__(
        self,
        upstream: Frame,
        transformer: Transformer | type[Transformer] | str,
        transformer_settings: dict | None = None,
    ) -> None:
        """Define a transform.

        Args:
            upstream: The frame to reshape.
            transformer: A `Transformer` instance, or a subclass / its registered name
                to build from `transformer_settings`.
            transformer_settings: The configuration dict, when `transformer` is a class
                or a name. Ignored when it is already an instance.
        """
        if isinstance(transformer, Transformer):
            self.transformer = transformer
        else:
            transformer_class = (
                _TRANSFORMER_CLASSES[transformer]
                if isinstance(transformer, str)
                else transformer
            )
            self.transformer = transformer_class(**(transformer_settings or {}))

        self.transformer_class = type(self.transformer)
        self._input = upstream
        super().__init__(upstream=(upstream,))

    # -- Step contract ----------------------------------------------------------------

    def __str__(self) -> str:
        """A transform is drawn with the transformer implementing it."""
        return f"{self.kind}({self.transformer_class.__name__})"

    @property
    def spec(self) -> TransformSpec:
        """The serialisable spec for this transform."""
        return TransformSpec(
            transformer_class=self.transformer_class.__name__,
            transformer_settings=self.transformer.model_dump(mode="json"),
        )

    def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:
        reshaped = self.transformer.apply(self._input._read_cache(adapter))
        adapter.store_transform(fp, reshaped)

    # -- Frame contract ---------------------------------------------------------------

    def _read_cache(self, adapter: Adapter) -> pl.DataFrame:
        if self._fp is None:  # collect orders upstream first
            raise RuntimeError(
                "This transform has not been collected. Call collect() first."
            )
        return adapter.read_transform(self._fp)

    @property
    def _identifier_reads(self) -> tuple[IdentifierRead, ...]:
        """A transform reads the same records as its input, so the reads delegate up."""
        return self._input._identifier_reads

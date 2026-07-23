"""Base class for client-side DAG step nodes."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar

import polars as pl
import pyarrow as pa

from matchbox.common.arrow import check_schema_subset
from matchbox.common.dtos import (
    SourceStepName,
    Step,
    StepName,
    StepPath,
)
from matchbox.common.hash import HASH_FUNC, hash_arrow_table
from matchbox.common.logging import profile_time

if TYPE_CHECKING:
    from matchbox.adapters import Adapter, Fingerprint
    from matchbox.client.dags import DAG
else:
    DAG = Any
    Adapter = Any
    Fingerprint = bytes

T = TypeVar("T")


class StepConfigProtocol(Protocol):
    """Minimal protocol required by client DAG step config DTOs."""

    @property
    def dependencies(self) -> list[StepName]:
        """Execution prerequisites required before running the step."""
        ...

    @property
    def parents(self) -> list[StepName]:
        """Direct DAG edges to this step."""
        ...

    def model_dump_json(self, **kwargs: Any) -> str:
        """Serialise the config for stable hashing."""
        ...


def post_run(method: Callable[..., T]) -> Callable[..., T]:
    """Decorator to ensure that a method is called after step run.

    Raises:
        RuntimeError: If the step hasn't been run yet.
    """

    @wraps(method)
    def wrapper(self: "StepABC", *args: Any, **kwargs: Any) -> T:
        if self._local_data is None:
            raise RuntimeError("The step must be run before attempting this operation.")
        return method(self, *args, **kwargs)

    return wrapper


class StepABC(ABC):
    """Base class for client-side DAG nodes that compute and sync data."""

    _local_data_schema: ClassVar[pa.Schema]
    # Domain-separation tag so empty artifacts of different step types (which all hash
    # to b"empty_table_hash") never share a storage key.
    _kind_tag: ClassVar[str]

    def __init__(
        self,
        dag: DAG,
        name: str,
        description: str | None = None,
    ) -> None:
        """Initialise the step."""
        self.dag = dag
        self.name = name
        self.description = description
        self._local_data: pl.DataFrame | None = None
        # Content fingerprint under which this step's artifact is stored in the
        # adapter. Set on sync() and preserved across clear_data() so downstream
        # steps can read this step's data after its local copy is dropped.
        self._fp: Fingerprint | None = None

    # Local data access

    @property
    def local_data(self) -> pl.DataFrame | None:
        """The locally computed results for this step."""
        return self._local_data

    def clear_data(self) -> None:
        """Drop locally computed data."""
        self._local_data = None

    # Abstract interface

    @property
    @abstractmethod
    def path(self) -> StepPath:
        """The step path used to identify this step on the server."""
        ...

    @property
    @abstractmethod
    def sources(self) -> set[SourceStepName]:
        """Set of source names upstream of this node."""
        ...

    @property
    @abstractmethod
    def config(self) -> StepConfigProtocol:
        """Config DTO for this step."""
        ...

    @abstractmethod
    def to_dto(self) -> Step:
        """Convert to Step DTO (serialisable DAG representation)."""
        ...

    @abstractmethod
    def _store(self, adapter: Adapter) -> None:
        """Persist this step's computed artifact to the adapter under `self._fp`."""
        ...

    @classmethod
    def from_dto(
        cls,
        step: Step,
        step_name: str,
        dag: DAG,
        **kwargs: Any,
    ) -> "StepABC":
        """Reconstruct from Step DTO. Subclasses should override this."""
        raise NotImplementedError(f"{cls.__name__} must implement from_dto.")

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> pl.DataFrame:
        """Execute the step, populate _local_data, and return it."""
        ...

    # Concrete shared behaviour

    @property
    def cache_path(self) -> Path:
        """Path within the DAG cache for storing this step's local data."""
        return self.dag.cache_path / f"{self.name}.parquet"

    def __hash__(self) -> int:
        """Return a hash of the step based on its config."""
        return hash(self.config.model_dump_json())

    def __eq__(self, other: object) -> bool:
        """Check equality of two step objects based on their config."""
        if type(other) is not type(self):
            return False
        return self.config == other.config

    @post_run
    def _fingerprint(self) -> bytes:
        """Compute a content hash of the local data for fingerprinting."""
        check_schema_subset(
            expected=self._local_data_schema, actual=self._local_data.to_arrow().schema
        )
        return hash_arrow_table(self._local_data.to_arrow())

    def _store_key(self) -> Fingerprint:
        """Content fingerprint namespaced by step type, used as the adapter key."""
        return HASH_FUNC(self._kind_tag.encode() + self._fingerprint()).digest()

    @post_run
    @profile_time(attr="name")
    def sync(self) -> None:
        """Persist this step's computed artifact to the DAG's adapter.

        The step is content-addressed by its fingerprint: if an identical artifact is
        already stored, storage is skipped (a cache hit). The fingerprint is retained
        on `self._fp` so downstream steps can read this data even after `clear_data`.
        """
        self._fp = self._store_key()
        if not self.dag.adapter.has(self._fp):
            self._store(self.dag.adapter)

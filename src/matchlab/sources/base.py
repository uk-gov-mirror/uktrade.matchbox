"""The contract every location implements.

A `Location` is a methodology, exactly as a `Deduper` or a `Transformer` is. A `Source`
names the class and hands it settings and resources separately, so the query travels in
a document and the client never does. See `matchlab.resources`.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator

import polars as pl
from pydantic import BaseModel, ConfigDict

from matchlab.core.dataframes import DataFrameClass, DataFrameType


class Location(BaseModel, ABC):
    """A place data is read from, holding everything needed to read it.

    Every field is a setting unless marked `matchlab.resources.FromResources`.

    A location declares no `version`, unlike the methodologies the other kinds of step
    run. A source hashes the rows it read into its own spec key, so a change in what a
    location returns already moves the fingerprint.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    def __str__(self) -> str:
        """Identify the location by class."""
        return type(self).__name__

    @abstractmethod
    def read(
        self,
        batch_size: int | None = None,
        rename: dict[str, str] | Callable | None = None,
        return_type: DataFrameType = DataFrameType.POLARS,
        schema_overrides: dict[str, pl.DataType] | None = None,
    ) -> Iterator[DataFrameClass]:
        """Get this location's rows, in batches.

        Args:
            batch_size: The size used for internal batching.
            rename: Renaming to apply to the rows that come back.

                * If a dictionary is provided, it will be used to rename the columns.
                * If a callable is provided, it will take the old name as input and
                    return the new name.
            return_type: The type of data to return. Defaults to "polars".
            schema_overrides: Types to force on the columns that come back, rather than
                letting the location infer them.
        """
        ...

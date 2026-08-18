"""Reading a source from a dataframe already in memory."""

from collections.abc import Callable, Generator

import polars as pl
from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame

from matchlab.core.dataframes import DataFrameClass, DataFrameType, to_dataframe
from matchlab.resources import FromResources
from matchlab.sources.base import Location


class DataFrame(Location):
    """A dataframe already in memory."""

    df: FromResources[DataFrameClass]
    """The rows to read."""

    @property
    def _polars(self) -> PolarsDataFrame:
        """The frame as polars, whatever it was handed as."""
        if isinstance(self.df, PolarsDataFrame):
            return self.df
        if isinstance(self.df, PandasDataFrame):
            return pl.from_pandas(self.df)
        return pl.DataFrame(self.df)

    def read(  # noqa: D102
        self,
        batch_size: int | None = None,
        rename: dict[str, str] | Callable | None = None,
        return_type: DataFrameType = DataFrameType.POLARS,
        schema_overrides: dict[str, pl.DataType] | None = None,
    ) -> Generator[DataFrameClass, None, None]:
        frame = self._polars

        if schema_overrides:
            frame = frame.with_columns(
                pl.col(column).cast(dtype)
                for column, dtype in schema_overrides.items()
                if column in frame.columns
            )

        if rename:
            frame = frame.rename(rename)

        batch_size = batch_size or 10_000
        for offset in range(0, frame.height, batch_size):
            yield to_dataframe(frame.slice(offset, batch_size), return_type)

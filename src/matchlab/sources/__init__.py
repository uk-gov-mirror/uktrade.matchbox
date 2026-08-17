"""Location methodologies and the Source plan node.

A `Source` reads rows through a `Location`, exactly as a `Model` scores them through a
`Deduper` or a `Linker`. `RelationalDB` runs a query, `DataFrame` reads a frame already
in memory, and `read_database`/`read_dataframe` are the shorthand for each. Register a
custom location with `add_location_class`, as `add_model_class` does a deduper.
"""

from matchlab.sources.base import Location
from matchlab.sources.dataframe import DataFrame
from matchlab.sources.relational import ClientType, DBClient, RelationalDB
from matchlab.sources.sources import (
    Source,
    add_location_class,
    read_database,
    read_dataframe,
    resolve_location_class,
)

__all__ = (
    "ClientType",
    "DBClient",
    "DataFrame",
    "Location",
    "RelationalDB",
    "Source",
    "add_location_class",
    "read_database",
    "read_dataframe",
    "resolve_location_class",
)

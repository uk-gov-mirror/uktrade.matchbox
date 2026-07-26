"""The logger and console everything in matchlab reports through.

Two objects, one each for the two ways output leaves the library:

* `logger` — the `matchlab` logger, wrapped so any call can take a `prefix=`. Like any
  library logger it has no handler of its own, and is silent until the application
  configures logging.
* `console` — the Rich console. `matchlab.progress` draws the collection tree on it,
  and tests swap it for a quiet one.

Both are module attributes rather than values passed around, which is why
`matchlab.progress` reads them through the module (`mlog.console`): that is what lets a
test patch one and have the library pick it up.
"""

import logging
from typing import Any, Final

from rich.console import Console


class PrefixedLoggerAdapter(logging.LoggerAdapter):
    """A logger adapter that supports adding a prefix enclosed in square brackets.

    This adapter allows passing an optional prefix parameter to any logging call
    without modifying the underlying logger.
    """

    def process(
        self,
        msg: Any,  # noqa: ANN401
        kwargs: dict[str, Any],  # noqa: ANN401
    ) -> tuple[Any, dict[str, Any]]:  # noqa: ANN401
        """Process the log message, adding a prefix if provided.

        Args:
            msg: The log message
            kwargs: Additional arguments to the logging method

        Returns:
            Tuple of (modified_message, modified_kwargs)
        """
        prefix = kwargs.pop("prefix", None)

        if prefix:
            msg = f"[{prefix}] {msg}"

        return msg, kwargs


logger: Final[PrefixedLoggerAdapter] = PrefixedLoggerAdapter(
    logging.getLogger("matchlab"), {}
)
"""Logger for matchlab.

Used for all logging in the matchlab library.

Allows passing a prefix to any logging call. A collection prefixes each record with the
step's position, which is what ties the record to the plan `collect` drew.

Examples:
    ```python
    logger.info("Ran in 0.041s", prefix="step 2")
    ```
"""

console: Final[Console] = Console()
"""Console for matchlab.

Where `matchlab.progress` draws a collection's plan, and where any CLI utility writes.
"""

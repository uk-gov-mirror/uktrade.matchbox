"""An interactive reviewer for resolved clusters.

`review()` opens a terminal app showing one cluster at a time: the records that were
grouped together, laid out so you can see what matched. You paint them into groups and
each decision is stored as a `Judgement`, which `EvalData.precision_recall` then scores
a resolution against.

Textual is an optional dependency — `pip install matchlab[tui]`.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matchlab.adapters import Adapter
    from matchlab.resolvers import Resolver


def review(
    resolver: "Resolver | str",
    n: int = 5,
    adapter: "Adapter | None" = None,
    tag: str | None = None,
    sample_file: str | Path | None = None,
    show_help: bool = True,
) -> None:
    """Review resolved clusters interactively, recording judgements.

    Takes either a live `Resolver` — collected first if it isn't already — or the
    *name* of a resolution already in a store. The second form needs no plan and no
    warehouse: record values come from the extract cached when each source was
    collected, which is the data the matching actually saw.

    ```python
    review(entities, tag="session-1")  # from a plan
    review("entities", adapter=DuckDBAdapter("run.duckdb"))  # from a store alone
    ```

    Args:
        resolver: The resolver whose clusters to review, or the name of a stored
            resolution.
        n: How many clusters to hold in the queue at once.
        adapter: Where judgements are stored, and where samples are drawn from.
            Defaults to the module-level adapter.
        tag: Tags every judgement made in this session, so a later
            `EvalData(adapter, tag=...)` can score against just these.
        sample_file: A parquet file written by `ResolverMatches.as_dump()`. Samples
            come from it rather than from the stored resolution, which is how you
            review the same clusters someone else did.
        show_help: Show the key bindings on start.

    Raises:
        ImportError: If Textual is not installed.
    """
    try:
        from matchlab.eval.tui.app import EntityResolutionApp  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "Reviewing clusters interactively needs Textual. "
            "Install it with `pip install matchlab[tui]`."
        ) from exc

    named = isinstance(resolver, str)
    if not named and not resolver.is_collected:
        resolver.collect(adapter)

    EntityResolutionApp(
        resolver=None if named else resolver,
        resolver_name=resolver if named else None,
        num_samples=n,
        adapter=adapter,
        session_tag=tag,
        sample_file=str(sample_file) if sample_file else None,
        show_help=show_help,
    ).run()


__all__ = ["review"]

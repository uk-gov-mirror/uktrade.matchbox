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
    resolver: "Resolver",
    n: int = 5,
    adapter: "Adapter | None" = None,
    tag: str | None = None,
    sample_file: str | Path | None = None,
    show_help: bool = True,
) -> None:
    """Review a resolver's clusters interactively, recording judgements.

    Collects the resolver first if it isn't already.

    Args:
        resolver: The resolver whose clusters to review. Required even when
            `sample_file` is given — a sample file records which records were shown
            together, not their values, so the sources are re-read to display them.
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

    if not resolver.is_collected:
        resolver.collect(adapter)

    EntityResolutionApp(
        resolver=resolver,
        num_samples=n,
        adapter=adapter,
        session_tag=tag,
        sample_file=str(sample_file) if sample_file else None,
        show_help=show_help,
    ).run()


__all__ = ["review"]

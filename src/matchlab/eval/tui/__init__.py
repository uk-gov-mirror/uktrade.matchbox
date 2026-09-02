"""An interactive reviewer for resolved clusters.

`review()` opens a terminal app showing one cluster at a time: the records that were
grouped together, laid out so you can see what matched. You paint them into groups and
each decision is stored as a `Judgement`, which `EvalData.precision_recall` then scores
a resolver against.

`app` is imported inside `review()` rather than here, so that importing `matchlab.eval`
for its scoring helpers doesn't pay for Textual's import.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matchlab.eval.samples import ResolverRef
    from matchlab.stores import Store


def review(
    resolver: "ResolverRef | Sequence[ResolverRef]",
    n: int = 5,
    store: "Store | None" = None,
    tag: str | None = None,
    seed: int | None = None,
    show_help: bool = True,
) -> None:
    """Review resolved clusters interactively, recording judgements.

    Takes either a live `Resolver`, collected first if it isn't already, or the
    *name* of a resolver already published in a store. The second form needs no plan
    and no warehouse. Record values come from the extract cached when each source was
    collected, which is the data the matching actually saw.

    ```python
    review(companies, tag="session-1")  # from a plan
    review("companies", store=DuckDBStore("run.duckdb"))  # from a store alone
    review([naive, splink], tag="bakeoff")  # judge both at once
    ```

    Args:
        resolver: The resolver whose clusters to review, or the name one was
            published under. Pass several and you review their merged components, so
            one session's judgements score every one of them.
        n: How many clusters to hold in the queue at once.
        store: Where judgements are stored, and where samples are drawn from.
            Defaults to the module-level store.
        tag: Tags every judgement made in this session, so a later
            `EvalData(store, tag=...)` can score against just these.
        seed: Fixes which clusters this session draws, so someone else with the same
            store can review the same ones.
        show_help: Show the key bindings on start.
    """
    from matchlab.eval.tui.app import EntityResolutionApp  # noqa: PLC0415 - lazy

    EntityResolutionApp(
        resolver=resolver,
        num_samples=n,
        store=store,
        session_tag=tag,
        seed=seed,
        show_help=show_help,
    ).run()


__all__ = ["review"]

"""The `matchlab` command.

Two commands. A terminal does two things better than a Python prompt. It tells you
what version you have, and it gives you a full-screen reviewer.

`review` takes a `module:attribute` naming a resolver in your code. With `--store`,
it instead takes the label of a resolver output already published. The second form
needs no plan and no warehouse. A stored resolver output knows which source artifacts
it covers, and their extracts hold the values. Name more than one target and you
review their merged clusters.
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matchlab.resolvers import Resolver


def _load_target(target: str) -> "Resolver":
    """Import `module:attribute` and return the resolver it names.

    The attribute may be a `Resolver`, or a callable returning one. The callable form
    is useful when building the plan needs a warehouse connection you'd rather not
    open at import time. A `Resolver` isn't callable, so the two never blur.

    Raises:
        SystemExit: If the target can't be imported or isn't a resolver.
    """
    from matchlab.resolvers import Resolver  # noqa: PLC0415 - keeps startup cheap

    if ":" not in target:
        raise SystemExit(
            f"Target '{target}' should look like 'module:attribute', for example "
            "'pipeline:entities', naming a Resolver in pipeline.py."
        )

    module_name, _, attribute = target.partition(":")

    # Lets a script in the working directory be named directly.
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"Could not import '{module_name}': {exc}") from exc

    try:
        obj: Any = getattr(module, attribute)
    except AttributeError as exc:
        raise SystemExit(f"'{module_name}' has no attribute '{attribute}'") from exc

    if callable(obj):
        obj = obj()

    if not isinstance(obj, Resolver):
        raise SystemExit(
            f"'{target}' is a {type(obj).__name__}, not a Resolver. Review needs the "
            "resolver whose clusters you want to look at."
        )

    return obj


def _redirect_logging(log_file: str) -> None:
    """Send all logging to a file, so it doesn't scribble over the UI."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="matchlab",
        description="Build, run and evaluate entity resolution pipelines.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("version", help="Show the installed matchlab version.")

    review = commands.add_parser(
        "review",
        help="Review a resolver's clusters interactively.",
        description=(
            "Open the cluster reviewer. TARGET names a Resolver as "
            "'module:attribute', such as 'pipeline:companies' for a resolver assigned "
            "to `companies` in pipeline.py, or a function returning one, if building "
            "the plan needs a connection you'd rather open on demand. "
            "With --store, TARGET is instead the label of a resolver output published "
            "in that store. Then no Python is imported. The values shown come from the "
            "extracts cached when the sources were collected, so no warehouse "
            "connection is needed. "
            "Name several targets to review their merged clusters, so one session's "
            "judgements score all of them against the same records."
        ),
    )
    review.add_argument(
        "targets",
        metavar="TARGET",
        nargs="+",
        help="module:attribute, or a resolver output's label when --store is given",
    )
    review.add_argument(
        "-n", "--samples", type=int, default=5, help="Clusters to queue (default: 5)."
    )
    review.add_argument(
        "-t", "--tag", help="Tag every judgement from this session, for later scoring."
    )
    review.add_argument(
        "--seed",
        type=int,
        help="Fix which clusters are drawn, so someone else can review the same ones.",
    )
    review.add_argument(
        "-s",
        "--store",
        help="DuckDB store to read from and write judgements to "
        "(default: the shared store in your cache directory).",
    )
    review.add_argument(
        "--log", dest="log_file", help="Send log output to this file instead."
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the `matchlab` command."""
    args = _build_parser().parse_args(argv)

    if args.command == "version":
        from importlib.metadata import version  # noqa: PLC0415

        print(version("matchlab"))
        return

    if args.log_file:
        _redirect_logging(args.log_file)

    from matchlab.eval import review  # noqa: PLC0415
    from matchlab.stores import DuckDBStore  # noqa: PLC0415

    store = DuckDBStore(args.store) if args.store else None

    # With a store, a bare word is a label on a stored resolver output, so it needs no
    # plan.
    targets: list[Any] = []
    for target in args.targets:
        if store is not None and ":" not in target:
            known = store.labels()
            if target not in known:
                raise SystemExit(
                    f"No resolver output is labelled '{target}' in {args.store}. "
                    f"Known labels: {', '.join(known) or 'none'}."
                )
            targets.append(target)
        else:
            targets.append(_load_target(target))

    review(
        targets if len(targets) > 1 else targets[0],
        n=args.samples,
        store=store,
        tag=args.tag,
        seed=args.seed,
    )

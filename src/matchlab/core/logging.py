"""The logger and consoles everything in matchlab reports through.

Three module attributes decide how output leaves the library:

* `logger` — the `matchlab` logger, wrapped so any record can carry a prefix saying
  which part of a plan produced it. Like any library logger it starts with no handler of
  its own, so an application that configures logging gets these records wherever it
  sends everything else.  If no application handler is configured, audible temporarily
  lends the logger a console handler for the duration of a collection.
* `console` — the Rich UI console. `matchlab.progress` draws the collection tree on
  it, and tests swap it for a quiet one.
* `diagnostic_console` — a Rich console bound to stderr. `ConsoleHandler` uses
  it when stdout may belong to data rather than to a human-facing display.

The consoles are module attributes rather than values passed around, so
`matchlab.progress` reads them through the module (`mlog.console`) instead of
importing them directly.
That indirection is what lets a test patch one and have the library pick it up.

**A prefix is usually ambient, not passed.** The thing worth quoting is a step's
position in the plan being collected, and the code that knows a message worth logging
is rarely the code that knows the position — a `Linker` counting matches is three
layers below the walk that numbered its step. `collect` therefore publishes the
position in `prefix`, a `ContextVar`, and every record emitted while that step runs
picks it up. Call sites stay ignorant, which is what keeps positions a property of the
walk rather than of the step (see `matchlab.steps.Step.__init__`).

The `prefix=` keyword remains for callers with something to say outside a collection,
where there is no position to quote. It is a fallback, not an override: an ambient
position wins, so records read consistently within the run that printed the tree.
"""

import logging
import sys
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import IO, Any, Final

from rich.console import Console
from rich.file_proxy import FileProxy

prefix: Final[ContextVar[str | None]] = ContextVar("matchlab_prefix", default=None)
"""The prefix records carry right now, or `None` outside any collection.

Set by `matchlab.progress.Progress` for the duration of each step, so that anything
logged while that step runs is tied to the plan the collection drew. Being a
`ContextVar` is what makes it restore correctly when a collection runs inside another:
the inner one holds the outer one's token and gives it back on the way out.

A handler or filter that wants to know which step is running can read it directly.
Setting it is `Progress`'s business — a record of your own takes `prefix=` instead.
"""


class PrefixedLoggerAdapter(logging.LoggerAdapter):
    """A logger adapter that tags records with the plan position they came from.

    The prefix is normally ambient, read from `prefix`. A `prefix=` keyword on the call
    supplies one for records emitted outside a collection.
    """

    def process(
        self,
        msg: Any,  # noqa: ANN401
        kwargs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        """Add the ambient or explicit prefix to `msg`, if either is set."""
        # Popped either way: `prefix` is ours, and `Logger.log` would reject it.
        explicit = kwargs.pop("prefix", None)
        # Ambient first: inside a collection the position is the prefix that matches
        # the tree, and it is the one a reader can look up. A caller's own idea of
        # what it is only applies where there is no position to quote.
        tag = prefix.get() or explicit

        if tag:
            msg = f"[{tag}] {msg}"

        return msg, kwargs


logger: Final[PrefixedLoggerAdapter] = PrefixedLoggerAdapter(
    logging.getLogger("matchlab"), {}
)
"""Logger for matchlab.

Used for all logging in the matchlab library.

Records logged while a collection is running are prefixed with the position of the step
that produced them, which is what ties a record to the plan `collect` drew. Nothing at
the call site arranges that; see `prefix`.

Examples:
    ```python
    logger.info(f"Round {round_num}: Found {len(matches):,} matches")
    ```
"""

console: Final[Console] = Console()
"""Console for matchlab.

Where `matchlab.progress` draws a collection's plan, and where any CLI utility writes.
"""

diagnostic_console: Final[Console] = Console(stderr=True)
"""Console for fallback diagnostics.

Used for log records when matchlab is running without a terminal-backed UI
console, so machine-readable stdout stays clean.
"""


@contextmanager
def through_console() -> Generator[None]:
    """Route handlers already writing to this terminal through the console instead.

    Rich redirects `sys.stdout` and `sys.stderr` while a live display is running, so a
    bare `print` lands above the frame rather than through it. Logging misses out:
    `logging.StreamHandler` reads `sys.stderr` **once**, when it is constructed, and
    `logging.basicConfig()` constructs one at import time — long before there is a
    frame to avoid. That handler then writes straight past Rich for the rest of the
    process.

    That is the ordinary case, not an exotic one. Rather than asking every caller to
    work around it, `matchlab.progress` closes the gap itself: for as long as it owns
    the terminal, it points any handler bound to that terminal at the same proxy Rich
    installs, and puts it back afterwards. Handlers writing anywhere else — a file, a
    socket, another stream — are left alone, having nothing to collide with.

    It restores the original streams on the way out, however the block ends. It's
    reentrant only in the sense that a handler already proxied is skipped, so a nested
    display can't double-wrap one.

    It does nothing in a notebook. The problem it solves is a cursor collision, and a
    notebook has no cursor: Rich draws a live display into an `ipywidgets.Output` that
    redraws itself in isolation, so nothing can smear it. Routing records through the
    console there would replace each one with its own `display()` call — a separate
    HTML block, boxed and spaced apart. Left alone, the stream they already write to
    renders them as ordinary contiguous text instead.
    """
    if console.is_jupyter:
        yield
        return

    # Read before Rich swaps them: afterwards `sys.stdout` is a proxy, and a handler
    # holding the real stream would no longer look like it points at the terminal.
    terminal = {sys.stdout, sys.stderr, console.file}
    patched: list[tuple[logging.StreamHandler, IO[str]]] = []

    target: logging.Logger | None = logger.logger
    while target is not None:
        for handler in target.handlers:
            stream = getattr(handler, "stream", None)
            if (
                isinstance(handler, logging.StreamHandler)
                and not isinstance(stream, FileProxy)
                and any(stream is candidate for candidate in terminal)
            ):
                patched.append((handler, stream))
                handler.setStream(FileProxy(console, stream))
        target = target.parent if target.propagate else None

    try:
        yield
    finally:
        for handler, stream in patched:
            handler.setStream(stream)


class ConsoleHandler(logging.Handler):
    """A logging handler that writes through matchlab's own console.

    Sharing the console is what lets logging coexist with the live tree
    `matchlab.progress` draws: Rich prints through the live region, so records land
    above the frame instead of over it. A handler holding a raw stream — which is what
    `logging.basicConfig()` gives you, bound at the moment it was called — writes
    straight past Rich and smears whatever is being redrawn.

    Deliberately plainer than `rich.logging.RichHandler`, which lays records out in a
    table and would wrap a logged plan inside its narrow message column. What matchlab
    logs is mostly trees, and a tree that wraps is not a tree. Records are written at
    the console's full width, unstyled, and formatted however the application asked.
    Where stdout is carrying data or has been redirected away from a terminal, they go
    to stderr through `diagnostic_console` so the data stream stays clean.

    You do not need this to run a collection at a terminal: `through_console` already
    borrows a handler bound to one for as long as a live tree is up. This is for saying
    so up front, and for the handler that outlives any one collection.

    Examples:
        ```python
        logging.basicConfig(level=logging.INFO, handlers=[ConsoleHandler()])
        ```
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Write `record` to the appropriate console."""
        try:
            # `markup=False` matters: a drawn tree is full of `[2]`, which Rich would
            # otherwise try to parse as a style tag. `highlight=False` likewise — its
            # guesses at numbers and brackets colour the tree into confetti.
            target = (
                console
                if console.is_terminal or console.is_jupyter
                else diagnostic_console
            )
            target.print(self.format(record), markup=False, highlight=False)
        except Exception:  # noqa: BLE001 - a handler must never raise into the caller
            self.handleError(record)


@contextmanager
def audible() -> Generator[None]:
    """Give matchlab's records somewhere to go when the application never said.

    A library logger has no handler of its own, so everything a collection reports —
    the plan, the per-step records, the closing summary — is dropped unless the
    application called `logging.basicConfig()` first. That is the right default for a
    library being embedded in something larger, and the wrong one for the ordinary
    case: someone at a prompt running a plan, watching the tree fill in, and never
    told why the line saying what it cost isn't there.

    So for the length of a collection, and only where nothing is listening, this
    attaches a `ConsoleHandler` and opens the `matchlab` logger to `INFO`. Through the
    console, because that is what coexists with a live tree — see `ConsoleHandler`.
    Both changes are undone on the way out, so global logging state is left exactly as
    it was found, and an application that configures logging later still gets what it
    configures.

    **A handler anywhere up the chain counts as listening**, and this then does nothing
    at all. An application that configured logging gets what it asked for, at the level
    it chose, including a level that drops these records. That is also how to turn this
    off:

    ```python
    logging.getLogger("matchlab").addHandler(logging.NullHandler())
    ```

    This is the the standard way to silence a library, which works here for the same
    reason a real handler does.
    """
    target = logger.logger
    if target.hasHandlers():
        yield
        return

    handler = ConsoleHandler()
    # A level and the message, nothing else. No timestamp or logger name, which would
    # push a `[step 7]` off to the right of every line and indent a logged tree out of
    # shape. The level stays because it is what separates a step that ran from one
    # that failed.
    handler.setFormatter(logging.Formatter("%(levelname)-5s %(message)s"))
    # Only where the level was left unset. A caller who set one without attaching a
    # handler has still said what they want to see, and `INFO` is not it.
    level = target.level
    target.addHandler(handler)
    if level == logging.NOTSET:
        target.setLevel(logging.INFO)
    try:
        yield
    finally:
        target.removeHandler(handler)
        target.setLevel(level)

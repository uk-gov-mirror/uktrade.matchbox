"""Exceptions raised by matchlab."""

from pyarrow import Schema


class MatchlabError(Exception):
    """An error occurred in matchlab."""

    def __init__(self, message: str | None = None) -> None:
        """Initialise the error, defaulting the message to the class docstring."""
        super().__init__(message or self.__doc__)


class SchemaMismatch(MatchlabError):
    """An Arrow table did not have the expected schema."""

    def __init__(self, expected: Schema, actual: Schema) -> None:
        """Initialise with the schemas that failed to match."""
        super().__init__(f"Schema mismatch. Expected:\n{expected}\nGot:\n{actual}")


class ExtractTransformError(MatchlabError):
    """A source's extract/transform SQL is not valid."""


class SourceTableError(MatchlabError):
    """A source's table could not be read from its location."""


class StepNotFound(MatchlabError):
    """A step could not be found."""

    def __init__(self, message: str | None = None, name: str | None = None) -> None:
        """Initialise with an explicit message, or a name to build one from."""
        if message is None:
            message = (
                f"Step {name} not found." if name is not None else "Step not found."
            )
        super().__init__(message)
        self.name = name

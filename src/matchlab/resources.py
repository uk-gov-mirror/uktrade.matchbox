"""Resources — the non-serialisable parameters to steps.

With the exception of sources, resources cannot influence the output of a step, as they
are not part of a node's spec and thus don't affect the fingerprint.
"""

from typing import Any, Generic, TypeVar

from matchlab.core.exceptions import ResourceError

T = TypeVar("T")


class Resource(Generic[T]):
    """A named object supplied at load time rather than serialised.

    The name is what a document records and what `matchlab.document.load` looks up.
    Share one `Resource` freely: two locations reading the same warehouse should be
    given the same object, and `dump` checks that one name never covers two.
    """

    __slots__ = ("name", "value")

    #: The name a document records. `None` only for an anonymous resource, which is
    #: usable in-process and refused by `dump`.
    name: str | None
    value: T

    def __init__(self, name: str, value: T) -> None:
        """Name a value so a document can ask for it again.

        Args:
            name: What `load` looks this up by. Distinct from a source's `key_field`,
                which identifies a record rather than a resource.
            value: The real object, used as-is in this process.

        Raises:
            ResourceError: If the name is not a non-empty string.
        """
        if not isinstance(name, str) or not name:
            raise ResourceError(
                f"A resource name must be a non-empty string, got {name!r}."
            )
        self.name = name
        self.value = value

    @classmethod
    def anonymous(cls, value: T) -> "Resource[T]":
        """Wrap a value that was passed without a name.

        Usable for the whole life of the process, and refused by `dump`.
        """
        resource: Resource[T] = cls.__new__(cls)
        resource.name = None
        resource.value = value
        return resource

    @property
    def is_anonymous(self) -> bool:
        """Whether this resource was passed without a name, so cannot be dumped."""
        return self.name is None

    def __repr__(self) -> str:
        """Show the name, never the value, which may carry a credential."""
        label = "anonymous" if self.is_anonymous else repr(self.name)
        return f"Resource({label}, <{type(self.value).__name__}>)"


def as_resources(supplied: dict[str, Any] | None) -> dict[str, Resource]:
    """Normalise a `*_resources` argument, wrapping any bare values.

    Lets a caller pass `client=engine` as readily as `client=Resource("wh", engine)`.
    The bare form works in this process and is refused by `dump`.
    """
    return {
        field: value if isinstance(value, Resource) else Resource.anonymous(value)
        for field, value in (supplied or {}).items()
    }


def values_of(resources: dict[str, Resource]) -> dict[str, Any]:
    """The real objects, keyed by the settings field each fills."""
    return {field: resource.value for field, resource in resources.items()}


def names_of(resources: dict[str, Resource]) -> dict[str, str]:
    """Field name to resource name, for the fields that have one.

    What a document records. Anonymous resources are absent, which is how `dump` knows
    to refuse them.
    """
    return {
        field: resource.name
        for field, resource in resources.items()
        if resource.name is not None
    }

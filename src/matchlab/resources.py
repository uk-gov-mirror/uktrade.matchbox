"""Resources, the parts of a step that cannot be serialised.

Every step is built from a class, a settings dict, and a resources dict. Settings reach
the plan document and the step's fingerprint. Resources reach neither. A document
records a resource's name, never its value, so a plan travels without its credentials
and `matchlab.document.load` asks for the named objects again.

A fingerprint cannot see a resource, so a resource must not change what a step computes.
A source is the exception. Its resource is where the rows come from. A source
fingerprint hashes those rows, so a change at the origin still moves it.

Two names in this module are easy to confuse.

`Resource` is an object you build at a call site. It pairs a value with the name a
document records for it.

`FromResources` marks a field in a methodology's class body. It declares that the field
is filled from the step's `*_resources` argument.

A step's constructor brings the two together. It records the name for the document and
passes the value into the marked field.
"""

from typing import Annotated, Any, Final, Generic, TypeAlias, TypeVar

from pydantic import BaseModel

from matchlab.core.exceptions import ResourceError

T = TypeVar("T")


class Resource(Generic[T]):
    """A value paired with the name a document records for it.

    Share one `Resource` between steps that need the same object. Give two locations
    reading one warehouse the same `Resource`.
    """

    __slots__ = ("name", "value")

    # What a document records. `None` for an anonymous resource, which works in this
    # process and is refused by `dump`.
    name: str | None
    value: T

    def __init__(self, name: str, value: T) -> None:
        """Name a value so a document can ask for it again.

        Args:
            name: What `matchlab.document.load` looks this up by.
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
        """Wrap a value passed without a name.

        The step works for the whole life of the process. Only `dump` refuses it,
        because a document has no name to record.
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


class _IsResource:
    """The marker `FromResources` attaches to a field. Carries no data."""

    __slots__ = ()

    def __repr__(self) -> str:
        """Keeps the rendered annotation readable as `FromResources[Engine]`."""
        return "resource"


IS_RESOURCE: Final = _IsResource()

FromResources: TypeAlias = Annotated[T, IS_RESOURCE]
"""Mark a field as a resource rather than a setting.

Every field of a location or methodology is a setting unless you mark it:

```python
class RelationalDB(Location):
    sql: SQLQuery                    # a setting
    client: FromResources[DBClient]  # a resource
```

The mark is what a step reads to decide which fields to leave out of the settings it
serialises and hashes. Passing a field in the wrong argument raises `ResourceError`.

Pydantic cannot validate most resource types. Add
`model_config = ConfigDict(arbitrary_types_allowed=True)` to the class that needs it.
"""


def resource_fields(methodology_class: type[BaseModel]) -> frozenset[str]:
    """The fields marked `FromResources`, inherited ones included."""
    return frozenset(
        field_name
        for field_name, field in methodology_class.model_fields.items()
        if any(isinstance(entry, _IsResource) for entry in field.metadata)
    )


def as_resources(supplied: dict[str, Any] | None) -> dict[str, Resource]:
    """Normalise a `*_resources` argument, wrapping any bare values.

    A caller can pass `client=engine` as readily as `client=Resource("wh", engine)`.
    """
    return {
        field: value if isinstance(value, Resource) else Resource.anonymous(value)
        for field, value in (supplied or {}).items()
    }


def values_of(resources: dict[str, Resource]) -> dict[str, Any]:
    """The real objects, keyed by the field each fills."""
    return {field: resource.value for field, resource in resources.items()}


def names_of(resources: dict[str, Resource]) -> dict[str, str]:
    """Field name to resource name, for the fields that have one.

    What a document records. Anonymous resources are missing from the result, which is
    how `dump` spots them.
    """
    return {
        field: resource.name
        for field, resource in resources.items()
        if resource.name is not None
    }

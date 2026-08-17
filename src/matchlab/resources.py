"""Resources — the non-serialisable parameters to steps.

With the exception of sources, resources cannot influence the output of a step, as they
are not part of a node's spec and thus don't affect the fingerprint.

Two names appear here, in two different places, and they are easy to confuse:

* `Resource` is an **object you make**, at a call site: a value plus the name a document
  records for it. `Resource("warehouse", engine)`.
* `FromResources` is a **note on a field**, in a methodology's class body. It holds
  nothing; it declares that the field is filled through the step's `*_resources`
  argument rather than its `*_settings`. `client: FromResources[DBClient]`.

They meet when a step is built: the constructor opens the box, records
`resource.name` for the document and passes `resource.value` into the marked field. By
the time the methodology exists the wrapper is gone, which is why a marked field is
typed as the value it ends up holding.
"""

from collections.abc import Mapping
from typing import Annotated, Any, Final, Generic, TypeAlias, TypeVar

from pydantic import BaseModel

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


class _IsResource:
    """The marker `FromResources` attaches to a field. Carries no data."""

    __slots__ = ()

    def __repr__(self) -> str:
        """Read as `FromResources[Engine]` wherever an annotation is rendered."""
        return "resource"


IS_RESOURCE: Final = _IsResource()

FromResources: TypeAlias = Annotated[T, IS_RESOURCE]
"""Mark a methodology's field as a resource rather than a setting.

Every field of a location, transformer, deduper, linker or resolver methodology is a
setting — serialised into a document, hashed into the fingerprint — unless it is marked
with this, in which case it is supplied through the step's `*_resources` argument and a
document records only its name:

```python
class RelationalDB(Location):
    sql: SQLQuery                    # a setting
    client: FromResources[DBClient]  # a resource
```

A field typed as anything pydantic cannot validate also needs
`model_config = ConfigDict(arbitrary_types_allowed=True)` on the class, which keeps that
loosening scoped to the methodology that needs it.
"""


def resource_fields(methodology_class: type[BaseModel]) -> frozenset[str]:
    """The fields of a methodology marked `FromResources`, including inherited ones."""
    return frozenset(
        name
        for name, field in methodology_class.model_fields.items()
        if any(isinstance(entry, _IsResource) for entry in field.metadata)
    )


def check_resource_split(
    methodology_class: type[BaseModel],
    settings: Mapping[str, Any],
    resources: Mapping[str, Any],
    prefix: str,
) -> None:
    """Check each field was passed in the argument its declaration calls for.

    Called by every step before it builds its methodology. Without it, `*_resources`
    would be taken on trust: a step excludes those fields when dumping its settings, so
    a setting passed as a resource would vanish from the spec and two steps doing
    different work would share a fingerprint.

    Args:
        methodology_class: The location or methodology about to be built.
        settings: The `*_settings` argument, as passed.
        resources: The `*_resources` argument, as passed.
        prefix: The argument prefix for this step kind, e.g. `"location"`.

    Raises:
        ResourceError: If a resource was passed among the settings, or a field that is
            not a declared resource was passed among the resources.
    """
    name = methodology_class.__name__
    declared = resource_fields(methodology_class)

    if misplaced := sorted(set(settings) & declared):
        fields = ", ".join(f"'{field}'" for field in misplaced)
        raise ResourceError(
            f"{fields} on {name} {'are' if len(misplaced) > 1 else 'is'} marked "
            f"`FromResources`, so must be passed in `{prefix}_resources` rather than "
            f"`{prefix}_settings`. A document then records the name rather than the "
            "value, and no credential is serialised."
        )

    for field in sorted(set(resources) - declared):
        if field not in methodology_class.model_fields:
            raise ResourceError(f"{name} has no field '{field}'.")
        raise ResourceError(
            f"'{field}' on {name} is a setting, not a resource, so must be passed in "
            f"`{prefix}_settings`. Only a field marked `FromResources` may go in "
            f"`{prefix}_resources`: a step excludes its resources from the settings it "
            "hashes, so this one would change with no re-run."
        )


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

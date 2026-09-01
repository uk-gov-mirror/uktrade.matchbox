"""What a methodology contributes to the fingerprint of the step that runs it."""

import secrets
from typing import Any

# Length of the nonce an unversioned methodology gets. Matches a `Fingerprint`.
NONCE_BYTES = 32

# Tagged apart so a version can never be read as a nonce, however the bytes fall.
_VERSION_TAG = b"|v"
_NONCE_TAG = b"|n"


def methodology_key(methodology_class: type | None) -> bytes:
    """Return the bytes `methodology_class` adds to its step's spec key.

    Args:
        methodology_class: The class whose code the step runs, or `None` for a step
            that runs no methodology. A source is the real case, since it is keyed by
            the data it read rather than by code.

    Returns:
        Nothing for `None`, the declared version for a versioned methodology, and a
        fresh nonce for an unversioned one. A nonce is what makes the step re-run.

    Raises:
        TypeError: If `version` is neither an `int` nor `None`.
    """
    if methodology_class is None:
        return b""

    version = declared_version(methodology_class)
    if version is None:
        return _NONCE_TAG + secrets.token_bytes(NONCE_BYTES)
    return _VERSION_TAG + str(version).encode()


def declared_version(methodology_class: type) -> int | None:
    """Return the version `methodology_class` declares, checking it is one.

    Raises:
        TypeError: If `version` is a settings field, or holds anything but an `int`.
    """
    if "version" in getattr(methodology_class, "model_fields", {}):
        raise TypeError(
            f"{methodology_class.__name__} declares `version` as a setting. It is a "
            "class attribute, so write `version: ClassVar[int] = 1`. Pydantic lets a "
            "field of that name shadow the inherited attribute, leaving the class "
            "reading as though it had declared nothing, and the step would refresh on "
            "every collect however the setting was filled in."
        )

    version: Any = getattr(methodology_class, "version", None)
    if version is None or isinstance(version, int):
        return version

    raise TypeError(
        f"{methodology_class.__name__}.version holds {type(version).__name__}, and a "
        "version must be an int or None. Count it up by one whenever the code changes "
        "what the class computes. A version is not a release number, so nothing else "
        "reads it and nothing orders two of them."
    )

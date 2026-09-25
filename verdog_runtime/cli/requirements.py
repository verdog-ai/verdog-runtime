"""PEP 508 requirements declared by one workflow environment."""

from collections.abc import Iterable

import packaging.requirements
import packaging.specifiers
import packaging.utils


def parse_requirements(source: str, path: str) -> tuple[str, ...]:
    """Parse one requirement per nonblank, non-comment line."""
    result: list[str] = []
    seen: dict[str, int] = {}
    for line_number, raw in enumerate(source.splitlines(), 1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        if value.startswith("-"):
            raise ValueError(
                f"{path}:{line_number}: pip directives are not supported"
            )
        try:
            name = packaging.utils.canonicalize_name(
                packaging.requirements.Requirement(value).name
            )
        except packaging.requirements.InvalidRequirement as error:
            raise ValueError(
                f"{path}:{line_number}: invalid PEP 508 requirement: {value}"
            ) from error
        if name in seen:
            raise ValueError(
                f"{path}:{line_number}: requirement {name} is already declared "
                f"at line {seen[name]}"
            )
        seen[name] = line_number
        result.append(value)
    return tuple(result)


def canonical_requirements(values: Iterable[str], /) -> tuple[str, ...]:
    """Serialize requirements exactly as the catalogue service does."""
    normalized: list[str] = []
    names: set[str] = set()
    for value in values:
        try:
            requirement = packaging.requirements.Requirement(value)
        except packaging.requirements.InvalidRequirement as error:
            raise ValueError(f"invalid PEP 508 requirement: {value}") from error
        name = packaging.utils.canonicalize_name(requirement.name)
        if name in names:
            raise ValueError(f"requirement {name} is declared more than once")
        names.add(name)
        normalized.append(str(requirement))
    return tuple(sorted(normalized, key=str.casefold))


def canonical_python_specifier(value: str, /) -> str:
    """Serialize a Python constraint exactly as the catalogue service does."""
    try:
        return str(packaging.specifiers.SpecifierSet(value))
    except packaging.specifiers.InvalidSpecifier as error:
        raise ValueError(
            f"invalid Python compatibility constraint: {value}"
        ) from error


__all__ = [
    "canonical_python_specifier",
    "canonical_requirements",
    "parse_requirements",
]

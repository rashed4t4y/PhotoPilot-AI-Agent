"""Shared exact-duplicate normalization; no invented keywords or synonym filler."""


def normalize_keywords(values: list[str]) -> list[str]:
    """Remove empty/exact duplicates using normalized whitespace and casefold.

    Preserve first spelling and relevance order. Semantic duplicates still need
    human review; this function does not claim to validate visual grounding.
    """
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError("Keywords must be a list of text values.")
    result, seen = [], set()
    for value in values:
        value = " ".join(value.split())
        identity = value.casefold()
        if value and identity not in seen:
            result.append(value)
            seen.add(identity)
    return result

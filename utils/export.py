"""In-memory exports from the current result, never from a cached model reply."""
import copy
import csv
from io import StringIO
from collections.abc import Mapping
from typing import Any
from utils.keywords import normalize_keywords
from utils.stock_readiness import DISCLAIMER


EDITABLE_FIELDS = ("analysis", "caption", "description", "keywords")


def with_edits(result: dict, edits: Mapping[str, str]) -> dict:
    """Copy the report and overlay edits; preserve original task statuses."""
    updated = copy.deepcopy(result)
    changed = []
    for field in EDITABLE_FIELDS:
        if field not in edits or field not in updated.get("outputs", {}):
            continue
        value = edits[field]
        if field == "keywords":
            value = normalize_keywords(value.replace("\n", ",").split(","))
        if value != updated["outputs"][field]:
            changed.append(field)
        updated["outputs"][field] = value
    updated["edited_fields"] = changed
    if "keywords" in updated.get("outputs", {}):
        updated["current_keyword_count"] = len(normalize_keywords(updated["outputs"]["keywords"]))
        updated["edit_notice"] = "Current keyword count reflects edits. Generation task statuses and generated counts are historical, not a revalidation of edited content."
    return updated


def _rows(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    rows = []
    if isinstance(value, dict):
        for key, item in value.items():
            rows.extend(_rows(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        rows.append((prefix, ", ".join(str(item) for item in value)))
    else:
        rows.append((prefix, "Not available" if value is None else str(value)))
    return rows


def _safe_cell(value: str) -> str:
    # Quote escaping alone does not prevent formulas. Account for leading
    # whitespace, control characters and BOMs that spreadsheet software ignores.
    probe = value.lstrip(" \t\r\n\v\f\ufeff\x00")
    if probe.startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


def export_txt(result: dict) -> str:
    return "PhotoPilot AI — photography report\n\n" + "\n\n".join(
        f"{field}\n{value}" for field, value in _rows({**result, "acceptance_disclaimer": DISCLAIMER})
    ) + "\n"


def export_csv(result: dict) -> str:
    """Portable field/value report, not a platform-specific submission template."""
    stream = StringIO(newline="")
    writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
    writer.writerow(("Field", "Value"))
    writer.writerows((_safe_cell(field), _safe_cell(value)) for field, value in _rows({**result, "acceptance_disclaimer": DISCLAIMER}))
    return stream.getvalue()

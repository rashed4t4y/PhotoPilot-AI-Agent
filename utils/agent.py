"""Bounded single-agent workflow with a conservative, offline English planner.

No arbitrary function dispatch, model-generated code, or open-ended tool loop.
Common intents are recognized locally; ambiguous/unsupported goals ask for
clarification. The validated plan is also suitable for a future model planner.
"""
import re
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from utils import gemini
from utils.export import export_csv, export_txt
from utils.image_tools import ImageValidationError, check_image_quality, validate_image
from utils.stock_readiness import DISCLAIMER, assess_resolution

Task = Literal["analysis", "caption", "description", "keywords", "quality", "readiness", "export"]
TASKS = ("analysis", "caption", "description", "keywords", "quality", "readiness", "export")
PATTERNS = {
    "analysis": r"\b(analy[sz]e|analysis|identify|what(?:'s| is) (?:in|visible)|composition|lighting)\b",
    "caption": r"\b(captions?|headline)\b",
    "description": r"\b(describe|descriptions?|alt[ -]?text)\b",
    "keywords": r"\b(keywords?|tags?)\b",
    "quality": r"\b(quality|blur(?:ry)?|sharpness|brightness|contrast|dimensions?|resolution|megapixels?|file size)\b",
    "readiness": r"\b(readiness|ready|eligibility)\b",
    "export": r"\b(export|download|csv|txt)\b",
}


class ClarificationNeeded(ValueError):
    """The goal cannot safely be mapped to supported work."""


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tasks: list[Task] = Field(min_length=1, max_length=7)
    caption_style: Literal["professional", "creative", "minimal", "descriptive"] = "professional"
    description_length: Literal["short", "medium", "long"] = "medium"
    keyword_count: int = Field(default=25, ge=1, le=50, strict=True)
    minimum_resolution_mp: float = Field(default=4.0, gt=0, allow_inf_nan=False)

    @field_validator("tasks")
    @classmethod
    def unique_tasks(cls, tasks):
        if len(tasks) != len(set(tasks)):
            raise ValueError("Duplicate tasks are not allowed.")
        if tasks == ["export"]:
            raise ValueError("Choose content to generate, or download an existing result.")
        return tasks


def validate_plan(proposal: dict | TaskPlan) -> TaskPlan:
    """Revalidate even model instances; never trust a bypassed constructor."""
    return TaskPlan.model_validate(proposal.model_dump() if isinstance(proposal, TaskPlan) else proposal)


def plan_goal(goal: str, **settings) -> TaskPlan:
    """Recognize explicit tasks or full stock packages without API calls.

    Negation, alternatives, and unsupported editing/publishing requests need
    clarification rather than risking extra work. Explicit task names override
    a platform mention (e.g. 'caption for Shutterstock' is caption-only).
    """
    if not isinstance(goal, str) or not goal.strip() or len(goal) > 2000:
        raise ClarificationNeeded("Enter a photography goal of 1–2000 characters.")
    if set(settings) - {"caption_style", "description_length", "keyword_count", "minimum_resolution_mp"}:
        raise ClarificationNeeded("Only caption style, description length, keyword count and resolution minimum are supported settings.")
    text = goal.lower().replace("’", "'")
    if re.search(r"\b(no|not|without|except|skip|don't|do not|or)\b", text):
        raise ClarificationNeeded("Please list just the tasks you want, without exclusions or alternatives.")
    if re.search(r"\b(delete|remove|retouch|resize|crop|upscale|publish|upload|send|email|execute|shell)\b", text):
        raise ClarificationNeeded("I can analyze, caption, describe, suggest keywords, check quality and export metadata—not edit or publish images.")
    selected = [task for task, pattern in PATTERNS.items() if re.search(pattern, text)]
    package = bool(re.search(r"\b(shutterstock|stock|portfolio)\b", text))
    full = bool(re.search(r"\b(full|complete|everything)\b", text))
    if package and (full or not selected or selected == ["export"]):
        selected = list(TASKS)
    if not selected or selected == ["export"]:
        raise ClarificationNeeded("Please specify analysis, caption, description, keywords or quality checks. Downloads are available after a run.")
    # Natural-language preferences override optional UI defaults when explicit.
    styles = [style for style in gemini.STYLES if re.search(rf"\b{style}\b", text)]
    if len(styles) > 1:
        raise ClarificationNeeded("Please choose one caption style.")
    if styles:
        settings["caption_style"] = styles[0]
    for length in gemini.LENGTHS:
        if re.search(rf"\b{length}\s+(?:image\s+)?description\b", text):
            settings["description_length"] = length
    count = re.search(r"\b(\d+)\s+keywords?\b", text)
    if count:
        settings["keyword_count"] = int(count.group(1))
    try:
        return validate_plan({"tasks": selected, **settings})
    except ValidationError:
        raise ClarificationNeeded("Check the settings: keyword count must be 1–50 and minimum resolution must be positive and finite.") from None


def execution_summary(result: dict) -> dict:
    """Terminal task statuses, reasons and next steps; not an acceptance verdict."""
    tasks = result.get("tasks", {})
    summary = {"goal": result.get("goal", ""), "planned": list(tasks)}
    for status in ("completed", "needs_review", "failed", "skipped"):
        summary[status] = [task for task, value in tasks.items() if value["status"] == status]
    summary["reasons"] = {task: value["message"] for task, value in tasks.items() if value.get("message")}
    summary["warnings"] = result.get("warnings", [])[:]
    summary["execution_success"] = bool(tasks) and not (summary["failed"] or summary["skipped"])
    steps = []
    if "keywords" in summary["needs_review"]:
        steps.append("Review keyword relevance and the requested count. Keep supported terms; lower the target rather than adding filler.")
    readiness = result.get("stock_readiness", {})
    if readiness.get("resolution_result") == "below_minimum":
        steps.append("Use a higher-resolution original or reshoot. No automatic upscaling is performed; upscaling does not guarantee acceptance.")
    if summary["failed"] or summary["skipped"]:
        steps.append("Resolve the listed validation/API errors, then request only the tasks that still need work. Successful results are retained.")
    if "description" in summary["needs_review"]:
        steps.append("Edit the description length without inventing unsupported details.")
    steps.append("Review generated facts, quality observations, licensing, intellectual property and platform policies before submission.")
    summary["next_steps"] = steps
    summary["acceptance_disclaimer"] = DISCLAIMER
    return summary


def run_agent(
    image_bytes: bytes, goal: str, *, plan: TaskPlan | dict | None = None,
    client=None, progress: Callable[[str, str], None] | None = None,
    **settings,
) -> dict:
    """Execute a validated, fixed-size plan and preserve independent successes.

    Visual fields share one Gemini request; quality/export are local. A supplied
    plan is an untrusted proposal: reject invalid or goal-inconsistent plans and
    fall back to the locally recognizable intent. No live request is made by
    planning, export, or quality-only runs.
    """
    expected = plan_goal(goal, **settings)
    notes = []
    if plan is not None:
        try:
            proposed = validate_plan(plan)
            if set(proposed.tasks) != set(expected.tasks):
                raise ValueError("Plan conflicts with requested tasks")
            # User goal/settings remain authoritative over proposal parameters.
        except (ValueError, TypeError):
            notes.append("Invalid or inconsistent plan ignored; used the recognized request.")
    plan = expected
    result = {
        "goal": goal, "plan": plan.model_dump(), "status": "running",
        "tasks": {task: {"status": "pending", "message": ""} for task in plan.tasks},
        "outputs": {}, "image_properties": {}, "warnings": notes,
        "acceptance_disclaimer": DISCLAIMER,
    }

    def update(task, status, message=""):
        result["tasks"][task] = {"status": status, "message": message}
        if progress:
            progress(task, status)

    try:
        validated = validate_image(image_bytes)
        try:
            width, height = validated.image.size
            result["image_properties"] = {
                "width": width, "height": height, "megapixels": width * height / 1_000_000,
                "file_format": validated.file_format, "file_size_bytes": validated.file_size_bytes,
            }
        finally:
            validated.image.close()
    except (ImageValidationError, OSError, ValueError):
        for task in plan.tasks:
            update(task, "skipped", "Not run: image validation failed. Use an intact JPEG/PNG within the size limits.")
        if "readiness" in plan.tasks:
            result["stock_readiness"] = assess_resolution(None, None, plan.minimum_resolution_mp)
            update("readiness", "failed", result["stock_readiness"]["message"])
        result["status"] = "failed"
        result["summary"] = execution_summary(result)
        return result

    if "readiness" in plan.tasks:
        update("readiness", "running")
        assessment = assess_resolution(width, height, plan.minimum_resolution_mp)
        result["stock_readiness"] = assessment
        update("readiness", assessment["status"], assessment["message"])
        if assessment["status"] != "completed":
            result["warnings"].append("readiness: " + assessment["message"])

    if "quality" in plan.tasks:
        update("quality", "running")
        try:
            result["outputs"]["quality"] = check_image_quality(image_bytes)
            observations = result["outputs"]["quality"].get("heuristic_flags", [])
            update("quality", "completed", "Measurements completed. " + " ".join(observations))
        except Exception:
            update("quality", "failed", "Local quality checks could not complete; other results are retained.")

    fields = tuple(field for field in gemini.FIELDS if field in plan.tasks)
    if fields:
        for task in fields:
            update(task, "running")
        try:
            metadata = gemini.generate_metadata(
                image_bytes, fields=fields, caption_style=plan.caption_style,
                description_length=plan.description_length, keyword_count=plan.keyword_count,
                client=client, allow_partial=True, user_goal=goal, keyword_retry=True,
            )
            if "keyword_review" in metadata:
                result["keyword_review"] = metadata["keyword_review"]
            result["warnings"].extend(metadata.get("warnings", []))
            for task in fields:
                if task in metadata and metadata[task]:
                    result["outputs"][task] = metadata[task]
                    flagged = any(note.startswith(task + ":") for note in metadata.get("warnings", []))
                    reasons = [note for note in metadata.get("warnings", []) if note.startswith(task + ":")]
                    if task == "keywords" and result.get("keyword_review"):
                        reasons = [result["keyword_review"]["reason"]]
                    update(task, "needs_review" if flagged else "completed", " ".join(reasons))
                else:
                    update(task, "failed", metadata.get("errors", {}).get(task, "No usable result returned."))
        except gemini.GeminiError as exc:
            for task in fields:
                update(task, "failed", str(exc))
        except Exception:
            for task in fields:
                update(task, "failed", "Metadata generation could not complete. Check settings and try again.")

    if "export" in plan.tasks:
        update("export", "running")
        try:
            # Validate serialization only; UI regenerates downloads from edits.
            export_txt(result)
            export_csv(result)
            update("export", "completed", "Report export is ready, including any failed-task statuses.")
        except Exception:
            update("export", "failed", "Report export could not be prepared.")

    substantive = [value["status"] for task, value in result["tasks"].items() if task != "export"]
    all_statuses = [value["status"] for value in result["tasks"].values()]
    if all(status == "completed" for status in all_statuses):
        result["status"] = "completed"
    elif any(status in ("completed", "needs_review") for status in substantive):
        result["status"] = "partial" if "failed" in all_statuses else "needs_review"
    else:
        result["status"] = "failed"
    result["summary"] = execution_summary(result)
    return result

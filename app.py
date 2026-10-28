"""PhotoPilot AI: upload, plan, run, review, edit and export."""
import hashlib
from io import BytesIO

import streamlit as st

from utils.agent import ClarificationNeeded, execution_summary, plan_goal, run_agent
from utils.export import EDITABLE_FIELDS, export_csv, export_txt, with_edits
from utils.image_tools import MAX_FILE_BYTES, ImageValidationError, validate_image
from utils.keywords import normalize_keywords

PRESETS = {
    "Caption + description (original workflow)": "Write a professional caption and a detailed description.",
    "Complete stock-photo package": "Prepare this image for my Shutterstock portfolio.",
    "Caption only": "Write a minimal caption.",
    "Keywords only": "Generate relevant stock photography keywords.",
    "Image analysis": "Analyze the visible subjects, lighting and composition.",
    "Quality checks (local, no API)": "Check image quality.",
    "Stock readiness (local, no API)": "Check Shutterstock readiness.",
}
LABELS = {
    "analysis": "Image analysis", "caption": "Caption", "description": "Description",
    "keywords": "Stock photography keywords", "quality": "Image quality", "export": "Prepare exports",
    "readiness": "Stock Readiness",
}


def clear_results():
    """Clear session results only; never change any file or uploaded image."""
    st.session_state.pop("pp_result", None)
    st.session_state.pop("pp_download_txt", None)
    st.session_state.pop("pp_download_csv", None)
    for field in EDITABLE_FIELDS:
        st.session_state.pop("pp_edit_" + field, None)


def select_preset():
    st.session_state["pp_goal"] = PRESETS[st.session_state["pp_preset"]]


def save_result(result):
    clear_results()
    st.session_state["pp_result"] = result
    for field in EDITABLE_FIELDS:
        if field in result["outputs"]:
            value = result["outputs"][field]
            st.session_state["pp_edit_" + field] = ", ".join(value) if field == "keywords" else value


def show_results():
    result = st.session_state.get("pp_result")
    if not result:
        return
    st.divider()
    st.subheader("Your results")
    messages = {
        "completed": "All requested tasks completed. Review the content before using it.",
        "needs_review": "Results are available, but some requested limits need review.",
        "partial": "Partially completed. Successful results are preserved; see failed tasks below.",
        "failed": "The requested content could not be completed. See the task messages below.",
    }
    if result["status"] == "completed":
        st.success(messages[result["status"]])
    elif result["status"] == "failed":
        st.error(messages[result["status"]])
    else:
        st.warning(messages[result["status"]])
    st.caption("Results belong to this completed run; changing the goal/settings does not regenerate them.")
    st.text("Run goal: " + result["goal"])
    summary = result.get("summary") or execution_summary(result)
    st.subheader("Agent execution summary")
    st.caption("Planned: " + ", ".join(LABELS[task] for task in summary["planned"]))
    st.caption(" · ".join(f"{status.replace('_', ' ').title()}: {len(summary[status])}"
                          for status in ("completed", "needs_review", "failed", "skipped")))
    st.caption("Execution completion is separate from content quality and stock-platform eligibility.")
    st.dataframe([
        {"Task": LABELS[task], "Status": value["status"], "Details": value["message"]}
        for task, value in result["tasks"].items()
    ], hide_index=True, width="stretch")
    for warning in result["warnings"]:
        st.warning(warning)
    with st.expander("Suggested next steps"):
        for step in summary["next_steps"]:
            st.write("• " + step)
    if "stock_readiness" in result:
        readiness = result["stock_readiness"]
        st.subheader("Stock Readiness")
        st.caption("Preliminary resolution check only — separate from image-quality measurements")
        st.text(f"Measured dimensions: {readiness['width']} × {readiness['height']} pixels")
        measured = readiness.get("megapixels")
        st.text(f"Measured resolution: {measured if measured is not None else 'unavailable'} MP · Required minimum: {readiness['minimum_megapixels']} MP")
        if readiness["resolution_result"] == "passed":
            st.success(readiness["message"])
        elif readiness["resolution_result"] == "unavailable":
            st.error(readiness["message"])
        else:
            st.warning(readiness["message"])
        st.caption(readiness["disclaimer"])
    if "keywords" in result["tasks"]:
        review = result.get("keyword_review", {})
        generated_count = review.get("actual_unique_count", len(normalize_keywords(result["outputs"].get("keywords", []))))
        st.caption(f"Keywords requested: {result['plan']['keyword_count']} · Generated unique keywords: {generated_count} · Refinement calls: {review.get('refinement_attempts', 0)}")
        st.text(review.get("reason") or result["tasks"]["keywords"]["message"])
    for field in EDITABLE_FIELDS:
        if field not in result["outputs"]:
            continue
        st.subheader(LABELS[field])
        st.text_area(
            "Edit " + LABELS[field].lower(), key="pp_edit_" + field,
            height=100 if field in ("caption", "keywords") else 160,
            help="Use commas or new lines between keywords." if field == "keywords" else "Your edits are included in both downloads.",
        )
        if field == "keywords":
            current_keywords = normalize_keywords(st.session_state["pp_edit_keywords"].replace("\n", ",").split(","))
            st.caption(f"Current edited unique keywords: {len(current_keywords)}. Duplicates and empty entries are removed in exports; relevance still needs human review.")
    if "quality" in result["outputs"]:
        quality = result["outputs"]["quality"]
        st.subheader("Image quality")
        st.caption("Measurements — not a platform acceptance decision")
        st.dataframe([
            {"Measurement": name, "Value": str(quality[name])}
            for name in ("width", "height", "megapixels", "file_format", "file_size_bytes",
                         "brightness_mean", "contrast_stddev", "laplacian_variance")
        ], hide_index=True, width="stretch")
        st.text(quality["measurement_notes"])
        st.markdown("**Heuristic observations**")
        for flag in quality["heuristic_flags"]:
            st.warning(flag)
        if not quality["heuristic_flags"]:
            st.info("No heuristic flags were triggered. This is not a quality guarantee.")
        st.caption(quality["limitations"])
    edits = {
        field: st.session_state["pp_edit_" + field]
        for field in EDITABLE_FIELDS if field in result["outputs"]
    }
    current = with_edits(result, edits)
    if current["edited_fields"]:
        st.caption("Downloads include your edits. Task statuses describe the original generation, not a revalidation of edits.")
    # Rebuild from widget state on every edit; no model calls here.
    st.session_state["pp_download_txt"] = export_txt(current)
    st.session_state["pp_download_csv"] = export_csv(current)
    st.subheader("Download your report")
    st.caption("Reports include available metadata, image properties, quality measurements and task statuses. CSV is a general report, not a Shutterstock upload template.")
    left, right = st.columns(2)
    left.download_button(
        "Download TXT", st.session_state["pp_download_txt"], file_name="photo_description.txt",
        mime="text/plain", on_click="ignore", width="stretch",
    )
    right.download_button(
        "Download CSV", st.session_state["pp_download_csv"].encode("utf-8-sig"),
        file_name="photopilot_metadata.csv", mime="text/csv", on_click="ignore", width="stretch",
    )


def main():
    st.set_page_config(page_title="PhotoPilot AI", page_icon="📸", layout="centered")
    # Static branding only; model content is never interpolated into HTML.
    st.markdown("""
    <style>
    .main-title{text-align:center;font-size:42px;font-weight:bold;color:#ff4b4b}
    .sub-title{text-align:center;color:gray;font-size:18px;margin-bottom:25px}
    .footer{text-align:center;color:gray;margin-top:40px;font-size:14px}
    </style>
    <div class='main-title'>📸 PhotoPilot AI</div>
    <div class='sub-title'>AI Powered Photography Assistant</div>
    """, unsafe_allow_html=True)
    st.info("AI tasks use your GEMINI_API_KEY environment variable or existing local .env configuration when you run them. Quality checks work without a key.")
    st.caption("AI runs send the image to Google Gemini and may use your API quota. Review generated facts and keywords; acceptance by stock platforms is not guaranteed.")
    uploaded = st.file_uploader(
        "Upload your image", type=["jpg", "jpeg", "png"], max_upload_size=20,
    )
    data = uploaded.getvalue() if uploaded is not None else None
    fingerprint = hashlib.sha256(data).hexdigest() if data is not None else None
    if fingerprint != st.session_state.get("pp_image_id"):
        clear_results()
        st.session_state["pp_image_id"] = fingerprint
        st.session_state.pop("pp_preview", None)
        st.session_state.pop("pp_image_error", None)
        if data is not None:
            try:
                validated = validate_image(data)
                try:
                    preview = validated.image.copy()
                    try:
                        preview.thumbnail((1200, 1200))
                        buffer = BytesIO()
                        preview.save(buffer, format="PNG")
                        st.session_state["pp_preview"] = buffer.getvalue()
                    finally:
                        preview.close()
                finally:
                    validated.image.close()
            except (ImageValidationError, OSError, ValueError):
                st.session_state["pp_image_error"] = "Invalid image. Use an intact single-frame JPEG/PNG, at most 20 MiB and 40 million pixels."
    if st.session_state.get("pp_image_error"):
        st.error(st.session_state["pp_image_error"])
    elif st.session_state.get("pp_preview"):
        st.image(st.session_state["pp_preview"], width="stretch")
        st.caption(f"Validated image · {len(data) / 1024:.1f} KiB")
    else:
        st.info("Start by uploading a photograph. Choose an example below or describe what you need.")
    st.caption(f"JPEG/PNG only · up to {MAX_FILE_BYTES // (1024 * 1024)} MiB · up to 40 megapixels · originals stay unchanged")
    st.selectbox("Example request", list(PRESETS), key="pp_preset", on_change=select_preset)
    if "pp_goal" not in st.session_state:
        st.session_state["pp_goal"] = next(iter(PRESETS.values()))
    goal = st.text_area("What would you like to do?", key="pp_goal", max_chars=2000)
    with st.expander("Optional preferences"):
        style = st.selectbox("Caption style", ["professional", "creative", "minimal", "descriptive"])
        length = st.selectbox("Description length", ["medium", "short", "long"])
        count = st.slider("Keyword count", min_value=1, max_value=50, value=25)
        minimum_mp = st.number_input("Stock resolution minimum (MP)", min_value=0.1, max_value=200.0, value=4.0, step=0.5)
        st.caption("Shutterstock-oriented default: 4 MP. This configurable preliminary check is not an acceptance guarantee. Stock readiness is local and never upscales an image.")
        st.caption("Explicit preferences in your request override these defaults. Word ranges: short 30–60, medium 80–120, long 150–200.")
    settings = {"caption_style": style, "description_length": length, "keyword_count": count,
                "minimum_resolution_mp": minimum_mp}
    plan = None
    try:
        plan = plan_goal(goal, **settings)
        st.caption("Selected tasks: " + " → ".join(LABELS[task] for task in plan.tasks))
    except ClarificationNeeded as exc:
        st.warning(str(exc))
    previous = st.session_state.get("pp_result")
    if previous and (goal != previous["goal"] or plan is None or plan.model_dump() != previous["plan"]):
        st.warning("Goal or settings changed. The results and edits below are from the previous run; press Run Agent to generate new results.")
    can_run = data is not None and not st.session_state.get("pp_image_error") and plan is not None
    if st.button("Run Agent", type="primary", width="stretch", disabled=not can_run):
        bar = st.progress(0.0, text="Validating image…")
        finished = set()

        def progress(task, status):
            if status in ("completed", "needs_review", "failed", "skipped"):
                finished.add(task)
            bar.progress(len(finished) / len(plan.tasks), text=f"{LABELS[task]}: {status.replace('_', ' ')}")

        try:
            result = run_agent(data, goal, plan=plan, progress=progress, **settings)
            save_result(result)
            bar.progress(1.0, text="Run finished — review results below")
        except ClarificationNeeded as exc:
            st.warning(str(exc))
        except Exception:
            st.error("The run could not finish. Previous results are retained. Try a smaller image or a simpler request.")
    st.button("Clear results / start another task", on_click=clear_results)
    show_results()
    st.markdown("<div class='footer'>Made with ❤️ using Streamlit &amp; Gemini AI</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

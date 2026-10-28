# 📸 PhotoPilot AI

An AI photography assistant built on the original Streamlit and Google Gemini
application. Upload a photo, describe your goal, review the selected tasks,
run the agent, edit the results, and download TXT or CSV reports.

## Features

- Original professional caption + detailed description workflow, preserved.
- Visible-content analysis: subjects, environment, lighting and composition.
- Caption styles: professional, creative, minimal and descriptive.
- Description targets: short (30–60 words), medium (80–120), long (150–200).
- Up to 50 stock-photo keywords, deduplicated in relevance order.
- One optional keyword-only refinement for valid shortfalls; requested, generated
  and edited unique counts are shown without adding unsupported filler.
- Separate Stock Readiness resolution assessment with a configurable minimum
  (4 MP for the Shutterstock-oriented preset), never an acceptance guarantee.
- Local dimensions, megapixels, actual format, file bytes, brightness, contrast
  and a documented blur heuristic.
- Structured, allowlisted task plans; progress and per-task completion status.
- Partial results survive independent failures. Shortfalls are marked for review.
- Editable outputs and session-state persistence; downloads use current edits.
- TXT and formula-protected, correctly escaped CSV exports.
- Local quality checks work without a Gemini key.

No image editing, publication, stock-site upload, batch processing or deployment
is performed. Original images are never overwritten or saved by this application.

## Windows setup

The pinned dependencies require **Python 3.12 or newer** (not Python 3.8).
The current project has been tested with **Python 3.13.14**. Other Python versions
are not certified by the test run in this checkout.

### Use this existing checkout

Open PowerShell:

```powershell
Set-Location 'C:\Users\rashe\Downloads\Photo to Description'
.\venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open http://localhost:8501. Stop the server with Ctrl+C.
Using the explicit interpreter avoids activation-policy and moved-launcher issues.
Do not recreate or delete an existing environment just to activate it.

If dependencies need installation in this existing environment, use:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

No additional dependencies are required for readiness or keyword refinement.

### Fresh clone only

These commands are for a new checkout without a virtual environment:

```powershell
git clone https://github.com/rashed4t4y/PhotoPilot-AI.git
Set-Location PhotoPilot-AI
py -3.13 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Optional activation (not necessary for the commands above):

```powershell
.\venv\Scripts\Activate.ps1
```

If activation is blocked, use the explicit interpreter commands rather than
changing machine-wide PowerShell security settings.

## API configuration

AI generation uses the existing `GEMINI_API_KEY` environment variable or a local
`.env` file. An environment variable already set takes precedence. Keep any
existing file and key; do not overwrite them with example values.

For a new configuration only, use a private local editor to create `.env` in the
project root with this placeholder replaced by your own key:

```dotenv
GEMINI_API_KEY=your_key_here
```

Do not paste keys into the goal field, commit them, print them in logs, or share
screenshots containing them. The repository ignores `.env`.
The app does not load the key at import/startup: configuration is loaded only
when you explicitly run an AI task. Missing configuration produces a task error,
not a startup crash. Quality-only requests and exports do not load it.

Images for AI tasks are sent to Google Gemini. Your account's API quota and
billing apply. Local checks remain on your machine. No real API call is part of
the automated tests.

The default model is `gemini-2.5-flash`. Model availability depends on your
account and Google's service; the Python integration accepts a `model` argument
for a future controlled change. This implementation has not been live-tested
against the service.

## Using the application

1. Upload a single-frame JPEG or PNG (up to 20 MiB and 40 million decoded pixels).
2. Select an example request or type a goal.
3. Adjust caption style, description length, keyword count and stock-resolution
   minimum if needed.
4. Review **Selected tasks** before pressing **Run Agent**.
5. Review task statuses, errors, warnings and image-quality measurements.
6. Edit analysis, caption, description and keywords in the text fields.
7. Leave the edited field (Tab or click outside) so Streamlit applies the edit,
   then download TXT or CSV.

Changing preferences or editing results does not call Gemini again. Pressing
Run Agent explicitly starts a new run. A new image clears results from the
previous image. “Clear results / start another task” clears only session results,
not files. Results live in the browser session and are not a permanent archive.
Changing the goal or settings displays a previous-run warning. Generated counts
and task statuses remain historical; edited keyword counts are shown separately.

### Example goals

- “Write a professional caption and a detailed description.”
- “Write a minimal caption for my portfolio.”
- “Prepare this image for my Shutterstock portfolio.”
- “Generate 15 keywords and a short description.”
- “Analyze the visible subjects, lighting and composition.”
- “Check image quality.”
- “Check Shutterstock readiness.” (local; no key required)
- “Create a caption and export CSV.”

The planner recognizes common English task terms and stock-package requests.
It is deliberately deterministic: no API call is needed to select tools.
Explicit tasks take precedence over a platform mention: “caption for
Shutterstock” is caption-only. A full stock-package request selects analysis,
caption, description, keywords, quality, stock readiness and export.

Unknown requests, exclusions, alternatives and unsupported image-editing or
publishing actions ask for clarification. Rephrase as a positive list of tasks
or choose a preset. This is not an unrestricted conversational planner.
Explicit recognized preferences such as “12 keywords” override UI defaults.

## Architecture and workflow

```text
app.py
  -> validate upload and preview
  -> agent.plan_goal: local intent recognition
  -> Pydantic TaskPlan: allowed tasks and bounded preferences
  -> agent.run_agent:
       separate local quality and resolution-readiness checks if requested
       one Gemini request for all requested visual metadata
       at most one keyword-only refinement after a valid shortfall
       independent field validation and task status
       local export preparation if requested
  -> session-state editors
  -> fresh TXT / CSV from current edits
```

| Module | Responsibility |
|---|---|
| [app.py](app.py) | Branding, upload, settings, progress, result editors and downloads |
| [utils/agent.py](utils/agent.py) | Bounded planning, allowlist validation, execution and completion checks |
| [utils/gemini.py](utils/gemini.py) | Lazy client, structured output, grounding and sanitized errors |
| [utils/image_tools.py](utils/image_tools.py) | Actual format/integrity validation and local quality measurements |
| [utils/stock_readiness.py](utils/stock_readiness.py) | Original-dimension resolution comparison and explicit unavailable outcomes |
| [utils/keywords.py](utils/keywords.py) | Shared whitespace/case duplicate removal without inventing terms |
| [utils/export.py](utils/export.py) | Edited report snapshots and TXT/CSV serialization |
| [tests/](tests/) | Generated-image, mocked-API and headless Streamlit integration tests |

Requested visual fields share one model request to avoid repeated image analysis.
An in-memory copy is reduced to at most 2048 pixels per side for the Gemini
payload; local quality metrics still use original resolution. Very small details
may be lost in this AI copy. The uploaded original is not changed.
The integration retries selected transient service/connection errors with bounded
backoff (two attempts by default, hard maximum three). SDK automatic retries are
disabled for clients created here. Authentication failures, malformed responses,
and read timeouts are not automatically retried.

The agent enables one keyword-only refinement when the initial response contains
valid, nonempty keywords but fewer unique terms than requested. That refinement
has **no additional transport retry**. The default agent maximum is therefore
three API calls: up to two initial attempts, plus at most one refinement. If the
initial request succeeds immediately, at most two calls are used. A sufficient
keyword count, invalid/empty initial keywords, authentication failure or malformed
initial response does not trigger refinement. Successful captions/descriptions
are never regenerated by this step. Standalone `generate_metadata` callers can
opt in with `keyword_retry=True`; its default remains false for compatibility.

Refinement errors preserve the earlier keywords and explain the failure. A
remaining shortfall is `needs_review`; the image may not support the requested
number of distinct concepts. No filler is inserted. Exact duplicates use
normalized whitespace and casefolding, preserving first spelling and order.
Semantic duplicates and factual relevance still need human review.

The original `generate_content(uploaded_image)` returns the same
`(caption, description)` tuple. The agent uses the richer `generate_metadata`
interface. Invalid fields are marked failed without losing other valid fields.
Malformed JSON or blocked/truncated responses are not salvaged.

### Status meanings

- **completed**: the task's operation finished; generated fields passed structural checks.
- **needs_review**: keyword/description targets were not met, optional keyword
  refinement failed, or measured resolution is below the configured minimum.
- **failed**: a required operation failed or an assessment was unavailable.
- **skipped**: a task was not executed because an earlier prerequisite, such as
  valid image data, was missing.

The overall run can additionally be **partial** when successful results coexist
with failures. This is an aggregate outcome, not a task status. The summary lists
the requested goal, planned tasks, each status group, reasons and suggested next
steps. Execution completion is separate from image suitability or acceptance.

An export can succeed even if it contains a failure report. Overall status is
not counted as success merely because the export task succeeded.
Statuses describe generation; user edits are separately recorded, not automatically
revalidated by AI.

## Stock Readiness: what it establishes

The separate **Stock Readiness** section compares the validated image's original
EXIF-oriented dimensions, before any preview/API resizing. Megapixels are
`(width * height) / 1_000_000`; the comparison uses the unrounded value.
The Shutterstock-oriented **preliminary configured default is 4 MP**. Change it
under Optional preferences → Stock resolution minimum (MP).

- Below the configured minimum: a clear warning says the image is not ready at
  its current dimensions, and the readiness task is `needs_review`.
- At or above the minimum: only **Resolution check passed** is reported.
- Missing/corrupt images or invalid measurement/configuration: the assessment is
  unavailable with an explanation, never marked passed.

No upscaling is performed. Use a higher-resolution original or reshoot when
appropriate; upscaling does not guarantee acceptance. Good blur/contrast scores
cannot override this resolution result. Neither check establishes compliance
with licensing, intellectual-property rules, content policies, other file
requirements or overall image quality. Platform policies can change; check the
platform's current contributor guidance separately. This is not an official
Shutterstock assessment or submission file.

## Quality metrics and limitations

The module checks real decoded content, not just the extension, and rejects
corrupt files, animation, unsupported formats and oversized images. EXIF
orientation is applied; transparency is composited onto white.

- Brightness: mean grayscale intensity, 0–255.
- Contrast: population standard deviation of grayscale intensity.
- Blur indicator: population variance of the four-neighbour Laplacian
  (up + down + left + right − 4 × centre), excluding the outer border.
- No resizing for metrics; no colour-profile correction. Tiny images below
  3 pixels on either axis have no blur score.

Flags use illustrative thresholds: brightness below 40 or above 215, contrast
below 20, and Laplacian variance below 100. They are **uncalibrated heuristics**.
Texture, noise, resolution, intentional blur and smooth backgrounds affect scores.
These are not aesthetic verdicts or stock-platform acceptance guarantees.

Grounding instructions discourage invented identities, locations, brands and
irrelevant keywords but cannot guarantee factual accuracy. Always review output.
Requested keyword counts are not filled with unsupported terms. Description
length is a target; insufficient visible detail may produce a shorter result.

CSV is a two-column field/value report, not a Shutterstock submission template.
Text starting like a spreadsheet formula is prefixed with an apostrophe. This
may visibly alter that cell when imported. TXT preserves ordinary text.
Downloads include edited metadata, image properties, quality values, warnings and
task statuses where available, plus resolution-readiness results and execution
summary when assessed. Every report includes an acceptance disclaimer. Missing
optional metadata is tolerated. Keyword exports remove empty/exact duplicate
entries; generated counts remain historical and current edited counts are separate.

## Tests

From the project directory:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use generated images and temporary files, with mocked Gemini clients and
configuration. They do not use your API key, read your `.env`, or make live
Gemini calls. Coverage includes validation, measurable quality checks, structured
responses, bounded retries, allowlist enforcement, selective tools, partial
failure, exports, missing-key UI behavior and persistence of edited results.
Readiness tests cover below/exactly-at/above-threshold images, configurable limits
and invalid input. Keyword tests cover normalization, one-refinement limits,
transient-call budgets, retained results and malformed refinement responses.

Streamlit AppTest checks the real app and widgets headlessly with a mocked
upload. It is not browser-based visual testing. A successful local server
startup does not verify Google authentication, billing, quota or model quality.

## Troubleshooting

- **Missing key:** configure `GEMINI_API_KEY` privately; local quality tools still work.
- **Authentication / permission:** verify the account key and model access.
- **Quota / rate limit:** wait and check your account quota; repeated runs may cost money.
- **Invalid response:** successful independent fields remain; review and rerun only what you need.
- **Invalid image:** use intact JPEG/PNG below the limits; don't overwrite your original.
- **Port occupied:** append `--server.port 8502`.
- **Virtual environment launcher error:** try `.\venv\Scripts\python.exe -m streamlit`;
  do not delete the environment without diagnosing it.

## Project and author

Created by **Rashed Ul Islam** — [@rashed4t4y](https://github.com/rashed4t4y).
Repository: [PhotoPilot-AI](https://github.com/rashed4t4y/PhotoPilot-AI).
Issues: [GitHub issues](https://github.com/rashed4t4y/PhotoPilot-AI/issues).

Built with Streamlit, Google Gemini, Pillow, NumPy and Pydantic.
The earlier README described an MIT license, but this checkout has no LICENSE
file; consult the owner before relying on a licensing grant.

### History

- Original release: image upload, Gemini captions/descriptions, TXT download.
- Agent MVP: bounded task selection, local quality checks, keywords, editable
  results, partial failures, CSV export and offline regression tests.

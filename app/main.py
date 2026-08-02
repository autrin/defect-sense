"""FastAPI web demo: upload a product photo, get back an inspection report.

Run:
    export DEFECT_SENSE_CKPT=results/Patchcore/MVTecAD/bottle/v5/weights/lightning/model.ckpt
    export DEFECT_SENSE_CATEGORY=bottle        # default category for the checkpoint
    uvicorn app.main:app --reload

Without a checkpoint the app falls back to VLM-only mode (every upload goes
straight to the VLM) so the demo works with just Ollama running.
"""

import base64
import html
import io
import os
from functools import lru_cache

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from defect_sense.pipeline.two_stage import Detection, TwoStagePipeline
from defect_sense.taxonomy import ALL_CATEGORIES, defect_types_for
from defect_sense.viz import draw_regions, overlay_heatmap
from defect_sense.vlm.adjudicator import VLMAdjudicator
from defect_sense.vlm.client import DEFAULT_MODEL, OllamaClient

app = FastAPI(title="defect-sense", version="0.2.0")

CKPT = os.environ.get("DEFECT_SENSE_CKPT")
CKPT_CATEGORY = os.environ.get("DEFECT_SENSE_CATEGORY", "bottle")
VLM_MODEL = os.environ.get("DEFECT_SENSE_VLM", DEFAULT_MODEL)
THRESHOLD = float(os.environ.get("DEFECT_SENSE_THRESHOLD", "0.5"))
DATASET_ROOT = os.environ.get("DEFECT_SENSE_DATA", "./datasets/MVTecAD")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class _AlwaysFlag:
    def predict(self, image: Image.Image) -> Detection:
        if image.width < 1 or image.height < 1:
            raise ValueError("Inspection image must not be empty")
        return Detection(score=1.0, anomaly_map=None)


@lru_cache(maxsize=4)
def get_pipeline(category: str) -> TwoStagePipeline:
    if CKPT and category == CKPT_CATEGORY:
        from defect_sense.detectors import AnomalibDetector

        detector, threshold = AnomalibDetector(ckpt_path=CKPT), THRESHOLD
    else:
        detector, threshold = _AlwaysFlag(), 0.0  # VLM-only fallback
    return TwoStagePipeline(
        detector=detector,
        adjudicator=VLMAdjudicator(
            client=OllamaClient(model=VLM_MODEL),
            category=category,
            defect_types=defect_types_for(category, DATASET_ROOT),
        ),
        threshold=threshold,
    )


@app.post("/inspect")
async def inspect(file: UploadFile = File(...), category: str = Form("bottle")):
    if category not in ALL_CATEGORIES:
        raise HTTPException(
            400, f"Unknown category {category!r}. Known: {ALL_CATEGORIES}"
        )
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(415, "Upload must be an image")
    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image exceeds the 10 MB upload limit")
        image = Image.open(io.BytesIO(contents))
        image.load()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Not a readable image: {e}") from e

    try:
        result = get_pipeline(category).inspect(image)
    except ConnectionError as e:
        raise HTTPException(503, str(e)) from e

    annotated = image.convert("RGB")
    if result.detection is not None and result.detection.anomaly_map is not None:
        annotated = overlay_heatmap(annotated, result.detection.anomaly_map)
    if result.regions:
        annotated = draw_regions(annotated, result.regions)
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")

    payload = result.to_dict()
    payload["annotated_png_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")
    payload["mode"] = (
        "two-stage" if (CKPT and category == CKPT_CATEGORY) else "vlm-only"
    )
    return payload


@app.get("/healthz")
async def health():
    return {
        "status": "ok",
        "mode": "two-stage" if CKPT else "vlm-only",
        "vlm_model": VLM_MODEL,
        "checkpoint_category": CKPT_CATEGORY if CKPT else None,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    options = "".join(
        f'<option value="{html.escape(c)}"{" selected" if c == CKPT_CATEGORY else ""}>'
        f'{html.escape(c.replace("_", " ").title())}</option>'
        for c in ALL_CATEGORIES
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Defect Sense | Inspection Console</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Two-stage industrial visual inspection console">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    :root {{ --ink: #17211b; --muted: #68736c; --line: #d9ded9; --paper: #f4f6f2;
        --surface: #fff; --signal: #d7ff46; --danger: #c5332a; --success: #16794b; --warning: #a35d00; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: var(--ink); background: var(--paper); font-family: Manrope, sans-serif; }}
    header {{ height: 68px; padding: 0 clamp(20px, 4vw, 64px); display: flex; align-items: center;
        justify-content: space-between; border-bottom: 1px solid var(--line); background: var(--surface); }}
    .brand {{ display: flex; align-items: center; gap: 12px; font-weight: 700; }}
    .mark {{ width: 30px; height: 30px; display: grid; place-items: center; background: var(--ink);
        color: var(--signal); font: 500 15px 'IBM Plex Mono', monospace; }}
    .status {{ display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 13px; }}
    .status::before {{ content: ''; width: 8px; height: 8px; border-radius: 50%; background: var(--success); }}
    main {{ min-height: calc(100vh - 68px); display: grid; grid-template-columns: minmax(320px, .85fr) minmax(420px, 1.15fr); }}
    .controls, .result {{ padding: clamp(28px, 5vw, 72px); }}
    .controls {{ border-right: 1px solid var(--line); background: var(--surface); }}
    .eyebrow {{ margin: 0 0 12px; color: var(--muted); font: 500 12px 'IBM Plex Mono', monospace;
        text-transform: uppercase; letter-spacing: 0; }}
    h1 {{ max-width: 580px; margin: 0 0 38px; font-size: clamp(32px, 4vw, 58px); line-height: 1.04; letter-spacing: 0; }}
    label {{ display: block; margin: 0 0 8px; font-size: 13px; font-weight: 600; }}
    select {{ width: 100%; height: 46px; margin-bottom: 22px; padding: 0 12px; border: 1px solid var(--line);
        border-radius: 4px; background: var(--surface); color: var(--ink); font: inherit; }}
    .dropzone {{ min-height: 210px; padding: 24px; display: grid; place-items: center; text-align: center;
        border: 1px dashed #9ba49e; border-radius: 6px; background: #fafbf9; cursor: pointer; overflow: hidden; }}
    .dropzone:hover, .dropzone.drag {{ border-color: var(--ink); background: #f2f5ed; }}
    .dropzone img {{ width: 100%; max-height: 300px; object-fit: contain; }}
    .drop-title {{ margin: 0 0 6px; font-weight: 600; }}
    .drop-meta {{ margin: 0; color: var(--muted); font-size: 13px; }}
    input[type=file] {{ position: absolute; width: 1px; height: 1px; opacity: 0; }}
    button {{ width: 100%; height: 50px; margin-top: 18px; border: 1px solid var(--ink); border-radius: 4px;
        background: var(--ink); color: white; font: 600 14px Manrope, sans-serif; cursor: pointer; }}
    button:hover {{ background: #28372e; }} button:disabled {{ opacity: .55; cursor: wait; }}
    .result {{ display: flex; flex-direction: column; background-image: linear-gradient(var(--line) 1px, transparent 1px),
        linear-gradient(90deg, var(--line) 1px, transparent 1px); background-size: 28px 28px; }}
    .result-shell {{ flex: 1; min-height: 420px; padding: clamp(24px, 4vw, 48px); background: rgba(255,255,255,.96);
        border: 1px solid var(--line); border-radius: 6px; }}
    .empty {{ height: 100%; min-height: 340px; display: grid; place-items: center; color: var(--muted); text-align: center; }}
    .empty-code {{ display: block; margin-bottom: 12px; color: var(--ink); font: 500 34px 'IBM Plex Mono', monospace; }}
    .verdict-row {{ display: flex; justify-content: space-between; gap: 20px; align-items: start; margin-bottom: 28px; }}
    .verdict {{ margin: 0; font-size: clamp(26px, 3vw, 42px); line-height: 1; text-transform: capitalize; }}
    .badge {{ padding: 7px 10px; border-radius: 3px; font: 500 11px 'IBM Plex Mono', monospace; text-transform: uppercase; }}
    .pass {{ color: var(--success); }} .defect {{ color: var(--danger); }} .false_alarm {{ color: var(--warning); }}
    .badge.pass {{ background: #ddf5e8; }} .badge.defect {{ background: #fbe4e1; }} .badge.false_alarm {{ background: #fff0d5; }}
    .annotated {{ width: 100%; max-height: 440px; object-fit: contain; background: #eef1ed; border-radius: 4px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); margin: 26px 0; border: 1px solid var(--line); }}
    .metric {{ padding: 14px; }} .metric + .metric {{ border-left: 1px solid var(--line); }}
    .metric span {{ display: block; color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    .metric strong {{ display: block; margin-top: 5px; font: 500 16px 'IBM Plex Mono', monospace; }}
    .report {{ margin: 0; line-height: 1.65; }}
    .error {{ padding: 14px; border-left: 3px solid var(--danger); background: #fbe4e1; color: #7b201b; }}
    .loader {{ width: 40px; height: 40px; border: 3px solid var(--line); border-top-color: var(--ink); border-radius: 50%;
        animation: spin .8s linear infinite; }} @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    @media (max-width: 860px) {{ main {{ grid-template-columns: 1fr; }} .controls {{ border-right: 0; border-bottom: 1px solid var(--line); }}
        h1 {{ font-size: 38px; }} .result {{ min-height: 560px; }} }}
    @media (max-width: 480px) {{ header {{ padding: 0 18px; }} .controls, .result {{ padding: 24px 18px; }}
        .metrics {{ grid-template-columns: 1fr; }} .metric + .metric {{ border-left: 0; border-top: 1px solid var(--line); }} }}
</style></head><body>
<header><div class="brand"><span class="mark">DS</span><span>Defect Sense</span></div>
<div class="status">Local inference · {html.escape(VLM_MODEL)}</div></header>
<main><section class="controls"><p class="eyebrow">Inspection console / 01</p><h1>Visual quality control, with evidence.</h1>
<form id="f"><label for="category">Product category</label><select id="category" name="category">{options}</select>
<label for="file">Inspection image</label><label class="dropzone" id="dropzone" for="file">
<span id="drop-copy"><p class="drop-title">Select an image</p><p class="drop-meta">PNG, JPG, TIFF · 10 MB maximum</p></span>
</label><input id="file" type="file" name="file" accept="image/*" required>
<button id="submit" type="submit">Run inspection</button></form></section>
<section class="result"><p class="eyebrow">Analysis / 02</p><div class="result-shell" id="out">
<div class="empty"><div><span class="empty-code">READY</span><span>Awaiting inspection image</span></div></div>
</div></section></main>
<script>
const form = document.getElementById('f');
const fileInput = document.getElementById('file');
const dropzone = document.getElementById('dropzone');
const out = document.getElementById('out');
const submit = document.getElementById('submit');
function preview(file) {{
    if (!file) return;
    const image = document.createElement('img'); image.alt = 'Selected inspection image'; image.src = URL.createObjectURL(file);
    dropzone.replaceChildren(image);
}}
fileInput.addEventListener('change', () => preview(fileInput.files[0]));
['dragenter', 'dragover'].forEach(name => dropzone.addEventListener(name, e => {{ e.preventDefault(); dropzone.classList.add('drag'); }}));
['dragleave', 'drop'].forEach(name => dropzone.addEventListener(name, e => {{ e.preventDefault(); dropzone.classList.remove('drag'); }}));
dropzone.addEventListener('drop', e => {{ if (e.dataTransfer.files.length) {{ fileInput.files = e.dataTransfer.files; preview(fileInput.files[0]); }} }});
function metric(label, value) {{
    const node = document.createElement('div'); node.className = 'metric';
    const name = document.createElement('span'); name.textContent = label;
    const number = document.createElement('strong'); number.textContent = value;
    node.append(name, number); return node;
}}
function renderResult(data) {{
    const row = document.createElement('div'); row.className = 'verdict-row';
    const title = document.createElement('h2'); title.className = 'verdict ' + data.verdict; title.textContent = data.defect_type || data.verdict.replace('_', ' ');
    const badge = document.createElement('span'); badge.className = 'badge ' + data.verdict; badge.textContent = data.verdict.replace('_', ' ');
    row.append(title, badge);
    const image = document.createElement('img'); image.className = 'annotated'; image.alt = 'Annotated inspection result';
    image.src = 'data:image/png;base64,' + data.annotated_png_base64;
    const metrics = document.createElement('div'); metrics.className = 'metrics';
    metrics.append(metric('Confidence', data.confidence == null ? 'N/A' : Math.round(data.confidence * 100) + '%'),
        metric('Stage 1', data.stage1_seconds.toFixed(3) + 's'), metric('Stage 2', data.stage2_seconds.toFixed(3) + 's'));
    const report = document.createElement('p'); report.className = 'report'; report.textContent = data.report || 'No report returned.';
    out.replaceChildren(row, image, metrics, report);
}}
form.addEventListener('submit', async (e) => {{
  e.preventDefault();
    submit.disabled = true; submit.textContent = 'Inspecting…';
    out.innerHTML = '<div class="empty"><div><div class="loader"></div></div></div>';
  const resp = await fetch('/inspect', {{ method: 'POST', body: new FormData(e.target) }});
  const data = await resp.json();
    if (!resp.ok) {{ const error = document.createElement('div'); error.className = 'error'; error.textContent = data.detail || 'Inspection failed'; out.replaceChildren(error); }}
    else {{ renderResult(data); }}
    submit.disabled = false; submit.textContent = 'Run inspection';
}});
</script>
</body></html>"""

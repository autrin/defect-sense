"""FastAPI web demo: upload a product photo, get back an inspection report.

Run:
    export DEFECT_SENSE_CKPT=results/Patchcore/MVTecAD/bottle/v5/weights/lightning/model.ckpt
    export DEFECT_SENSE_CATEGORY=bottle        # default category for the checkpoint
    uvicorn app.main:app --reload

Without a checkpoint the app falls back to VLM-only mode (every upload goes
straight to the VLM) so the demo works with just Ollama running.
"""
import base64
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


class _AlwaysFlag:
    def predict(self, image) -> Detection:
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
        raise HTTPException(400, f"Unknown category {category!r}. Known: {ALL_CATEGORIES}")
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image.load()
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
    payload["mode"] = "two-stage" if (CKPT and category == CKPT_CATEGORY) else "vlm-only"
    return payload


@app.get("/", response_class=HTMLResponse)
async def index():
    options = "".join(
        f'<option value="{c}"{" selected" if c == CKPT_CATEGORY else ""}>{c}</option>'
        for c in ALL_CATEGORIES
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>defect-sense</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }}
  .row {{ display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }}
  img {{ max-width: 100%; border-radius: 8px; margin-top: 1rem; }}
  pre {{ background: #f4f4f4; padding: 1rem; border-radius: 8px; overflow-x: auto; white-space: pre-wrap; }}
  .verdict {{ font-size: 1.3rem; font-weight: 700; margin-top: 1rem; }}
  .pass {{ color: #1a7f37; }} .defect {{ color: #cf222e; }} .false_alarm {{ color: #9a6700; }}
  button {{ padding: .5rem 1.2rem; }}
</style></head><body>
<h1>defect-sense</h1>
<p>Upload a product photo. Stage 1 (anomaly detector) triages it; flagged images
are adjudicated by a local VLM that names the defect and writes a report.</p>
<form id="f" class="row">
  <input type="file" name="file" accept="image/*" required>
  <select name="category">{options}</select>
  <button type="submit">Inspect</button>
</form>
<div id="out"></div>
<script>
document.getElementById('f').addEventListener('submit', async (e) => {{
  e.preventDefault();
  const out = document.getElementById('out');
  out.innerHTML = '<p>Inspecting… (VLM adjudication can take ~10-60s)</p>';
  const resp = await fetch('/inspect', {{ method: 'POST', body: new FormData(e.target) }});
  const data = await resp.json();
  if (!resp.ok) {{ out.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>'; return; }}
  const img = data.annotated_png_base64;
  delete data.annotated_png_base64;
  out.innerHTML =
    '<div class="verdict ' + data.verdict + '">' + data.verdict.toUpperCase() +
    (data.defect_type ? ' — ' + data.defect_type : '') + '</div>' +
    (data.report ? '<p>' + data.report + '</p>' : '') +
    '<img src="data:image/png;base64,' + img + '">' +
    '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
}});
</script>
</body></html>"""

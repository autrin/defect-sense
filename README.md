# defect-sense

Industrial defect detection combining fast anomaly detection with vision-language reasoning.

A two-stage pipeline: a lightweight anomaly detector (PatchCore / EfficientAD) flags candidate defects, and a vision-language model (Qwen3-VL) classifies the defect type and generates a natural-language inspection report with grounded bounding boxes.

Benchmarked on the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset.

## Status

🚧 In active development.

- [x] Repo scaffolded
- [x] Dataset download via Hugging Face mirror (bypasses broken upstream URL)
- [x] PatchCore baseline on `bottle` category
- [x] PatchCore baseline across all 15 MVTec AD categories
- [x] VLM adjudication stage (Qwen3-VL 8B via Ollama): heatmap → region proposals → structured JSON verdict
- [x] Evaluation harness (triage precision/recall, defect-type accuracy, latency, VLM call rate)
- [x] Web UI (FastAPI + minimal frontend)
- [x] Unit test suite + CI (pipeline logic runs without GPU/Ollama)
- [ ] EfficientAD comparison (`benchmark.py --model efficient_ad` — run pending)
- [ ] Adjudication results across all 15 categories (run pending)
- [ ] Deployed demo

## Why this project

Most defect detection systems either use classical CV (fast, opaque, brittle) or throw a VLM at the whole problem (slow, expensive, hallucination-prone). The realistic production pattern is neither: a cheap specialist model handles the volume, and a VLM adjudicates edge cases and produces human-readable output.

This repo implements and measures that pattern end-to-end.

## Architecture

```
Input image
    |
    v
+------------------------+
|  PatchCore /           |  --> anomaly score + heatmap
|  EfficientAD           |
+------------------------+
    |
    v
+------------------------+
|  Threshold / triage    |  --> clean images pass through
+------------------------+
    |
    v
+------------------------+
|  Qwen3-VL 8B          |  --> defect class
|  (adjudication)        |      + bounding box
|                        |      + written report
+------------------------+
```

## Setup

Requires Python 3.11 (3.12 also works; 3.13 wheels for some ML deps can still be spotty) and, for GPU acceleration, an NVIDIA GPU with a driver that supports CUDA 12.1+.

```bash
git clone https://github.com/autrin/defect-sense.git
cd defect-sense

# Create venv with Python 3.11 explicitly
py -3.11 -m venv .venv                              # Windows
# python3.11 -m venv .venv                          # Linux / macOS

# Activate
source .venv/Scripts/activate                       # Windows (Git Bash)
# .venv\Scripts\activate                            # Windows (cmd / PowerShell)
# source .venv/bin/activate                         # Linux / macOS
```

**Install PyTorch with CUDA first, then verify before installing anything else.**
This order prevents accidentally ending up with a CPU-only build.

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is actually enabled — must print True and your GPU name
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If the check prints `False`, stop and fix before proceeding — reinstalling later on top of a broken torch is worse than starting clean.

Then install the rest:

```bash
python -m pip install -r requirements.txt
```

Install Ollama and pull the VLM (for the Stage 2 adjudication work — not needed for the Stage 1 baseline):

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh
# Windows: https://ollama.com/download

ollama pull qwen3-vl:8b
```

## Dataset

Uses the [MVTec Anomaly Detection dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) (CC BY-NC-SA 4.0, free for non-commercial research).

Anomalib's built-in downloader has been unreliable since late 2025 (upstream 404), so this repo pulls from the Hugging Face mirror `TheoM55/mvtec_all_objects_split` and lays the files out in the folder structure anomalib expects:

```bash
python scripts/download_mvtec.py
```

Downloads all 15 categories to `./datasets/MVTecAD/`.

## Usage

Install the package first (from the repo root, after the setup above):

```bash
python -m pip install -e .
```

### Stage 1: benchmark an anomaly detector

```bash
python benchmark.py                                  # PatchCore, all 15 categories
python benchmark.py bottle hazelnut                  # subset
python benchmark.py --model efficient_ad             # EfficientAD comparison
```

Writes `results/<model>_baseline.csv` and leaves trained checkpoints under
`results/<Model>/MVTecAD/<category>/<version>/weights/lightning/model.ckpt`.

### Stage 2: evaluate VLM adjudication on a category

Requires Ollama running with the VLM pulled (`ollama pull qwen3-vl:8b`).

```bash
# Full two-stage pipeline: detector triages, VLM adjudicates flagged images
python scripts/eval_adjudication.py bottle \
    --ckpt results/Patchcore/MVTecAD/bottle/v5/weights/lightning/model.ckpt

# Ablation: send every image straight to the VLM (no stage 1)
python scripts/eval_adjudication.py bottle --mode vlm-only --limit 5
```

Reports triage precision/recall, defect-type classification accuracy against
MVTec's ground-truth labels, per-stage latency, and the VLM call rate (the
fraction of images that pay stage 2 cost). Outputs land in
`results/adjudication/`.

### Inspect a single image

```bash
python scripts/inspect_image.py datasets/MVTecAD/bottle/test/contamination/001.png \
    --category bottle \
    --ckpt results/Patchcore/MVTecAD/bottle/v5/weights/lightning/model.ckpt
```

Prints the JSON inspection result and saves an annotated heatmap+boxes image.

### Serve the web demo

```bash
export DEFECT_SENSE_CKPT=results/Patchcore/MVTecAD/bottle/v5/weights/lightning/model.ckpt
export DEFECT_SENSE_CATEGORY=bottle
uvicorn app.main:app --reload
```

Open http://localhost:8000, upload an image, get back a verdict, defect type,
annotated image, and written report. Without a checkpoint the app falls back
to VLM-only mode, so the demo also works with just Ollama.

### Run the tests

The unit tests cover region extraction, VLM response parsing, pipeline triage
logic, and the eval math — no GPU, dataset, or Ollama needed:

```bash
pytest
```

## Results

PatchCore baseline across all 15 MVTec AD categories, on an RTX 4070 Laptop (8GB VRAM).
Mean image AUROC: **0.982**. Mean pixel AUROC: **0.980**. Full benchmark runs in ~75 minutes.

| Category    | Image AUROC | Image F1 | Pixel AUROC | Pixel F1 |
|-------------|-------------|----------|-------------|----------|
| bottle      | 1.000       | 0.992    | 0.986       | 0.727    |
| cable       | 0.983       | 0.967    | 0.985       | 0.638    |
| capsule     | 0.992       | 0.977    | 0.990       | 0.517    |
| carpet      | 0.986       | 0.972    | 0.991       | 0.606    |
| grid        | 0.986       | 0.965    | 0.982       | 0.389    |
| hazelnut    | 1.000       | 0.993    | 0.988       | 0.631    |
| leather     | 1.000       | 0.995    | 0.992       | 0.438    |
| metal_nut   | 0.999       | 0.989    | 0.987       | 0.841    |
| pill        | 0.947       | 0.953    | 0.981       | 0.720    |
| screw       | 0.964       | 0.951    | 0.989       | 0.370    |
| tile        | 1.000       | 0.994    | 0.956       | 0.620    |
| toothbrush  | 0.908       | 0.936    | 0.989       | 0.596    |
| transistor  | 0.995       | 0.962    | 0.973       | 0.608    |
| wood        | 0.986       | 0.958    | 0.932       | 0.467    |
| zipper      | 0.979       | 0.979    | 0.981       | 0.543    |
| **Mean**    | **0.982**   | **0.972**| **0.980**   | **0.581**|

**Takeaways:**
- Image-level classification is near-saturated (12/15 categories ≥ 0.98 AUROC). PatchCore is a strong baseline out of the box.
- Pixel F1 ranges from 0.37 (`screw`, `grid`) to 0.84 (`metal_nut`), correlated with defect size and structural complexity. Thin structures with small defects are where mask coarseness shows.
- Weakest image-level results (`toothbrush`, `pill`, `screw`) map to known-hard MVTec categories — small test sets, high intra-class variability, or subtle defect types.
- These localization gaps are the motivation for the Stage 2 VLM adjudicator.

## Project structure

```
defect-sense/
├── benchmark.py                  # Stage 1 benchmark CLI (PatchCore / EfficientAD)
├── datasets/                     # MVTec AD data (gitignored)
├── results/                      # benchmark CSVs + adjudication eval outputs
├── scripts/
│   ├── download_mvtec.py         # dataset download via HF mirror
│   ├── eval_adjudication.py      # Stage 2 eval: two-stage vs vlm-only
│   └── inspect_image.py          # single-image inspection CLI
├── src/defect_sense/
│   ├── detectors/                # anomalib (PatchCore/EfficientAD) wrapper
│   ├── vlm/                      # Ollama client + VLM adjudicator (prompt, schema, parsing)
│   ├── pipeline/                 # two-stage triage → adjudication orchestration
│   ├── eval/                     # eval harness: triage, typing accuracy, latency, cost
│   ├── regions.py                # anomaly heatmap → bounding-box proposals
│   ├── taxonomy.py               # per-category defect types
│   └── viz.py                    # heatmap overlays, box drawing (PIL-only)
├── app/                          # FastAPI web demo
└── tests/                        # unit tests (run without GPU/Ollama)
```

### How Stage 2 works

1. The detector's anomaly heatmap is thresholded relative to its own dynamic
   range; connected components become ranked bounding-box **region proposals**.
2. The VLM receives the original photo *and* a heatmap overlay with numbered
   boxes, plus the anomaly score and the category's closed defect taxonomy.
3. Ollama's structured-output mode constrains the reply to a JSON schema
   (`is_defect`, `defect_type` enum, `bbox`, `confidence`, `report`); parsing
   is defensive and unparseable replies fail safe as "unknown, keep flagged".
4. The VLM can overrule stage 1 (`false_alarm`), which is scored explicitly
   by the eval harness — that's the precision the two-stage design buys.

## Stack

- **Anomaly detection:** [anomalib](https://github.com/openvinotoolkit/anomalib) (PatchCore, EfficientAD)
- **VLM:** VLM: Qwen3-VL 8B (primary); qwen3.5:9b and gemma4:12b benchmarked as alternatives. All served locally via [Ollama](https://ollama.com)
- **Backend:** FastAPI
- **Evaluation:** custom harness, results in CSV + Markdown reports

## References

- Roth et al., [Towards Total Recall in Industrial Anomaly Detection (PatchCore)](https://arxiv.org/abs/2106.08265), CVPR 2022
- Batzner et al., [EfficientAD](https://arxiv.org/abs/2303.14535), WACV 2024
- Qwen3-VL (Alibaba, 2025) — [model card](https://ollama.com/library/qwen3-vl)
- Bergmann et al., [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad), CVPR 2019

## License

MIT — see `LICENSE`.

Author: Autrin Hakimi
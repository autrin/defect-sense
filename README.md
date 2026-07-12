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
- [ ] EfficientAD comparison
- [ ] VLM adjudication stage (Qwen3-VL 8B via Ollama)
- [ ] Evaluation harness (precision, recall, latency, cost per inspection)
- [ ] Web UI (FastAPI + minimal frontend)
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
python scripts/download_bottle.py
```

Downloads to `./datasets/MVTecAD/bottle/`. Additional per-category downloaders will be added as the benchmark expands.

## Usage

### Train a PatchCore baseline on the `bottle` category

```bash
python train_patchcore.py
```

Category is currently hardcoded in the script. A `--category` flag will be added when the full benchmark loop lands.

### Run the full benchmark (planned)

```bash
python benchmark.py --model patchcore --output results/patchcore.csv
```

### Serve the web demo (planned)

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000, upload an image, get back a defect report.

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

Target layout — directories are created as the phases in **Status** are completed. Current state has only `scripts/`, `results/`, `datasets/`, and the two top-level scripts.

```
defect-sense/
├── datasets/             # MVTec AD data (gitignored)
├── results/              # benchmark outputs
├── scripts/              # data download, batch runs
├── src/
│   ├── detectors/        # PatchCore, EfficientAD wrappers
│   ├── vlm/              # Qwen3-VL 8B client via Ollama
│   ├── pipeline/         # two-stage adjudication logic
│   └── eval/             # metrics, harness
├── app/                  # FastAPI web demo
├── notebooks/            # exploration, visualizations
└── tests/
```

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
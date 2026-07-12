# defect-sense

Industrial defect detection combining fast anomaly detection with vision-language reasoning.

A two-stage pipeline: a lightweight anomaly detector (PatchCore / EfficientAD) flags candidate defects, and a vision-language model (Qwen2.5-VL) classifies the defect type and generates a natural-language inspection report with grounded bounding boxes.

Benchmarked on the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset.

## Status

🚧 In active development. Current phase: baseline anomaly detection on MVTec AD.

- [x] Repo scaffolded
- [x] Dataset download via Hugging Face mirror (bypasses broken upstream URL)
- [ ] PatchCore baseline on `bottle` category
- [ ] PatchCore baseline across all 15 MVTec AD categories
- [ ] EfficientAD comparison
- [ ] VLM adjudication stage (Qwen2.5-VL via Ollama)
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
|  Qwen2.5-VL            |  --> defect class
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

ollama pull qwen2.5vl:7b
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

_Populated after first full run._

| Category | Model      | Image AUROC | Pixel AUROC | Latency (ms) |
|----------|------------|-------------|-------------|--------------|
| bottle   | PatchCore  | –           | –           | –            |
| hazelnut | PatchCore  | –           | –           | –            |
| ...      | ...        | ...         | ...         | ...          |

## Project structure

Target layout — directories are created as the phases in **Status** are completed. Current state has only `scripts/`, `results/`, `datasets/`, and the two top-level scripts.

```
defect-sense/
├── datasets/             # MVTec AD data (gitignored)
├── models/               # trained weights (gitignored)
├── results/              # benchmark outputs
├── scripts/              # data download, batch runs
├── src/
│   ├── detectors/        # PatchCore, EfficientAD wrappers
│   ├── vlm/              # Qwen2.5-VL client via Ollama
│   ├── pipeline/         # two-stage adjudication logic
│   └── eval/             # metrics, harness
├── app/                  # FastAPI web demo
├── notebooks/            # exploration, visualizations
└── tests/
```

## Stack

- **Anomaly detection:** [anomalib](https://github.com/openvinotoolkit/anomalib) (PatchCore, EfficientAD)
- **VLM:** Qwen2.5-VL 7B served locally via [Ollama](https://ollama.com)
- **Backend:** FastAPI
- **Evaluation:** custom harness, results in CSV + Markdown reports

## References

- Roth et al., [Towards Total Recall in Industrial Anomaly Detection (PatchCore)](https://arxiv.org/abs/2106.08265), CVPR 2022
- Batzner et al., [EfficientAD](https://arxiv.org/abs/2303.14535), WACV 2024
- Wang et al., [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
- Bergmann et al., [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad), CVPR 2019

## License

MIT — see `LICENSE`.

Author: Autrin Hakimi
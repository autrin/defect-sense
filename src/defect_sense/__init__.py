"""defect-sense: two-stage industrial defect detection.

Stage 1: a fast anomaly detector (PatchCore / EfficientAD via anomalib)
scores every image and localizes suspect regions.
Stage 2: a vision-language model (Qwen3-VL via Ollama) adjudicates flagged
images — classifying the defect type and writing an inspection report.
"""

__version__ = "0.2.0"

"""MVTec AD defect taxonomy.

The VLM adjudicator classifies defects into a closed set of types per
category. When the dataset is on disk we derive the set from the test
folder names (the ground truth); otherwise we fall back to the canonical
MVTec AD taxonomy below.
"""
from pathlib import Path

# Canonical defect types per MVTec AD category (test subfolders minus "good").
MVTEC_DEFECT_TYPES: dict[str, list[str]] = {
    "bottle": ["broken_large", "broken_small", "contamination"],
    "cable": [
        "bent_wire", "cable_swap", "combined", "cut_inner_insulation",
        "cut_outer_insulation", "missing_cable", "missing_wire", "poke_insulation",
    ],
    "capsule": ["crack", "faulty_imprint", "poke", "scratch", "squeeze"],
    "carpet": ["color", "cut", "hole", "metal_contamination", "thread"],
    "grid": ["bent", "broken", "glue", "metal_contamination", "thread"],
    "hazelnut": ["crack", "cut", "hole", "print"],
    "leather": ["color", "cut", "fold", "glue", "poke"],
    "metal_nut": ["bent", "color", "flip", "scratch"],
    "pill": [
        "color", "combined", "contamination", "crack",
        "faulty_imprint", "pill_type", "scratch",
    ],
    "screw": [
        "manipulated_front", "scratch_head", "scratch_neck",
        "thread_side", "thread_top",
    ],
    "tile": ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
    "toothbrush": ["defective"],
    "transistor": ["bent_lead", "cut_lead", "damaged_case", "misplaced"],
    "wood": ["color", "combined", "hole", "liquid", "scratch"],
    "zipper": [
        "broken_teeth", "combined", "fabric_border", "fabric_interior",
        "rough", "split_teeth", "squeezed_teeth",
    ],
}

ALL_CATEGORIES = sorted(MVTEC_DEFECT_TYPES)


def defect_types_for(category: str, dataset_root: str | Path | None = None) -> list[str]:
    """Defect types for a category, preferring the on-disk test folders."""
    if dataset_root is not None:
        test_dir = Path(dataset_root) / category / "test"
        if test_dir.is_dir():
            types = sorted(
                d.name for d in test_dir.iterdir()
                if d.is_dir() and d.name != "good"
            )
            if types:
                return types
    if category not in MVTEC_DEFECT_TYPES:
        raise KeyError(
            f"Unknown category {category!r} and no dataset folder to derive from. "
            f"Known: {ALL_CATEGORIES}"
        )
    return list(MVTEC_DEFECT_TYPES[category])

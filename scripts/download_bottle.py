"""Download MVTec AD categories from the HF mirror and lay them out
in the folder structure anomalib expects.

Auto-detects column names because HF mirrors vary in schema.
"""
from pathlib import Path
from shutil import copyfile
from datasets import load_dataset

CATEGORIES = ["bottle"]   # add more later: "hazelnut", "capsule", etc.
DATASET = "TheoM55/mvtec_all_objects_split"
OUTPUT_ROOT = Path("./datasets/MVTecAD")


def find_key(sample: dict, candidates: list[str]) -> str | None:
    for k in candidates:
        if k in sample:
            return k
    return None


def save_image_like(obj, out_path: Path) -> bool:
    """Save a PIL Image or copy from a file path. Returns True on success."""
    if obj is None:
        return False
    if hasattr(obj, "save"):        # PIL image
        obj.save(out_path)
        return True
    if isinstance(obj, str) and obj:
        copyfile(obj, out_path)
        return True
    return False


def dump_split(split_name: str, out_root: Path, category: str) -> None:
    print(f"\n== {split_name} ==")
    ds = load_dataset(DATASET, split=split_name)
    if len(ds) == 0:
        print("  Empty split, skipping.")
        return

    print(f"  Columns: {ds.column_names}")
    sample0 = ds[0]

    image_key = find_key(sample0, ["image", "image_path", "img", "picture"])
    defect_key = find_key(sample0, ["defect", "defect_type", "label_name", "class"])
    label_key = find_key(sample0, ["label", "is_anomaly", "anomaly"])
    mask_key = find_key(sample0, ["mask", "mask_path", "ground_truth", "gt_mask", "segmentation"])

    if image_key is None:
        raise RuntimeError(f"Can't find image column. Available: {ds.column_names}")

    is_train = split_name.endswith(".train")

    counts: dict[str, int] = {}
    masks_saved = 0
    for sample in ds:
        defect = "good" if is_train else (sample[defect_key] if defect_key else "unknown")
        counts[defect] = counts.get(defect, 0) + 1
        idx = counts[defect]

        subfolder = "train" if is_train else "test"
        img_dir = out_root / category / subfolder / defect
        img_dir.mkdir(parents=True, exist_ok=True)
        save_image_like(sample[image_key], img_dir / f"{idx:03d}.png")

        # Ground-truth masks: only for defective test samples
        if not is_train and mask_key and defect != "good":
            mask_dir = out_root / category / "ground_truth" / defect
            mask_dir.mkdir(parents=True, exist_ok=True)
            if save_image_like(sample[mask_key], mask_dir / f"{idx:03d}_mask.png"):
                masks_saved += 1

    print(f"  Wrote {len(ds)} images. Defect groups: {counts}")
    if not is_train:
        print(f"  Masks saved: {masks_saved}")


def main() -> None:
    for category in CATEGORIES:
        print(f"\n### Category: {category} ###")
        dump_split(f"{category}.train", OUTPUT_ROOT, category)
        dump_split(f"{category}.test", OUTPUT_ROOT, category)
    print(f"\nDone. Data in {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
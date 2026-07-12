"""Download MVTec AD categories from the HF mirror and lay them out
in the folder structure anomalib expects.
"""
from pathlib import Path
from datasets import load_dataset

CATEGORIES = ["bottle"]   # add more later: "hazelnut", "capsule", etc.
DATASET = "TheoM55/mvtec_all_objects_split"
OUTPUT_ROOT = Path("./datasets/MVTecAD")


def find_key(sample: dict, candidates: list[str]) -> str | None:
    """Return the first candidate key present in the sample."""
    for k in candidates:
        if k in sample:
            return k
    return None


def dump_split(split_name: str, out_root: Path, category: str) -> None:
    print(f"\n== {split_name} ==")
    ds = load_dataset(DATASET, split=split_name)

    if len(ds) == 0:
        print(f"  Empty split, skipping.")
        return

    print(f"  Columns: {ds.column_names}")
    sample0 = ds[0]

    image_key = find_key(sample0, ["image", "image_path", "img", "picture"])
    defect_key = find_key(sample0, ["defect", "defect_type", "label_name", "class"])
    label_key = find_key(sample0, ["label", "is_anomaly", "anomaly"])
    mask_key = find_key(sample0, ["mask", "ground_truth", "gt_mask", "segmentation"])

    if image_key is None:
        raise RuntimeError(f"Can't find image column. Available: {ds.column_names}")

    is_train = split_name.endswith(".train")

    counts: dict[str, int] = {}
    for sample in ds:
        img = sample[image_key]

        if is_train:
            # All training images are "good" (defect-free) by MVTec convention
            defect = "good"
        else:
            defect = sample[defect_key] if defect_key else "unknown"

        counts[defect] = counts.get(defect, 0) + 1
        idx = counts[defect]

        subfolder = "train" if is_train else "test"
        img_dir = out_root / category / subfolder / defect
        img_dir.mkdir(parents=True, exist_ok=True)
        img.save(img_dir / f"{idx:03d}.png")

        # Save ground-truth masks (test set only, defective samples only)
        if not is_train and mask_key and sample.get(mask_key) is not None:
            if not label_key or sample[label_key] == 1:
                mask_dir = out_root / category / "ground_truth" / defect
                mask_dir.mkdir(parents=True, exist_ok=True)
                sample[mask_key].save(mask_dir / f"{idx:03d}_mask.png")

    print(f"  Wrote {len(ds)} images. Defect groups: {counts}")


def main() -> None:
    for category in CATEGORIES:
        print(f"\n### Category: {category} ###")
        dump_split(f"{category}.train", OUTPUT_ROOT, category)
        dump_split(f"{category}.test", OUTPUT_ROOT, category)
    print(f"\nDone. Data in {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
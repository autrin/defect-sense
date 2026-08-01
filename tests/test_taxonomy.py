import pytest

from defect_sense.taxonomy import ALL_CATEGORIES, MVTEC_DEFECT_TYPES, defect_types_for


def test_all_15_categories_present():
    assert len(ALL_CATEGORIES) == 15
    assert "bottle" in ALL_CATEGORIES and "zipper" in ALL_CATEGORIES


def test_fallback_taxonomy():
    assert defect_types_for("bottle") == ["broken_large", "broken_small", "contamination"]


def test_derives_from_dataset_folders(tmp_path):
    test_dir = tmp_path / "bottle" / "test"
    for name in ("good", "weird_new_defect", "contamination"):
        (test_dir / name).mkdir(parents=True)
    types = defect_types_for("bottle", tmp_path)
    assert types == ["contamination", "weird_new_defect"]  # sorted, no "good"


def test_missing_dataset_falls_back(tmp_path):
    assert defect_types_for("bottle", tmp_path) == MVTEC_DEFECT_TYPES["bottle"]


def test_unknown_category_raises():
    with pytest.raises(KeyError):
        defect_types_for("flux_capacitor")

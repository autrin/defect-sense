import numpy as np

from defect_sense.regions import Region, extract_regions


def test_flat_map_yields_no_regions():
    assert extract_regions(np.zeros((64, 64))) == []
    assert extract_regions(np.full((64, 64), 3.7)) == []


def test_single_hot_blob():
    amap = np.zeros((100, 100))
    amap[20:30, 40:55] = 1.0
    regions = extract_regions(amap)
    assert len(regions) == 1
    assert regions[0].bbox == (40, 20, 55, 30)
    assert regions[0].area == 10 * 15
    assert regions[0].peak_score == 1.0


def test_regions_sorted_by_peak_and_capped():
    amap = np.zeros((100, 100))
    amap[10:15, 10:15] = 0.6   # weaker blob
    amap[60:70, 60:70] = 1.0   # stronger blob
    regions = extract_regions(amap)
    assert len(regions) == 2
    assert regions[0].bbox == (60, 60, 70, 70)  # strongest first

    regions = extract_regions(amap, max_regions=1)
    assert len(regions) == 1
    assert regions[0].peak_score == 1.0


def test_small_noise_filtered():
    amap = np.zeros((100, 100))
    amap[50, 50] = 1.0  # single pixel: below min_area at default 0.05% of 10k = 5px
    assert extract_regions(amap) == []


def test_diagonal_blobs_are_separate_components():
    # 4-connectivity: diagonal touch does not merge
    amap = np.zeros((50, 50))
    amap[10:20, 10:20] = 1.0
    amap[20:30, 20:30] = 1.0
    assert len(extract_regions(amap, min_area_frac=0.0)) == 2


def test_region_scaling():
    r = Region(bbox=(10, 20, 30, 40), area=1, peak_score=1.0, mean_score=1.0)
    assert r.scaled(2.0, 0.5).bbox == (20, 10, 60, 20)


def test_rejects_non_2d_input():
    import pytest

    with pytest.raises(ValueError):
        extract_regions(np.zeros((3, 64, 64)))

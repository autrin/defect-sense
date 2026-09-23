import pytest

from scripts.compare_saved_decisions import compare


def test_comparison_separates_helpful_and_harmful_vetoes():
    records = [
        {"true_type": "crack", "verdict": "false_alarm", "anomaly_score": "0.9"},
        {"true_type": "good", "verdict": "false_alarm", "anomaly_score": "0.8"},
        {"true_type": "crack", "verdict": "defect", "anomaly_score": "0.5"},
        {"true_type": "good", "verdict": "pass", "anomaly_score": "0.1"},
    ]
    result = compare(records, 0.5)
    assert result["saved"]["recall"] == 0.5
    assert result["detector_preserved"]["recall"] == 1.0
    assert result["detector_preserved"]["fp"] == 1
    assert result["true_defects_removed_by_stage2"] == 1
    assert result["false_alarms_removed_by_stage2"] == 1


def test_comparison_rejects_empty_input_and_nonfinite_scores():
    with pytest.raises(ValueError, match="No saved records"):
        compare([], 0.5)
    with pytest.raises(ValueError, match="finite"):
        compare([{"true_type": "good", "verdict": "pass", "anomaly_score": "nan"}], 0.5)

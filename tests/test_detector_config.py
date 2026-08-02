import pytest

from defect_sense.detectors.anomalib_detector import (
    EFFICIENT_AD_TRAINING_STEPS,
    trainer_kwargs,
)


def test_default_training_schedules_are_model_appropriate():
    assert trainer_kwargs("patchcore") == {"max_epochs": 1}
    assert trainer_kwargs("efficient_ad") == {
        "max_steps": EFFICIENT_AD_TRAINING_STEPS,
    }


def test_training_schedule_can_be_overridden():
    assert trainer_kwargs("efficient_ad", max_epochs=10) == {"max_epochs": 10}
    assert trainer_kwargs("patchcore", max_steps=500) == {"max_steps": 500}


def test_training_schedule_rejects_conflicting_limits():
    with pytest.raises(ValueError, match="only one"):
        trainer_kwargs("efficient_ad", max_epochs=10, max_steps=500)

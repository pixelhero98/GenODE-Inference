from __future__ import annotations

import pytest

from genode.gico.policy import MODEL_PAYLOAD_VERSION, require_current_gico_checkpoint_payload
from genode.gico.report_locked_test import _teacher_final_retrain_metadata
from genode.models.conditioning import FROZEN_BACKBONE_POLICY_CONTEXT_PROTOCOL


def _current_payload() -> dict[str, object]:
    return {
        "model_payload_version": MODEL_PAYLOAD_VERSION,
        "context_embedding_protocol": FROZEN_BACKBONE_POLICY_CONTEXT_PROTOCOL,
    }


def test_current_gico_payload_rejects_retired_final_teacher_alias() -> None:
    payload = {
        **_current_payload(),
        "final_teacher_retrain": {"enabled": True},
    }

    with pytest.raises(ValueError, match="retired metadata keys.*final_teacher_retrain"):
        require_current_gico_checkpoint_payload(payload)


def test_current_gico_payload_rejects_nested_retired_final_teacher_alias() -> None:
    payload = {
        **_current_payload(),
        "teacher_training": {
            "final_teacher_retrain": {"enabled": True},
        },
    }

    with pytest.raises(ValueError, match="retired metadata keys.*final_teacher_retrain"):
        require_current_gico_checkpoint_payload(payload)


def test_teacher_final_retrain_metadata_rejects_retired_alias() -> None:
    with pytest.raises(ValueError, match="retired metadata key 'final_teacher_retrain'"):
        _teacher_final_retrain_metadata(
            {"final_teacher_retrain": {"enabled": True}},
            {},
        )


def test_teacher_final_retrain_metadata_rejects_conflicting_sources() -> None:
    with pytest.raises(ValueError, match="conflicting teacher_final_retrain values"):
        _teacher_final_retrain_metadata(
            {"teacher_final_retrain": {"enabled": True, "selected_step": 10}},
            {"teacher_final_retrain": {"enabled": True, "selected_step": 20}},
        )


def test_teacher_final_retrain_metadata_accepts_one_consistent_contract() -> None:
    metadata = {"enabled": True, "selected_step": 10}

    assert (
        _teacher_final_retrain_metadata(
            {"teacher_final_retrain": metadata},
            {"teacher_final_retrain": dict(metadata)},
        )
        == metadata
    )

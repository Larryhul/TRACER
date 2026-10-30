"""Metadata and construction helpers for external H1 response baselines.

Names in this module are deliberately claim-safe: the external predictors are
method-level reimplementations or proxies, never assertions of official code
or training-protocol reproduction.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from .models import ResponseModelConfig, TeamResponsePredictor, build_response_predictor

METHOD_TAXONOMY: dict[str, dict[str, str]] = {
    "history-only": {"implementation": "history", "type": "action-agnostic lower bound"},
    "nearest-robot": {"implementation": "nearest", "type": "single-robot locality baseline"},
    "additive": {"implementation": "additive", "type": "matched internal reduction"},
    "eiffert2019-style": {
        "implementation": "eiffert2019-style",
        "type": "external action-conditioned response reimplementation",
        "claim": "method-level reimplementation; not official Eiffert reproduction",
    },
    "ScePT-style": {
        "implementation": "team-concat",
        "type": "external scene-context proxy",
        "claim": "team-concat proxy; not official ScePT reproduction",
    },
    "Kedia-style": {
        "implementation": "kedia-style",
        "type": "external game-theoretic latent-hypothesis adapter",
        "claim": "method-level adaptation; not official Kedia reproduction",
    },
    "joint": {"implementation": "joint", "type": "TRACER response model"},
    "matched-additive": {"implementation": "matched-additive", "type": "capacity-matched internal control"},
    "shuffled-joint": {"implementation": "shuffled-joint", "type": "action-response pairing negative control"},
    "oracle": {"implementation": "oracle", "type": "simulator upper bound"},
}


def external_method_metadata(model: nn.Module, method: str | None = None) -> dict[str, Any]:
    """Return serializable architecture/input metadata for reports."""
    name = method or str(getattr(model, "variant", "unknown"))
    if name == "team-concat":
        report_name = "ScePT-style"
    elif name == "history":
        report_name = "history-only"
    elif name == "nearest":
        report_name = "nearest-robot"
    elif name == "kedia-style":
        report_name = "Kedia-style"
    else:
        report_name = name
    taxonomy = METHOD_TAXONOMY.get(report_name, {})
    config = getattr(model, "config", None)
    config_dict = vars(config) if isinstance(config, ResponseModelConfig) else {}
    return {
        "method": report_name,
        "implementation": name,
        "type": taxonomy.get("type", "internal baseline"),
        "claim": taxonomy.get("claim", "internal implementation"),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "input_contract": {
            "entity_history": "(batch, history, 2)",
            "robot_current": "(batch, robots, 4)",
            "candidate_robot_trajectory": "(batch, robots, horizon, 4)",
            "robot_mask": "(batch, robots), fixed padded slots",
            "optional_multi_robot_context": "neighbor_history/neighbor_mask",
        },
        "model_config": config_dict,
    }


__all__ = ["METHOD_TAXONOMY", "external_method_metadata", "build_response_predictor"]

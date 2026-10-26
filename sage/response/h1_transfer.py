from pathlib import Path
from typing import Any

import torch

from sage.experiments.formal_manifest import sha256_file


def initialize_h1_encoder(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    if model.config.architecture != "tracer":
        raise ValueError("H1 encoder transfer requires a TRACER H2 model")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["model_config"]
    for name in (
        "architecture", "history", "horizon", "hidden_dim", "num_heads",
        "graph_layers", "dt", "geometry_features", "dynamic_interaction_features",
        "temporal_pooling",
    ):
        if source.get(name) != getattr(model.config, name):
            raise ValueError(f"H1/H2 encoder configuration mismatch: {name}")
    prefix = "tracer_encoder."
    weights = {name[len(prefix):]: value for name, value in checkpoint["model"].items() if name.startswith(prefix)}
    target = model.tracer_encoder.state_dict()
    if weights.keys() != target.keys() or any(weights[name].shape != target[name].shape for name in target):
        raise ValueError("H1/H2 encoder parameter names or shapes do not match")
    model.tracer_encoder.load_state_dict(weights, strict=True)
    return {
        "scope": "encoder_only_not_response_decoder_or_factorial_objective",
        "checkpoint": str(checkpoint_path), "sha256": sha256_file(checkpoint_path),
        "source_model_seed": checkpoint.get("seed"), "source_data_seed": checkpoint.get("data_seed"),
        "source_robots": source["num_robots"], "target_robots": model.config.num_robots,
        "parameter_tensors": len(weights), "frozen": False,
    }


def initialize_single_robot_response(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    """Preserve a single-robot joint predictor in every initially neutral mode."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint["model_config"]
    if source["num_robots"] != 1 or model.config.num_robots != 1 or checkpoint.get("variant") != "joint":
        raise ValueError("full response transfer currently supports only one-robot joint predictors")
    if not model.config.joint_uncertainty_decoder or model.config.static_geometry_features or model.config.evidence_conditioning:
        raise ValueError("full response transfer requires joint uncertainty and no extra map/evidence conditioning")
    for name in ("min_log_std", "max_log_std"):
        if source[name] != getattr(model.config, name):
            raise ValueError(f"response transfer configuration mismatch: {name}")
    if checkpoint.get("response_delay_steps", 0) != model.config.response_delay_steps:
        raise ValueError("response transfer must preserve the causal response delay")
    metadata = initialize_h1_encoder(model, checkpoint_path)
    weights = checkpoint["model"]
    hidden = model.config.hidden_dim
    mappings = (
        (model.base_decoder, "tracer_base_mean_decoder", [(0, 0)]),
        (model.single_decoder, "tracer_single_decoder", [(0, 0), (2 * hidden, hidden)]),
        (model.log_std_decoder, "tracer_log_std_decoder", [(0, 0), (3 * hidden, hidden)]),
    )
    with torch.no_grad():
        for decoder, prefix, columns in mappings:
            first_weight = weights[f"{prefix}.0.weight"]
            if first_weight.shape[0] != decoder[0].out_features or weights[f"{prefix}.3.weight"].shape != decoder[3].weight.shape:
                raise ValueError("response decoder dimensions do not match")
            decoder[0].weight.zero_()
            for target_column, source_column in columns:
                decoder[0].weight[:, target_column:target_column + hidden].copy_(first_weight[:, source_column:source_column + hidden])
            decoder[0].bias.copy_(weights[f"{prefix}.0.bias"])
            decoder[3].weight.copy_(weights[f"{prefix}.3.weight"])
            decoder[3].bias.copy_(weights[f"{prefix}.3.bias"])
    metadata["scope"] = "single_robot_encoder_and_complete_predictive_mean_std_heads; initially neutral modes become trainable"
    metadata["mode_conditioning_columns_initially_zero"] = True
    return metadata

from __future__ import annotations

from collections.abc import Mapping


def apply_checkpoint_compatibility(cfg, state_dict: Mapping[str, object]):
    """Adjust config fields so older checkpoints can still be evaluated."""

    keys = set(state_dict.keys())
    history_weight = state_dict.get("encoder.history_encoder.weight_ih_l0")
    has_lstm_history = False
    if history_weight is not None and hasattr(history_weight, "shape"):
        hidden_dim = int(getattr(cfg.encoder, "hidden_dim", 0))
        has_lstm_history = hidden_dim > 0 and int(history_weight.shape[0]) == 4 * hidden_dim
    if hasattr(cfg.encoder, "history_encoder_type") and not has_lstm_history:
        cfg.encoder.history_encoder_type = "gru"

    has_concat_fusion = any(key.startswith("encoder.feature_fusion_proj.") for key in keys)
    if hasattr(cfg.encoder, "feature_fusion") and not has_concat_fusion:
        cfg.encoder.feature_fusion = "sum"

    has_semantic_relations = any("encoder.layers.0.q.4." in key for key in keys)
    if hasattr(cfg.encoder, "relation_granularity") and not has_semantic_relations:
        cfg.encoder.relation_granularity = "control"

    is_new_encoder = has_lstm_history and has_concat_fusion
    if hasattr(cfg.diffusion, "guidance_update") and not is_new_encoder:
        cfg.diffusion.guidance_update = "x0"
    if hasattr(cfg.diffusion, "use_graph_context_token") and not is_new_encoder:
        cfg.diffusion.use_graph_context_token = False

    has_compact_hgt = "encoder.layers.0.q.weight" in keys
    if has_compact_hgt:
        cfg.encoder.relation_attention_type = "compact"
        cfg.encoder.use_entity_entity_edges = True
    has_robot_current = any(key.startswith("encoder.robot_state_encoder.") for key in keys)
    if not has_robot_current:
        cfg.encoder.use_robot_current_state = False
    has_geometry = any(key.startswith("encoder.geometry_embedding.") for key in keys)
    if not has_geometry:
        cfg.encoder.use_geometry_embedding = False
    has_legacy_mlp = any(key.endswith("denoiser.net.0.weight") or ".denoiser.net." in key for key in keys)
    if has_legacy_mlp:
        cfg.diffusion.denoiser_type = "mlp"
    has_agent_attention = any(".agent_attn." in key for key in keys)
    has_transformer = any("denoiser.layers." in key for key in keys)
    if has_transformer and not has_agent_attention:
        cfg.diffusion.use_agent_attention = False
    return cfg

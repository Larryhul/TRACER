from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from sage.data.heterogeneous_scene import ROLE_TO_ID
from sage.guidance.safety_social import ComfortMargin, SafetySocialGuidanceConfig
from sage.models.hgt_encoder import HGTEncoderConfig
from sage.models.joint_diffusion import DiffusionConfig
from sage.models.sage_model import SAGEConfig


def load_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _allowed_kwargs(cls, data: Mapping[str, Any] | None) -> dict[str, Any]:
    if not data:
        return {}
    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in data.items() if key in allowed}


def _comfort_from_raw(raw: Any) -> dict[int, ComfortMargin] | None:
    if not isinstance(raw, Mapping):
        return None

    comfort: dict[int, ComfortMargin] = {}
    for role, margin in raw.items():
        role_id = ROLE_TO_ID.get(str(role).lower(), None)
        if role_id is None:
            try:
                role_id = int(role)
            except (TypeError, ValueError):
                continue
        if isinstance(margin, Mapping):
            comfort[int(role_id)] = ComfortMargin(
                sigma_long=float(margin.get("sigma_long", margin.get("sigma_lon", 1.5))),
                sigma_lat=float(margin.get("sigma_lat", 0.8)),
            )
    return comfort or None


def sage_config_from_dict(raw: Mapping[str, Any] | None) -> SAGEConfig:
    """Build a SAGEConfig from either configs/sage.yaml or a checkpoint dict."""

    raw = raw or {}
    model = raw.get("model", raw)

    encoder = HGTEncoderConfig(**_allowed_kwargs(HGTEncoderConfig, model.get("encoder", raw.get("encoder", {}))))
    diffusion = DiffusionConfig(**_allowed_kwargs(DiffusionConfig, model.get("diffusion", raw.get("diffusion", {}))))

    guidance_raw = dict(model.get("guidance", raw.get("guidance", {})) or {})
    guidance_scale = guidance_raw.pop("guidance_scale", model.get("guidance_scale", raw.get("guidance_scale", 0.03)))

    comfort = _comfort_from_raw(guidance_raw.pop("comfort", None))
    guidance_kwargs = _allowed_kwargs(SafetySocialGuidanceConfig, guidance_raw)
    if comfort is not None:
        guidance_kwargs["comfort"] = comfort
    guidance = SafetySocialGuidanceConfig(**guidance_kwargs)

    generation_mode = model.get("generation_mode", raw.get("generation_mode", "joint"))

    return SAGEConfig(
        encoder=encoder,
        diffusion=diffusion,
        guidance=guidance,
        guidance_scale=float(guidance_scale),
        generation_mode=str(generation_mode),
    )


def load_sage_config(path: str | Path | None) -> tuple[SAGEConfig, dict[str, Any]]:
    raw = load_yaml(path)
    return sage_config_from_dict(raw), raw

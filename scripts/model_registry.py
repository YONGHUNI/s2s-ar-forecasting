"""Explicit Earth2Studio forecast model dispatch.

NeuralGCM is intentionally not registered here. Its stochastic rollout
requires a separate adapter rather than Earth2Studio run.deterministic().
"""
from __future__ import annotations

from importlib import import_module

_MODEL_CLASSES = {
    "GraphCastOperational": "GraphCastOperational",
    "Pangu6": "Pangu6",
    "SFNO": "SFNO",
}


def get_model_type(model_name: str):
    """Resolve a supported deterministic model or fail before loading weights."""
    try:
        class_name = _MODEL_CLASSES[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_MODEL_CLASSES))
        raise ValueError(
            f"Unsupported deterministic model {model_name!r}. Supported: {supported}"
        ) from exc
    module = import_module("earth2studio.models.px")
    return getattr(module, class_name)

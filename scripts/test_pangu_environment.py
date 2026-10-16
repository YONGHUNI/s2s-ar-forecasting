#!/usr/bin/env python3
"""Validate the isolated Pangu environment without downloading model weights."""
import importlib.metadata as metadata

import onnxruntime as ort
import torch
from earth2studio.models.px import Pangu6

print("earth2studio:", metadata.version("earth2studio"))
print("torch:", torch.__version__)
print("Pangu6 import: OK")
print("torch CUDA available:", torch.cuda.is_available())
print("ONNX execution providers:", ort.get_available_providers())

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access the allocated CUDA device.")
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise RuntimeError("ONNX Runtime CUDAExecutionProvider not available.")

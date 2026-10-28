"""Find and load the official checkpoint (.nemo tar: model_config.yaml + model_weights.ckpt) without NeMo."""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import torch
import yaml

from .config import ModelConfig
from .modeling import SortformerModel, convert_nemo_state_dict, rope_tables

DEFAULT_REPO = "nvidia/Nemotron-3-Diarization"


def cache_dir() -> Path:
    root = os.environ.get("NEMOTRON_DIAR_TRT_CACHE")
    if root:
        return Path(root)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "nemotron_diar_trt"


def resolve_nemo(path_or_repo: str | os.PathLike = DEFAULT_REPO, revision: str | None = None) -> Path:
    """A .nemo file, a directory holding exactly one, or a Hugging Face repo id."""
    p = Path(path_or_repo)
    if p.is_file():
        return p
    if p.is_dir():
        files = sorted(p.glob("*.nemo"))
        if len(files) != 1:
            raise FileNotFoundError(f"expected exactly one .nemo file in {p}, found {len(files)}")
        return files[0]
    from huggingface_hub import snapshot_download

    local = snapshot_download(str(path_or_repo), revision=revision, allow_patterns=["*.nemo"])
    return resolve_nemo(local)


def _member(tar: tarfile.TarFile, name: str) -> bytes:
    for m in tar.getmembers():
        if m.isfile() and Path(m.name).name == name:
            return tar.extractfile(m).read()
    raise FileNotFoundError(f"{name} not in {tar.name}")


def read_nemo(path: str | os.PathLike) -> tuple[dict, dict[str, torch.Tensor]]:
    """(model_config.yaml as a dict, NeMo state dict as stored: bf16 tensors)."""
    with tarfile.open(path, "r:*") as tar:
        cfg = yaml.safe_load(_member(tar, "model_config.yaml"))
        sd = torch.load(io.BytesIO(_member(tar, "model_weights.ckpt")), map_location="cpu", weights_only=True)
    return cfg, sd


def load_model(path_or_repo: str | os.PathLike = DEFAULT_REPO, device="cpu",
               revision: str | None = None) -> SortformerModel:
    """fp32 model with the checkpoint's values (NeMo also runs the bf16-stored weights in fp32)."""
    raw_cfg, sd = read_nemo(resolve_nemo(path_or_repo, revision))
    cfg = ModelConfig.from_nemo_yaml(raw_cfg)
    with torch.device("meta"):
        model = SortformerModel(cfg)
    state = {k: v.to(torch.float32) for k, v in convert_nemo_state_dict(sd).items()}
    model.load_state_dict(state, strict=True, assign=True)
    cos, sin = rope_tables(cfg)  # non-persistent buffers: rebuild off the meta device
    model.encoder.rope_cos, model.encoder.rope_sin = cos, sin
    return model.eval().requires_grad_(False).to(device)

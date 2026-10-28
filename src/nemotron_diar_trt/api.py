"""Public API: `Diarizer.from_pretrained(...).diarize(audio) -> Result`."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import numpy as np
import torch

from .checkpoint import DEFAULT_REPO, cache_dir, load_model, resolve_nemo
from .config import StreamingConfig
from .features import FEATURE_CHUNK_SAMPLES, compute_features
from .modeling import SortformerModel
from .postprocess import Turn, turns_from_probs
from .runtime import GraphStreamer
from .streaming import stream_features


@dataclass
class Result:
    probs: np.ndarray  # [frames, n_spk] float32 speaker probabilities, one frame per `frame_seconds`
    turns: list[Turn]  # grouped by speaker index, time order within a speaker (NeMo's order)
    frame_seconds: float

    @property
    def lines(self) -> list[str]:
        """NeMo's `diarize()` strings: "start end speaker_k"."""
        return [t.line() for t in self.turns]


def read_audio(path: str | os.PathLike, sample_rate: int = 16000) -> np.ndarray:
    """Mono float32 (channel mean), which must already be at `sample_rate`."""
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != sample_rate:
        raise ValueError(f"{path}: {sr} Hz; resample to {sample_rate} Hz first")
    return wav.mean(axis=1)


class Diarizer:
    def __init__(self, model: SortformerModel, streaming: StreamingConfig | None = None, encode=None,
                 nemo_compat: bool = False, graphs: bool = True, pad_to: int | None = None):
        self.model = model
        self.streaming = streaming or StreamingConfig()
        self.encode = encode  # None: torch
        self.nemo_compat = nemo_compat
        self.feature_chunk = FEATURE_CHUNK_SAMPLES
        self.streamer = None
        self.stream = None
        if self.device.type == "cuda":
            self.stream = torch.cuda.Stream(self.device)  # TensorRT synchronizes more on the default stream
            if graphs:
                self.streamer = GraphStreamer(model, self.streaming, encode, pad_to=pad_to)

    @classmethod
    def from_pretrained(cls, path_or_repo: str | os.PathLike = DEFAULT_REPO, backend: str = "trt",
                        dtype: str = "float32", device: str = "cuda", streaming: StreamingConfig | None = None,
                        nemo_compat: bool = False, revision: str | None = None, attention: str = "sdpa",
                        graphs: bool = True, rebuild: bool = False, engine_head: bool | None = None,
                        pad: bool | None = None) -> Diarizer:
        """backend "trt" (default: TensorRT encoder in `dtype`, engine built on first use and cached under
        cache_dir()) or "torch" (fp32; attention "sdpa" or NeMo's "flex"). dtype "float32" (default) gives
        NeMo's turns (3x NeMo's speed); "float16" is ~3x faster again, with turns that differ from NeMo's in a
        few percent of frames (DER on par; README "Precision"). engine_head: the head inside the
        engine (default for float16 / bfloat16) or in torch fp32 (default for float32: the head's conv then runs
        in TF32 as in NeMo, and the output matches NeMo's). graphs=False runs the eager reference loop (every
        step as NeMo runs it). pad: pad every step to the longest sequence so the encoder sees one shape
        (default: on for TensorRT, off for torch, whose steps then match NeMo's shapes exactly)."""
        path = resolve_nemo(path_or_repo, revision)
        model = load_model(path, device=device)
        model.encoder.attention = attention
        streaming = streaming or StreamingConfig()
        from .trt import max_seq_len

        encode, n = None, max_seq_len(streaming)
        if backend == "trt":
            from .trt import BuildOptions, TrtEncoder, ensure_engine

            if engine_head is None:
                engine_head = dtype != "float32"
            opts = BuildOptions(dtype=dtype, head=engine_head, max_len=n, opt_len=n)
            engine = ensure_engine(model, f"{path.stem}-{path.stat().st_size}", opts, cache_dir(), rebuild)
            encode = TrtEncoder(engine, model.cfg.upsample, model.cfg.n_spk, None if engine_head else model.head)
        elif backend != "torch":
            raise ValueError(f"backend must be 'trt' or 'torch', not {backend!r}")
        if pad is None:
            pad = backend == "trt"
        return cls(model, streaming, encode, nemo_compat, graphs, n if pad else None)

    @property
    def device(self) -> torch.device:
        return self.model.frontend.window.device

    @torch.no_grad()
    def probabilities(self, audio) -> np.ndarray:
        """audio: 16 kHz mono float32 samples (array / tensor) or a path -> [frames, n_spk] float32."""
        if isinstance(audio, (str, os.PathLike)):
            audio = read_audio(audio, self.model.cfg.sample_rate)
        audio = torch.as_tensor(np.asarray(audio, np.float32) if not isinstance(audio, torch.Tensor) else audio)
        with contextlib.nullcontext() if self.stream is None else torch.cuda.stream(self.stream):
            feats = compute_features(self.model.frontend, audio.float(), self.feature_chunk)
            if self.streamer is not None:
                return self.streamer(feats).cpu().numpy()
            lengths = torch.tensor([feats.shape[1]], device=self.device)
            out = stream_features(self.model, feats[None], lengths, self.streaming, encode=self.encode)
            return out[0].float().cpu().numpy()

    def diarize(self, audio) -> Result:
        probs = self.probabilities(audio)
        frame = self.model.cfg.frame_seconds
        return Result(probs, turns_from_probs(probs, frame, self.streaming.threshold, self.nemo_compat), frame)

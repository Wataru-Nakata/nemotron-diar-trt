"""TensorRT engine for the encoder (and head): ONNX export, build on first use, and the runtime call.

The engine maps a pre-encoded [cache | fifo | chunk] sequence [B, T, 512] (float32) and its valid lengths [B]
(int32; keys beyond them are masked) to 10 ms speaker probabilities [B, 8T, 8] (head in the engine) or to the
encoder output [B, T, 512] (head=False: the head then runs in torch fp32, its conv in TF32 as in NeMo), both
float32. Pre-encode, the FIFO / cache update and the compression stay in fp32 torch (see streaming.py).
Engines are cached under cache_dir() by GPU, TensorRT version and options; build.json is written last and
marks a complete build. (A mixed engine, fp32 residual stream and LayerNorms around fp16 attention and
feed-forward, agreed no better with NeMo than plain fp16: docs/j3-2779198.md.)
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .config import StreamingConfig
from .modeling import SortformerModel

log = logging.getLogger("nemotron_diar_trt")

EXPORT_VERSION = 3  # bump whenever the exported graph changes
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


@dataclass(frozen=True)
class BuildOptions:
    dtype: str = "float32"
    head: bool = False  # head in the engine (else the engine returns the encoder output and torch runs the head)
    max_batch: int = 1
    max_len: int = 684  # longest [cache | fifo | chunk] in encoder frames (max_seq_len of the settings)
    opt_len: int = 684
    tf32: bool = False  # float32 engines: allow TF32 matmuls
    opt_level: int = 3
    workspace_gb: float = 8.0


def max_seq_len(sc: StreamingConfig) -> int:
    return sc.spkcache_len + sc.fifo_len + sc.chunk_left_context + sc.chunk_len + sc.chunk_right_context


class _EncoderGraph(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module | None, in_dtype: torch.dtype):
        super().__init__()
        self.encoder, self.head, self.in_dtype = encoder, head, in_dtype

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.encoder(seq.to(self.in_dtype), lengths)
        return (x if self.head is None else self.head(x, lengths)).float()


def export_onnx(model: SortformerModel, path: Path, opts: BuildOptions) -> Path:
    dtype = DTYPES[opts.dtype]
    model = copy.deepcopy(model)
    model.encoder.attention = "math"
    model.encoder.to(dtype)
    model.head.to(dtype)
    graph = _EncoderGraph(model.encoder, model.head if opts.head else None, dtype).eval()
    dev = model.frontend.window.device
    batch = torch.export.Dim("batch", min=1, max=max(2, opts.max_batch))
    length = torch.export.Dim("length", min=1, max=opts.max_len)
    t = min(300, opts.max_len - 1)  # sizes > 1 and below the maximum, so the exporter keeps them dynamic
    seq = torch.randn(2, t, model.cfg.d_model, device=dev)
    lengths = torch.tensor([t, t - 7], dtype=torch.int32, device=dev)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        torch.onnx.export(graph, (seq, lengths), path, input_names=["seq", "lengths"], output_names=["out"],
                          dynamic_shapes={"seq": {0: batch, 1: length}, "lengths": {0: batch}}, dynamo=True,
                          verbose=False, external_data=True)
    return path


_LOGGER = None
_RUNTIME = None


def trt_logger(trt):
    """One logger for the process: TensorRT keeps the first one it is given and ignores later ones."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = _make_logger(trt)
    return _LOGGER


def _make_logger(trt):
    class Logger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)

        def log(self, severity, msg):
            if severity in (trt.ILogger.Severity.ERROR, trt.ILogger.Severity.INTERNAL_ERROR):
                log.error("TensorRT: %s", msg)
            elif severity == trt.ILogger.Severity.WARNING:
                log.warning("TensorRT: %s", msg)

    return Logger()


def build_engine(onnx_path: Path, engine_path: Path, opts: BuildOptions, d_model: int) -> float:
    import tensorrt as trt

    logger = trt_logger(trt)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"failed to parse {onnx_path}:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(opts.workspace_gb * (1 << 30)))
    config.builder_optimization_level = opts.opt_level
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if not opts.tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    profile.set_shape("seq", (1, 1, d_model), (1, opts.opt_len, d_model), (opts.max_batch, opts.max_len, d_model))
    profile.set_shape("lengths", (1,), (1,), (opts.max_batch,))
    config.add_optimization_profile(profile)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {engine_path}")
    engine_path.write_bytes(serialized)
    return time.time() - t0


def engine_tag(opts: BuildOptions) -> str:
    import tensorrt as trt

    major, minor = torch.cuda.get_device_capability()
    gpu = torch.cuda.get_device_name().replace(" ", "_").replace("/", "_")
    digest = hashlib.sha1(json.dumps(asdict(opts), sort_keys=True).encode()).hexdigest()[:8]
    return f"v{EXPORT_VERSION}-{gpu}-sm{major}{minor}-trt{trt.__version__}-{opts.dtype}-{digest}"


def ensure_engine(model: SortformerModel, model_tag: str, opts: BuildOptions, root: Path,
                  rebuild: bool = False) -> Path:
    """The engine for these options, built (and cached) if missing. Returns the engine file."""
    engine_dir = root / model_tag / engine_tag(opts)
    meta = engine_dir / "build.json"
    if not rebuild and meta.exists() and json.loads(meta.read_text()).get("options") == asdict(opts):
        return engine_dir / "encoder.engine"
    engine_dir.mkdir(parents=True, exist_ok=True)
    meta.unlink(missing_ok=True)
    log.warning("building the TensorRT engine in %s (once per GPU / TensorRT / options)", engine_dir)
    t0 = time.time()
    onnx_path = export_onnx(model, engine_dir / "onnx" / "encoder.onnx", opts)
    export_seconds = time.time() - t0
    build_seconds = build_engine(onnx_path, engine_dir / "encoder.engine", opts, model.cfg.d_model)
    shutil.rmtree(engine_dir / "onnx")
    meta.write_text(json.dumps({"options": asdict(opts), "export_seconds": export_seconds,
                                "build_seconds": build_seconds, "torch": torch.__version__}, indent=1))
    return engine_dir / "encoder.engine"


class TrtEncoder:
    """Runs the engine on torch tensors; callable as `streaming.Encode`. Each CUDA graph gets an execution
    context of its own (TensorRT state must not be shared between captured graphs); eager calls share one.
    All contexts share one scratch buffer: they run on the same stream, one at a time. `head`: the torch head,
    for engines built with head=False."""

    def __init__(self, engine_path: Path, upsample: int, n_spk: int, head: nn.Module | None = None):
        import tensorrt as trt

        global _RUNTIME
        self.trt = trt
        if _RUNTIME is None:
            _RUNTIME = trt.Runtime(trt_logger(trt))
        self.engine = _RUNTIME.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}; rebuild it")
        self.upsample, self.n_spk, self.head = upsample, n_spk, head
        self.scratch = torch.empty(max(self.engine.device_memory_size_v2, 1), dtype=torch.uint8, device="cuda")
        self.contexts: dict[object, object] = {}
        self.graph_mode = None

    def _context(self, shape: tuple):
        # graph_mode is set (to a key of the graph) by runtime._Graph around its warm-up and capture: the
        # graph's own context, which also gets the one enqueue TensorRT needs after a shape change before capture
        key = "eager" if self.graph_mode is None else self.graph_mode
        ctx = self.contexts.get(key)
        if ctx is None:
            ctx = self.engine.create_execution_context(self.trt.ExecutionContextAllocationStrategy.USER_MANAGED)
            ctx.set_device_memory(self.scratch.data_ptr(), self.scratch.numel())
            self.contexts[key] = ctx
        if tuple(ctx.get_tensor_shape("seq")) != shape:
            ctx.set_input_shape("seq", shape)
            ctx.set_input_shape("lengths", shape[:1])
        return ctx

    def __call__(self, seq: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        seq = seq.contiguous()
        b, t, d = seq.shape
        valid = torch.full((b,), t, dtype=torch.int32, device=seq.device) if lengths is None else lengths.int()
        if self.head is None:
            out = torch.empty(b, t * self.upsample, self.n_spk, device=seq.device, dtype=torch.float32)
        else:
            out = torch.empty(b, t, d, device=seq.device, dtype=torch.float32)
        ctx = self._context(tuple(seq.shape))
        ctx.set_tensor_address("seq", seq.data_ptr())
        ctx.set_tensor_address("lengths", valid.data_ptr())
        ctx.set_tensor_address("out", out.data_ptr())
        if not ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT encoder execution failed")
        return out if self.head is None else self.head(out, lengths)

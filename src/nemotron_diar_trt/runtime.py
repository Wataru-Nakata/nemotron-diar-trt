"""Fast batch-1 streaming: whole steps (encoder, head, FIFO / cache update, compression) replayed as CUDA graphs.

With fixed streaming settings almost every step has the same shapes: with the default settings step 0 sees
[chunk 380] and every later step [cache 264 | fifo 40 | chunk 380], until the last one or two. A step's
control flow depends only on its signature (cache / FIFO lengths, compressed or not, chunk length and
contexts). Steps with a full chunk and full right context are run eagerly the first time their signature
occurs and captured as a CUDA graph the second time; from then on one replay runs the step. Graphs persist
across recordings, so step 0 is a graph from the second recording on. The last one or two steps of a
recording (shorter chunks, lengths that vary from recording to recording) always run eagerly, which keeps
the number of graphs small. Outputs go into one preallocated buffer.
"""

from __future__ import annotations

import math
import time

import torch

from .config import StreamingConfig
from .modeling import SortformerModel
from .streaming import Encode, StreamingState, chunk_plan, stream_step, torch_encode


class _Graph:
    def __init__(self, model, encode, sc, sig, state: StreamingState, chunk: torch.Tensor, lc, rc, update, pad_to):
        self.inputs = {"spkcache": state.spkcache.clone(), "fifo": state.fifo.clone(), "chunk": chunk.clone()}
        if state.compressed:
            self.inputs["spkcache_preds"] = state.spkcache_preds.clone()
        self.compressed = state.compressed

        def run():
            st = StreamingState(self.inputs["spkcache"], self.inputs["fifo"], self.inputs.get("spkcache_preds"),
                                self.compressed)
            return stream_step(model, encode, st, self.inputs["chunk"], lc, rc, sc, update=update, pad_to=pad_to)

        graph_mode = hasattr(encode, "graph_mode")  # TrtEncoder: an execution context for this graph alone
        if graph_mode:
            encode.graph_mode = ("graph", sig)
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):  # warm-up on a side stream, as torch.cuda.graph requires
                run()
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.state, self.out, _ = run()
        finally:
            if graph_mode:
                encode.graph_mode = None

    def __call__(self, state: StreamingState, chunk: torch.Tensor):
        self.inputs["spkcache"].copy_(state.spkcache)
        self.inputs["fifo"].copy_(state.fifo)
        if self.compressed:
            self.inputs["spkcache_preds"].copy_(state.spkcache_preds)
        self.inputs["chunk"].copy_(chunk)
        self.graph.replay()
        return self.state, self.out


class GraphStreamer:
    """Batch-1 streaming with per-signature CUDA graphs. Same numbers as `stream_features` (same kernels and
    shapes); the last step's state update, whose result is never used, is skipped.

    pad_to: pad every step's sequence to this length (masked), so the encoder sees one shape: a TensorRT
    context then never re-plans for the varying lengths of a recording's first and last steps.
    time_steps: record (kind, seconds) per step in `step_times` (synchronizes; for profiling).
    """

    def __init__(self, model: SortformerModel, sc: StreamingConfig, encode: Encode | None = None,
                 graphs: bool = True, pad_to: int | None = None):
        self.model, self.sc = model, sc
        self.encode = encode or torch_encode(model)
        self.use_graphs = graphs
        self.pad_to = pad_to
        self.graphs: dict[tuple, _Graph] = {}
        self.seen: set[tuple] = set()
        self.time_steps = False
        self.step_times: list[tuple[str, float]] = []

    @torch.no_grad()
    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        """feats [n_mels, T] on the GPU -> probabilities [T, n_spk] (GPU, float32)."""
        cfg = self.model.cfg
        factor, up = cfg.subsampling, cfg.upsample
        n_frames = feats.shape[1]
        plan = chunk_plan(n_frames, self.sc, factor)
        out = torch.empty(len(plan) * self.sc.chunk_len * up, cfg.n_spk, device=feats.device)
        state = StreamingState.empty(1, cfg.d_model, feats.device)
        pos = 0
        feats_t = feats.t()[None]  # [1, T, n_mels]
        for k, (stt, end, left, right) in enumerate(plan):
            chunk = feats_t[:, stt - left:end + right]
            lc, rc = round(left / factor), math.ceil(right / factor)
            update = k + 1 < len(plan)
            full = end - stt == self.sc.chunk_len * factor and rc == self.sc.chunk_right_context and update
            sig = (state.spkcache.shape[1], state.fifo.shape[1], state.compressed, chunk.shape[1], lc, rc, update)
            if self.time_steps:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            if sig in self.graphs:
                kind = "graph"
                state, piece = self.graphs[sig](state, chunk)
            elif self.use_graphs and full and sig in self.seen:
                kind = "capture"
                self.graphs[sig] = _Graph(self.model, self.encode, self.sc, sig, state, chunk, lc, rc, update,
                                          self.pad_to)
                state, piece = self.graphs[sig](state, chunk)
            else:
                kind = "eager"
                if full:
                    self.seen.add(sig)
                state, piece, _ = stream_step(self.model, self.encode, state, chunk, lc, rc, self.sc, update=update,
                                              pad_to=self.pad_to)
            out[pos:pos + piece.shape[1]].copy_(piece[0])
            pos += piece.shape[1]
            if self.time_steps:
                torch.cuda.synchronize()
                self.step_times.append((kind, time.perf_counter() - t0))
        return out[:n_frames]

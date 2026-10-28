"""Streaming Sortformer inference: chunk loop, FIFO / speaker-cache update and cache compression.

This is the reference path: it follows NeMo's synchronous `forward_streaming` (sortformer_diar_models.py) and
`SortformerModules.streaming_update` / `_compress_spkcache` (sortformer_modules.py) step for step, with the
same tensor shapes, so fp32 results match NeMo's. The compression avoids NeMo's boolean-mask assignment and
CPU index tensors (same values via torch.where / gather / scatter) so that it can also run inside a CUDA graph.
See docs/streaming.md for the per-step walk-through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from .config import ModelConfig, StreamingConfig
from .modeling import SortformerModel


@dataclass
class StreamingState:
    spkcache: torch.Tensor  # [B, S, D] pre-encoded embeddings (before embed_norm)
    fifo: torch.Tensor  # [B, F, D]
    spkcache_preds: torch.Tensor | None = None  # [B, S, n_spk], kept from the last compression
    compressed: bool = False

    @classmethod
    def empty(cls, batch: int, d_model: int, device) -> StreamingState:
        return cls(torch.zeros(batch, 0, d_model, device=device), torch.zeros(batch, 0, d_model, device=device))


def chunk_plan(n_frames: int, sc: StreamingConfig, factor: int) -> list[tuple[int, int, int, int]]:
    """(start, end, left_offset, right_offset) in feature frames for each step (NeMo streaming_feat_loader)."""
    plan, stt, end = [], 0, 0
    while end < n_frames:
        left = min(sc.chunk_left_context * factor, stt)
        end = min(stt + sc.chunk_len * factor, n_frames)
        right = min(sc.chunk_right_context * factor, n_frames - end)
        plan.append((stt, end, left, right))
        stt = end
    return plan


def downsample_preds(preds: torch.Tensor, factor: int) -> torch.Tensor:
    """Mean of each `factor` consecutive 10 ms frames -> encoder rate (partial last window kept)."""
    return F.avg_pool1d(preds.transpose(1, 2), kernel_size=factor, stride=factor, ceil_mode=True,
                        count_include_pad=False).transpose(1, 2)


def _log_pred_scores(preds: torch.Tensor, cfg: ModelConfig) -> torch.Tensor:
    log_probs = torch.log(torch.clamp(preds, min=cfg.pred_score_threshold))
    log_1_probs = torch.log(torch.clamp(1.0 - preds, min=cfg.pred_score_threshold))
    log_1_probs_sum = log_1_probs.sum(dim=2).unsqueeze(-1).expand(-1, -1, preds.shape[2])
    return log_probs - log_1_probs + log_1_probs_sum - math.log(0.5)


def _disable_low_scores(preds: torch.Tensor, scores: torch.Tensor, min_pos_scores_per_spk: int) -> torch.Tensor:
    is_speech = preds > 0.5
    scores = torch.where(is_speech, scores, float("-inf"))
    is_pos = scores > 0
    replace = (~is_pos) & is_speech & (is_pos.sum(dim=1).unsqueeze(1) >= min_pos_scores_per_spk)
    return torch.where(replace, float("-inf"), scores)


def _boost_topk_scores(scores: torch.Tensor, n_boost_per_spk: int, scale_factor: float) -> torch.Tensor:
    _, idx = torch.topk(scores, n_boost_per_spk, dim=1, largest=True, sorted=False)
    boosted = torch.gather(scores, 1, idx) - scale_factor * math.log(0.5)
    return scores.scatter(1, idx, boosted)


def select_spkcache(preds: torch.Tensor, spkcache_len: int, cfg: ModelConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Frame indices [B, spkcache_len] to keep (speaker-major, time order within a speaker) and which of them
    are silence slots, from predictions [B, n, n_spk] (n > spkcache_len)."""
    batch, n_frames, n_spk = preds.shape
    per_spk = spkcache_len // n_spk - cfg.spkcache_sil_frames_per_spk
    strong = math.floor(per_spk * cfg.strong_boost_rate)
    weak = math.floor(per_spk * cfg.weak_boost_rate)
    min_pos = math.floor(per_spk * cfg.min_pos_scores_rate)

    scores = _disable_low_scores(preds, _log_pred_scores(preds, cfg), min_pos)
    if cfg.scores_boost_latest > 0:  # newly added frames (position-based, as in NeMo)
        scores[:, spkcache_len:, :] += cfg.scores_boost_latest
    scores = _boost_topk_scores(scores, strong, 2)  # each speaker gets at least `strong` frames
    scores = _boost_topk_scores(scores, weak, 1)  # no speaker dominates
    sil = cfg.spkcache_sil_frames_per_spk
    if sil > 0:  # silence slots at the end of each speaker block
        scores = torch.cat([scores, torch.full((batch, sil, n_spk), float("inf"), device=scores.device)], dim=1)

    n_frames_sil = scores.shape[1]
    flat = scores.permute(0, 2, 1).reshape(batch, -1)  # speaker-major
    values, idx = torch.topk(flat, spkcache_len, dim=1, sorted=False)
    idx = torch.where(values != float("-inf"), idx, cfg.max_index)
    idx, _ = torch.sort(idx, dim=1)  # speaker blocks in order, frames in time order within a block
    disabled = idx == cfg.max_index
    idx = torch.remainder(idx, n_frames_sil)
    disabled = disabled | (idx >= n_frames_sil - sil)
    return torch.where(disabled, 0, idx), disabled


def compress_spkcache(emb_seq: torch.Tensor, preds: torch.Tensor, sil_emb: torch.Tensor, spkcache_len: int,
                      cfg: ModelConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the `spkcache_len` most informative frames, grouped by speaker, each block ending in silence slots.

    emb_seq [B, n, D], preds [B, n, n_spk] (n > spkcache_len) -> ([B, spkcache_len, D], [B, spkcache_len, n_spk]).
    """
    batch, _, n_spk = preds.shape
    idx, disabled = select_spkcache(preds, spkcache_len, cfg)
    d = emb_seq.shape[2]
    emb = torch.gather(emb_seq, 1, idx.unsqueeze(-1).expand(-1, -1, d))
    sil_emb = sil_emb.to(dtype=emb_seq.dtype, device=emb_seq.device).unsqueeze(0).expand(batch, -1)
    emb = torch.where(disabled.unsqueeze(-1), sil_emb.unsqueeze(1).expand(-1, spkcache_len, -1), emb)
    kept = torch.gather(preds, 1, idx.unsqueeze(-1).expand(-1, -1, n_spk))
    kept = torch.where(disabled.unsqueeze(-1), 0.0, kept)
    return emb, kept


def streaming_update(state: StreamingState, chunk: torch.Tensor, preds: torch.Tensor, lc: int, rc: int,
                     sc: StreamingConfig, cfg: ModelConfig, sil_emb: torch.Tensor) -> StreamingState:
    """Append the chunk to the FIFO; move popped frames into the speaker cache and compress it when full.

    chunk [B, lc + L + rc, D]; preds [B, S + F + lc + L + rc, n_spk] at encoder rate.
    """
    s, f, n = state.spkcache.shape[1], state.fifo.shape[1], chunk.shape[1] - lc - rc
    fifo_preds = preds[:, s:s + f]
    chunk_preds = preds[:, s + f + lc:s + f + n + lc]
    fifo = torch.cat([state.fifo, chunk[:, lc:n + lc]], dim=1)
    fifo_preds = torch.cat([fifo_preds, chunk_preds], dim=1)
    spkcache, spkcache_preds, compressed = state.spkcache, state.spkcache_preds, state.compressed
    if f + n > sc.fifo_len:
        pop = min(max(sc.spkcache_update_period, n - sc.fifo_len + f), f + n)
        pop_embs, pop_preds = fifo[:, :pop], fifo_preds[:, :pop]
        fifo = fifo[:, pop:]
        spkcache = torch.cat([spkcache, pop_embs], dim=1)
        # Before the first compression the cache frames carry this step's predictions; afterwards they keep
        # the ones they were selected with.
        spkcache_preds = torch.cat([spkcache_preds if compressed else preds[:, :s], pop_preds], dim=1)
        if spkcache.shape[1] > sc.spkcache_len:
            spkcache, spkcache_preds = compress_spkcache(spkcache, spkcache_preds, sil_emb, sc.spkcache_len, cfg)
            compressed = True
    return StreamingState(spkcache, fifo, spkcache_preds, compressed)


StepHook = Callable[[int, dict], None]
Encode = Callable[[torch.Tensor, "torch.Tensor | None"], torch.Tensor]


def torch_encode(model: SortformerModel) -> Encode:
    """[B, T, D] pre-encoded sequence (+ valid lengths or None) -> 10 ms probabilities [B, T * upsample, n_spk]."""
    return lambda seq, lengths: model.head(model.encoder(seq, lengths), lengths)


def stream_step(model: SortformerModel, encode: Encode, state: StreamingState, chunk_feats: torch.Tensor, lc: int,
                rc: int, sc: StreamingConfig, seq_lengths: torch.Tensor | None = None, update: bool = True,
                pad_to: int | None = None):
    """One streaming step (NeMo forward_streaming_step, synchronous path).

    chunk_feats [B, lc*8 + L*8 + rc*8, n_mels] -> (next state, this chunk's 10 ms probabilities [B, L * 8, n_spk],
    tensors for inspection). seq_lengths: valid [cache | fifo | chunk] lengths when rows are padded (else None).
    pad_to: zero-pad the sequence to this length and mask the padding (one encoder shape for every step; the
    numbers differ from the unpadded step by rounding only). Control flow depends only on shapes and
    `state.compressed`, so a step can be captured in a CUDA graph.
    """
    cfg, up = model.cfg, model.cfg.upsample
    emb = model.encoder.pre_encode(chunk_feats)
    s, f = state.spkcache.shape[1], state.fifo.shape[1]
    seq = torch.cat([state.spkcache, state.fifo, emb], dim=1)
    if pad_to is not None and seq.shape[1] < pad_to:
        if seq_lengths is None:
            seq_lengths = torch.full((seq.shape[0],), seq.shape[1], dtype=torch.int32, device=seq.device)
        seq = F.pad(seq, (0, 0, 0, pad_to - seq.shape[1]))
    hr = encode(seq, seq_lengths)
    preds = downsample_preds(hr, up)
    if seq_lengths is not None:
        preds = preds * (torch.arange(preds.shape[1], device=preds.device)[None, :] < seq_lengths[:, None])[..., None]
    n = emb.shape[1] - lc - rc
    start = (s + f + lc) * up
    out = hr[:, start:start + n * up]
    if update:
        state = streaming_update(state, emb, preds, lc, rc, sc, cfg, model.head.learnable_sil_emb)
    return state, out, {"emb": emb, "preds": preds, "hr": hr}


@torch.no_grad()
def stream_features(model: SortformerModel, feats: torch.Tensor, lengths: torch.Tensor, sc: StreamingConfig,
                    hook: StepHook | None = None, encode: Encode | None = None) -> torch.Tensor:
    """feats [B, n_mels, T] (frames >= lengths ignored) -> speaker probabilities [B, T, n_spk] at 10 ms.

    The reference loop: eager, every step exactly as NeMo runs it (including the last state update).
    """
    cfg = model.cfg
    encode = encode or torch_encode(model)
    factor = cfg.subsampling
    batch, _, n_frames = feats.shape
    state = StreamingState.empty(batch, cfg.d_model, feats.device)
    outs = []
    for step, (stt, end, left, right) in enumerate(chunk_plan(n_frames, sc, factor)):
        chunk_feats = feats[:, :, stt - left:end + right]
        lc, rc = round(left / factor), math.ceil(right / factor)
        seq_lengths = None  # batch 1: every position is valid
        if batch > 1:
            feat_len = (lengths - stt + left).clamp(0, chunk_feats.shape[2])
            seq_lengths = state.spkcache.shape[1] + state.fifo.shape[1] + (feat_len + factor - 1) // factor
        before = state
        state, out, t = stream_step(model, encode, state, chunk_feats.transpose(1, 2), lc, rc, sc, seq_lengths)
        outs.append(out)
        if hook is not None:
            hook(step, {"spkcache": before.spkcache, "fifo": before.fifo, "lc": lc, "rc": rc, **t})
    total = torch.cat(outs, dim=1)[:, :n_frames]
    if batch > 1:
        total = total * (torch.arange(n_frames, device=total.device)[None, :] < lengths[:, None])[..., None]
    return total

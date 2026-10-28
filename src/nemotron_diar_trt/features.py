"""Whole-recording mel features, computed in pieces for long inputs."""

from __future__ import annotations

import torch

from .modeling import MelFrontend

# Inputs longer than this get their features in pieces, with enough context that each piece's frames equal
# the whole-signal ones: NeMo's single STFT over a 12-20 h recording needs tens of GiB.
FEATURE_CHUNK_SAMPLES = 600 * 16000


@torch.no_grad()
def compute_features(frontend: MelFrontend, audio: torch.Tensor,
                     chunk: int = FEATURE_CHUNK_SAMPLES) -> torch.Tensor:
    """audio [N] float32 (any device; pieces are moved to the frontend's device) -> features [n_mels, N // hop].

    Inputs up to `chunk` samples go through one frontend call; longer ones are split into `chunk`-sample
    pieces, each with (n_fft // 2 // hop + 2) hops of context on both sides.
    """
    device = frontend.window.device
    n = int(audio.shape[0])
    hop, n_fft = frontend.cfg.hop_length, frontend.cfg.n_fft
    if n <= chunk:
        x = audio.to(device, torch.float32)[None]
        feats, seq_len = frontend(x, torch.tensor([n], device=device))
        return feats[0, :, :int(seq_len[0])]
    if chunk % hop:
        raise ValueError(f"chunk ({chunk}) must be a multiple of the hop ({hop})")
    context = (n_fft // 2 // hop + 2) * hop
    total = int(frontend.seq_len(torch.tensor([n]))[0])
    pieces = []
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        first_sample, last_sample = max(0, start - context), min(n, stop + context)
        piece = audio[first_sample:last_sample].to(device, torch.float32)[None]
        feats, _ = frontend(piece, torch.tensor([last_sample - first_sample], device=device))
        first = (start - first_sample) // hop
        count = (stop - start) // hop if stop < n else total - start // hop
        pieces.append(feats[0, :, first:first + count])
    return torch.cat(pieces, dim=1)

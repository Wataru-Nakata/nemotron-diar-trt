"""Speaker turns from per-frame probabilities.

Default: exact times (frame index x frame length in float64) and every turn.
nemo_compat=True reproduces NeMo's `diarize()` text output exactly (vad_utils.predlist_to_timestamps and
speaker_utils.generate_diarization_output_lines): times pass through float32 (`fl2int` / `int2fl`), so they
drift by up to ~1 ms past ~168 s, and every turn ending after 2**31 / 1e5 s (~21474.84 s, 5 h 58 m) is
silently dropped because `fl2int` overflows int32.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: int

    def line(self) -> str:
        return f"{self.start:.3f} {self.end:.3f} speaker_{self.speaker}"


def active_runs(probs: np.ndarray, onset: float = 0.5, offset: float = 0.5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per speaker: (start, end) frame indices of active runs, end exclusive. probs [T, n_spk].

    With onset >= offset (NeMo binarization_vectorized): p > onset switches on, p < offset switches off and
    values in between keep the previous state (initially off).
    """
    if onset < offset:
        raise NotImplementedError("onset < offset (toggling hysteresis) is not supported")
    probs = np.asarray(probs)
    n_frames, n_spk = probs.shape
    runs = []
    for k in range(n_spk):
        p = probs[:, k]
        on, off = p > onset, p < offset
        event = np.where(on | off, np.arange(1, n_frames + 1), 0)
        last = np.maximum.accumulate(event) if n_frames else event
        above = np.concatenate([[False], on])[last]
        diff = np.diff(np.concatenate([[0], above.astype(np.int8), [0]]))
        runs.append((np.flatnonzero(diff > 0), np.flatnonzero(diff < 0)))
    return runs


def turns_from_probs(probs, frame_seconds: float = 0.01, threshold: float = 0.5,
                     nemo_compat: bool = False) -> list[Turn]:
    """Turns grouped by speaker index, in time order within a speaker (NeMo's line order)."""
    probs = probs.float().cpu().numpy() if isinstance(probs, torch.Tensor) else np.asarray(probs, np.float32)
    runs = active_runs(probs, threshold, threshold)
    if nemo_compat:
        return [t for k, (s, e) in enumerate(runs) for t in _nemo_turns(s, e, k, frame_seconds)]
    return [Turn(float(s) * frame_seconds, float(e) * frame_seconds, k)
            for k, (starts, ends) in enumerate(runs) for s, e in zip(starts.tolist(), ends.tolist())]


def _nemo_turns(starts: np.ndarray, ends: np.ndarray, speaker: int, frame_seconds: float) -> list[Turn]:
    if len(starts) == 0:
        return []
    # predlist_to_timestamps: float32 seconds, + offset 0.0, Python round(., 2)
    start_t = torch.clamp(torch.from_numpy(starts).float() * frame_seconds - 0.0, min=0.0)
    end_t = torch.from_numpy(ends).float() * frame_seconds + 0.0
    valid = end_t > start_t
    seg = torch.stack([start_t[valid], end_t[valid]], dim=1) + 0.0
    ranges = [[round(s, 2), round(e, 2)] for s, e in seg.tolist()]
    return [Turn(s, e, speaker) for s, e in _merge_float_intervals(ranges)]


def _fl2int(x: float, decimals: int) -> int:
    return torch.round(torch.tensor([x * (10**decimals)]), decimals=0).int().item()


def _int2fl(x: int, decimals: int) -> float:
    return torch.round(torch.tensor([x / (10**decimals)]), decimals=decimals).item()


def _merge_float_intervals(ranges: list[list[float]], decimals: int = 5, margin: int = 2) -> list[list[float]]:
    ranges_int = []
    for x in ranges:
        stt, end = int(_fl2int(x[0], decimals) + margin), int(_fl2int(x[1], decimals))
        if stt < end:
            ranges_int.append([stt, end])
    return [[_int2fl(a - margin, decimals), _int2fl(b, decimals)] for a, b in _merge_int_intervals(ranges_int)]


def _merge_int_intervals(intervals: list[list[int]]) -> list[list[int]]:
    if len(intervals) <= 1:
        return intervals
    # NeMo sorts the start and end columns independently
    ordered = [[int(a), int(b)] for a, b in torch.sort(torch.tensor(intervals), dim=0)[0]]
    merged, (start, end) = [], ordered[0]
    for stt2, end2 in ordered[1:]:
        if end >= stt2:
            end = max(end2, end)
        else:
            merged.append([int(start), int(end)])
            start, end = stt2, max(end2, end)
    merged.append([int(start), int(end)])
    return merged

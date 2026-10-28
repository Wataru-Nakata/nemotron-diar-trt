"""Reference outputs, profile and determinism checks from NeMo itself.

Runs in a NeMo environment (tools/nemo_ref/pyproject.toml; scripts/gpu_job.pbs REF_CMD / $REF_PY) with this
repo's src on PYTHONPATH. The reference is NeMo's `SortformerEncLabelModel.diarize()` with the same streaming
settings as this package (`StreamingConfig()`). Inputs longer than 600 s get their features in 600 s pieces
with context, which gives the whole-signal features (checked by --whole) without NeMo's single STFT over the
whole recording.

Per item (lists from tools/prepare_data.py) it writes to <out>/<set>/:
  <id>.probs.npy   float32 [T, 8], the per-frame probabilities (10 ms) NeMo returned
  <id>.lines.txt   NeMo's turn strings ("start end speaker_k")
  <id>.steps.npz   per streaming step: encoder input layout and the selected speaker-cache indices
and <out>/summary.json with timings (pass B, no hooks), the in-process repeat comparison, whole-signal vs
chunked features and the per-function profile.

    python tools/nemo_ref.py --lists .cache/lists/short.jsonl ... --out outputs/x [--compare outputs/y]
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from nemotron_diar_trt.api import read_audio as read_mono
from nemotron_diar_trt.config import StreamingConfig
from nemotron_diar_trt.features import FEATURE_CHUNK_SAMPLES


def default_model() -> str:
    """The checkpoint: $NEMOTRON_DIAR_MODEL (a .nemo file or its directory), else the Hugging Face repo."""
    return os.environ.get("NEMOTRON_DIAR_MODEL", "nvidia/Nemotron-3-Diarization")


def load_items(paths):
    items = []
    for p in paths:
        items += [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
    return items


def read_audio(item):
    audio = read_mono(item["path"])
    start = int(round(item.get("start", 0.0) * 16000))
    stop = None if item.get("dur") is None else start + int(round(item["dur"] * 16000))
    return np.ascontiguousarray(audio[start:stop])


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def chunked_features(preprocessor, audio, length, device, chunk):
    """NeMo's preprocessor over `chunk`-sample pieces with enough context on both sides that every piece's
    frames equal the whole-signal ones (the same split as nemotron_diar_trt.features.compute_features)."""
    featurizer = preprocessor.featurizer
    hop, n_fft = featurizer.hop_length, featurizer.n_fft
    context = (n_fft // 2 // hop + 2) * hop
    n = int(length[0])
    total = int(featurizer.get_seq_len(torch.tensor([n]))[0])
    pieces = []
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        first_sample, last_sample = max(0, start - context), min(n, stop + context)
        piece = audio[:, first_sample:last_sample].to(device)
        feats, _ = preprocessor(input_signal=piece, length=torch.tensor([last_sample - first_sample], device=device))
        first = (start - first_sample) // hop
        count = (stop - start) // hop if stop < n else total - start // hop
        pieces.append(feats[:, :, first:first + count])
    feats = torch.cat(pieces, dim=2)
    return feats, torch.tensor([feats.shape[2]], device=device)


class NemoDiarizer:
    """NeMo's streaming Sortformer `diarize()` with this package's default streaming settings and threshold."""

    def __init__(self, model_path: str, device: str = "cuda", sc: StreamingConfig | None = None):
        from nemo.collections.asr.models import SortformerEncLabelModel
        from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig
        from nemo.collections.asr.parts.utils.vad_utils import load_postprocessing_from_yaml

        from nemotron_diar_trt.checkpoint import resolve_nemo

        sc = sc or StreamingConfig()
        self.settings = {k: getattr(sc, k) for k in ("chunk_len", "chunk_left_context", "chunk_right_context",
                                                       "fifo_len", "spkcache_len", "spkcache_update_period")}
        self.model = SortformerEncLabelModel.restore_from(str(resolve_nemo(model_path)), map_location=device).eval()
        for name, value in self.settings.items():
            setattr(self.model.sortformer_modules, name, value)
        self.model._check_streaming_parameters()
        self.feature_chunk = FEATURE_CHUNK_SAMPLES
        whole_signal = self.model.process_signal

        def process_signal(audio_signal, audio_signal_length):
            if audio_signal.shape[0] != 1 or audio_signal.shape[1] <= self.feature_chunk:
                return whole_signal(audio_signal, audio_signal_length)
            return chunked_features(self.model.preprocessor, audio_signal, audio_signal_length, self.model.device,
                                    self.feature_chunk)

        self.model.process_signal = process_signal
        postprocessing = load_postprocessing_from_yaml()
        postprocessing.onset = postprocessing.offset = sc.threshold
        self.options = DiarizeConfig(batch_size=1, num_workers=0, sample_rate=16000,
                                     postprocessing_params=postprocessing, include_tensor_outputs=True)

    def __call__(self, audio: np.ndarray) -> tuple[list[str], np.ndarray]:
        with torch.inference_mode():
            lines, preds = self.model.diarize(audio=[audio], override_config=self.options)
        return lines[0], preds[0][0].float().cpu().numpy()


class Recorder:
    """Monkeypatches NeMo methods to record per-step data and/or synchronized timings."""

    def __init__(self, model):
        self.model = model
        self.on = False
        self.timing = False
        self.reset()
        sm = self.model.sortformer_modules
        topk = sm._get_topk_indices

        def topk_hook(scores):
            idx, disabled = topk(scores)
            if self.on:
                self.steps["topk"].append(idx[0].to(torch.int16).cpu().numpy())
                self.steps["disabled"].append(disabled[0].cpu().numpy())
            return idx, disabled

        sm._get_topk_indices = topk_hook
        compress = sm._compress_spkcache

        def compress_hook(emb_seq, preds, mean_sil_emb, permute_spk=False):
            out = compress(emb_seq=emb_seq, preds=preds, mean_sil_emb=mean_sil_emb, permute_spk=permute_spk)
            if self.on and len(self.compress_io) < 4:
                self.compress_io.append({"emb_seq": emb_seq.cpu(), "preds": preds.cpu(), "spkcache": out[0].cpu(),
                                         "spkcache_preds": out[1].cpu()})
            return out

        sm._compress_spkcache = compress_hook
        step = self.model.forward_streaming_step

        def step_hook(processed_signal, processed_signal_length, streaming_state, total_preds, **kw):
            if self.on:
                self.steps["layout"].append((streaming_state.spkcache.shape[1], streaming_state.fifo.shape[1],
                                             processed_signal.shape[1], kw.get("left_offset", 0),
                                             kw.get("right_offset", 0)))
            return step(processed_signal=processed_signal, processed_signal_length=processed_signal_length,
                        streaming_state=streaming_state, total_preds=total_preds, **kw)

        self.model.forward_streaming_step = step_hook
        process_signal = self.model.process_signal

        def features_hook(audio_signal, audio_signal_length):
            out = process_signal(audio_signal, audio_signal_length)
            if self.on and self.keep_features:
                self.features = out[0][0, :, :int(out[1][0])].cpu()
            return out

        self.model.process_signal = features_hook
        self.keep_features = False
        for owner, name in [(self.model, "process_signal"), (self.model, "forward_streaming"),
                            (self.model, "forward_streaming_step"), (self.model, "_call_pre_encode"),
                            (self.model, "frontend_encoder"), (self.model, "forward_infer"),
                            (sm, "streaming_update"), (sm, "_compress_spkcache"), (sm, "downsample_preds"),
                            (self.model, "_diarize_output_processing")]:
            self._time(owner, name)

    def _time(self, owner, name):
        fn = getattr(owner, name)

        def timed(*a, **k):
            if not self.timing:
                return fn(*a, **k)
            sync()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            sync()
            self.profile[name][0] += time.perf_counter() - t0
            self.profile[name][1] += 1
            return out

        setattr(owner, name, timed)

    def reset(self):
        self.steps = defaultdict(list)
        self.compress_io = []
        self.features = None
        self.profile = defaultdict(lambda: [0.0, 0])


def run(diarizer, audio):
    sync()
    t0 = time.perf_counter()
    lines, probs = diarizer(audio)
    sync()
    return lines, probs, time.perf_counter() - t0


def compare(a, b):
    same_shape = a.shape == b.shape
    return {"same_shape": same_shape, "max_abs": float(np.abs(a - b).max()) if same_shape and a.size else None,
            "mask_diff_frames": int(((a > 0.5) != (b > 0.5)).any(1).sum()) if same_shape else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=default_model(), help=".nemo file, its directory or Hugging Face repo")
    ap.add_argument("--no-repeat", action="store_true", help="skip pass B (timing + in-process determinism)")
    ap.add_argument("--whole", nargs="*", default=[], help="ids to rerun with whole-signal features")
    ap.add_argument("--profile", nargs="*", default=[], help="ids to time per NeMo function")
    ap.add_argument("--torch-profile", default=None, help="id to run under torch.profiler")
    ap.add_argument("--features", nargs="*", default=[], help="ids whose NeMo features are saved")
    ap.add_argument("--compare", type=Path, default=None, help="earlier output dir to compare probs with")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"torch": torch.__version__,
               "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
               "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
               "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "items": {}}

    t0 = time.perf_counter()
    diarizer = NemoDiarizer(args.model, args.device)
    summary["model_load_seconds"] = time.perf_counter() - t0
    summary["inference_settings"] = diarizer.settings
    rec = Recorder(diarizer.model)
    items = load_items(args.lists)
    t0 = time.perf_counter()
    run(diarizer, read_audio(items[0])[: 30 * 16000])  # warm-up (flex attention compile)
    summary["warmup_seconds"] = time.perf_counter() - t0

    for item in items:
        audio = read_audio(item)
        seconds = len(audio) / 16000
        out_dir = args.out / item["set"]
        out_dir.mkdir(exist_ok=True)
        info = {"seconds": seconds}
        # pass A: hooks on (outputs and per-step data)
        rec.reset()
        rec.on, rec.keep_features = True, item["id"] in args.features
        lines, probs, wall_a = run(diarizer, audio)
        rec.on = False
        np.save(out_dir / f"{item['id']}.probs.npy", probs)
        (out_dir / f"{item['id']}.lines.txt").write_text("".join(line + "\n" for line in lines))
        info["turns"] = len(lines)
        np.savez(out_dir / f"{item['id']}.steps.npz", layout=np.array(rec.steps["layout"], np.int64),
                 topk=np.array(rec.steps["topk"]), disabled=np.array(rec.steps["disabled"]))
        if rec.compress_io:
            torch.save(rec.compress_io, out_dir / f"{item['id']}.compress.pt")
        if rec.features is not None:
            torch.save(rec.features, out_dir / f"{item['id']}.features.pt")
        info["wall_hooked"] = wall_a
        # pass B: no hooks; timing and in-process determinism
        if not args.no_repeat:
            lines_b, probs_b, wall_b = run(diarizer, audio)
            info["wall"] = wall_b
            info["rtfx"] = seconds / wall_b
            info["repeat"] = compare(probs, probs_b)
            info["repeat"]["same_lines"] = lines_b == lines
        if item["id"] in args.profile:
            rec.reset()
            rec.timing = True
            _, _, wall_p = run(diarizer, audio)
            rec.timing = False
            info["profile"] = {"wall": wall_p, **{k: {"seconds": v[0], "calls": v[1]}
                                                  for k, v in rec.profile.items()}}
        if item["id"] in args.whole:
            chunk = diarizer.feature_chunk
            diarizer.feature_chunk = 1 << 62
            lines_w, probs_w, wall_w = run(diarizer, audio)
            diarizer.feature_chunk = chunk
            info["whole_vs_chunked"] = compare(probs, probs_w)
            info["whole_vs_chunked"]["same_lines"] = lines_w == lines
            info["whole_vs_chunked"]["wall"] = wall_w
        if item["id"] == args.torch_profile:
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                run(diarizer, audio)
            (args.out / f"torch_profile_{item['id']}.txt").write_text(
                prof.key_averages().table(sort_by="cuda_time_total", row_limit=40) + "\n\n" +
                prof.key_averages().table(sort_by="count", row_limit=25))
        if args.compare is not None:
            other = args.compare / item["set"] / f"{item['id']}.probs.npy"
            if other.exists():
                info["cross_process"] = compare(probs, np.load(other))
                info["cross_process"]["same_lines"] = (
                    (args.compare / item["set"] / f"{item['id']}.lines.txt").read_text()
                    == (out_dir / f"{item['id']}.lines.txt").read_text())
        summary["items"][f"{item['set']}/{item['id']}"] = info
        print(json.dumps({"item": f"{item['set']}/{item['id']}", **{k: v for k, v in info.items()
                                                                   if k != "profile"}}), flush=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1))

    walls = [v for v in summary["items"].values() if "wall" in v]
    if walls:
        summary["total"] = {"seconds": sum(v["seconds"] for v in walls), "wall": sum(v["wall"] for v in walls)}
        summary["total"]["rtfx"] = summary["total"]["seconds"] / summary["total"]["wall"]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "items"}), flush=True)


if __name__ == "__main__":
    main()

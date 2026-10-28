"""Compare this package's torch path with NeMo outputs saved by tools/nemo_ref.py.

Runs in the NeMo reference env (same torch build as NeMo) or the TensorRT env (trt-* variants) with
this repo's src on PYTHONPATH. Per item and attention kernel it reports the "agreement ladder":
  step0_max_abs       max |p - p_nemo| over the first step's frames (no speaker cache yet)
  first_cache_diff    first compression whose selected cache frames differ (None: all equal)
  max_abs, mask_diff_frames   over the whole output (frames where any speaker's p > 0.5 differs)
  same_lines          turn strings equal to NeMo's (nemo_compat=True)
  same_track          the uint8 track round(p * 255) equal
plus features vs NeMo's (saved ids) and the compression on NeMo's own inputs (saved steps).

    python tools/parity.py --ref outputs/job-X/ref --lists .cache/lists/short.jsonl ... --out outputs/job-Y/parity
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nemo_ref import default_model, load_items, read_audio, sync  # noqa: E402

from nemotron_diar_trt import streaming  # noqa: E402
from nemotron_diar_trt.api import Diarizer  # noqa: E402
from nemotron_diar_trt.features import compute_features  # noqa: E402
from nemotron_diar_trt.postprocess import turns_from_probs  # noqa: E402


def track(p):
    return np.round(np.clip(p, 0, 1) * 255).astype(np.uint8)


def find(refs, item, suffix):
    for r in refs:
        p = r / item["set"] / f"{item['id']}{suffix}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, nargs="+", required=True)
    ap.add_argument("--lists", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", default=["flex", "sdpa", "sdpa-graph"],
                    help="attention kernel ('-graph': CUDA-graph runner, else the eager reference loop) or "
                         "trt-<dtype>[+torchhead][+nopad] (TensorRT encoder, CUDA graphs)")
    ap.add_argument("--model", default=default_model(), help=".nemo file, its directory or Hugging Face repo")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-probs", action="store_true", help="save turn lines only (large sets)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    items = [i for i in load_items(args.lists) if find(args.ref, i, ".probs.npy")]
    summary = {"torch": torch.__version__, "checks": {}, "items": {},
               "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}

    selections = []
    select = streaming.select_spkcache

    def record(preds, spkcache_len, cfg):
        idx, disabled = select(preds, spkcache_len, cfg)
        selections.append((idx[0].to(torch.int16).cpu().numpy(), disabled[0].cpu().numpy()))
        return idx, disabled

    for variant in args.variants:
        t0 = time.perf_counter()
        if variant.startswith("trt-"):  # TensorRT encoder (engine built on first use), CUDA graphs
            dtype, *flags = variant[4:].split("+")
            graphs = True
            diarizer = Diarizer.from_pretrained(args.model, backend="trt", dtype=dtype, device=args.device,
                                                nemo_compat=True, pad="nopad" not in flags,
                                                engine_head=False if "torchhead" in flags else None)
        else:
            attention, graphs = variant.split("-")[0], variant.endswith("-graph")
            diarizer = Diarizer.from_pretrained(args.model, backend="torch", device=args.device,
                                                attention=attention, nemo_compat=True, graphs=graphs)
        summary[f"load_seconds_{variant}"] = time.perf_counter() - t0
        diarizer.probabilities(read_audio(items[0])[: 30 * 16000])  # warm-up (compiles flex attention)
        model = diarizer.model

        if variant == args.variants[0]:
            for item in items:  # frontend and compression on NeMo's own tensors
                feats = find(args.ref, item, ".features.pt")
                if feats is not None:
                    ref = torch.load(feats)
                    ours = compute_features(model.frontend, torch.from_numpy(read_audio(item))).cpu()
                    summary["checks"][f"features/{item['id']}"] = {
                        "same_shape": ours.shape == ref.shape,
                        "equal": bool(ours.shape == ref.shape and torch.equal(ours, ref)),
                        "max_abs": float((ours - ref).abs().max()) if ours.shape == ref.shape else None}
                comp = find(args.ref, item, ".compress.pt")
                if comp is not None:
                    for k, io in enumerate(torch.load(comp)):
                        emb, kept = streaming.compress_spkcache(
                            io["emb_seq"].to(args.device), io["preds"].to(args.device), model.head.learnable_sil_emb,
                            diarizer.streaming.spkcache_len, model.cfg)
                        summary["checks"][f"compress/{item['id']}/{k}"] = {
                            "emb_equal": bool(torch.equal(emb.cpu(), io["spkcache"])),
                            "preds_equal": bool(torch.equal(kept.cpu(), io["spkcache_preds"]))}
            print(json.dumps(summary["checks"]), flush=True)

        if not graphs:  # graph replays run no Python: selections are recorded on eager loops only
            streaming.select_spkcache = record
        for item in items:
            audio = read_audio(item)
            selections.clear()
            sync()
            t0 = time.perf_counter()
            probs = diarizer.probabilities(audio)
            sync()
            wall = time.perf_counter() - t0
            lines = [t.line() for t in turns_from_probs(probs, nemo_compat=True)]
            exact = turns_from_probs(probs)
            ref = np.load(find(args.ref, item, ".probs.npy"))
            ref_lines = find(args.ref, item, ".lines.txt").read_text().splitlines()
            steps = np.load(find(args.ref, item, ".steps.npz"))
            first = None if not graphs else "n/a"
            for k, (idx, dis) in enumerate(selections if not graphs else []):
                if k >= len(steps["topk"]) or not (np.array_equal(idx, steps["topk"][k])
                                                   and np.array_equal(dis, steps["disabled"][k])):
                    first = k
                    break
            if first is None and len(selections) != len(steps["topk"]):
                first = min(len(selections), len(steps["topk"]))
            info = {"seconds": len(audio) / 16000, "wall": wall, "rtfx": len(audio) / 16000 / wall,
                    "same_shape": probs.shape == ref.shape, "compressions": len(selections),
                    "first_cache_diff": first}
            if probs.shape == ref.shape:
                d = np.abs(probs - ref)
                info.update(step0_max_abs=float(d[:2720].max()), max_abs=float(d.max()),
                            mask_diff_frames=int(((probs > 0.5) != (ref > 0.5)).any(1).sum()),
                            same_track=bool(np.array_equal(track(probs), track(ref))),
                            track_diff_values=int((track(probs) != track(ref)).sum()))
            info.update(same_lines=lines == ref_lines, lines=len(lines), ref_lines=len(ref_lines),
                        exact_turns=len(exact))
            out = args.out / variant / item["set"]
            out.mkdir(parents=True, exist_ok=True)
            if not args.no_probs:
                np.save(out / f"{item['id']}.probs.npy", probs)
            (out / f"{item['id']}.lines.txt").write_text("".join(x + "\n" for x in lines))
            # exact turns (every turn, also past 21474.84 s) for DER against NeMo's probabilities' exact turns
            for name, p in ((f"{variant}_exact", probs), ("nemo_exact", ref)):
                d = args.out / name / item["set"]
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{item['id']}.lines.txt").write_text("".join(t.line() + "\n" for t in turns_from_probs(p)))
            summary["items"][f"{variant}/{item['set']}/{item['id']}"] = info
            print(json.dumps({"item": f"{variant}/{item['set']}/{item['id']}", **info}), flush=True)
            (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
        streaming.select_spkcache = select
        del diarizer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

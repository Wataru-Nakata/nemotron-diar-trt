"""Speed of this package's `Diarizer.diarize` (probabilities + turns) per backend, on the files of some lists.

NeMo's speed on the same files comes from tools/nemo_ref.py in the same job (its pass B timings).
Per variant: load seconds (engine build included when the cache is empty), warm-up, then each file once:
wall seconds with the GPU synchronized, RTFx = audio seconds / wall seconds, peak GPU memory.

    python tools/bench.py --lists .cache/lists/ami.jsonl ... --variants trt-float16 torch --out outputs/x/bench.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nemo_ref import default_model, load_items, read_audio, sync  # noqa: E402

from nemotron_diar_trt.api import Diarizer  # noqa: E402


def make(variant, model):
    """trt-<dtype>[+torchhead][+nopad], torch (CUDA graphs) or torch-eager."""
    if variant.startswith("trt-"):
        dtype, *flags = variant[4:].split("+")
        return Diarizer.from_pretrained(model, backend="trt", dtype=dtype, pad="nopad" not in flags,
                                        engine_head=False if "torchhead" in flags else None)
    return Diarizer.from_pretrained(model, backend="torch", graphs=variant != "torch-eager")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["trt-float16", "torch"])
    ap.add_argument("--model", default=default_model(), help=".nemo file, its directory or Hugging Face repo")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile-item", default=None, help="id to run under torch.profiler per variant")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    items = load_items(args.lists)
    audios = [read_audio(i) for i in items]
    report = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "variants": {}}
    for variant in args.variants:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        diarizer = make(variant, args.model)
        load = time.perf_counter() - t0
        for _ in range(2):  # warm-up: graphs are captured on a step signature's second occurrence
            diarizer.diarize(audios[0][: 120 * 16000])
        files, total_s, total_wall, first = {}, 0.0, 0.0, []
        for item, audio in zip(items, audios):
            sync()
            t0 = time.perf_counter()
            result = diarizer.diarize(audio)
            sync()
            wall = time.perf_counter() - t0
            first.append(result.probs)
            seconds = len(audio) / 16000
            files[f"{item['set']}/{item['id']}"] = {"seconds": seconds, "wall": wall, "rtfx": seconds / wall,
                                                    "turns": len(result.turns)}
            total_s += seconds
            total_wall += wall
        report["variants"][variant] = {"load_seconds": load, "seconds": total_s, "wall": total_wall,
                                       "rtfx": total_s / total_wall,
                                       "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30, "files": files}
        streamer = diarizer.streamer
        if streamer is not None:  # where a step's time goes: graph replays, captures, eager steps
            streamer.time_steps, streamer.step_times = True, []
            repeat = 0.0  # second pass (graphs replayed) against the first: must be identical
            for audio, probs in zip(audios, first):
                repeat = max(repeat, float(abs(diarizer.diarize(audio).probs - probs).max()))
            streamer.time_steps = False
            report["variants"][variant]["repeat_max_abs"] = repeat
            report["variants"][variant]["graphs"] = len(streamer.graphs)
            kinds = {}
            for kind, sec in streamer.step_times:
                k = kinds.setdefault(kind, {"steps": 0, "seconds": 0.0})
                k["steps"] += 1
                k["seconds"] += sec
            for k in kinds.values():
                k["ms_per_step"] = 1000 * k["seconds"] / k["steps"]
            report["variants"][variant]["steps"] = kinds
        if args.profile_item:
            from torch.profiler import ProfilerActivity, profile

            audio = next(a for i, a in zip(items, audios) if i["id"] == args.profile_item)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                diarizer.diarize(audio)
            (args.out.parent / f"profile_{variant}.txt").write_text(
                prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
        print(json.dumps({"variant": variant, **{k: v for k, v in report["variants"][variant].items()
                                                if k != "files"}}), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        del diarizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

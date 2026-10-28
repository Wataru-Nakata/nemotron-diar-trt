"""DER of saved turn lines against reference RTTMs, and of each system against NeMo's output.

Runs on the CPU in tools/der's env (pyannote.metrics; a few hundred MB: create it in /tmp, delete it after):
    UV_PROJECT_ENVIRONMENT=/tmp/der-env-$USER uv run --project tools/der python tools/der.py \
        --lists .cache/lists/ami.jsonl ... --hyp nemo=outputs/job-X/ref,outputs/job-X/der \
        --hyp trt16=outputs/job-Y/parity_trt/trt-float16 [--vs nemo] [--out der.json]

Hypotheses are <dir>/<set>/<id>.lines.txt ("start end speaker_k", as NeMo's diarize() returns them).
Scoring: pyannote.metrics DiarizationErrorRate with overlap counted, collar 0.25 s and 0 s, within the
UEM (AMI's files; the whole recording elsewhere). --vs NAME also scores every other system with NAME's output
as the reference (collar 0), i.e. how much the turns differ from that implementation's.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import soundfile
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate


def read_rttm(path):
    ann = Annotation()
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if f and f[0] == "SPEAKER":
            start, dur = float(f[3]), float(f[4])
            ann[Segment(start, start + dur)] = f[7]
    return ann


def read_lines(path):
    ann = Annotation()
    for line in Path(path).read_text().splitlines():
        start, end, spk = line.split()
        ann[Segment(float(start), float(end))] = spk
    return ann


def read_uem(item):
    if item.get("uem"):
        tl = Timeline()
        for line in Path(item["uem"]).read_text().splitlines():
            f = line.split()
            tl.add(Segment(float(f[2]), float(f[3])))
        return tl
    duration = soundfile.info(item["path"]).duration
    start = item.get("start", 0.0)
    return Timeline([Segment(start, start + (item.get("dur") or duration - start))])


def find(dirs, item):
    for d in dirs:
        p = Path(d) / item["set"] / f"{item['id']}.lines.txt"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lists", nargs="+", required=True)
    ap.add_argument("--hyp", action="append", required=True, help="name=dir[,dir...]")
    ap.add_argument("--vs", default=None, help="also score the others against this system's output")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    systems = {h.split("=", 1)[0]: h.split("=", 1)[1].split(",") for h in args.hyp}
    items = [json.loads(line) for p in args.lists for line in Path(p).read_text().splitlines() if line.strip()]
    results = defaultdict(dict)
    for set_name in dict.fromkeys(i["set"] for i in items):
        subset = [i for i in items if i["set"] == set_name]
        for name, dirs in systems.items():
            for collar in (0.25, 0.0):
                if not all(i.get("rttm") for i in subset):
                    continue
                metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
                missing = 0
                for item in subset:
                    hyp = find(dirs, item)
                    if hyp is None:
                        missing += 1
                        continue
                    metric(read_rttm(item["rttm"]), read_lines(hyp), uem=read_uem(item))
                comp, total = metric.accumulated_, metric.accumulated_.get("total", 0.0)
                results[set_name][f"{name}/collar{collar}"] = {
                    "der": abs(metric), "files": len(subset) - missing, "missing": missing,
                    **({k: comp[k] / total for k in ("missed detection", "false alarm", "confusion")}
                       if total else {})}
            if args.vs and name != args.vs:
                metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
                for item in subset:
                    ref, hyp = find(systems[args.vs], item), find(dirs, item)
                    if ref is not None and hyp is not None:
                        metric(read_lines(ref), read_lines(hyp), uem=read_uem(item))
                results[set_name][f"{name}/vs_{args.vs}"] = {"der": abs(metric)}
        for key, val in results[set_name].items():
            print(f"{set_name:18s} {key:28s} " + " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in val.items()))
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

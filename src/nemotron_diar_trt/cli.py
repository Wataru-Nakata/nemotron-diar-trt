"""nemotron-diar-trt diarize <audio>... : print NeMo-style turn lines per file."""

from __future__ import annotations

import argparse
import os


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nemotron-diar-trt")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diarize", help="print 'start end speaker_k' lines for 16 kHz audio files")
    d.add_argument("audio", nargs="+")
    d.add_argument("--model", default=os.environ.get("NEMOTRON_DIAR_MODEL", "nvidia/Nemotron-3-Diarization"),
                   help=".nemo file, its directory or Hugging Face repo (default: $NEMOTRON_DIAR_MODEL or the repo)")
    d.add_argument("--backend", default="trt", choices=["trt", "torch"])
    d.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"],
                   help="TensorRT precision: float32 matches NeMo, float16 is ~3x faster (see README)")
    d.add_argument("--nemo-compat", action="store_true", help="reproduce NeMo's turn output exactly")
    args = ap.parse_args(argv)

    from .api import Diarizer

    diarizer = Diarizer.from_pretrained(args.model, backend=args.backend, dtype=args.dtype,
                                        nemo_compat=args.nemo_compat)
    for path in args.audio:
        result = diarizer.diarize(path)
        for line in result.lines:
            print(f"{path}\t{line}")


if __name__ == "__main__":
    main()

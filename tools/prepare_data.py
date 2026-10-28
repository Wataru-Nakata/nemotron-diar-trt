"""Write evaluation lists (.cache/lists/<set>.jsonl) and reference RTTMs for the downloaded / local datasets.

Each list line: {"id", "set", "path", "rttm" (or null), "uem" (or null), "start" (s), "dur" (s or null)}.
Sets: ami (AMI test, Mix-Headset, only_words references), voxconverse_test / voxconverse_dev,
csj (CSJ dialogues: both channels mixed, references from the TRN inter-pausal units, one speaker per
channel), ami_long (AMI test meetings concatenated per series and per two series: 2-5 h with 4 or 8
speakers), ami_all (all 16 in one 9 h recording), bench50 (50 VoxConverse dev recordings, for speed),
short (slices of AMI for quick checks).

Data locations: $NEMOTRON_DIAR_DATA (default ./data) holds ami/audio/<meeting>.Mix-Headset.wav,
ami/AMI-diarization-setup (git clone of BUTSpeechFIT/AMI-diarization-setup), voxconverse/{test,dev}/**/*.wav
and voxconverse/rttm-repo (git clone of joonson/voxconverse); the AMI concatenations are written to
$NEMOTRON_DIAR_DATA/ami_long. $CSJ_ROOT is the CSJ distribution (WAV/, TRN/), if you have it.

    python -I tools/prepare_data.py --out .cache
"""

import argparse
import json
import os
import re
from pathlib import Path

DATA = Path(os.environ.get("NEMOTRON_DIAR_DATA", "data"))
CSJ = Path(os.environ.get("CSJ_ROOT", DATA / "CSJ"))


def ami(out: Path) -> list[dict]:
    setup = DATA / "ami/AMI-diarization-setup"
    items = []
    for m in (setup / "lists/test.meetings.txt").read_text().split():
        wav = DATA / f"ami/audio/{m}.Mix-Headset.wav"
        if wav.exists():
            items.append({"id": m, "set": "ami", "path": str(wav),
                          "rttm": str(setup / f"only_words/rttms/test/{m}.rttm"),
                          "uem": str(setup / f"uems/test/{m}.uem"), "start": 0.0, "dur": None})
    return items


def voxconverse(out: Path, split: str) -> list[dict]:
    root = DATA / "voxconverse"
    wavs = {p.stem: p for p in sorted((root / split).rglob("*.wav")) if "__MACOSX" not in p.parts}
    items = []
    for rttm in sorted((root / "rttm-repo" / split).glob("*.rttm")):
        if rttm.stem in wavs:
            items.append({"id": rttm.stem, "set": f"voxconverse_{split}", "path": str(wavs[rttm.stem]),
                          "rttm": str(rttm), "uem": None, "start": 0.0, "dur": None})
    return items


AMI_LONG = {  # meetings of one series share their 4 participants; two series give 8 speakers
    "ES2004": ["ES2004a", "ES2004b", "ES2004c", "ES2004d"],
    "IS1009": ["IS1009a", "IS1009b", "IS1009c", "IS1009d"],
    "TS3003": ["TS3003a", "TS3003b", "TS3003c", "TS3003d"],
    "EN2002": ["EN2002a", "EN2002b", "EN2002c", "EN2002d"],
    "ES2004+IS1009": ["ES2004a", "ES2004b", "ES2004c", "ES2004d", "IS1009a", "IS1009b", "IS1009c", "IS1009d"],
    "TS3003+EN2002": ["TS3003a", "TS3003b", "TS3003c", "TS3003d", "EN2002a", "EN2002b", "EN2002c", "EN2002d"],
}


def ami_long(out: Path, groups: dict = AMI_LONG, set_name: str = "ami_long") -> list[dict]:
    """Long recordings with real labels: AMI test meetings concatenated (audio, RTTM and UEM offset)."""
    import numpy as np
    import soundfile as sf

    setup = DATA / "ami/AMI-diarization-setup"
    audio_dir = DATA / set_name
    refs = out / f"refs/{set_name}"
    audio_dir.mkdir(parents=True, exist_ok=True)
    refs.mkdir(parents=True, exist_ok=True)
    items = []
    for name, meetings in groups.items():
        wav, rttm, uem = audio_dir / f"{name}.wav", [], []
        offset, pieces = 0.0, []
        for m in meetings:
            x, sr = sf.read(DATA / f"ami/audio/{m}.Mix-Headset.wav", dtype="float32", always_2d=True)
            assert sr == 16000
            pieces.append(x.mean(axis=1))
            for line in (setup / f"only_words/rttms/test/{m}.rttm").read_text().splitlines():
                f = line.split()
                rttm.append(f"SPEAKER {name} 1 {float(f[3]) + offset:.3f} {f[4]} <NA> <NA> {f[7]} <NA> <NA>\n")
            for line in (setup / f"uems/test/{m}.uem").read_text().splitlines():
                f = line.split()
                uem.append(f"{name} 1 {float(f[2]) + offset:.3f} {float(f[3]) + offset:.3f}\n")
            offset += len(pieces[-1]) / sr
        if not wav.exists():
            sf.write(wav, np.concatenate(pieces), 16000, subtype="PCM_16")
        (refs / f"{name}.rttm").write_text("".join(rttm))
        (refs / f"{name}.uem").write_text("".join(uem))
        items.append({"id": name, "set": set_name, "path": str(wav), "rttm": str(refs / f"{name}.rttm"),
                      "uem": str(refs / f"{name}.uem"), "start": 0.0, "dur": None})
    return items


UNIT = re.compile(r"^\d{4} (\d+\.\d+)-(\d+\.\d+) ([LR]):")


def csj(out: Path) -> list[dict]:
    refs = out / "refs/csj"
    refs.mkdir(parents=True, exist_ok=True)
    items = []
    for trn in sorted((CSJ / "TRN/Form1/noncore").glob("D*.trn")):
        wav = CSJ / f"WAV/noncore/{trn.stem}.wav"
        if not wav.exists():
            continue
        lines = []
        for line in trn.read_bytes().decode("shift_jis", errors="replace").splitlines():
            m = UNIT.match(line)
            if m:
                start, end = float(m[1]), float(m[2])
                lines.append(f"SPEAKER {trn.stem} 1 {start:.3f} {end - start:.3f} <NA> <NA> {m[3]} <NA> <NA>\n")
        rttm = refs / f"{trn.stem}.rttm"
        rttm.write_text("".join(lines))
        items.append({"id": trn.stem, "set": "csj", "path": str(wav), "rttm": str(rttm), "uem": None,
                      "start": 0.0, "dur": None})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(".cache"))
    args = ap.parse_args()
    args.out = args.out.resolve()
    lists = args.out / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    sets = {"ami": ami(args.out), "voxconverse_test": voxconverse(args.out, "test"),
            "voxconverse_dev": voxconverse(args.out, "dev"), "csj": csj(args.out),
            "ami_long": ami_long(args.out),
            # all 16 test meetings in one 9 h recording (16 speakers, more than the model's 8): long enough to
            # show NeMo's turn output stopping at 5 h 58 m
            "ami_all": ami_long(args.out, {"AMI-test-16": [m for series in ("IS1009", "ES2004", "TS3003", "EN2002")
                                                           for m in AMI_LONG[series]]}, "ami_all")}
    # a speed benchmark: the first 50 VoxConverse dev recordings by id (1-16 min, 4.8 h)
    sets["bench50"] = [dict(i, set="bench50") for i in sorted(sets["voxconverse_dev"], key=lambda i: i["id"])[:50]]
    meeting = next((i for i in sets["ami"] if i["id"] == "ES2004a"), None)
    if meeting:
        sets["short"] = [dict(meeting, id=f"ES2004a_{dur}s", set="short", rttm=None, uem=None, dur=float(dur))
                         for dur in (20, 100)]
    for name, items in sets.items():
        (lists / f"{name}.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
        print(f"{name}: {len(items)}")


if __name__ == "__main__":
    main()

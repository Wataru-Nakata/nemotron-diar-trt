# nemotron-diar-trt

This package runs the streaming Sortformer speaker diarizer [`nvidia/Nemotron-3-Diarization`](https://huggingface.co/nvidia/Nemotron-3-Diarization)
(8 speakers, 10 ms output frames) quickly:
* **No NeMo dependency**: a plain-PyTorch reimplementation loads the official `.nemo` weights with `strict=True`.
* **TensorRT**: the encoder runs as a TensorRT engine, built on first use and cached.
* **CUDA graphs**: each steady streaming step is one graph replay.

On public benchmarks, the default (fp32) gives NeMo's turns on 479 of 504 recordings at 3.2× NeMo's speed on the
same GPU. fp16 is 10× (see Precision below).

```bash
pip install git+https://github.com/Wataru-Nakata/nemotron-diar-trt     # torch >= 2.10, TensorRT >= 10.16
```

```python
from nemotron_diar_trt import Diarizer

d = Diarizer.from_pretrained("nvidia/Nemotron-3-Diarization")   # or a .nemo file / its directory
r = d.diarize("meeting.wav")       # 16 kHz audio file, or a float32 numpy array / tensor of 16 kHz samples
r.probs                            # [frames, 8] float32 speaker probabilities, 10 ms per frame
r.turns                            # [Turn(start, end, speaker)], grouped by speaker, then by time
r.lines                            # NeMo's diarize() strings: "start end speaker_k"

Diarizer.from_pretrained(..., dtype="float16")      # ~3x faster again; turns differ from NeMo's in a few % of frames
Diarizer.from_pretrained(..., backend="torch")      # no TensorRT: fp32 PyTorch (+ CUDA graphs)
Diarizer.from_pretrained(..., nemo_compat=True)     # turn strings exactly as NeMo prints them (see below)
```

```bash
nemotron-diar-trt diarize a.wav b.wav             # or: python -m nemotron_diar_trt.cli diarize --model x.nemo a.wav
```

**Streaming settings**: the defaults are chunk 340, right context 40, FIFO 40, speaker cache 264 and update
period 300 (encoder frames of 80 ms), with threshold 0.5. Pass a `StreamingConfig` to change them.

**Inputs over 600 s**: features are computed in 600 s pieces with enough context that the result equals the
whole-signal features. NeMo computes them in one call; for 12 h of audio, its complex STFT alone is ~9 GB.

## Speed and agreement with NeMo

All measurements were made on one H200, with NeMo and this package in the same job. NeMo is
`SortformerEncLabelModel.diarize()` from [NeMo Speech](https://github.com/NVIDIA-NeMo/Speech) `cf724ac3` (fp32),
with the same streaming settings and the same 600 s feature pieces.
* **Speed**: 22.9 h of audio: 50 VoxConverse dev recordings (1–16 min) and six concatenations of AMI test
  meetings (2–5 h; see Accuracy).
* **RTFx** is audio seconds / wall seconds, warm, one recording at a time. It includes features, the copy to the
  host and the turns.
* **Load time**: once the engine is in the cache, or including the first-use build.
* **Identical turns**: recordings whose turn strings equal NeMo's, over AMI, CSJ and VoxConverse (504 recordings).
  Measured with `nemo_compat=True`. The default output has exact times, so its strings differ from NeMo's
  float32-rounded ones by up to 1 ms, and it keeps turns past 5 h 58 m.
* **DER vs NeMo**: the DER of the turns with NeMo's turns as the reference, per evaluation set.

| | RTFx | vs NeMo | load | identical turns | DER vs NeMo |
|---|---|---|---|---|---|
| NeMo | 990 | 1× | 15 s (+ imports) | – | – |
| **TensorRT fp32 (default)** | **3,165** | **3.2×** | 1.1 s / 37 s with build | 479 / 504 | ≤ 0.18% |
| TensorRT fp16 | 10,174 | 10.3× | 0.9 s / 99 s with build | 29 / 504 | 0.4–2.6% |
| PyTorch fp32 (`backend="torch"`) | 2,702 | 2.7× | 0.2 s | 474 / 504 | ≤ 0.38% |

**Where the speed comes from**: a step covers 27.2 s of audio.
* NeMo spends ~27 ms on it, ~26 ms of that in the encoder (fp32 FlexAttention and GEMMs).
* Averaged over the benchmark, including per-recording work, the fp32 engine takes ~8.6 ms per step and fp16
  ~2.7 ms. Each steady step is a single CUDA-graph replay that also holds the FIFO / speaker-cache update and the
  compression.
* Model load drops from NeMo's 15 s (plus its imports) to about 1 s.
* The command line on a 17.5 min meeting takes 46 s from process start to exit on an empty cache, including the
  engine build, and 3 s once the engine is cached.

**Bitwise check**: with NeMo's own attention kernel (`attention="flex"`, eager, slow), this implementation
reproduces NeMo bit for bit on the 16 AMI test meetings (9 h) and two clips:
* the probabilities and turn strings
* every speaker-cache selection

Everything else differs only by kernel rounding, which the cache compression can amplify (see Precision).

### Accuracy (DER %, collar 0.25 s / 0 s, overlap scored, pyannote.metrics)

| set | NeMo | TensorRT fp32 | TensorRT fp16 |
|---|---|---|---|
| AMI test, Mix-Headset, only_words references (16 meetings) | 24.37 / 25.96 | 24.37 / 25.96 | 24.39 / 25.98 |
| AMI test meetings concatenated (6 recordings, 2–5 h, 4 or 8 speakers) | 24.39 / 26.04 | 24.39 / 26.04 | 24.34 / 25.99 |
| CSJ dialogues (40, Japanese) | 8.39 / 12.99 | 8.39 / 12.99 | 8.35 / 12.95 |
| VoxConverse test (232) | 6.43 / 8.38 | 6.45 / 8.40 | 6.19 / 8.16 |
| VoxConverse dev (216) | 4.17 / 5.78 | 4.17 / 5.78 | 4.59 / 6.18 |

* **AMI concatenations**: the four meetings of each test series (same 4 participants) in one recording, and two
  series per recording (8 speakers). They test long-form behaviour with real labels.
* **VoxConverse**: up to 21 speakers against the model's 8. The numbers compare implementations, not the
  model's published results.

### Precision

**Why outputs diverge**: every step's speaker cache is chosen by hard decisions (p > 0.5, score > 0, top-k).
* A rounding difference near such a boundary changes which frames are kept.
* From then on, the run continues from a different cache.

**fp32 (default)**: rounding differences are ~1e-5, so this rarely happens.
* All 16 AMI meetings and 39 of 40 CSJ dialogues give NeMo's turns exactly.
* On the 2–5 h AMI concatenations, 0.03% of frames change.
* It can still happen on hard input. On all 16 AMI test meetings in one 9 h recording (16 speakers, twice the
  model's capacity), fp32 follows NeMo to within 4e-4 for 3.2 h. Then one cache decision flips, and 9.5% of the
  frames differ from there on. The PyTorch path flips at the same step.

**fp16**: rounding differences are ~1e-2.
* 0.3% of frames change on AMI and CSJ, 2–3% on VoxConverse, and 0.6% on the AMI concatenations (DER vs NeMo
  0.84%).
* Against the human labels it is as accurate on average (−0.24 / +0.42 pt on VoxConverse test / dev, ±0.05
  elsewhere), but it is not NeMo's output.

**The head's conv**: the encoder runs in TensorRT. With fp32, the head runs in PyTorch, because NeMo's
subpixel conv runs in TF32 (cuDNN's default) and a full-fp32 conv in the engine would differ by 3e-4.

## NeMo drops turns past 5 h 58 m

NeMo's `diarize()` text output (`speaker_utils.generate_diarization_output_lines` → `merge_float_intervals` →
`fl2int`) converts seconds × 10⁵ to int32 through a float32 tensor. That has two effects:
* Every turn ending after 2³¹ / 10⁵ s = 21,474.84 s is silently dropped, because the value overflows and the turn
  fails `stt < end`. Reported upstream as [NVIDIA-NeMo/Speech#16448](https://github.com/NVIDIA-NeMo/Speech/issues/16448).
* Earlier times drift by up to ~1 ms. Past 2²⁴ / 10⁵ ≈ 168 s they are float32-quantized, e.g. 16384.03 →
  `16384.029`.

The probabilities are fine. On the 16 AMI test meetings concatenated into one 9.06 h recording, NeMo returns
11,432 turns, the last ending at 21,474.4 s. NeMo's own probabilities converted here give 22,098 turns, up to the
end of the recording at 32,617.5 s; 10,666 of them end after 5 h 58 m. By default this package computes exact
times (frame index × 10 ms) and keeps every turn. `nemo_compat=True` reproduces NeMo's
strings exactly, drop included, which is what the agreement numbers above use.

## How it works

The streaming loop is NeMo's synchronous `forward_streaming`, step for step (see [docs/streaming.md](docs/streaming.md)):
1. The step's 27.2 s chunk (+3.2 s right context) is stacked and projected.
2. It is appended to `[speaker cache (264) | FIFO (40)]`.
3. The 684-frame sequence is re-encoded by the 31-layer Transformer (RoPE).
4. The 8× subpixel head gives 10 ms probabilities.
5. Popped FIFO frames go to the cache, which is compressed back to 264 frames.

**What differs from NeMo's loop**:
* **Fixed shapes**: all steps but the first have the same shapes. The first and steady steps are captured as
  CUDA graphs. A recording's last one or two steps run eagerly, padded to the same shape with a key mask, so
  TensorRT never re-plans.
* **Precision split**: only the encoder (and for fp16 the head) is in TensorRT. Pre-encode, the cache update and
  the compression stay fp32 PyTorch.
* **No growing buffers**: outputs go into one preallocated buffer instead of NeMo's growing `torch.cat`.

Engines are cached under `$NEMOTRON_DIAR_TRT_CACHE` (default `~/.cache/nemotron_diar_trt`), keyed by:
* GPU
* TensorRT version
* options
* `EXPORT_VERSION`

## License

Apache-2.0. The streaming algorithm and the cache compression follow NVIDIA NeMo (Apache-2.0). The model
weights are NVIDIA's and come with their own license (see the model card).

## Development

Layout:
* `src/nemotron_diar_trt/`
  * `modeling.py`: frontend, encoder, head
  * `streaming.py`: loop, cache update, compression
  * `runtime.py`: CUDA graphs
  * `trt.py`: export, build, run
  * `postprocess.py`: turns
* `tools/`: evaluation against NeMo. The NeMo reference runs in its own environment (`tools/nemo_env`, NeMo
  Speech `cf724ac3`, torch 2.14.0) and is never a dependency of the package.
* `scripts/`: PBS job helpers. Environments are packed as tarballs and unpacked on the node's local disk; local
  paths go in `.cache/job.env`.

```bash
uv run pytest                                      # CPU tests (tiny random models); the GPU tests run in jobs
scripts/build_env_tar.sh main                      # torch + TensorRT env -> .cache/envs/nemotron-env.tar.zst
scripts/build_env_tar.sh ref                       # NeMo reference env   -> .cache/envs/nemo-ref.tar.zst
NEMOTRON_DIAR_DATA=data python -I tools/prepare_data.py --out .cache    # evaluation lists and references
scripts/qsub_pool.sh "CMD=tools/eval.sh~@,REF_ENV=1"     # NeMo, agreement and speed, as one GPU job
UV_PROJECT_ENVIRONMENT=/tmp/der-env uv run --project tools/der python tools/der.py \
  --lists .cache/lists/ami.jsonl --hyp nemo=outputs/job-X/nemo --hyp trt=outputs/job-X/parity/trt-float32 --vs nemo
```

**Evaluation data** (downloaded or local, not in the repository):
* [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) Mix-Headset test meetings, with
  [BUTSpeechFIT/AMI-diarization-setup](https://github.com/BUTSpeechFIT/AMI-diarization-setup) references.
* [VoxConverse](https://github.com/joonson/voxconverse) dev / test.
* [CSJ](https://clrd.ninjal.ac.jp/csj/en/) dialogues (licensed; optional).

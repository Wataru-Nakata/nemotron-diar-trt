# What NeMo runs per chunk (Nemotron-3-Diarization, this package's default settings)

Reference: NeMo Speech `cf724ac3` ([NVIDIA-NeMo/Speech](https://github.com/NVIDIA-NeMo/Speech)). Paths below
are relative to `nemo/collections/asr/`. Settings:
`chunk_len 340, chunk_right_context 40, fifo_len 40, spkcache_len 264, spkcache_update_period 300`
(encoder frames of 80 ms), threshold 0.5, batch 1. Everything else comes from the checkpoint's
`model_config.yaml`.

## Model

* **Weights**: 363 tensors, all stored as bf16. NeMo builds fp32 modules and copies the bf16 values in. That
  includes the Hann `window` (400) and the mel filterbank `fb` [1, 128, 257], so both must come from the checkpoint.
* **Unused at inference**: `sortformer_modules.hidden_to_spks` and `activity_head` never feed the
  probabilities. There is no `transformer_encoder`, so `tf_d_model` 192 is only the head width.
* **Frontend** (`parts/preprocessing/features.py` `FilterbankFeatures.forward`):
  * pre-emphasis 0.97, then `torch.stft(n_fft 512, hop 160, win 400, center, pad_mode constant)`
  * magnitude, then `^2`, then `fb @`, then `log(x + 2**-24)`
  * `normalize: NA`, so no normalization.
  * `diarize()` sets dither 0 and `pad_to` 0.
  * `seq_len = N // 160`. The STFT yields one more frame, which `forward` cuts (`models/sortformer_diar_models.py`
    `forward`: `processed_signal[:, :, :length.max()]`).
  * For inputs over 600 s, the features can be computed in 600 s pieces with 480 samples of context on each side
    (`features.compute_features`). The result is identical to NeMo's single call, which needs tens of GiB for
    12–20 h.
* **Encoder** (`modules/transformer_encoder.py` `TransformerEncoder`):
  * `FeatureStacking` (`parts/submodules/subsampling.py`): zero-pad T to a multiple of 8, reshape
    `[T/8, 8*128]`, then `Linear(1024, 512, bias=False)`. The streaming loop calls it per chunk ("pre-encode").
  * Then `embed_norm` LayerNorm, then 31 pre-norm blocks:
    * `x += out_proj(attn(LN1 x))`
    * `x += W2 GELU(W1 LN2 x)` (exact GELU, 2048 hidden)
  * Then `final_norm`.
  * **Attention**: `w_qkv` (no bias) gives `[q|k|v]`, 8 heads of 64.
    * RoPE on q and k: NeoX half split, all 64 dims, base 1e4, positions 0..T-1. The cos/sin tables are built on
      the CPU in fp32 (`parts/submodules/multi_head_attention.py` `RotaryPositionalEncoding.create_pe`).
    * Kernel: FlexAttention compiled with `dynamic=True` (`modules/transformer_encoder_utils.py`), with a
      padding-only block mask and scale 1/8.
* **Head** (`modules/sortformer_modules.py`):
  * `encoder_proj` 512→192
  * `subpixel_upsample` Conv1d(192, 1536, k 3, pad 1), reshaped to `[8T, 192]`
  * `relu → Linear(192,192) → relu → Linear(192, 8) → sigmoid`
  * The result is 10 ms frames ("high resolution").
* **Numerics**: on the H200, `cudnn.allow_tf32` is True, so the conv runs in TF32. `matmul.allow_tf32` is
  False, so the Linears are IEEE fp32.

## Chunk loop (`models/sortformer_diar_models.py` `forward_streaming`, `forward_streaming_step`)

Features `[128, T]` are cut in `SortformerModules.streaming_feat_loader`. In feature frames:
`stt += 2720`, `end = min(stt + 2720, T)`, `right = min(320, T - end)`, no left context; there are `ceil(T/2720)`
steps. Each step does the following:

1. **Pre-encode the chunk** `feat[stt : end + right]`, giving `C' = ceil((end + right - stt)/8)` embeddings.
2. **Concatenate** `[spkcache | fifo | chunk]` (pre-encode space, before `embed_norm`). Lengths are
   0+0+380 at step 0 and 264+40+380 = **684** in steady state. They are shorter only in the last one or two steps.
3. **Encode** the whole concatenation (`bypass_pre_encode=True`), then apply the head. This recomputes everything
   every step; there is no KV cache. The output is `preds_hr [8·len, 8]`.
4. **Downsample**: `preds = avg_pool1d(preds_hr, 8, ceil_mode)` gives encoder-rate predictions for the
   state update.
5. **Output**: `preds_hr[(S+F)·8 : (S+F+L)·8]`, where `L = C' - ceil(right/8)` and S, F are the lengths
   *before* the update. The pieces are appended to `total_preds` with `torch.cat`, which copies quadratically.
6. **`streaming_update`** (`modules/sortformer_modules.py`), synchronous version:
   * `fifo_preds = preds[S:S+F]`. The FIFO predictions are refreshed every step.
   * Append the chunk's first L embeddings and predictions to the FIFO.
   * If `F + L > 40`, pop `min(max(300, L - 40 + F), F + L)` frames. That is 300 at step 0, then 340: the
     update period only matters at step 0. The popped embeddings and their *current* predictions go to the
     cache.
   * Until the first compression, the cache predictions are this step's `preds[:S]`. After it, they are the
     ones gathered at the last compression.
   * If the cache exceeds 264 frames (300 at step 0, 604 afterwards), compress it.
7. **`_compress_spkcache`**:
   * `per_spk = 264/8 - 1 = 32`; strong = 24, weak = 48, min_pos = 16.
   * `score = log(max(p,.25)) - log(max(1-p,.25)) + Σ_s log(max(1-p_s,.25)) - log .5`
   * `-inf` where `p <= 0.5`. If a speaker has at least 16 positive scores, its non-positive ones also become `-inf`.
   * `+0.05` for positions ≥ 264, i.e. the frames that just arrived.
   * Per speaker, top-24 gets `+2 ln 2`, then top-48 gets `+ln 2`.
   * Append one `+inf` silence slot per speaker.
   * Top-264 over the speaker-major flattening `[8 × (n+1)]`, sorted by index. Slots that are `-inf` or a
     silence slot get `learnable_sil_emb` and prediction 0.
   * The resulting cache is speaker-major blocks, time order within a block, and each block ends with its
     silence slot. Its RoPE positions follow this layout.
   * Inference uses no randomness: no speaker permutation and no score noise.
   * The `use_learnable_sil_emb` path never updates the running silence mean.

After the loop, `total_preds[:, :T]` is returned on the CPU as float32 `[1, T, 8]`.

## Postprocessing (`parts/utils/vad_utils.py`, `parts/utils/speaker_utils.py`)

* `binarization_vectorized`: `p > 0.5` switches on, `p < 0.5` switches off, and `p == 0.5` keeps the state.
* Start and end are `fp32(frame) * 0.01`, then Python `round(·, 2)`. Pads and minimum durations are 0, so
  `filtering` is a no-op.
* `generate_diarization_output_lines` → `merge_float_intervals(decimals 5)`:
  * `fl2int` is `torch.tensor([x*1e5])` (float32) → `.int()`.
  * Times are float32-quantized past ~168 s.
  * Every turn ending after 21,474.84 s is dropped (int32 overflow).
  * Lines are `"{s:.3f} {e:.3f} speaker_k"`, grouped by speaker.
* `nemotron_diar_trt` reproduces this with `nemo_compat=True`. By default it uses exact times and keeps every turn.

## Consequences for a fast implementation

* **Fixed shapes**:
  * Steady-state steps all have exactly the same shapes: input 684, pop 340, compression 604→264.
  * Step 0 has its own shape (380, pop 300, compression 300→264). Only the last one or two steps differ.
  * CUDA graphs therefore need no padding at batch 1.
* **Sensitivity**: the compression's discrete choices (`p > 0.5`, `score > 0`, three top-k) turn tiny numerical
  differences into a different cache. Parity is therefore checked per step: which cache frames are selected.

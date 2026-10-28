"""Model and streaming settings, read from the NeMo checkpoint's model_config.yaml."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    # preprocessor (AudioToMelSpectrogramPreprocessor; dither and pad_to are off at inference)
    sample_rate: int = 16000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 128
    preemph: float = 0.97
    log_zero_guard: float = 2**-24
    # encoder (TransformerEncoder: feature stacking, pre-norm blocks, RoPE)
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 31
    ff_dim: int = 2048
    subsampling: int = 8
    rope_base: float = 10000.0
    pos_max_len: int = 5000
    # Sortformer head and speaker-cache compression
    hidden: int = 192
    n_spk: int = 8
    upsample: int = 8  # high_resolution: one output frame per feature frame (10 ms)
    spkcache_sil_frames_per_spk: int = 1
    pred_score_threshold: float = 0.25
    scores_boost_latest: float = 0.05
    strong_boost_rate: float = 0.75
    weak_boost_rate: float = 1.5
    min_pos_scores_rate: float = 0.5
    max_index: int = 99999

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def frame_seconds(self) -> float:
        return self.hop_length / self.sample_rate

    @classmethod
    def from_nemo_yaml(cls, cfg: dict) -> ModelConfig:
        pre, enc, sm = cfg["preprocessor"], cfg["encoder"], cfg["sortformer_modules"]
        sr = int(pre.get("sample_rate", cfg.get("sample_rate", 16000)))
        unsupported = {
            "preprocessor.normalize": (pre.get("normalize", "NA"), "NA"),
            "preprocessor.window": (pre.get("window", "hann"), "hann"),
            "preprocessor.frame_splicing": (pre.get("frame_splicing", 1), 1),
            "encoder.subsampling": (enc.get("subsampling"), "feature_stacking"),
            "encoder.self_attention_model": (enc.get("self_attention_model"), "rope"),
            "encoder.attn_mode": (enc.get("attn_mode", "full"), "full"),
            "encoder.xscaling": (enc.get("xscaling", False), False),
            "encoder.qkv_bias": (enc.get("qkv_bias", False), False),
            "encoder.qk_norm": (enc.get("qk_norm", False), False),
            "encoder.pre_block_norm": (enc.get("pre_block_norm", True), True),
            "encoder.feat_out": (enc.get("feat_out", -1), -1),
            "encoder.rotary_fraction": (enc.get("rotary_fraction", 1.0), 1.0),
            "sortformer_modules.use_learnable_sil_emb": (sm.get("use_learnable_sil_emb"), True),
            "high_resolution": (cfg.get("high_resolution", False), True),
            "output_subsampling_factor": (cfg.get("output_subsampling_factor", 1), 1),
            "streaming_mode": (cfg.get("streaming_mode", False), True),
            "transformer_encoder": (cfg.get("transformer_encoder"), None),
        }
        bad = {k: v for k, (v, want) in unsupported.items() if v != want}
        if bad:
            raise NotImplementedError(f"checkpoint settings this implementation does not cover: {bad}")
        return cls(
            sample_rate=sr,
            n_fft=int(pre.get("n_fft", 512)),
            win_length=int(round(float(pre.get("window_size", 0.025)) * sr)),
            hop_length=int(round(float(pre.get("window_stride", 0.01)) * sr)),
            n_mels=int(pre.get("features", 128)),
            d_model=int(enc["d_model"]),
            n_heads=int(enc["n_heads"]),
            n_layers=int(enc["n_layers"]),
            ff_dim=int(float(enc.get("ff_expansion", 4.0)) * int(enc["d_model"])),
            subsampling=int(enc["subsampling_factor"]),
            rope_base=float(enc.get("rope_base", 10000.0)),
            pos_max_len=int(enc.get("pos_emb_max_len", 5000)),
            hidden=int(sm["tf_d_model"]),
            n_spk=int(sm["num_spks"]),
            upsample=int(enc["subsampling_factor"]),
            spkcache_sil_frames_per_spk=int(sm["spkcache_sil_frames_per_spk"]),
            pred_score_threshold=float(sm["pred_score_threshold"]),
            scores_boost_latest=float(sm["scores_boost_latest"]),
            strong_boost_rate=float(sm["strong_boost_rate"]),
            weak_boost_rate=float(sm["weak_boost_rate"]),
            min_pos_scores_rate=float(sm["min_pos_scores_rate"]),
            max_index=int(sm.get("max_index", 99999)),
        )


@dataclass(frozen=True)
class StreamingConfig:
    """Streaming settings in encoder frames (80 ms)."""

    chunk_len: int = 340
    chunk_left_context: int = 0
    chunk_right_context: int = 40
    fifo_len: int = 40
    spkcache_len: int = 264
    spkcache_update_period: int = 300
    threshold: float = 0.5  # onset = offset for the turns

    def __post_init__(self):
        if min(self.chunk_len, self.spkcache_len, self.spkcache_update_period) < 1:
            raise ValueError("chunk_len, spkcache_len and spkcache_update_period must be positive")
        if min(self.chunk_left_context, self.chunk_right_context, self.fifo_len) < 0:
            raise ValueError("contexts and fifo_len must be non-negative")

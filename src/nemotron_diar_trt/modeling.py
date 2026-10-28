"""Plain-PyTorch Nemotron-3-Diarization (streaming Sortformer): mel frontend, Transformer encoder, head.

Module and parameter names follow NeMo (FilterbankFeatures, TransformerEncoder, SortformerModules), so the
official weights load with strict=True after dropping the training-only heads. The arithmetic follows NeMo
op by op (same tensors, same order), so fp32 results can be compared bit for bit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class MelFrontend(nn.Module):
    """NeMo FilterbankFeatures at inference: pre-emphasis, centred STFT, power, mel, log; no normalization."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # Both come from the checkpoint (bf16-rounded values), not from torch.hann_window / librosa.
        self.register_buffer("window", torch.zeros(cfg.win_length))
        self.register_buffer("fb", torch.zeros(1, cfg.n_mels, cfg.n_fft // 2 + 1))

    def seq_len(self, length: torch.Tensor) -> torch.Tensor:
        """Valid frames: the centred STFT's last frame is masked (NeMo get_seq_len)."""
        return torch.floor_divide(length + self.cfg.n_fft // 2 * 2 - self.cfg.n_fft, self.cfg.hop_length).long()

    def forward(self, x: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, N] float32 waveform -> features [B, n_mels, N // hop + 1] (frames >= seq_len zeroed)."""
        cfg = self.cfg
        seq_len = self.seq_len(length)
        seq_len = torch.where(length == 0, torch.zeros_like(seq_len), seq_len)
        timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < length.unsqueeze(1)
        x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - cfg.preemph * x[:, :-1]), dim=1)
        x = x.masked_fill(~timemask, 0.0)
        x = torch.stft(x, n_fft=cfg.n_fft, hop_length=cfg.hop_length, win_length=cfg.win_length, center=True,
                       window=self.window.to(dtype=torch.float, device=x.device), return_complex=True,
                       pad_mode="constant")
        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1))
        x = x.pow(2.0)
        x = torch.matmul(self.fb.to(x.dtype), x)
        x = torch.log(x + cfg.log_zero_guard)
        mask = torch.arange(x.size(-1), device=x.device).repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1), 0.0)
        return x, seq_len


class FeatureStacking(nn.Module):
    """Stack `subsampling` consecutive feature frames (zero-padded at the end) and project them."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.factor = cfg.subsampling
        self.proj = nn.Linear(cfg.subsampling * cfg.n_mels, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, n_mels] -> [B, ceil(T / factor), d_model]."""
        b, t, c = x.shape
        pad = (self.factor - t % self.factor) % self.factor
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        return self.proj(x.reshape(b, (t + pad) // self.factor, c * self.factor))


def rope_tables(cfg: ModelConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin for positions 0..pos_max_len-1, computed on the CPU in fp32 exactly like NeMo."""
    d = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(0, d, 2, dtype=torch.float32, device="cpu") / d))
    positions = torch.arange(0, cfg.pos_max_len, dtype=torch.float32, device="cpu")
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


_flex = None


def flex_attention(q, k, v, key_lengths=None):
    """NeMo's attention kernel: FlexAttention compiled with dynamic shapes, padding-only block mask."""
    global _flex
    from torch.nn.attention.flex_attention import create_block_mask
    from torch.nn.attention.flex_attention import flex_attention as fa

    if _flex is None:
        _flex = torch.compile(fa, dynamic=True) if q.is_cuda else fa
    attn = _flex if q.is_cuda else fa
    b, _, t, _ = q.shape
    lengths = key_lengths if key_lengths is not None else torch.full((b,), t, device=q.device)

    def pad_mask(bi, h, q_idx, kv_idx):
        return kv_idx < lengths[bi]

    block_mask = create_block_mask(pad_mask, B=b, H=1, Q_LEN=t, KV_LEN=t, device=q.device)
    return attn(q, k, v, block_mask=block_mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.head_dim, self.d_model = cfg.n_heads, cfg.head_dim, cfg.d_model
        self.w_qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x, cos, sin, key_mask=None, key_lengths=None, backend: str = "sdpa"):
        """x [B, T, D]; cos/sin [T, head_dim]; key_mask [B, T] bool (True = valid) or None (all valid)."""
        b, t, _ = x.shape
        qkv = self.w_qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        cos, sin = cos.to(q.dtype).view(1, 1, t, self.head_dim), sin.to(q.dtype).view(1, 1, t, self.head_dim)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        if backend == "flex":
            out = flex_attention(q, k, v, key_lengths)
        elif backend == "math":  # written out for ONNX export (TensorRT fuses it); scale on the query only
            scores = torch.matmul(q * self.head_dim**-0.5, k.transpose(-2, -1))
            if key_mask is not None:
                scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
            out = torch.matmul(torch.softmax(scores, dim=-1), v)
        else:
            mask = None if key_mask is None else key_mask[:, None, None, :]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.d_model, cfg.ff_dim), nn.GELU(), nn.Dropout(0.0),
                                 nn.Linear(cfg.ff_dim, cfg.d_model))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x, cos, sin, key_mask=None, key_lengths=None, backend="sdpa"):
        x = x + self.attn(self.norm1(x), cos, sin, key_mask, key_lengths, backend)
        return x + self.ffn(self.norm2(x))


class Encoder(nn.Module):
    """TransformerEncoder; `forward` takes pre-encoded embeddings (NeMo's bypass_pre_encode=True path)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.pre_encode = FeatureStacking(cfg)
        self.embed_norm = nn.LayerNorm(cfg.d_model)
        self.layers = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.LayerNorm(cfg.d_model)
        cos, sin = rope_tables(cfg)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.attention = "sdpa"  # or "flex" (NeMo's kernel, for parity checks) or "math" (for export)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """x [B, T, D] pre-encoded -> [B, T, D]; lengths masks padded keys (None: all valid)."""
        t = x.shape[1]
        cos, sin = self.rope_cos[:t].to(x.dtype), self.rope_sin[:t].to(x.dtype)
        key_mask = None
        if lengths is not None and self.attention != "flex":
            key_mask = torch.arange(t, device=x.device)[None, :] < lengths[:, None]
        x = self.embed_norm(x)
        for layer in self.layers:
            x = layer(x, cos, sin, key_mask, lengths, self.attention)
        return self.final_norm(x)


class SortformerHead(nn.Module):
    """SortformerModules at inference: projection, subpixel upsampling to 10 ms, two-layer speaker MLP."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.upsample = cfg.upsample
        self.encoder_proj = nn.Linear(cfg.d_model, cfg.hidden)
        self.subpixel_upsample = nn.Conv1d(cfg.hidden, cfg.hidden * cfg.upsample, kernel_size=3, padding=1)
        self.first_hidden_to_hidden = nn.Linear(cfg.hidden, cfg.hidden)
        self.single_hidden_to_spks = nn.Linear(cfg.hidden, cfg.n_spk)
        self.learnable_sil_emb = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, enc: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """enc [B, T, D] -> speaker probabilities [B, T * upsample, n_spk] (zero beyond lengths * upsample).

        With lengths, padded positions are zeroed before the conv, so the last valid frames see the same zero
        padding as an unpadded row.
        """
        h = self.encoder_proj(enc)
        b, t, c = h.shape
        if lengths is not None:
            h = h * (torch.arange(t, device=enc.device)[None, :] < lengths[:, None]).unsqueeze(-1).to(h.dtype)
        h = self.subpixel_upsample(h.transpose(1, 2)).transpose(1, 2)
        h = h.reshape(b, t, self.upsample, c).reshape(b, t * self.upsample, c)
        h = self.first_hidden_to_hidden(F.relu(h))
        logits = self.single_hidden_to_spks(F.relu(h))
        preds = torch.sigmoid(logits)
        if lengths is not None:
            mask = torch.arange(t, device=enc.device)[None, :] < lengths[:, None]
            preds = preds * mask.repeat_interleave(self.upsample, dim=1).unsqueeze(-1)
        return preds


class SortformerModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.frontend = MelFrontend(cfg)
        self.encoder = Encoder(cfg)
        self.head = SortformerHead(cfg)


# NeMo prefixes -> ours; training-only heads that never feed the probabilities are dropped.
RENAMES = {"preprocessor.featurizer.": "frontend.", "encoder.": "encoder.", "sortformer_modules.": "head."}
UNUSED = ("sortformer_modules.hidden_to_spks.", "sortformer_modules.activity_head.")


def convert_nemo_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in sd.items():
        if key.startswith(UNUSED):
            continue
        for old, new in RENAMES.items():
            if key.startswith(old):
                out[new + key[len(old):]] = value
                break
        else:
            raise KeyError(f"unexpected checkpoint key {key}")
    return out

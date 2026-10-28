import math

import pytest
import torch

from nemotron_diar_trt.config import ModelConfig, StreamingConfig
from nemotron_diar_trt.modeling import SortformerModel, convert_nemo_state_dict
from nemotron_diar_trt.streaming import chunk_plan, compress_spkcache, stream_features

TINY = ModelConfig(n_fft=64, win_length=48, hop_length=16, n_mels=16, d_model=32, n_heads=2, n_layers=2, ff_dim=64,
                   hidden=12, n_spk=4, pos_max_len=600)
TINY_SC = StreamingConfig(chunk_len=6, chunk_right_context=2, fifo_len=2, spkcache_len=12, spkcache_update_period=5)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    model = SortformerModel(TINY).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.3)
    return model


def nemo_compress(emb_seq, preds, sil_emb, spkcache_len, cfg):
    """NeMo's _compress_spkcache (inference branch), op for op, as the reference for ours."""
    batch_size, n_frames, n_spk = preds.shape
    mean_sil_emb = sil_emb.unsqueeze(0).expand(batch_size, -1)
    per_spk = spkcache_len // n_spk - cfg.spkcache_sil_frames_per_spk
    strong = math.floor(per_spk * cfg.strong_boost_rate)
    weak = math.floor(per_spk * cfg.weak_boost_rate)
    min_pos = math.floor(per_spk * cfg.min_pos_scores_rate)
    log_probs = torch.log(torch.clamp(preds, min=cfg.pred_score_threshold))
    log_1_probs = torch.log(torch.clamp(1.0 - preds, min=cfg.pred_score_threshold))
    log_1_probs_sum = log_1_probs.sum(dim=2).unsqueeze(-1).expand(-1, -1, n_spk)
    scores = log_probs - log_1_probs + log_1_probs_sum - math.log(0.5)
    is_speech = preds > 0.5
    scores = torch.where(is_speech, scores, torch.tensor(float("-inf")))
    is_pos = scores > 0
    is_nonpos_replace = (~is_pos) * is_speech * (is_pos.sum(dim=1).unsqueeze(1) >= min_pos)
    scores = torch.where(is_nonpos_replace, torch.tensor(float("-inf")), scores)
    scores[:, spkcache_len:, :] += cfg.scores_boost_latest
    for n_boost, scale in ((strong, 2), (weak, 1)):
        _, topk = torch.topk(scores, n_boost, dim=1, largest=True, sorted=False)
        bi = torch.arange(batch_size).unsqueeze(1).unsqueeze(2)
        si = torch.arange(n_spk).unsqueeze(0).unsqueeze(0)
        scores[bi, topk, si] -= scale * math.log(0.5)
    pad = torch.full((batch_size, cfg.spkcache_sil_frames_per_spk, n_spk), float("inf"))
    scores = torch.cat([scores, pad], dim=1)
    n_frames = scores.shape[1]
    flat = scores.permute(0, 2, 1).reshape(batch_size, -1)
    values, idx = torch.topk(flat, spkcache_len, dim=1, sorted=False)
    idx = torch.where(values != float("-inf"), idx, torch.tensor(cfg.max_index))
    idx, _ = torch.sort(idx, dim=1)
    disabled = idx == cfg.max_index
    idx = torch.remainder(idx, n_frames)
    disabled += idx >= n_frames - cfg.spkcache_sil_frames_per_spk
    idx[disabled] = 0
    emb = torch.gather(emb_seq, 1, idx.unsqueeze(-1).expand(-1, -1, emb_seq.shape[2]))
    emb = torch.where(disabled.unsqueeze(-1), mean_sil_emb.unsqueeze(1).expand(-1, spkcache_len, -1), emb)
    kept = torch.gather(preds, 1, idx.unsqueeze(-1).expand(-1, -1, n_spk))
    kept = torch.where(disabled.unsqueeze(-1), torch.tensor(0.0), kept)
    return emb, kept


def test_chunk_plan_matches_nemo_loader():
    sc = StreamingConfig()
    plan = chunk_plan(10000, sc, 8)
    assert plan[0] == (0, 2720, 0, 320)
    assert plan[-1] == (8160, 10000, 0, 0)
    assert plan[-2] == (5440, 8160, 0, 320)
    plan = chunk_plan(5500, sc, 8)  # less than a full right context before the last chunk
    assert plan == [(0, 2720, 0, 320), (2720, 5440, 0, 60), (5440, 5500, 0, 0)]
    assert len(chunk_plan(2720, sc, 8)) == 1


def test_compress_matches_nemo_ops_with_ties():
    cfg, n_spk, d = ModelConfig(), 8, 16
    gen = torch.Generator().manual_seed(0)
    for trial in range(20):
        n = 300 if trial % 2 else 604
        preds = torch.rand(2, n, n_spk, generator=gen)
        # saturated / repeated values give exact score ties, as in real streams
        preds[:, ::3] = torch.where(preds[:, ::3] > 0.5, 1.0, 0.0)
        preds[:, 1::7, : trial % n_spk] = 0.9
        emb = torch.randn(2, n, d, generator=gen)
        sil = torch.randn(d, generator=gen)
        ours = compress_spkcache(emb, preds, sil, 264, cfg)
        ref = nemo_compress(emb, preds.clone(), sil, 264, cfg)
        assert torch.equal(ours[0], ref[0]) and torch.equal(ours[1], ref[1])


def test_state_dict_conversion_is_strict():
    model = tiny_model()
    nemo = {}
    for k, v in model.state_dict().items():
        prefix = {"frontend.": "preprocessor.featurizer.", "encoder.": "encoder.", "head.": "sortformer_modules."}
        for ours, theirs in prefix.items():
            if k.startswith(ours):
                nemo[theirs + k[len(ours):]] = v
    nemo["sortformer_modules.hidden_to_spks.weight"] = torch.zeros(4, 24)
    nemo["sortformer_modules.activity_head.1.bias"] = torch.zeros(3)
    fresh = SortformerModel(TINY)
    fresh.load_state_dict(convert_nemo_state_dict(nemo), strict=True)


def test_stream_shapes_and_batch_consistency():
    model = tiny_model()
    feats = torch.randn(2, TINY.n_mels, 300)
    lengths = torch.tensor([300, 300])
    out = stream_features(model, feats, lengths, TINY_SC)
    assert out.shape == (2, 300, TINY.n_spk)
    assert torch.all((out >= 0) & (out <= 1))
    single = stream_features(model, feats[:1], lengths[:1], TINY_SC)
    torch.testing.assert_close(out[:1], single, rtol=1e-5, atol=1e-6)


def test_stream_steps_cover_every_frame_once():
    model = tiny_model(1)
    seen = []
    for n in (37, 48, 49, 95, 200):
        feats = torch.randn(1, TINY.n_mels, n)
        steps = []
        out = stream_features(model, feats, torch.tensor([n]), TINY_SC,
                              hook=lambda i, t: steps.append(t["hr"].shape[1]))
        assert out.shape[1] == n
        seen.append(len(steps))
    assert seen == [len(chunk_plan(n, TINY_SC, 8)) for n in (37, 48, 49, 95, 200)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_streamer_matches_eager_loop():
    from nemotron_diar_trt.runtime import GraphStreamer

    model = tiny_model(2).cuda()
    streamer = GraphStreamer(model, TINY_SC)
    for n in (300, 300, 333, 300, 47):  # graphs are captured on a signature's second occurrence
        feats = torch.randn(TINY.n_mels, n, device="cuda")
        ref = stream_features(model, feats[None], torch.tensor([n], device="cuda"), TINY_SC)[0]
        assert torch.equal(streamer(feats), ref)
    assert streamer.graphs


def test_padded_steps_match_unpadded():
    from nemotron_diar_trt.streaming import StreamingState, stream_step, torch_encode

    model = tiny_model(3)
    encode = torch_encode(model)
    feats = torch.randn(1, 300, TINY.n_mels)
    state = StreamingState.empty(1, TINY.d_model, "cpu")
    for (stt, end, left, right) in chunk_plan(300, TINY_SC, 8):  # same input state for both, every step
        chunk = feats[:, stt - left:end + right]
        ref_state, ref, _ = stream_step(model, encode, state, chunk, 0, -(-right // 8), TINY_SC)
        pad_state, pad, t = stream_step(model, encode, state, chunk, 0, -(-right // 8), TINY_SC, pad_to=24)
        assert t["hr"].shape[1] == 24 * 8
        torch.testing.assert_close(pad, ref, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(pad_state.fifo, ref_state.fifo)
        state = ref_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_padded_graph_streamer_close_to_eager():
    from nemotron_diar_trt.runtime import GraphStreamer

    model = tiny_model(4).cuda()
    streamer = GraphStreamer(model, TINY_SC, pad_to=24)
    for n in (300, 300, 133, 47):
        feats = torch.randn(TINY.n_mels, n, device="cuda")
        ref = stream_features(model, feats[None], torch.tensor([n], device="cuda"), TINY_SC)[0]
        torch.testing.assert_close(streamer(feats), ref, rtol=1e-3, atol=1e-4)

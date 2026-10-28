import numpy as np

from nemotron_diar_trt.postprocess import active_runs, turns_from_probs


def probs_with_runs(n_frames, runs, n_spk=2):
    p = np.zeros((n_frames, n_spk), np.float32)
    for spk, s, e in runs:
        p[s:e, spk] = 0.9
    return p


def test_hysteresis_keeps_state_at_exact_threshold():
    p = np.array([[0.2], [0.7], [0.5], [0.5], [0.3], [0.5], [0.8], [0.1]], np.float32)
    (starts, ends), = active_runs(p)
    assert starts.tolist() == [1, 6] and ends.tolist() == [4, 7]


def test_exact_and_nemo_times():
    p = probs_with_runs(2_200_000, [(0, 307, 512), (1, 1638403, 1638500), (0, 2147400, 2147600)])
    exact = turns_from_probs(p)
    assert [t.line() for t in exact] == ["3.070 5.120 speaker_0", "21474.000 21476.000 speaker_0",
                                         "16384.030 16385.000 speaker_1"]
    nemo = turns_from_probs(p, nemo_compat=True)
    # float32 times past ~168 s, and the turn ending after 21474.84 s is dropped (int32 overflow in fl2int)
    assert [t.line() for t in nemo] == ["3.070 5.120 speaker_0", "16384.029 16385.000 speaker_1"]


def test_turns_grouped_by_speaker_then_time():
    p = probs_with_runs(1000, [(1, 10, 20), (0, 500, 600), (0, 30, 40), (1, 700, 800)])
    lines = [t.line() for t in turns_from_probs(p, nemo_compat=True)]
    assert lines == ["0.300 0.400 speaker_0", "5.000 6.000 speaker_0", "0.100 0.200 speaker_1",
                     "7.000 8.000 speaker_1"]
    assert lines == [t.line() for t in turns_from_probs(p)]

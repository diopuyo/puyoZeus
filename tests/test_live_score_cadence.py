"""実観測間隔への適応、従来30/60fps同値、欠測と境界の回帰検査。"""
from dataclasses import asdict
import json
import pytest
from src.b3_score_ledger import ScoreLedgerV2, ScoreLedger


@pytest.mark.parametrize('fps', [30, 60])
def test_dense_observations_unchanged(fps: int) -> None:
    old, live = ScoreLedgerV2(adaptive_score_cadence=False), ScoreLedgerV2(adaptive_score_cadence=True)
    for n in range(240):
        score = None if n == 40 else (9000 if n == 70 else n // 5)
        assert old._adopt_score(0, score, n/fps) == live._adopt_score(0, score, n/fps)
        assert asdict(old.st[0]) == asdict(live.st[0])


def test_sparse_jitter_spike_missing_and_duplicate() -> None:
    ledger = ScoreLedgerV2(adaptive_score_cadence=True)
    for t, score, expected in [(0., 0, None), (.1, 0, 0), (.27, 9999, 0),
                               (.48, 15, 15), (.48, 25, 15), (.6, None, 15),
                               (.8, 20, 15), (1.1, 21, 20)]:
        ledger._adopt_score(0, score, t)
        assert ledger.st[0].adopted == expected


def test_game_reset_and_causal_prefix() -> None:
    ledger = ScoreLedger(adaptive_score_cadence=True)
    frames = [dict(t_sec=t, frame=i, game_idx=game, match_active=True,
                   sides=[dict(side=s, state='STABLE', score=score) for s in ('1P', '2P')])
              for i, (t, game, score) in enumerate([(0., 1, 0), (.2, 1, 0),
                  (.4, 1, 10), (.7, 1, 12), (1., 2, 0), (1.2, 2, 0)])]
    snapshots = [ledger.update(frame) for frame in frames]
    assert snapshots[4]['history_waiting']
    assert not snapshots[5]['history_waiting']
    for end in range(1, len(frames)+1):
        replay = ScoreLedger(adaptive_score_cadence=True)
        options = dict(sort_keys=True, default=str)  # NaNと監査イベントも決定的な表現で照合する。
        assert json.dumps([replay.update(f) for f in frames[:end]], **options) == json.dumps(snapshots[:end], **options)


def test_explicit_off_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PUYO_LIVE_SCORE_CADENCE', '1')
    assert ScoreLedgerV2().adaptive_score_cadence
    ledger = ScoreLedgerV2(adaptive_score_cadence=False)
    for t in (0., .1, .3):
        ledger._adopt_score(0, 0, t)
    assert ledger.st[0].adopted is None

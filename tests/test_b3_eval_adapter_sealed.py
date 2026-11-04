"""B3 採点口アダプタ (表示格子・原票書出し・既存採点器呼出) と封印開封手順のテスト。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.b3_eval_adapter import (EvaluatorTrace, SOURCE_UNAVOIDABLE, death_events_from_trace, resample_hold,
                                 run_false_death_scan, score_m_time_agree, write_replay_record)
from src.b3_eval_sealed import (OpeningPlan, SealedError, build_manifest, manifest_sha, open_sealed, verify_manifest)

EXEV = Path("C:/Users/Example/.codex/worktrees/exev/puyo_analyzer")
EXEV_LINUX = Path("/mnt/c/Users/Example/.codex/worktrees/exev/puyo_analyzer")


def _exev_root():
    for r in (EXEV, EXEV_LINUX):
        if (r / "scripts/eval_set_score_20261001.py").exists():
            return r
    return None


def test_resample_hold_step_function_and_no_lookahead():
    tr = EvaluatorTrace(np.array([10.0, 10.5, 12.0]), np.array([0.2, 0.7, 0.9]))
    out = resample_hold(tr, 9.0, 13.0, hz=2.0)       # 格子 9.0, 9.5, 10.0, ...
    d = dict(zip(out["t_sec"].tolist(), out["display_p1"].tolist()))
    assert d[9.0] == 0.5 and d[9.5] == 0.5           # 最初の評価行より前は null = 0.5 (未来値で埋めない)
    assert d[10.0] == 0.2 and d[10.5] == 0.7 and d[11.5] == 0.7 and d[12.0] == 0.9 and d[12.5] == 0.9
    assert np.all(out["display_adv"] == pytest.approx((out["display_p1"] - 0.5) * 200))
    assert out["source"][:2].tolist() == ["null", "null"] and set(out["source"][2:].tolist()) == {"prob"}
    far = resample_hold(tr, 13.0, 16.0, hz=1.0)      # 最終行 (12.0) から 2 秒を超えると null (2 秒ちょうどは保持)
    assert far["display_p1"].tolist() == [0.9, 0.9, 0.5] and far["source"].tolist() == ["prob", "prob", "null"]


def test_trace_validation():
    with pytest.raises(ValueError):
        EvaluatorTrace(np.array([1.0, 0.5]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        EvaluatorTrace(np.array([1.0]), np.array([np.nan]))


def test_write_record_and_death_source_and_events(tmp_path):
    tr = EvaluatorTrace(np.array([0.0, 1.0, 2.0]), np.array([0.5, 0.3, 0.01]), dead_side=np.array([0, 0, 1]),
                        dead_source=np.array(["", "", SOURCE_UNAVOIDABLE]))
    info = write_replay_record(tmp_path / "rec", tr, [(0.0, 3.0)],
                               events=death_events_from_trace(tr, lambda t: 1))
    z = np.load(tmp_path / "rec/display.npz")
    assert set(z.files) == {"t_sec", "display_p1", "display_adv", "source"}
    assert (z["source"][z["t_sec"] >= 2.0] == SOURCE_UNAVOIDABLE).all() and (z["source"][z["t_sec"] < 2.0] == "prob").all()
    ev = [json.loads(l) for l in (tmp_path / "rec/events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ev[0]["values"][0]["dead_sides"] == ["1P"] and ev[0]["values"][0]["t_sec"] == 2.0
    assert info["frames"] == len(z["t_sec"]) and len(info["display_sha256"]) == 64
    stale = write_replay_record(tmp_path / "e", tr, [(100.0, 101.0)])      # 評価行が無い窓は全 null (消さない)
    assert set(np.load(tmp_path / "e/display.npz")["source"].tolist()) == {"null"} and stale["frames"] > 0


@pytest.mark.skipif(_exev_root() is None, reason="exev worktree が無い環境")
def test_existing_scorer_reads_adapter_output(tmp_path):
    root = _exev_root()
    games = [{"game": 1, "winner": "1P", "start": 0.0, "end": 10.0, "part": 1},
             {"game": 2, "winner": "2P", "start": 10.0, "end": 20.0, "part": 1}]
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"games": games}), encoding="utf-8")
    t = np.arange(0.0, 20.0, 0.05)
    p = np.where(t < 10.0, 0.8, 0.3)                       # 両試合とも当たり
    tr = EvaluatorTrace(t, p)
    write_replay_record(tmp_path / "r", tr, [(0.0, 20.0)])
    res = score_m_time_agree(root, [tmp_path / "r" / "display.npz"], labels)
    s = res["summary"]
    expect = (-np.log(0.8) + -np.log(0.7)) / 2
    assert s["games"] == 2 and s["m_time"] == pytest.approx(expect, rel=1e-6)
    assert s["m_agree"] == pytest.approx(1.0)               # 終盤1/3 の符号が全フレーム勝者側


# ---- 封印 ----
def _sealed_root(tmp_path):
    root = tmp_path / "sealed"
    (root / "receipts").mkdir(parents=True)
    (root / "TARGETS.json").write_text("[1]", encoding="utf-8")
    (root / "receipts/43.json").write_text('{"a": 1}', encoding="utf-8")
    return root


def _plan(**kw):
    base = dict(prereg_commit="8f0294a", candidate_sha="c" * 64, reference_sha="r" * 64, gate_conditions_sha="g" * 64,
                labels_sha="l" * 64, executor="test", purpose="unit test")
    base.update(kw)
    return OpeningPlan(**base)


def test_manifest_detects_tamper_and_missing(tmp_path):
    root = _sealed_root(tmp_path)
    m = build_manifest(root, ["TARGETS.json", "receipts/43.json"])
    assert verify_manifest(root, m) == []
    (root / "TARGETS.json").write_text("[2]", encoding="utf-8")
    bad = verify_manifest(root, m)
    assert bad and bad[0]["problem"] == "sha_mismatch"
    (root / "receipts/43.json").unlink()
    assert {b["problem"] for b in verify_manifest(root, m)} == {"sha_mismatch", "missing"}


def test_dry_run_never_scores_and_reports_token(tmp_path):
    root = _sealed_root(tmp_path)
    m = build_manifest(root, ["TARGETS.json"])
    called = []
    r = open_sealed(root, m, _plan(), ["a", "b"], lambda n: called.append(n) or {}, tmp_path / "rec.json")
    assert r["executed"] is False and called == [] and r["expected_token"].startswith("OPEN-B3-SEALED-")
    assert not (tmp_path / "rec.json").exists()


def test_execute_requires_matching_token_and_clean_manifest(tmp_path):
    root = _sealed_root(tmp_path)
    m = build_manifest(root, ["TARGETS.json"])
    plan = _plan()
    called = []
    with pytest.raises(SealedError):
        open_sealed(root, m, plan, ["a"], lambda n: called.append(n) or {}, tmp_path / "rec.json", token="wrong", dry_run=False)
    assert called == []
    tok = plan.token(manifest_sha(m))
    assert tok != _plan(candidate_sha="x" * 64).token(manifest_sha(m))        # 候補が違えば token も違う
    (root / "TARGETS.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(SealedError):
        open_sealed(root, m, plan, ["a"], lambda n: called.append(n) or {}, tmp_path / "rec.json", token=tok, dry_run=False)
    assert called == [] and not (tmp_path / "rec.json").exists()


def test_execute_scores_all_sets_in_one_pass_records_before_return_and_refuses_reopen(tmp_path):
    root = _sealed_root(tmp_path)
    m = build_manifest(root, ["TARGETS.json"])
    plan = _plan()
    tok = plan.token(manifest_sha(m))
    order = []

    def score(name):
        order.append(name)
        if name == "b":
            raise RuntimeError("壊れたセット")
        return {"ll": 0.6}

    rec = tmp_path / "rec.json"
    r = open_sealed(root, m, plan, ["a", "b", "c"], score, rec, token=tok, dry_run=False, now=lambda: 123.0)
    assert order == ["a", "b", "c"]                                          # b が壊れても c まで一括で採点する
    assert r["results"]["b"]["status"] == "broken" and r["results"]["c"]["status"] == "ok"
    saved = json.loads(rec.read_text(encoding="utf-8"))
    assert saved["opened_at_unix"] == 123.0 and saved["opening_complete"] is False and saved["executor"] == "test"
    assert saved["plan"]["candidate_sha"] == "c" * 64 and saved["manifest_sha256"] == manifest_sha(m)
    assert "0.6" not in json.dumps(saved)                                      # 開封記録に成績を書かない
    with pytest.raises(SealedError):
        open_sealed(root, m, plan, ["a"], score, rec, token=tok, dry_run=False)   # 再開封は拒否


def test_false_death_scan_subprocess_missing_root_fails_loudly(tmp_path):
    with pytest.raises(Exception):
        run_false_death_scan(tmp_path / "nowhere", [tmp_path], [], python=sys.executable)

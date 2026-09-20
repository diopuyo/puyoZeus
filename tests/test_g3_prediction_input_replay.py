"""予測入力の再生台の検査 (2026-09-17)。

この台の役目は「窓を直したらどれだけ観測できるか」の見積もりだが、
**その前に既知の失敗を再現できること**が条件 (`.claude/rules/01-verification-ladder.md` 原則3)。
再現の判定そのものが壊れていれば、見積もりは全部無意味になる。ここではその判定を固定する。

動画も実走も使わない。数秒で終わる (段1)。
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts import g3_prediction_input_replay as R


def make_run_dir(tmp_path: Path, *, adoptions: int = 1, rejections: int = 1096,
                 sides: tuple[int, int] = (508, 588),
                 native: tuple[int, int] = (32494, 36900)) -> Path:
    """v12 相当の成果物を最小限だけ作る。"""
    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    one_p, two_p = sides
    rejects = []
    for index in range(rejections):
        side = "1P" if index < one_p else "2P"
        rejects.append(dict(
            frame=str(3812 + index * 26), side=side, tokens=[],
            native_first=str(native[0]), native_last=str(native[1]),
            reason="outside_original_directional_scope", adoption_permission="False",
        ))
    payload = dict(
        adoptions=[dict(adoption_frame="34796", segment_id="1P:epoch:8:segment:23")
                   for _ in range(adoptions)],
        unwitnessed_appends=rejects,
    )
    (run / "BASELINE_ADOPTION_STATUS.json").write_text(
        json.dumps(payload), encoding="utf-8")
    calls = [dict(frame=34796 + i * 2, side="1P", make_invoked=True, made=False)
             for i in range(182)]
    calls += [dict(frame=i, side="2P", make_invoked=False, made=False)
              for i in range(36300 - 182)]
    (run / "CONDITIONAL_CURRENT_CALLS.json").write_text(
        json.dumps(calls), encoding="utf-8")
    return run


def test_reproduces_known_failure(tmp_path: Path) -> None:
    """v12 と同じ値なら「再現できた」と判定する。"""
    run = make_run_dir(tmp_path)
    problems = R.reproduce_known(R.load_adoption(run), R.load_make_calls(run))
    assert problems == [], problems


def test_detects_when_numbers_drift(tmp_path: Path) -> None:
    """既知と1つでもずれたら、必ず名前つきで落とす。黙って通さない。"""
    run = make_run_dir(tmp_path, adoptions=2)
    problems = R.reproduce_known(R.load_adoption(run), R.load_make_calls(run))
    assert len(problems) == 1
    assert "採用件数" in problems[0]


def test_detects_wrong_window(tmp_path: Path) -> None:
    """窓の値が違えば落とす。ここがこの調査の核心なので必ず検査する。"""
    run = make_run_dir(tmp_path, native=(0, 36298))
    problems = R.reproduce_known(R.load_adoption(run), R.load_make_calls(run))
    assert any("窓" in p for p in problems)


def test_rejections_are_all_before_window_start(tmp_path: Path) -> None:
    """却下が全件「窓より手前」であることが、この欠陥の決定的な証拠。"""
    run = make_run_dir(tmp_path)
    adoption = R.load_adoption(run)
    low, high = adoption["rejection_frame_range"]
    first, _ = adoption["native_scope"]
    assert low >= 0 and high < first, "却下が窓の中にもあるなら別の原因がある"
    assert adoption["rejection_reasons"] == {
        "outside_original_directional_scope": 1096
    }


def test_side_breakdown_is_carried(tmp_path: Path) -> None:
    """2P が 0 採用であることを見落とさないよう、側の内訳を必ず持つ。"""
    adoption = R.load_adoption(make_run_dir(tmp_path))
    assert adoption["rejection_sides"] == {"1P": 508, "2P": 588}
    assert adoption["adoption_sides"] == ["1P"]


def test_make_reach_rate_is_reported_with_denominator(tmp_path: Path) -> None:
    """到達率は必ず母数と並べる。0.50%% という数字は母数なしでは意味がない。"""
    calls = R.load_make_calls(make_run_dir(tmp_path))
    assert calls["rows"] == 36300
    assert calls["make_invoked"] == 182
    assert calls["made"] == 0
    assert calls["sides"] == {"1P": 182}

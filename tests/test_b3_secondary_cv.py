"""副次CV実行器の合成データ検証。本番行は読まない。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import b3_secondary_cv as cv
from src.b3_train_config import PLAYER_CTRL_SALT
from src.b3_train_model import HgbConfig, feature_columns
from src.b3_train_synth import make_rows


def fixture_plan() -> dict:
    titles = {f"video_s{i:02d}": f"P vs 相手{i} 30先" if i < 3 else
              f"別{i} vs 他{i} 30先" for i in range(8)}
    groups = {v: v for v in titles}
    jobs = cv.build_jobs(titles, groups)
    return dict(sha256="合成計画", jobs=jobs, groups=groups, columns=["cells_color_p1", "cells_color_p2"],
                estimate=cv.estimate(jobs), aggregation="試合等重み")


def write_parts(out: Path, plan: dict, differences: tuple[float, ...] = (0.01, 0.03, 0.07),
                games: tuple[int, ...] = (1, 2, 7)) -> None:
    for job in plan["jobs"]:
        scores = {}
        for v in job["test"]:
            idx = int(v[-2:])
            scores[v] = dict(ll=0.5 + (differences[idx] if job["arm"] == "A" else 0), games=games[idx])
        cv.atomic_json(out / "parts" / f"{job['id']}.json",
                       dict(plan_sha256=plan["sha256"], job=job, scores=scores))


def test_splits_follow_addendum_and_reuse_a() -> None:
    plan = fixture_plan()
    jobs = plan["jobs"]
    assert len(jobs) == 4
    a = jobs[0]
    assert a["arm"] == "A" and len(a["test"]) == 3
    unrelated = a["train"]
    expected = sorted(unrelated, key=lambda v: hashlib.sha256(
        (PLAYER_CTRL_SALT + "p|" + v).encode()).hexdigest())[:2]
    for b in jobs[1:]:
        assert b["control_dropped"] == sorted(expected)
        assert set(b["train"]) == set(plan["groups"]) - {b["test"][0], *expected}
        assert len(a["train"]) == len(b["train"])
        assert len(set(a["test"]) & set(b["train"])) == 2


def test_order_and_normalization_are_deterministic() -> None:
    titles = {f"v{i}": f"「 Ｐ vs 相手{i} 」" if i < 3 else f"別{i} vs 他{i} 30先" for i in range(8)}
    groups = {v: v for v in titles}
    first = cv.build_jobs(titles, groups)
    assert first == cv.build_jobs(dict(reversed(list(titles.items()))), groups)
    assert {j["player"] for j in first} == {"p"}


def test_group_sibling_is_excluded_and_reported() -> None:
    titles = {f"v{i}": f"P vs 相手{i} 30先" if i < 3 else f"別{i} vs 他{i} 30先" for i in range(8)}
    groups = {v: v for v in titles}
    groups["v1"] = groups["v0"]
    jobs = cv.build_jobs(titles, groups)
    b = next(j for j in jobs if j["arm"] == "B" and j["test"] == ["v0"])
    assert "v1" not in b["train"] and b["group_extra"] == ["v1"]
    assert len(b["train"]) == len(jobs[0]["train"]) - 1


@pytest.mark.parametrize("titles", [{"v": "抽出不能"}, {"v": "P vs Q 30先"},
                                    {f"v{i}": "P vs Q 30先" for i in range(3)}])
def test_invalid_population_stops(titles: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        cv.build_jobs(titles, {v: v for v in titles})


def test_estimate_counts_saved_a_jobs() -> None:
    estimate = fixture_plan()["estimate"]
    assert estimate["naive_jobs"] == 6 and estimate["jobs"] == 4
    assert estimate["saved_jobs"] == 2
    assert estimate["hours"] == pytest.approx([170 / 3600, 1200 / 3600])


def test_aggregate_uses_match_weights(tmp_path: Path) -> None:
    plan = fixture_plan()
    write_parts(tmp_path, plan)
    result = cv.aggregate(plan, tmp_path)
    assert result["per_player"]["p"]["d"] == pytest.approx(0.056)
    assert result["d_mean"] == pytest.approx(0.056)
    assert result["per_player"]["p"]["test_games"] == 10
    assert result["mean_hold"] and result["players_over"] == []
    assert result["verdict"] == "HOLD"


@pytest.mark.parametrize("difference,games,hold", [
    (0.0, (1, 2, 7), False), (0.06, (1, 2, 17), True),
    (0.05, (1, 2, 17), True),
])
def test_receipt_hold_conditions(tmp_path: Path, difference: float, games: tuple[int, ...],
                                 hold: bool) -> None:
    plan = fixture_plan()
    write_parts(tmp_path, plan, (difference,) * 3, games)
    result = cv.aggregate(plan, tmp_path)
    assert result["hold"] is hold
    # 差分は実際の浮動小数点値で判定する。
    expected = ["p"] if result["per_player"]["p"]["d"] > 0.05 and sum(games) >= 20 else []
    assert result["players_over"] == expected


def test_missing_or_foreign_parts_never_produce_receipt(tmp_path: Path) -> None:
    plan = fixture_plan()
    write_parts(tmp_path, plan)
    part = tmp_path / "parts" / f"{plan['jobs'][-1]['id']}.json"
    part.unlink()
    with pytest.raises(FileNotFoundError):
        cv.aggregate(plan, tmp_path)
    assert not (tmp_path / "RECEIPT.json").exists()
    write_parts(tmp_path, plan)
    data = json.loads(part.read_text())
    data["plan_sha256"] = "他の計画"
    cv.atomic_json(part, data)
    with pytest.raises(ValueError, match="別の入力"):
        cv.aggregate(plan, tmp_path)


@pytest.mark.parametrize("change", ["games", "ll", "test"])
def test_corrupt_scores_are_rejected(tmp_path: Path, change: str) -> None:
    plan = fixture_plan()
    write_parts(tmp_path, plan)
    job = plan["jobs"][-1]
    path = tmp_path / "parts" / f"{job['id']}.json"
    data = json.loads(path.read_text())
    score = data["scores"][job["test"][0]]
    if change == "games":
        score["games"] += 1
    elif change == "ll":
        score["ll"] = float("nan")
    else:
        data["scores"] = {}
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        cv.aggregate(plan, tmp_path)


def test_real_synthetic_training_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    df, _ = make_rows(n_videos=8, games_per_video=2, rows_per_game=8)
    plan = fixture_plan()
    monkeypatch.setattr(cv, "CONFIG", HgbConfig("合成用", 3, 0.1, 3, 2))
    monkeypatch.setenv("B3_HGB_THREADS", "1")
    monkeypatch.setenv("B3_MONOTONIC", "0")
    for job in plan["jobs"]:
        result = cv.run_job(df, plan, job, tmp_path / "parts")
        assert result["monotonic_cst"]["static"] == {"cells_color_p1": 1, "cells_color_p2": -1}
        assert all(s["games"] == 2 for s in result["scores"].values())
    receipt = cv.aggregate(plan, tmp_path)
    assert receipt["per_player"]["p"]["test_games"] == 6

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("完了済みジョブで再学習した")

    monkeypatch.setattr(cv, "fit_two_state", forbidden)
    for job in reversed(plan["jobs"]):
        cv.run_job(df, plan, job, tmp_path / "parts")
    assert cv.aggregate(plan, tmp_path) == receipt


def test_fixed_configuration_and_fast_exclusion() -> None:
    assert cv.CONFIG.name == "ref" and cv.FINAL_SEED == 1003
    assert not set(feature_columns(cv.COLUMN_NAMES)) & set(cv.EXCLUDED_FAST_COLUMNS)


def test_worker_partition_and_plan_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from argparse import Namespace
    plan = fixture_plan()
    seen = []
    monkeypatch.setattr(cv, "load_training_rows", lambda *args: pd.DataFrame({"video_id": list(plan["groups"])}))
    monkeypatch.setattr(cv, "precheck_rows", lambda *args: {"violations_total": 0})
    monkeypatch.setattr(cv, "run_job", lambda df, p, j, parts: seen.append(j["id"]))
    args = Namespace(run=False, aggregate=False, out=tmp_path, job=None, worker_index=None)
    cv.execute(args, plan, plan["groups"])
    assert seen == []
    args.run = True
    for index in range(2):
        args.worker_index = index
        cv.execute(args, plan, plan["groups"])
    assert sorted(seen) == sorted(j["id"] for j in plan["jobs"])


def test_plan_fingerprint_changes_with_input(tmp_path: Path) -> None:
    targets = tmp_path / "targets.json"
    records = [dict(video_id=f"v{i}", youtube_id=f"yt{i}",
                    title=f"P vs 相手{i} 30先" if i < 3 else f"別{i} vs 他{i} 30先") for i in range(8)]
    cv.atomic_json(targets, records)
    paths = {r["video_id"]: tmp_path / f"{r['video_id']}.parquet" for r in records}
    for path in paths.values():
        path.write_bytes(b"synthetic")
    plan = cv.create_plan(paths, targets, 1)
    assert cv.create_plan(paths, targets, 1) == plan
    paths["v0"].write_bytes(b"changed")
    assert cv.create_plan(paths, targets, 1)["sha256"] != plan["sha256"]

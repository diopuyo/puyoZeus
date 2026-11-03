"""対戦者単位の副次CV（事前登録 追記A §2）。既定は計画のみ、学習は --run。

python -m scripts.b3_secondary_cv --rows-dir ROWS --out OUT
同じ引数に --run --worker-index 0（別プロセスは1）を追加して2並列。
単独ジョブは --run --job PLAN内のid、集計だけは --aggregate。
完了済み parts は再利用する。Aは選手ごと1回、Bは各動画ごと1回。
本文のgroup分離も維持するため、Bではvの同一groupも追加除外する。
追加除外とA/Bの学習動画数は受領票に明記する（c110/c112ではBが1本少ない）。
段1の登録集合のみを対象とし、段2・封印の行は読み込まない。
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import hashlib
import json
import math
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.b3_train_stage1 import REPOST_MERGES
from src.b3_column_spec import COLUMN_NAMES, EXCLUDED_FAST_COLUMNS
from src.b3_eval_gates import secondary_hold
from src.b3_eval_metrics import game_values, row_log_loss
from src.b3_eval_secondary import d_mean, eligible_players, leave_player_split, video_players
from src.b3_train_config import FINAL_SEED, PLAYER_CTRL_SALT
from src.b3_train_data import list_complete_parquets, load_group_map, load_training_rows, precheck_rows
from src.b3_train_folds import assert_no_group_leak, control_drop_order
from src.b3_train_model import (DEFAULT_GRID, feature_columns, fit_two_state, hgb_threads,
                                model_monotonic_constraints)

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "docs/b3_prereg_fill/training_manifest.json"
CONFIG = next(c for c in DEFAULT_GRID if c.name == "ref")
WORKERS = 2
JOB_SECONDS = (85, 600)
SECONDS_PER_HOUR = 3600
VERSION = 1


def digest(value: object) -> str:
    """JSONの内容を決定的な識別子にする。"""
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: object) -> None:
    """中断時に完成扱いの部分ファイルを残さない。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as stream:
        tmp = Path(stream.name)
        stream.write(text)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def make_job(player: str, arm: str, train: list[str], test: list[str],
             dropped: list[str], group_extra: list[str]) -> dict:
    key = digest([player, arm, test])
    return dict(id=f"{arm}_{key}", player=player, arm=arm, train=sorted(train),
                test=sorted(test), control_dropped=sorted(dropped), group_extra=sorted(group_extra))


def build_jobs(titles: dict[str, str], groups: dict[str, str]) -> list[dict]:
    """選手抽出・A分割を再利用し、追記AのBを作る。検証groupは学習から除く。"""
    players = video_players(titles)
    jobs = []
    for player, videos in eligible_players(players).items():
        split = leave_player_split(player, players, groups)
        extra = sorted(set(split["test"]) - set(videos))
        jobs.append(make_job(player, "A", split["train"], videos, [], extra))
        unrelated = [v for v in players if player not in players[v]]
        count = len(videos) - 1
        if len(unrelated) < count:
            raise ValueError(f"{player}: 対照の除外候補が不足")
        dropped = control_drop_order(player, unrelated)[:count]
        for video in videos:
            excluded = {video, *dropped}
            siblings = {v for v in players if groups[v] == groups[video]}
            extra = sorted(siblings - excluded)
            train = sorted(set(players) - excluded - siblings)
            assert_no_group_leak([groups[v] for v in train], [groups[video]])
            jobs.append(make_job(player, "B", train, [video], dropped, extra))
    if not jobs or any(not j["train"] for j in jobs):
        raise ValueError("対象選手または学習動画が無い")
    return jobs


def estimate(jobs: list[dict], workers: int = WORKERS) -> dict:
    """指定の1ジョブ85秒〜10分から、均等分配時の概算を返す。"""
    n_a = sum(j["arm"] == "A" for j in jobs)
    n_b = len(jobs) - n_a
    waves = math.ceil(len(jobs) / workers)
    return dict(players=n_a, pairs=n_b, naive_jobs=n_b * 2, jobs=len(jobs),
                a_jobs=n_a, b_jobs=n_b, saved_jobs=n_b - n_a, workers=workers,
                job_seconds=list(JOB_SECONDS),
                hours=[waves * sec / SECONDS_PER_HOUR for sec in JOB_SECONDS],
                note="読込・hash・採点・2並列による競合は別途。学習時間は推定で実測ではない")


def create_plan(paths: dict[str, Path], targets: Path, threads: int) -> dict:
    """段1の登録動画だけを計画し、入力・コード・設定をpartsと結び付ける。"""
    records = json.loads(targets.read_text(encoding="utf-8"))
    titles = {r["video_id"]: r["title"] for r in records if r["video_id"] in paths}
    if set(titles) != set(paths):
        raise ValueError("対象動画のタイトルが不足")
    group_all = load_group_map(targets, REPOST_MERGES)
    groups = {v: group_all[v] for v in paths}
    jobs = build_jobs(titles, groups)
    sources = [Path(__file__), ROOT / "scripts/b3_train_stage1.py", *sorted((ROOT / "src").glob("b3_*.py"))]
    plan = dict(version=VERSION, jobs=jobs, groups=groups, estimate=estimate(jobs),
                config=asdict(CONFIG), seed=FINAL_SEED, monotonic=True, threads=threads,
                columns=feature_columns(COLUMN_NAMES), excluded_fast=list(EXCLUDED_FAST_COLUMNS),
                salt=PLAYER_CTRL_SALT, targets_sha256=file_hash(targets),
                inputs={v: file_hash(p) for v, p in paths.items()},
                sources={str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                aggregation="動画内も選手内も試合等重み、選手間はtest試合数重み")
    return {**plan, "sha256": digest(plan)}


def read_part(path: Path, plan: dict, job: dict) -> dict:
    part = json.loads(path.read_text(encoding="utf-8"))
    if part.get("plan_sha256") != plan["sha256"] or part.get("job") != job:
        raise ValueError(f"別の入力・設定・ジョブのparts: {path}")
    scores = part["scores"]
    if set(scores) != set(job["test"]):
        raise ValueError(f"採点動画の欠落: {path}")
    for score in scores.values():
        if score["games"] < 1 or not math.isfinite(score["ll"]) or score["ll"] < 0:
            raise ValueError(f"不正な採点値: {path}")
    return part


def run_job(df: pd.DataFrame, plan: dict, job: dict, parts: Path) -> dict:
    """独立ジョブ。Aは全出演動画を1モデルで採点し、完成済みなら学習を省く。"""
    path = parts / f"{job['id']}.json"
    if path.exists():
        return read_part(path, plan, job)
    started = time.perf_counter()
    train = df[df["video_id"].isin(job["train"])]
    model = fit_two_state(train, plan["columns"], CONFIG, FINAL_SEED, monotonic=True)
    scores = {}
    for video in job["test"]:
        test = df[df["video_id"] == video]
        prediction = model.predict(test)
        if len(test) == 0 or not np.isfinite(prediction).all():
            raise ValueError(f"{video}: 空の検証行または非有限予測")
        values = game_values(test, row_log_loss(test["won_p1"].to_numpy(), prediction), plan["groups"])
        scores[video] = dict(ll=float(values["value"].mean()), games=len(values))
    result = dict(plan_sha256=plan["sha256"], job=job, scores=scores,
                  train_rows=len(train), seconds=time.perf_counter() - started,
                  monotonic_cst=model_monotonic_constraints(model))
    atomic_json(path, result)
    return result


def aggregate(plan: dict, out: Path) -> dict:
    """全partsを照合した場合だけ、登録済みの保留条件を判定する。"""
    parts = [read_part(out / "parts" / f"{j['id']}.json", plan, j) for j in plan["jobs"]]
    by_a = {p["job"]["player"]: p for p in parts if p["job"]["arm"] == "A"}
    per_player: dict[str, dict] = {}
    for part in parts:
        job = part["job"]
        if job["arm"] != "B":
            continue
        player, video = job["player"], job["test"][0]
        a, b = by_a[player]["scores"][video], part["scores"][video]
        if a["games"] != b["games"]:
            raise ValueError(f"{player}/{video}: AとBの試合数が不一致")
        result = per_player.setdefault(player, {"videos": {}})
        result["videos"][video] = dict(ll_a=a["ll"], ll_b=b["ll"], d=a["ll"] - b["ll"],
            test_games=a["games"], train_videos_a=len(by_a[player]["job"]["train"]),
            train_videos_b=len(job["train"]), group_extra=job["group_extra"])
    for result in per_player.values():
        values = list(result["videos"].values())
        result["test_games"] = sum(v["test_games"] for v in values)
        for key in ("d", "ll_a", "ll_b"):
            result[key] = sum(v[key] * v["test_games"] for v in values) / result["test_games"]
    mean = d_mean(per_player)
    hold = secondary_hold(mean, per_player)
    receipt = dict(plan_sha256=plan["sha256"], status="complete", per_player=per_player,
                   d_mean=mean, **hold, verdict="HOLD" if hold["hold"] else "NO_HOLD",
                   estimate=plan["estimate"], aggregation=plan["aggregation"],
                   note="副次CVの報告項目。全体の合格判定ではない。group追加除外による本数差は動画別に記録")
    atomic_json(out / "RECEIPT.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-dir", type=Path, required=True)
    parser.add_argument("--targets", type=Path, default=TARGETS)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--aggregate", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--job", help="PLAN.jsonのジョブidを1つ指定")
    selection.add_argument("--worker-index", type=int, choices=range(WORKERS))
    return parser.parse_args()


def execute(a: argparse.Namespace, plan: dict, paths: dict[str, Path]) -> None:
    if a.aggregate:
        aggregate(plan, a.out)
        return
    if not a.run:
        return
    selected = [j for i, j in enumerate(plan["jobs"]) if
                (a.job is None or j["id"] == a.job) and
                (a.worker_index is None or i % WORKERS == a.worker_index)]
    if not selected:
        raise ValueError("指定ジョブが存在しない")
    pending = []
    for job in selected:
        path = a.out / "parts" / f"{job['id']}.json"
        if path.exists():
            read_part(path, plan, job)
        else:
            pending.append(job)
    if pending:
        df = load_training_rows(paths, plan["columns"])
        pre = precheck_rows(df, plan["groups"])
        if set(df["video_id"]) != set(paths) or pre["violations_total"]:
            raise ValueError(f"段1行の前提検査に違反: {pre}")
        for job in pending:
            run_job(df, plan, job, a.out / "parts")
            print(f"完了: {job['player']} {job['arm']} {job['test']}", flush=True)
    if all((a.out / "parts" / f"{j['id']}.json").exists() for j in plan["jobs"]):
        aggregate(plan, a.out)


def main() -> None:
    a = parse_args()
    if (a.job is not None or a.worker_index is not None) and not a.run:
        raise ValueError("ジョブ指定には --run が必要")
    if os.environ.get("B3_MONOTONIC", "1") != "1":
        raise ValueError("副次CVは B3_MONOTONIC=1 に固定")
    # 段1の登録集合に限定し、別集合のdoneが同居しても読まない。
    registered = {r["video_id"] for r in json.loads(TARGETS.read_text(encoding="utf-8"))}
    supplied = {r["video_id"] for r in json.loads(a.targets.read_text(encoding="utf-8"))}
    if supplied != registered:
        raise ValueError("targetsは段1の登録動画集合と一致させること")
    paths = list_complete_parquets(a.rows_dir, sorted(registered))
    plan = create_plan(paths, a.targets, hgb_threads())
    plan_path = a.out / "PLAN.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("既存計画と入力・コード・設定が不一致。別の出力先を指定")
    atomic_json(plan_path, plan)
    print(json.dumps(plan["estimate"], ensure_ascii=False), flush=True)
    execute(a, plan, paths)


if __name__ == "__main__":
    main()

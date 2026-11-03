"""F1: 非飽和火力と打ち返し余地を固定15foldで比較する。"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from src.board import Board, BOARD_COLS
from src.chain import ChainSimulator
from src import indicators_v2 as iv
from src import puyo_core_bridge as native
from src.exchange_event_features import S1_COLUMNS, S3_COLUMNS
from src.exchange_event_landing import remaining_hands
from src.production_config import GHOST_CHAIN_RULE_ENABLED
from src.scoring import calculate_chain_score, compute_effective_rate, score_to_ojama
from scripts import train_exchange_event_models_20260926 as v1

ROOT, REFERENCE, SHARED = v1.ROOT, v1.REFERENCE, v1.SHARED
OUT = ROOT / "logs/f1b_counter_features"
RAW = SHARED / "data/indicators_v2/boards_lean_phase_l_2026-08-11"
QUEUE = ("next1_a", "next1_b", "dnext_a", "dnext_b")
WORKERS, NICE, BOOTSTRAPS, SEED, CHECKPOINT = 6, 19, 2000, 0, 50
LL_LIMIT, AUC_LIMIT, OVERALL_LIMIT = -.002, .01, .001
QUANTILES = (.025, .975)
RESPONSE_WIDTH = 2 * BOARD_COLS + 2 * (BOARD_COLS - 1)
SIDES = ("1P", "2P")
NEW_COLUMNS = tuple(f"NF_ojama_k{k}_{s}" for s in ("self", "opp", "diff")
                    for k in iv.NEAR_FUTURE_K_LEVELS) + ("counter_margin_self", "counter_margin_opp")
NAMES = ("S1", "S1_prime", "S3", "S3_prime", "S1_light", "S1_prime_light",
         "S3_light", "S3_prime_light")
DEFINITIONS = {
    "NF": "NF_ojama_k1..5=log1p(near_future_fire_power.values[k].raw)、rawはscoringのマージン換算済み個数。差列は変換後self−opp。",
    "counter": "counter_margin_side=sign(x)log1p(abs(x)); x=n手以内の送り個数−max(相手送量−自分送量,0)。",
    "hands": "n=remaining_hands（残演出/SEC_PER_HANDのfloor+1）、探索K=n−既知NEXT2手、応手幅22。",
    "S1": "発火前確定盤面でNF・応手を計算し、発火側だけChainSimulator完走予測を相殺に使用。",
    "S3": "F1b: 発火前確定盤面・NEXT・発火時換算率を凍結。NFは凍結盤面、応手は既発火群の予測完走盤面でn=1。得点差で相殺。post_idsの盤面・NEXT・時刻は使用禁止。",
    "overall": "全位相のE1 G_fe OOFを共通土台とし、T11採用中盤行だけ各SモデルのOOFに置換。",
    "detail": "得点確定・着地時刻が未収録のためS3時点の最新盤面は復元不能。発火前の最終確定盤面を保守的に凍結し、得点差はT11/E1既存列を維持。得点の着地前利用可能性の実測証明は含まない。",
}


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    """半端な途中ファイルを読まないよう原子的に保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def read_npz(path: Path) -> dict[str, np.ndarray]:
    """入力ファイルを閉じてから計算する。"""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


class NativeSimulator:
    """T11と同じnative連鎖計算を探索に渡す。"""
    def simulate(self, board: Board) -> Any:
        """幽霊連鎖の採用設定に従う。"""
        return native.simulate_chain(board, exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)


def native_score(result: Any) -> Any:
    """nativeが計算済みの厳密得点を探索へ返す。"""
    if isinstance(result, native.ChainSimResult):
        return SimpleNamespace(total_score=result.exact_score)
    return calculate_chain_score(result)


def init_worker() -> None:
    """CPUのみ、低優先度、探索内部だけnative計算にする。"""
    os.nice(max(0, NICE - os.nice(0)))
    if not native.NATIVE_AVAILABLE:
        raise RuntimeError("T11と同じnative計算環境が必要")


@lru_cache(maxsize=32768)
def fire(raw: bytes, queue: tuple[int, ...], elapsed: float,
         levels: tuple[int, ...], response: bool = False) -> np.ndarray:
    """正規化前の個数を返し、飽和したscoreから逆算しない。"""
    grid = np.frombuffer(raw, dtype=np.int8).reshape(iv.BOARD_ROWS, BOARD_COLS)
    board = Board.from_list(grid.tolist())
    width = RESPONSE_WIDTH if response else iv.NEAR_FUTURE_BEAM_WIDTH
    result = iv.near_future_fire_power(board, queue[:2], queue[2:], elapsed,
        simulator=NativeSimulator(), k_levels=levels, beam_width=width,
        active_colors=iv._near_future_active_colors(board), resolve_before_death=response,
        use_exact_score=True)
    return np.asarray([result.values[k].raw for k in levels])


def queue_at(raw: dict, idx: int) -> tuple[int, ...]:
    """未知色は既存T11同様に0へ変換する。"""
    return tuple(int(raw[key][idx]) if 1 <= raw[key][idx] <= 5 else 0 for key in QUEUE)


def completion(raw: dict, ids: np.ndarray, firing: np.ndarray, elapsed: float) -> tuple:
    """E12と同じ確定盤面の完走計算で既発火分を一度だけ相殺する。"""
    sends, counts, boards = np.zeros(2), np.zeros(2, dtype=int), []
    simulator = ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)
    for side, idx in enumerate(ids):
        if idx < 0:
            boards.append(None)
            sends[side] = np.nan
            continue
        board = Board.from_list(raw["grids"][idx].tolist())
        if firing[side]:
            result = simulator.simulate(board)
            sends[side] = score_to_ojama(calculate_chain_score(result).total_score,
                                       elapsed_sec=elapsed).ojama_count
            counts[side], board = result.chain_count, result.final_board
        boards.append(board)
    return sends, counts, boards


def stage_features(raw: dict, ids: np.ndarray, sends: np.ndarray, counts: np.ndarray,
                   elapsed: float, boards: list | None = None) -> np.ndarray:
    """側別の非飽和火力と、二重相殺しない打ち返し余地を算出する。"""
    values = np.full((2, len(iv.NEAR_FUTURE_K_LEVELS) + 1), np.nan)
    for side, idx in enumerate(ids):
        if idx < 0:
            continue
        grid, queue = raw["grids"][idx].astype(np.int8), queue_at(raw, int(idx))
        values[side, :-1] = np.log1p(fire(grid.tobytes(), queue, elapsed, iv.NEAR_FUTURE_K_LEVELS))
        response_board = boards[side]._grid.astype(np.int8) if boards is not None else grid
        busy = iv.estimate_chain_anim_duration_sec(int(counts[side]), "empirical_table_2026_08_14")
        hands = remaining_hands(int(counts[1-side]), 0., 0., busy_sec=busy)
        level = hands - iv.NEAR_FUTURE_KNOWN_HAND_SLOTS
        available = fire(response_board.tobytes(), queue, elapsed, (level,), True)[0]
        margin = available - np.maximum(sends[1-side] - sends[side], 0)
        values[side, -1] = np.sign(margin) * np.log1p(abs(margin))
    return values


def orient(values: np.ndarray, sign: float) -> np.ndarray:
    """絶対値は交換のみ、差分だけ反転する。"""
    own, opp = values if sign > 0 else values[::-1]
    return np.r_[own[:-1], opp[:-1], own[:-1]-opp[:-1], own[-1], opp[-1]]


def event_features(raw: dict, pre: np.ndarray, post: np.ndarray, trigger: float,
                   starts: dict, firing: np.ndarray) -> tuple:
    """同じ区間に属する学習行は一回だけ物理特徴を計算する。"""
    elapsed = float(trigger - starts[raw["game_idx"][post[0]]])
    sent, counts, boards = completion(raw, pre, firing, elapsed)
    s1 = stage_features(raw, pre, sent, counts, elapsed, boards)
    before = np.asarray([raw["score"][i] if i >= 0 else np.nan for i in pre])
    after = raw["score"][post].astype(float)
    missing = ~np.isfinite(before) | (before < 0) | (after < 0)
    confirmed = np.where(missing, np.nan, np.maximum(after-before, 0)/compute_effective_rate(elapsed))
    # 終点の盤面・NEXT・経過時刻を特徴へ入れない。既発火火力は応手へ再加算しない。
    s3 = stage_features(raw, pre, confirmed, np.zeros(2, dtype=int), elapsed, boards)
    return s1, s3, counts


def video_job(job: tuple[str, str]) -> str:
    """動画内50区間ごとに保存し、正常なNaNも再計算しない。"""
    video, output = job
    path = Path(output) / "features" / (video + ".npz")
    pre = read_npz(v1.run_path(REFERENCE, "s1") / "features" / (video + ".npz"))
    post = read_npz(v1.run_path(REFERENCE, "s3") / "features" / (video + ".npz"))
    np.testing.assert_array_equal(pre["row_id"], post["row_id"])
    np.testing.assert_array_equal(pre["pre_ids"], post["pre_ids"])
    size = len(pre["row_id"])
    data = read_npz(path) if path.exists() else dict(row_id=pre["row_id"],
        s1=np.full((size, len(NEW_COLUMNS)), np.nan, np.float32),
        s3=np.full((size, len(NEW_COLUMNS)), np.nan, np.float32),
        done=np.zeros(size, bool), predicted_counts=np.zeros((size, 2), np.int16))
    if data["done"].all():
        return video
    raw = read_npz(RAW / (video.removeprefix("video_") + ".npz"))
    rows = pd.read_csv(v1.run_path(REFERENCE, "s1") / "rows.csv", usecols=["row_id", "sign"])
    signs = rows.set_index("row_id").loc[pre["row_id"], "sign"].to_numpy()
    starts = {g: raw["t_sec"][raw["game_idx"] == g].min() for g in np.unique(raw["game_idx"])}
    cache = {}
    for number, i in enumerate(np.flatnonzero(~data["done"])):
        key = (*pre["pre_ids"][i], *post["post_ids"][i], float(pre["trigger"][i]))
        if key not in cache:
            firing = pre["values"][i, -2:]
            if signs[i] < 0:
                firing = firing[::-1]
            cache[key] = event_features(raw, pre["pre_ids"][i], post["post_ids"][i],
                                       float(pre["trigger"][i]), starts, firing)
        left, right, counts = cache[key]
        data["s1"][i], data["s3"][i] = orient(left, signs[i]), orient(right, signs[i])
        data["predicted_counts"][i], data["done"][i] = counts, True
        if (number + 1) % CHECKPOINT == 0:
            save_npz(path, **data)
    save_npz(path, **data)
    print(f"特徴完了 {video} {size}行 {len(cache)}区間", flush=True)
    return video


def prepare(output: Path) -> pd.DataFrame:
    """T11と同じ行順で既存列を保持して追加列だけ後置する。"""
    rows = pd.read_csv(v1.run_path(REFERENCE, "s1") / "rows.csv")
    videos = sorted(rows.video_id.unique())
    with ProcessPoolExecutor(WORKERS, initializer=init_worker) as pool:
        list(pool.map(video_job, [(v, str(output)) for v in videos]))
    chunks = [read_npz(output / "features" / (v + ".npz")) for v in videos]
    ids = np.concatenate([c["row_id"] for c in chunks])
    order = np.argsort(ids)
    np.testing.assert_array_equal(ids[order], rows.row_id)
    d = np.load(v1.run_path(REFERENCE, "base") / "design.npy", mmap_mode="r")[rows.row_id]
    s1 = np.column_stack((d, np.load(v1.run_path(REFERENCE, "s1") / "s1_features.npy")))
    s3 = np.column_stack((s1, np.load(ROOT / "logs/exchange_event_v1/score_features_margin.npy")))
    for name, original, key in (("S1", s1, "s1"), ("S3", s3, "s3")):
        new = np.concatenate([c[key] for c in chunks])[order]
        np.save(output / (name + ".npy"), original)
        np.save(output / (name + "_prime.npy"), np.column_stack((original, new)))
    rows.to_csv(output / "rows.csv", index=False)
    v1.save_json(output / "FEATURE_COUNTS.json", dict(rows=len(rows), videos=len(videos),
        positive_completion_rows=int(sum((c["predicted_counts"] > 0).any(axis=1).sum() for c in chunks))))
    return rows


def fold_job(job: tuple[str, int, int]) -> str:
    """固定動画holdoutで旧新・標準軽量を同条件に再学習する。"""
    output_text, seed, fold = job
    output = Path(output_text)
    path = output / f"seed_{seed}/fold_{fold}/predictions.csv"
    if path.exists():
        return str(path)
    rows = pd.read_csv(output / "rows.csv")
    ref = pd.read_csv(v1.run_path(REFERENCE, "s1") / f"seed_{seed}/fold_{fold}/predictions.csv")
    test = rows.video_id.isin(ref.video_id.unique()).to_numpy()
    assert not set(rows.loc[test, "video_id"]) & set(rows.loc[~test, "video_id"])
    sign, label = rows.sign.to_numpy(), rows.label.to_numpy()
    target = np.where(sign > 0, label, 1-label)
    result = rows.loc[test, ["row_id", "video_id", "label", "phase"]].copy()
    for name in NAMES:
        x = np.load(output / (name.removesuffix("_light") + ".npy"), mmap_mode="r")
        params = v1.LIGHT_PARAMS if name.endswith("_light") else {}
        model = HistGradientBoostingClassifier(random_state=SEED, **params).fit(x[~test], target[~test])
        p = model.predict_proba(x[test])[:, 1]
        result[name] = np.where(sign[test] > 0, p, 1-p)
    result["seed"], result["fold"] = seed, fold
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path.with_suffix(".tmp"), index=False)
    path.with_suffix(".tmp").replace(path)
    print(f"CV完了 seed={seed} fold={fold}", flush=True)
    return str(path)


def prediction_frames(output: Path) -> None:
    """全位相では非対象行の共通G_feを保持する。"""
    middle, overall = [], []
    for seed in v1.SEEDS:
        for fold in v1.FOLDS:
            part = pd.read_csv(output / f"seed_{seed}/fold_{fold}/predictions.csv")
            middle.append(part)
            frame = pd.read_csv(ROOT / f"logs/exchange_event_v1/seed_{seed}/fold_{fold}/g.csv")
            frame = frame.set_index("row_id")
            assert set(part.row_id).issubset(frame.index)
            np.testing.assert_array_equal(frame.loc[part.row_id, "label"], part.label)
            for name in NAMES:
                frame[name] = frame.G_fe
                frame.loc[part.row_id, name] = part[name].to_numpy()
            frame["seed"], frame["fold"] = seed, fold
            overall.append(frame.reset_index().drop(columns="G_fe"))
    for scope, parts in (("middle", middle), ("overall", overall)):
        frame = pd.concat(parts, ignore_index=True)
        assert not frame.duplicated(["seed", "row_id"]).any()
        frame.to_pickle(output / (scope + ".pkl"))


def auc_pair_matrix(y: np.ndarray, p: np.ndarray, groups: np.ndarray, count: int) -> np.ndarray:
    """動画対ごとの勝ち順位数を作り、2000回の対応AUCを厳密に高速化する。"""
    order = np.argsort(p, kind="stable")
    y, p, groups = y[order], p[order], groups[order]
    starts = np.r_[0, np.flatnonzero(np.diff(p)) + 1]
    lengths = np.diff(np.r_[starts, len(p)])
    matrix = np.empty((count, count))
    for video in range(count):
        negatives = np.add.reduceat(((groups == video) & (y == 0)).astype(float), starts)
        ranks = np.cumsum(negatives) - negatives / 2
        weights = y * np.repeat(ranks, lengths)
        matrix[:, video] = np.bincount(groups, weights=weights, minlength=count)
    return matrix


def bootstrap_job(job: tuple[str, str, str]) -> str:
    """動画を復元抽出し、同一抽出列で全モデルを比較する。"""
    output_text, scope, name = job
    output = Path(output_text)
    path = output / f"bootstrap_{scope}_{name}.npz"
    if path.exists():
        return str(path)
    frame = pd.read_pickle(output / (scope + ".pkl"))
    y, p = frame.label.to_numpy(), frame[name].to_numpy()
    videos = np.unique(frame.video_id.to_numpy(dtype=str))
    groups = pd.Categorical(frame.video_id, categories=videos).codes.astype(np.int64)
    count = len(videos)
    eps = np.finfo(float).eps
    clipped = np.clip(p, eps, 1-eps)
    loss = -(y*np.log(clipped)+(1-y)*np.log1p(-clipped))
    sums = np.bincount(groups, weights=loss, minlength=count)
    sizes = np.bincount(groups, minlength=count)
    positive = np.bincount(groups, weights=y, minlength=count)
    negative = sizes-positive
    pairs = auc_pair_matrix(y, p, groups, count)
    rng = np.random.default_rng(SEED)
    draws = np.stack([np.bincount(rng.integers(0, count, count), minlength=count) for _ in range(BOOTSTRAPS)])
    ll = (draws @ sums) / (draws @ sizes)
    auc = np.einsum("bi,ij,bj->b", draws, pairs, draws) / ((draws @ positive)*(draws @ negative))
    point = np.array([log_loss(y, p), roc_auc_score(y, p)])
    np.testing.assert_allclose(pairs.sum() / (positive.sum()*negative.sum()), point[1], atol=1e-12)
    save_npz(path, values=np.column_stack((ll, auc)), point=point,
             rows=np.array(len(frame)), videos=videos)
    print(f"bootstrap完了 {scope} {name}", flush=True)
    return str(path)


def decisions(report: dict) -> dict:
    """事前登録の不等号をそのまま適用する。"""
    result = {}
    for name, middle in report["differences"]["middle"].items():
        ll, auc = middle["log_loss"], middle["auc"]
        improvement = ((ll["delta"] <= LL_LIMIT and ll["ci95"][1] < 0)
                       or (auc["delta"] >= AUC_LIMIT and auc["ci95"][0] > 0))
        upper = report["differences"]["overall"][name]["log_loss"]["ci95"][1]
        result[name] = dict(adopt=bool(improvement and upper <= OVERALL_LIMIT),
                            middle_improvement=bool(improvement), overall_ll_upper=upper)
    return result


def collect_report(output: Path) -> dict:
    """中盤と全体について点推定・差の95%CIを揃える。"""
    report: dict[str, Any] = dict(metrics={}, differences={}, definitions=DEFINITIONS)
    for scope in ("middle", "overall"):
        saved = {name: read_npz(output / f"bootstrap_{scope}_{name}.npz") for name in NAMES}
        report["metrics"][scope] = {name: dict(zip(("log_loss", "auc"), data["point"].tolist()))
                                    for name, data in saved.items()}
        report["differences"][scope] = {}
        for stage in ("S1", "S3"):
            for suffix in ("", "_light"):
                old, new = stage+suffix, stage+"_prime"+suffix
                diff = saved[new]["values"]-saved[old]["values"]
                point = saved[new]["point"]-saved[old]["point"]
                report["differences"][scope][new] = {metric: dict(delta=float(point[i]),
                    ci95=np.quantile(diff[:, i], QUANTILES).tolist())
                    for i, metric in enumerate(("log_loss", "auc"))}
        report[scope+"_predictions"] = int(saved["S1"]["rows"])
    report["decision"] = decisions(report)
    return report


def final_fit(output: Path, report: dict) -> dict:
    """基準を満たしたモデルだけv2専用ディレクトリへ保存する。"""
    accepted = [name for name, value in report["decision"].items() if value["adopt"]]
    if not accepted:
        return {}
    directory = ROOT / "models/exchange_event_v2"
    directory.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(output / "rows.csv")
    target = np.where(rows.sign > 0, rows.label, 1-rows.label)
    manifest = {}
    for name in accepted:
        x = np.load(output / (name.removesuffix("_light") + ".npy"), mmap_mode="r")
        params = v1.LIGHT_PARAMS if name.endswith("_light") else {}
        model = HistGradientBoostingClassifier(random_state=SEED, **params).fit(x, target)
        path = directory / (name + ".joblib")
        joblib.dump(model, path)
        columns = (S1_COLUMNS if name.startswith("S1") else S3_COLUMNS) + NEW_COLUMNS
        manifest[name] = dict(file=path.name, columns=columns, sha256=v1.file_sha256(path),
                              parameters=model.get_params(), decision=report["decision"][name])
    v1.save_json(directory / "manifest.json", dict(models=manifest, definitions=DEFINITIONS,
        production_enabled=False, protocol_sha256=v1.file_sha256(output / "PROTOCOL.json")))
    return manifest


def summary(output: Path, report: dict) -> None:
    """要求された数値と物理量の定義を短い日本語で残す。"""
    lines = ["# F1 非飽和火力・打ち返し余地", "", *DEFINITIONS.values(), ""]
    for scope in ("middle", "overall"):
        lines.append(f"{scope}: {report[scope+'_predictions']} OOF予測（3seed分）")
        for stage in ("S1", "S3"):
            for suffix in ("", "_light"):
                old, new = stage+suffix, stage+"_prime"+suffix
                a, b = (report["metrics"][scope][n] for n in (old, new))
                diff = report["differences"][scope][new]
                lines.append(f"- {old}→{new}: AUC {a['auc']:.8f}→{b['auc']:.8f}; "
                    f"LL {a['log_loss']:.8f}→{b['log_loss']:.8f}; 差CI {json.dumps(diff, ensure_ascii=False)}")
    lines += ["判定: "+json.dumps(report["decision"], ensure_ascii=False),
              f"所要秒: {report['elapsed_seconds']:.1f}", "15fold、動画bootstrap 2000回 seed0、CPU6並列 nice19。"]
    (output / "SUMMARY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def protocol(output: Path) -> None:
    """結果を見る前に列・CV・比較・判定・入力ハッシュを固定する。"""
    paths = [v1.run_path(REFERENCE, "base") / "rows.csv",
             v1.run_path(REFERENCE, "s1") / "rows.csv",
             ROOT / "logs/exchange_event_v1/score_features_margin.npy"]
    paths += sorted(v1.run_path(REFERENCE, "s1").glob("seed_*/fold_*/predictions.csv"))
    value = dict(definitions=DEFINITIONS, columns=NEW_COLUMNS, workers=WORKERS, nice=NICE,
        gbm=HistGradientBoostingClassifier(random_state=SEED).get_params(), light=v1.LIGHT_PARAMS,
        seeds=list(v1.SEEDS), folds=list(v1.FOLDS), bootstrap=dict(unit="video", n=BOOTSTRAPS, seed=SEED),
        adoption=dict(ll=LL_LIMIT, auc=AUC_LIMIT, overall_ll_ci_upper=OVERALL_LIMIT),
        input_sha256={str(p): v1.file_sha256(p) for p in paths},
        s3_baseline="E1 S3 margin版（既存v1）。T11固定70版との混同を避ける。")
    path = output / "PROTOCOL.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(value)):
        raise ValueError("再開先の事前登録と入力が変化した")
    v1.save_json(path, value)


def execute(output: Path) -> None:
    """特徴生成・固定CV・対応CI・採用モデル保存まで完了する。"""
    started_path = output / "STARTED.json"
    if not started_path.exists():
        v1.save_json(started_path, dict(time=time.time()))
    started = json.loads(started_path.read_text())["time"]
    protocol(output)
    v1.validate_rows(REFERENCE, SHARED / "data/video_tier_index_2026-08-07.tsv")
    v1.save_json(output / "STATUS.json", dict(stage="FEATURES", pid=os.getpid()))
    prepare(output)
    v1.save_json(output / "STATUS.json", dict(stage="CV", pid=os.getpid()))
    jobs = [(str(output), s, f) for s in v1.SEEDS for f in v1.FOLDS]
    with ProcessPoolExecutor(WORKERS, initializer=init_worker) as pool:
        list(pool.map(fold_job, jobs))
    prediction_frames(output)
    v1.save_json(output / "STATUS.json", dict(stage="BOOTSTRAP", pid=os.getpid()))
    jobs = [(str(output), scope, name) for scope in ("middle", "overall") for name in NAMES]
    with ProcessPoolExecutor(WORKERS, initializer=init_worker) as pool:
        list(pool.map(bootstrap_job, jobs))
    report = collect_report(output)
    report["models"] = final_fit(output, report)
    report.update(status="COMPLETE", elapsed_seconds=time.time()-started)
    v1.save_json(output / "METRICS.json", report)
    summary(output, report)
    v1.save_json(output / "STATUS.json", dict(stage="COMPLETE", pid=os.getpid()))
    (output / "ERROR.json").unlink(missing_ok=True)


def main() -> None:
    """旧エントリポイントからもF1bの是正・監査付きパイプラインを使う。"""
    from scripts.audit_retrain_f1b_20260927 import main as causal_main

    causal_main()


if __name__ == "__main__":
    main()

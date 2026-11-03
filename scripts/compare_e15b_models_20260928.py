"""E15の行・分割・重みを固定した、開始盤面と未来得点の補足比較。"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

from scripts import train_exchange_event_models_v3_20260928 as v3
from scripts.e15_training_rows_20260928 import Exchange, samples
from src.exchange_event_count_features import CountObservation, orient, side_features
from src.exchange_event_features import score_features

v2, np, pd, v1 = v3.v2, v3.np, v3.pd, v3.v1
OUT = v2.ROOT / "logs/e15b"
VARIANTS = ("frozen_observed", "frozen_endpoint")
SCORE_START, COUNT_START = len(v2.S1_COLUMNS), len(v2.S3_COLUMNS)
EXPECTED_ROWS = 84445


def replacement(exchange: Exchange, raw: dict, start: float, count: int, source: int) -> np.ndarray:
    """盤面・NEXT・換算率を初回発火前に凍結し、指定個数の発火得点を加える。"""
    elapsed = max(0., exchange.trigger-start)
    observation = CountObservation(raw["grids"][exchange.pre],
        np.array([v2.queue_at(raw, i) for i in exchange.pre]), elapsed)
    firing = tuple(any(c.side == i and c.trigger == exchange.trigger for c in exchange.chains)
                   for i in range(2))
    totals = np.array([sum(c.score for c in exchange.chains[:count] if c.side == i) for i in range(2)])
    counts = orient(side_features(observation, firing, np.zeros(2), totals), source)
    return np.r_[score_features(np.zeros(2), totals, elapsed, source), counts]


def video_job(video: str) -> str:
    """採否を再計算せずE15保存行へ結合し、動画単位で再開可能にする。"""
    path = OUT / "features" / (video + ".npz")
    if path.exists():
        return video
    rows = pd.read_csv(v3.OUT / "rows.csv", float_precision="round_trip")
    rows = rows[rows.video_id == video]
    raw = v2.read_npz(v2.RAW / (video.removeprefix("video_") + ".npz"))
    matrices = {name: [] for name in VARIANTS}
    row_ids = []
    future_rows = 0
    for game, part in rows.groupby("game", sort=False):
        ids = np.flatnonzero(raw["game_idx"] == game)
        ids = ids[np.argsort(raw["t_sec"][ids], kind="stable")]
        start = float(raw["t_sec"][ids].min())
        # 全更新を読み終えた参照には、同じ撃ち合いの後続発火だけが加わっている。
        states = {(e.number, t): e for e, t, _, _ in samples(raw, ids)}
        for row in part.itertuples():
            exchange = states[(row.exchange, row.t_sec)]
            assert exchange.trigger == row.trigger_sec and (exchange.pre >= 0).all()
            assert (raw["t_sec"][exchange.pre] < exchange.trigger).all()
            observed = [c for c in exchange.chains if c.observed <= row.t_sec]
            assert len(observed) == row.fired_chains
            future_rows += int(len(exchange.chains) > row.fired_chains)
            for name, count in zip(VARIANTS, (row.fired_chains, len(exchange.chains))):
                matrices[name].append(replacement(exchange, raw, start, count, int(row.sign < 0)))
            row_ids.append(row.row_id)
    v2.save_npz(path, row_id=np.array(row_ids), future_rows=np.array(future_rows),
                **{name: np.asarray(values, np.float32) for name, values in matrices.items()})
    print(f"凍結特徴完了 {video} {len(row_ids)}行", flush=True)
    return video


def prepare() -> None:
    """同じ行番号へ差分列だけを差し替え、観測得点列の一致も検査する。"""
    rows = pd.read_csv(v3.OUT / "rows.csv")
    assert len(rows) == EXPECTED_ROWS and np.array_equal(rows.row_id, np.arange(EXPECTED_ROWS))
    with ProcessPoolExecutor(v2.WORKERS, initializer=v2.init_worker) as pool:
        list(pool.map(video_job, rows.video_id.unique()))
    original = np.load(v3.OUT / "S3_prime.npy")
    matrices = {name: original.copy() for name in VARIANTS}
    seen = np.zeros(len(rows), int)
    future_rows = 0
    for path in sorted((OUT / "features").glob("*.npz")):
        data = v2.read_npz(path)
        ids = data["row_id"]
        seen[ids] += 1
        future_rows += int(data["future_rows"])
        for name in VARIANTS:
            matrices[name][ids, SCORE_START:] = data[name]
    assert (seen == 1).all()
    np.testing.assert_array_equal(matrices[VARIANTS[0]][:, SCORE_START:COUNT_START],
                                  original[:, SCORE_START:COUNT_START])
    for name, matrix in matrices.items():
        dest = OUT / name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(v3.OUT / "rows.csv", dest / "rows.csv")
        np.testing.assert_array_equal(matrix[:, :SCORE_START], original[:, :SCORE_START])
        np.save(dest / "S3_prime.npy", matrix)
    v1.save_json(OUT / "COHORT.json", dict(rows=len(rows), videos=rows.video_id.nunique(),
        exchanges=len(rows.groupby(["video_id", "game", "exchange"])), future_score_rows=future_rows,
        rows_sha256=v1.file_sha256(v3.OUT / "rows.csv"), unchanged="D・arrival・行・重み・観測得点"))


def measurements(frame: pd.DataFrame, name: str) -> dict:
    """従来の行平均と撃ち合い重み付きの両方を報告する。"""
    return dict(auc=float(v2.roc_auc_score(frame.label, frame[name])),
        log_loss=float(v2.log_loss(frame.label, frame[name])),
        weighted_auc=float(v2.roc_auc_score(frame.label, frame[name], sample_weight=frame.weight)),
        weighted_log_loss=float(v2.log_loss(frame.label, frame[name], sample_weight=frame.weight)))


def report() -> None:
    """同一OOF行の比較だけを追記し、E15の事前登録判定は変更しない。"""
    results, reference = {}, None
    for name, root in (("live_observed", v3.OUT), *((n, OUT / n) for n in VARIANTS)):
        frame = pd.concat([pd.read_csv(p) for p in sorted(root.glob("seed_*/fold_*/predictions.csv"))])
        frame = frame.sort_values(["seed", "row_id"]).reset_index(drop=True)
        identity = frame[["row_id", "video_id", "seed", "fold", "weight", "label", "sign"]]
        if reference is None:
            reference = identity
        pd.testing.assert_frame_equal(identity, reference)
        assert len(frame) == EXPECTED_ROWS * len(v1.SEEDS)
        results[name] = {model: measurements(frame, model) for model in v3.NAMES}
    cohort = json.loads((OUT / "COHORT.json").read_text())
    v1.save_json(OUT / "METRICS.json", dict(cohort=cohort, folds=len(v1.SEEDS)*len(v1.FOLDS),
        predictions=len(reference), metrics=results, E15_verdict="FAIL維持・本比較は補足",
        endpoint_warning="未来の後続発火を含む参考条件。学習・推論への採用不可"))


def main() -> None:
    """10分超の処理を特徴・fold単位のログとチェックポイントで実行する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    v1.save_json(OUT / "STATUS.json", dict(stage="FEATURES"))
    prepare()
    v1.save_json(OUT / "STATUS.json", dict(stage="CV"))
    jobs = [(str(OUT / n), s, f) for n in VARIANTS for s in v1.SEEDS for f in v1.FOLDS]
    with ProcessPoolExecutor(v2.WORKERS, initializer=v2.init_worker) as pool:
        list(pool.map(v3.fold_job, jobs))
    report()
    v1.save_json(OUT / "STATUS.json", dict(stage="COMPLETE"))


if __name__ == "__main__":
    main()

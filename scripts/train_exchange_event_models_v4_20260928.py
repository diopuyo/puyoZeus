"""E16: 同じE15行・重み・15foldで、手番同期countだけを差し替える。"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil

from scripts import train_exchange_event_models_v3_20260928 as v3
from scripts.e15_training_rows_20260928 import samples, Exchange, Chain
from src.exchange_event_sync import CountTurnSynchronizer, SyncedCountSide
from src.exchange_event_count_features import CountObservation, side_features, orient, completion

v2, v1, np, pd = v3.v2, v3.v1, v3.np, v3.pd
OUT = v2.ROOT / "logs/e16/train"
MODELS = v2.ROOT / "models/exchange_event_v4"
SIDES = ("1P", "2P")
EXPECTED_ROWS, AUC_MIN = 84445, .702
FOLD_COUNT = len(v1.SEEDS)*len(v1.FOLDS)
FEATURES = OUT / "features_aligned"


def synchronized(raw: dict, ids: np.ndarray) -> dict[float, tuple]:
    """原票にはslideが無いので、同じ色収支・NEXT繰上がり判定だけを適用する。"""
    sync = [CountTurnSynchronizer(), CountTurnSynchronizer()]
    result = {}
    for stamp in np.unique(raw["t_sec"][ids]):
        for i in ids[raw["t_sec"][ids] == stamp]:
            side = int(raw["side"][i] == "2P")
            active = raw["chain_mechanism"][i] != ""
            sync[side].observe(raw["grids"][i], np.array(v2.queue_at(raw, i)), float(stamp),
                               not active, chaining=active)
        result[float(stamp)] = tuple(s.accepted for s in sync)
    return result


def count_vectors(exchange: Exchange, active: list[Chain], synced: tuple[SyncedCountSide, ...],
                  raw: dict, start: float, stamp: float, count: int, source: int) -> tuple:
    """既発火分の得点と連鎖中の完走盤面はE15の定義を維持する。"""
    if any(side is None for side in synced):
        missing = np.full(len(v2.NEW_COLUMNS), np.nan)
        return missing, missing
    grids = np.stack([s.grid for s in synced])
    for chain in active:
        if chain.final is None:
            _, _, chain.final = completion(raw["grids"][chain.pre].tobytes(), True, chain.trigger-start)
        grids[chain.side] = np.frombuffer(chain.final, np.int8).reshape(grids.shape[1:])
    totals = np.array([sum(c.score for c in exchange.chains[:count] if c.side == side) for side in range(2)])
    obs = CountObservation(grids, np.stack([s.queue for s in synced]), stamp-start,
                           live=True, score_elapsed_sec=max(0., exchange.trigger-start))
    first = orient(side_features(obs, (False, False)), source)
    third = orient(side_features(obs, (False, False), np.zeros(2), totals), source)
    return first, third


def video_job(video: str) -> str:
    """試合ごとに同期器をリセットし、同一行番号へのcount差分を動画単位保存する。"""
    path = FEATURES / (video+".npz")
    if path.exists():
        return video
    rows = pd.read_csv(v3.OUT / "rows.csv", float_precision="round_trip")
    rows = rows[rows.video_id == video]
    raw = v2.read_npz(v2.RAW / (video.removeprefix("video_")+".npz"))
    row_ids, first, third, ages = [], [], [], []
    for game, part in rows.groupby("game", sort=False):
        ids = np.flatnonzero(raw["game_idx"] == game)
        ids = ids[np.argsort(raw["t_sec"][ids], kind="stable")]
        start = float(raw["t_sec"][ids].min())
        aligned = synchronized(raw, ids)
        states = {(e.number, t): (e, active) for e, t, _, active in samples(raw, ids)}
        for row in part.itertuples():
            exchange, active = states[(row.exchange, row.t_sec)]
            synced = aligned[row.t_sec]
            s1, s3 = count_vectors(exchange, active, synced, raw, start, row.t_sec,
                                    row.fired_chains, int(row.sign < 0))
            row_ids.append(row.row_id)
            first.append(s1)
            third.append(s3)
            ages.append([row.t_sec-s.t_sec if s is not None else np.nan for s in synced])
    v2.save_npz(path, row_id=np.array(row_ids), s1=np.array(first, np.float32),
                s3=np.array(third, np.float32), age=np.array(ages))
    print(f"同期特徴完了 {video} {len(row_ids)}行", flush=True)
    return video


def prepare() -> None:
    """D・到着・得点列、行順、重みを完全保存してcount列だけを更新する。"""
    rows = pd.read_csv(v3.OUT / "rows.csv")
    assert len(rows) == EXPECTED_ROWS
    with ProcessPoolExecutor(v2.WORKERS, initializer=v2.init_worker) as pool:
        list(pool.map(video_job, rows.video_id.unique()))
    values = {s: np.load(v3.OUT / f"{s}_prime.npy") for s in ("S1", "S3")}
    seen = np.zeros(len(rows), int)
    ages = np.full((len(rows), 2), np.nan)
    for path in FEATURES.glob("*.npz"):
        chunk = v2.read_npz(path)
        ids = chunk["row_id"]
        seen[ids] += 1
        ages[ids] = chunk["age"]
        for stage in values:
            values[stage][ids, -len(v2.NEW_COLUMNS):] = chunk[stage.lower()]
    assert (seen == 1).all()
    for stage, matrix in values.items():
        original = np.load(v3.OUT / f"{stage}_prime.npy")
        np.testing.assert_array_equal(matrix[:, :-len(v2.NEW_COLUMNS)], original[:, :-len(v2.NEW_COLUMNS)])
        np.save(OUT / f"{stage}_prime.npy", matrix)
    shutil.copyfile(v3.OUT / "rows.csv", OUT / "rows.csv")
    v1.save_json(OUT / "COHORT.json", dict(rows=len(rows), videos=rows.video_id.nunique(),
        exchanges=len(rows.groupby(["video_id", "game", "exchange"])),
        missing_initial=int(np.isnan(ages).any(axis=1).sum()),
        input_age_median=float(np.nanmedian(ages)), input_age_max=float(np.nanmax(ages))))


def final_fit() -> None:
    """v3を残し、新しいモデルディレクトリだけへ全行学習結果を保存する。"""
    MODELS.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(OUT / "rows.csv")
    target = np.where(rows.sign > 0, rows.label, 1-rows.label)
    manifest = dict(models={}, production_enabled=False, version="E16_synchronized_count")
    for stage, columns in (("S1", v2.S1_COLUMNS), ("S3", v2.S3_COLUMNS)):
        x = np.load(OUT / f"{stage}_prime.npy")
        for suffix in ("", "_light"):
            name = stage+"_prime"+suffix
            params = v1.LIGHT_PARAMS if suffix else {}
            model = v2.HistGradientBoostingClassifier(random_state=v2.SEED, **params)
            model.fit(x, target, sample_weight=rows.weight)
            path = MODELS / (name+".joblib")
            v2.joblib.dump(model, path)
            manifest["models"][name] = dict(file=path.name, sha256=v1.file_sha256(path),
                columns=columns+v2.NEW_COLUMNS, valid=True, version=manifest["version"])
    v1.save_json(MODELS / "manifest.json", manifest)


def main() -> None:
    """E15と同じCVを実行し、閾値を事後変更せず結果を保存する。"""
    OUT.mkdir(parents=True, exist_ok=True)
    v1.save_json(OUT / "STATUS.json", dict(stage="FEATURES"))
    prepare()
    v1.save_json(OUT / "STATUS.json", dict(stage="CV"))
    with ProcessPoolExecutor(v2.WORKERS, initializer=v2.init_worker) as pool:
        list(pool.map(v3.fold_job, [(str(OUT), s, f) for s in v1.SEEDS for f in v1.FOLDS]))
    from scripts.compare_e15b_models_20260928 import measurements
    frames = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("seed_*/fold_*/predictions.csv"))])
    metrics = {name: measurements(frames, name) for name in v3.NAMES}
    assert len(frames) == EXPECTED_ROWS * len(v1.SEEDS)
    v1.save_json(OUT / "METRICS.json", dict(metrics=metrics, rows=EXPECTED_ROWS, folds=FOLD_COUNT,
        predictions=len(frames), threshold=AUC_MIN, passed=metrics["S3_prime_light"]["auc"] >= AUC_MIN))
    final_fit()
    v1.save_json(OUT / "STATUS.json", dict(stage="COMPLETE"))


if __name__ == "__main__":
    main()

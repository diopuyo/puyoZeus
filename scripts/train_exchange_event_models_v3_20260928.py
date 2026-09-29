"""E15案A: STABLE更新行・撃ち合い単位重み・固定15動画foldでv3を学習する。"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import traceback

from scripts import train_exchange_event_models_v2_20260927 as v2
from scripts.e15_training_rows_20260928 import game_rows

np, pd, v1 = v2.np, v2.pd, v2.v1
OUT = v2.ROOT / "logs/e15"
MODELS = v2.ROOT / "models/exchange_event_v3"
WORKERS = v2.WORKERS
ORACLE_LIMIT = .722
NAMES = ("S3_prime", "S3_prime_light")
PROTOCOL = dict(version="E15_live_count", row_unit="いずれかの側のSTABLE盤面/NEXT更新",
    admission="評価時刻t_secで中盤判定。未来の終点による採否なし",
    scores="観測済み連鎖だけの側別終点得点差を発火時完走予測の代理とする",
    weight="撃ち合い単位で総和1", folds="既存3seed×5動画fold",
    a=dict(baseline="logs/f1b_counter_features/METRICS.json", oracle_max=ORACLE_LIMIT),
    b=dict(q_log_loss_max=.5708, zenchi_agreement_min=.7955, false_death_max=1, death_denominator=28),
    c=dict(nf_2613_9_min=700, jump_2614_08_max=.66, p_2694_3_min=.566,
           prefire_probability="発火前から段階的に上昇"),
    d=dict(report="同一入力のE14/v3 OFF/ON count同値率、更新追従率"))


@lru_cache(maxsize=1)
def reference_inputs() -> tuple[dict[str, pd.DataFrame], np.ndarray]:
    """全動画の巨大CSVはworkerごとに一度だけ読む。値と行順は不変。"""
    rows = pd.read_csv(v1.run_path(v2.REFERENCE, "base") / "rows.csv")
    parts = dict(tuple(rows.groupby("video_id", sort=False)))
    parts[""] = rows.iloc[:0]
    design = np.load(v1.run_path(v2.REFERENCE, "base") / "design.npy", mmap_mode="r")
    return parts, design


def video_job(job: tuple[str, str]) -> str:
    """動画・試合ごとのチェックポイントから再開する。"""
    video, output = job
    root = Path(output) / "features" / video
    raw = v2.read_npz(v2.RAW / (video.removeprefix("video_") + ".npz"))
    parts, design = reference_inputs()
    rows = parts.get(video, parts[""])
    for game in np.unique(raw["game_idx"]):
        path = root / f"game_{game}.npz"
        meta = path.with_suffix(".json")
        if path.exists() and meta.exists():
            continue
        ids = np.flatnonzero(raw["game_idx"] == game)
        ids = ids[np.argsort(raw["t_sec"][ids], kind="stable")]
        part = rows[rows.game_id == f"{video}:{game}"]
        values = list(game_rows(raw, ids, part, design))
        info = [dict(video_id=video, **v[0]) for v in values]
        v2.save_npz(path, s1=np.asarray([v[1] for v in values], dtype=np.float32),
                    s3=np.asarray([v[2] for v in values], dtype=np.float32))
        v1.save_json(meta, info)
    print(f"特徴完了 {video}", flush=True)
    return video


def prepare(output: Path) -> pd.DataFrame:
    """148原票を走査し、ラベル・過去Dのある行だけを採用する。"""
    v1.validate_rows(v2.REFERENCE, v2.SHARED / "data/video_tier_index_2026-08-07.tsv")
    tiers = pd.read_csv(v2.SHARED / "data/video_tier_index_2026-08-07.tsv", sep="\t").set_index("video_name")
    videos = ["video_"+p.stem for p in sorted(v2.RAW.glob("*.npz"))]
    for video in videos:
        if video not in tiers.index or not any(t in str(tiers.loc[video, "tier"]) for t in v1.ALLOWED_TIERS):
            raise ValueError(f"ティア未確認: {video}")
    with ProcessPoolExecutor(WORKERS, initializer=v2.init_worker) as pool:
        list(pool.map(video_job, [(v, str(output)) for v in videos]))
    records, first, third = [], [], []
    for path in sorted((output / "features").glob("*/game_*.npz")):
        info = json.loads(path.with_suffix(".json").read_text())
        if info:
            chunk = v2.read_npz(path)
            records.extend(info)
            first.append(chunk["s1"])
            third.append(chunk["s3"])
    rows = pd.DataFrame(records)
    rows["row_id"] = np.arange(len(rows))
    rows["weight"] = 1 / rows.groupby(["video_id", "game", "exchange"]).t_sec.transform("size")
    assert (rows[["board_sec_1p", "board_sec_2p"]].max(axis=1) <= rows.t_sec).all()
    rows.to_csv(output / "rows.csv", index=False)
    np.save(output / "S1_prime.npy", np.concatenate(first))
    np.save(output / "S3_prime.npy", np.concatenate(third))
    v1.save_json(output / "COUNTS.json", dict(input_videos=len(videos), videos=rows.video_id.nunique(),
        rows=len(rows), exchanges=len(rows.groupby(["video_id", "game", "exchange"]))))
    return rows


def fold_job(job: tuple[str, int, int]) -> str:
    """参照動画分割を固定し、学習重みを撃ち合い単位で正規化する。"""
    output_text, seed, fold = job
    output = Path(output_text)
    path = output / f"seed_{seed}/fold_{fold}/predictions.csv"
    if path.exists():
        return str(path)
    rows = pd.read_csv(output / "rows.csv")
    ref = pd.read_csv(v1.run_path(v2.REFERENCE, "base") / f"seed_{seed}/fold_{fold}/de_predictions.csv")
    test = rows.video_id.isin(ref.video_id.unique()).to_numpy()
    assert not set(rows.loc[test, "video_id"]) & set(rows.loc[~test, "video_id"])
    target = np.where(rows.sign > 0, rows.label, 1-rows.label)
    x = np.load(output / "S3_prime.npy", mmap_mode="r")
    result = rows.loc[test].copy()
    for name in NAMES:
        params = v1.LIGHT_PARAMS if name.endswith("_light") else {}
        model = v2.HistGradientBoostingClassifier(random_state=v2.SEED, **params)
        model.fit(x[~test], target[~test], sample_weight=rows.weight.to_numpy()[~test])
        p = model.predict_proba(x[test])[:, 1]
        result[name] = np.where(rows.sign.to_numpy()[test] > 0, p, 1-p)
    result["seed"], result["fold"] = seed, fold
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path.with_suffix(".tmp"), index=False)
    path.with_suffix(".tmp").replace(path)
    print(f"CV完了 {seed}/{fold}", flush=True)
    return str(path)


def cv_report(output: Path) -> dict:
    """固定閾値を適用し、オラクル超過は合格にせず停止する。"""
    frame = pd.concat([pd.read_csv(p) for p in sorted(output.glob("seed_*/fold_*/predictions.csv"))])
    assert not frame.duplicated(["seed", "row_id"]).any()
    counts = json.loads((output / "COUNTS.json").read_text())
    assert len(frame) == counts["rows"] * len(v1.SEEDS)
    baseline = json.loads((v2.OUT / "METRICS.json").read_text())["metrics"]["middle"]
    metrics = {}
    for name in NAMES:
        auc = v2.roc_auc_score(frame.label, frame[name])
        metrics[name] = dict(auc=float(auc), weighted_auc=float(v2.roc_auc_score(
            frame.label, frame[name], sample_weight=frame.weight)),
            log_loss=float(v2.log_loss(frame.label, frame[name])),
            v2_auc=baseline[name]["auc"], pass_a=bool(baseline[name]["auc"] <= auc <= ORACLE_LIMIT))
    suspect = any(m["auc"] > ORACLE_LIMIT for m in metrics.values())
    return dict(protocol=PROTOCOL, a=dict(**counts, predictions=len(frame), folds=len(v1.SEEDS)*len(v1.FOLDS),
        metrics=metrics, leak_suspected=suspect, passed=metrics["S3_prime_light"]["pass_a"] and not suspect),
        b=dict(status="未実行"), c=dict(status="未実行"), d=dict(status="未実行"),
        status="STOP_ORACLE_EXCEEDED" if suspect else "CV_COMPLETE")


def final_fit(output: Path) -> None:
    """v2を破壊せずv3ディレクトリだけに実験モデルを保存する。"""
    MODELS.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(output / "rows.csv")
    target = np.where(rows.sign > 0, rows.label, 1-rows.label)
    manifest = dict(models={}, production_enabled=False, version="E15_live_count", protocol=PROTOCOL)
    for stage, columns in (("S1", v2.S1_COLUMNS), ("S3", v2.S3_COLUMNS)):
        x = np.load(output / (stage+"_prime.npy"), mmap_mode="r")
        for suffix in ("", "_light"):
            name = stage+"_prime"+suffix
            params = v1.LIGHT_PARAMS if suffix else {}
            model = v2.HistGradientBoostingClassifier(random_state=v2.SEED, **params)
            model.fit(x, target, sample_weight=rows.weight)
            path = MODELS / (name+".joblib")
            v2.joblib.dump(model, path)
            manifest["models"][name] = dict(file=path.name, sha256=v1.file_sha256(path),
                columns=columns+v2.NEW_COLUMNS, valid=True, version="E15_live_count")
    v1.save_json(MODELS / "manifest.json", manifest)


def main() -> None:
    """長時間実行はログと試合・foldチェックポイントを残す。"""
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "train.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            path = args.output / "PROTOCOL.json"
            if path.exists() and json.loads(path.read_text()) != PROTOCOL:
                raise ValueError("事前登録が変更された")
            v1.save_json(path, PROTOCOL)
            v1.save_json(args.output / "STATUS.json", dict(stage="FEATURES"))
            prepare(args.output)
            v1.save_json(args.output / "STATUS.json", dict(stage="CV"))
            with ProcessPoolExecutor(WORKERS, initializer=v2.init_worker) as pool:
                list(pool.map(fold_job, [(str(args.output), s, f) for s in v1.SEEDS for f in v1.FOLDS]))
            report = cv_report(args.output)
            v1.save_json(args.output / "METRICS.json", report)
            if not report["a"]["leak_suspected"]:
                final_fit(args.output)
            v1.save_json(args.output / "STATUS.json", dict(stage=report["status"]))
        except Exception:
            v1.save_json(args.output / "ERROR.json", dict(traceback=traceback.format_exc()))
            raise


if __name__ == "__main__":
    main()

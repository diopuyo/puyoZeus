"""F1b: 盤面参照時刻を全行監査し、未来盤面を除いたS3を再評価する。"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from typing import Any

from scripts import train_exchange_event_models_v2_20260927 as f1

np, pd, v1 = f1.np, f1.pd, f1.v1
ROOT, OUT = f1.ROOT, f1.OUT
PREVIOUS = ROOT / "logs/f1_counter_features"
S3_NAMES = ("S3_prime", "S3_prime_light")


def board_times(raw: dict, ids: np.ndarray) -> np.ndarray:
    """未取得盤面を末尾行へ誤参照せず欠測のまま返す。"""
    return np.where(ids >= 0, raw["t_sec"][np.maximum(ids, 0)], np.nan)


def audit_video(video: str) -> pd.DataFrame:
    """評価時刻の欠測を代理時刻で捏造せず、各側の参照時刻を保存する。"""
    raw = f1.read_npz(f1.RAW / (video.removeprefix("video_") + ".npz"))
    pre = f1.read_npz(v1.run_path(f1.REFERENCE, "s1") / "features" / (video + ".npz"))
    post = f1.read_npz(v1.run_path(f1.REFERENCE, "s3") / "features" / (video + ".npz"))
    np.testing.assert_array_equal(pre["row_id"], post["row_id"])
    before, after = board_times(raw, pre["pre_ids"]), board_times(raw, post["post_ids"])
    trigger = pre["trigger"]
    frame = pd.DataFrame(dict(video_id=video, row_id=pre["row_id"], trigger_sec=trigger,
        s3_evaluation_sec=np.full(len(trigger), np.nan), t10_endpoint_sec=pre["endpoint"]))
    for side, idx in enumerate(f1.SIDES):
        frame[f"s1_{idx}_board_sec"] = before[:, side]
        frame[f"old_s3_{idx}_board_sec"] = after[:, side]
        frame[f"corrected_s3_{idx}_source_sec"] = before[:, side]
        frame[f"s1_{idx}_available"] = np.isfinite(before[:, side]) & (before[:, side] < trigger)
        frame[f"s1_{idx}_future"] = np.isfinite(before[:, side]) & (before[:, side] > trigger)
    frame["s1_both_available"] = frame[[f"s1_{s}_available" for s in f1.SIDES]].all(axis=1)
    frame["s1_any_future"] = frame[[f"s1_{s}_future" for s in f1.SIDES]].any(axis=1)
    frame["s1_any_missing"] = ~np.isfinite(before).all(axis=1)
    frame["old_s3_oracle_endpoint"] = True
    frame["old_s3_after_trigger"] = (after > trigger[:, None]).all(axis=1)
    frame["old_s3_status"] = "unknown_score_finalize_timestamp"
    frame["corrected_s3_any_future"] = frame.s1_any_future
    frame["corrected_s3_both_guaranteed_before_evaluation"] = frame.s1_both_available
    return frame


def audit(output: Path) -> dict:
    """同じ294616行で件数と割合を計算し、区間平均で水増ししない。"""
    path = output / "AUDIT.json"
    if path.exists():
        return json.loads(path.read_text())
    rows = pd.read_csv(PREVIOUS / "rows.csv")
    with ProcessPoolExecutor(f1.WORKERS, initializer=f1.init_worker) as pool:
        frame = pd.concat(pool.map(audit_video, sorted(rows.video_id.unique())), ignore_index=True)
    frame = frame.sort_values("row_id")
    np.testing.assert_array_equal(frame.row_id, rows.row_id)
    frame.to_csv(output / "BOARD_TIME_AUDIT.csv.gz", index=False, compression="gzip")
    keys = ("s1_both_available", "s1_any_future", "s1_any_missing", "old_s3_oracle_endpoint",
            "old_s3_after_trigger", "corrected_s3_any_future",
            "corrected_s3_both_guaranteed_before_evaluation")
    report = {key: dict(count=int(frame[key].sum()), fraction=float(frame[key].mean())) for key in keys}
    report.update(rows=len(frame), videos=int(frame.video_id.nunique()),
        exact_s3_evaluation_missing=dict(count=len(frame), fraction=1.),
        old_s3_exact_before=0, old_s3_exact_after=0, old_s3_exact_unknown=len(frame),
        correction="NF/応手の実観測盤面・NEXTは全てtriggerより前。応手だけその盤面の完走予測を使用。",
        score_timing="T11/E1の得点差は固定。着地前に確定していたかは本資産では検証不能。")
    if report["s1_any_future"]["count"]:
        raise ValueError("S1の発火前盤面に未来参照あり")
    v1.save_json(path, report)
    return report


def preserve_previous(output: Path) -> None:
    """旧成果を退避し、リーク版S3の採用表示を無効化する。"""
    directory = ROOT / "models/exchange_event_v2"
    backup = output / "superseded_models"
    if not backup.exists():
        shutil.copytree(directory, backup)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in S3_NAMES:
        if name in manifest["models"]:
            manifest["models"][name]["valid"] = False
            manifest["models"][name]["invalidation_reason"] = "T10着地後盤面をS3へ使用。F1bで差替え。"
    manifest["production_enabled"] = False
    v1.save_json(directory / "manifest.json", manifest)
    v1.save_json(PREVIOUS / "SUPERSEDED_S3.json", dict(reason="着地後盤面の未来参照",
        replacement=str(output), withdrawn_models=list(S3_NAMES)))


def corrected_video(job: tuple[str, str]) -> str:
    """S1特徴は再用し、S3応手だけ発火前入力から計算し直す。"""
    video, output_text = job
    path = Path(output_text) / "features" / (video + ".npz")
    old = f1.read_npz(PREVIOUS / "features" / (video + ".npz"))
    data = f1.read_npz(path) if path.exists() else dict(row_id=old["row_id"], s1=old["s1"],
        s3=np.full_like(old["s3"], np.nan), done=np.zeros(len(old["row_id"]), bool))
    if data["done"].all():
        return video
    raw = f1.read_npz(f1.RAW / (video.removeprefix("video_") + ".npz"))
    pre = f1.read_npz(v1.run_path(f1.REFERENCE, "s1") / "features" / (video + ".npz"))
    post = f1.read_npz(v1.run_path(f1.REFERENCE, "s3") / "features" / (video + ".npz"))
    rows = pd.read_csv(PREVIOUS / "rows.csv", usecols=["row_id", "sign"]).set_index("row_id")
    signs = rows.loc[pre["row_id"], "sign"].to_numpy()
    starts = {g: raw["t_sec"][raw["game_idx"] == g].min() for g in np.unique(raw["game_idx"])}
    cache = {}
    for number, i in enumerate(np.flatnonzero(~data["done"])):
        key = (*pre["pre_ids"][i], *post["post_ids"][i], float(pre["trigger"][i]))
        if key not in cache:
            firing = pre["values"][i, -2:]
            if signs[i] < 0:
                firing = firing[::-1]
            cache[key] = corrected_event(raw, pre["pre_ids"][i], post["post_ids"][i],
                                         float(pre["trigger"][i]), starts, firing)
        data["s3"][i] = f1.orient(cache[key], signs[i])
        np.testing.assert_allclose(data["s3"][i, :-2], old["s1"][i, :-2], equal_nan=True)
        data["done"][i] = True
        if (number + 1) % f1.CHECKPOINT == 0:
            f1.save_npz(path, **data)
    f1.save_npz(path, **data)
    print(f"F1b特徴完了 {video}", flush=True)
    return video


def corrected_event(raw: dict, pre: np.ndarray, post: np.ndarray, trigger: float,
                    starts: dict, firing: np.ndarray) -> np.ndarray:
    """postから読む列を得点だけに限定し、時刻とNEXTの間接リークも除く。"""
    valid = pre[pre >= 0]
    if not len(valid):
        return np.full((2, len(f1.iv.NEAR_FUTURE_K_LEVELS)+1), np.nan)
    game = raw["game_idx"][valid[0]]
    elapsed = float(trigger-starts[game])
    _, _, boards = f1.completion(raw, pre, firing, elapsed)
    before = np.asarray([raw["score"][i] if i >= 0 else np.nan for i in pre])
    after = raw["score"][post].astype(float)
    missing = ~np.isfinite(before) | (before < 0) | (after < 0)
    sends = np.where(missing, np.nan, np.maximum(after-before, 0)/f1.compute_effective_rate(elapsed))
    return f1.stage_features(raw, pre, sends, np.zeros(2, dtype=int), elapsed, boards)


def prepare(output: Path) -> None:
    """行・既存列・S1を固定し、S3の追加17列だけを差し替える。"""
    rows = pd.read_csv(PREVIOUS / "rows.csv")
    videos = sorted(rows.video_id.unique())
    with ProcessPoolExecutor(f1.WORKERS, initializer=f1.init_worker) as pool:
        list(pool.map(corrected_video, [(v, str(output)) for v in videos]))
    chunks = [f1.read_npz(output / "features" / (v + ".npz")) for v in videos]
    ids = np.concatenate([c["row_id"] for c in chunks])
    order = np.argsort(ids)
    np.testing.assert_array_equal(ids[order], rows.row_id)
    new = np.concatenate([c["s3"] for c in chunks])[order]
    original = np.load(PREVIOUS / "S3.npy", mmap_mode="r")
    np.save(output / "S3_prime.npy", np.column_stack((original, new)))
    rows.to_csv(output / "rows.csv", index=False)


def fold_job(job: tuple[str, int, int]) -> str:
    """旧モデルとS1の保存OOFは不変、S3の追加特徴版だけを再学習する。"""
    output_text, seed, fold = job
    output = Path(output_text)
    relative = Path(f"seed_{seed}/fold_{fold}/predictions.csv")
    path = output / relative
    if path.exists():
        return str(path)
    rows = pd.read_csv(output / "rows.csv")
    result = pd.read_csv(PREVIOUS / relative)
    test = rows.video_id.isin(result.video_id.unique()).to_numpy()
    np.testing.assert_array_equal(rows.loc[test, "row_id"], result.row_id)
    sign, label = rows.sign.to_numpy(), rows.label.to_numpy()
    target = np.where(sign > 0, label, 1-label)
    x = np.load(output / "S3_prime.npy", mmap_mode="r")
    for name in S3_NAMES:
        params = v1.LIGHT_PARAMS if name.endswith("_light") else {}
        model = f1.HistGradientBoostingClassifier(random_state=f1.SEED, **params).fit(x[~test], target[~test])
        p = model.predict_proba(x[test])[:, 1]
        result[name] = np.where(sign[test] > 0, p, 1-p)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path.with_suffix(".tmp"), index=False)
    path.with_suffix(".tmp").replace(path)
    print(f"F1b CV完了 seed={seed} fold={fold}", flush=True)
    return str(path)


def bootstrap(output: Path) -> None:
    """同じ動画抽出列で比較し、不変な旧モデルの集計値は再用する。"""
    f1.prediction_frames(output)
    for scope in ("middle", "overall"):
        for name in f1.NAMES:
            if name not in S3_NAMES:
                filename = f"bootstrap_{scope}_{name}.npz"
                shutil.copyfile(PREVIOUS / filename, output / filename)
    jobs = [(str(output), scope, name) for scope in ("middle", "overall") for name in S3_NAMES]
    with ProcessPoolExecutor(f1.WORKERS, initializer=f1.init_worker) as pool:
        list(pool.map(f1.bootstrap_job, jobs))


def save_models(output: Path, report: dict) -> dict:
    """旧リーク版を残さず是正版へ上書きし、採否と運用OFFを明示する。"""
    directory = ROOT / "models/exchange_event_v2"
    manifest = json.loads((output / "superseded_models/manifest.json").read_text())
    rows = pd.read_csv(output / "rows.csv")
    x = np.load(output / "S3_prime.npy", mmap_mode="r")
    target = np.where(rows.sign > 0, rows.label, 1-rows.label)
    for name in S3_NAMES:
        params = v1.LIGHT_PARAMS if name.endswith("_light") else {}
        model = f1.HistGradientBoostingClassifier(random_state=f1.SEED, **params).fit(x, target)
        path = directory / (name + ".joblib")
        f1.joblib.dump(model, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
        loaded = f1.joblib.load(path)
        assert np.isfinite(loaded.predict_proba(x[:100])).all()
        manifest["models"][name] = dict(file=path.name, sha256=v1.file_sha256(path),
            columns=f1.S3_COLUMNS+f1.NEW_COLUMNS, parameters=model.get_params(),
            decision=report["decision"][name], valid=True, version="F1b_prefire_frozen",
            board_causality="入力は発火前に限定。S3時点の最新性・得点の観測時刻は未検証。")
    manifest.update(definitions=f1.DEFINITIONS, production_enabled=False,
        protocol_sha256=v1.file_sha256(output / "PROTOCOL.json"), audit=str(output / "AUDIT.json"))
    v1.save_json(directory / "manifest.json", manifest)
    return manifest["models"]


def protocol(output: Path) -> None:
    """旧CV・閾値を保ち、盤面選択是正だけを結果確認前に固定する。"""
    value = json.loads((PREVIOUS / "PROTOCOL.json").read_text())
    value.update(definitions=f1.DEFINITIONS, correction="F1b_prefire_frozen",
        original_protocol_sha256=v1.file_sha256(PREVIOUS / "PROTOCOL.json"),
        exact_s3_timestamp_available=False,
        model_policy="S3標準・軽量を是正版で上書き、機械採否を記録し運用OFF。S1/v1は不変。")
    path = output / "PROTOCOL.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("F1bの事前登録が再開元と不一致")
    v1.save_json(path, value)


def execute(output: Path) -> None:
    """監査から採用判定・是正モデルの保存まで再開可能に進める。"""
    started = time.time()
    protocol(output)
    preserve_previous(output)
    v1.save_json(output / "STATUS.json", dict(stage="AUDIT", pid=os.getpid()))
    audited = audit(output)
    v1.save_json(output / "STATUS.json", dict(stage="FEATURES", pid=os.getpid()))
    prepare(output)
    v1.save_json(output / "STATUS.json", dict(stage="CV", pid=os.getpid()))
    jobs = [(str(output), s, f) for s in v1.SEEDS for f in v1.FOLDS]
    with ProcessPoolExecutor(f1.WORKERS, initializer=f1.init_worker) as pool:
        list(pool.map(fold_job, jobs))
    v1.save_json(output / "STATUS.json", dict(stage="BOOTSTRAP", pid=os.getpid()))
    bootstrap(output)
    report = f1.collect_report(output)
    report.update(audit=audited, status="COMPLETE_CONSERVATIVE_BOARD_CORRECTION",
        exact_latest_board_audit_complete=False, elapsed_seconds=time.time()-started)
    report["models"] = save_models(output, report)
    v1.save_json(output / "METRICS.json", report)
    f1.summary(output, report)
    v1.save_json(output / "STATUS.json", dict(stage="COMPLETE", pid=os.getpid()))
    (output / "ERROR.json").unlink(missing_ok=True)


def main() -> None:
    """排他ロック付きでCPU6並列・nice19の独立runを起動する。"""
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    f1.init_worker()
    with (args.output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(args.output)
        except Exception:
            v1.save_json(args.output / "ERROR.json", dict(traceback=traceback.format_exc()))
            raise


if __name__ == "__main__":
    main()

"""保存済み同一行CVを照合し、同期保持の時間分布と損失差を出す。"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scripts import train_exchange_event_models_v3_20260928 as v3
from scripts.compare_e15b_models_20260928 import measurements
from scripts.report_e16_diagnostics_20260928 import cv_identity
from scripts.run_e3_exchange_eval_20260926 import save_json

OUT, SYNC = Path("logs/e17"), Path("logs/e16/train")
THRESHOLDS = (1., 3., 5., 10., 30., 60.)
QUANTILES = (.5, .9, .95, .99, 1.)
IDENTITY = ["row_id", "video_id", "game", "exchange", "t_sec", "weight", "label", "sign", "seed", "fold"]
TIME_TOLERANCE = 1e-5


def input_differences(rows: pd.DataFrame, ages: np.ndarray) -> np.ndarray:
    """保持値とE15最新確定値が本当に異なるか、元盤面/NEXTで判定する。"""
    different = np.zeros(ages.shape, bool)
    for video, part in rows.groupby("video_id", sort=False):
        raw = v3.v2.read_npz(v3.v2.RAW/(video.removeprefix("video_")+".npz"))
        for game, group in part.groupby("game", sort=False):
            for side, label in enumerate(("1P", "2P")):
                ids = np.flatnonzero((raw["game_idx"] == game) & (raw["side"] == label))
                ids = ids[np.argsort(raw["t_sec"][ids], kind="stable")]
                stamps = raw["t_sec"][ids].astype(float)
                for row in group.itertuples():
                    if not np.isfinite(ages[row.row_id, side]):
                        continue
                    accepted = row.t_sec-ages[row.row_id, side]
                    latest = (row.board_sec_1p, row.board_sec_2p)[side]
                    pair = [int(np.argmin(abs(stamps-t))) for t in (accepted, latest)]
                    assert all(abs(stamps[i]-t) < TIME_TOLERANCE for i, t in zip(pair, (accepted, latest)))
                    a, b = ids[pair]
                    different[row.row_id, side] = (not np.array_equal(raw["grids"][a], raw["grids"][b])
                        or v3.v2.queue_at(raw, a) != v3.v2.queue_at(raw, b))
    return different


def distribution(age: np.ndarray, mask: np.ndarray) -> dict:
    """各行で長い側の保持時間を用い、全行を分母として長期保持率を出す。"""
    valid = mask & np.isfinite(age)
    return dict(rows=len(age), eligible=int(valid.sum()),
        quantiles_sec={str(q): float(np.quantile(age[valid], q)) for q in QUANTILES},
        over_seconds={str(t): dict(rows=int(np.count_nonzero(valid & (age > t))),
            fraction=float(np.mean(valid & (age > t)))) for t in THRESHOLDS})


def cv_by_age(ages: np.ndarray) -> dict:
    """全OOF行の同一性を検査し、保持時間帯別の行平均損失を比較する。"""
    frames = {}
    for name, root in (("E15", v3.OUT), ("sync", SYNC)):
        frame = pd.concat([pd.read_csv(p) for p in sorted(root.glob("seed_*/fold_*/predictions.csv"))])
        frames[name] = frame.sort_values(["seed", "row_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(frames["E15"][IDENTITY], frames["sync"][IDENTITY], check_exact=True)
    age = ages[frames["E15"].row_id.to_numpy()]
    metrics = {k: {m: measurements(f, m) for m in v3.NAMES} for k, f in frames.items()}
    bins = {}
    for low, high in zip((0., *THRESHOLDS), (*THRESHOLDS, np.inf)):
        mask = (age >= low) & (age < high)
        values = {k: measurements(f[mask], "S3_prime_light") for k, f in frames.items()}
        bins[f"{low:g}..{high:g}"] = dict(predictions=int(mask.sum()), rows=int(mask.sum()/len(v3.v1.SEEDS)),
            metrics=values, delta_log_loss=values["sync"]["log_loss"]-values["E15"]["log_loss"])
    return dict(identity=cv_identity(), metrics=metrics, by_age=bins)


def main() -> None:
    """同期特徴を再学習せず、生成済み全行の保持証跡を検査する。"""
    rows = pd.read_csv(v3.OUT/"rows.csv", float_precision="round_trip")
    ages = np.full((len(rows), 2), np.nan)
    seen = np.zeros(len(rows), int)
    for path in sorted((SYNC/"features_aligned").glob("*.npz")):
        with np.load(path) as chunk:
            ages[chunk["row_id"]] = chunk["age"]
            seen[chunk["row_id"]] += 1
    assert (seen == 1).all()
    different = input_differences(rows, ages)
    maximum = np.max(ages, axis=1)
    stale_max = np.max(np.where(different, ages, 0.), axis=1)
    before = np.load(v3.OUT/"S3_prime.npy")
    after = np.load(SYNC/"S3_prime.npy")
    start = len(v3.v2.S3_COLUMNS)
    np.testing.assert_array_equal(before[:, :start], after[:, :start])
    changed = ~np.all((before == after) | (np.isnan(before) & np.isnan(after)), axis=1)
    report = dict(rows=len(rows), missing_initial=int(np.isnan(ages).any(axis=1).sum()),
        input_age=distribution(maximum, np.ones(len(rows), bool)),
        different_from_latest=distribution(stale_max, different.any(axis=1)),
        count_feature_changed_rows=int(changed.sum()), cv=cv_by_age(maximum))
    save_json(OUT/"HOLD_AUDIT.json", report)
    np.savez_compressed(OUT/"hold_rows.npz", age=ages, different=different, feature_changed=changed)
    print(report, flush=True)


if __name__ == "__main__":
    main()

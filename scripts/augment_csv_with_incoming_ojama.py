"""score_series_cache.json から incoming_ojama を集計して CSV に列追加する。

入力:
    data/training/match_features_v2.csv  (1390 行 × 17 特徴量、incoming_ojama_pressure は定数)
    data/training/score_series_cache.json

処理:
    各サンプル (video_id, match_idx, time_phase) について:
      1. 該当試合の score 時系列をキャッシュから取得
      2. infer_timeline_from_score_series で連鎖イベントの ojama 予測を生成
      3. 「サンプル時刻 t の直近 PRESSURE_WINDOW_SEC 秒間に 1P/2P が受けた ojama 累計」を計算
      4. 1P 視点の incoming_ojama_diff = (1P 受けた累計) - (2P 受けた累計)
      5. 既存 CSV の incoming_ojama_pressure 列を上書き or 新規列を追加
        - 0..1 正規化: pressure = clip(diff / NORM_DIVISOR, -1, 1) で対称化
        - 既存指標と同じ「1P 視点 - 2P 視点」差分で揃える

出力:
    data/training/match_features_v3.csv  (incoming_ojama_pressure 修復済)
    data/training/match_features_v3_meta.json  (集計メタ)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ojama_score_inferrer import OjamaScoreInferrer

DEFAULT_CSV_IN = Path("data/training/match_features_v2.csv")
DEFAULT_CACHE = Path("data/training/score_series_cache.json")
DEFAULT_CSV_OUT = Path("data/training/match_features_v3.csv")
DEFAULT_META_OUT = Path("data/training/match_features_v3_meta.json")

# 集計窓: サンプル時刻 t の直前 X 秒間の ojama を「現在の incoming pressure」とする
PRESSURE_WINDOW_SEC: float = 10.0
# 正規化分母: 30 個 (= ojama 1 段) を pressure=1.0 とする
PRESSURE_NORM_DIVISOR: float = 30.0
# OjamaScoreInferrer の min_chain_score
MIN_CHAIN_SCORE: int = 40


def compute_phase_time(start: float, end: float, phase: str) -> float | None:
    """generate_training_dataset_v2.py と同じロジックで時刻を計算。"""
    duration = end - start
    mid = (start + end) / 2.0
    table = {
        "start_plus_0": start + 1.0,
        "start_plus_15": start + 15.0,
        "start_plus_30": start + 30.0,
        "mid_minus_30": mid - 30.0,
        "mid_minus_15": mid - 15.0,
        "midpoint": mid,
        "mid_plus_15": mid + 15.0,
        "mid_plus_30": mid + 30.0,
        "end_minus_15": end - 15.0,
        "end_minus_5": end - 5.0,
    }
    return table.get(phase)


def aggregate_incoming(
    predictions: list,
    sample_time: float,
    window: float,
) -> tuple[int, int]:
    """サンプル時刻 t の直前 window 秒間に 1P/2P が受けた ojama 累計を返す。

    OjamaPrediction.fired_at_sec は match_start からの elapsed なので、
    呼び出し側で sample_time も elapsed に変換しておくこと。
    """
    incoming_1p = 0
    incoming_2p = 0
    lower = sample_time - window
    upper = sample_time
    for p in predictions:
        if not (lower <= p.fired_at_sec <= upper):
            continue
        if p.side == "1P":
            incoming_1p += p.pending
        elif p.side == "2P":
            incoming_2p += p.pending
    return incoming_1p, incoming_2p


def normalize_pressure(diff: int) -> float:
    """incoming_diff を [-1, 1] に正規化 (1P 視点)。"""
    v = diff / PRESSURE_NORM_DIVISOR
    return max(-1.0, min(1.0, v))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-in", default=str(DEFAULT_CSV_IN))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--csv-out", default=str(DEFAULT_CSV_OUT))
    parser.add_argument("--meta-out", default=str(DEFAULT_META_OUT))
    parser.add_argument("--window", type=float, default=PRESSURE_WINDOW_SEC)
    parser.add_argument("--norm", type=float, default=PRESSURE_NORM_DIVISOR)
    args = parser.parse_args()

    csv_in = Path(args.csv_in)
    cache_path = Path(args.cache)
    csv_out = Path(args.csv_out)
    meta_out = Path(args.meta_out)

    # キャッシュ読込
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    print(f"cache loaded: videos={list(cache.keys())}")

    # 各試合の予測時系列を事前計算 (キャッシュ → predictions)
    inferrer = OjamaScoreInferrer()
    match_predictions: dict[tuple[str, str], tuple[float, list]] = {}
    boundaries: dict[tuple[str, str], tuple[float, float]] = {}
    for vid, matches in cache.items():
        for match_idx, samples in matches.items():
            valid_series = [
                (s["t"], int(s["1p"]), int(s["2p"]))
                for s in samples
                if s["1p"] is not None and s["2p"] is not None
            ]
            if len(valid_series) < 2:
                match_predictions[(vid, match_idx)] = (0.0, [])
                continue
            # 試合開始 = キャッシュ内最初の time
            start = valid_series[0][0]
            preds = inferrer.infer_timeline_from_score_series(
                valid_series, match_start_sec=start, min_chain_score=MIN_CHAIN_SCORE,
            )
            match_predictions[(vid, match_idx)] = (start, preds)
            # 試合範囲も保存 (start, end)
            boundaries[(vid, match_idx)] = (start, valid_series[-1][0])
    print(f"predictions ready for {len(match_predictions)} matches")

    # CSV 読込 + 列上書き
    rows: list[dict] = []
    with open(csv_in, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for r in reader:
            rows.append(r)
    print(f"csv loaded: rows={len(rows)} cols={len(fieldnames)}")
    if "incoming_ojama_pressure" not in fieldnames:
        print("[error] CSV に incoming_ojama_pressure 列がない")
        return 1

    # incoming_ojama_pressure を上書き
    updated = 0
    skipped = 0
    pressure_values: list[float] = []
    for r in rows:
        # match_idx は CSV では数値文字列
        # キャッシュ側でも数値文字列キーで保存しているので合わせる
        # CSV の video_id は "01"/"02"/"03"、キャッシュキーは "video_01" 等
        raw_vid = r["video_id"]
        vid = raw_vid if raw_vid.startswith("video_") else f"video_{raw_vid}"
        midx = r["match_idx"]
        phase = r["time_phase"]
        bdy = boundaries.get((vid, midx))
        preds_pair = match_predictions.get((vid, midx))
        if bdy is None or preds_pair is None or not preds_pair[1]:
            skipped += 1
            continue
        match_start, preds = preds_pair
        # サンプル時刻 (絶対秒) を計算する場合、generate_training_dataset の
        # boundaries は match_boundaries_v4 由来。キャッシュ側の最初の t と
        # match_boundaries の start_sec はほぼ同じはずなので、相対 elapsed を
        # 計算するために phase ごとに再構築する。
        # キャッシュには boundaries.tsv の start_sec が反映されないため、
        # 本スクリプトでは キャッシュ最初のサンプル時刻を起点とする。
        # phase 時刻 = match_start + offset
        offset_table = {
            "start_plus_0": 1.0, "start_plus_15": 15.0, "start_plus_30": 30.0,
            "mid_minus_30": None, "mid_minus_15": None, "midpoint": None,
            "mid_plus_15": None, "mid_plus_30": None,
            "end_minus_15": None, "end_minus_5": None,
        }
        # mid/end は試合長に依存するので boundaries (start, end) から計算
        match_end = bdy[1]
        duration = match_end - match_start
        mid = duration / 2.0
        offsets = {
            "start_plus_0": 1.0, "start_plus_15": 15.0, "start_plus_30": 30.0,
            "mid_minus_30": mid - 30.0, "mid_minus_15": mid - 15.0,
            "midpoint": mid,
            "mid_plus_15": mid + 15.0, "mid_plus_30": mid + 30.0,
            "end_minus_15": duration - 15.0, "end_minus_5": duration - 5.0,
        }
        elapsed = offsets.get(phase)
        if elapsed is None:
            skipped += 1
            continue
        in1, in2 = aggregate_incoming(preds, elapsed, args.window)
        # 1P 視点 - 2P 視点。1P が多く受ける = 1P 不利 = 負の pressure
        # ただしこれまでの差分指標は「1P 値 - 2P 値」、incoming は「受ける」=不利方向
        # 既存 DEFAULT_WEIGHTS で incoming_ojama_pressure: -1.0 (受ける=不利)
        # CSV 列も「1P が受けた - 2P が受けた」、Scorer 側で負係数を掛けて反転
        diff = in1 - in2
        pressure = normalize_pressure(diff)
        r["incoming_ojama_pressure"] = f"{pressure:.6f}"
        pressure_values.append(pressure)
        updated += 1

    # 出力
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nupdated rows: {updated} / {len(rows)}, skipped: {skipped}")

    # 統計
    if pressure_values:
        import statistics
        non_zero = [v for v in pressure_values if v != 0.0]
        meta = {
            "input_csv": str(csv_in),
            "output_csv": str(csv_out),
            "score_cache": str(cache_path),
            "window_sec": args.window,
            "norm_divisor": args.norm,
            "min_chain_score": MIN_CHAIN_SCORE,
            "rows_total": len(rows),
            "rows_updated": updated,
            "rows_skipped": skipped,
            "pressure_stats": {
                "n": len(pressure_values),
                "n_nonzero": len(non_zero),
                "mean": statistics.mean(pressure_values),
                "stdev": (
                    statistics.stdev(pressure_values)
                    if len(pressure_values) > 1 else 0.0
                ),
                "min": min(pressure_values),
                "max": max(pressure_values),
            },
        }
        meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"出力: {csv_out}")
        print(f"meta: {meta_out}")
        print(f"pressure stats: mean={meta['pressure_stats']['mean']:+.4f} "
              f"std={meta['pressure_stats']['stdev']:.4f} "
              f"nonzero={meta['pressure_stats']['n_nonzero']}/{len(pressure_values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

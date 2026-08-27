"""診断専用 (読み取り専用・本体コード非改変): OjamaAccountingTracker の
PENDING_ABS_CAP=216 到達バグが npz/CSV 学習データを歪めているかを定量化する。

【背景】
docs/KNOWN_WEAKNESSES.md W25節末尾で「score OCR異常由来」と仮説的に記録された
forecast cap 到達 (src/ojama_accounting.py:715-727) が、現在走行中の148動画
再収集ログに大量出現。本スクリプトは既存の W12 ラベル付きCSV
(data/verify/labeled_win_w12_85_2026-08-16/labeled_win_w12_85.csv) を読み取り、
pegged 行 (ojama_forecast_uncapped>=216) の tsumo (手数近似, W18既知の限界あり)
分布を見て、マージンタイム由来 (正常) か会計異常 (バグ) かを切り分ける。

動画の再走査・重い計算は一切行わない (既存CSVの列読み取りのみ、単スレッド)。

実行: WSL側venvで
  PYTHONPATH=. ./venv/bin/python scripts/_diag_ojama_forecast_cap_2026-08-18.py
出力: logs/_diag_ojama_forecast_cap_2026-08-18.log へ手動リダイレクトして保存。
"""
from __future__ import annotations

import pandas as pd

CSV_PATH = "data/verify/labeled_win_w12_85_2026-08-16/labeled_win_w12_85.csv"


def main() -> None:
    cols = ["video_id", "side", "t_sec", "tsumo", "ojama_forecast_uncapped", "ojama_forecast"]
    df = pd.read_csv(CSV_PATH, usecols=lambda c: c in cols)
    print("rows:", len(df))
    print(df["ojama_forecast_uncapped"].describe())
    print("max:", df["ojama_forecast_uncapped"].max())

    pegged = df[df["ojama_forecast_uncapped"] >= 216].copy()
    print("\n=== pegged (>=216) 集計 ===")
    print("pegged件数:", len(pegged), "/ 全体", len(df),
          f"({100 * len(pegged) / len(df):.3f}%)")
    print(">216 (クランプ漏れがあれば非ゼロのはず):",
          (df["ojama_forecast_uncapped"] > 216).sum())
    print("対象video数:", pegged["video_id"].nunique(), "/ 全体video数:",
          df["video_id"].nunique())

    bins = [0, 10, 20, 30, 40, 60, 80, 100, 150, 200, 300]
    pegged["tsumo_bin"] = pd.cut(pegged["tsumo"], bins=bins)
    print("\n=== pegged行のtsumo帯別ヒストグラム (マージンタイム閾値=概ねtsumo30-40以降で妥当) ===")
    print(pegged["tsumo_bin"].value_counts().sort_index())

    early = pegged[pegged["tsumo"] < 10]
    print("\n=== tsumo<10 (超早期、マージンタイム適用外のはず=会計異常の疑い) ===")
    print("件数:", len(early), f"({100 * len(early) / len(pegged):.4f}% of pegged, "
          f"{100 * len(early) / len(df):.6f}% of all rows)")
    print(early[["video_id", "side", "t_sec", "tsumo"]])
    print("\n対象video:", sorted(early["video_id"].unique().tolist()))


if __name__ == "__main__":
    main()

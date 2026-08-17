"""連鎖演出時間 実測テーブルを恒久JSONに変換する
(docs/DEMO_REVIEW_2026-08-13.md #12 案B、2026-08-14)。

入力: data/verify/chain_anim_duration_2026-08-14/table_by_chain_count.csv
      (旧23動画+新10動画Phase L実測、ピクセルdiffベース盤面settle実測。
      詳細は同ディレクトリの README.txt 参照)
出力: data/verify/chain_anim_duration_median_table_2026-08-14.json
      (data/verify/chain_length_conditional_2026-08-13.json と同パターン、
      中央値+P25/P75+n を保持する監査用の恒久レコード)

N=13以上は実測サンプル数 n が極小 (6/3/3件) のため、生の中央値をそのまま
較正値に使わず、N=1..12 (n=11〜105件、十分な母数) の実測に対する線形フィット
結果 (a=3.3, b=0.89、診断報告 案B 指定値) で外挿する。

⚠️ 本スクリプトが生成する JSON は監査・記録用途 (「何を実測し、どの値を
採用したか」の恒久ログ)。実際の較正計算は src/indicators_v2.py の
CHAIN_ANIM_DURATION_MEDIAN_SEC_TABLE_2026_08_14 /
CHAIN_ANIM_DURATION_EXTRAPOLATION_{A,B}_SEC_2026_08_14 (ハードコード定数、
indicators_v2.py は非stateless実装を避けるためファイルI/Oを行わない設計)
が単一情報源であり、本 JSON はそれと同じ値を人間が確認できる形で保持する
だけの副産物 (値の変更時は両方を同時に更新すること)。

使い方:
    python -m scripts._build_chain_anim_duration_table_2026-08-14
"""
from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path

SOURCE_CSV = Path("data/verify/chain_anim_duration_2026-08-14/table_by_chain_count.csv")
OUT_PATH = Path("data/verify/chain_anim_duration_median_table_2026-08-14.json")

# src/indicators_v2.py の
# CHAIN_ANIM_DURATION_EXTRAPOLATION_{A_SEC,B_SEC_PER_CHAIN,MIN_CHAIN_COUNT}_2026_08_14
# と同一の値 (意図的に複製: 本スクリプトは軽量な使い捨てツールであり、
# src.indicators_v2 の重い依存を持ち込まないため)。
EXTRAPOLATION_A: float = 3.3
EXTRAPOLATION_B: float = 0.89
EXTRAPOLATION_MIN_N: int = 13


def build_table(rows: "list[dict]") -> dict:
    """CSV の行リストから JSON 保存用の dict を構築する。"""
    table: dict[str, dict] = {}
    for row in rows:
        n_chain = int(row["chain_count"])
        extrapolated = n_chain >= EXTRAPOLATION_MIN_N
        raw_median = float(row["median_sec"])
        median = EXTRAPOLATION_A + EXTRAPOLATION_B * n_chain if extrapolated else raw_median
        table[str(n_chain)] = {
            "n": int(row["n"]),
            "median_sec": median,
            "p25_sec": float(row["p25_sec"]),
            "p75_sec": float(row["p75_sec"]),
            "extrapolated": extrapolated,
            "raw_median_sec_low_n": raw_median if extrapolated else None,
        }
    return {
        "table_by_chain_count": table,
        "extrapolation": {
            "formula": "a + b * chain_count",
            "a": EXTRAPOLATION_A,
            "b": EXTRAPOLATION_B,
            "applies_to_chain_count_gte": EXTRAPOLATION_MIN_N,
            "reason": (
                "N>=13 は実測サンプル数が6/3/3件と極小のため、生の中央値を"
                "採用せず線形フィットで外挿する (診断報告 案B 指定値)。"
            ),
        },
        "source_csv": str(SOURCE_CSV),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "methodology": (
            "data/verify/chain_anim_duration_2026-08-14/README.txt "
            "(旧23動画[2026-07-24計測]+新10動画Phase L[2026-08-14計測]、"
            "ピクセルdiffベース盤面settle実測) の連鎖数別中央値をそのまま"
            "較正値として採用。N=13以上は線形フィット a+b*N "
            "(docs/DEMO_REVIEW_2026-08-13.md #12 案B 指定値) で外挿する。"
        ),
    }


def main() -> None:
    with SOURCE_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    table = build_table(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    max_n = max(int(r["chain_count"]) for r in rows)
    print(f"[build_chain_anim_duration_table] {OUT_PATH} に保存 (N=1..{max_n})")


if __name__ == "__main__":
    main()

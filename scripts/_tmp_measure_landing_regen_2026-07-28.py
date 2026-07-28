"""#24 Step0 前提ゲート再検証 (2026-07-28): 認識強化後 npz での着弾遅延・被覆率再測定。

scripts/measure_ojama_landing_delay.py / scripts/measure_exchange_dynamics.py は
「別コーダが並行作業中のため書き換えない」方針(両ファイル docstring 明記) のため、
本スクリプトは import のみで新しい npz ディレクトリ
(data/indicators_v2/boards_lean_fixed_regen_2026-07-28/) に対して同じロジックを
再実行する専用の一時スクリプトとして新規作成する (既存ファイルは一切変更しない)。

2026-07-22 測定 (旧 boards_lean_fixed、#51 反映高速化3修正 適用前) との
比較可能性のため、発火検出ロジック (_process_video) ・着弾検出ロジック
(_measure_landing_for_video) は完全に同一のものを import して使う。

被覆率 (= opp_available 率、OPP_GAP_THRESHOLD_SEC=2.0 のゲート判定対象) を
動画別に集計し、最小/中央/最大を報告する (userタスク必須項目)。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.chain import ChainSimulator  # noqa: E402
from scripts.measure_exchange_dynamics import TIER_MAP, _process_video  # noqa: E402
from scripts.measure_ojama_landing_delay import (  # noqa: E402
    OPP_GAP_THRESHOLD_SEC, _measure_landing_for_video, _print_report,
)

# 再収集先 (2026-07-28、認識強化後の RecognitionPipeline.load_default 既定値を
# 継承した collect_boards_lean.py で c 系動画を再収集した npz 群)
NPZ_DIR_REGEN: Path = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_fixed_regen_2026-07-28"
OUTPUT_CSV_REGEN: Path = PROJ_ROOT / "data" / "indicators_v2" / "exchange_landing_delay_regen_2026-07-28.csv"

# 被覆率ゲート判定の閾値 (userタスク指定)
COVERAGE_GATE_PCT: float = 20.0


def _report_coverage_by_video(df: pd.DataFrame) -> pd.DataFrame:
    """動画別の被覆率 (opp_available 率) を集計する。"""
    g = df.groupby("video_stem")["opp_available"].agg(["mean", "count"])
    g = g.rename(columns={"mean": "coverage_rate", "count": "n_fire"})
    g["coverage_pct"] = g["coverage_rate"] * 100.0
    return g.sort_values("coverage_pct")


def main() -> None:
    warnings.filterwarnings("ignore")
    npz_paths = [NPZ_DIR_REGEN / f"{stem}.npz" for stem in TIER_MAP]
    present = [p for p in npz_paths if p.exists()]
    missing = [p.stem for p in npz_paths if not p.exists()]
    print(f"[INFO] 対象23動画中 存在={len(present)} / 欠落={len(missing)}")
    if missing:
        print(f"[INFO] 欠落 (未完走またはスキップ): {missing}")

    if not present:
        print("[ERROR] regen npz が1件もありません。", file=sys.stderr)
        sys.exit(1)

    sim = ChainSimulator()
    all_rows: list[dict] = []
    seq_id = 0
    for npz_path in sorted(present, key=lambda p: p.stem):
        _, defrag_events, seq_id = _process_video(npz_path, sim, seq_id)
        rows = _measure_landing_for_video(npz_path, defrag_events)
        all_rows.extend(rows)
        n_avail = sum(1 for r in rows if r["opp_available"])
        pct = (n_avail / len(rows) * 100.0) if rows else float("nan")
        print(
            f"  {npz_path.stem}: 発火{len(defrag_events)}件 -> "
            f"着弾測定{len(rows)}件 (被覆率 {pct:.1f}%)",
        )

    if not all_rows:
        print("[ERROR] 発火イベントが0件でした。", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    OUTPUT_CSV_REGEN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV_REGEN, index=False)
    print(f"\n[DONE] {len(df)} 行を {OUTPUT_CSV_REGEN} に保存しました")

    cov = _report_coverage_by_video(df)
    print(f"\n[被覆率 (opp_available率, OPP_GAP_THRESHOLD_SEC={OPP_GAP_THRESHOLD_SEC}s) 動画別]")
    print(cov.to_string())
    overall = float(df["opp_available"].mean() * 100.0)
    print(f"\n[被覆率 全体] {overall:.1f}%")
    print(
        f"[被覆率 動画別分布] 最小={cov['coverage_pct'].min():.1f}% "
        f"中央={cov['coverage_pct'].median():.1f}% "
        f"最大={cov['coverage_pct'].max():.1f}%",
    )
    verdict = "前進OK" if overall > COVERAGE_GATE_PCT else "閾値緩和 or 保留をuser判断へ"
    print(f"\n[ゲート判定] 全体被覆率 {overall:.1f}% (閾値{COVERAGE_GATE_PCT}%) -> {verdict}")

    _print_report(df)


if __name__ == "__main__":
    main()

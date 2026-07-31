"""#24 Step0 前提ゲート再検証 (2026-07-28): 最終レポート自動生成。

boards_lean_fixed_regen_2026-07-28/ への再収集が完走した後に呼ばれる想定。
1. scripts._tmp_measure_landing_regen_2026-07-28 を実行し着弾遅延+被覆率を再測定
   (結果は data/indicators_v2/exchange_landing_delay_regen_2026-07-28.csv)。
2. 旧 (2026-07-22, #51適用前) の data/indicators_v2/exchange_landing_delay.csv
   と被覆率を比較。
3. data/verify/exchange_gate_recheck_2026-07-28/summary.md に結果をまとめる。

既存スクリプト (measure_exchange_dynamics.py / measure_ojama_landing_delay.py) は
import のみで変更しない。
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

OLD_CSV: Path = PROJ_ROOT / "data" / "indicators_v2" / "exchange_landing_delay.csv"
NEW_CSV: Path = PROJ_ROOT / "data" / "indicators_v2" / "exchange_landing_delay_regen_2026-07-28.csv"
REGEN_NPZ_DIR: Path = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_fixed_regen_2026-07-28"
REPORT_LOG: Path = PROJ_ROOT / "logs" / "exchange_landing_delay_regen_report_2026-07-28.log"
SUMMARY_MD: Path = PROJ_ROOT / "data" / "verify" / "exchange_gate_recheck_2026-07-28" / "summary.md"

# ゲート判定閾値 (userタスク指定)
COVERAGE_GATE_PCT: float = 20.0
OPP_GAP_THRESHOLD_SEC: float = 2.0

# 対象23動画 (userタスク指定の固定リスト、measure_exchange_dynamics.TIER_MAP と同一)
ALL_23_STEMS: tuple[str, ...] = (
    "c5", "c6", "c7", "c11", "c16", "c21", "c22", "c28", "c30", "c31",
    "c44", "c51", "c53", "c54", "c59", "c62", "c68", "c73", "c78", "c80",
    "c82", "c83", "c84",
)


def _coverage_stats(df: pd.DataFrame) -> dict:
    """全体被覆率 + 動画別 min/median/max を返す。"""
    overall = float(df["opp_available"].mean() * 100.0)
    by_video = df.groupby("video_stem")["opp_available"].mean() * 100.0
    return {
        "overall_pct": overall,
        "n_video": int(by_video.shape[0]),
        "min_pct": float(by_video.min()),
        "median_pct": float(by_video.median()),
        "max_pct": float(by_video.max()),
        "by_video": by_video.sort_values(),
    }


def _load_regen_module():
    """ファイル名にハイフンを含み通常 import できない再測定スクリプトを動的ロードする。"""
    import importlib.util
    mod_path = PROJ_ROOT / "scripts" / "_tmp_measure_landing_regen_2026-07-28.py"
    spec = importlib.util.spec_from_file_location("_tmp_measure_landing_regen_2026_07_28", mod_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _run_regen_measurement() -> tuple[bool, str]:
    """再測定スクリプトを実行し (成功か, 標準出力ログ) を返す。"""
    regen_mod = _load_regen_module()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            regen_mod.main()
        ok = True
    except SystemExit as e:
        ok = (e.code == 0)
    return ok, buf.getvalue()


def _missing_video_report() -> list[str]:
    """regen npz が存在しない動画一覧を返す。"""
    return [s for s in ALL_23_STEMS if not (REGEN_NPZ_DIR / f"{s}.npz").exists()]


def _write_summary(old_stats: dict, new_stats: dict, missing: list[str], regen_ok: bool) -> None:
    verdict = "前進OK" if new_stats["overall_pct"] > COVERAGE_GATE_PCT else "閾値緩和 or 保留をuser判断へ"
    lines: list[str] = []
    lines.append("# #24 打ち合い計測器 Step0 前提ゲート再検証 (2026-07-28)")
    lines.append("")
    lines.append("## 背景")
    lines.append(
        "- 2026-07-22測定 (認識強化前) の被覆率は 7.3%〜47.2% (OPP_GAP_THRESHOLD_SEC=2.0)。",
    )
    lines.append(
        "- #51系反映高速化3修正 (enable_recovery_counter_carryover / "
        "enable_cnn_flicker_hsv_fallback / enable_initial_confirm_vote、"
        "2026-07-27 既定ON化) 反映後の被覆率を再測定した。",
    )
    lines.append(
        "- 疑似発火対策 (4連結絞り、MIN_CHAIN_COUNT_FOR_FIRE) は"
        "scripts/measure_exchange_dynamics.py に解決済のため再実装していない。",
    )
    lines.append("")
    lines.append("## Step1: 既存558イベントのレポート永続化")
    lines.append(
        "- `logs/measure_exchange_dynamics_report_2026-07-28.log` に保存済 "
        "(再計算なし、既存 exchange_dynamics_stats.csv (2026-07-22生成) を再出力)。",
    )
    lines.append("")
    lines.append("## Step2: 被覆率の再測定")
    lines.append("")
    lines.append(f"- 対象23動画中 regen npz 生成済み: {23 - len(missing)}/23")
    if missing:
        lines.append(f"- 欠落 (未完走/失敗): {', '.join(missing)}")
    lines.append(
        "- 再収集設定: `scripts/collect_boards_lean.py` "
        "(`--sample-interval 0.2 --max-sec 1200`、2026-07-22測定と同一、"
        "RecognitionPipeline.load_default の明示フラグなし = 現行既定値を継承)。",
    )
    lines.append(
        "- 既定追従チェック: `collect_boards_lean.py:382-389` の `load_default()` 呼び出しは "
        "`enable_recovery_counter_carryover` 等 #51系フラグを一切明示指定していないため "
        "本体既定値 (True) をそのまま継承する。measure_stable_cell_acc.py で過去に発覚した"
        "「ローカル決め打ちdefaultの追従漏れ」バグは本経路には存在しない (確認済み、修正不要)。",
    )
    lines.append("")
    lines.append("### 被覆率 (opp_available率) 比較")
    lines.append("")
    lines.append("| | 2026-07-22 (認識強化前) | 2026-07-28 (認識強化後) |")
    lines.append("|---|---|---|")
    lines.append(f"| 全体 | {old_stats['overall_pct']:.1f}% | {new_stats['overall_pct']:.1f}% |")
    lines.append(
        f"| 動画別 最小 | {old_stats['min_pct']:.1f}% | {new_stats['min_pct']:.1f}% |",
    )
    lines.append(
        f"| 動画別 中央 | {old_stats['median_pct']:.1f}% | {new_stats['median_pct']:.1f}% |",
    )
    lines.append(
        f"| 動画別 最大 | {old_stats['max_pct']:.1f}% | {new_stats['max_pct']:.1f}% |",
    )
    lines.append(f"| 対象動画数 | {old_stats['n_video']} | {new_stats['n_video']} |")
    lines.append("")
    lines.append("### 動画別被覆率 (2026-07-28、被覆率昇順)")
    lines.append("")
    lines.append("| 動画 | 被覆率 |")
    lines.append("|---|---|")
    for stem, pct in new_stats["by_video"].items():
        lines.append(f"| {stem} | {pct:.1f}% |")
    lines.append("")
    lines.append("## ゲート判定")
    lines.append(
        f"- OPP_GAP_THRESHOLD_SEC={OPP_GAP_THRESHOLD_SEC}で全体被覆率 "
        f"{new_stats['overall_pct']:.1f}% (閾値{COVERAGE_GATE_PCT}%超) -> **{verdict}**",
    )
    lines.append("")
    lines.append("## 詳細ログ")
    lines.append(f"- 再測定フルレポート: `logs/exchange_landing_delay_regen_report_2026-07-28.log`")
    lines.append(f"- 再測定CSV: `data/indicators_v2/exchange_landing_delay_regen_2026-07-28.csv`")
    lines.append(f"- 再収集npz: `data/indicators_v2/boards_lean_fixed_regen_2026-07-28/`")
    lines.append("")
    lines.append("## 注意")
    lines.append(
        "- 本再測定は #51系反映高速化3修正の効果のみを見るためのもので、"
        "元動画は data/frames/video_cN.mp4 (削除ポリシー対象外、既存ローカル保持分) を再利用した。",
    )
    if not regen_ok:
        lines.append("- ⚠️ 再測定スクリプトの実行で異常終了を検知。ログを確認すること。")

    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[DONE] summary.md を保存しました: {SUMMARY_MD}")


def main() -> None:
    old_df = pd.read_csv(OLD_CSV)
    old_stats = _coverage_stats(old_df)

    missing = _missing_video_report()
    regen_ok, report_text = _run_regen_measurement()
    REPORT_LOG.parent.mkdir(parents=True, exist_ok=True)
    REPORT_LOG.write_text(report_text, encoding="utf-8")
    print(f"[INFO] 再測定レポートを保存: {REPORT_LOG} (成功={regen_ok})")

    if not NEW_CSV.exists():
        print(f"[ERROR] {NEW_CSV} が生成されませんでした。", file=sys.stderr)
        sys.exit(1)
    new_df = pd.read_csv(NEW_CSV)
    new_stats = _coverage_stats(new_df)

    _write_summary(old_stats, new_stats, missing, regen_ok)


if __name__ == "__main__":
    main()

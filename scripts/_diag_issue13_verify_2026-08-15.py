"""指摘13 (docs/DEMO_REVIEW_2026-08-13.md #13) の検証用スクリプト。

「84%で固定でなく、2Pのみ連鎖中なら受け側の状況は0.5秒ごとに変わるので動くのが
普通では」への対処 (enable_resolved_live_defender=True) を、
scripts/_selfverify_final2_confirm_2026-08-14.py (指摘9/10/12検証) と同一の
動画・同一フラグ構成 + 新フラグ追加のみで、以下を検証する:

  - 指摘12窓 (source 234.87-243s、2Pの本線発火・1Pは片側先行終了):
    disp_adv (=画面表示値) の 0.5秒刻みタイムラインを OFF/ON 両変種で比較。
    ONでは84%起点から1Pの盤面変化に応じて連続的に動き、t≈244の撃ち返しで
    反転する挙動になっているかを確認する。
  - 指摘10窓 (source 194.53-200s、両者本物の連鎖中=ケース1):
    ON でも完全凍結が維持され (#13はケース2=片側のみ連鎖中限定のはず)、
    真に応手不能な高止まりに変化が無いことを確認する。
  - 全域 (162-310s) の極端値率・隣接差分 (乱高下) 分布が悪化しないこと。

本体コード (scripts/visualize_advantage_overlay.py) は変更しない (計装は
monkeypatch/generate() 呼び出しのみ、フラグは既定OFFのまま追加した新機能を
明示的に有効化して比較する)。
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.visualize_advantage_overlay as ov  # noqa: E402
from scripts.visualize_advantage_overlay import load_timeline_dump  # noqa: E402

START_SEC = 162.0
END_SEC = 310.0
VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT = Path("data/verify/demo_fixed_2026-08-13/_unused_diag_issue13.mp4")

ISSUE12_LO, ISSUE12_HI = 234.87, 243.0
ISSUE10_LO, ISSUE10_HI = 194.53, 200.0


def _run(live_defender: bool, dump_path: Path) -> list[tuple[float, float]]:
    """指定変種 (enable_resolved_live_defender) で通し実行し history を返す。"""
    history: list[tuple[float, float]] = []
    ov.generate(
        VIDEO, OUT, max_sec=0.0, sample_interval=0.0,
        start_sec=START_SEC, end_sec=END_SEC,
        show_recognition=True,
        enable_early_fire_reaction=True, enable_per_side_settled=True,
        disable_score_lead_bias=True, disable_pressure=True,
        enable_counter_remaining_time=True, enable_counter_defender_only=True,
        stable_majority_window=True,
        enable_ojama_fall_placement_override=True,
        enable_ojama_fall_entry_hardening=True,
        enable_ojama_fall_scoped_exit=True,
        enable_resolved_exchange_eval=True,
        enable_resolved_decisive_amplify=True,
        enable_resolved_live_defender=live_defender,
        enable_pseudo_chain_score_fill=True,
        layout="panel",
        render=False,
        debug_history_out=history,
        dump_timeline_path=dump_path,
    )
    return history


def _bucketed(history: list[tuple[float, float]], lo: float, hi: float,
              bucket_sec: float = 0.5) -> list[tuple[float, float]]:
    """[lo, hi] 窓を bucket_sec 刻みで間引く (連続タイムライン表示用)。"""
    out: list[tuple[float, float]] = []
    last_bucket = None
    for t_sec, disp_adv in history:
        if not (lo <= t_sec <= hi):
            continue
        bucket = round(t_sec / bucket_sec)
        if bucket == last_bucket:
            continue
        last_bucket = bucket
        out.append((t_sec, disp_adv))
    return out


def _print_timeline(label: str, rows: list[tuple[float, float]]) -> None:
    print(f"\n---- {label} ----")
    for t_sec, disp_adv in rows:
        p1 = 0.5 + disp_adv / 200.0
        print(f"  t={t_sec:7.2f} disp_adv={disp_adv:+7.2f} "
              f"p1(1P)={p1 * 100:5.1f}% p2(2P)={100 - p1 * 100:5.1f}%")


def _fullscan_stats(dump_path: Path) -> dict:
    video_id, rows = load_timeline_dump(dump_path)
    n_total = len(rows)
    n_extreme_p1 = sum(1 for r in rows if r.p1 >= 0.97)
    n_extreme_p2 = sum(1 for r in rows if r.p1 <= 0.03)
    diffs = [abs(rows[i].p1 - rows[i - 1].p1) for i in range(1, n_total)]
    return {
        "video_id": video_id, "n_total": n_total,
        "extreme_p1_rate": n_extreme_p1 / n_total if n_total else 0.0,
        "extreme_p2_rate": n_extreme_p2 / n_total if n_total else 0.0,
        "diff_mean": statistics.mean(diffs) if diffs else 0.0,
        "diff_p95": (
            statistics.quantiles(diffs, n=20)[18] if len(diffs) >= 20 else max(diffs, default=0.0)
        ),
        "diff_max": max(diffs, default=0.0),
    }


def main() -> int:
    dump_off = Path("data/verify/demo_fixed_2026-08-13/diag_issue13_off_2026-08-15.npz")
    dump_on = Path("data/verify/demo_fixed_2026-08-13/diag_issue13_on_2026-08-15.npz")

    print("===== OFF (enable_resolved_live_defender=False, 既定=従来#12挙動) 実行中 =====")
    hist_off = _run(live_defender=False, dump_path=dump_off)
    print("===== ON  (enable_resolved_live_defender=True, 指摘13新実装) 実行中 =====")
    hist_on = _run(live_defender=True, dump_path=dump_on)

    print("\n===== 指摘12窓 (source 234.87-243s): disp_adv 0.5秒刻み比較 =====")
    _print_timeline("OFF (従来: 片側終了後も完全凍結)", _bucketed(hist_off, ISSUE12_LO, ISSUE12_HI))
    _print_timeline("ON  (指摘13: 受け側ライブ再評価)", _bucketed(hist_on, ISSUE12_LO, ISSUE12_HI))

    print("\n===== 指摘10窓 (source 194.53-200s、両者本物の連鎖中=ケース1): "
          "disp_adv 0.5秒刻み比較 (ON でも凍結維持のはず) =====")
    rows_off10 = _bucketed(hist_off, ISSUE10_LO, ISSUE10_HI)
    rows_on10 = _bucketed(hist_on, ISSUE10_LO, ISSUE10_HI)
    _print_timeline("OFF", rows_off10)
    _print_timeline("ON ", rows_on10)
    vals_off10 = [v for _, v in rows_off10]
    vals_on10 = [v for _, v in rows_on10]
    if vals_off10 and vals_on10:
        print(f"\n  OFF stdev={statistics.pstdev(vals_off10):.3f} "
              f"range=[{min(vals_off10):+.2f},{max(vals_off10):+.2f}]")
        print(f"  ON  stdev={statistics.pstdev(vals_on10):.3f} "
              f"range=[{min(vals_on10):+.2f},{max(vals_on10):+.2f}]")

    print("\n===== 全域 (source 162-310s) 極端値率・隣接差分 (乱高下) 比較 =====")
    stats_off = _fullscan_stats(dump_off)
    stats_on = _fullscan_stats(dump_on)
    for label, s in (("OFF", stats_off), ("ON ", stats_on)):
        print(f"  {label}: video_id={s['video_id']} n={s['n_total']} "
              f"極端値p1>=97%={s['extreme_p1_rate'] * 100:.1f}% "
              f"極端値p1<=3%={s['extreme_p2_rate'] * 100:.1f}% "
              f"|Δp1|平均={s['diff_mean']:.4f} p95={s['diff_p95']:.4f} max={s['diff_max']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

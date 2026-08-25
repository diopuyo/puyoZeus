"""連鎖「演出時間」(発火検知〜連鎖終了) を連鎖数別に層別実測するテーブル化スクリプト
(2026-08-14、デバッガ計装タスク)。

## 目的
応手判定の時間予算 `CHAIN_ANIM_PER_STEP_SEC=0.4秒/連鎖` (23動画418イベント平均)
が大連鎖に系統的過小である疑い (指摘12: 6連鎖=実測8.1秒 vs 予算換算2.4秒) を、
連鎖数(N)別の分布 (中央値/P25/P75/n) で定量化する。

## 既存資産の使い方 (read-only)
- 主資産: data/verify/recognition_diag_chain_anim_duration_multi/
  (2026-07-24計測、23動画・418イベント、chain_count 1〜7個別 + 8以上まとめ)。
  この CSV をそのまま読み込み、層別テーブルのベースにする (再計測しない、
  対象動画のmp4は既にストレージ節約のため削除済で再現不可)。
- 補強資産: 148動画Phase L再生成 npz
  (data/indicators_v2/boards_lean_phase_l_2026-08-11/) のうち、生mp4が
  まだ data/frames/ に残っている10動画 (c109,c13,c130-c137、Phase L
  wave2の未削除分) を対象に、同一の測定ロジック
  (scripts/_diag_chain_anim_duration_multi.py の
  `_measure_visual_duration`/`_process_video` を import して再利用、
  本体コード変更なし) で追加実測する。旧23動画とは別の動画・別の認識
  パイプライン世代のため、独立クロスチェックとして扱う。

## 除外基準 (既存スクリプトと同一、変更しない)
- no_motion_detected: 発火検知後 MOTION_START_MAX_WAIT_SEC=1.5秒以内に
  盤面ROIの動きが一度も観測できない → 除外 (誤発火/認識ズレの疑い)。
- truncated_no_settle: 探索窓上限 (5+3.5*chain_count、上限35秒) 内に
  SETTLE_MIN_SEC=1.2秒連続静止が観測できない → 除外 (試合終了跨ぎ・
  おじゃま落下混入等で「連鎖の終わり」が窓内に収まらないケース)。
- 上記いずれでもない「ok」のみを分布集計に使う。

## サンプリング (補強資産のみ、front-run 時間を抑えるため)
- chain_count>=9 (レア): 全件採用。
- chain_count 1〜8: (video_stem, chain_count) ごとに最大 SAMPLE_CAP 件を
  決定的な等間隔サンプリング (乱数不使用、再現性のため)。

出力: data/verify/chain_anim_duration_2026-08-14/
    - table_by_chain_count.csv : N別 (中央値/P25/P75/n/出典内訳)
    - new_events_raw.csv       : 補強資産の生イベント一覧
    - distribution.png         : 分布図 (箱ひげ、旧+新を重ねる)
    - README.txt               : 検算結果 (6連鎖8.1秒・8連鎖14.5秒の収まり具合)

Usage (前面実行のみ、バックグラウンド分離禁止):
    PYTHONPATH=. ./venv/bin/python scripts/_diag_chain_anim_duration_by_n_2026-08-14.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.chain import ChainSimulator  # noqa: E402
from scripts.measure_exchange_dynamics import _process_video  # noqa: E402
from scripts._diag_chain_anim_duration_multi import (  # noqa: E402
    _EventMeasurement, _measure_visual_duration,
)
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402

import cv2  # noqa: E402

OLD_DIR = PROJ_ROOT / "data" / "verify" / "recognition_diag_chain_anim_duration_multi"
OLD_EVENTS_CSV = OLD_DIR / "events_raw.csv"

NEW_NPZ_DIR = PROJ_ROOT / "data" / "indicators_v2" / "boards_lean_phase_l_2026-08-11"
NEW_VIDEO_DIR = PROJ_ROOT / "data" / "frames"
# Phase L wave2 のうち生mp4がまだ data/frames/ に残っている動画
# (2026-08-14 時点で確認。他は収集後削除済でこのスクリプトでは測れない)。
NEW_VIDEO_IDS: tuple[str, ...] = (
    "c109", "c13", "c130", "c131", "c132", "c133", "c134", "c135", "c136", "c137",
)

OUT_DIR = PROJ_ROOT / "data" / "verify" / "chain_anim_duration_2026-08-14"

RARE_CHAIN_COUNT_THRESHOLD = 9  # これ以上は全件採用
SAMPLE_CAP = 3  # 1〜8連鎖は (video, chain_count) ごとに最大この件数


def _collect_new_events() -> list:
    sim = ChainSimulator()
    seq_id = 0
    all_events = []
    for stem in NEW_VIDEO_IDS:
        p = NEW_NPZ_DIR / f"{stem}.npz"
        if not p.exists():
            print(f"[WARN] npz不在: {stem}")
            continue
        _, defrag, seq_id = _process_video(p, sim, seq_id)
        events = [e for e in defrag if e.chain_count >= 1]
        all_events.append((stem, events))
        print(f"[collect] {stem}: chain_count>=1 が {len(events)} 件")
    return all_events


def _sample(by_video_events: list) -> list:
    """front-run 時間短縮のための決定的サンプリング。"""
    grouped: dict[tuple[str, int], list] = {}
    for stem, events in by_video_events:
        for e in events:
            grouped.setdefault((stem, e.chain_count), []).append(e)

    sampled = []
    for (stem, cc), group in grouped.items():
        if cc >= RARE_CHAIN_COUNT_THRESHOLD or len(group) <= SAMPLE_CAP:
            sampled.extend(group)
        else:
            idx = np.linspace(0, len(group) - 1, num=SAMPLE_CAP).astype(int)
            sampled.extend([group[i] for i in sorted(set(idx))])
    return sampled


def _measure_new(selected: list) -> list[_EventMeasurement]:
    by_video: dict[str, list] = {}
    for e in selected:
        by_video.setdefault(e.video_stem, []).append(e)

    results: list[_EventMeasurement] = []
    for stem, evs in sorted(by_video.items()):
        video_path = NEW_VIDEO_DIR / f"video_{stem}.mp4"
        if not video_path.exists():
            print(f"[WARN] mp4不在: {stem}")
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] mp4 open失敗: {stem}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        n_done = 0
        for ev in evs:
            region = DEFAULT_P1_REGION if ev.fire_side == "1P" else DEFAULT_P2_REGION
            dur, n_peaks, status = _measure_visual_duration(
                cap, fps, region, ev.t_chain_start, ev.chain_count,
            )
            results.append(_EventMeasurement(
                video_stem=stem, tier=ev.tier, fire_side=ev.fire_side,
                chain_count=ev.chain_count, t_chain_start=ev.t_chain_start,
                t_fire=ev.t_fire, visual_duration_sec=dur, n_motion_peaks=n_peaks,
                status=status,
            ))
            n_done += 1
        cap.release()
        print(f"[measure] {stem}: {n_done}件処理完了 (累計{len(results)}件)")
    return results


def _write_new_events_csv(results: list[_EventMeasurement], out_path: Path) -> None:
    cols = [
        "video_stem", "tier", "fire_side", "chain_count", "t_chain_start", "t_fire",
        "visual_duration_sec", "n_motion_peaks", "status",
    ]
    lines = [",".join(cols)]
    for r in results:
        vdur = r.visual_duration_sec if r.visual_duration_sec is not None else ""
        lines.append(",".join(str(v) for v in [
            r.video_stem, r.tier, r.fire_side, r.chain_count,
            f"{r.t_chain_start:.3f}", f"{r.t_fire:.3f}", vdur,
            r.n_motion_peaks, r.status,
        ]))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _load_old_events() -> list[dict]:
    """既存23動画資産 (events_raw.csv) を読み込む (status='ok' のみ)。"""
    rows = []
    if not OLD_EVENTS_CSV.exists():
        print(f"[WARN] 旧資産不在: {OLD_EVENTS_CSV}")
        return rows
    with open(OLD_EVENTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "ok" or row["visual_duration_sec"] == "":
                continue
            rows.append({
                "source": "old_23videos_2026-07-24",
                "video_stem": row["video_stem"],
                "chain_count": int(row["chain_count"]),
                "visual_duration_sec": float(row["visual_duration_sec"]),
            })
    return rows


def _percentile_table(all_rows: list[dict]) -> list[dict]:
    """chain_count 別 (中央値/P25/P75/n、出典内訳込み) テーブルを作る。"""
    by_n: dict[int, list[dict]] = {}
    for r in all_rows:
        by_n.setdefault(r["chain_count"], []).append(r)

    out = []
    for n in sorted(by_n):
        group = by_n[n]
        vals = np.array([g["visual_duration_sec"] for g in group])
        n_old = sum(1 for g in group if g["source"].startswith("old"))
        n_new = sum(1 for g in group if g["source"].startswith("new"))
        out.append({
            "chain_count": n,
            "n": len(vals),
            "n_old_23videos": n_old,
            "n_new_10videos_phase_l": n_new,
            "median_sec": float(np.median(vals)),
            "p25_sec": float(np.percentile(vals, 25)),
            "p75_sec": float(np.percentile(vals, 75)),
            "mean_sec": float(np.mean(vals)),
            "std_sec": float(np.std(vals)),
            "min_sec": float(np.min(vals)),
            "max_sec": float(np.max(vals)),
            "current_formula_0.4persec": 0.4 * n,
        })
    return out


def _write_table_csv(table: list[dict], out_path: Path) -> None:
    if not table:
        return
    cols = list(table[0].keys())
    lines = [",".join(cols)]
    for row in table:
        lines.append(",".join(str(row[c]) for c in cols))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_viz(all_rows: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    old_rows = [r for r in all_rows if r["source"].startswith("old")]
    new_rows = [r for r in all_rows if r["source"].startswith("new")]

    fig, ax = plt.subplots(figsize=(11, 7))
    ns = sorted(set(r["chain_count"] for r in all_rows))
    old_data = [[r["visual_duration_sec"] for r in old_rows if r["chain_count"] == n] for n in ns]
    new_data = [[r["visual_duration_sec"] for r in new_rows if r["chain_count"] == n] for n in ns]

    pos_old = [n - 0.15 for n in ns]
    pos_new = [n + 0.15 for n in ns]
    bp1 = ax.boxplot(old_data, positions=pos_old, widths=0.25, showfliers=True,
                      patch_artist=True, boxprops=dict(facecolor="lightblue"))
    bp2 = ax.boxplot(new_data, positions=pos_new, widths=0.25, showfliers=True,
                      patch_artist=True, boxprops=dict(facecolor="lightsalmon"))
    ax.set_xticks(ns)
    xs = np.linspace(min(ns), max(ns), 50)
    ax.plot(xs, 0.4 * xs, color="green", linestyle=":", label="現行式 CHAIN_ANIM_PER_STEP_SEC=0.4*n")
    ax.set_xlabel("連鎖数 (最終確定N)")
    ax.set_ylabel("演出時間実測 (秒、発火検知〜盤面settle)")
    ax.set_title("連鎖数別 演出時間実測 (青=旧23動画418件/2026-07-24, 橙=新10動画/Phase L 2026-08-14)")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["旧23動画", "新10動画(Phase L)"], loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[STEP1] 旧23動画資産を読み込み中...")
    old_rows = _load_old_events()
    print(f"[STEP1] 完了: {len(old_rows)} 件 (status=ok のみ)")

    print("[STEP2] 新10動画 (Phase L, mp4現存分) からFireEvent収集中...")
    by_video_events = _collect_new_events()
    total_new = sum(len(evs) for _, evs in by_video_events)
    print(f"[STEP2] 完了: chain_count>=1 総 {total_new} 件")

    selected = _sample(by_video_events)
    print(f"[STEP3] サンプリング後 {len(selected)} 件を実測 (前面実行、完了まで待機)")

    new_results = _measure_new(selected)
    _write_new_events_csv(new_results, OUT_DIR / "new_events_raw.csv")

    new_rows = [
        {
            "source": "new_10videos_phase_l_2026-08-14",
            "video_stem": r.video_stem,
            "chain_count": r.chain_count,
            "visual_duration_sec": r.visual_duration_sec,
        }
        for r in new_results if r.visual_duration_sec is not None
    ]
    n_new_ok = len(new_rows)
    n_new_no_motion = sum(1 for r in new_results if r.status == "no_motion_detected")
    n_new_truncated = sum(1 for r in new_results if r.status == "truncated_no_settle")
    print(f"[STEP4] 新規実測成功 {n_new_ok} / 動き未検出 {n_new_no_motion} / "
          f"打ち切り {n_new_truncated}")

    all_rows = old_rows + new_rows
    table = _percentile_table(all_rows)
    _write_table_csv(table, OUT_DIR / "table_by_chain_count.csv")
    _write_viz(all_rows, OUT_DIR / "distribution.png")

    # 検算: 指摘12 6連鎖=8.1秒 / 過去8連鎖=14.5秒
    row6 = next((r for r in table if r["chain_count"] == 6), None)
    row8 = next((r for r in table if r["chain_count"] == 8), None)
    lines = [
        "==== 連鎖数別 演出時間 検算 (2026-08-14) ====",
        f"旧23動画 (2026-07-24計測): status=ok {len(old_rows)}件",
        f"新10動画 (Phase L, mp4現存分, 2026-08-14計測): "
        f"収集{total_new}件 → サンプリング後{len(selected)}件測定 → "
        f"成功{n_new_ok}件 (動き未検出{n_new_no_motion}/打ち切り{n_new_truncated})",
        "",
    ]
    if row6:
        in_range6 = row6["p25_sec"] <= 8.1 <= row6["p75_sec"]
        lines.append(
            f"6連鎖: n={row6['n']} (旧{row6['n_old_23videos']}+新{row6['n_new_10videos_phase_l']}) "
            f"中央値={row6['median_sec']:.2f}s P25={row6['p25_sec']:.2f}s P75={row6['p75_sec']:.2f}s "
            f"→ 指摘12実測8.1秒は P25-P75帯内: {in_range6}"
        )
    if row8:
        in_range8 = row8["p25_sec"] <= 14.5 <= row8["p75_sec"]
        lines.append(
            f"8連鎖: n={row8['n']} (旧{row8['n_old_23videos']}+新{row8['n_new_10videos_phase_l']}) "
            f"中央値={row8['median_sec']:.2f}s P25={row8['p25_sec']:.2f}s P75={row8['p75_sec']:.2f}s "
            f"→ 過去実測14.5秒は P25-P75帯内: {in_range8}"
        )
    text = "\n".join(lines)
    (OUT_DIR / "README.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[DONE] 出力先: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

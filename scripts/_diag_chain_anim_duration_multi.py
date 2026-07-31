"""真因較正: 連鎖「消去演出」総時間の実測 (chain_count別分布) + 現行パイプライン
の早期離脱秒数を多動画で定量化する (2026-07-24, アーキ追加要求対応)。

完全 read-only 診断スクリプト。src/ は一切変更しない
(recognition_pipeline.py / board_state_machine.py は読むだけ)。
scripts/measure_exchange_dynamics.py も import のみ (書き換えない、
別コーダ作業中ファイルではないが既存資産の再利用に徹する)。

目的 (アーキ設計 案A0 の較正材料):
    現行 CHAIN_HOLD_PER_STEP_SEC=0.3 (src/recognition_pipeline.py:368) は
    「1連鎖ステップあたりの消去演出時間」の仮値。8連鎖の実測が旧計測で
    ~14.5秒 (計測器 memory `project_exchange_measurement_foundation`) に対し
    式は 0.3*8=2.4秒 と大幅過小評価の疑いがある。本スクリプトは:
        1. 実映像の「盤面全体モーション」から連鎖アニメ総時間を chain_count
           別に多動画・多イベントで実測し、分布 (中央値・外れ値・n) を出す。
        2. 線形モデル (a + b*chain_count) と原点通過モデル (b*chain_count)
           の当てはめを比較し、固定オーバーヘッド項の要否を判定する。
        3. 8連鎖実測が既知値 (~14.5秒) と整合するか確認する。
        4. 現行パイプラインが実際に CHAIN 状態を保持している秒数
           (t_fire - t_chain_start、npz 記録の実測値) との差分から
           「早期離脱秒数」を求める。
        5. 較正後の CHAIN_HOLD_PER_STEP_SEC 候補値を、ラグ増のトレード
           オフ込みで提示する。

方式:
    - イベント検出: scripts/measure_exchange_dynamics._process_video を
      再利用し、boards_lean_fixed npz から (video_stem, chain_count,
      t_chain_start, t_fire) を持つ FireEvent を得る (既存の pre-chain
      静止盤面アンカー方式、CNN 再推論不要で高速)。
    - 総時間実測: 該当 mp4 を t_chain_start から seek し、盤面 ROI の
      グレースケール frame間 diff (mean abs) を計測。「診断開始後に
      一度でも動きを検出し、その後 SETTLE_MIN_SEC 秒連続で diff が
      閾値以下に収まった」時刻を「アニメ完全終了 (視覚的 settle)」と
      判定する (CNN 不要、ChainAnimationDetector と同じ考え方の流用)。

出力先: data/verify/recognition_diag_chain_anim_duration_multi/
    - events_raw.csv         : イベント毎の実測値一覧
    - chain_count_bins.csv   : chain_count 別統計 (実測/pipeline/formula)
    - model_fit.json         : 線形回帰結果 (切片あり/なし両方)
    - duration_by_chain_count.png : 分布 viz (箱ひげ + 回帰直線)
    - summary.txt / summary.json  : 較正結論 + 閾値候補

Usage:
    PYTHONPATH=. ./venv/bin/python scripts/_diag_chain_anim_duration_multi.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "2")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from src.chain import ChainSimulator  # noqa: E402
from src.image_reader import BoardRegion, DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.measure_exchange_dynamics import (  # noqa: E402
    CHAIN_BIN_CAP, NPZ_DIR, TIER_MAP, FireEvent, _process_video,
)

# ============================
# 定数
# ============================
VIDEO_DIR: Path = PROJ_ROOT / "data" / "frames"
OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "recognition_diag_chain_anim_duration_multi"

# 現行仮値 (src/recognition_pipeline.py:368 と同一の class 定数を参照、
# マジックナンバー複製を避ける)。
CHAIN_HOLD_PER_STEP_SEC: float = RecognitionPipeline.CHAIN_HOLD_PER_STEP_SEC

# settle 判定: 盤面全体 grayscale diff (mean abs, 0-255) がこの値以下を「静止」
# とみなす (ChainAnimationDetector.LUMA_DIFF_THRESHOLD=18.0 は「アニメ中」判定用の
# 閾値なので、より保守的な低め閾値 (=誤って早期 settle 判定しない) を採用)。
SETTLE_DIFF_THRESHOLD: float = 6.0
# 上記閾値以下が何秒連続したら「settle 完了」とみなすか。
# 2026-07-24 較正: 初期値 0.5s は多段連鎖の「段間の短い静止 (実測 0.4-0.8s)」を
# 誤って最終 settle と判定する早期停止バグを起こしていた
# (scripts/_diag_settle_trace_inspect.py の生 diff trace で実証、
#  c11 8連鎖: 段間lullは全て<=0.6s、真の最終settleは~1.0s以上連続)。
# 1.2s に引き上げ、段間lullを跨いで真の完了のみ検出するよう修正。
SETTLE_MIN_SEC: float = 1.2
# 動き検出開始判定: 発火直後この秒数以内に SETTLE_DIFF_THRESHOLD 超の動きが
# 最低1回観測されなければ「アニメ未検出」として除外する (誤発火ガード)。
MOTION_START_MAX_WAIT_SEC: float = 1.5

# 探索窓の上限 (秒)。chain_count に応じて動的に決める (長い連鎖ほど長時間)。
SEARCH_WINDOW_BASE_SEC: float = 5.0
SEARCH_WINDOW_PER_CHAIN_SEC: float = 3.5
SEARCH_WINDOW_CAP_SEC: float = 35.0

# イベントサンプリング上限 (実行時間管理、userタスクの計測目的に対し十分な
# n を確保しつつ暴走を防ぐ)。chain_count>=RARE_CHAIN_COUNT_THRESHOLD は
# 全件採用 (レア高連鎖は較正の生命線のため間引かない)。
RARE_CHAIN_COUNT_THRESHOLD: int = 5
MAX_EVENTS_PER_VIDEO_PER_BIN: int = 5

# t_chain_start から実際にアニメ開始 (フレーム動き) を待つ最大遅延。
# (score OCR 検知と映像動きに若干ズレがあるため小さめのマージン。)
PRE_ROLL_SEC: float = 0.3


@dataclass
class _EventMeasurement:
    """1 イベント分の実測結果。"""

    video_stem: str
    tier: str
    fire_side: str
    chain_count: int
    t_chain_start: float
    t_fire: float
    visual_duration_sec: float | None  # None = 検出失敗 (truncated)
    n_motion_peaks: int  # diff が閾値を上回った回数 (視覚的ステップ数の目安)
    status: str  # "ok" / "no_motion_detected" / "truncated_no_settle"


# ============================
# イベント選定 (npz ベース、CNN 不要で高速)
# ============================


def _collect_fire_events() -> list[FireEvent]:
    """全 23 動画の npz から FireEvent (chain_count>=1) を収集する。"""
    sim = ChainSimulator()
    all_events: list[FireEvent] = []
    seq_id = 0
    for stem in sorted(TIER_MAP):
        npz_path = NPZ_DIR / f"{stem}.npz"
        if not npz_path.exists():
            print(f"[WARN] npz 不在スキップ: {stem}")
            continue
        _, defrag, seq_id = _process_video(npz_path, sim, seq_id)
        events = [e for e in defrag if e.chain_count >= 1]
        all_events.extend(events)
        print(f"[collect] {stem}: {len(events)} 件 (chain_count>=1)")
    return all_events


def _select_events(events: list[FireEvent]) -> list[FireEvent]:
    """chain_count 分布を保ちつつ実行時間を抑えるためサンプリングする。

    chain_count >= RARE_CHAIN_COUNT_THRESHOLD は全件採用 (較正の生命線)。
    それ未満は (video_stem, chain_count) 毎に最大 MAX_EVENTS_PER_VIDEO_PER_BIN
    件を均等間隔サンプリングする (ランダムでなく決定的、再現性のため)。
    """
    rare = [e for e in events if e.chain_count >= RARE_CHAIN_COUNT_THRESHOLD]
    common = [e for e in events if e.chain_count < RARE_CHAIN_COUNT_THRESHOLD]

    grouped: dict[tuple[str, int], list[FireEvent]] = {}
    for e in common:
        grouped.setdefault((e.video_stem, e.chain_count), []).append(e)

    sampled: list[FireEvent] = []
    for key, group in grouped.items():
        if len(group) <= MAX_EVENTS_PER_VIDEO_PER_BIN:
            sampled.extend(group)
        else:
            idx = np.linspace(0, len(group) - 1, num=MAX_EVENTS_PER_VIDEO_PER_BIN).astype(int)
            sampled.extend([group[i] for i in sorted(set(idx))])
    return rare + sampled


# ============================
# 実測: 盤面ROI grayscale diff による settle 検出
# ============================


def _region_gray(frame: np.ndarray, region: BoardRegion) -> np.ndarray:
    x1, y1 = region.x, region.y
    x2, y2 = region.x + region.width, region.y + region.height
    h_img, w_img = frame.shape[:2]
    x1, x2 = max(0, x1), min(w_img, x2)
    y1, y2 = max(0, y1), min(h_img, y2)
    return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)


def _measure_visual_duration(
    cap: cv2.VideoCapture, fps: float, region: BoardRegion,
    t_chain_start: float, chain_count: int,
) -> tuple[float | None, int, str]:
    """t_chain_start から盤面 ROI の動きを追跡し settle 時刻を返す。

    Returns:
        (visual_duration_sec, n_motion_peaks, status)
    """
    search_window = min(
        SEARCH_WINDOW_CAP_SEC,
        SEARCH_WINDOW_BASE_SEC + SEARCH_WINDOW_PER_CHAIN_SEC * chain_count,
    )
    start_fi = int(round(max(0.0, t_chain_start - PRE_ROLL_SEC) * fps))
    end_fi = int(round((t_chain_start + search_window) * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_fi))

    prev_gray: np.ndarray | None = None
    motion_seen = False
    motion_start_t: float | None = None
    settle_run_start_t: float | None = None
    n_peaks = 0
    was_above = False
    fi = start_fi
    while fi <= end_fi:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t = fi / fps
        gray = _region_gray(frame, region)
        if prev_gray is not None and prev_gray.shape == gray.shape:
            diff = float(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16)).mean())
            above = diff > SETTLE_DIFF_THRESHOLD
            if above and not was_above:
                n_peaks += 1
            was_above = above
            if above:
                motion_seen = True
                if motion_start_t is None:
                    motion_start_t = t
                settle_run_start_t = None
            else:
                if motion_seen:
                    if settle_run_start_t is None:
                        settle_run_start_t = t
                    elif t - settle_run_start_t >= SETTLE_MIN_SEC:
                        return (t - t_chain_start, n_peaks, "ok")
                if not motion_seen and (t - t_chain_start) > MOTION_START_MAX_WAIT_SEC:
                    return (None, n_peaks, "no_motion_detected")
        prev_gray = gray
        fi += 1

    if not motion_seen:
        return (None, n_peaks, "no_motion_detected")
    return (None, n_peaks, "truncated_no_settle")


def _measure_events_for_video(stem: str, events: list[FireEvent]) -> list[_EventMeasurement]:
    """1 動画分のイベントをまとめて計測する (VideoCapture を使い回す)。"""
    video_path = VIDEO_DIR / f"video_{stem}.mp4"
    if not video_path.exists():
        print(f"[WARN] mp4 不在スキップ: {stem}")
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] mp4 open失敗: {stem}")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    results: list[_EventMeasurement] = []
    for ev in events:
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
    cap.release()
    return results


# ============================
# 集計 + モデル当てはめ
# ============================


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """a + b*x モデルの最小二乗フィット + R^2 を返す。"""
    if len(x) < 3:
        return {"a": None, "b": None, "r2": None, "n": len(x)}
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {"a": float(a), "b": float(b), "r2": r2, "n": len(x)}


def _origin_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """b*x (切片なし) モデルの最小二乗フィット + R^2 を返す。"""
    if len(x) < 3:
        return {"b": None, "r2": None, "n": len(x)}
    b = float(np.sum(x * y) / np.sum(x * x))
    pred = b * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {"b": b, "r2": r2, "n": len(x)}


def _write_bins_csv(results: list[_EventMeasurement], out_path: Path) -> list[dict]:
    """chain_count 別 (実測 / pipeline / formula) 統計を CSV + dict list で返す。"""
    ok = [r for r in results if r.visual_duration_sec is not None]
    bins: dict[int, list[_EventMeasurement]] = {}
    for r in ok:
        b = min(r.chain_count, CHAIN_BIN_CAP)
        bins.setdefault(b, []).append(r)

    rows = []
    for b in sorted(bins):
        group = bins[b]
        visual = np.array([r.visual_duration_sec for r in group])
        pipeline = np.array([r.t_fire - r.t_chain_start for r in group])
        formula = np.array([CHAIN_HOLD_PER_STEP_SEC * r.chain_count for r in group])
        early_exit = visual - pipeline
        rows.append({
            "chain_bin": b, "n": len(group),
            "visual_median": float(np.median(visual)), "visual_mean": float(np.mean(visual)),
            "visual_std": float(np.std(visual)), "visual_max": float(np.max(visual)),
            "visual_min": float(np.min(visual)),
            "pipeline_median": float(np.median(pipeline)), "pipeline_mean": float(np.mean(pipeline)),
            "formula_sec_0.3perchain": float(np.mean(formula)),
            "early_exit_median": float(np.median(early_exit)),
            "early_exit_mean": float(np.mean(early_exit)),
            "early_exit_max": float(np.max(early_exit)),
        })
    cols = list(rows[0].keys()) if rows else []
    lines = [",".join(cols)]
    for row in rows:
        lines.append(",".join(str(row[c]) for c in cols))
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return rows


def _write_events_csv(results: list[_EventMeasurement], out_path: Path) -> None:
    cols = [
        "video_stem", "tier", "fire_side", "chain_count", "t_chain_start", "t_fire",
        "visual_duration_sec", "pipeline_duration_sec", "formula_duration_sec",
        "early_exit_sec", "n_motion_peaks", "status",
    ]
    lines = [",".join(cols)]
    for r in results:
        pipeline_dur = r.t_fire - r.t_chain_start
        formula_dur = CHAIN_HOLD_PER_STEP_SEC * r.chain_count
        early_exit = (
            r.visual_duration_sec - pipeline_dur if r.visual_duration_sec is not None else ""
        )
        vdur = r.visual_duration_sec if r.visual_duration_sec is not None else ""
        lines.append(",".join(str(v) for v in [
            r.video_stem, r.tier, r.fire_side, r.chain_count,
            f"{r.t_chain_start:.3f}", f"{r.t_fire:.3f}", vdur,
            f"{pipeline_dur:.3f}", f"{formula_dur:.3f}", early_exit,
            r.n_motion_peaks, r.status,
        ]))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_viz(results: list[_EventMeasurement], model_linear: dict, model_origin: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in results if r.visual_duration_sec is not None]
    if not ok:
        return
    x = np.array([min(r.chain_count, CHAIN_BIN_CAP) for r in ok])
    y = np.array([r.visual_duration_sec for r in ok])

    fig, ax = plt.subplots(figsize=(10, 7))
    bins_present = sorted(set(x.tolist()))
    data = [y[x == b] for b in bins_present]
    ax.boxplot(data, positions=bins_present, widths=0.5, showfliers=True)
    ax.scatter(x + np.random.uniform(-0.08, 0.08, size=len(x)), y, s=10, alpha=0.35, color="gray")

    xs = np.linspace(min(bins_present), max(bins_present), 50)
    if model_linear["a"] is not None:
        ax.plot(xs, model_linear["a"] + model_linear["b"] * xs, color="red",
                 label=f"線形 a+b*n (a={model_linear['a']:.2f},b={model_linear['b']:.2f},R2={model_linear['r2']:.2f})")
    if model_origin["b"] is not None:
        ax.plot(xs, model_origin["b"] * xs, color="blue", linestyle="--",
                 label=f"原点通過 b*n (b={model_origin['b']:.2f},R2={model_origin['r2']:.2f})")
    ax.plot(xs, CHAIN_HOLD_PER_STEP_SEC * xs, color="green", linestyle=":",
             label=f"現行式 0.3*n (CHAIN_HOLD_PER_STEP_SEC)")
    ax.set_xlabel("chain_count (8+はまとめ)")
    ax.set_ylabel("実測 消去演出総時間 (秒)")
    ax.set_title("連鎖数別 消去演出総時間 実測分布 + モデル当てはめ (23動画)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _format_summary(bin_rows: list[dict], model_linear: dict, model_origin: dict, results: list[_EventMeasurement]) -> tuple[str, dict]:
    n_total = len(results)
    n_ok = sum(1 for r in results if r.visual_duration_sec is not None)
    n_no_motion = sum(1 for r in results if r.status == "no_motion_detected")
    n_truncated = sum(1 for r in results if r.status == "truncated_no_settle")

    eight_bin = next((row for row in bin_rows if row["chain_bin"] == CHAIN_BIN_CAP), None)
    margin_ratio = 1.3
    if model_linear["b"] is not None:
        candidate_slope = model_linear["b"] * margin_ratio
    elif model_origin["b"] is not None:
        candidate_slope = model_origin["b"] * margin_ratio
    else:
        candidate_slope = None

    summary_dict = {
        "n_events_total_sampled": n_total,
        "n_events_measured_ok": n_ok,
        "n_no_motion_detected": n_no_motion,
        "n_truncated_no_settle": n_truncated,
        "model_linear_a_plus_bx": model_linear,
        "model_origin_bx": model_origin,
        "known_8chain_bin_stats": eight_bin,
        "current_formula_coef_sec_per_chain": CHAIN_HOLD_PER_STEP_SEC,
        "candidate_coef_sec_per_chain_margin30pct": candidate_slope,
        "chain_count_bins": bin_rows,
    }
    lines = [
        "==== 連鎖「消去演出」総時間 較正 サマリ (23動画・chain_count別) ====",
        f"サンプル数: 総{n_total} 実測成功{n_ok} 動き未検出{n_no_motion} settle未検出(打ち切り){n_truncated}",
        f"線形モデル (a+b*chain_count): a={model_linear['a']} b={model_linear['b']} R2={model_linear['r2']} n={model_linear['n']}",
        f"原点通過モデル (b*chain_count): b={model_origin['b']} R2={model_origin['r2']} n={model_origin['n']}",
        f"現行仮値 (CHAIN_HOLD_PER_STEP_SEC): {CHAIN_HOLD_PER_STEP_SEC} 秒/連鎖",
        f"8連鎖ビン実測: {eight_bin}",
        f"較正候補 (フィット係数 × 1.3マージン): {candidate_slope} 秒/連鎖"
        + (f" (+ 固定項 {model_linear['a']:.2f}秒)" if model_linear.get("a") else ""),
        "--- chain_count別ビン統計 ---",
    ]
    for row in bin_rows:
        lines.append(
            f"  bin={row['chain_bin']} n={row['n']} visual_median={row['visual_median']:.2f} "
            f"pipeline_median={row['pipeline_median']:.2f} formula={row['formula_sec_0.3perchain']:.2f} "
            f"early_exit_median={row['early_exit_median']:.2f} early_exit_max={row['early_exit_max']:.2f}"
        )
    return "\n".join(lines), summary_dict


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[STEP1] npz から FireEvent 収集中 (23動画)...")
    all_events = _collect_fire_events()
    print(f"[STEP1] 完了: chain_count>=1 総 {len(all_events)} 件")

    selected = _select_events(all_events)
    print(f"[STEP2] サンプリング後: {len(selected)} 件を実測対象とする")

    by_video: dict[str, list[FireEvent]] = {}
    for ev in selected:
        by_video.setdefault(ev.video_stem, []).append(ev)

    all_results: list[_EventMeasurement] = []
    n_video_done = 0
    for stem, evs in sorted(by_video.items()):
        res = _measure_events_for_video(stem, evs)
        all_results.extend(res)
        n_video_done += 1
        print(f"[STEP3] {stem} 完了 ({n_video_done}/{len(by_video)}動画, 累計{len(all_results)}件測定)")

    _write_events_csv(all_results, OUTPUT_DIR / "events_raw.csv")
    bin_rows = _write_bins_csv(all_results, OUTPUT_DIR / "chain_count_bins.csv")

    ok = [r for r in all_results if r.visual_duration_sec is not None]
    x = np.array([min(r.chain_count, CHAIN_BIN_CAP) for r in ok], dtype=float)
    y = np.array([r.visual_duration_sec for r in ok], dtype=float)
    model_linear = _linear_fit(x, y)
    model_origin = _origin_fit(x, y)
    (OUTPUT_DIR / "model_fit.json").write_text(
        json.dumps({"linear": model_linear, "origin": model_origin}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_viz(all_results, model_linear, model_origin, OUTPUT_DIR / "duration_by_chain_count.png")

    text, summary_dict = _format_summary(bin_rows, model_linear, model_origin, all_results)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary_dict, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    (OUTPUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print(f"[DONE] 出力先: {OUTPUT_DIR}")
    print(text)


if __name__ == "__main__":
    main()

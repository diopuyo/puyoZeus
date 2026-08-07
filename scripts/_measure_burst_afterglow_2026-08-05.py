"""BURST_GATE_POST_CLOSE_COOLDOWN_SEC の物理根拠測定 (2026-08-05)。

`scripts/_verify_c29_stage15_bypass_2026-08-05.py` で確定した第5機構
(窓close直後0.27〜0.33秒の残光が無防備経路から漏れる) を、c29単発のシーン
逆算ではなく**個体群の減衰時間分布**として測定し、クールダウン定数の候補値
に物理的根拠を与える (`feedback_overfitting_awareness_2026-08-04` 準拠、
シーン逆算禁止)。

## 測定対象 (2母集団、混同せず層別報告する)
- `onset93`: `build_error_onset_sheet_2026-08-04.py::diagnose_all_samples` の
  93セルonset (誤りの「焼き付いた瞬間」、重複onsetは (video,side,onset_t)
  0.1秒丸めで de-dup)。
- `calib17`: `labeled_cell_features_v3.csv` の `is_true_burst=True` 17行
  (人手ラベルの「明確にバーストが写っている」瞬間、onsetそのものではなく
  ラベル時点である点が onset93 と意味論的に異なるため合算しない)。

## 測定方法
各イベントについて動画から `center_t ± PROBE_WINDOW_SEC` を
`PROBE_STEP_SEC` 間隔でシーケンシャル読み込みし、
`compute_effect_glow_score` (rows1-3、本番と同一ロジック) を実測する。
1. `t_last_open`: score が `BURST_GATE_OPEN_THRESHOLD_RECALIBRATED` (0.954)
   を最後に上回った時刻。
2. 減衰時間 (`decay_to_baseline_sec`): `t_last_open` から、score が
   イベント前の基底レベル (`t∈[-3.0,-2.0]` 区間の中央値 + マージン) に
   最初に戻る時刻までの経過秒数。
3. 残光帯滞在時間 (`afterglow_band_sec`): `t_last_open` から、score が
   固定下限 `AFTERGLOW_BAND_LOW` (0.5) を最初に下回る時刻までの経過秒数
   (coordinator指定の副次指標、baseline推定に依存しないため頑健)。
右censoring (`PROBE_WINDOW_SEC` 内に戻り切らない) は明示フラグで除外集計
する (打ち切り値を無自覚に混ぜると分布が過小評価される)。

## 既存資産の再利用 (コピペ禁止指示への対応)
- `scripts/build_error_onset_sheet_2026-08-04.py::diagnose_all_samples`
  (importlib動的import、onset検出ロジックは再実装しない)
- `src/effect_glow_detector.py::compute_effect_glow_score` (Stage0抽出済み)
- `src/image_reader.py::DEFAULT_P1_REGION/DEFAULT_P2_REGION`

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._measure_burst_afterglow_2026-08-05
"""
from __future__ import annotations

import csv
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board_state_machine import EFFECT_GATE_TOP_ROWS  # noqa: E402
from src.effect_glow_detector import compute_effect_glow_score  # noqa: E402
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION  # noqa: E402

# ファイル名にハイフンを含むため通常の `from ... import` は不可 (動的import)。
_SHEET = importlib.import_module("scripts.build_error_onset_sheet_2026-08-04")

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

VIDEO_DIR: Path = Path("/home/ryouj/frames")
CALIBRATION_CSV: Path = Path(
    "data/verify/effect_detector_calibration_v3_2026-08-04/labeled_cell_features_v3.csv"
)
OUTPUT_CSV_PATH: Path = Path(
    "data/verify/burst_guard_2026-08-05/burst_afterglow_events.csv"
)

# on_v2_full backtest で実際に使われた閾値 (較正済み推奨値)。
BURST_GATE_OPEN_THRESHOLD_RECALIBRATED: float = 0.954

# 測定窓 (onset/ラベル時点 ± この秒数)。
PROBE_WINDOW_SEC: float = 3.0
# censored (窓内に戻らない) イベント専用の再測定窓 (連鎖中の複数バーストが
# 3秒以内に再点火し続けている疑いを切り分けるため、より広く取る)。
WIDE_PROBE_WINDOW_SEC: float = 8.0
# サンプリング間隔目標値 (実際は fps に応じて最も近いフレーム数に量子化)。
PROBE_STEP_SEC: float = 0.1

# 残光帯 (coordinator指定、baseline推定に依存しない固定下限)。
AFTERGLOW_BAND_LOW: float = 0.5

# 基底レベル推定に使う「イベント前の静穏区間」(center_t からの相対秒)。
BASELINE_WINDOW_REL_SEC: "tuple[float, float]" = (-3.0, -2.0)
BASELINE_RETURN_MARGIN: float = 0.05

# 動画フレームの正規化サイズ (他スクリプトと同一、1920x1080前提のregionを使うため)。
FRAME_TARGET_WIDTH: int = 1920
FRAME_TARGET_HEIGHT: int = 1080

SOURCE_ONSET93: str = "onset93"
SOURCE_CALIB17: str = "calib17"


# =============================================================================
# データ構造
# =============================================================================


@dataclass(frozen=True)
class AfterglowEvent:
    """1イベント (video, side, center_t) の減衰測定入力。"""

    source: str
    video: str
    side: str
    center_t_sec: float


@dataclass
class AfterglowResult:
    """1イベント分の測定結果。"""

    event: AfterglowEvent
    max_score: "float | None"
    ever_opened: bool
    t_last_open_rel: "float | None"
    decay_to_baseline_sec: "float | None"
    decay_censored: bool
    afterglow_band_sec: "float | None"
    afterglow_censored: bool
    baseline_level: "float | None"


# =============================================================================
# 1. イベント列挙 (2母集団、意味論が異なるため混同しない)
# =============================================================================


def _dedup_events(events: list[AfterglowEvent]) -> list[AfterglowEvent]:
    """(video, side, center_t 0.1秒丸め) で重複イベントを除く。"""
    seen: set[tuple[str, str, str, float]] = set()
    out: list[AfterglowEvent] = []
    for e in events:
        key = (e.source, e.video, e.side, round(e.center_t_sec, 1))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def load_onset93_events() -> list[AfterglowEvent]:
    """93セルonsetのうち burst layer (row1-3) かつ onset特定済みの一意イベント。"""
    records = _SHEET.diagnose_all_samples()
    events = [
        AfterglowEvent(SOURCE_ONSET93, r.video, r.side, r.onset_t_sec)
        for r in records
        if r.row in EFFECT_GATE_TOP_ROWS and r.onset_t_sec is not None
    ]
    return _dedup_events(events)


def load_calib17_events() -> list[AfterglowEvent]:
    """較正セットの真バースト正例17件 (人手ラベル時点、onsetとは意味論が異なる)。"""
    df = pd.read_csv(CALIBRATION_CSV, encoding="utf-8-sig")
    pos = df[df["is_true_burst"] == True]  # noqa: E712 (pandasの規約上明示比較)
    events = [
        AfterglowEvent(SOURCE_CALIB17, str(row.video_stem), str(row.side), float(row.t_sec))
        for row in pos.itertuples()
    ]
    return _dedup_events(events)


# =============================================================================
# 2. スコア時系列抽出 (シーケンシャル読み込み、seek 1回のみ)
# =============================================================================


def _sample_scores(
    event: AfterglowEvent, cap_cache: dict, fps_cache: dict,
    window_sec: float = PROBE_WINDOW_SEC,
) -> "list[tuple[float, float]] | None":
    """center_t ± window_sec を PROBE_STEP_SEC 間隔で実測する (相対秒,score)。"""
    region = DEFAULT_P1_REGION if event.side == "1P" else DEFAULT_P2_REGION
    if event.video not in cap_cache:
        video_path = VIDEO_DIR / f"video_{event.video}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        cap_cache[event.video] = cap
        fps_cache[event.video] = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap = cap_cache[event.video]
    fps = fps_cache[event.video]
    start_t = max(0.0, event.center_t_sec - window_sec)
    start_frame = int(round(start_t * fps))
    step = max(1, int(round(PROBE_STEP_SEC * fps)))
    n_frames_needed = int(round((event.center_t_sec + window_sec - start_t) * fps)) + step
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    samples: list[tuple[float, float]] = []
    for i in range(n_frames_needed):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if i % step != 0:
            continue
        if frame.shape[:2] != (FRAME_TARGET_HEIGHT, FRAME_TARGET_WIDTH):
            frame = cv2.resize(frame, (FRAME_TARGET_WIDTH, FRAME_TARGET_HEIGHT))
        score = compute_effect_glow_score(frame, region, EFFECT_GATE_TOP_ROWS)
        t_rel = (start_frame + i) / fps - event.center_t_sec
        samples.append((t_rel, score))
    return samples if samples else None


# =============================================================================
# 3. 減衰時間 / 残光帯滞在時間の算出
# =============================================================================


def _baseline_level(samples: "list[tuple[float, float]]") -> "float | None":
    """イベント前の静穏区間 (BASELINE_WINDOW_REL_SEC) の score 中央値。"""
    lo, hi = BASELINE_WINDOW_REL_SEC
    vals = [s for t, s in samples if lo <= t <= hi]
    return float(np.median(vals)) if vals else None


def _last_open_index(samples: "list[tuple[float, float]]") -> "int | None":
    """score が閾値を最後に上回ったサンプルの index (一度も無ければ None)。"""
    idxs = [
        i for i, (_t, s) in enumerate(samples)
        if s >= BURST_GATE_OPEN_THRESHOLD_RECALIBRATED
    ]
    return idxs[-1] if idxs else None


def _first_crossing_after(
    samples: "list[tuple[float, float]]", start_idx: int, level: float,
) -> "tuple[float, bool]":
    """start_idx より後で score が level 以下になる最初の時刻を探す (相対秒)。

    見つからなければ (窓終端の相対秒, censored=True) を返す。
    """
    for t, s in samples[start_idx:]:
        if s <= level:
            return t, False
    return samples[-1][0], True


def measure_one_event(
    event: AfterglowEvent, cap_cache: dict, fps_cache: dict,
    window_sec: float = PROBE_WINDOW_SEC,
) -> "AfterglowResult | None":
    """1イベント分の減衰時間・残光帯滞在時間を測定する (window_sec: 観測窓の半幅)。"""
    samples = _sample_scores(event, cap_cache, fps_cache, window_sec)
    if samples is None:
        return None
    max_score = max(s for _t, s in samples)
    last_open_idx = _last_open_index(samples)
    if last_open_idx is None:
        return AfterglowResult(
            event, max_score, False, None, None, False, None, False, None,
        )
    t_last_open = samples[last_open_idx][0]
    baseline = _baseline_level(samples)
    decay_sec: "float | None" = None
    decay_censored = False
    if baseline is not None:
        t_ret, censored = _first_crossing_after(
            samples, last_open_idx, baseline + BASELINE_RETURN_MARGIN,
        )
        decay_sec, decay_censored = (t_ret - t_last_open), censored
    t_ag, ag_censored = _first_crossing_after(samples, last_open_idx, AFTERGLOW_BAND_LOW)
    return AfterglowResult(
        event, max_score, True, t_last_open,
        decay_sec, decay_censored, (t_ag - t_last_open), ag_censored, baseline,
    )


def reprobe_censored_wide(
    results: list[AfterglowResult], cap_cache: dict, fps_cache: dict,
) -> "tuple[list[AfterglowResult], int]":
    """3秒窓で censored だったイベントのみ WIDE_PROBE_WINDOW_SEC で再測定する。

    連鎖中の複数バーストが再点火を続けている (=単発の残光ではない) 疑いを
    切り分ける。戻り値は (更新後の全結果リスト, 8秒窓でも censored のまま
    残った件数=連鎖持続型で固定クールダウンでは守れない疑いのある件数)。
    """
    updated: list[AfterglowResult] = []
    still_censored = 0
    for r in results:
        if not r.ever_opened or not r.afterglow_censored:
            updated.append(r)
            continue
        wide = measure_one_event(r.event, cap_cache, fps_cache, WIDE_PROBE_WINDOW_SEC)
        if wide is None:
            updated.append(r)
            continue
        if wide.afterglow_censored:
            still_censored += 1
        updated.append(wide)
    return updated, still_censored


# =============================================================================
# 4. 出力 (CSV + 分布サマリ)
# =============================================================================


def write_events_csv(results: list[AfterglowResult], out_path: Path) -> None:
    """イベント単位の測定結果をCSVに書き出す。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source", "video", "side", "center_t_sec", "max_score", "ever_opened",
        "t_last_open_rel", "decay_to_baseline_sec", "decay_censored",
        "afterglow_band_sec", "afterglow_censored", "baseline_level",
    ]
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "source": r.event.source, "video": r.event.video, "side": r.event.side,
                "center_t_sec": r.event.center_t_sec, "max_score": f"{r.max_score:.3f}",
                "ever_opened": r.ever_opened,
                "t_last_open_rel": _fmt(r.t_last_open_rel),
                "decay_to_baseline_sec": _fmt(r.decay_to_baseline_sec),
                "decay_censored": r.decay_censored,
                "afterglow_band_sec": _fmt(r.afterglow_band_sec),
                "afterglow_censored": r.afterglow_censored,
                "baseline_level": _fmt(r.baseline_level),
            })


def _fmt(v: "float | None") -> str:
    return "" if v is None else f"{v:.3f}"


def _dist_line(label: str, values: list[float]) -> str:
    """min/p50/p90/max を1行で報告する。"""
    if not values:
        return f"  {label}: データなし"
    arr = np.array(values)
    return (
        f"  {label} (n={len(arr)}): min={arr.min():.3f} p50={np.median(arr):.3f} "
        f"p90={np.percentile(arr,90):.3f} max={arr.max():.3f}"
    )


def build_summary(results: list[AfterglowResult], source: str) -> str:
    """1母集団分の分布サマリ (減衰時間 + 残光帯滞在時間、censored別集計)。"""
    group = [r for r in results if r.event.source == source]
    opened = [r for r in group if r.ever_opened]
    lines = [
        f"--- {source} (全{len(group)}件、窓が開いた={len(opened)}件、"
        f"一度も開かず={len(group) - len(opened)}件) ---",
    ]
    decay_ok = [r.decay_to_baseline_sec for r in opened if not r.decay_censored and r.decay_to_baseline_sec is not None]
    decay_cens = [r for r in opened if r.decay_censored]
    lines.append(_dist_line("減衰時間(baseline方式,非censored)", decay_ok))
    lines.append(f"  減衰時間 censored (窓内に戻らず): {len(decay_cens)}件")
    ag_ok = [r.afterglow_band_sec for r in opened if not r.afterglow_censored and r.afterglow_band_sec is not None]
    ag_cens = [r for r in opened if r.afterglow_censored]
    lines.append(_dist_line("残光帯滞在時間(0.5-0.954方式,非censored)", ag_ok))
    lines.append(f"  残光帯滞在時間 censored (窓内に戻らず): {len(ag_cens)}件")
    return "\n".join(lines)


def recommend_cooldown(results: list[AfterglowResult]) -> str:
    """p90ベース/max+マージンベースの推奨クールダウン値 (2母集団統合・非censored限定)。"""
    opened = [r for r in results if r.ever_opened]
    ag_ok = [
        r.afterglow_band_sec for r in opened
        if not r.afterglow_censored and r.afterglow_band_sec is not None
    ]
    if not ag_ok:
        return "[推奨クールダウン] 非censoredデータなし、判断不能"
    arr = np.array(ag_ok)
    p90 = float(np.percentile(arr, 90))
    mx = float(np.max(arr))
    margin = PROBE_STEP_SEC  # サンプリング量子化誤差 (0.1秒) をマージンとして加算
    n_censored = sum(1 for r in opened if r.afterglow_censored)
    return (
        f"[推奨クールダウン] (根拠=残光帯0.5-0.954滞在時間、非censored n={len(arr)}、"
        f"censored除外={n_censored}件)\n"
        f"  p90ベース: {p90:.3f}秒 (+量子化マージン{margin}秒 → {p90 + margin:.3f}秒)\n"
        f"  max+マージンベース: {mx:.3f}秒 (+量子化マージン{margin}秒 → {mx + margin:.3f}秒)\n"
        f"  [注] censored{n_censored}件は真の減衰がこの範囲を超える可能性を示す下限値のため、"
        f"上記推奨値は保守的な下限として扱うこと"
    )


# =============================================================================
# 5. main
# =============================================================================


def main() -> None:
    cv2.setNumThreads(1)
    onset_events = load_onset93_events()
    calib_events = load_calib17_events()
    print(f"[1/3] イベント列挙: onset93={len(onset_events)}件 / calib17={len(calib_events)}件")

    cap_cache: dict = {}
    fps_cache: dict = {}
    results: list[AfterglowResult] = []
    for event in onset_events + calib_events:
        r = measure_one_event(event, cap_cache, fps_cache)
        if r is not None:
            results.append(r)
    print(f"[2/4] 測定完了(±{PROBE_WINDOW_SEC}秒窓): {len(results)}件 (フレーム取得不能は除外)")

    results, still_censored = reprobe_censored_wide(results, cap_cache, fps_cache)
    for cap in cap_cache.values():
        cap.release()
    print(
        f"[3/4] censored再測定(±{WIDE_PROBE_WINDOW_SEC}秒窓): "
        f"8秒窓でも戻らず=連鎖持続の疑い {still_censored}件"
    )

    write_events_csv(results, OUTPUT_CSV_PATH)
    print(f"[出力] イベント単位CSV: {OUTPUT_CSV_PATH}")

    print("\n[4/4] " + build_summary(results, SOURCE_ONSET93))
    print("\n" + build_summary(results, SOURCE_CALIB17))
    print(
        f"\n[連鎖持続の疑い] 8秒窓でも残光帯から戻らないイベント: {still_censored}件 "
        "(固定クールダウンでは保護できない可能性、quiescence側の設計対応を要検討)"
    )
    print("\n" + recommend_cooldown(results))


if __name__ == "__main__":
    main()

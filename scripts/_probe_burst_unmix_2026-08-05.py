"""バーストunmix (α合成除去) の実現可能性検証 (2026-08-05、research probe)。

user案: 「バースト = 5色の半透明レイヤーの不透明度を記録して除去 (unmix)」の
実現可能性を、今夜確定した誤りセル93件 (onset時点の観測値=誤読 と
correct_grid=正解 が両方既知) を教師データとして数値検証する。

## モデル
観測画素 = α×バースト色 + (1-α)×真の画素 (BGR 線形合成、HSV空間では合成が
非線形になるため BGR で解く)。

## α・バースト色の推定方式
同一 onset (同一フレーム) で複数セルが同時に汚染される「同時onsetグループ」
(`build_error_onset_sheet_2026-08-04.py` の `group_key`/`group_size` を再利用)
では、各セルの correct_value から得る「標準色 (ColorClassifier の
DEFAULT_COLOR_RANGES を import、再実装しない)」を真の画素の近似値として使い、
observed_i = α×burst + (1-α)×true_i を全セルについて連立させる:
    observed_i - true_i = (α×burst) - α×true_i
未知数を B=α×burst (3ch) と α (1) に置くと **線形**になり、group_size>=2 なら
最小二乗で一意に解ける (3ch × N セル の方程式、未知数4)。
group_size==1 の単独セルは単独では α と burst を分離できないため、同一動画の
他グループから推定した burst (無ければ全動画横断の平均 burst) を固定して
α のみを解く (1D最小二乗)。

## unmix後の再分類テスト
推定した α・burst で observed から true を逆算し (`(observed-α×burst)/(1-α)`)、
既存 `ColorClassifier.classify()` (import、再実装しない) に均一パッチとして
渡して correct_value に戻るか判定する。

## 負例チェック
onset-2秒 (バースト前、タイムライン上で既に correct_value を示している行が
存在する場合のみ) の観測画素に同じ α・burst を **無条件に**適用し、
誤って別の色に変換してしまう率を測る (過剰適用の危険度)。

これは実現可能性検証 (research probe) であり本番実装ではない。窓保留方式
(別途設計中) との比較材料として使う。

Usage:
    PYTHONPATH=. ./venv/bin/python -m scripts._probe_burst_unmix_2026-08-05
"""
from __future__ import annotations

import importlib
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import COLOR_EMPTY, COLOR_OJAMA  # noqa: E402
from src.cell_recovery_refiner import (  # noqa: E402
    OJM_RECOVERY_S_MAX,
    OJM_RECOVERY_V_MAX,
    OJM_RECOVERY_V_MIN,
)
from src.image_reader import ColorClassifier, DEFAULT_COLOR_RANGES  # noqa: E402

# ファイル名にハイフンを含むため動的 import (コピペ禁止指示への対応)。
_SHEET = importlib.import_module("scripts.build_error_onset_sheet_2026-08-04")
_DIAG = importlib.import_module("scripts._diag_c_zero_effect_2026-08-04")
_MC = importlib.import_module("scripts.measure_effect_gate_c_2026-08-04")

# =============================================================================
# 定数 (マジックナンバー禁止のため全て定数化)
# =============================================================================

# 負例チェック用の「バースト前」オフセット (onset 基準、秒)。
NEGATIVE_CHECK_OFFSET_SEC: float = -2.0

# α >= この値なら「情報がほぼ残っていない (復元原理的に不能)」と判定する。
ALPHA_INFO_LOST_MIN: float = 0.90
# 数値安定化のための α 上限 (1.0 で除算不能になるため)。
ALPHA_DIV_CLIP_MAX: float = 0.999

# 再分類テスト用の合成パッチサイズ (均一色、classify() の median 計算に十分な大きさ)。
SYNTH_PATCH_SIZE: int = 5

# 標準色 (真の画素) 近似時の彩度・輝度マージン (DEFAULT_COLOR_RANGES の
# s_min/v_min に加算し、「本物のぷよの典型値」に寄せる。範囲全体の中央値だと
# 現実の彩度分布より高くなりすぎるため、下限寄りの典型値を採用)。
COLOR_REF_S_MARGIN: int = 40
COLOR_REF_V_MARGIN: int = 60
# COLOR_EMPTY (背景/空セル) の標準 HSV 近似値 (暗め・低彩度の盤面背景)。
EMPTY_REF_HSV: tuple[int, int, int] = (0, 20, 60)

# 有効な burst 推定として採用する α の範囲 (この範囲外は数値的に不安定・
# モデル不適合とみなし、per-video/global 平均には使わない)。
VALID_BURST_ALPHA_MIN: float = 0.05
VALID_BURST_ALPHA_MAX: float = 0.98
# burst_BGR が物理的に妥当とみなす範囲 (uint8 の 0-255 に多少の余裕を持たせる)。
# 連立方程式が rank 不足 (グループ内の correct_value が偏っている等) で
# 解が不安定になると数千のような非物理値が出るため、これを弾く。
VALID_BURST_BGR_MIN: float = -30.0
VALID_BURST_BGR_MAX: float = 285.0

STATUS_SUCCESS: str = "success"
STATUS_INFO_LOST: str = "info_lost"
STATUS_MODEL_MISMATCH: str = "model_mismatch"
STATUS_NO_DATA: str = "no_data"


# =============================================================================
# データ構造
# =============================================================================


@dataclass
class GroupBurstEstimate:
    """同時onsetグループ1件分の α・burst 推定結果。"""

    video: str
    group_key: str
    n_cells: int
    alpha: float
    burst_bgr: "np.ndarray | None"  # None = α≈0 で burst 未分離
    valid: bool  # VALID_BURST_ALPHA_MIN/MAX 範囲内か


@dataclass
class CellUnmixResult:
    """1誤りセル分の unmix 再分類結果。"""

    video: str
    side: str
    row: int
    col: int
    wrong_value: int
    correct_value: int
    alpha: "float | None"
    burst_source: str  # "joint" (グループ同時推定) / "fallback" (per-video/global)
    reclass_value: "int | None"
    status: str
    neg_check_flipped: "bool | None" = None  # 負例チェック: 誤変換したか


# =============================================================================
# 1. 標準色参照 (ColorClassifier の基準値を import、再定義しない)
# =============================================================================


def color_reference_bgr(color: int) -> "np.ndarray":
    """correct_value から「標準的な真の画素」BGR を近似する (研究probe用の粗い近似)。"""
    if color == COLOR_EMPTY:
        h, s, v = EMPTY_REF_HSV
    elif color == COLOR_OJAMA:
        h, s, v = 0, OJM_RECOVERY_S_MAX // 2, (OJM_RECOVERY_V_MIN + OJM_RECOVERY_V_MAX) // 2
    else:
        rng = DEFAULT_COLOR_RANGES[color][0]
        h = (rng.h_min + rng.h_max) // 2
        s = min(255, rng.s_min + COLOR_REF_S_MARGIN)
        v = min(255, rng.v_min + COLOR_REF_V_MARGIN)
    hsv = np.array([[[h, s, v]]], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float64)


def _patch_mean_bgr(frame_bgr: "np.ndarray", region: "object", row: int, col: int) -> "np.ndarray":
    """対象セルパッチの平均 BGR (線形合成モデルに使う連続値、median でなく mean)。"""
    x1, y1, x2, y2 = region.cell_sample_rect(row, col)
    patch = frame_bgr[y1:y2, x1:x2].astype(np.float64)
    return patch.reshape(-1, 3).mean(axis=0)


# =============================================================================
# 2. α・burst 推定 (同時onsetグループの連立最小二乗)
# =============================================================================


def solve_group_alpha_burst(
    observed_list: list["np.ndarray"], true_list: list["np.ndarray"],
) -> "tuple[float, np.ndarray] | None":
    """observed_i = α×burst + (1-α)×true_i を全セル・全チャンネルで連立して解く。

    未知数 x=[Br,Bg,Bb,α] (B=α×burst)。3ch×N件の方程式を最小二乗で解く。
    N<2 (方程式が未知数4個に対して不足) の場合は None を返す。
    """
    n = len(observed_list)
    if n < 2:
        return None
    rows_a: list[list[float]] = []
    rows_y: list[float] = []
    for obs, true in zip(observed_list, true_list):
        for c in range(3):
            coef = [0.0, 0.0, 0.0, -float(true[c])]
            coef[c] = 1.0
            rows_a.append(coef)
            rows_y.append(float(obs[c] - true[c]))
    a_mat = np.array(rows_a)
    y_vec = np.array(rows_y)
    x, *_ = np.linalg.lstsq(a_mat, y_vec, rcond=None)
    b_vec, alpha = x[:3], float(x[3])
    if abs(alpha) < 1e-6:
        return alpha, None
    return alpha, b_vec / alpha


def solve_alpha_given_burst(
    observed: "np.ndarray", true_bgr: "np.ndarray", burst_bgr: "np.ndarray",
) -> float:
    """burst 固定のもとで1セル分の α を1次元最小二乗で解く (単独セル用フォールバック)。"""
    d = burst_bgr - true_bgr
    y = observed - true_bgr
    denom = float(np.dot(d, d))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(y, d) / denom)


def estimate_group_burst(
    records: list["object"], observed_map: dict[int, "np.ndarray"],
) -> dict[str, GroupBurstEstimate]:
    """group_size>=2 の同時onsetグループごとに α・burst を連立推定する。"""
    by_group: dict[str, list["object"]] = defaultdict(list)
    for r in records:
        by_group[r.group_key].append(r)
    results: dict[str, GroupBurstEstimate] = {}
    for key, group in by_group.items():
        if len(group) < 2:
            continue
        obs = [observed_map[id(r)] for r in group if id(r) in observed_map]
        true = [color_reference_bgr(r.correct_value) for r in group if id(r) in observed_map]
        solved = solve_group_alpha_burst(obs, true)
        if solved is None:
            continue
        alpha, burst = solved
        valid = (
            burst is not None
            and VALID_BURST_ALPHA_MIN <= alpha <= VALID_BURST_ALPHA_MAX
            and bool(np.all(burst >= VALID_BURST_BGR_MIN) and np.all(burst <= VALID_BURST_BGR_MAX))
        )
        results[key] = GroupBurstEstimate(
            video=group[0].video, group_key=key, n_cells=len(group),
            alpha=alpha, burst_bgr=burst, valid=valid,
        )
    return results


def aggregate_burst_fallback(
    group_estimates: dict[str, GroupBurstEstimate],
) -> tuple[dict[str, "np.ndarray"], "np.ndarray | None"]:
    """有効なグループ推定から動画別・全体平均の burst_bgr を作る (単独セル用)。"""
    by_video: dict[str, list["np.ndarray"]] = defaultdict(list)
    for est in group_estimates.values():
        if est.valid and est.burst_bgr is not None:
            by_video[est.video].append(est.burst_bgr)
    per_video = {v: np.mean(vals, axis=0) for v, vals in by_video.items()}
    all_vals = [v for vals in by_video.values() for v in vals]
    global_burst = np.mean(all_vals, axis=0) if all_vals else None
    return per_video, global_burst


# =============================================================================
# 3. 観測画素抽出 (onset+0.2秒フレームを再抽出)
# =============================================================================


def extract_observed_bgr(
    records: list["object"], cap_cache: dict, fps_cache: dict,
) -> dict[int, "np.ndarray"]:
    """全セルの onset+0.2秒時点の平均BGRを抽出する (build_error_onset_sheet の中央コマ相当)。"""
    observed: dict[int, "np.ndarray"] = {}
    for r in records:
        if r.onset_t_sec is None:
            continue
        t = r.onset_t_sec + _SHEET.CAPTURE_POST_ONSET_SEC
        frame = _DIAG._read_frame_at(cap_cache[r.video], fps_cache[r.video], max(t, 0.0))
        if frame is None:
            continue
        region = _SHEET._region_for_side(r.side)
        observed[id(r)] = _patch_mean_bgr(frame, region, r.row, r.col)
    return observed


# =============================================================================
# 4. unmix 復元 + 再分類テスト
# =============================================================================


def unmix_recover_bgr(
    observed: "np.ndarray", alpha: float, burst_bgr: "np.ndarray",
) -> "np.ndarray":
    """observed = α×burst + (1-α)×true から true を逆算する (α をクリップして数値安定化)。"""
    a = min(alpha, ALPHA_DIV_CLIP_MAX)
    return (observed - a * burst_bgr) / (1.0 - a)


def _reclassify_bgr(classifier: "ColorClassifier", bgr: "np.ndarray") -> int:
    """復元 BGR を均一パッチにして既存 ColorClassifier で再分類する (再実装しない)。"""
    clipped = np.clip(bgr, 0, 255).astype(np.uint8)
    patch = np.tile(clipped, (SYNTH_PATCH_SIZE, SYNTH_PATCH_SIZE, 1))
    return classifier.classify(patch)


def unmix_one_cell(
    r: "object", observed: "np.ndarray", alpha: float, burst_bgr: "np.ndarray",
    burst_source: str, classifier: "ColorClassifier",
) -> CellUnmixResult:
    """1セル分の unmix→再分類→成否判定を行う。"""
    if alpha >= ALPHA_INFO_LOST_MIN:
        status = STATUS_INFO_LOST
        reclass = None
    else:
        recovered = unmix_recover_bgr(observed, alpha, burst_bgr)
        reclass = _reclassify_bgr(classifier, recovered)
        status = STATUS_SUCCESS if reclass == r.correct_value else STATUS_MODEL_MISMATCH
    return CellUnmixResult(
        video=r.video, side=r.side, row=r.row, col=r.col,
        wrong_value=r.wrong_value, correct_value=r.correct_value,
        alpha=alpha, burst_source=burst_source, reclass_value=reclass, status=status,
    )


# =============================================================================
# 5. 負例チェック (バースト前フレームへの無条件適用)
# =============================================================================


def _value_at_or_before(
    timeline: list[tuple[int, float, int]], t_sec: float,
) -> "int | None":
    """timeline ((frame_idx, t_sec, value) の frame_idx 昇順リスト) から t_sec 直前の値を返す。"""
    candidates = [v for _fi, t, v in timeline if t <= t_sec]
    return candidates[-1] if candidates else None


def check_negative_case(
    r: "object", anchor_idx: "object", alpha: float, burst_bgr: "np.ndarray",
    cap_cache: dict, fps_cache: dict, classifier: "ColorClassifier",
) -> "bool | None":
    """onset-2秒時点が既に correct_value なら、同じ unmix を適用して誤変換するか調べる。

    Returns:
        True = 誤変換 (過剰適用の危険)、False = 変換しても正解のまま、
        None = onset-2秒時点で timeline 上まだ correct_value でない (適用対象外)。
    """
    if r.onset_t_sec is None:
        return None
    neg_t = r.onset_t_sec + NEGATIVE_CHECK_OFFSET_SEC
    if neg_t < 0:
        return None
    timeline = _DIAG._find_value_timeline(anchor_idx, r.side, r.game_idx, r.row, r.col)
    value_at_neg_t = _value_at_or_before(timeline, neg_t)
    if value_at_neg_t != r.correct_value:
        return None  # まだバースト前に正しい値になっていない → 検査対象外
    frame = _DIAG._read_frame_at(cap_cache[r.video], fps_cache[r.video], neg_t)
    if frame is None:
        return None
    region = _SHEET._region_for_side(r.side)
    observed = _patch_mean_bgr(frame, region, r.row, r.col)
    a = min(alpha, ALPHA_DIV_CLIP_MAX)
    recovered = unmix_recover_bgr(observed, a, burst_bgr)
    reclass = _reclassify_bgr(classifier, recovered)
    return reclass != r.correct_value


# =============================================================================
# 6. キャッシュ構築
# =============================================================================


def build_caches(records: list["object"]) -> tuple[dict, dict, dict]:
    """動画ごとに anchor_idx / cv2.VideoCapture / fps をキャッシュする。"""
    anchor_cache: dict[str, "object"] = {}
    cap_cache: dict[str, "cv2.VideoCapture"] = {}
    fps_cache: dict[str, float] = {}
    for r in records:
        if r.video in anchor_cache:
            continue
        npz_path = _MC.ANCHOR_NPZ_DIR / f"{r.video}.npz"
        anchor_cache[r.video] = _MC._load_npz_index(npz_path)
        video_path = _DIAG.VIDEO_DIR / f"video_{r.video}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        cap_cache[r.video] = cap
        fps_cache[r.video] = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return anchor_cache, cap_cache, fps_cache


def release_caps(cap_cache: dict) -> None:
    """VideoCapture を全て解放する。"""
    for cap in cap_cache.values():
        cap.release()


# =============================================================================
# 7. メイン診断ループ
# =============================================================================


def resolve_burst_for_record(
    r: "object", observed_map: dict[int, "np.ndarray"],
    group_estimates: dict[str, GroupBurstEstimate],
    per_video_fallback: dict[str, "np.ndarray"], global_fallback: "np.ndarray | None",
) -> "tuple[float, np.ndarray, str] | None":
    """1セル分の (α, burst_bgr, 推定方式) を確定する。データ不足なら None。"""
    if id(r) not in observed_map:
        return None
    est = group_estimates.get(r.group_key)
    if est is not None and est.valid and est.burst_bgr is not None:
        return est.alpha, est.burst_bgr, "joint"
    burst = per_video_fallback.get(r.video, global_fallback)
    if burst is None:
        return None
    true_bgr = color_reference_bgr(r.correct_value)
    alpha = solve_alpha_given_burst(observed_map[id(r)], true_bgr, burst)
    return alpha, burst, "fallback"


def run_probe(records: list["object"]) -> tuple[list[CellUnmixResult], dict]:
    """全93セルについて unmix→再分類→負例チェックまで一括実行する。"""
    anchor_cache, cap_cache, fps_cache = build_caches(records)
    observed_map = extract_observed_bgr(records, cap_cache, fps_cache)
    group_estimates = estimate_group_burst(records, observed_map)
    per_video_fallback, global_fallback = aggregate_burst_fallback(group_estimates)
    classifier = ColorClassifier()

    results: list[CellUnmixResult] = []
    for r in records:
        resolved = resolve_burst_for_record(
            r, observed_map, group_estimates, per_video_fallback, global_fallback,
        )
        if resolved is None:
            results.append(CellUnmixResult(
                video=r.video, side=r.side, row=r.row, col=r.col,
                wrong_value=r.wrong_value, correct_value=r.correct_value,
                alpha=None, burst_source="none", reclass_value=None,
                status=STATUS_NO_DATA,
            ))
            continue
        alpha, burst, source = resolved
        res = unmix_one_cell(r, observed_map[id(r)], alpha, burst, source, classifier)
        res.neg_check_flipped = check_negative_case(
            r, anchor_cache[r.video], alpha, burst, cap_cache, fps_cache, classifier,
        )
        results.append(res)
    release_caps(cap_cache)
    return results, group_estimates


# =============================================================================
# 8. 集計レポート
# =============================================================================


def report_alpha_distribution(results: list[CellUnmixResult]) -> str:
    """α分布 (min/median/max) を報告する。"""
    alphas = [r.alpha for r in results if r.alpha is not None]
    if not alphas:
        return "α分布: データなし"
    arr = np.array(alphas)
    return (
        f"α分布 (n={len(arr)}): min={arr.min():.3f} median={np.median(arr):.3f} "
        f"max={arr.max():.3f} mean={arr.mean():.3f}"
    )


def report_burst_colors(group_estimates: dict[str, GroupBurstEstimate]) -> str:
    """有効な burst 色推定の一覧 (5色仮説の検証材料)。"""
    lines = ["--- burst色推定 (同時onsetグループの連立最小二乗、有効のみ) ---"]
    valid = [e for e in group_estimates.values() if e.valid]
    lines.append(f"有効グループ数: {len(valid)}/{len(group_estimates)}")
    for e in sorted(valid, key=lambda e: e.video):
        # uint8 キャスト前に [0,255] へクリップする (VALID_BURST_BGR_MAX=285 まで
        # 許容しているため、素の cast だと 256 以上が折り返り HSV 表示が壊れる)。
        clipped = np.clip(e.burst_bgr, 0, 255).astype(np.uint8)
        b, g, r = clipped
        hsv = cv2.cvtColor(clipped.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
        lines.append(
            f"  {e.video} {e.group_key} n={e.n_cells} α={e.alpha:.3f} "
            f"burst_BGR=({b},{g},{r}) burst_HSV={tuple(int(x) for x in hsv)}"
        )
    return "\n".join(lines)


def report_reclassification(results: list[CellUnmixResult]) -> str:
    """復元成功/情報消失/モデル不整合の内訳。"""
    lines = ["--- unmix後の再分類結果 ---"]
    n = len(results)
    for status in (STATUS_SUCCESS, STATUS_INFO_LOST, STATUS_MODEL_MISMATCH, STATUS_NO_DATA):
        matched = [r for r in results if r.status == status]
        lines.append(f"{status}: {len(matched)}/{n} 件")
    return "\n".join(lines)


def report_negative_check(results: list[CellUnmixResult]) -> str:
    """負例チェック (バースト前フレームへの無条件適用) の誤変換率。"""
    checked = [r for r in results if r.neg_check_flipped is not None]
    flipped = [r for r in checked if r.neg_check_flipped]
    if not checked:
        return "負例チェック: 対象0件 (onset-2秒時点で timeline 上 correct_value の行なし)"
    return (
        f"負例チェック: 対象{len(checked)}件中 誤変換{len(flipped)}件 "
        f"({len(flipped) / len(checked) * 100:.1f}%)"
    )


# =============================================================================
# 9. main
# =============================================================================


def main() -> None:
    records = _SHEET.diagnose_all_samples()
    print(f"[1/3] 診断対象: {len(records)} 件 (batch1+batch2 fixed 盤面の全誤りセル)")

    results, group_estimates = run_probe(records)
    print("\n[2/3] α・burst推定")
    print(report_alpha_distribution(results))
    print(report_burst_colors(group_estimates))

    print("\n[3/3] 再分類テスト + 負例チェック")
    print(report_reclassification(results))
    print(report_negative_check(results))


if __name__ == "__main__":
    main()

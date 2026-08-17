"""タスク#7 (W3対処): れんさテロップテンプレの複数動画採取ブートストラップ。

## 背景

`models/ui_templates/chain_count_digits/digit_N.png` は単一動画
(video_c54、一部 video_c11) 採取のため、他動画では通用しない疑いがある
(docs/KNOWN_WEAKNESSES.md W3、C-1a 解決率0%)。本スクリプトは、テロップに
依存しない得点逆算の高信頼帯 (`src.chain_count_truth.
select_chain_count_high_confidence_band`) で「最終連鎖数がほぼ確実にNである」
と分かっているイベントを手がかりに、実際の動画フレームから「N れんさ!」
ポップアップの実クロップを採取し、追加テンプレ候補 (`digit_N_src_<video_id>.png`
相当) を作る。

## 採取方式 (低閾値ピーク探索)

対象イベントの窓 [t_center-CAPTURE_WINDOW_BEFORE_SEC,
t_center+CAPTURE_WINDOW_AFTER_SEC] を `CAPTURE_SAMPLE_INTERVAL_SEC` 間隔で
サンプリングし、既存の単一ソーステンプレ (`digit_N.png`) を使って
`cv2.matchTemplate` のピークスコアが最大になるフレーム・位置を探す
(閾値なし、ピーク位置を仮の採取候補として使うだけ)。既存テンプレが
他動画でも大まかな位置の手がかりとして機能するかどうか自体が未知数のため、
本スクリプトはピークスコアも正直に記録し、閾値判定は行わない
(採否は user レビュー、`docs/KNOWN_WEAKNESSES.md` 運用ルール参照)。

## 出力先

採取した候補クロップは **models/ui_templates/ に直接書き込まない**
(未検証のテンプレを本番資産に混ぜないため)。
`data/verify/chain_count_v2_2026-08-14/candidate_templates/` に
`digit_{n}_src_{video_id}_g{game_idx}.png` として保存し、採用判定は別途
user レビュー (タスク#7 item 3 のレビューシートと合わせて実施) を経る。

## 使い方
    python -m scripts._capture_chain_telop_templates_multisource_2026-08-14
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.chain_count_ocr import (
    CHAIN_DIGIT_LABELS,
    DEFAULT_CHAIN_TEMPLATE_DIR,
    Side,
    _crop_search_roi,
    _to_gray,
)
from src.chain_count_truth import compute_telop_search_window

_spec = importlib.util.spec_from_file_location(
    "_review20_for_capture",
    Path(__file__).resolve().parent / "_build_review20_chain_count_v2_2026-08-14.py",
)
assert _spec is not None and _spec.loader is not None
_review20_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_review20_mod)

DEFAULT_OUT_DIR = Path("data/verify/chain_count_v2_2026-08-14/candidate_templates")
# WSL 上の ~/frames/ (16動画・削除禁止資産、CLAUDE.md/MEMORY.md 記載) を既定にする。
# 本スクリプトは WSL 上の venv から実行する前提 (CLAUDE.md プロセス管理ルール)。
DEFAULT_VIDEO_DIR = Path.home() / "frames"

# 採取窓は `compute_telop_search_window` (src/chain_count_truth.py) に一元化。
# 【2026-08-14 続行タスクで撤回】旧実装は CAPTURE_WINDOW_BEFORE_SEC=3.0 /
# _AFTER_SEC=1.0 という t_center (発火タグ行時刻) 基準の固定窓を使っていたが、
# 実測で「発火前盤面行 (before_t_sec) 〜発火タグ行 (trigger_sec)」の自然な
# 区間そのものを使う方が正確・網羅的と判明したため、この区間 (+小さな
# 安全バッファ) に統一する。
CAPTURE_SAMPLE_INTERVAL_SEC: float = 0.05
# 採取候補として保存する最低スコア (閾値未満は junk crop の可能性が高く
# 保存しても user レビューの手間を増やすだけなので除外する)。
CAPTURE_MIN_SAVE_SCORE: float = 0.55
# 採取対象イベント数の上限 (review20 と同じ選定ロジックを再利用)。
CAPTURE_MAX_TARGETS: int = 20


@dataclass(frozen=True)
class CaptureTarget:
    """採取対象イベント1件 (高信頼帯の得点逆算で expected_n が既知)。"""
    video_id: str
    side: Side
    game_idx: int
    before_t_sec: float
    trigger_sec: float
    expected_n: int


@dataclass(frozen=True)
class CaptureResult:
    target: CaptureTarget
    peak_score: float
    peak_t_sec: float | None
    saved_path: str | None


def _load_primary_template_gray(label: int, template_dir: Path) -> np.ndarray | None:
    path = template_dir / f"digit_{label}.png"
    if not path.is_file():
        return None
    img = cv2.imread(str(path))
    if img is None:
        return None
    return _to_gray(img)


def capture_one(
    target: CaptureTarget,
    video_path: Path,
    template_dir: Path,
    out_dir: Path,
) -> CaptureResult:
    """1イベントについて、既存単一ソーステンプレでピーク位置を探し候補クロップを保存する。"""
    tpl_gray = _load_primary_template_gray(target.expected_n, template_dir)
    if tpl_gray is None:
        return CaptureResult(target, 0.0, None, None)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return CaptureResult(target, 0.0, None, None)

    best_score = -1.0
    best_loc: tuple[int, int] | None = None
    best_t: float | None = None
    best_roi: np.ndarray | None = None

    t, t_end = compute_telop_search_window(target.before_t_sec, target.trigger_sec)
    while t <= t_end:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            roi = _crop_search_roi(frame, target.side)
            if roi is not None and roi.size > 0:
                gray_roi = _to_gray(roi)
                if gray_roi.shape[0] >= tpl_gray.shape[0] and gray_roi.shape[1] >= tpl_gray.shape[1]:
                    res = cv2.matchTemplate(gray_roi, tpl_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val > best_score:
                        best_score = float(max_val)
                        best_loc = (int(max_loc[0]), int(max_loc[1]))
                        best_t = t
                        best_roi = roi
        t += CAPTURE_SAMPLE_INTERVAL_SEC
    cap.release()

    saved_path: str | None = None
    if best_loc is not None and best_roi is not None and best_score >= CAPTURE_MIN_SAVE_SCORE:
        x, y = best_loc
        h, w = tpl_gray.shape[:2]
        crop = best_roi[y:y + h, x:x + w]
        if crop.size > 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"digit_{target.expected_n}_src_{target.video_id}_g{target.game_idx}.png"
            out_path = out_dir / fname
            cv2.imwrite(str(out_path), crop)
            saved_path = str(out_path)

    return CaptureResult(target, max(0.0, best_score), best_t, saved_path)


def _build_targets_from_review20_events() -> tuple[CaptureTarget, ...]:
    """review20 と同じ選定ロジック (得点逆算高信頼帯) からイベントを取得する。

    2026-08-14 続行タスクでの窓修正 (`compute_telop_search_window`) を経て、
    手動選定した6件の固定リスト (旧実装) から動的選定に切り替える
    (成功窓で候補を広く取り直すため)。
    """
    events = _review20_mod._select_review_events()[:CAPTURE_MAX_TARGETS]
    return tuple(
        CaptureTarget(
            video_id=ev["video_id"], side=ev["side"], game_idx=ev["game_idx"],
            before_t_sec=ev["before_t_sec"], trigger_sec=ev["t_sec"],
            expected_n=ev["expected_n"],
        )
        for ev in events
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--template-dir", type=Path, default=DEFAULT_CHAIN_TEMPLATE_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    targets = _build_targets_from_review20_events()
    print(f"[capture] 採取対象イベント数={len(targets)}")

    results: list[dict] = []
    for target in targets:
        video_path = args.video_dir / f"video_{target.video_id}.mp4"
        if not video_path.is_file():
            print(f"[capture] SKIP {target.video_id}: 動画ファイル不在 ({video_path})")
            continue
        r = capture_one(target, video_path, args.template_dir, args.out_dir)
        print(f"[capture] {target.video_id} {target.side} g{target.game_idx} "
              f"expected_n={target.expected_n}: peak_score={r.peak_score:.3f} "
              f"peak_t={r.peak_t_sec} saved={r.saved_path}")
        results.append({
            "video_id": target.video_id,
            "side": target.side,
            "game_idx": target.game_idx,
            "expected_n": target.expected_n,
            "peak_score": r.peak_score,
            "peak_t_sec": r.peak_t_sec,
            "saved_path": r.saved_path,
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "_capture_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[capture] summary -> {summary_path}")


if __name__ == "__main__":
    main()

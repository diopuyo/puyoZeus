"""V3.2: 試合終了告知 (やった! / ばたんきゅー) 検出 + ロックダウン管理。

ぷよぷよe スポーツの試合終了時、勝者側に「(全消し) やった!」、敗者側に
「ばたんきゅー」のロゴ画像が数秒間表示される。この期間に CNN が盤面を
読みに行くと、ロゴ画像と背景の組み合わせで誤認識が発生する (m27 で実証)。

本モジュールは:
    1. テンプレート NCC マッチで終了告知を検出
    2. 検出時刻を記録、その時刻から lockdown_sec の間は「ロック中」フラグ
    3. 呼び出し側 (phase_u_render など) はロック中フレームの read_both_boards
       結果を採用せず、前盤面を保持する

設計上の利点:
    - matches.tsv に依存しない (動画単独で動作可能、リアルタイム対応)
    - テンプレート 1 枚 (やった or ばたんきゅーのいずれか) でも検出可能
    - 検出後に表示が消えても (テンプレマッチが切れても) 内部タイマーで
      lockdown_sec までロックを維持

ROI:
    - やった!: 1P 盤面中央 (x=180-700, y=150-520)
    - ばたんきゅー: 2P 盤面中央 (x=1230-1700, y=300-600)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# 既定テンプレートディレクトリ
DEFAULT_TEMPLATE_DIR: Path = Path("models/ui_templates")
# テンプレ名 prefix (match_end_*.png をすべて読み込む)
MATCH_END_PREFIX: str = "match_end_"

# NCC マッチ閾値 (0-1)
DEFAULT_NCC_THRESHOLD: float = 0.55

# 検出後ロックダウン秒数 (前盤面保持期間)
DEFAULT_LOCKDOWN_SEC: float = 5.0

# 探索領域 (1920x1080 基準、テンプレ位置の周囲を少し広めに探索)
# やった! は P1 盤面中央付近 (180-700, 150-520) に表示
SEARCH_P1: tuple[int, int, int, int] = (100, 100, 800, 600)  # (x, y, w, h)
# ばたんきゅー は P2 盤面中央 (1230-1700, 300-600) に表示
SEARCH_P2: tuple[int, int, int, int] = (1150, 200, 700, 500)


@dataclass(frozen=True)
class MatchEndDetectionResult:
    """1 フレーム検出結果。"""
    detected: bool
    template_name: str | None
    score: float


class MatchEndDetector:
    """試合終了告知検出 + 自動ロックダウン管理。

    使い方:
        detector = MatchEndDetector.load_default()
        for t_sec, frame in frames:
            detector.update(frame, t_sec)
            if detector.is_locked(t_sec):
                # 前盤面保持
                pass
    """

    def __init__(
        self,
        templates: dict[str, np.ndarray],
        threshold: float = DEFAULT_NCC_THRESHOLD,
        lockdown_sec: float = DEFAULT_LOCKDOWN_SEC,
    ) -> None:
        self._templates = templates
        self._threshold = threshold
        self._lockdown_sec = lockdown_sec
        self._last_detected_t: float | None = None

    @classmethod
    def load_default(
        cls,
        template_dir: Path = DEFAULT_TEMPLATE_DIR,
        threshold: float = DEFAULT_NCC_THRESHOLD,
        lockdown_sec: float = DEFAULT_LOCKDOWN_SEC,
    ) -> "MatchEndDetector":
        """既定ディレクトリから match_end_*.png を読み込む。"""
        templates: dict[str, np.ndarray] = {}
        if template_dir.exists():
            for p in sorted(template_dir.glob(f"{MATCH_END_PREFIX}*.png")):
                img = cv2.imread(str(p))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                templates[p.stem] = gray
        return cls(
            templates=templates,
            threshold=threshold,
            lockdown_sec=lockdown_sec,
        )

    def detect(self, frame_bgr: np.ndarray) -> MatchEndDetectionResult:
        """1 フレームに対してテンプレートマッチ (検出のみ、状態更新せず)。"""
        if not self._templates or frame_bgr is None or frame_bgr.size == 0:
            return MatchEndDetectionResult(
                detected=False, template_name=None, score=0.0,
            )
        h, w = frame_bgr.shape[:2]

        best_name: str | None = None
        best_score: float = -1.0
        for name, tmpl in self._templates.items():
            # テンプレ名から探索領域決定 (yatta → P1, batan → P2)
            if "batan" in name:
                sx, sy, sw, sh = SEARCH_P2
            else:
                sx, sy, sw, sh = SEARCH_P1
            # 画像範囲で clamp
            x1 = max(0, min(sx, w - 1))
            y1 = max(0, min(sy, h - 1))
            x2 = max(x1 + 1, min(sx + sw, w))
            y2 = max(y1 + 1, min(sy + sh, h))
            roi = frame_bgr[y1:y2, x1:x2]
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            tH, tW = tmpl.shape[:2]
            if roi_gray.shape[0] < tH or roi_gray.shape[1] < tW:
                continue
            result = cv2.matchTemplate(roi_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = float(max_val)
                best_name = name

        return MatchEndDetectionResult(
            detected=best_score >= self._threshold,
            template_name=best_name,
            score=best_score,
        )

    def update(self, frame_bgr: np.ndarray, t_sec: float) -> bool:
        """フレーム検出 + 状態更新。検出されたら _last_detected_t を t_sec に。

        Returns:
            bool: 現在 (= t_sec 時点で) ロック中か。
        """
        result = self.detect(frame_bgr)
        if result.detected:
            self._last_detected_t = t_sec
        return self.is_locked(t_sec)

    def is_locked(self, t_sec: float) -> bool:
        """指定時刻でロック中か (最後の検出から lockdown_sec 以内)。"""
        if self._last_detected_t is None:
            return False
        return (t_sec - self._last_detected_t) <= self._lockdown_sec

    def reset(self) -> None:
        """内部タイマーをリセット (試合間で呼ぶ場合)。"""
        self._last_detected_t = None

    @property
    def last_detected_t(self) -> float | None:
        return self._last_detected_t


__all__ = [
    "DEFAULT_LOCKDOWN_SEC",
    "DEFAULT_NCC_THRESHOLD",
    "DEFAULT_TEMPLATE_DIR",
    "MATCH_END_PREFIX",
    "MatchEndDetectionResult",
    "MatchEndDetector",
    "SEARCH_P1",
    "SEARCH_P2",
]

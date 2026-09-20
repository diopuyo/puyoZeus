"""NextValidator: next/dnext 検出の自己整合性検査.

ロジック:
    1. 配置 trace:
       state==STABLE 時の next pair を記録 → TSUMO_FALL → 次 STABLE で
       盤面の色 count delta が next pair と一致するはず。
    2. 連続性:
       直前 dnext == 現 next (ツモが進む = dnext が next に slide) であるはず。

擬似ラベル形式:
    component="next"
    input_data={"patch_top": ndarray, "patch_bot": ndarray, "side": "1P"/"2P"}
    label=(top_color, bot_color)
    confidence=0.95+
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.board import (
    BOARD_COLS,
    BOARD_ROWS,
    COLOR_EMPTY,
    COLOR_OJAMA,
    COLOR_UNKNOWN,
    Board,
)
from src.board_state_machine import BoardState
from src.next_detector import (
    ROI_1P_NEXT_BOT,
    ROI_1P_NEXT_TOP,
    ROI_2P_NEXT_BOT,
    ROI_2P_NEXT_TOP,
)
from src.self_supervised.cross_validator import CrossValidator
from src.self_supervised.pseudo_label import (
    COMPONENT_NEXT,
    PseudoLabelSample,
)


# 配置 delta 一致判定の confidence
PLACEMENT_MATCH_CONFIDENCE: float = 0.95
# 連続性 (dnext→next) 一致の confidence
CONTINUITY_MATCH_CONFIDENCE: float = 0.92
# 直近フレーム保持数
NEXT_HISTORY_WINDOW: int = 8

# Phase I 改良: STABLE 中の next_pair が連続観測 → 高確度として emit
# 5 fps サンプリングだと placement_trace / continuity が稀にしか発火しないため、
# STABLE 中の安定性ベースで MEDIUM 信頼の擬似ラベルも収集する。
STABLE_AGREE_MIN: int = 3  # STABLE 中 (state==STABLE) で同 next_pair 連続観測数
STABLE_MATCH_CONFIDENCE: float = 0.85  # STABLE 安定 emit の信頼度


@dataclass
class _NextFrame:
    """1 frame の next 観測."""

    frame_idx: int
    t_sec: float
    state_1p: BoardState
    state_2p: BoardState
    confirmed_1p: Board | None
    confirmed_2p: Board | None
    next_pair_1p: tuple[int, int] | None
    next_pair_2p: tuple[int, int] | None
    dnext_pair_1p: tuple[int, int] | None
    dnext_pair_2p: tuple[int, int] | None
    next_patches_1p: dict[str, np.ndarray] | None = field(default=None)
    next_patches_2p: dict[str, np.ndarray] | None = field(default=None)


class NextValidator(CrossValidator):
    """next/dnext 検出の自己整合性検査."""

    def __init__(
        self,
        history_window: int = NEXT_HISTORY_WINDOW,
        stable_agree_min: int = STABLE_AGREE_MIN,
        enable_stable_emit: bool = True,
    ) -> None:
        super().__init__()
        if history_window < 3:
            raise ValueError("history_window must be >= 3")
        if stable_agree_min < 2:
            raise ValueError("stable_agree_min must be >= 2")
        self._history: deque[_NextFrame] = deque(maxlen=history_window)
        # 直前 STABLE の (board, next_pair, frame_idx) (side 別)
        self._stable_1p: tuple[Board, tuple[int, int], int, np.ndarray | None,
                               np.ndarray | None] | None = None
        self._stable_2p: tuple[Board, tuple[int, int], int, np.ndarray | None,
                               np.ndarray | None] | None = None
        # emit 済 frame_idx (重複防止)
        self._emitted: set[tuple[int, str]] = set()
        # 安定 emit 重複防止 (per side, per pair, 試合単位)
        self._stable_emitted: set[tuple[str, tuple[int, int]]] = set()
        self._stable_agree_min = int(stable_agree_min)
        self._enable_stable_emit = bool(enable_stable_emit)

    def reset(self) -> None:
        super().reset()
        self._history.clear()
        self._stable_1p = None
        self._stable_2p = None
        self._emitted.clear()
        self._stable_emitted.clear()

    def update(
        self,
        frame_idx: int,
        t_sec: float,
        pipeline_result: Any,
        frame_bgr: np.ndarray | None,
    ) -> None:
        """1 frame の更新."""
        if not getattr(pipeline_result, "is_match_active", False):
            return
        next_pair_1p = _safe_get_next_pair(pipeline_result.p1)
        next_pair_2p = _safe_get_next_pair(pipeline_result.p2)
        dnext_pair_1p = _safe_get_dnext_pair(pipeline_result.p1)
        dnext_pair_2p = _safe_get_dnext_pair(pipeline_result.p2)
        patches_1p = (
            _extract_next_patches(frame_bgr, "1P")
            if frame_bgr is not None else None
        )
        patches_2p = (
            _extract_next_patches(frame_bgr, "2P")
            if frame_bgr is not None else None
        )
        rec = _NextFrame(
            frame_idx=frame_idx,
            t_sec=t_sec,
            state_1p=pipeline_result.p1.state,
            state_2p=pipeline_result.p2.state,
            confirmed_1p=pipeline_result.p1.confirmed_board,
            confirmed_2p=pipeline_result.p2.confirmed_board,
            next_pair_1p=next_pair_1p,
            next_pair_2p=next_pair_2p,
            dnext_pair_1p=dnext_pair_1p,
            dnext_pair_2p=dnext_pair_2p,
            next_patches_1p=patches_1p,
            next_patches_2p=patches_2p,
        )
        self._history.append(rec)
        self._check_placement("1P", rec)
        self._check_placement("2P", rec)
        self._check_continuity(rec)
        if self._enable_stable_emit:
            self._check_stable_persistence(rec)

    # ------------------------------------------------------------------
    # placement trace: STABLE → TSUMO → STABLE で次盤面の delta 確認
    # ------------------------------------------------------------------

    def _check_placement(self, side: str, rec: _NextFrame) -> None:
        """STABLE 検出時、直前 STABLE の next_pair が今 STABLE への color delta と
        一致するかを確認、一致すれば擬似ラベル emit."""
        state = rec.state_1p if side == "1P" else rec.state_2p
        confirmed = rec.confirmed_1p if side == "1P" else rec.confirmed_2p
        next_pair = (
            rec.next_pair_1p if side == "1P" else rec.next_pair_2p
        )
        patches = (
            rec.next_patches_1p if side == "1P" else rec.next_patches_2p
        )
        if state != BoardState.STABLE or confirmed is None:
            return
        # 直前 STABLE と比較
        prev = self._stable_1p if side == "1P" else self._stable_2p
        if prev is None:
            # 初 STABLE: 記録のみ
            if next_pair is not None:
                self._record_stable(side, confirmed, next_pair, rec, patches)
            return
        prev_board, prev_next, prev_frame_idx, prev_patch_top, prev_patch_bot = prev
        # 同一 STABLE (盤面変化なし) は skip
        if _board_equal(prev_board, confirmed):
            return
        # delta 計算
        delta_counts = _color_count_delta(prev_board, confirmed)
        # delta が +1 puyo だけの色 2 種類になっているはず (横置き) または
        # 単一色 +2 (同色ペア縦/横)
        expected_counts = _expected_delta_from_pair(prev_next)
        match = _delta_matches(delta_counts, expected_counts)
        if match:
            # 高信頼擬似ラベル emit (prev_next の patch を正解と一致確認)
            key = (prev_frame_idx, side)
            if key not in self._emitted and prev_patch_top is not None:
                # patches は dict {next_top, next_bot}
                sample = PseudoLabelSample(
                    component=COMPONENT_NEXT,
                    timestamp=rec.t_sec,
                    input_data={
                        "patch_top": prev_patch_top.copy(),
                        "patch_bot": (
                            prev_patch_bot.copy()
                            if prev_patch_bot is not None else None
                        ),
                        "side": side,
                    },
                    label={
                        "top_color": int(prev_next[0]),
                        "bot_color": int(prev_next[1]),
                    },
                    confidence=PLACEMENT_MATCH_CONFIDENCE,
                    metadata={
                        "frame_idx": int(prev_frame_idx),
                        "verified_at_frame": int(rec.frame_idx),
                        "delta_counts": {
                            int(k): int(v) for k, v in delta_counts.items()
                        },
                        "source": "placement_trace_match",
                    },
                )
                self._emit(sample)
                self._emitted.add(key)
        else:
            # delta が clean だが next_pair と不一致 → next が misread の可能性
            # delta が exactly 2 puyo の場合のみ「正解 = delta から逆算」を emit
            recovered = _recover_pair_from_delta(delta_counts)
            if (
                recovered is not None
                and prev_patch_top is not None
            ):
                key = (prev_frame_idx, side)
                if key not in self._emitted:
                    sample = PseudoLabelSample(
                        component=COMPONENT_NEXT,
                        timestamp=rec.t_sec,
                        input_data={
                            "patch_top": prev_patch_top.copy(),
                            "patch_bot": (
                                prev_patch_bot.copy()
                                if prev_patch_bot is not None else None
                            ),
                            "side": side,
                        },
                        label={
                            "top_color": int(recovered[0]),
                            "bot_color": int(recovered[1]),
                        },
                        confidence=PLACEMENT_MATCH_CONFIDENCE,
                        metadata={
                            "frame_idx": int(prev_frame_idx),
                            "verified_at_frame": int(rec.frame_idx),
                            "predicted_next": [int(c) for c in prev_next],
                            "delta_counts": {
                                int(k): int(v) for k, v in delta_counts.items()
                            },
                            "source": "placement_trace_correct",
                        },
                    )
                    self._emit(sample)
                    self._emitted.add(key)
        # 新しい STABLE を記録
        if next_pair is not None and patches is not None:
            self._record_stable(side, confirmed, next_pair, rec, patches)

    def _record_stable(
        self,
        side: str,
        board: Board,
        next_pair: tuple[int, int],
        rec: _NextFrame,
        patches: dict[str, np.ndarray] | None,
    ) -> None:
        """直前 STABLE 情報を記録."""
        patch_top = (
            patches.get("next_top") if patches is not None else None
        )
        patch_bot = (
            patches.get("next_bot") if patches is not None else None
        )
        record = (
            board.copy(), next_pair, rec.frame_idx, patch_top, patch_bot,
        )
        if side == "1P":
            self._stable_1p = record
        else:
            self._stable_2p = record

    # ------------------------------------------------------------------
    # STABLE 持続性: STABLE 中に同一 next_pair が連続観測 → MEDIUM emit
    # ------------------------------------------------------------------

    def _check_stable_persistence(self, rec: _NextFrame) -> None:
        """STABLE 中に同一 next_pair が一定 frame 連続観測されたら emit.

        placement_trace / continuity が稀にしか発火しないサンプリング条件
        (5fps) でも、STABLE が長く続けば次ツモ表示は安定しているはずなので、
        その安定区間で next_detector の出力を高確度として保存する。
        """
        self._emit_stable_for_side("1P", rec)
        self._emit_stable_for_side("2P", rec)

    def _emit_stable_for_side(self, side: str, rec: _NextFrame) -> None:
        """1 side の STABLE 持続性 emit."""
        next_pair = rec.next_pair_1p if side == "1P" else rec.next_pair_2p
        if next_pair is None:
            return
        # 直近 stable_agree_min 件の frame で
        # (state==STABLE かつ next_pair==同値) を要求
        if len(self._history) < self._stable_agree_min:
            return
        recent = list(self._history)[-self._stable_agree_min:]
        for f in recent:
            f_state = f.state_1p if side == "1P" else f.state_2p
            f_pair = f.next_pair_1p if side == "1P" else f.next_pair_2p
            if f_state != BoardState.STABLE or f_pair != next_pair:
                return
        # 重複防止: 1 試合 (= reset まで) で同 (side, pair) は 1 回だけ emit
        dedup_key = (side, next_pair)
        if dedup_key in self._stable_emitted:
            return
        patches = (
            rec.next_patches_1p if side == "1P" else rec.next_patches_2p
        )
        if patches is None:
            return
        patch_top = patches.get("next_top")
        patch_bot = patches.get("next_bot")
        if patch_top is None:
            return
        sample = PseudoLabelSample(
            component=COMPONENT_NEXT,
            timestamp=rec.t_sec,
            input_data={
                "patch_top": patch_top.copy(),
                "patch_bot": (
                    patch_bot.copy() if patch_bot is not None else None
                ),
                "side": side,
            },
            label={
                "top_color": int(next_pair[0]),
                "bot_color": int(next_pair[1]),
            },
            confidence=STABLE_MATCH_CONFIDENCE,
            metadata={
                "frame_idx": int(rec.frame_idx),
                "agree_count": int(self._stable_agree_min),
                "source": "stable_persistence",
            },
        )
        self._emit(sample)
        self._stable_emitted.add(dedup_key)

    # ------------------------------------------------------------------
    # 連続性: 直前 dnext == 現 next
    # ------------------------------------------------------------------

    def _check_continuity(self, rec: _NextFrame) -> None:
        """直前 frame の dnext と現 frame の next が一致するか.

        ツモ消費 1 回ごとに dnext → next にスライドするので、
        STABLE → TSUMO_FALL の境界 frame でスライド発生。
        """
        if len(self._history) < 2:
            return
        prev = self._history[-2]
        # 1P 側
        self._emit_continuity_for(prev, rec, "1P")
        self._emit_continuity_for(prev, rec, "2P")

    def _emit_continuity_for(
        self, prev: _NextFrame, cur: _NextFrame, side: str,
    ) -> None:
        """1 side の連続性チェック."""
        prev_dnext = (
            prev.dnext_pair_1p if side == "1P" else prev.dnext_pair_2p
        )
        cur_next = cur.next_pair_1p if side == "1P" else cur.next_pair_2p
        if prev_dnext is None or cur_next is None:
            return
        # 連続性が成立する遷移: state が STABLE→TSUMO_FALL に移った瞬間
        # 簡略: dnext と next が一致したら高信頼で patch ラベル化
        if prev_dnext != cur_next:
            return
        cur_patches = (
            cur.next_patches_1p if side == "1P" else cur.next_patches_2p
        )
        if cur_patches is None:
            return
        key = (cur.frame_idx, side)
        if key in self._emitted:
            return
        patch_top = cur_patches.get("next_top")
        patch_bot = cur_patches.get("next_bot")
        if patch_top is None:
            return
        sample = PseudoLabelSample(
            component=COMPONENT_NEXT,
            timestamp=cur.t_sec,
            input_data={
                "patch_top": patch_top.copy(),
                "patch_bot": patch_bot.copy() if patch_bot is not None else None,
                "side": side,
            },
            label={
                "top_color": int(cur_next[0]),
                "bot_color": int(cur_next[1]),
            },
            confidence=CONTINUITY_MATCH_CONFIDENCE,
            metadata={
                "frame_idx": int(cur.frame_idx),
                "prev_frame_idx": int(prev.frame_idx),
                "source": "continuity_match",
            },
        )
        self._emit(sample)
        self._emitted.add(key)


# ============================
# helpers
# ============================


def _safe_get_next_pair(side_result: Any) -> tuple[int, int] | None:
    """SideResult から next_pair を取得 (state context 経由)."""
    # SideResult には next_pair が直接無いので、内部 state machine の
    # next_queue から最後を取る…ただし SideResult には公開されていない。
    # 当面は metadata.get("next_pair") を許容、なければ None。
    np_attr = getattr(side_result, "next_pair", None)
    if np_attr is not None and len(np_attr) == 2:
        return (int(np_attr[0]), int(np_attr[1]))
    return None


def _safe_get_dnext_pair(side_result: Any) -> tuple[int, int] | None:
    """SideResult から dnext_pair を取得."""
    dn = getattr(side_result, "dnext_pair", None)
    if dn is not None and len(dn) == 2:
        return (int(dn[0]), int(dn[1]))
    return None


def _extract_next_patches(
    frame: np.ndarray, side: str,
) -> dict[str, np.ndarray] | None:
    """next ROI からパッチを切り出し.

    入力が 1080p 以外の 16:9 解像度 (例: 720p) の場合は 1080p
    にリサイズして ROI 適用する。
    """
    if frame is None or frame.ndim != 3:
        return None
    h, w = frame.shape[:2]
    if (h, w) != (1080, 1920):
        scaled = _ensure_1080p_frame(frame)
        if scaled is None:
            return None
        frame = scaled
        h, w = frame.shape[:2]
    if side == "1P":
        rois = {
            "next_top": ROI_1P_NEXT_TOP,
            "next_bot": ROI_1P_NEXT_BOT,
        }
    else:
        rois = {
            "next_top": ROI_2P_NEXT_TOP,
            "next_bot": ROI_2P_NEXT_BOT,
        }
    out: dict[str, np.ndarray] = {}
    for key, roi in rois.items():
        y1, y2, x1, x2 = roi
        if y2 > h or x2 > w:
            continue
        out[key] = frame[y1:y2, x1:x2].copy()
    return out if out else None


def _ensure_1080p_frame(frame: np.ndarray) -> np.ndarray | None:
    """16:9 フレームを 1080p (1920x1080) にリサイズ. 16:9 以外は None."""
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return None
    if abs(w * 9 - h * 16) > max(w, h):
        return None
    try:
        import cv2
        return cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    except Exception:
        return None


def _board_equal(a: Board, b: Board) -> bool:
    """盤面が完全に等しいか."""
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if a.get(r, c) != b.get(r, c):
                return False
    return True


def _color_count_delta(prev: Board, cur: Board) -> dict[int, int]:
    """色別 count の delta を返す (cur - prev)."""
    prev_counts: dict[int, int] = {}
    cur_counts: dict[int, int] = {}
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            pv = int(prev.get(r, c))
            cv = int(cur.get(r, c))
            if pv != COLOR_EMPTY and pv != COLOR_UNKNOWN:
                prev_counts[pv] = prev_counts.get(pv, 0) + 1
            if cv != COLOR_EMPTY and cv != COLOR_UNKNOWN:
                cur_counts[cv] = cur_counts.get(cv, 0) + 1
    delta: dict[int, int] = {}
    for color in set(prev_counts) | set(cur_counts):
        d = cur_counts.get(color, 0) - prev_counts.get(color, 0)
        if d != 0:
            delta[color] = d
    return delta


def _expected_delta_from_pair(pair: tuple[int, int]) -> dict[int, int]:
    """next pair から期待 delta を算出 (置く puyo 2 個分)."""
    expected: dict[int, int] = {}
    for c in pair:
        expected[c] = expected.get(c, 0) + 1
    return expected


def _delta_matches(
    actual: dict[int, int], expected: dict[int, int],
) -> bool:
    """actual delta が expected と完全一致 (おじゃま列除外して比較)."""
    # actual がぴったり expected (2 puyo 分の +) になっているか
    # 色は puyo 5 色のみ (1..5)、おじゃま (9) や UNKNOWN (10) は除外
    a_filtered = {
        k: v for k, v in actual.items()
        if k not in (COLOR_OJAMA, COLOR_UNKNOWN, COLOR_EMPTY)
    }
    e_filtered = {
        k: v for k, v in expected.items()
        if k not in (COLOR_OJAMA, COLOR_UNKNOWN, COLOR_EMPTY)
    }
    return a_filtered == e_filtered


def _recover_pair_from_delta(
    delta: dict[int, int],
) -> tuple[int, int] | None:
    """delta から next pair を逆算 (= 期待 delta 形式の場合のみ)."""
    pos_only = {
        k: v for k, v in delta.items()
        if v > 0 and k not in (COLOR_OJAMA, COLOR_UNKNOWN, COLOR_EMPTY)
    }
    total = sum(pos_only.values())
    if total != 2:
        return None
    if len(pos_only) == 1:
        # 同色 ペア (+2)
        c = next(iter(pos_only))
        return (c, c)
    if len(pos_only) == 2:
        cs = sorted(pos_only.keys())
        return (cs[0], cs[1])
    return None


__all__ = [
    "CONTINUITY_MATCH_CONFIDENCE",
    "NEXT_HISTORY_WINDOW",
    "NextValidator",
    "PLACEMENT_MATCH_CONFIDENCE",
    "STABLE_AGREE_MIN",
    "STABLE_MATCH_CONFIDENCE",
]

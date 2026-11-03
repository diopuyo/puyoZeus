"""本体とB3の読取条件を照合し、当該呼出しの生得点だけを共有する。"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from src.recog_speed_py import ITEM_DISPLAY_SCORES, STATS, audit_enabled, enabled
from src.score_ocr import ScoreOcr


@dataclass(frozen=True)
class RawScores:
    score_1p: int | None
    score_2p: int | None


def prepare(owner: Any) -> None:
    """早期returnや同一indexの再投入でも前回の値を使わない。"""
    owner._live_share_scores = os.environ.get('PUYO_B3_SHARE_SCORE_READ') == '1'
    if owner._live_share_scores:
        owner._live_raw_scores = None


def compatible(left: Any, right: Any) -> bool:
    """未知の派生OCRは共有しない。既知実装の可変設定も毎回照合する。"""
    if type(left) is not ScoreOcr or type(right) is not ScoreOcr:
        return False
    names = ('_min_confidence', '_margin_min', '_avg_min_confidence',
             '_enable_matmul_ncc', '_region_offsets')
    if any(getattr(left, name) != getattr(right, name) for name in names):
        return False
    for name in ('_templates_gray', '_templates_mask'):
        a, b = getattr(left, name), getattr(right, name)
        if a.keys() != b.keys() or any(not np.array_equal(a[k], b[k]) for k in a):
            return False
    return True


def shared_scores(owner: Any, ocr: Any, frame: Any) -> RawScores | None:
    """欠測(None)も有効な生観測。ScoreTrackerの前回値は使わない。"""
    if not getattr(owner, '_live_share_scores', False):
        return None
    raw = getattr(owner, '_live_raw_scores', None)
    if raw is None or raw[:2] != (frame.index, frame.media_sec):
        return None
    if not compatible(getattr(owner, '_score_ocr', None), ocr):
        return None
    return RawScores(*raw[2:])


def displayed_scores(pipe: Any, frame: Any) -> tuple:
    """recognize() の表示得点 (1P, 2P)。従来は同じコマの得点をもう一度 OCR していた。

    live_recog_speed_py の display_scores が ON で、本体がこのコマで読んだ生得点が共有されている
    (b3_share_score_read が ON・同じコマ) ときだけ、それを使う。生得点は両側の ScoreTracker が
    owner._score_ocr.read_side(画像, 側)[0] で読んだ値で、従来の再読 (displayed_scores_from_pipeline) と
    同じ関数・同じ画像・同じ ROI 補正 (補正は update の先頭で確定し、このコマではもう変わらない) の結果なので
    値は同じ。tracker が読んだ OCR が owner._score_ocr と別物なら (差し替えられた場合) 使わない。
    使えない場合は従来どおり再読する。
    """
    from src.exchange_event_overlay import displayed_scores_from_pipeline
    if enabled(ITEM_DISPLAY_SCORES):
        owner = getattr(pipe, 'pipe', pipe)
        raw = getattr(owner, '_live_raw_scores', None)
        ocr = getattr(owner, '_score_ocr', None)
        trackers = (getattr(owner, '_score_tracker_1p', None), getattr(owner, '_score_tracker_2p', None))
        if (getattr(owner, '_live_share_scores', False) and raw is not None and ocr is not None
                and raw[:2] == (frame.index, frame.media_sec)
                and all(getattr(tracker, '_ocr', None) is ocr for tracker in trackers)):
            STATS['display_scores_shared'] += 1
            if audit_enabled():     # 検証用: 共有した値を、従来どおり再読した値と突き合わせて数える
                STATS['audit_checked'] += 1
                if tuple(displayed_scores_from_pipeline(pipe, frame.image)) != (raw[2], raw[3]):
                    STATS['audit_mismatch'] += 1
            return (raw[2], raw[3])
    return displayed_scores_from_pipeline(pipe, frame.image)

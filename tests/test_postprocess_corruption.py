"""後処理破壊検知 (postprocess_corruption) のユニットテスト。

合成ボードを使う純粋ユニットテスト (動画・pipeline 不要)。
アナリスト ↔ テスター合意 12 ケース + α を網羅。
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pytest

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.measure_stable_cell_acc import (
    CORRUPTION_LOG_LIMIT,
    POSTPROCESS_CORRUPTION_REJECT_RATE,
    POSTPROCESS_CORRUPTION_WARNING_RATE,
    POSTPROCESS_SIDE_BIAS_THRESHOLD,
    POSTPROCESS_SIDE_BIAS_MIN_CELLS,
    VideoStats,
    _aggregate_corruption,
    _check_postprocess_corruption,
    _judge_corruption_metrics,
    COLOR_NAMES,
)
from src.board import (
    COLOR_EMPTY,
    COLOR_RED,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_PURPLE,
    COLOR_UNKNOWN,
)

# ============================
# ヘルパー
# ============================


def _mk_stats(video_id: str = "vtest") -> VideoStats:
    """空の VideoStats を返す。"""
    return VideoStats(video_id=video_id, is_holdout=False)


def _call_check(
    stats: VideoStats,
    raw_cnn: int,
    raw_hsv: int,
    confirmed: int,
    side: str = "1P",
    row: int = 5,
    col: int = 2,
    fi: int = 0,
    t_sec: float = 1.0,
) -> None:
    """_check_postprocess_corruption の薄いラッパー (テスト用)。"""
    _check_postprocess_corruption(
        "vtest", fi, t_sec, side, row, col,
        raw_cnn, raw_hsv, confirmed, stats,
    )


# ============================
# ケース 1: 破壊検知 - CNN==HSV だが confirmed が異なる
# ============================

def test_corruption_detected_cnn_hsv_agree_confirmed_differs():
    """raw_cnn==raw_hsv==GREEN, confirmed==RED → count==1 かつ log に記録される。"""
    stats = _mk_stats()
    _call_check(stats, raw_cnn=COLOR_GREEN, raw_hsv=COLOR_GREEN, confirmed=COLOR_RED)
    assert stats.postprocess_corruption_count == 1
    assert len(stats.postprocess_corruption_log) == 1
    entry = stats.postprocess_corruption_log[0]
    assert entry["raw_cnn"] == "green"
    assert entry["confirmed"] == "red"
    assert entry["side"] == "1P"


# ============================
# ケース 2: 検知なし - 3 者全員一致
# ============================

def test_corruption_not_detected_all_agree():
    """3 者全員一致の場合は corruption_count==0。"""
    stats = _mk_stats()
    _call_check(stats, raw_cnn=COLOR_RED, raw_hsv=COLOR_RED, confirmed=COLOR_RED)
    assert stats.postprocess_corruption_count == 0
    assert len(stats.postprocess_corruption_log) == 0


# ============================
# ケース 3: 検知なし - CNN と HSV が不一致 (対象外)
# ============================

def test_corruption_not_detected_cnn_hsv_disagree():
    """CNN != HSV の場合は corruption 対象外 → count==0。"""
    stats = _mk_stats()
    _call_check(stats, raw_cnn=COLOR_RED, raw_hsv=COLOR_BLUE, confirmed=COLOR_RED)
    assert stats.postprocess_corruption_count == 0


# ============================
# ケース 4: 検知なし - UNKNOWN が含まれる
# ============================

def test_corruption_not_detected_unknown_present_raw_cnn():
    """raw_cnn が UNKNOWN の場合は corruption 対象外。"""
    stats = _mk_stats()
    _call_check(stats, raw_cnn=COLOR_UNKNOWN, raw_hsv=COLOR_UNKNOWN, confirmed=COLOR_RED)
    assert stats.postprocess_corruption_count == 0


def test_corruption_not_detected_unknown_present_confirmed():
    """confirmed が UNKNOWN の場合は corruption 対象外。"""
    stats = _mk_stats()
    _call_check(stats, raw_cnn=COLOR_GREEN, raw_hsv=COLOR_GREEN, confirmed=COLOR_UNKNOWN)
    assert stats.postprocess_corruption_count == 0


# ============================
# ケース 5: side_bias - 1P 側が GREEN に 7 件
# ============================

def test_side_bias_1p_green_7cells():
    """7 件すべて 1P → by_side['1P']==7 で集計される。"""
    stats = _mk_stats()
    for i in range(7):
        _call_check(stats, raw_cnn=COLOR_GREEN, raw_hsv=COLOR_GREEN, confirmed=COLOR_RED,
                    side="1P", row=i, fi=i)
    assert stats.postprocess_corruption_by_side["1P"] == 7
    assert stats.postprocess_corruption_by_side.get("2P", 0) == 0

    # _aggregate_corruption で side_bias 検知
    corruption = _aggregate_corruption([stats], total_stable_cells=7000)
    # dominant_side は 1P (7/7 = 100%)
    assert corruption["side_bias"]["detected"] is True
    assert corruption["side_bias"]["dominant_side"] == "1P"


# ============================
# ケース 6: side_bias 閾値 50% - ちょうど境界
# ============================

def test_side_bias_threshold_50pct():
    """5 件: 1P=3, 2P=2 → 1P の rate は 3/5=0.6 >= 0.50 → bias 検知。"""
    stats = _mk_stats()
    for i in range(3):
        _call_check(stats, raw_cnn=COLOR_GREEN, raw_hsv=COLOR_GREEN, confirmed=COLOR_RED,
                    side="1P", row=i, fi=i)
    for i in range(2):
        _call_check(stats, raw_cnn=COLOR_GREEN, raw_hsv=COLOR_GREEN, confirmed=COLOR_RED,
                    side="2P", row=i + 3, fi=i + 3)
    corruption = _aggregate_corruption([stats], total_stable_cells=5000)
    assert corruption["side_bias"]["detected"] is True
    assert corruption["side_bias"]["dominant_side"] == "1P"


# ============================
# ケース 7: log_limit - 300 件投入でも log は LIMIT 以内
# ============================

def test_corruption_log_limit():
    """300 件投入しても postprocess_corruption_log は CORRUPTION_LOG_LIMIT 以内。"""
    stats = _mk_stats()
    for i in range(300):
        _call_check(stats, raw_cnn=COLOR_GREEN, raw_hsv=COLOR_GREEN, confirmed=COLOR_RED,
                    row=i % 13, col=i % 6, fi=i)
    assert stats.postprocess_corruption_count == 300
    assert len(stats.postprocess_corruption_log) <= CORRUPTION_LOG_LIMIT


# ============================
# ケース 8: rate >= 0.1% で _judge_corruption_metrics が FAIL
# ============================

def test_corruption_rate_threshold_reject():
    """rate >= POSTPROCESS_CORRUPTION_REJECT_RATE (0.1%) で FAIL が返ること。"""
    corruption_section = {
        "rate": POSTPROCESS_CORRUPTION_REJECT_RATE,
        "side_bias": {"detected": False},
    }
    failures = _judge_corruption_metrics(corruption_section)
    assert len(failures) > 0
    assert any("REJECT" in f for f in failures)


# ============================
# ケース 9: rate==0 で PASS
# ============================

def test_corruption_rate_zero_pass():
    """rate==0 の場合は FAIL なし。"""
    corruption_section = {
        "rate": 0.0,
        "side_bias": {"detected": False},
    }
    failures = _judge_corruption_metrics(corruption_section)
    assert failures == []


# ============================
# ケース 10: side_bias.detected==True で FAIL
# ============================

def test_side_bias_rate_threshold_reject():
    """side_bias.detected==True の場合は FAIL が返ること。"""
    corruption_section = {
        "rate": 0.0,
        "side_bias": {
            "detected": True,
            "dominant_side": "2P",
            "dominant_color": "green",
            "dominant_rate": 0.8,
        },
    }
    failures = _judge_corruption_metrics(corruption_section)
    assert len(failures) > 0
    assert any("side_bias" in f for f in failures)


# ============================
# ケース 11: corruption_ratio - fix=10, corruption=0 → ratio==0.0
# ============================

def test_corruption_ratio_symmetric():
    """physics_fix=10, corruption=0 → corruption_ratio==0.0。"""
    stats = _mk_stats()
    # physics_fix はここでは手動で設定
    stats.physics_fix_count = 10
    corruption = _aggregate_corruption([stats], total_stable_cells=1000)
    assert corruption["corruption_ratio"] == 0.0


# ============================
# ケース 12: corruption_ratio - fix=3, corruption=7 → ratio>0.5 (ネット負)
# ============================

def test_corruption_ratio_net_negative():
    """physics_fix=3, corruption=7 → corruption_ratio=7/10=0.7 > 0.5 (ネット負)。"""
    stats = _mk_stats()
    stats.physics_fix_count = 3
    # corruption を 7 件投入
    for i in range(7):
        _call_check(stats, raw_cnn=COLOR_GREEN, raw_hsv=COLOR_GREEN, confirmed=COLOR_RED,
                    row=i % 13, col=i % 6, fi=i)
    corruption = _aggregate_corruption([stats], total_stable_cells=1000)
    assert corruption["corruption_ratio"] > 0.5


# ============================
# 追加ケース: VideoStats の新フィールドが後方互換デフォルトを持つ
# ============================

def test_videostats_corruption_fields_backward_compat():
    """VideoStats の corruption フィールドがデフォルト 0/空で初期化されること。"""
    s = VideoStats(video_id="v_compat", is_holdout=False)
    assert s.postprocess_corruption_count == 0
    assert s.postprocess_corruption_by_side["1P"] == 0
    assert s.postprocess_corruption_color_pairs[(COLOR_RED, COLOR_BLUE)] == 0
    assert s.postprocess_corruption_log == []


# ============================
# 追加ケース: color_pair が正しく集計される
# ============================

def test_corruption_color_pairs_accumulated():
    """color_pair (GREEN→RED) が複数回呼ばれたとき正しくカウントされること。"""
    stats = _mk_stats()
    for i in range(5):
        _call_check(stats, raw_cnn=COLOR_GREEN, raw_hsv=COLOR_GREEN, confirmed=COLOR_RED,
                    row=i, fi=i)
    assert stats.postprocess_corruption_color_pairs[(COLOR_GREEN, COLOR_RED)] == 5


# ============================
# 追加ケース: _aggregate_corruption の blind_spot_note 存在確認
# ============================

def test_aggregate_corruption_blind_spot_note():
    """_aggregate_corruption の結果に blind_spot_note が含まれること。"""
    stats = _mk_stats()
    corruption = _aggregate_corruption([stats], total_stable_cells=1000)
    assert "blind_spot_note" in corruption
    note = corruption["blind_spot_note"]
    assert "全列崩壊" in note or "全列崩壊型" in note


# ============================
# 追加ケース: constraint_fill 無効時に corruption_count が 0 でも rate が 0.0
# ============================

def test_corruption_rate_zero_when_no_corruption():
    """corruption_count=0 で total_stable_cells>0 なら rate==0.0。"""
    stats = _mk_stats()
    corruption = _aggregate_corruption([stats], total_stable_cells=9999)
    assert corruption["rate"] == 0.0
    assert corruption["count"] == 0

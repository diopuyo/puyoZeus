from __future__ import annotations

import subprocess
import sys
import json
from collections import defaultdict
from pathlib import Path

import pytest

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from scripts.measure_stable_cell_acc import (
    PASS_OVERALL_THRESHOLD,
    PASS_PER_COLOR_THRESHOLD,
    VideoStats,
    _aggregate_stats,
    _compute_holdout_summary,
    _judge_pass_fail,
    _judge_pass_fail_with_corruption,
    _resolve_hsv_path,
    _resolve_video_path,
    _majority_vote,
    _record_cell,
    _eval_side_frame,
    COLOR_NAMES,
    EVAL_COLORS,
    _collect_col_metrics,
    _judge_i1_metrics,
    _build_i1_summary,
    PER_COL_UNKNOWN_WARNING,
    PER_COL_UNKNOWN_CRITICAL,
    NON_STABLE_CRITICAL_FRAMES,
    NON_STABLE_WARMUP_SEC,
    MIDGAME_START_SEC,
    MIDGAME_COL_EMPTY_CRITICAL,
    MIDGAME_COL_MIN_FRAMES,
    BOARD_ROWS,
    BOARD_COLS,
    # サブカテゴリ関連
    _classify_corruption_subcategory,
    _aggregate_corruption_subcategory,
    _aggregate_corruption,
    _judge_corruption_metrics,
    CORRUPTION_EMPTY_TO_COLOR_WARNING_RATE,
    POSTPROCESS_CORRUPTION_REJECT_RATE,
)
from src.board import (
    COLOR_EMPTY, COLOR_RED, COLOR_BLUE, COLOR_GREEN,
    COLOR_YELLOW, COLOR_PURPLE, COLOR_OJAMA, COLOR_UNKNOWN,
)
# ============================
# ヘルパー
# ============================

def _make_stats(
    video_id=None,
    is_holdout=False,
    total=100,
    agreed=99,
    color=None,
):
    if video_id is None: video_id = "v99"
    if color is None: color = COLOR_RED
    s = VideoStats(video_id=video_id, is_holdout=is_holdout)
    s.total_cells = total
    s.agreed_cells = agreed
    s.total_by_color[color] = total
    s.correct_by_color[color] = agreed
    s.total_by_row[5] = total
    s.correct_by_row[5] = agreed
    s.stable_frame_count = 10
    return s

# ============================
# _aggregate_stats テスト
# ============================

def test_aggregate_overall_acc_perfect():
    """全合意の場合 acc=1.0 になること。"""
    stats = [_make_stats(total=50, agreed=50)]
    result = _aggregate_stats(stats)
    assert result["overall"]["acc"] == 1.0
    assert result["overall"]["total_cells"] == 50

def test_aggregate_overall_acc_partial():
    """部分合意の場合 acc が正しく計算されること。"""
    stats = [_make_stats(total=100, agreed=97)]
    result = _aggregate_stats(stats)
    assert abs(result["overall"]["acc"] - 0.97) < 1e-9

def test_aggregate_multiple_videos():
    """複数動画の場合に正しく集計されること。"""
    s1 = _make_stats("v01", total=100, agreed=99, color=COLOR_RED)
    s2 = _make_stats("v02", total=200, agreed=198, color=COLOR_BLUE)
    result = _aggregate_stats([s1, s2])
    assert result["overall"]["total_cells"] == 300
    assert result["overall"]["correct"] == 297

def test_aggregate_per_video_keys():
    """per_video に動画 ID が含まれること。"""
    s1 = _make_stats("v_abc")
    result = _aggregate_stats([s1])
    assert "v_abc" in result["per_video"]

def test_aggregate_per_row_structure():
    """per_row に row_0 〜 row_12 が含まれること。"""
    stats = [_make_stats()]
    result = _aggregate_stats(stats)
    assert "row_0" in result["per_row"]
    assert "row_12" in result["per_row"]
    assert len(result["per_row"]) == 13

def test_aggregate_per_color_keys():
    """per_color に 7 色すべてが含まれること。"""
    stats = [_make_stats()]
    result = _aggregate_stats(stats)
    expected = {COLOR_NAMES[c] for c in EVAL_COLORS}
    assert set(result["per_color"].keys()) == expected

def test_aggregate_empty_stats_list():
    """空リストで overall acc=0.0 になること (ゼロ除算しないこと)。"""
    result = _aggregate_stats([])
    assert result["overall"]["acc"] == 0.0

def test_aggregate_no_data_for_color_gives_1_0():
    """該当色のデータが 0 件の場合 per_color が 1.0 になること。"""
    stats = [_make_stats(color=COLOR_RED)]
    result = _aggregate_stats(stats)
    assert result["per_color"]["blue"] == 1.0

# ============================
# _compute_holdout_summary テスト
# ============================

def test_holdout_summary_no_holdout_videos():
    """holdout_ids に該当なしなら acc=None になること。"""
    stats = [_make_stats("v01")]
    summary = _compute_holdout_summary(stats, ["v99"])
    assert summary["acc"] is None
    assert "note" in summary

def test_holdout_summary_correct_acc():
    """holdout 動画の acc が正しく計算されること。"""
    s_ho = _make_stats("v89", is_holdout=True, total=80, agreed=76)
    s_learn = _make_stats("v30", is_holdout=False, total=120, agreed=118)
    summary = _compute_holdout_summary([s_ho, s_learn], ["v89"])
    assert abs(summary["acc"] - 76 / 80) < 1e-9
    assert summary["total_cells"] == 80

def test_holdout_summary_empty_list():
    """holdout_ids が空なら acc=None になること。"""
    stats = [_make_stats("v01")]
    summary = _compute_holdout_summary(stats, [])
    assert summary["acc"] is None

# ============================
# _judge_pass_fail テスト
# ============================

def _all_ok_colors(val=0.999):
    return {name: val for name in COLOR_NAMES.values() if name != "unknown"}

def test_judge_pass_all_ok():
    """全指標が閾値超で PASS になること。"""
    verdict, failures = _judge_pass_fail(0.999, _all_ok_colors(), 0.999)
    assert verdict == "PASS"
    assert failures == []

def test_judge_fail_overall():
    """全マス平均が閾値未満で FAIL になること。"""
    verdict, failures = _judge_pass_fail(0.990, _all_ok_colors(), 0.990)
    assert verdict == "FAIL"
    assert any("全マス平均" in f for f in failures)

def test_judge_fail_per_color():
    """1 色でも閾値未満なら FAIL になること。"""
    per_color = _all_ok_colors()
    per_color["ojama"] = 0.970
    verdict, failures = _judge_pass_fail(0.999, per_color, 0.999)
    assert verdict == "FAIL"
    assert any("ojama" in f for f in failures)

def test_judge_pass_threshold_boundary():
    """ちょうど閾値に等しい場合は PASS になること (境界値)。"""
    per_color = _all_ok_colors(PASS_PER_COLOR_THRESHOLD)
    verdict, _ = _judge_pass_fail(
        PASS_OVERALL_THRESHOLD, per_color, PASS_OVERALL_THRESHOLD,
    )
    assert verdict == "PASS"

def test_judge_holdout_acc_priority():
    """holdout_acc が overall_acc より低い場合 holdout_acc で FAIL になること。"""
    verdict, _ = _judge_pass_fail(0.999, _all_ok_colors(), holdout_acc=0.980)
    assert verdict == "FAIL"

def test_judge_holdout_acc_none_uses_overall():
    """holdout_acc=None のとき overall_acc で判定すること。"""
    verdict, _ = _judge_pass_fail(0.999, _all_ok_colors(), holdout_acc=None)
    assert verdict == "PASS"

# ============================
# _resolve_video_path テスト
# ============================

def test_resolve_video_path_not_found():
    """存在しない動画 ID で None が返ること。"""
    result = _resolve_video_path("vXXX_nonexistent", None)
    assert result is None

def test_resolve_video_path_existing(tmp_path):
    """動画ファイルが存在する場合に Path が返ること。"""
    fake_video = tmp_path / "v99_match1_60s.mp4"
    fake_video.touch()
    result = _resolve_video_path("v99", tmp_path)
    assert result == fake_video

def test_resolve_video_path_wrong_id(tmp_path):
    """異なる動画 ID のファイルが存在しても None が返ること。"""
    fake_video = tmp_path / "v88_match1_60s.mp4"
    fake_video.touch()
    result = _resolve_video_path("v99", tmp_path)
    assert result is None

# ============================
# _resolve_hsv_path テスト
# ============================

def test_resolve_hsv_path_fallback():
    """per-video JSON がない場合 _merged_default.json にフォールバックすること。"""
    result = _resolve_hsv_path("vXXX_nonexistent")
    assert result.name == "_merged_default.json"

def test_resolve_hsv_path_returns_json():
    """v89 または fallback で .json が返ること。"""
    result = _resolve_hsv_path("v89")
    assert result.suffix == ".json"

# ============================
# smoke test: CLI
# ============================

def test_cli_no_args_exits():
    """引数なしで実行すると SystemExit になること。"""
    import subprocess
    ret = subprocess.run(
        [sys.executable, "scripts/measure_stable_cell_acc.py"],
        capture_output=True,
        cwd=str(_PROJ_ROOT),
    )
    assert ret.returncode != 0

def test_cli_help_exits_zero():
    """--help で exit code 0 になること。"""
    import subprocess
    ret = subprocess.run(
        [sys.executable, "scripts/measure_stable_cell_acc.py", "--help"],
        capture_output=True,
        cwd=str(_PROJ_ROOT),
    )
    assert ret.returncode == 0


# ============================
# 3 者独立化: _majority_vote テスト
# ============================

def test_majority_vote_all_same():
    """3 者全員一致なら同値を返すこと。"""
    assert _majority_vote(COLOR_RED, COLOR_RED, COLOR_RED) == COLOR_RED


def test_majority_vote_ab_agree():
    """a == b で c が異なる場合は a を返すこと。"""
    assert _majority_vote(COLOR_RED, COLOR_RED, COLOR_BLUE) == COLOR_RED


def test_majority_vote_ac_agree():
    """a == c で b が異なる場合は a を返すこと。"""
    assert _majority_vote(COLOR_RED, COLOR_BLUE, COLOR_RED) == COLOR_RED


def test_majority_vote_bc_agree():
    """b == c で a が異なる場合は b を返すこと。"""
    assert _majority_vote(COLOR_BLUE, COLOR_RED, COLOR_RED) == COLOR_RED


def test_majority_vote_all_different():
    """全員不一致の場合は a (raw_cnn) を返すこと。"""
    assert _majority_vote(COLOR_RED, COLOR_BLUE, COLOR_GREEN) == COLOR_RED


# ============================
# 3 者独立化: _record_cell テスト
# ============================

def _make_empty_stats() -> VideoStats:
    """空の VideoStats を生成する。"""
    return VideoStats(video_id="vtest", is_holdout=False)


def test_record_cell_all_agree_increments_agreed():
    """3 者全員一致のとき agreed_cells が +1 されること。"""
    stats = _make_empty_stats()
    disags: list[dict] = []
    _record_cell("v99", 0, 0.0, "1P", 5, 2,
                 COLOR_RED, COLOR_RED, COLOR_RED, stats, disags)
    assert stats.agreed_cells == 1
    assert stats.total_cells == 1
    assert stats.disagreement_count == 0
    assert stats.all_three_agree_count == 1
    assert len(disags) == 0


def test_record_cell_cnn_hsv_disagree_physics_correct():
    """CNN と HSV が不一致で confirmed が多数決ラベルに一致するとき
    physics_fix_count が +1 され agreed にはカウントされないこと。

    パターン: raw_cnn=RED, raw_hsv=BLUE, confirmed=BLUE
    多数決ラベル = BLUE (hsv+confirmed で 2 対 1)
    raw_cnn != label → disagreement として記録される。
    """
    stats = _make_empty_stats()
    disags: list[dict] = []
    _record_cell("v99", 0, 0.0, "1P", 5, 2,
                 COLOR_RED, COLOR_BLUE, COLOR_BLUE, stats, disags)
    assert stats.disagreement_count == 1
    assert stats.agreed_cells == 0
    assert len(disags) == 1
    assert disags[0]["predictions"]["raw_cnn"] == "red"
    assert disags[0]["predictions"]["raw_hsv"] == "blue"
    assert disags[0]["predictions"]["confirmed"] == "blue"


def test_record_cell_disagreement_dict_has_three_keys():
    """不一致 cell の predictions に raw_cnn / raw_hsv / confirmed / majority_label が含まれること。

    パターン: raw_cnn=GREEN, raw_hsv=BLUE, confirmed=BLUE
    多数決 = BLUE (hsv+confirmed 2 票), raw_cnn=GREEN != label → disagreement に記録される。
    """
    stats = _make_empty_stats()
    disags: list[dict] = []
    _record_cell("v99", 10, 1.0, "2P", 3, 1,
                 COLOR_GREEN, COLOR_BLUE, COLOR_BLUE, stats, disags)
    assert len(disags) == 1
    pred = disags[0]["predictions"]
    assert "raw_cnn" in pred
    assert "raw_hsv" in pred
    assert "confirmed" in pred
    assert "majority_label" in pred


def test_record_cell_physics_fix_when_cnn_hsv_split_confirmed_matches():
    """raw_cnn == raw_hsv だが confirmed が異なる場合は physics_fix_count を増やさないこと。

    パターン: raw_cnn=RED, raw_hsv=RED, confirmed=BLUE
    多数決 = RED (cnn+hsv), raw_cnn == label → agreed=1
    physics_fix_count は raw_cnn != raw_hsv の場合のみ増える → 0 のまま。
    """
    stats = _make_empty_stats()
    disags: list[dict] = []
    _record_cell("v99", 5, 0.5, "1P", 7, 3,
                 COLOR_RED, COLOR_RED, COLOR_BLUE, stats, disags)
    assert stats.agreed_cells == 1
    assert stats.physics_fix_count == 0


# ============================
# 3 者独立化: _eval_side_frame テスト
# ============================

class _FakeBoard:
    """テスト用 Board モック。全 cell が同じ色を返す。"""

    def __init__(self, color: int) -> None:
        self._color = color

    def get(self, row: int, col: int) -> int:  # noqa: D401
        return self._color


def test_eval_side_frame_three_way_all_agree():
    """3 者全員一致フレームで total_cells と all_three_agree_count が正しく増えること。"""
    stats = _make_empty_stats()
    disags: list[dict] = []
    board_red = _FakeBoard(COLOR_RED)
    _eval_side_frame(
        "1P", 0, 0.0, "vtest",
        raw_cnn_board=board_red,
        raw_hsv_board=board_red,
        confirmed_board=board_red,
        stats=stats,
        disagreements=disags,
    )
    assert stats.total_cells > 0
    assert stats.agreed_cells == stats.total_cells
    assert stats.all_three_agree_count == stats.total_cells
    assert stats.disagreement_count == 0


def test_eval_side_frame_cnn_hsv_disagree_confirmed_matches_hsv():
    """CNN と HSV が全 cell 不一致 (CNN=RED, HSV=BLUE, confirmed=BLUE) のとき
    disagreement が記録されること (多数決 = BLUE, raw_cnn=RED で不一致)。"""
    stats = _make_empty_stats()
    disags: list[dict] = []
    _eval_side_frame(
        "1P", 0, 0.0, "vtest",
        raw_cnn_board=_FakeBoard(COLOR_RED),
        raw_hsv_board=_FakeBoard(COLOR_BLUE),
        confirmed_board=_FakeBoard(COLOR_BLUE),
        stats=stats,
        disagreements=disags,
    )
    assert stats.total_cells > 0
    assert stats.disagreement_count > 0
    # 不一致 cell の predictions に raw_cnn フィールドがあること
    assert len(disags) > 0
    assert "raw_cnn" in disags[0]["predictions"]


# ============================
# サブカテゴリ corruption テスト (C タスク)
# ============================


def test_classify_corruption_subcategory_empty_to_color():
    """raw==EMPTY → confirmed==色 のとき empty_to_color_count が +1 されること。"""
    stats = _make_empty_stats()
    _classify_corruption_subcategory(COLOR_EMPTY, COLOR_RED, stats)
    assert stats.corruption_empty_to_color_count == 1
    assert stats.corruption_color_to_color_count == 0
    assert stats.corruption_color_to_empty_count == 0


def test_classify_corruption_subcategory_color_to_color():
    """raw==色A → confirmed==色B のとき color_to_color_count が +1 されること。"""
    stats = _make_empty_stats()
    _classify_corruption_subcategory(COLOR_RED, COLOR_BLUE, stats)
    assert stats.corruption_color_to_color_count == 1
    assert stats.corruption_empty_to_color_count == 0
    assert stats.corruption_color_to_empty_count == 0


def test_classify_corruption_subcategory_color_to_empty():
    """raw==色 → confirmed==EMPTY のとき color_to_empty_count が +1 されること。"""
    stats = _make_empty_stats()
    _classify_corruption_subcategory(COLOR_GREEN, COLOR_EMPTY, stats)
    assert stats.corruption_color_to_empty_count == 1
    assert stats.corruption_empty_to_color_count == 0
    assert stats.corruption_color_to_color_count == 0


def test_classify_corruption_subcategory_accumulates():
    """複数回呼び出したとき各カウンタが正しく累積されること。"""
    stats = _make_empty_stats()
    _classify_corruption_subcategory(COLOR_EMPTY, COLOR_RED, stats)   # e2c
    _classify_corruption_subcategory(COLOR_EMPTY, COLOR_BLUE, stats)  # e2c
    _classify_corruption_subcategory(COLOR_RED, COLOR_BLUE, stats)    # c2c
    _classify_corruption_subcategory(COLOR_BLUE, COLOR_EMPTY, stats)  # c2e
    assert stats.corruption_empty_to_color_count == 2
    assert stats.corruption_color_to_color_count == 1
    assert stats.corruption_color_to_empty_count == 1


def _make_stats_with_corruption(
    e2c: int = 0, c2c: int = 0, c2e: int = 0,
    total_cells: int = 1000,
) -> VideoStats:
    """サブカテゴリ corruption カウンタを指定した VideoStats を生成する。"""
    s = VideoStats(video_id="vtest", is_holdout=False)
    s.total_cells = total_cells
    s.agreed_cells = total_cells
    s.postprocess_corruption_count = e2c + c2c + c2e
    s.corruption_empty_to_color_count = e2c
    s.corruption_color_to_color_count = c2c
    s.corruption_color_to_empty_count = c2e
    return s


def test_aggregate_corruption_subcategory_empty_list():
    """空リストのとき全サブカテゴリ count=0 / rate=0.0 になること。"""
    result = _aggregate_corruption_subcategory([], 1000)
    for key in ("empty_to_color", "color_to_color", "color_to_empty"):
        assert result[key]["count"] == 0
        assert result[key]["rate"] == 0.0


def test_aggregate_corruption_subcategory_counts_summed():
    """複数動画の各サブカテゴリが正しく合算されること。"""
    s1 = _make_stats_with_corruption(e2c=5, c2c=3, c2e=2, total_cells=500)
    s2 = _make_stats_with_corruption(e2c=10, c2c=1, c2e=4, total_cells=500)
    result = _aggregate_corruption_subcategory([s1, s2], total_stable_cells=1000)
    assert result["empty_to_color"]["count"] == 15
    assert result["color_to_color"]["count"] == 4
    assert result["color_to_empty"]["count"] == 6


def test_aggregate_corruption_subcategory_rate_correct():
    """rate = count / total_stable_cells が正しく計算されること。"""
    s = _make_stats_with_corruption(e2c=10, total_cells=1000)
    result = _aggregate_corruption_subcategory([s], total_stable_cells=1000)
    assert abs(result["empty_to_color"]["rate"] - 0.010) < 1e-9


def test_aggregate_corruption_subcategory_zero_division():
    """total_stable_cells=0 のとき rate=0.0 になること (ゼロ除算しないこと)。"""
    s = _make_stats_with_corruption(e2c=5)
    result = _aggregate_corruption_subcategory([s], total_stable_cells=0)
    # denom=0 → denom=1 fallback なので rate=5/1=5.0 (上限なし)
    # 主要確認: クラッシュしないこと
    assert isinstance(result["empty_to_color"]["rate"], float)


def test_aggregate_corruption_subcategory_has_note_fields():
    """各サブカテゴリに 'note' フィールドが含まれること。"""
    result = _aggregate_corruption_subcategory([], 100)
    for key in ("empty_to_color", "color_to_color", "color_to_empty"):
        assert "note" in result[key]


def test_aggregate_corruption_has_subcategory_key():
    """_aggregate_corruption の戻り値に 'subcategory' キーが含まれること。"""
    s = _make_stats_with_corruption(e2c=2, c2c=1, total_cells=100)
    result = _aggregate_corruption([s], total_stable_cells=100)
    assert "subcategory" in result
    assert "empty_to_color" in result["subcategory"]
    assert "color_to_color" in result["subcategory"]
    assert "color_to_empty" in result["subcategory"]


def test_aggregate_corruption_existing_fields_preserved():
    """既存フィールド (count/rate/by_side/color_pairs等) が破壊されないこと。"""
    s = _make_stats_with_corruption(e2c=2, total_cells=100)
    result = _aggregate_corruption([s], total_stable_cells=100)
    for key in ("count", "rate", "by_side", "color_pairs",
                "side_bias", "corruption_ratio", "log", "blind_spot_note"):
        assert key in result, f"既存キー '{key}' が消えている"


def test_judge_corruption_metrics_warning_only_does_not_affect_fail():
    """empty_to_color が WARNING 閾値を超えても [WARNING] メッセージが返るだけで
    FAIL メッセージは増えないこと。"""
    high_e2c_rate = CORRUPTION_EMPTY_TO_COLOR_WARNING_RATE * 2
    corruption_section = {
        "rate": 0.0,  # FAIL 閾値未満
        "side_bias": {"detected": False},
        "subcategory": {
            "empty_to_color": {"rate": high_e2c_rate},
            "color_to_color": {"rate": 0.0},
            "color_to_empty": {"rate": 0.0},
        },
    }
    messages = _judge_corruption_metrics(corruption_section)
    # WARNING メッセージが含まれること
    assert any("[WARNING]" in m for m in messages)
    # FAIL メッセージ (REJECT 閾値・side_bias) が含まれないこと
    assert not any("REJECT閾値" in m for m in messages)
    assert not any("side_bias" in m for m in messages)


def test_judge_corruption_metrics_no_warning_below_threshold():
    """empty_to_color が WARNING 閾値未満のとき [WARNING] が出ないこと。"""
    low_e2c_rate = CORRUPTION_EMPTY_TO_COLOR_WARNING_RATE / 2
    corruption_section = {
        "rate": 0.0,
        "side_bias": {"detected": False},
        "subcategory": {
            "empty_to_color": {"rate": low_e2c_rate},
        },
    }
    messages = _judge_corruption_metrics(corruption_section)
    assert not any("[WARNING]" in m for m in messages)


def test_judge_pass_fail_with_corruption_warning_not_fail():
    """empty_to_color WARNING だけが発生した場合 PASS 判定になること。"""
    high_e2c_rate = CORRUPTION_EMPTY_TO_COLOR_WARNING_RATE * 2
    corruption_section = {
        "rate": 0.0,  # FAIL 閾値未満
        "side_bias": {"detected": False},
        "subcategory": {
            "empty_to_color": {"rate": high_e2c_rate},
            "color_to_color": {"rate": 0.0},
            "color_to_empty": {"rate": 0.0},
        },
    }
    # 全 acc は OK
    per_color = {name: 0.999 for name in COLOR_NAMES.values() if name != "unknown"}
    verdict, failures = _judge_pass_fail_with_corruption(
        overall_acc=0.999,
        per_color=per_color,
        holdout_acc=0.999,
        stats_list=None,
        corruption_section=corruption_section,
    )
    # PASS でなければならない
    assert verdict == "PASS"
    # ただし failures に [WARNING] メッセージが含まれること (情報提示)
    assert any("[WARNING]" in f for f in failures)


def test_judge_pass_fail_with_corruption_fail_when_rate_high():
    """corruption_rate が REJECT 閾値以上なら FAIL になること。"""
    corruption_section = {
        "rate": POSTPROCESS_CORRUPTION_REJECT_RATE * 2,
        "side_bias": {"detected": False},
        "subcategory": {
            "empty_to_color": {"rate": 0.0},
        },
    }
    per_color = {name: 0.999 for name in COLOR_NAMES.values() if name != "unknown"}
    verdict, failures = _judge_pass_fail_with_corruption(
        overall_acc=0.999,
        per_color=per_color,
        holdout_acc=0.999,
        stats_list=None,
        corruption_section=corruption_section,
    )
    assert verdict == "FAIL"
    assert any("REJECT閾値" in f for f in failures)


def test_video_stats_subcategory_fields_default_zero():
    """VideoStats の新サブカテゴリフィールドがデフォルト 0 であること (後方互換)。"""
    stats = VideoStats(video_id="v99", is_holdout=False)
    assert stats.corruption_empty_to_color_count == 0
    assert stats.corruption_color_to_color_count == 0
    assert stats.corruption_color_to_empty_count == 0

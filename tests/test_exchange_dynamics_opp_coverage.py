"""#24 打ち合い計測器 Step1 の回帰テスト (2026-07-29)。

対象:
    1. FireEvent に付与する相手盤面の被覆状態 (OppCoverageStatus 4値+UNKNOWN)。
    2. _classify_exchange の整地 (せいち) 分類追加 (既存の催促/本線/不明判定は不変)。
    3. 上記2点が _build_fire_event / _process_video まで正しく配線されていること。

軽量なダミーデータのみを使用し、実動画・実npzの重い処理は一切行わない。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.chain import ChainSimulator
from scripts.measure_exchange_dynamics import (
    ANOMALY_DELTA_SCORE_MIN,
    MATCH_END_REMAINING_SEC,
    OPP_COVERAGE_GAP_THRESHOLD_SEC,
    SCORE_DELTA_FIRE,
    SEICHI_OJAMA_MAX_COUNT,
    FireEvent,
    NpzRecord,
    OppCoverageStatus,
    _build_fire_event,
    _classify_exchange,
    _classify_gap_cause,
    _classify_opp_coverage,
    _process_video,
)


def _make_record(
    t_sec: list[float],
    score: list[float],
    side: str = "2P",
) -> NpzRecord:
    """テスト用の最小 NpzRecord を組み立てる (grids/game_idx はダミー値)。"""
    n = len(t_sec)
    return NpzRecord(
        video_id="test_video",
        side=side,
        t_sec=np.array(t_sec, dtype=np.float32),
        game_idx=np.zeros(n, dtype=np.int32),
        grids=np.zeros((n, 13, 6), dtype=np.int8),
        score=np.array(score, dtype=np.int32),
    )


# ============================
# (1) OppCoverageStatus 分類 (_classify_opp_coverage / _classify_gap_cause)
# ============================


def test_opp_coverage_observed_when_gap_small() -> None:
    """ギャップが閾値以下なら OBSERVED (連続観測できている)。"""
    opp = _make_record(t_sec=[0.0, 1.0, 2.5], score=[0, 0, 0])
    status = _classify_opp_coverage(t_fire=1.0, own_game_end_t=10.0, opp_window=opp)
    assert status == OppCoverageStatus.OBSERVED


def test_opp_coverage_observed_at_exact_threshold_boundary() -> None:
    """ギャップがちょうど閾値 (2.0秒) なら OBSERVED (<=、境界は観測扱い)。"""
    opp = _make_record(t_sec=[0.0, 1.0, 1.0 + OPP_COVERAGE_GAP_THRESHOLD_SEC], score=[0, 0, 0])
    status = _classify_opp_coverage(t_fire=1.0, own_game_end_t=10.0, opp_window=opp)
    assert status == OppCoverageStatus.OBSERVED


def test_opp_coverage_opp_chaining_when_gap_large_with_score_jump() -> None:
    """ギャップが閾値超 かつ 相手スコアが連鎖跳ね相当なら OPP_CHAINING。"""
    opp = _make_record(t_sec=[0.0, 1.0, 8.0], score=[0, 0, SCORE_DELTA_FIRE])
    status = _classify_opp_coverage(t_fire=1.0, own_game_end_t=10.0, opp_window=opp)
    assert status == OppCoverageStatus.OPP_CHAINING


def test_opp_coverage_unobserved_when_gap_large_without_score_jump() -> None:
    """ギャップが閾値超 かつ 相手スコアが微増のみなら UNOBSERVED (本物の取りこぼし)。"""
    opp = _make_record(t_sec=[0.0, 1.0, 8.0], score=[0, 0, SCORE_DELTA_FIRE - 1])
    status = _classify_opp_coverage(t_fire=1.0, own_game_end_t=10.0, opp_window=opp)
    assert status == OppCoverageStatus.UNOBSERVED


def test_opp_coverage_unknown_when_score_missing_at_gap_boundary() -> None:
    """ギャップ境界の相手スコアが欠損 (-1) なら判別不能 UNKNOWN。"""
    opp = _make_record(t_sec=[0.0, 1.0, 8.0], score=[0, -1, 0])
    status = _classify_opp_coverage(t_fire=1.0, own_game_end_t=10.0, opp_window=opp)
    assert status == OppCoverageStatus.UNKNOWN


def test_opp_coverage_unknown_when_delta_is_anomalous() -> None:
    """ギャップ境界の得点増分が異常値域なら判別不能 UNKNOWN (UNOBSERVEDに混ぜない)。"""
    opp = _make_record(t_sec=[0.0, 1.0, 8.0], score=[0, 0, ANOMALY_DELTA_SCORE_MIN])
    status = _classify_opp_coverage(t_fire=1.0, own_game_end_t=10.0, opp_window=opp)
    assert status == OppCoverageStatus.UNKNOWN


def test_opp_coverage_match_end_when_no_future_frame_near_own_game_end() -> None:
    """相手側フレームが尽きており、攻撃側自身の残り試合時間が短いなら MATCH_END。"""
    opp = _make_record(t_sec=[0.0, 1.0], score=[0, 0])
    status = _classify_opp_coverage(t_fire=1.0, own_game_end_t=1.0 + MATCH_END_REMAINING_SEC, opp_window=opp)
    assert status == OppCoverageStatus.MATCH_END


def test_opp_coverage_unobserved_when_no_future_frame_far_from_match_end() -> None:
    """相手側フレームが尽きていても、攻撃側自身の試合がまだ長く続くなら
    「試合終了」ではなく長時間ブラックアウト = UNOBSERVED として扱う。
    """
    opp = _make_record(t_sec=[0.0, 1.0], score=[0, 0])
    status = _classify_opp_coverage(
        t_fire=1.0, own_game_end_t=1.0 + MATCH_END_REMAINING_SEC + 1.0, opp_window=opp,
    )
    assert status == OppCoverageStatus.UNOBSERVED


def test_opp_coverage_unobserved_when_no_opp_frame_before_fire() -> None:
    """発火直前の相手フレームが1件も無い (試合開始直後等) 場合は保守的に UNOBSERVED。"""
    opp = _make_record(t_sec=[5.0, 6.0], score=[0, 0])
    status = _classify_opp_coverage(t_fire=1.0, own_game_end_t=10.0, opp_window=opp)
    assert status == OppCoverageStatus.UNOBSERVED


def test_classify_gap_cause_matches_opp_coverage_boundary_values() -> None:
    """_classify_gap_cause 単体でも SCORE_DELTA_FIRE / ANOMALY 境界が期待通り。"""
    assert _classify_gap_cause(0, SCORE_DELTA_FIRE) == OppCoverageStatus.OPP_CHAINING
    assert _classify_gap_cause(0, SCORE_DELTA_FIRE - 1) == OppCoverageStatus.UNOBSERVED
    assert _classify_gap_cause(0, ANOMALY_DELTA_SCORE_MIN) == OppCoverageStatus.UNKNOWN
    assert _classify_gap_cause(0, -1) == OppCoverageStatus.UNKNOWN  # 負差分は異常
    assert _classify_gap_cause(-1, 0) == OppCoverageStatus.UNKNOWN  # score欠損


# ============================
# (2) _classify_exchange の整地分類追加 (既存2値ロジック不変の確認込み)
# ============================


def _grid_with_color_puyos(n: int) -> np.ndarray:
    """色ぷよ (COLOR_RED=1) が n 個ある盤面グリッドを返す (先頭 n セルを埋める)。"""
    grid = np.zeros((13, 6), dtype=np.int8)
    flat = grid.reshape(-1)
    flat[:n] = 1
    return grid


def test_classify_exchange_backward_compat_saisoku_without_ojama_arg() -> None:
    """ojama_sent_count 省略時は従来通り催促 (ratio<0.6) のまま (backwards compat)。"""
    before = _grid_with_color_puyos(5)
    after = _grid_with_color_puyos(3)  # consumed=2, ratio=0.4
    label, ratio = _classify_exchange(before, after)
    assert label == "催促"
    assert ratio == pytest.approx(0.4)


def test_classify_exchange_backward_compat_honsen_without_ojama_arg() -> None:
    """ojama_sent_count 省略時は従来通り本線 (ratio>=0.6) のまま (backwards compat)。"""
    before = _grid_with_color_puyos(5)
    after = _grid_with_color_puyos(1)  # consumed=4, ratio=0.8
    label, ratio = _classify_exchange(before, after)
    assert label == "本線"
    assert ratio == pytest.approx(0.8)


def test_classify_exchange_fumei_unaffected_by_ojama_arg() -> None:
    """before_n=0 (不明) は ojama_sent_count が小さくても整地に上書きされない。"""
    before = _grid_with_color_puyos(0)
    after = _grid_with_color_puyos(0)
    label, ratio = _classify_exchange(before, after, ojama_sent_count=1.0)
    assert label == "不明"
    assert np.isnan(ratio)


def test_classify_exchange_seichi_overrides_saisoku_when_ojama_small() -> None:
    """送りお邪魔が SEICHI_OJAMA_MAX_COUNT 以下なら催促相当でも整地に上書きされる。"""
    before = _grid_with_color_puyos(5)
    after = _grid_with_color_puyos(3)  # ratio=0.4 (催促相当)
    label, ratio = _classify_exchange(before, after, ojama_sent_count=SEICHI_OJAMA_MAX_COUNT)
    assert label == "整地"
    assert ratio == pytest.approx(0.4)  # ratio自体は上書きしない (比率計算は不変)


def test_classify_exchange_seichi_overrides_honsen_when_ojama_small() -> None:
    """送りお邪魔が SEICHI_OJAMA_MAX_COUNT 以下なら本線相当でも整地に上書きされる。"""
    before = _grid_with_color_puyos(5)
    after = _grid_with_color_puyos(1)  # ratio=0.8 (本線相当)
    label, _ratio = _classify_exchange(before, after, ojama_sent_count=0.0)
    assert label == "整地"


def test_classify_exchange_no_seichi_when_ojama_above_threshold() -> None:
    """送りお邪魔が閾値超なら整地に上書きされず、通常の催促/本線判定を維持する。"""
    before = _grid_with_color_puyos(5)
    after = _grid_with_color_puyos(1)  # ratio=0.8 (本線)
    label, _ratio = _classify_exchange(
        before, after, ojama_sent_count=SEICHI_OJAMA_MAX_COUNT + 1.0,
    )
    assert label == "本線"


def test_classify_exchange_no_seichi_override_when_ojama_is_nan() -> None:
    """ojama_sent_count が NaN (スコア欠損) の場合は上書きしない。"""
    before = _grid_with_color_puyos(5)
    after = _grid_with_color_puyos(3)  # ratio=0.4 (催促)
    label, _ratio = _classify_exchange(before, after, ojama_sent_count=float("nan"))
    assert label == "催促"


# ============================
# (3) FireEvent への配線確認 (_build_fire_event / _process_video)
# ============================


def test_fireevent_default_opp_coverage_status_is_unknown() -> None:
    """FireEvent を直接構築した場合 (新フィールド省略) は既定 UNKNOWN (backwards compat)。"""
    ev = FireEvent(
        video_stem="v", tier="マスター", game_idx=0, fire_side="1P",
        fi_idx=0, t_fire=0.0, delta_score=0, chain_count=0, ratio=0.0,
        label="催促", ojama_sent_count=0.0,
    )
    assert ev.opp_coverage_status == OppCoverageStatus.UNKNOWN.value


def _four_stack_grid() -> np.ndarray:
    """col=0 の最下段4マス (row9-12) に赤ぷよ4個を積んだ物理的に妥当な盤面。"""
    grid = np.zeros((13, 6), dtype=np.int8)
    grid[9:13, 0] = 1
    return grid


def test_build_fire_event_without_opp_rec_full_keeps_unknown() -> None:
    """opp_rec_full 省略時は opp_coverage_status が UNKNOWN のまま (backwards compat)。"""
    rec = NpzRecord(
        video_id="v", side="1P",
        t_sec=np.array([0.0, 2.0], dtype=np.float32),
        game_idx=np.array([0, 0], dtype=np.int32),
        grids=np.stack([_four_stack_grid(), np.zeros((13, 6), dtype=np.int8)]),
        score=np.array([0, 500], dtype=np.int32),
    )
    ev = _build_fire_event(
        rec, fi=1, delta_score=500, sim=ChainSimulator(),
        game_start_t=0.0, tier="マスター", video_stem="v", before_idx=0,
    )
    assert ev.opp_coverage_status == OppCoverageStatus.UNKNOWN.value
    assert ev.chain_count >= 1  # 物理的に妥当な4連結なので本物の発火として検出される


def test_build_fire_event_with_opp_rec_full_computes_observed() -> None:
    """opp_rec_full 指定時は opp_coverage_status が実計算される (OBSERVED になる例)。

    own側 (rec) に発火後 (t=10.0) のフレームを1つ足し own_game_end_t を
    t_fire (t=2.0) から引き離しておく。そうしないと "own の最終フレーム=
    発火そのもの" になり、_restrict_to_time_window が opp の t=2.5 を
    window外として切り捨て、MATCH_END に化けてしまう (実測で確認済み、
    _process_video 統合テストの extra_1p_tail_frame と同じ理由)。
    """
    rec = NpzRecord(
        video_id="v", side="1P",
        t_sec=np.array([0.0, 2.0, 10.0], dtype=np.float32),
        game_idx=np.array([0, 0, 0], dtype=np.int32),
        grids=np.stack([
            _four_stack_grid(), np.zeros((13, 6), dtype=np.int8), np.zeros((13, 6), dtype=np.int8),
        ]),
        score=np.array([0, 500, 500], dtype=np.int32),
    )
    opp_rec_full = _make_record(t_sec=[0.0, 1.0, 2.5], score=[0, 0, 0], side="2P")
    ev = _build_fire_event(
        rec, fi=1, delta_score=500, sim=ChainSimulator(),
        game_start_t=0.0, tier="マスター", video_stem="v", before_idx=0,
        opp_rec_full=opp_rec_full,
    )
    assert ev.opp_coverage_status == OppCoverageStatus.OBSERVED.value


def _save_synthetic_npz(path: Path, extra_1p_tail_frame: bool) -> None:
    """1P (攻撃側・4連結発火1回) + 2P (発火なし) の最小合成npzを保存する。

    Args:
        extra_1p_tail_frame: True で 1P に t=10.0 の追加フレームを足す
            (own_game_end_t を発火時刻から引き離し、OBSERVED 判定を誘発)。
            False なら発火が最終フレームのまま (MATCH_END 判定を誘発)。
    """
    four_stack = _four_stack_grid()
    empty = np.zeros((13, 6), dtype=np.int8)
    t_1p = [0.0, 1.0, 3.0] + ([10.0] if extra_1p_tail_frame else [])
    score_1p = [0, 0, 500] + ([500] if extra_1p_tail_frame else [])
    grids_1p = [four_stack, four_stack, empty] + ([empty] if extra_1p_tail_frame else [])

    t_2p = [0.0, 2.0, 3.5]
    score_2p = [0, 0, 0]
    grids_2p = [empty, empty, empty]

    n1, n2 = len(t_1p), len(t_2p)
    np.savez_compressed(
        str(path),
        video_id=np.array(["test_video"] * (n1 + n2)),
        side=np.array(["1P"] * n1 + ["2P"] * n2),
        t_sec=np.array(t_1p + t_2p, dtype=np.float32),
        game_idx=np.zeros(n1 + n2, dtype=np.int32),
        grids=np.stack(grids_1p + grids_2p).astype(np.int8),
        score=np.array(score_1p + score_2p, dtype=np.int32),
    )


def test_process_video_end_to_end_observed(tmp_path: Path) -> None:
    """own_game_end_t が t_fire から離れている場合、_process_video 経由でも
    opp_coverage_status が OBSERVED になる (フル配線確認)。
    """
    npz_path = tmp_path / "test_video.npz"
    _save_synthetic_npz(npz_path, extra_1p_tail_frame=True)
    _, defrag_events, _ = _process_video(npz_path, ChainSimulator(), 0)
    assert len(defrag_events) == 1
    ev = defrag_events[0]
    assert ev.fire_side == "1P"
    assert ev.label == "本線"
    assert ev.chain_count >= 1
    assert ev.opp_coverage_status == OppCoverageStatus.OBSERVED.value


def test_process_video_end_to_end_match_end(tmp_path: Path) -> None:
    """own側自身の試合が発火直後に終わっている場合、_process_video 経由でも
    opp_coverage_status が MATCH_END になる (フル配線確認)。
    """
    npz_path = tmp_path / "test_video.npz"
    _save_synthetic_npz(npz_path, extra_1p_tail_frame=False)
    _, defrag_events, _ = _process_video(npz_path, ChainSimulator(), 0)
    assert len(defrag_events) == 1
    ev = defrag_events[0]
    assert ev.opp_coverage_status == OppCoverageStatus.MATCH_END.value

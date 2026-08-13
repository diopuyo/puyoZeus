"""下部グラフの横軸相対時間化 (#8) の回帰テスト。

docs/DEMO_REVIEW_2026-08-13.md #8: 試合境界検知時に history.clear() する
一方で横軸が動画全体の絶対時間のままだったため、境界後の曲線が絶対位置
(=途中) から始まって見えるバグだった。純粋関数
`_graph_relative_time`/`_reset_graph_origin` の単体テスト (フラグ不要の
表示バグ修正、常時適用)。
"""
from __future__ import annotations

from scripts.visualize_advantage_overlay import (
    _graph_relative_time,
    _reset_graph_origin,
)


def test_relative_time_before_any_boundary_matches_legacy() -> None:
    """境界が一度も起きていない (game_start_sec=0.0) 場合、従来の
    `t - start_sec` と完全に一致する (backwards compat)。"""
    assert _graph_relative_time(t_sec=12.5, start_sec=2.0, game_start_sec=0.0) == 10.5


def test_relative_time_resets_to_zero_at_boundary() -> None:
    """境界検知直後 (game_start_sec が (t-start_sec) と一致) は 0 になる。"""
    t_sec, start_sec = 40.0, 2.0
    game_start_sec, _ = _reset_graph_origin(t_sec, start_sec, n_frames=3000, fps=30.0)
    assert _graph_relative_time(t_sec, start_sec, game_start_sec) == 0.0


def test_relative_time_grows_from_zero_after_boundary() -> None:
    """境界後は 0 から単調に増える (絶対時間ではなく相対時間)。"""
    start_sec = 2.0
    boundary_t = 40.0
    game_start_sec, _ = _reset_graph_origin(boundary_t, start_sec, n_frames=9000, fps=30.0)
    later_t = 45.5
    rel = _graph_relative_time(later_t, start_sec, game_start_sec)
    assert rel == 5.5  # 絶対値 43.5 ではなく境界からの経過 5.5 秒


def test_reset_graph_origin_scale_is_remaining_video_length() -> None:
    """スケールは「この時点からの残り動画尺」(既存 total_dur と同じ式の
    境界時刻版) に巻き直る。"""
    n_frames, fps = 3000, 30.0  # 動画全体 100 秒
    t_sec = 40.0
    _, graph_total = _reset_graph_origin(t_sec, start_sec=0.0, n_frames=n_frames, fps=fps)
    assert graph_total == 60.0  # 100 - 40


def test_reset_graph_origin_scale_has_floor_of_one_second() -> None:
    """動画末尾ギリギリで境界が起きても最低 1.0 秒は確保する (ゼロ除算回避、
    既存 total_dur と同じ下限)。"""
    n_frames, fps = 300, 30.0  # 動画全体 10 秒
    _, graph_total = _reset_graph_origin(t_sec=9.99, start_sec=0.0, n_frames=n_frames, fps=fps)
    assert graph_total == 1.0

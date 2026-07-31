"""#24 被覆ゲート測定バグ修正の回帰テスト (2026-07-29)。

真因: scripts/measure_ojama_landing_delay.py の相手フレーム絞り込みが
「相手側 game_idx ラベル一致」に依存しており、1P/2P 独立カウンタである
game_idx がスコアリセット見逃しでズレると、実在する相手観測を
「別試合」として誤って捨てていた。

修正: `_opponent_frame_mask` を追加し、既定 (use_game_idx_mask=False) では
攻撃側自身の game_idx 境界時刻 [開始, 終了] で相手フレームを絞る
時刻ベース方式に変更した (game_idx ラベル自体は信用しない)。
`use_game_idx_mask=True` で旧挙動 (backwards compat) を再現できる。

本テストは、game_idx がズレた合成データで
「時刻ベース (新) なら相手観測を拾える／game_idx一致 (旧) では拾えない」
ことを検証する。
"""
from __future__ import annotations

import numpy as np

from scripts.measure_exchange_dynamics import NpzRecord
from scripts.measure_ojama_landing_delay import _opponent_frame_mask


def _make_record(side: str, t_sec: list[float], game_idx: list[int]) -> NpzRecord:
    """テスト用の最小 NpzRecord を組み立てる (grids/score はダミー値)。"""
    n = len(t_sec)
    return NpzRecord(
        video_id="test_video",
        side=side,
        t_sec=np.array(t_sec, dtype=np.float32),
        game_idx=np.array(game_idx, dtype=np.int32),
        grids=np.zeros((n, 13, 6), dtype=np.int8),
        score=np.zeros(n, dtype=np.int32),
    )


def test_time_window_mask_recovers_frames_missed_by_offset_game_idx() -> None:
    """相手側の game_idx が +1 ズレた合成データで、新方式は拾い旧方式は落とす。"""
    # 攻撃側 (own): 第1試合 (game_idx=0, t=0..4) -> 第2試合 (game_idx=1, t=5..9)
    own_rec = _make_record(
        "1P",
        t_sec=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        game_idx=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    )
    # 相手側 (opp): 同じ実試合の同じ時間帯 (t=5..9) なのに、リセット見逃しで
    # game_idx ラベルが 2 に飛んでいる (1 というラベルは一度も出現しない)。
    opp_rec = _make_record(
        "2P",
        t_sec=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        game_idx=[0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
    )
    target_game_idx = 1  # 攻撃側の第2試合

    # 旧方式 (相手側 game_idx ラベル一致): ラベル1が存在しないため全滅
    old_mask = _opponent_frame_mask(opp_rec, own_rec, target_game_idx, use_game_idx_mask=True)
    assert not old_mask.any(), "旧方式は相手観測を全く拾えないはず(バグ再現)"

    # 新方式 (既定、時刻ベース): 攻撃側の試合区間 [5,9] にある相手フレーム5件を拾う
    new_mask = _opponent_frame_mask(opp_rec, own_rec, target_game_idx)
    assert int(new_mask.sum()) == 5, "新方式は攻撃側の試合区間内の相手フレームを全て拾うはず"
    assert list(opp_rec.t_sec[new_mask]) == [5.0, 6.0, 7.0, 8.0, 9.0]


def test_time_window_mask_default_matches_explicit_false() -> None:
    """use_game_idx_mask 省略時の既定値が False (新方式) であることを確認する。"""
    own_rec = _make_record("1P", t_sec=[0, 1, 2], game_idx=[0, 0, 0])
    opp_rec = _make_record("2P", t_sec=[0, 1, 2], game_idx=[5, 5, 5])
    default_mask = _opponent_frame_mask(opp_rec, own_rec, 0)
    explicit_mask = _opponent_frame_mask(opp_rec, own_rec, 0, use_game_idx_mask=False)
    assert list(default_mask) == list(explicit_mask)


def test_old_mask_still_reproducible_for_backward_compat() -> None:
    """game_idx ラベルが一致している通常ケースでは新旧方式が一致することを確認する
    (ズレが無ければ結果が変わらないことの安全確認、既存 cycle への影響がないケース)。
    """
    own_rec = _make_record("1P", t_sec=[0, 1, 2, 3], game_idx=[0, 0, 1, 1])
    opp_rec = _make_record("2P", t_sec=[0, 1, 2, 3], game_idx=[0, 0, 1, 1])
    old_mask = _opponent_frame_mask(opp_rec, own_rec, 1, use_game_idx_mask=True)
    new_mask = _opponent_frame_mask(opp_rec, own_rec, 1, use_game_idx_mask=False)
    assert list(old_mask) == list(new_mask) == [False, False, True, True]

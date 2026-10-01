"""疎な保存から連続フレームを捏造しないことと、未来非参照を検証する。"""
import numpy as np
import pandas as pd
from pathlib import Path

from scripts.plan_c_input_audit import assigned, inspect_video


def data(changing: bool = False) -> dict[str, np.ndarray]:
    """同一盤面の連続読みと、配置ごとの保存を分ける。"""
    n = 9
    grids = np.zeros((n, 13, 6), np.int8)
    grids[3:, -1, 0] = 1
    if changing:
        grids[:, -1, 1] = np.arange(n)
    return dict(grids=grids, t_sec=np.arange(n)/30, game_idx=np.zeros(n, int),
                side=np.array(['1P']*n), next1_a=np.ones(n, int), next1_b=np.full(n, 2),
                dnext_a=np.full(n, 3), dnext_b=np.full(n, 4))


def test_sparse_rows_do_not_adopt_queue() -> None:
    assert not assigned(data(changing=True)).any()


def test_actual_three_readings_can_adopt() -> None:
    raw = data()
    raw['next1_a'][3:] = 3
    assert (assigned(raw)[5] > 0).all()


def test_every_prefix_matches_full() -> None:
    raw = data()
    full = assigned(raw)
    for cut in range(1, len(full)+1):
        short = assigned({k: v[:cut] for k, v in raw.items()})
        np.testing.assert_array_equal(short, full[:cut])


def test_game_boundary_drops_history() -> None:
    raw = data()
    raw['game_idx'][3:] = 1
    assert not assigned(raw).any()


def test_unselected_video_still_audited(tmp_path: Path) -> None:
    raw = data()
    raw['frame_idx'] = np.arange(len(raw['t_sec']))
    path = tmp_path/'excluded.npz'
    np.savez(path, **raw)
    report = inspect_video(path, pd.DataFrame(columns=['game_id', 'source_side', 'frame']))
    assert report['training_rows'] == report['matched_rows'] == 0
    assert report['raw_rows'] == 9

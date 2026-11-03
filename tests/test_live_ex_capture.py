"""非STABLEの未来NEXTを採用せず、入力コピーを破壊しない。"""
from types import SimpleNamespace

from src.phase_j.live_ex_capture import CaptureEx, capture


def test_next_changes_only_with_stable_confirmed_board() -> None:
    owner = CaptureEx()
    side = dict(state=dict(state='STABLE'), confirmed_board=dict(board=dict(array=[1])),
                next_pair=[1, 2], dnext_pair=[3, 4])
    row = dict(game_idx=1, _v49f=dict(raw=[side, side]))
    frame = SimpleNamespace(media_sec=1.)
    first = owner.capture(None, frame, row)
    side['next_pair'] = [4, 4]
    side['state']['state'] = 'CHAIN'
    second = owner.capture(None, frame, row)
    assert first['raw'][0]['next_pair'] == second['raw'][0]['next_pair'] == [1, 2]
    assert row['_v49f']['raw'][0]['next_pair'] == [4, 4]
    side['state']['state'] = 'STABLE'
    side['confirmed_board']['board']['array'] = [2]
    assert owner.capture(None, frame, row)['raw'][0]['next_pair'] == [4, 4]


def test_epoch_keeps_formal_next_until_stable_board_changes() -> None:
    """世代境界だけでは正式StableNextの保持値を更新しない。"""
    owner = SimpleNamespace()
    side = dict(state=dict(state='STABLE'), confirmed_board=dict(board=dict(array=[1])),
                next_pair=[1, 2], dnext_pair=[3, 4])
    row = dict(game_idx=1, _v49f=dict(raw=[side, side]))
    frame = SimpleNamespace(media_sec=1.)
    capture(owner, None, frame, row, epoch=1)
    side.update(next_pair=[5, 2], dnext_pair=None)
    capture(owner, None, frame, row, epoch=2)
    assert row['_ex']['raw'][0]['next_pair'] == [1, 2]
    assert row['_ex']['raw'][0]['dnext_pair'] == [3, 4]
    assert owner._ex_epoch == 2
    side['confirmed_board']['board']['array'] = [2]
    capture(owner, None, frame, row, epoch=2)
    assert row['_ex']['raw'][0]['next_pair'] == [5, 2]
    assert row['_ex']['raw'][0]['dnext_pair'] is None


def test_new_owner_starts_independent_next_history() -> None:
    """別の運転では前の所有者のNEXTを継承しない。"""
    side = dict(state=dict(state='STABLE'), confirmed_board=dict(board=dict(array=[1])),
                next_pair=[1, 2], dnext_pair=[3, 4])
    row = dict(game_idx=1, _v49f=dict(raw=[side, side]))
    frame = SimpleNamespace(media_sec=1.)
    capture(SimpleNamespace(), None, frame, row, epoch=1)
    side.update(next_pair=[5, 2], dnext_pair=None)
    capture(SimpleNamespace(), None, frame, row, epoch=2)
    assert row['_ex']['raw'][0]['next_pair'] == [5, 2]
    assert row['_ex']['raw'][0]['dnext_pair'] is None

"""EX のツモのテープにだけ毎コマの NEXT を送る旗 (live_24h_tape_next、既定 OFF) を検査する。"""
import argparse
from types import SimpleNamespace

import pytest

from src.phase_j import live_24h_config as config
from src.phase_j import live_ex_capture
from src.phase_j.live_ex_capture import CaptureEx
from src.phase_j.live_ex_engine import TapeNextKnowledge
from src.phase_j._live_ex_frozen.src.prefire_tsumo_v411 import Knowledge

BOARD_A = [[0] * 6 for _ in range(13)]
BOARD_B = [[0] * 6 for _ in range(12)] + [[1, 0, 0, 0, 0, 0]]


def side(state: str, board: list, nxt: tuple, dnx: tuple) -> dict:
    return dict(state=dict(state=state), confirmed_board=dict(board=dict(array=board)),
                next_pair=dict(tuple=list(nxt)), dnext_pair=dict(tuple=list(dnx)), next_slide_motion=False)


def frames() -> list[tuple[float, dict]]:
    """1P だけ動く。盤が変わるのは最初のコマだけで、その後 NEXT が 1 手ずつ 2 回繰り上がる。"""
    still = side('STABLE', BOARD_B, (3, 4), (1, 1))
    rows = [(0.0, side('STABLE', BOARD_A, (1, 2), (3, 4)))]
    rows += [(0.1 * i, side('TSUMO_FALL', None, (1, 2), (3, 4))) for i in range(1, 4)]
    rows += [(0.1 * i, side('TSUMO_FALL', None, (3, 4), (1, 1))) for i in range(4, 8)]
    rows += [(0.1 * i, still) for i in range(8, 10)]
    rows += [(0.1 * i, side('TSUMO_FALL', None, (1, 1), (2, 2))) for i in range(10, 14)]
    return rows


def capture_all(enabled: bool, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    monkeypatch.setenv(live_ex_capture.ENV_TAPE_NEXT, '1' if enabled else '0')
    monkeypatch.setenv('PUYO_LIVE_24H_RETR', '0')
    capture, out = CaptureEx(), []
    other = side('STABLE', BOARD_A, (5, 5), (5, 5))
    for t, p1 in frames():
        row = dict(game_idx=0, _v49f=dict(raw=[p1, other]))
        out.append(capture.capture(None, SimpleNamespace(media_sec=t), row))
    return out


def test_off_packets_have_no_tape_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for packet in capture_all(False, monkeypatch):
        assert all('tape_next_pair' not in s and 'tape_dnext_pair' not in s for s in packet['raw'])


def test_on_keeps_held_next_and_adds_per_frame_next(monkeypatch: pytest.MonkeyPatch) -> None:
    off, on = capture_all(False, monkeypatch), capture_all(True, monkeypatch)
    for a, b, (_, p1) in zip(off, on, frames()):
        stripped = [{k: v for k, v in s.items() if not k.startswith('tape_')} for s in b['raw']]
        assert stripped == a['raw']   # 他の消費者が読む値は旗 OFF と同じ
        assert b['raw'][0]['tape_next_pair'] == p1['next_pair']
        assert b['raw'][0]['tape_dnext_pair'] == p1['dnext_pair']


def run_tape(knowledge: Knowledge, packets: list[dict]) -> Knowledge:
    for packet in packets:
        knowledge.observe(packet['t'], packet['game'], packet['raw'])
    return knowledge


def test_tape_follows_every_advance_only_with_tape_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    on = capture_all(True, monkeypatch)
    held = run_tape(Knowledge(), on).tape.sides[0]
    tape = run_tape(TapeNextKnowledge(), on).tape.sides[0]
    assert tape.own == [(1, 2), (3, 4), (1, 1), (2, 2)] and tape.restarts == 0
    assert len(held.own) < len(tape.own)   # 持ち越しの NEXT では繰り上がりを取りこぼす


def test_tape_knowledge_without_keys_matches_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    off = capture_all(False, monkeypatch)
    a, b = run_tape(Knowledge(), off), run_tape(TapeNextKnowledge(), off)
    assert [s.own for s in a.tape.sides] == [s.own for s in b.tape.sides]
    assert a.tape.events == b.tape.events


@pytest.mark.parametrize('value', ['yes', 1])
def test_invalid_flag_is_rejected(value: object, tmp_path) -> None:
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    assert parser.parse_args([]).live_24h_tape_next is False
    with pytest.raises(SystemExit):
        config.configure(SimpleNamespace(live_24h=True, live_24h_tape_next=value), parser)
    with pytest.raises(ValueError, match='live_24h_tape_next'):
        config.launcher_values(dict(live_24h=True, live_24h_tape_next=value), tmp_path)


def test_launcher_to_ex_settings(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    from src.phase_j import live_ex_runtime
    from src.phase_j.launcher import build_pipeline_config, parse_user_config, USER_KEYS
    from tests.test_live_24h_launcher import data
    for key in (config.ENV_XC, config.ENV_RETR, config.ENV_TAPE_NEXT, config.ENV_ENABLED, config.ENV_AMA,
                config.ENV_DELAY, config.ENV_MODEL, 'PUYO_EX_WORK_BUDGET', 'PUYO_EX_WORK_LIMIT'):
        monkeypatch.setenv(key, os.environ.get(key, ''))
    assert config.ENV_TAPE_NEXT == live_ex_capture.ENV_TAPE_NEXT
    user = parse_user_config(data(tmp_path, live_24h=True, live_24h_retr=True, live_24h_tape_next=True,
                                  live_24h_model='ex.json'), tmp_path)
    pipeline = build_pipeline_config(user)
    assert 'live_24h_tape_next' in USER_KEYS and pipeline['live_24h_tape_next'] is True
    config.configure(SimpleNamespace(**pipeline), argparse.ArgumentParser())
    assert live_ex_capture.tape_next_enabled()
    monkeypatch.setattr(live_ex_runtime, 'settings_from_env', lambda defaults: defaults)
    assert config.ex_settings()['tape_next'] is True

"""コマ落ちに強い状態判定の箇所別 OFF (live_frame_drop_rules_off)。既定は何も外さない。

外す箇所の判定は「OFF と同じ数え方」へ戻る。全コマでは刻み = 回数なので、箇所を外しても外さなくても同じ出力になる。
ここでは (1) 指定の検証と配線 (puyo_live.json → ランチャ → 環境変数 → 認識 process)、(2) 空なら何も差し替えない、
(3) 箇所ごとに、刻みの判定が成立する入力で「外すと回数どおりに戻る」ことを、差し替え前の関数との比較で確かめる。
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

pytest.importorskip('torch')
from src.media_tick import TickShadow, TickStamp, min_obs, audit_reset  # noqa: E402
from src.phase_j import live_fdr_rules, live_process, live_speed  # noqa: E402
from src.phase_j.launcher import LauncherConfigError, build_pipeline_config, parse_user_config  # noqa: E402
from src.phase_j.live_speed import (FRAME_DROP_RULES, FRAME_DROP_RULES_OFF_ENV, FRAME_DROP_RULES_OFF_KEY,  # noqa: E402
                                    parse_rules_off)

STEP = 8   # 動画の stride を想定した刻みの間隔 (1 回の呼び出しで 8 刻み進む)


def patch_targets() -> list[tuple[Any, str]]:
    """箇所別 OFF が差し替えうる (対象, 属性名) の全部。試験が元へ戻すために使う。"""
    import src.b3_live_queues as queues
    import src.board_state_machine as machine
    import src.ojama_visual_detector as ojama
    import src.recognition_pipeline as pipeline
    import src.state_detectors as detectors
    return [(machine.BoardStateMachine, '_pending_reached'), (machine.BoardStateMachine, '_resume_vote_weights'),
            (machine.BoardStateMachine, '_append_non_stable_history_ticked'), (machine, '_recovery_count_reached'),
            (detectors.GravitySettleDetector, '_settle_reached_ticked'),
            (ojama.OjamaVisualDetector, '_board_settled'), (queues.LiveQueues, 'move_window_frame'),
            (pipeline.RecognitionPipeline, '_r35_missed_placement'), (pipeline.RecognitionPipeline, '_r15_warmup_step')]


@pytest.fixture
def restore_targets() -> Iterator[dict[tuple[int, str], Any]]:
    """差し替えを元へ戻す。originals は {(id(対象), 名前): 元の属性}。"""
    saved = [(target, name, target.__dict__[name]) for target, name in patch_targets()]
    yield {(id(target), name): value for target, name, value in saved}
    for target, name, value in saved:
        setattr(target, name, value)


def current(target: Any, name: str) -> Any:
    return target.__dict__[name]


def ticked_shadow(observed: int) -> TickShadow:
    """observed 回続いた影。1 回ごとに STEP 刻み進んだので、刻みは 1 + STEP*(observed-1)。
    判定の関数が自分で observe を呼ぶ箇所 (R04・R06・R10) には、1 回手前までの影を渡す。"""
    shadow = TickShadow()
    for k in range(1, observed+1):
        shadow.observe(k, TickStamp(now=STEP*k, prev_call=None if k == 1 else STEP*(k-1)))
    return shadow


# --- 指定の検証 -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('value, expected', [
    (None, ()), ('', ()), ([], ()), ('R06', ('R06',)), (' R10 , R06 ', ('R06', 'R10')),
    (['R35', 'R01', 'R01'], ('R01', 'R35')), (('B07',), ('B07',)), (list(FRAME_DROP_RULES), FRAME_DROP_RULES),
])
def test_parse_rules_off(value: Any, expected: tuple[str, ...]) -> None:
    assert parse_rules_off(value) == expected   # FRAME_DROP_RULES の順、重複は 1 つ


@pytest.mark.parametrize('value', ['R99', ['R06', 'r10'], 'R03', [6], 6, {'R06': True}, [None]])
def test_parse_rules_off_rejects_unknown_and_non_strings(value: Any) -> None:
    """知らない名前 (R03 は R02 に含むので単独では持たない) や文字列以外は黙って無視せず止める。"""
    with pytest.raises(ValueError, match=FRAME_DROP_RULES_OFF_KEY):
        parse_rules_off(value)


def test_rules_off_needs_frame_drop_robust() -> None:
    values = dict(live_frame_drop_robust=False)
    assert live_speed.rules_off_of({}, values) == ()
    with pytest.raises(ValueError, match='live_frame_drop_robust'):
        live_speed.rules_off_of({FRAME_DROP_RULES_OFF_KEY: ['R06']}, values)
    assert live_speed.rules_off_of({FRAME_DROP_RULES_OFF_KEY: ['R06']}, dict(live_frame_drop_robust=True)) == ('R06',)


# --- 配線: puyo_live.json → ランチャ → pipeline_config.json → 環境変数 ------------------------------------------

def test_launcher_accepts_and_forwards_rules_only_when_given(tmp_path: Path) -> None:
    base = dict(device_name='test', live_frame_drop_robust=True)
    off = parse_user_config(base, tmp_path)
    assert off.live_frame_drop_rules_off == () and FRAME_DROP_RULES_OFF_KEY not in build_pipeline_config(off)
    on = parse_user_config(dict(base, **{FRAME_DROP_RULES_OFF_KEY: ['R10', 'R06']}), tmp_path)
    assert on.live_frame_drop_rules_off == ('R06', 'R10')
    assert build_pipeline_config(on)[FRAME_DROP_RULES_OFF_KEY] == ['R06', 'R10']


def test_launcher_rejects_bad_rules(tmp_path: Path) -> None:
    with pytest.raises(LauncherConfigError, match='知らない箇所'):
        parse_user_config(dict(device_name='t', live_frame_drop_robust=True, **{FRAME_DROP_RULES_OFF_KEY: ['R99']}),
                          tmp_path)
    with pytest.raises(LauncherConfigError, match='live_frame_drop_robust'):
        parse_user_config(dict(device_name='t', **{FRAME_DROP_RULES_OFF_KEY: ['R06']}), tmp_path)


def test_configure_sets_and_resets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, 'environ', {})
    on = argparse.Namespace(source='video', cpu_threads=1, live_frame_drop_robust=True,
                            **{FRAME_DROP_RULES_OFF_KEY: ['R10', 'R06']})
    live_speed.configure(on, argparse.ArgumentParser())
    assert os.environ[FRAME_DROP_RULES_OFF_ENV] == 'R06,R10' and live_speed.rules_off_from_env() == ('R06', 'R10')
    record = live_speed.trial_flags_record(on)
    assert record['frame_drop_rules_off'] == ['R06', 'R10']
    live_speed.configure(argparse.Namespace(source='video', cpu_threads=1, live_frame_drop_robust=True),
                         argparse.ArgumentParser())
    assert os.environ[FRAME_DROP_RULES_OFF_ENV] == '' and live_speed.rules_off_from_env() == ()


def test_cli_delivers_rules_from_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """pipeline_config.json の配列が、実際の CLI の解析を通って環境変数になる (起動口と同じ経路)。"""
    import sys
    from scripts.run_live_pipeline_20260928 import parse_args
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    path = tmp_path/'input.json'
    path.write_text(json.dumps(dict(source='dshow', name='test', index=0, cpu_threads=1, live_frame_drop_robust=True,
                                    **{FRAME_DROP_RULES_OFF_KEY: ['R15', 'R06']})))
    monkeypatch.setattr(sys, 'argv', ['live', '--config', str(path), '--output', str(tmp_path/'out')])
    options = parse_args()
    assert os.environ[FRAME_DROP_RULES_OFF_ENV] == 'R06,R15' and options.live_frame_drop_robust is True


# --- 差し替え --------------------------------------------------------------------------------------------------

def test_empty_rules_replace_nothing(restore_targets: dict) -> None:
    before = {(id(t), n): current(t, n) for t, n in patch_targets()}
    assert live_fdr_rules.apply_rules_off(()) == ()
    assert {(id(t), n): current(t, n) for t, n in patch_targets()} == before


def test_each_rule_replaces_only_its_own_sites(restore_targets: dict) -> None:
    sites = {'R01': ['_pending_reached'], 'R02': ['_resume_vote_weights', '_append_non_stable_history_ticked'],
             'R04': ['_recovery_count_reached'], 'R06': ['_settle_reached_ticked'], 'R10': ['_board_settled'],
             'R15': ['_r15_warmup_step'], 'B07': ['move_window_frame'], 'R35': ['_r35_missed_placement']}
    assert set(sites) == set(FRAME_DROP_RULES)
    for rule, names in sites.items():
        for target, name in patch_targets():
            setattr(target, name, restore_targets[(id(target), name)])
        live_fdr_rules.apply_rules_off((rule,))
        replaced = {name for target, name in patch_targets()
                    if current(target, name) is not restore_targets[(id(target), name)]}
        assert replaced == set(names), rule


def test_apply_rejects_unknown_rule_and_missing_site(restore_targets: dict) -> None:
    with pytest.raises(ValueError, match='知らない箇所'):
        live_fdr_rules.apply_rules_off(('R99',))
    with pytest.raises(AttributeError, match='名前が変わった'):
        live_fdr_rules.replace(SimpleNamespace(__name__='x'), '_missing', lambda: None)


def test_r01_off_counts_by_calls_only(restore_targets: dict) -> None:
    from src.board_state_machine import BoardStateMachine
    n = 3
    count = min_obs(n)   # 回数は下限どおり、刻みは N 以上
    ctx = SimpleNamespace(pending_count=count, pending_tick=ticked_shadow(count))
    machine = SimpleNamespace(_ctx=ctx, _stable_n=n, _enable_stable_majority_window=False)
    stamp = TickStamp(now=STEP*count, prev_call=STEP*(count-1))
    assert BoardStateMachine._pending_reached(machine, stamp) is True        # 刻みモード: 早く成立
    live_fdr_rules.apply_rules_off(('R01',))
    assert BoardStateMachine._pending_reached(machine, stamp) is False       # 外すと回数どおり (count < N)
    ctx.pending_count = n
    assert BoardStateMachine._pending_reached(machine, stamp) is True


def test_r02_off_keeps_count_history_and_no_weights(restore_targets: dict) -> None:
    from src.board_state_machine import BoardStateMachine
    board = SimpleNamespace(copy=lambda: object())
    ctx = SimpleNamespace(non_stable_cnn_history=[object(), object(), object()],
                          non_stable_cnn_weights=[1, 1, 1], non_stable_last_append_tick=None)
    machine = SimpleNamespace(_ctx=ctx, _non_stable_history_size=3)
    signals = SimpleNamespace(media_tick=TickStamp(now=STEP, prev_call=0))
    assert BoardStateMachine._resume_vote_weights(machine, signals) is not None   # 刻みモードは重みを返す
    live_fdr_rules.apply_rules_off(('R02',))
    assert BoardStateMachine._resume_vote_weights(machine, signals) is None
    BoardStateMachine._append_non_stable_history_ticked(machine, board, signals.media_tick)
    assert len(ctx.non_stable_cnn_history) == 3 and ctx.non_stable_cnn_weights == [1, 1, 1]   # 件数で古い順に捨てる


def test_r04_off_counts_calls_only(restore_targets: dict) -> None:
    import src.board_state_machine as machine
    n = 8
    count = min_obs(n)
    ctx = SimpleNamespace(recovery_tick_shadows={(0, 0): ticked_shadow(count-1)})
    stamp = TickStamp(now=STEP*count, prev_call=STEP*(count-1))
    assert machine._recovery_count_reached(ctx, (0, 0), count, n, stamp) is True
    live_fdr_rules.apply_rules_off(('R04',))
    assert machine._recovery_count_reached(ctx, (0, 0), count, n, stamp) is False
    assert machine._recovery_count_reached(ctx, (0, 0), n, n, stamp) is True


def test_r06_off_counts_calls_only(restore_targets: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('src.production_config.LIVE_FRAME_DROP_ROBUST_SITES', frozenset({'R06', 'R10', 'R15'}))
    from src.state_detectors import GRAVITY_SETTLE_MIN_FRAMES, GravitySettleDetector
    count = min_obs(GRAVITY_SETTLE_MIN_FRAMES)
    detector = SimpleNamespace(_stable_consec=count, _stable_tick=ticked_shadow(count-1))
    stamp = TickStamp(now=STEP*count, prev_call=STEP*(count-1))
    assert GravitySettleDetector._settle_reached_ticked(detector, stamp) is True
    live_fdr_rules.apply_rules_off(('R06',))
    assert GravitySettleDetector._settle_reached_ticked(detector, stamp) is False
    detector._stable_consec = GRAVITY_SETTLE_MIN_FRAMES
    assert GravitySettleDetector._settle_reached_ticked(detector, stamp) is True


def test_r10_off_counts_calls_only(restore_targets: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    # 既定の採否 (production_config) は R15 だけで R10 は元から回数どおり。差し替え前の刻みの動きを見るため R10 を載せる。
    monkeypatch.setattr('src.production_config.LIVE_FRAME_DROP_ROBUST_SITES', frozenset({'R06', 'R10', 'R15'}))
    from src.ojama_visual_detector import OJAMA_FALL_SETTLE_STABLE_FRAMES, OjamaVisualDetector
    count = min_obs(OJAMA_FALL_SETTLE_STABLE_FRAMES)
    detector = SimpleNamespace(_board_stable_consec=count, _board_stable_tick=ticked_shadow(count-1))
    stamp = TickStamp(now=STEP*count, prev_call=STEP*(count-1))
    assert OjamaVisualDetector._board_settled(detector, stamp) is True
    live_fdr_rules.apply_rules_off(('R10',))
    assert OjamaVisualDetector._board_settled(detector, stamp) is False
    detector._board_stable_consec = OJAMA_FALL_SETTLE_STABLE_FRAMES
    assert OjamaVisualDetector._board_settled(detector, stamp) is True


def test_r15_off_decrements_by_one(restore_targets: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('src.production_config.LIVE_FRAME_DROP_ROBUST_SITES', frozenset({'R15'}))   # 既定は空なので、差し替え前の動きを見るため R15 を載せる
    from src.recognition_pipeline import RecognitionPipeline
    audit_reset()
    pipeline = SimpleNamespace(_media_tick_stamp=TickStamp(now=STEP*3, prev_call=STEP*2))
    assert RecognitionPipeline._r15_warmup_step(pipeline) == STEP
    live_fdr_rules.apply_rules_off(('R15',))
    assert RecognitionPipeline._r15_warmup_step(pipeline) == 1


def test_r35_off_never_fills_missed_placement(restore_targets: dict) -> None:
    from src.recognition_pipeline import RecognitionPipeline
    live_fdr_rules.apply_rules_off(('R35',))
    pipeline = SimpleNamespace(_media_tick_stamp=TickStamp(now=STEP*3, prev_call=0),
                               _pending_tsumo_1p=deque([(1, 2)]), _pending_tsumo_2p=deque())
    assert RecognitionPipeline._r35_missed_placement(pipeline, '1P', 1.0, 'stable') is None
    assert list(pipeline._pending_tsumo_1p) == [(1, 2)]   # 補完で消費されない


def test_b07_off_returns_no_window(restore_targets: dict) -> None:
    from src.b3_live_queues import LiveQueues
    queues = SimpleNamespace(prev_frame=100)
    pipe = SimpleNamespace(media_tick_step=STEP)
    assert LiveQueues.move_window_frame(queues, pipe, 200) == 100+STEP        # 前回 + 1 刻みで見る
    live_fdr_rules.apply_rules_off(('B07',))
    assert LiveQueues.move_window_frame(queues, pipe, 200) is None


# --- 認識 process の起動 ----------------------------------------------------------------------------------------

def stub_pipe() -> SimpleNamespace:
    pipe = SimpleNamespace(media_tick_step=None, _enable_media_tick_counters=True)
    pipe.set_media_tick_step = lambda step: setattr(pipe, 'media_tick_step', step)
    return pipe


def test_configure_media_tick_applies_rules_and_logs_them(restore_targets: dict, monkeypatch: pytest.MonkeyPatch,
                                                          capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(os, 'environ', {'PUYO_LIVE_FRAME_DROP_ROBUST': '1', FRAME_DROP_RULES_OFF_ENV: 'R06,R15'})
    live_process.configure_media_tick(stub_pipe(), STEP)
    line = capsys.readouterr().out.strip()
    assert line.startswith(live_process.FRAME_DROP_ROBUST_MARK)
    assert json.loads(line[len(live_process.FRAME_DROP_ROBUST_MARK):])['rules_off'] == ['R06', 'R15']
    import src.state_detectors as detectors
    assert current(detectors.GravitySettleDetector, '_settle_reached_ticked') is not restore_targets[
        (id(detectors.GravitySettleDetector), '_settle_reached_ticked')]


def test_configure_media_tick_default_is_unchanged(restore_targets: dict, monkeypatch: pytest.MonkeyPatch,
                                                   capsys: pytest.CaptureFixture) -> None:
    """箇所別 OFF が空 (既定) なら、印に rules_off の鍵は無く、何も差し替えない。FDR OFF なら印も出ない。"""
    before = {(id(t), n): current(t, n) for t, n in patch_targets()}
    monkeypatch.setattr(os, 'environ', {'PUYO_LIVE_FRAME_DROP_ROBUST': '1'})
    live_process.configure_media_tick(stub_pipe(), STEP)
    state = json.loads(capsys.readouterr().out.strip()[len(live_process.FRAME_DROP_ROBUST_MARK):])
    assert 'rules_off' not in state and state['enabled'] is True
    assert state['sites'] == []               # 認識 process が読んだ採否 (production_config の既定は空。R06・R10・R15 は外す)
    monkeypatch.setattr(os, 'environ', {FRAME_DROP_RULES_OFF_ENV: 'R06'})   # FDR が OFF の指定は効かない
    live_process.configure_media_tick(stub_pipe(), STEP)
    assert capsys.readouterr().out == ''
    assert {(id(t), n): current(t, n) for t, n in patch_targets()} == before

"""録画の流し直し起動口: 合言葉の門 (無ければ何も作らない)、計画の中身、子の実行時差し替え。"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import live_x_replay_child as child
from scripts import run_live_x_replay as replay


@pytest.mark.parametrize('token,environ', [
    (None, {}), (replay.RUN_TOKEN, {}), (None, {replay.CONFIRM_ENV: replay.CONFIRM_WORD}),
    ('違う', {replay.CONFIRM_ENV: replay.CONFIRM_WORD}), (replay.RUN_TOKEN, {replay.CONFIRM_ENV: '違う'})])
def test_gate_refuses_without_both_words(token: str | None, environ: dict) -> None:
    with pytest.raises(SystemExit, match='起動しない'):
        replay.authorize(token, environ)


def test_gate_accepts_both_words() -> None:
    replay.authorize(replay.RUN_TOKEN, {replay.CONFIRM_ENV: replay.CONFIRM_WORD})


def test_run_without_token_creates_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(replay.CONFIRM_ENV, raising=False)
    monkeypatch.setattr(replay, 'blocked', lambda: pytest.fail('門より先へ進んだ'))
    root = tmp_path/'replay'
    with pytest.raises(SystemExit, match='起動しない'):
        replay.main(['run', '--root', str(root)])
    assert not root.exists() and list(tmp_path.iterdir()) == []


def saved_dir(tmp_path: Path) -> Path:
    saved = tmp_path/'saved'
    saved.mkdir()
    (saved/'puyo_live.json').write_text(json.dumps(dict(
        source='dshow', device_name='OBS Virtual Camera', device_index=3, port=8765, output_dir='x',
        dshow_color_correction='601to709', b3_judgment=True, live_v49f=True)), encoding='utf-8')
    (saved/'recognition_config.json').write_text(json.dumps(dict(
        enable_placement_signal_reconcile=False, stable_frame_count=3)), encoding='utf-8')
    return saved


@pytest.mark.parametrize('arm,condition', [('OFF', 'L'), ('ON', 'F')])
def test_config_keeps_21c_and_overrides_only_input(arm: str, condition: str, tmp_path: Path) -> None:
    saved = saved_dir(tmp_path)
    config = replay.user_config(arm, condition, replay.OFFSET_SEC+100., replay.OFFSET_SEC+200., tmp_path, saved)
    assert config['source'] == 'video' and config['live_x_direct_publish'] is (arm == 'ON')
    assert (config['start_sec'], config['end_sec']) == pytest.approx((100., 200.))
    assert config['b3_judgment'] is True and config['live_v49f'] is True and config['port'] == replay.PORT
    assert not {'device_name', 'device_index', 'dshow_color_correction'} & config.keys()
    plan = replay.child_plan(condition, tmp_path, ['src/x.py'], saved)
    assert plan['realtime'] is replay.CONDITIONS[condition]
    assert plan['recognition_overrides'] == dict(enable_placement_signal_reconcile=False)
    assert plan['crop'] == list(replay.CROP) and plan['offset_sec'] == replay.OFFSET_SEC


def test_plan_is_side_effect_free_abba(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(replay, 'overlay_modules', lambda: ['src/phase_j/live_v49f.py'])
    monkeypatch.setattr(replay, 'bundle_drift', lambda modules: [])
    saved = saved_dir(tmp_path)
    monkeypatch.setattr(replay, 'SAVED', saved)
    monkeypatch.setattr(replay, 'user_config', lambda arm, *rest: dict(live_x_direct_publish=arm == 'ON'))
    monkeypatch.setattr(replay, 'child_plan', lambda condition, out, modules: dict(modules=modules))
    runs = replay.plan(replay.arguments(['plan', '--root', str(tmp_path/'r')]))
    assert [r['arm'] for r in runs] == list(replay.ARMS) and [r['name'] for r in runs] == [
        'L1_OFF', 'L2_ON', 'L3_ON', 'L4_OFF']
    assert not (tmp_path/'r').exists()


def test_plan_refuses_bundle_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(replay, 'overlay_modules', lambda: [])
    monkeypatch.setattr(replay, 'bundle_drift', lambda modules: ['src/b3_state.py'])
    with pytest.raises(SystemExit, match='ソースが違う'):
        replay.plan(replay.arguments(['plan', '--root', str(tmp_path)]))


def test_run_one_receipt_counts_patched_processes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved = saved_dir(tmp_path)
    monkeypatch.setattr(replay, 'SAVED', saved)
    out = tmp_path/'L1_OFF'
    run = dict(name='L1_OFF', arm='OFF', out=str(out), config=dict(a=1), command=['python'],
               child=dict(source_modules=['src/x.py']))

    def fake_run(command: list, **kwargs: object) -> SimpleNamespace:
        assert json.loads(kwargs['env']['LIVE_X_REPLAY_CHILD']) == run['child']
        lines = [dict(pid=1, patch='video_source'), dict(pid=2, patch='video_source'), dict(pid=1, patch='launcher')]
        (out/'patches.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in lines), encoding='utf-8')
        (out/'output').mkdir()
        (out/'output/recognition_config.json').write_text((saved/'recognition_config.json').read_text())
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(replay.subprocess, 'run', fake_run)
    receipt = replay.run_one(run)
    assert receipt['patches'] == dict(video_source=2, recognition_config=0, launcher=1)
    assert receipt['recognition_matches_21c'] and receipt['exit_code'] == 0
    assert json.loads((out/'puyo_live.json').read_text(encoding='utf-8')) == dict(a=1)


def test_same_text_ignores_only_line_endings(tmp_path: Path) -> None:
    left, right, other = tmp_path/'a', tmp_path/'b', tmp_path/'c'
    left.write_bytes(b'x = 1\r\ny = 2\r\n')
    right.write_bytes(b'x = 1\ny = 2\n')
    other.write_bytes(b'x = 1\ny = 3\n')
    assert replay.same_text(left, right) and not replay.same_text(left, other)


def test_video_source_patch_crops_corrects_and_shifts(tmp_path: Path) -> None:
    from src.phase_j.live_source import CapturedFrame
    image = np.zeros((1080, 1920, 3), np.uint8)
    image[:920, :1636] = 200                      # 切出し範囲だけ明るい
    image[920:, :] = 10

    class Source:
        def __iter__(self):
            yield CapturedFrame(30, 1., 5., 5., image)
    module = SimpleNamespace(VideoFileSource=Source)
    plan = dict(crop=[1636, 920], offset_sec=715.5, color_correction='601to709', patch_log=str(tmp_path/'p.jsonl'))
    child.cropped_source(plan)(module)
    frame = next(iter(module.VideoFileSource()))
    assert frame.image.shape == (1080, 1920, 3) and frame.media_sec == pytest.approx(716.5)
    assert frame.index == 30 and frame.captured_at == 5. and int(frame.image.min()) > 100
    assert json.loads((tmp_path/'p.jsonl').read_text())['patch'] == 'video_source'


def test_recognition_and_launcher_patches(tmp_path: Path) -> None:
    plan = dict(recognition_overrides=dict(enable_placement_signal_reconcile=False), app=str(tmp_path),
                realtime=False, patch_log=str(tmp_path/'p.jsonl'))
    speed = SimpleNamespace(recognition_config=lambda config, live_input: dict(config, live=live_input))
    child.recognition_reference(plan)(speed)
    assert speed.recognition_config(dict(enable_placement_signal_reconcile=True), False) == dict(
        enable_placement_signal_reconcile=False, live=False)
    launcher = SimpleNamespace(APP_ROOT=Path('worktree'), build_pipeline_config=lambda user: dict(realtime=True, a=1))
    child.launcher_paths(plan)(launcher)
    assert launcher.APP_ROOT == tmp_path and launcher.build_pipeline_config(None) == dict(realtime=False, a=1)


def test_post_import_patch_runs_after_module_body(tmp_path: Path) -> None:
    (tmp_path/'live_x_dummy_module.py').write_text('VALUE = 1\n', encoding='utf-8')
    hook = child.PostImportPatch({'live_x_dummy_module': lambda module: setattr(module, 'VALUE', module.VALUE+1)})
    sys.path.insert(0, str(tmp_path))
    sys.meta_path.insert(0, hook)
    try:
        import live_x_dummy_module
        assert live_x_dummy_module.VALUE == 2
    finally:
        sys.meta_path.remove(hook)
        sys.path.remove(str(tmp_path))
        sys.modules.pop('live_x_dummy_module', None)


def test_overlay_modules_include_only_listed_scripts(monkeypatch) -> None:
    """差し替える scripts は OVERLAY_SCRIPTS (collect_boards_lean) だけ。run_live_pipeline は ROOT が __file__ 由来で壊れるので対象外、
    台・採点の scripts も差し替えない。"""
    assert replay.OVERLAY_SCRIPTS == {'scripts/collect_boards_lean.py'}
    monkeypatch.setattr(replay, 'WORKTREE', Path(__file__).resolve().parents[1])
    changed = ['src/phase_j/live_speed.py', 'scripts/collect_boards_lean.py', 'scripts/run_live_pipeline_20260928.py',
               'scripts/run_live_x_video30.py', 'scripts/measure_live_x_timing.py', 'src/production_config.py',
               'src/phase_j/overlay_x_prefire.js']
    monkeypatch.setattr(replay, 'git_lines', lambda *args: changed if args[0] == 'diff' else [])
    assert replay.overlay_modules() == ['scripts/collect_boards_lean.py', 'src/phase_j/live_speed.py',
                                        'src/production_config.py']

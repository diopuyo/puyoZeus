"""video_30 の流し直しの入口: 入力仕様 (区間・stride・試合)、計画の中身、門、受領票の合否、子の入力差し替え。"""
from __future__ import annotations

import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import live_x_replay_child as child
from scripts import live_x_video30_spec as spec
from scripts import run_live_x_replay as replay
from scripts import run_live_x_video30 as video30


def saved_dir(tmp_path: Path) -> Path:
    saved = tmp_path/'saved'
    saved.mkdir()
    (saved/'puyo_live.json').write_text(json.dumps(dict(
        source='dshow', device_name='OBS Virtual Camera', device_index=3, port=8765, output_dir='x',
        dshow_color_correction='601to709', b3_judgment=True, live_v49f=True, live_x_prefire=True)), encoding='utf-8')
    (saved/'recognition_config.json').write_text(json.dumps(dict(
        enable_placement_signal_reconcile=False, stable_frame_count=3)), encoding='utf-8')
    return saved


def arguments(tmp_path: Path, *extra: str) -> object:
    return video30.arguments(['plan', '--root', str(tmp_path/'r'), *extra])


# --- 入力仕様 ---------------------------------------------------------------------------------

def test_start_sec_maps_to_first_processed_frame() -> None:
    start = spec.live_start_sec()
    assert start == 176.0
    # 配信ソフトは int((start_sec-warmup)*fps) コマ目から認識し、int(start_sec*fps) コマ目から評価・公開する。
    assert int((start-spec.WARMUP_SEC)*spec.VIDEO_FPS) == spec.first_frame() == 10500
    assert spec.write_frame(start) == 10560


def test_warmup_constant_matches_launcher() -> None:
    from src.phase_j.launcher import VIDEO_WARMUP_SEC
    assert spec.WARMUP_SEC == VIDEO_WARMUP_SEC


def test_start_sec_refuses_rounding_mismatch() -> None:
    with pytest.raises(ValueError, match='丸め'):
        spec.live_start_sec(first_sec=1./60., fps=60.)   # (1/60+1-1)*60 が 1 に届かず int で 0 コマ目になる


def test_stride_and_processed_frames_for_60fps() -> None:
    assert spec.stride_for(60.) == 2 and spec.stride_for(30.) == 1
    frames = spec.processed_frames(spec.first_frame(), spec.end_frame(), spec.stride_for())
    assert frames[0] == 10500 and frames[-1] == 17038 and len(frames) == 3270
    assert spec.end_frame() == 17040 == spec.VIDEO_FRAMES


def test_workload_names_its_unit() -> None:
    work = spec.workload()
    assert work == dict(first_frame=10500, end_frame=17040, stride=2, processed_frames=3270,
                        decoded_frames=6540, video_seconds=109.0, evaluated_from_frame=10560)


def test_match_ranges_match_verification_json() -> None:
    for match in spec.MATCHES:
        first, last = match.frames()
        assert last-first+1 == spec.EXPECTED_MATCH_FRAMES[match.index]
    assert [m.frames() for m in spec.MATCHES] == [(11276, 13296), (13966, 16726)]


@pytest.mark.skipif(not (spec.REVIEW_ROOT/'manifest.json').exists(), reason='レビュー成果物が無い環境')
def test_spec_agrees_with_review_manifest() -> None:
    manifest = json.loads((spec.REVIEW_ROOT/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['capture_start'] == spec.FIRST_PROCESSED_SEC and manifest['capture_end'] == spec.END_SEC
    assert [(m['first_move'], m['last_scoring_row']) for m in manifest['matches']] == [
        (m.first_move_sec, m.last_scoring_sec) for m in spec.MATCHES]


# --- 門 ---------------------------------------------------------------------------------------

def test_run_without_token_creates_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(replay.CONFIRM_ENV, raising=False)
    monkeypatch.setattr(replay, 'blocked', lambda: pytest.fail('門より先へ進んだ'))
    root = tmp_path/'replay'
    with pytest.raises(SystemExit, match='起動しない'):
        video30.main(['run', '--root', str(root)])
    assert not root.exists() and list(tmp_path.iterdir()) == []


def test_run_refuses_wrong_video_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(replay.CONFIRM_ENV, replay.CONFIRM_WORD)
    monkeypatch.setattr(replay, 'blocked', lambda: None)
    monkeypatch.setattr(replay, 'digest', lambda path: 'x'*64)
    root = tmp_path/'replay'
    with pytest.raises(SystemExit, match='SHA256'):
        video30.main(['run', '--root', str(root), '--token', replay.RUN_TOKEN])
    assert not root.exists()


# --- 計画 -------------------------------------------------------------------------------------

@pytest.mark.parametrize('arm,preset,expected', [
    ('OFF', 'direct', dict(live_x_direct_publish=False, live_x_prefire_shadow=False, live_x_prefire=False)),
    ('ON', 'direct', dict(live_x_direct_publish=True, live_x_prefire_shadow=False, live_x_prefire=False)),
    ('ON', 'integrated', dict(live_x_direct_publish=True, live_x_prefire_shadow=True, live_x_prefire=True)),
    ('OFF', 'integrated', dict(live_x_direct_publish=False, live_x_prefire_shadow=False, live_x_prefire=False))])
def test_config_overrides_input_range_and_x_flags(arm: str, preset: str, expected: dict, tmp_path: Path) -> None:
    config = video30.user_config(arm, tmp_path/'out', spec.FIRST_PROCESSED_SEC, spec.END_SEC,
                                 video30.ON_PRESETS[preset], saved_dir(tmp_path))
    assert config['source'] == 'video' and config['video_path'] == str(spec.VIDEO)
    assert (config['start_sec'], config['end_sec']) == (176.0, 284.0)
    assert config['port'] == replay.PORT and config['output_dir'] == str(tmp_path/'out'/'output')
    assert {k: config[k] for k in video30.X_FLAGS} == expected
    assert config['b3_judgment'] is True and config['live_v49f'] is True      # 21c の設定は保つ
    assert not {'device_name', 'device_index', 'dshow_color_correction'} & config.keys()


@pytest.mark.parametrize('condition', ['F', 'L'])
def test_child_plan_disables_only_input_rewrite(condition: str, tmp_path: Path) -> None:
    plan = video30.child_plan(condition, tmp_path, ['src/x.py'], saved_dir(tmp_path))
    assert plan['crop'] is None and plan['offset_sec'] == 0. and plan['color_correction'] is None
    assert plan['realtime'] is replay.CONDITIONS[condition]
    assert plan['recognition_overrides'] == dict(enable_placement_signal_reconcile=False)
    assert plan['source_modules'] == ['src/x.py'] and plan['patch_log'] == str(tmp_path/'patches.jsonl')


def test_plan_is_side_effect_free_abba(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(replay, 'overlay_modules', lambda: ['src/phase_j/live_v49f.py'])
    monkeypatch.setattr(replay, 'bundle_drift', lambda modules: [])
    monkeypatch.setattr(video30, 'user_config', lambda arm, *rest: dict(live_x_direct_publish=arm == 'ON'))
    monkeypatch.setattr(video30, 'child_plan', lambda condition, out, modules: dict(modules=modules))
    runs = video30.plan(arguments(tmp_path, '--condition', 'L'))
    assert [r['arm'] for r in runs] == list(replay.ARMS)
    assert [r['name'] for r in runs] == ['L1_OFF', 'L2_ON', 'L3_ON', 'L4_OFF']
    assert all('--dev-allow-video' in r['command'] and r['command'][-2] == '--skip-manifest' for r in runs)
    assert not (tmp_path/'r').exists()


def test_plan_document_leads_with_workload_and_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(replay, 'overlay_modules', lambda: [])
    monkeypatch.setattr(replay, 'bundle_drift', lambda modules: [])
    document = video30.plan_document(arguments(tmp_path, '--on', 'integrated'))
    assert document['workload']['processed_frames'] == 3270 and document['on_flags'] == list(video30.X_FLAGS)
    assert [m['frames'] for m in document['matches']] == [[11276, 13296], [13966, 16726]]
    assert document['gate']['token'] == replay.RUN_TOKEN and document['video_sha256'] == spec.VIDEO_SHA256
    assert [r['name'] for r in document['runs']] == ['F1_OFF', 'F2_ON', 'F3_ON', 'F4_OFF']


def test_plan_refuses_bundle_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(replay, 'overlay_modules', lambda: [])
    monkeypatch.setattr(replay, 'bundle_drift', lambda modules: ['src/b3_state.py'])
    with pytest.raises(SystemExit, match='ソースが違う'):
        video30.plan(arguments(tmp_path))


@pytest.mark.parametrize('first,end', [(183.0, 284.), (-1., 284.), (175., 270.), (175., 290.)])
def test_plan_refuses_ranges_that_miss_the_matches(first: float, end: float) -> None:
    with pytest.raises(SystemExit, match='--first-sec|--end-sec'):
        video30.check_range(first, end)


def test_default_range_passes_check() -> None:
    video30.check_range(spec.FIRST_PROCESSED_SEC, spec.END_SEC)


# --- 受領票の合否 -----------------------------------------------------------------------------

def receipt(**override: object) -> dict:
    base = dict(exit_code=0, patches=dict(video_source=2), recognition_matches_21c=True,
                expected_review_frames=3000, evaluated_review_frames=3000, coverage=1.0)
    return dict(base, **override)


@pytest.mark.parametrize('condition,override,reason', [
    ('F', {}, None), ('L', dict(coverage=.4, evaluated_review_frames=1200), None),
    ('F', dict(exit_code=1), '終了コード'), ('F', dict(patches=dict(video_source=1)), '差し替え件数'),
    ('F', dict(recognition_matches_21c=False), '21c'), ('L', dict(evaluated_review_frames=0, coverage=0.), '母数'),
    ('F', dict(expected_review_frames=0, evaluated_review_frames=0), '母数'),
    ('F', dict(coverage=.9, evaluated_review_frames=2700), '下限')])
def test_accept_requires_denominator_and_f_coverage(condition: str, override: dict, reason: str | None) -> None:
    failure = video30.accept(receipt(**override), condition)
    assert (failure is None) if reason is None else (failure is not None and reason in failure)


def write_evaluations(out: Path, frames: list[int]) -> None:
    path = out/video30.EVALUATIONS
    path.parent.mkdir(parents=True)
    with path.open('wb') as stream:
        for frame in frames:
            pickle.dump(dict(frame=frame, t_sec=frame/60.), stream)


def test_output_check_counts_expected_frames_in_review_range(tmp_path: Path) -> None:
    review_end = spec.REVIEW_FIRST_FRAME+spec.REVIEW_FRAME_COUNT
    frames = list(range(10560, 17040, 2))
    write_evaluations(tmp_path, frames[:-100]+[10501])   # 末尾 100 コマ分は欠け・格子に乗らない 1 行は数えない
    result = video30.output_check(tmp_path, spec.FIRST_PROCESSED_SEC, spec.END_SEC)
    assert result['expected_review_frames'] == len(range(11040, review_end, 2)) == 3000
    assert result['evaluated_review_frames'] == 2900 and result['coverage'] == pytest.approx(2900/3000)
    assert result['workload']['stride'] == 2


def test_output_check_without_spool_reports_zero(tmp_path: Path) -> None:
    result = video30.output_check(tmp_path, spec.FIRST_PROCESSED_SEC, spec.END_SEC)
    assert result['evaluated_review_frames'] == 0 and result['coverage'] == 0.


# --- 子の入力差し替え (crop=None は何も書き換えない) --------------------------------------------

def test_child_without_crop_keeps_image_and_time(tmp_path: Path) -> None:
    from src.phase_j.live_source import CapturedFrame
    image = np.full((1080, 1920, 3), 77, np.uint8)

    class Source:
        def __iter__(self):
            yield CapturedFrame(10560, 176., 5., 5., image)
    module = SimpleNamespace(VideoFileSource=Source)
    plan = dict(crop=None, offset_sec=0., color_correction=None, patch_log=str(tmp_path/'p.jsonl'))
    child.cropped_source(plan)(module)
    frame = next(iter(module.VideoFileSource()))
    assert frame.image is image and frame.media_sec == 176. and frame.index == 10560 and frame.captured_at == 5.
    assert json.loads((tmp_path/'p.jsonl').read_text())['patch'] == 'video_source'   # 受領票の件数は数える


def test_child_without_crop_key_at_all_also_works(tmp_path: Path) -> None:
    from src.phase_j.live_source import CapturedFrame
    image = np.zeros((1080, 1920, 3), np.uint8)

    class Source:
        def __iter__(self):
            yield CapturedFrame(1, 1., 0., 0., image)
    module = SimpleNamespace(VideoFileSource=Source)
    child.cropped_source(dict(offset_sec=2., patch_log=str(tmp_path/'p.jsonl')))(module)
    assert next(iter(module.VideoFileSource())).media_sec == 3.


# --- 配信ソフト本体の設定検証を通ること (起動しないで、起動口と同じ検証を通す) ---------------------------------

needs_21c = pytest.mark.skipif(not (replay.SAVED/'puyo_live.json').exists(), reason='21c の保存設定が無い環境')


def real_config(arm: str, preset: str, tmp_path: Path) -> dict:
    """21c の実設定から作った video_30 の設定。モデルの置き場だけ試験用の実在ファイルへ向ける。"""
    model, table = tmp_path/'model.joblib', tmp_path/'timing_table.json'
    model.write_bytes(b'x')
    table.write_bytes(b'{}')
    config = video30.user_config(arm, tmp_path/'out', spec.FIRST_PROCESSED_SEC, spec.END_SEC, video30.ON_PRESETS[preset])
    return dict(config, b3_model=str(model), b3_timing_table=str(table))


@needs_21c
@pytest.mark.parametrize('arm,preset', [('OFF', 'direct'), ('ON', 'direct'), ('ON', 'integrated')])
def test_generated_config_passes_launcher_validation(arm: str, preset: str, tmp_path: Path) -> None:
    from src.phase_j import launcher
    user = launcher.parse_user_config(real_config(arm, preset, tmp_path), tmp_path, allow_video=True)
    pipeline = launcher.build_pipeline_config(user)
    # 認識は int((176-1)*60)=10500 コマ目から、評価・公開は int(176*60)=10560 コマ目から。終端は 284 秒。
    assert (pipeline['start_sec'], pipeline['end_sec'], pipeline['warmup_sec']) == (176.0, 284.0, 1.0)
    assert user.video_path == spec.VIDEO and user.live_x_direct_publish is (arm == 'ON')
    assert user.live_x_prefire is user.live_x_prefire_shadow is (arm == 'ON' and preset == 'integrated')
    assert user.live_v49f and user.b3_async and user.b3_m3          # 21c の B3 M3・v4.9f の構成は保つ


def test_all_preset_adds_trial_flags_to_the_on_arm_only(tmp_path: Path) -> None:
    """23y の ALL (X + コマ落ちに強い状態判定 + 予告 CNN の試作 2 つ)。OFF の腕の設定は従来と同じで何も足さない。"""
    from src.phase_j.live_speed import SPEED_TRIAL_KEYS, TRIAL_FLAG_KEYS, X_TRIAL_KEYS
    saved = saved_dir(tmp_path)
    all_keys = X_TRIAL_KEYS+SPEED_TRIAL_KEYS   # 23y の ALL。24a 以降の試作フラグ (TRIAL_FLAG_KEYS の後ろ) は preset に含めない
    on = video30.user_config('ON', tmp_path/'out', spec.FIRST_PROCESSED_SEC, spec.END_SEC, video30.ON_PRESETS['all'], saved)
    off = video30.user_config('OFF', tmp_path/'out', spec.FIRST_PROCESSED_SEC, spec.END_SEC, video30.ON_PRESETS['all'], saved)
    assert video30.ON_PRESETS['all'] == all_keys and set(SPEED_TRIAL_KEYS) <= set(on)
    assert all(on[key] is True for key in all_keys)
    assert not any(on.get(key) for key in TRIAL_FLAG_KEYS if key not in all_keys)   # 後から足した鍵は ON にしない
    assert not any(off.get(key) for key in TRIAL_FLAG_KEYS) and not set(SPEED_TRIAL_KEYS) & set(off)
    direct = video30.user_config('ON', tmp_path/'out', spec.FIRST_PROCESSED_SEC, spec.END_SEC, video30.ON_PRESETS['direct'], saved)
    assert not set(SPEED_TRIAL_KEYS) & set(direct)                       # 既存の preset は新しい鍵を足さない

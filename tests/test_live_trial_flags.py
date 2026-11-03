"""試作フラグ (X・コマ落ちに強い状態判定・予告 CNN の試作) の実効値の記録 (live_trial_flags.json)。

OBS 試験版 23y は、同じ入口から OFF / X / ALL の 3 つの設定を切り替えて走らせる。
設定が puyo_live.json から実行時の process まで届いたことを、走行の記録だけで後から確かめられるようにする。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

from src.phase_j import live_speed
from src.phase_j.live_speed import (ARM_ALL, ARM_CUSTOM, ARM_OFF, ARM_X, SPEED_DEFAULTS, SPEED_TRIAL_KEYS,
                                    TRIAL_FLAG_KEYS, TRIAL_FLAGS_FILE, X_TRIAL_KEYS, classify_arm,
                                    trial_flags_record, write_trial_flags)

ENV_ON, ENV_OFF = '1', '0'


def test_trial_keys_are_known_speed_flags_and_default_off() -> None:
    """記録するフラグは配線済み (SPEED_DEFAULTS にある) で、どれも既定 OFF。順序は X → 速度の試作。"""
    assert TRIAL_FLAG_KEYS == X_TRIAL_KEYS+SPEED_TRIAL_KEYS+(live_speed.MATCH_END_PAIR_KEY, live_speed.VERIFY_REASON_KEY, 'b3_warning_async', live_speed.NATIVE_HSV_MAIN_KEY, live_speed.VERIFY_ASYNC_KEY, 'live_match_end_feed_on_hold', 'match_end_yatta_text_template', live_speed.CAPTURE_RECONNECT_KEY, live_speed.WORKER_GRACEFUL_STOP_KEY, live_speed.BOARD_VECTORIZED_KEY, live_speed.VERIFY_SOFT_HOLD_KEY)
    assert len(set(TRIAL_FLAG_KEYS)) == len(TRIAL_FLAG_KEYS) == 18
    for key in TRIAL_FLAG_KEYS:
        assert SPEED_DEFAULTS[key] is False, key


@pytest.mark.parametrize('enabled, arm', [
    ((), ARM_OFF),
    (X_TRIAL_KEYS, ARM_X),
    (TRIAL_FLAG_KEYS, ARM_ALL),
    (SPEED_TRIAL_KEYS, ARM_CUSTOM),          # X 無しの速度だけ
    (X_TRIAL_KEYS[:1], ARM_CUSTOM),          # X の一部だけ
    (live_speed.ARM_TRIAL_KEYS[:-1], ARM_CUSTOM),  # 既存6項目から 1 つ欠けた
])
def test_classify_arm(enabled: tuple[str, ...], arm: str) -> None:
    assert classify_arm({key: key in enabled for key in TRIAL_FLAG_KEYS}) == arm


def test_classify_arm_ignores_unknown_and_missing_keys() -> None:
    assert classify_arm({}) == ARM_OFF
    assert classify_arm({'live_row1_guard': True}) == ARM_OFF   # 本番 ON 済みのフラグは試作ではない


def make_options(enabled: tuple[str, ...] = (), **extra: object) -> argparse.Namespace:
    return argparse.Namespace(**{key: key in enabled for key in TRIAL_FLAG_KEYS}, **extra)


def test_record_has_flags_env_and_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, 'environ', {'PUYO_'+k.upper(): ENV_ON if k in X_TRIAL_KEYS else ENV_OFF
                                        for k in TRIAL_FLAG_KEYS})
    record = trial_flags_record(make_options(X_TRIAL_KEYS))
    assert record['schema'] == live_speed.TRIAL_FLAGS_SCHEMA and record['arm'] == ARM_X
    from src.production_config import LIVE_FRAME_DROP_ROBUST_SITES
    assert record['frame_drop_robust_sites'] == sorted(LIVE_FRAME_DROP_ROBUST_SITES) == []   # 2026-10-23 夜の既定 (R15 も外す)
    assert record['flags'] == {k: k in X_TRIAL_KEYS for k in TRIAL_FLAG_KEYS}
    assert record['env'] == {'PUYO_'+k.upper(): ENV_ON if k in X_TRIAL_KEYS else ENV_OFF for k in TRIAL_FLAG_KEYS}


def test_record_shows_missing_env_as_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """環境変数が届いていなければ None で残る (flags と食い違うので検査で見つかる)。"""
    monkeypatch.setattr(os, 'environ', {})
    record = trial_flags_record(make_options(TRIAL_FLAG_KEYS))
    assert record['arm'] == ARM_ALL
    assert set(record['env'].values()) == {None}


def test_record_uses_default_off_for_options_without_the_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, 'environ', {})
    record = trial_flags_record(argparse.Namespace())
    assert record['arm'] == ARM_OFF and not any(record['flags'].values())


def test_write_creates_file_even_when_all_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """OFF でも書く (ファイルが無い = 記録できなかった、と区別するため)。"""
    monkeypatch.setattr(os, 'environ', {})
    write_trial_flags(make_options(output=tmp_path))
    record = json.loads((tmp_path/TRIAL_FLAGS_FILE).read_text(encoding='utf-8'))
    assert record['arm'] == ARM_OFF and list(record['flags']) == list(TRIAL_FLAG_KEYS)


def test_configure_env_matches_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """configure が子 process へ渡す環境変数と、記録の flags・env が一致する (ALL)。"""
    monkeypatch.setattr(os, 'environ', {})
    options = argparse.Namespace(source='video', cpu_threads=1, live_v49f=True,
                                 **{key: True for key in TRIAL_FLAG_KEYS})
    live_speed.configure(options, argparse.ArgumentParser())
    record = trial_flags_record(options)
    assert record['arm'] == ARM_ALL
    assert all(value == ENV_ON for value in record['env'].values())
    assert live_speed.frame_drop_robust_enabled()
    assert live_speed.observation_options().get('enable_warning_batch') is True
    assert live_speed.recognition_config({}, False)['enable_media_tick_counters'] is True


def test_configure_off_leaves_recognition_config_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """OFF の bit-identical: 認識設定に鍵を足さず、観測器の追加引数も無い。"""
    monkeypatch.setattr(os, 'environ', {})
    live_speed.configure(argparse.Namespace(source='video', cpu_threads=1), argparse.ArgumentParser())
    assert 'enable_media_tick_counters' not in live_speed.recognition_config(dict(stable_frame_count=3), False)
    assert live_speed.observation_options() == {}
    assert not live_speed.frame_drop_robust_enabled()


def test_configure_writes_record_only_when_output_dir_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ランチャが output を作った後の configure が記録を書く。output が無ければ何も作らない (単体の呼び出しで勝手に作らない)。"""
    monkeypatch.setattr(os, 'environ', {})
    options = argparse.Namespace(source='video', cpu_threads=1, live_v49f=True, output=tmp_path/'later',
                                 **{key: True for key in TRIAL_FLAG_KEYS})
    live_speed.configure(options, argparse.ArgumentParser())
    assert not (tmp_path/'later').exists()                                   # 作らない
    (tmp_path/'later').mkdir()
    live_speed.configure(options, argparse.ArgumentParser())
    record = json.loads((tmp_path/'later'/TRIAL_FLAGS_FILE).read_text(encoding='utf-8'))
    assert record['arm'] == ARM_ALL and all(value == ENV_ON for value in record['env'].values())   # env 設定の後に書く
    live_speed.configure(argparse.Namespace(source='video', cpu_threads=1), argparse.ArgumentParser())  # output 属性なし = 何もしない
    assert live_speed.write_trial_flags_if_ready(argparse.Namespace(output='not-a-path')) is False


def test_run_live_pipeline_script_is_unchanged_from_the_distributed_one() -> None:
    """配布 app が実行時に読む scripts は台が差し替えられない (ROOT が __file__ 由来のため)。記録は src から書き、この script は変えない。

    基準は 24c で二重起動の排他 bind を足したコミット (387e73f9)。23y/24b の配布物はそれ以前の版で、24c の配布物から新しい版になる。"""
    import subprocess
    root = Path(__file__).resolve().parents[1]
    if subprocess.run(['git', '-C', str(root), 'cat-file', '-e', '387e73f9^{commit}'], capture_output=True).returncode:
        pytest.skip('基準点 387e73f9 がこの clone に無い')
    changed = subprocess.run(['git', '-C', str(root), 'diff', '--quiet', '387e73f9', 'HEAD', '--',
                              'scripts/run_live_pipeline_20260928.py'])
    assert changed.returncode == 0


@pytest.mark.parametrize('enabled', [True, False])
def test_r37_is_appended_and_recorded_each_start(
    enabled: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """起動ごとに R37 の実効値を上書きし、既存の鍵順序と ALL 分類を保つ。"""
    monkeypatch.setattr(os, 'environ', {})
    legacy = ('live_x_direct_publish', 'live_x_prefire_shadow', 'live_x_prefire',
              'live_frame_drop_robust', 'b3_warning_batch', 'b3_warning_prefetch')
    key = live_speed.SCORE_ZERO_THEN_MOVE_KEY
    options = argparse.Namespace(source='video', cpu_threads=1, live_v49f=True, output=tmp_path,
                                 **{name: True for name in legacy}, **{key: not enabled})
    live_speed.configure(options, argparse.ArgumentParser())
    setattr(options, key, enabled)
    live_speed.configure(options, argparse.ArgumentParser())
    record = json.loads((tmp_path/TRIAL_FLAGS_FILE).read_text(encoding='utf-8'))
    assert list(record['flags']) == [*legacy, key, live_speed.MATCH_END_PAIR_KEY,
                                     live_speed.VERIFY_REASON_KEY, 'b3_warning_async', live_speed.NATIVE_HSV_MAIN_KEY, live_speed.VERIFY_ASYNC_KEY,
                                     'live_match_end_feed_on_hold', 'match_end_yatta_text_template',
                                     live_speed.CAPTURE_RECONNECT_KEY, live_speed.WORKER_GRACEFUL_STOP_KEY,
                                     live_speed.BOARD_VECTORIZED_KEY, live_speed.VERIFY_SOFT_HOLD_KEY]
    assert list(record['env']) == ['PUYO_'+name.upper() for name in record['flags']]
    assert record['flags'][key] is enabled
    assert record['env'][live_speed.SCORE_ZERO_THEN_MOVE_ENV] == str(int(enabled))
    assert record['arm'] == ARM_ALL

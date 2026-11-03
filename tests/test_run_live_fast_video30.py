"""予告 CNN 試作の流し直し入口: 腕ごとのフラグと、合言葉なしでは何も作らないこと。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def saved_dir(tmp_path: Path) -> Path:
    saved = tmp_path/'saved'
    saved.mkdir()
    (saved/'puyo_live.json').write_text(json.dumps(dict(source='dshow', device_name='OBS', device_index=3,
        dshow_color_correction='601to709', b3_warning_parallel=True, live_x_direct_publish=True)), encoding='utf-8')
    return saved


def test_arms_only_differ_in_fast_flags(tmp_path: Path) -> None:
    from scripts import run_live_fast_video30 as fast
    saved = saved_dir(tmp_path)
    off = fast.user_config('OFF', tmp_path/'a', 175.0, 284.0, saved)
    on = fast.user_config('ON', tmp_path/'a', 175.0, 284.0, saved)
    assert {k: off[k] for k in fast.FAST_FLAGS} == dict(b3_warning_batch=False, b3_warning_prefetch=False)
    assert {k: on[k] for k in fast.FAST_FLAGS} == dict(b3_warning_batch=True, b3_warning_prefetch=True)
    assert {k: v for k, v in on.items() if k not in fast.FAST_FLAGS} == \
        {k: v for k, v in off.items() if k not in fast.FAST_FLAGS}
    assert off['live_x_direct_publish'] is False and off['b3_warning_parallel'] is True
    assert 'device_name' not in off and 'dshow_color_correction' not in off


def test_run_refuses_without_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_live_fast_video30 as fast
    monkeypatch.delenv(fast.CONFIRM_ENV, raising=False)
    root = tmp_path/'replay'
    with pytest.raises(SystemExit, match='起動しない'):
        fast.main(['run', '--root', str(root), '--token', fast.RUN_TOKEN])
    monkeypatch.setenv(fast.CONFIRM_ENV, fast.CONFIRM_WORD)
    with pytest.raises(SystemExit, match='起動しない'):
        fast.main(['run', '--root', str(root), '--token', 'wrong'])
    assert not root.exists()

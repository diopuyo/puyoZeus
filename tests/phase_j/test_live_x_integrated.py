"""配信の X 段1 (live_x_direct_publish) と段3 (live_x_prefire) を統合したときの契約。

2 本のブランチは別々に試験済みだが、同時 ON の組合せは統合で初めて生まれる。ここでは
- どの X フラグも既定 OFF で、設定の入口 (production_config / SPEED_DEFAULTS / UserConfig / 環境変数) に両方残る
- 段1 ON でも、段3 の記録 (live_x_prefire.jsonl) と summary が close で閉じる
- 判定ID は (従来 6 要素, 段3 の印, 段1 の X 完了時刻) の順に伸びる
- 配布物 (import 閉包 + src/phase_j 丸ごと) に新規 3 ファイルが入る
を見る。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.phase_j import live_v49f_prefire as prefire
from src.phase_j.live_x_direct import ENV_NAME as ENV_DIRECT, direct_publish_enabled
from tests.phase_j.test_live_x_prefire import SHOWN
from tests.test_live_x_direct_publish import RESULT_TIMEOUT_SEC, X_BASE, owner, value

ROOT = Path(__file__).resolve().parents[2]
X_KEYS = ('live_x_direct_publish', 'live_x_prefire_shadow', 'live_x_prefire')
X_DEFAULT_NAMES = ('LIVE_X_DIRECT_PUBLISH_DEFAULT', 'LIVE_X_PREFIRE_SHADOW_DEFAULT', 'LIVE_X_PREFIRE_DEFAULT')
NEW_BUNDLE_FILES = ('src/phase_j/live_x_direct.py', 'src/phase_j/live_v49f_prefire.py',
                    'src/phase_j/overlay_x_prefire.js')
B3_ID_LEN = 6          # LiveB3.decorate が作る従来の判定ID
GENERATION, RESULT_FRAME, X_DONE_AT = 2, 7, 12.5


def test_every_x_flag_is_default_off_and_wired_in_every_entry(tmp_path: Path) -> None:
    from src import production_config
    from src.phase_j.launcher import build_pipeline_config, parse_user_config
    from src.phase_j.live_speed import SPEED_DEFAULTS
    assert [getattr(production_config, name) for name in X_DEFAULT_NAMES] == [False, False, False]
    assert all(SPEED_DEFAULTS[key] is False for key in X_KEYS)
    base = dict(device_name='OBS Virtual Camera')
    user = parse_user_config(base, tmp_path)
    assert all(getattr(user, key) is False for key in X_KEYS)
    assert not set(X_KEYS) & set(build_pipeline_config(user))   # OFF は pipeline_config に出ない
    enabled = parse_user_config(dict(base, **{key: True for key in X_KEYS}), tmp_path)
    assert all(build_pipeline_config(enabled)[key] is True for key in X_KEYS)


def test_all_on_reaches_spawn_children_through_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.phase_j.live_speed import SPEED_DEFAULTS, configure
    monkeypatch.setattr(os, 'environ', {})
    options = argparse.Namespace(**dict(SPEED_DEFAULTS, **{key: True for key in X_KEYS}), cpu_threads=1,
        source='dshow', b3_judgment=True, split_evaluation=True, worker_mode='process', compare=False,
        live_v49f=True)
    configure(options, argparse.ArgumentParser())
    assert direct_publish_enabled() is True
    assert prefire.flags() == (True, True)
    assert os.environ[ENV_DIRECT] == '1'


def packet(index: int) -> dict[str, Any]:
    """段3 が足す 'prefire' つきの v4.9f 入力。作業側 (echo) へは 'prefire' を除いて届く。"""
    data = dict(t=float(index), events=[dict(kind='provisional', t=.9, **SHOWN),
                                       dict(kind='table', t=.5, build_ms=.4)],
                shown=[SHOWN, None], cost_ms=1.5, read_ms=.7, reads=2)
    return dict(game=0, t=float(index), origins=[], echo_base=X_BASE, echo_value=value([40]), prefire=data)


def test_direct_publish_on_still_closes_prefire_records(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """統合の落とし穴: 段1 の close_direct が早期 return で段3 の close を飛ばすと summary が消える。"""
    monkeypatch.setenv(prefire.ENV_DISPLAY, '1')
    view = owner(monkeypatch, tmp_path, True)
    assert view.direct is True and view.prefire is not None
    try:
        view.update(dict(frame=0, game_idx=0, _v49f=packet(0)), 1)
        result = view.outputs.get(timeout=RESULT_TIMEOUT_SEC)
        assert result['x_fresh'] is True                       # 段1: X の初出
        row = dict(t_sec=1.25, raw_probability=.6, match_active=True, judgment_id=(1, 2))
        view.decorate(row)
        assert row['x_provisional']['sides'][0]['score'] == SHOWN['score']   # 段3: 別欄
        assert row['judgment_id'][:2] == (1, 2) and len(row['judgment_id']) == 3
    finally:
        view.close()
    summary = json.loads((tmp_path/'live_x_prefire_summary.json').read_text(encoding='utf-8'))
    assert summary['counts'] == dict(provisional=1, table=1) and summary['cost_ms']['count'] == 1
    assert len((tmp_path/'live_x_prefire.jsonl').read_text(encoding='utf-8').splitlines()) == 1
    status = json.loads((tmp_path/'v49f_display_status.json').read_text(encoding='utf-8'))
    assert status['direct_publish'] is True and status['pump_alive'] is False
    assert (status['x_fresh'], status['x_fresh_fires'], status['x_wakes']) == (1, 1, 1)   # 段1 の起床の記録


def test_prefire_off_with_direct_on_writes_no_prefire_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(prefire.ENV_DISPLAY, raising=False)
    monkeypatch.delenv(prefire.ENV_SHADOW, raising=False)
    view = owner(monkeypatch, tmp_path, True)
    try:
        assert view.direct is True and view.prefire is None
    finally:
        view.close()
    assert not (tmp_path/'live_x_prefire.jsonl').exists()
    assert not (tmp_path/'live_x_prefire_summary.json').exists()


def test_judgment_id_grows_in_fixed_order_when_both_on(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.phase_j.live_b3 import LiveB3
    monkeypatch.delenv('PUYO_B3_M3', raising=False)
    monkeypatch.delenv('PUYO_OBS_FAST_AUDIT', raising=False)
    view = prefire.PrefireView(tmp_path, display=True)
    view.shown = [SHOWN, None]
    b3 = LiveB3.__new__(LiveB3)
    b3.completion_driven, b3.generation, b3.result_frame = True, GENERATION, RESULT_FRAME
    b3.display = SimpleNamespace(shown=(1., .55), source='null', raw=.5, game=0, contents=None)
    b3.ama = b3.time_gate = b3.v49f_baseline = None
    b3.v49f = SimpleNamespace(direct=True, x_completed_at=X_DONE_AT,
        decorate=lambda row, *a, **k: (row.update(v49f={}), view.decorate(row)))
    row = dict(match_active=True, t_sec=1.25)
    b3.decorate(row)
    view.close()
    marks = ((SHOWN['status'], SHOWN['landing_t'], SHOWN['score']), None)
    assert row['judgment_id'] == (GENERATION, RESULT_FRAME, 'null', .5, True, 0, marks, X_DONE_AT)
    assert len(row['judgment_id']) == B3_ID_LEN + 2


def import_bundle_tools() -> tuple[Any, Any]:
    sys.path.insert(0, str(ROOT/'packaging'))
    try:
        import build_bundle
        import import_closure
    finally:
        sys.path.remove(str(ROOT/'packaging'))
    return build_bundle, import_closure


def test_new_modules_and_overlay_script_are_in_the_distribution() -> None:
    """配布物は「import 閉包の .py」+「資産リストの src/phase_j 丸ごと」。新規 3 ファイルが両方から落ちない。"""
    build_bundle, import_closure = import_bundle_tools()
    closure: set[str] = set()
    for entry in build_bundle.ENTRY_MODULES:
        modules, _ = import_closure.internal_closure(ROOT, entry)
        closure |= set(modules)
    assert {'src.phase_j.live_x_direct', 'src.phase_j.live_v49f_prefire'} <= closure
    assert 'src/phase_j' in build_bundle.read_asset_list()
    expanded = set(build_bundle.expand_asset('src/phase_j', ROOT))
    assert set(NEW_BUNDLE_FILES) <= expanded
    assert not any(build_bundle.is_runtime_file(f'app/{rel}') for rel in NEW_BUNDLE_FILES)

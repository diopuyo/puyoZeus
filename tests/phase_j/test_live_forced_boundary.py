"""強制境界 (入力世代切替) が得点リセット境界の5秒デバウンス枠を消費しないことを確認する。

2026-10-11 OBS通し試験 (soak game11): 強制境界の3.81秒後に来た本物の得点リセット境界が
デバウンスで拒否され、前試合の死亡確定が次試合に残った。注入文 (adapt_loop) を、本物の
accept_formal_boundary / 境界ブロックと同形の最小generate (再生台 scripts/replay_forced_boundary_20261011.py
の run_mini_generate) へ適用して、境界の通り方を検証する。
"""
from __future__ import annotations

import ast
import inspect
from typing import Any

import pytest

from src.phase_j import live_bridge
from src.phase_j.live_bridge import (
    ISOLATED_FORCED_BOUNDARY_AFTER, ISOLATED_FORCED_BOUNDARY_BEFORE, LEGACY_FORCED_BOUNDARY_BEFORE,
    RecognitionBridge, adapt_loop, build_live_generate, forced_boundary_statements)
from scripts.replay_forced_boundary_20261011 import run_mini_generate

MID_SCORES = (29247, 15423)   # 試合中の得点 (soak t=699.91 直前の値)
ZERO = (0, 0)                 # 得点リセット後
INCIDENT_GAP_SEC = 3.81       # 強制境界から本物の得点リセット境界までの実測差


def run(events: list[tuple[float, tuple[int, int], bool]], isolate: bool) -> list[tuple[int, int, Any]]:
    """events = (t_sec, 得点, 強制境界か)。戻りは各通知の (frame, game_idx, デバウンス枠の時刻)。"""
    return run_mini_generate(events, isolate)


def games(trace: list[tuple[int, int, Any]]) -> list[int]:
    return [game for _, game, _ in trace]


def windows(trace: list[tuple[int, int, Any]]) -> list[Any]:
    return [window for _, _, window in trace]


# --- (a) 強制境界単独 -------------------------------------------------------

def test_forced_only_advances_game_but_keeps_debounce_window() -> None:
    events = [(9.0, (500, 300), False), (10.0, ZERO, False), (11.0, (100, 50), False),
              (50.0, (900, 700), True), (51.0, (950, 700), False)]
    on = run(events, True)
    assert games(on) == [0, 1, 1, 2, 2]
    assert windows(on) == [None, 10.0, 10.0, 10.0, 10.0]  # 強制境界 (t=50) でも枠は10.0のまま
    off = run(events, False)
    assert games(off) == [0, 1, 1, 2, 2]
    assert windows(off) == [None, 10.0, 10.0, 50.0, 50.0]  # 従来は枠を50.0へ更新


def test_forced_only_with_no_previous_boundary_restores_none() -> None:
    on = run([(1.0, MID_SCORES, True), (2.0, MID_SCORES, False)], True)
    assert games(on) == [1, 1]
    assert windows(on) == [None, None]


# --- (b) 強制境界の3.8秒後の本物の得点リセット境界 (本件の再現) ------------------

def incident_events() -> list[tuple[float, tuple[int, int], bool]]:
    base = 96.0
    return [(base, MID_SCORES, True), (base + 1.0, MID_SCORES, False), (base + 2.0, MID_SCORES, False),
            (base + INCIDENT_GAP_SEC, ZERO, False), (base + INCIDENT_GAP_SEC + 1.0, ZERO, False)]


def test_real_score_reset_after_forced_boundary_is_accepted() -> None:
    on = run(incident_events(), True)
    assert games(on) == [1, 1, 1, 2, 2]  # 強制境界 +1、本物の得点リセット境界 +1
    assert windows(on)[3] == pytest.approx(99.81)


def test_incident_is_reproduced_when_flag_off() -> None:
    off = run(incident_events(), False)
    assert games(off) == [1, 1, 1, 1, 1]  # 本物の境界が拒否される (修正前の欠陥の再現)
    assert windows(off)[3] == 96.0


# --- (c) 得点リセット単独は従来どおり ------------------------------------------

@pytest.mark.parametrize('isolate', [False, True])
def test_score_reset_alone_and_quick_second_reset_unchanged(isolate: bool) -> None:
    events = [(9.0, (500, 300), False), (10.0, ZERO, False),         # 受理
              (11.0, (120, 60), False), (13.0, ZERO, False),         # 立ち上がりだが3秒後 → 拒否
              (14.0, (130, 60), False), (30.0, ZERO, False)]         # 20秒後 → 受理
    trace = run(events, isolate)
    assert games(trace) == [0, 1, 1, 1, 1, 2]
    assert windows(trace) == [None, 10.0, 10.0, 10.0, 10.0, 30.0]


# --- (d) 同一frameで得点リセット境界と強制境界が重なる --------------------------

def test_same_frame_score_reset_and_forced_counts_once_and_updates_window() -> None:
    events = [(9.0, (500, 300), False), (10.0, ZERO, True), (11.0, (80, 40), False),
              (13.0, ZERO, False)]
    on = run(events, True)
    assert games(on) == [0, 1, 1, 1]            # 境界は1回 (+1)
    assert windows(on) == [None, 10.0, 10.0, 10.0]  # 得点リセット境界なので枠は更新される
    # 枠が更新されているため、3秒後の再立ち上がり (13.0) は従来どおり拒否される
    assert games(run(events, False)) == games(on)
    assert windows(run(events, False)) == windows(on)


# --- 強制境界は枠の内側でも拒否されない・連続しても毎回進む --------------------------

def test_forced_boundary_is_never_debounced() -> None:
    events = [(9.0, (500, 300), False), (10.0, ZERO, False), (11.0, (80, 40), True),
              (12.0, (90, 40), True), (13.0, (100, 40), False)]
    for isolate in (False, True):
        assert games(run(events, isolate)) == [0, 1, 2, 3, 3]
    assert windows(run(events, True)) == [None, 10.0, 10.0, 10.0, 10.0]


# --- (e) フラグ OFF は従来と一致 -----------------------------------------------

def real_generate_loop(isolate: bool) -> ast.For:
    import scripts.visualize_advantage_overlay as overlay
    tree = ast.parse(inspect.getsource(overlay.generate))
    loop = next(node for node in tree.body[0].body
                if isinstance(node, ast.For) and ast.unparse(node.target) == 'fi')
    adapt_loop(loop, isolate)
    return loop


def starts(node: ast.stmt, *prefixes: str) -> bool:
    return ast.unparse(node).startswith(prefixes)


def test_off_matches_legacy_injection_exactly() -> None:
    """ONの注入文 (前後) を除き、旧1文へ戻すと、OFFのループとASTが一致する。"""
    off, on = real_generate_loop(False), real_generate_loop(True)
    legacy = ast.parse(LEGACY_FORCED_BOUNDARY_BEFORE).body
    assert ast.unparse(legacy[0]) == ('_formal_boundary = _live_bridge.consume_input_boundary() '
                                      'or _formal_boundary')
    body, rebuilt, injected = list(on.body), [], 0
    for node in body:
        if starts(node, '_scored_boundary =', '_forced_boundary =', '_debounce_before_forced =',
                  'if _forced_boundary and'):
            injected += 1
        elif starts(node, '_formal_boundary = _forced_boundary or _formal_boundary'):
            injected += 1
            rebuilt.extend(legacy)
        else:
            rebuilt.append(node)
    assert injected == 5  # 前の4文 + 後の1文 (前の4文のうち最後の1文を旧1文へ置換)
    assert [ast.unparse(node) for node in rebuilt] == [ast.unparse(node) for node in off.body]
    assert sum(starts(node, 'if _formal_boundary:') for node in off.body) == 1


def test_injection_surrounds_the_real_boundary_block_only() -> None:
    """本物の境界ブロックが想定形 (枠更新と加算が各1回) で、注入が前後に隣接している。"""
    loop = real_generate_loop(True)
    body = loop.body
    index = next(i for i, node in enumerate(body) if starts(node, 'if _formal_boundary:'))
    block = ast.unparse(body[index])
    assert block.count('_last_formal_boundary_t = t') == 1 and block.count('game_idx += 1') == 1
    before = [ast.unparse(node) for node in body[index - 4:index]]
    assert before == [ast.unparse(node) for node in ast.parse(ISOLATED_FORCED_BOUNDARY_BEFORE).body]
    assert ast.unparse(body[index + 1]) == ast.unparse(ast.parse(ISOLATED_FORCED_BOUNDARY_AFTER).body[0])
    assert sum('consume_input_boundary' in ast.unparse(node) for node in body) == 1


def test_forced_boundary_statements_selects_by_flag() -> None:
    before, after = forced_boundary_statements(False)
    assert len(before) == 1 and after == []
    before, after = forced_boundary_statements(True)
    assert len(before) == 4 and len(after) == 1


# --- 配線: bridge 属性・CLI・単一情報源 ---------------------------------------------

@pytest.mark.parametrize(('attribute', 'expected'), [(None, True), (True, True), (False, False)])
def test_build_live_generate_passes_flag_to_adapt_loop(monkeypatch: pytest.MonkeyPatch,
                                                       attribute: bool | None, expected: bool) -> None:
    import scripts.visualize_advantage_overlay as overlay
    seen: list[bool] = []
    original = live_bridge.adapt_loop
    monkeypatch.setattr(live_bridge, 'adapt_loop',
                        lambda loop, isolate=True: (seen.append(isolate), original(loop, isolate))[1])
    bridge = RecognitionBridge(False, lambda *args: None)
    if attribute is not None:
        bridge.forced_boundary_debounce_isolation = attribute
    build_live_generate(overlay, bridge)
    assert seen == [expected]


def test_cli_default_on_and_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.run_live_pipeline_20260928 import build_command, parse_args
    monkeypatch.setattr('sys.argv', ['live'])
    options = parse_args()
    assert options.forced_boundary_debounce_isolation is True
    options.no_render = True  # run_live が build_command の前に設定する値
    assert not any('forced-boundary' in token for token in build_command(options))  # 描画CLIへは渡さない
    monkeypatch.setattr('sys.argv', ['live', '--no-forced-boundary-debounce-isolation'])
    assert parse_args().forced_boundary_debounce_isolation is False


def test_registered_in_production_config_as_single_source() -> None:
    from src import production_config as config
    assert config.FORCED_BOUNDARY_DEBOUNCE_ISOLATION_DEFAULT is True
    (entry,) = config.LIVE_BRIDGE_ADOPTED
    assert entry.flag == '--forced-boundary-debounce-isolation' and entry.adopted == '2026-10-11'
    for evidence in ('user承認 2026-10-11', 'soak game11', '699.91'):
        assert evidence in entry.reason
    assert entry.flag in config.describe()
    # 描画CLI・収集CLIが受け付けないフラグなので、それらの集約文字列へは混ぜない。
    for joined in (config.advantage_overlay_flags(), config.exchange_event_flags(),
                   config.recognition_flags(), config.collect_flags(), config.visualization_flags(),
                   config.placement_reconcile_flags()):
        assert 'forced-boundary' not in joined
    assert live_bridge.FORCED_BOUNDARY_DEBOUNCE_ISOLATION_DEFAULT is config.FORCED_BOUNDARY_DEBOUNCE_ISOLATION_DEFAULT

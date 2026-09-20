"""原76本体の再実行前に時計・件数と派生関数のアンカーを確かめる。"""
from __future__ import annotations
from pathlib import Path
import sys
import json

RESET_ROOT = Path(__file__).resolve().parents[1]/'g2_empty_tail_reset_integration_2026-09-11_v1'
sys.path.insert(0, str(RESET_ROOT))
import extension as X


def test_original_extension_anchors() -> None:
    extend = X.continuation()
    assert extend.__globals__['I'].RESET == 34932
    assert extend.__globals__['I'].POST[0] == 34932
    assert extend.__globals__['I'].POST[-1] == 34980
    assert 80 in extend.__code__.co_consts
    assert 80 in extend.__globals__['verify'].__code__.co_consts


def test_shifted_actual_input_clock() -> None:
    original = X.Q.module('_desync_test_original_reset_inputs', X.Q.RECOVERY/'reset_inputs.py')
    result = X.inputs().shifted(original)
    assert result.RESET == 34932
    assert result.DEADLINE == 34946
    assert result.FIRST == 34950
    assert result.LANDED == 34952
    assert result.SECOND == 34960
    assert result.FRAMES == X.POST


def test_entry_does_not_shadow_original_continuation() -> None:
    entry = X.Q.module('_desync_test_entry', RESET_ROOT/'run_desync.py')
    original = X.Q.module('_desync_test_hidden_bundle',
        RESET_ROOT.parent/'g2_hidden_repeat_integration_2026-09-10_v1/hidden_bundle.py')
    assert Path(original.HR.B.__file__).parent.name == 'g2_hidden_continuation_candidate_2026-09-10_v1'
    assert original.HR.B is not entry.R


def test_desync_keeps_original_palette() -> None:
    source = RESET_ROOT/'prefix_cpu_v10/directional_history.jsonl'
    with source.open() as stream:
        row = next(r for r in map(json.loads, stream) if r['scope']['frame_idx'] == 34922)
    palette = {c for line in row['raw_capture']['raw']['grid'] for c in line if 1 <= c <= 5}
    assert palette == {2, 3, 4, 5}
    assert not set((1, 1)+X.YY) <= palette  # 旧RR入力の反例を保持。
    assert set(X.BB+X.YY) <= palette
    assert X.BB != (3, 3)  # 後継GGとの不一致条件は残す。


def test_entry_continuation_original_anchors() -> None:
    import entry_continuation as entry
    result = entry.derived()
    assert result.__globals__['I'].RESET == 34932
    assert result.__globals__['ENTRY'].install is entry.E.install
    assert 80 in result.__code__.co_consts

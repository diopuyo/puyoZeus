"""video39候補の収集範囲/設定票だけを接続する。後段時計の完了とは別。"""
from __future__ import annotations

from contextlib import ExitStack
import importlib.util
import inspect
from pathlib import Path
import sys
from typing import Any

from scripts import g3_video39_baseline as B
from scripts import g3_observer_scope as O

G = B.G
ENTRY_SHA = '4af78d137ea43c03bffc4415d1814cc31b30317df09eee0bb379cc044caccf66'
OLD_FPS, OLD_STRIDE, OLD_FIRST = 60, 2, 29052
OLD_ENDS = (36300, 36902)
OLD_SOURCE_SHA = 'b3728078cc2e8282065e5dd78ca1c13e6a443ffdf1dc4333e3da06f0852757d3'
OLD_WINDOW_STRIDE, OLD_WINDOW_FPS = 2, 60
# common.py の SOURCE は既存 bind_bounds が所有するため、ここでは扱わない。
IDENTITY_TARGETS = {
    'g2_next_motion_observables_2026-09-09_v1/observables.py': (('VIDEO_SHA', False),),
    'g2_normal_completion_transaction_2026-09-09_v1/live_provider.py': (('SOURCE', True),),
    'g2_palette_finish_proof_2026-09-09_v1/adapter.py': (('SOURCE_ID', True),),
}


def bind_bounds(stack: ExitStack, main: Any, history: Any) -> list[dict]:
    """旧sourceの固定値を検査してから、収集commonの時計を同時に置換する。"""
    G.require(B.M.digest(Path(G.__file__)) == ENTRY_SHA, 'source39_original_entry')
    ns, seen, receipts = main.__globals__, set(), []
    modules = [ns['K'], ns['S'].K, ns['Q'].K, ns['R'].K]
    for module in modules:
        G.require(Path(module.__file__).resolve() == G.COMMON_FILE, 'source39_common_owner')
        spaces = [vars(module)] + [fn.__globals__ for fn in vars(module).values()
            if inspect.isfunction(fn) and Path(fn.__code__.co_filename).resolve() == G.COMMON_FILE]
        for space in spaces:
            if id(space) in seen:
                continue
            seen.add(id(space))
            G.require(space['FIRST'] == OLD_FIRST and space['END'] in OLD_ENDS
                and (space['FPS'], space['STRIDE']) == (OLD_FPS, OLD_STRIDE), 'source39_old_collection_scope')
            G.require(space['SOURCE'] == OLD_SOURCE_SHA, 'source39_old_identity')
            receipts.append(dict(kind='common', old_fps=space['FPS'], old_stride=space['STRIDE'],
                                 fps=B.FPS, stride=B.STRIDE, first=0, end_exclusive=B.END))
            G.patch_dict(stack, space, dict(FIRST=0, END=B.END, FPS=B.FPS, STRIDE=B.STRIDE, SOURCE=G.SOURCE_SHA,
                                           FRAMES=tuple(range(0, B.END, B.STRIDE))))
    space = history.HistoryRecorder.begin_frame.__globals__
    G.require(space['FIRST_FRAME'] == OLD_FIRST and space['END_FRAME'] == OLD_ENDS[-1], 'source39_history_owner')
    G.patch_dict(stack, space, dict(FIRST_FRAME=0, END_FRAME=B.END))
    receipts.append(dict(kind='HistoryRecorder.begin_frame', first=0, end=B.END,
                         inner_clock_not_yet_adapted=True))
    G.require(ns['K'].bounds()['first_frame'] == 0 and ns['S'].K.bounds()['end_exclusive'] == B.END,
              'source39_actual_bounds')
    return receipts


def receipt_for(plan: dict, output: Path, base: Any) -> tuple[dict, dict]:
    """旧票は版参照に限定し、実source/較正/連続frame数を新runへ固定する。"""
    prior = B.M.document(B.REFERENCE_PLAN, B.REFERENCE_SHA)
    G.require((plan['reference_prepare'], plan['reference_sha']) ==
              (prior['prepare_receipt'], prior['prepare_sha256']), 'source39_reference')
    receipt = B.M.document(Path(plan['reference_prepare']), plan['reference_sha'])
    G.require(receipt['video_path'] == prior['video_path'], 'source39_original_prepare')
    G.require(plan['source_id'] == 'sha256:' + G.SOURCE_SHA, 'source39_identity')
    G.require(int(B.END / B.FPS * B.FPS) == B.END, 'source39_end_rounding')
    receipt['input_and_code_sha256'].update(plan['runtime_pins'])
    G.require(receipt['input_and_code_sha256'][str(G.SOURCE)] == G.SOURCE_SHA, 'source39_source_pin')
    base.assert_unchanged(receipt['input_and_code_sha256'])
    tokens = list(plan['tokens'])
    G.require(tokens.count('--score-region-calibration') == 1 and
              tokens[tokens.index('--score-region-calibration')+1] == str(B.CALIBRATION), 'source39_calibration')
    original = B.source_contract()
    G.require(original['source_video_sha256'] == G.SOURCE_SHA
              and (original['time_base_numerator'], original['time_base_denominator']) == (1, B.FPS)
              and original['processing_start_frame'] == 0
              and original['processing_end_frame_exclusive'] >= B.END,
              'source39_original_source_contract')
    receipt.update(original_source=original, video_path=str(G.SOURCE), source_id=G.SOURCE_SHA, target_board={'sha256': None}, run_id=str(output),
        target_reference_disabled=True, requested_interval_sec=[0, B.END / B.FPS],
        initialization=f'cold_at_frame_0_same_pipeline_through_frame_{B.END-1}',
        output_boundary=dict(path=str(output), exclusive_entry_directory=True,
            frozen_prepare_reference=plan['reference_prepare'], shared_live_preflight=False,
            original_prepare_returned_before_mkdir=False),
        expected_frame_count=len(range(0, B.END, B.STRIDE)), quality_gate_clear=False,
        original_collection_tokens=tokens, gt_scoring=plan['gt_scoring'], g3_plan_sha256=plan['_sha256'],
        source39_entry_scope_only=True, downstream_source_clock_not_verified=True)
    return receipt, dict(collection_tokens=tokens)


def arguments(collector: Any, config: dict) -> dict:
    """検収済みsource39 CLI捕捉で実videoとstride1を固定する。"""
    kwargs, receipt = B.arguments(collector, config['collection_tokens'], G.SOURCE)
    G.require(receipt['video'] == str(G.SOURCE) and kwargs['sample_interval_frames'] == B.STRIDE,
              'source39_actual_sampling')
    return kwargs


def observer_rebind(stack: Any, window: Any, values: dict, replace: Any) -> dict:
    """元video38窓の旧時計を明示pinで検査し、source39の全区間/stride1へ置換する。"""
    scope, context, metadata = (values[key] for key in ('scope', 'context', 'metadata'))
    expected = (((O.OLD_FIRST, O.OLD_LAST),), O.OLD_FIRST, O.OLD_LAST, O.OLD_FIRST, O.OLD_LAST)
    actual = (scope.WINDOWS, context.OUTER_FIRST, context.OUTER_LAST, metadata.FIRST, metadata.LAST)
    G.require(actual == expected, 'source39_observer_owner_bounds')
    G.require((window.FIRST, window.LAST, window.STRIDE) == (O.OLD_FIRST, O.OLD_LAST, OLD_WINDOW_STRIDE)
              and tuple(window.FRAMES) == tuple(range(O.OLD_FIRST, O.OLD_LAST + OLD_WINDOW_STRIDE,
                                                      OLD_WINDOW_STRIDE)),
              'source39_observer_verifier_bounds')
    G.require(context.STRIDE == metadata.STRIDE == OLD_WINDOW_STRIDE and metadata.FPS == OLD_WINDOW_FPS,
              'source39_observer_old_clock')
    last, frames = B.END - B.STRIDE, tuple(range(0, B.END, B.STRIDE))
    changes = [(scope, 'WINDOWS', ((0, last),)), (context, 'OUTER_FIRST', 0), (context, 'OUTER_LAST', last),
               (context, 'STRIDE', B.STRIDE), (metadata, 'FIRST', 0), (metadata, 'LAST', last),
               (metadata, 'STRIDE', B.STRIDE), (metadata, 'FPS', B.FPS), (window, 'FIRST', 0),
               (window, 'LAST', last), (window, 'STRIDE', B.STRIDE), (window, 'FRAMES', frames)]
    receipt = dict(first=0, last=last, stride=B.STRIDE, fps=B.FPS, updates=len(frames),
                   source39_observer_scope_only=True, quality_gate_clear=False)
    previous = [(owner, name, getattr(owner, name)) for owner, name, _ in changes]
    def restored() -> None:
        receipt['inner_binding_restored'] = all(getattr(owner, name) is value for owner, name, value in previous)
    stack.callback(restored)
    for owner, name, value in changes:
        replace(stack, owner, name, value)
    return receipt


def retarget(stack: Any, module: Any, replace: Any, done: set) -> list[dict]:
    """元video38識別子を持つ実loadedmoduleだけを、旧値検査つきでsource39へ置換する。"""
    file = getattr(module, '__file__', None)
    if not file:
        return []
    posix, receipts = Path(file).resolve().as_posix(), []
    for suffix, entries in IDENTITY_TARGETS.items():
        if not posix.endswith(suffix):
            continue
        for attribute, prefixed in entries:
            key = (id(module), attribute)
            if key in done:
                continue
            old = ('sha256:' if prefixed else '') + OLD_SOURCE_SHA
            new = ('sha256:' if prefixed else '') + G.SOURCE_SHA
            G.require(getattr(module, attribute) == old, 'source39_identity_old_value')
            replace(stack, module, attribute, new)
            done.add(key)
            receipts.append(dict(file=suffix, attribute=attribute, prefixed=prefixed,
                                 module=module.__name__, actual_loaded=True))
    return receipts


def sweep(stack: Any, replace: Any, done: set) -> list[dict]:
    """私有loadを経ない既loadedの同一定数も同じ契約で拾う。件数0は成功にしない。"""
    receipts = []
    for module in list(sys.modules.values()):
        receipts.extend(retarget(stack, module, replace, done))
    return receipts


def targeted(location: Any) -> bool:
    """4つの固定targetだけを対象にし、他のmodule loadには触れない。"""
    posix = Path(location).resolve().as_posix()
    return any(posix.endswith(suffix) for suffix in IDENTITY_TARGETS)


def install_source_identity(stack: Any, adapter: Any) -> dict:
    """私有loaderが各自実装でも通る共通spec生成点で、target4件だけ実load直後に揃える。"""
    replace = adapter.A.A.A.V4.replace_owned
    done: set = set()
    receipts = sweep(stack, replace, done)
    original = importlib.util.spec_from_file_location
    def spec_from_file_location(name: Any, location: Any = None, *args: Any, **kwargs: Any) -> Any:
        spec = original(name, location, *args, **kwargs)
        if spec is None or getattr(spec, 'loader', None) is None or location is None or not targeted(location):
            return spec
        loader, previous = spec.loader, spec.loader.exec_module
        def exec_module(module: Any) -> None:
            previous(module)
            receipts.extend(retarget(stack, module, replace, done))
        loader.exec_module = exec_module
        return spec
    replace(stack, importlib.util, 'spec_from_file_location', spec_from_file_location)
    def ended() -> None:
        receipt['loaded_targets'] = sorted({item['file'] for item in receipts})
        receipt['replacements'] = len(receipts)
    receipt = dict(receipts=receipts, targets=len(IDENTITY_TARGETS), source39_identity_only=True,
                   quality_gate_clear=False)
    stack.callback(ended)
    return receipt

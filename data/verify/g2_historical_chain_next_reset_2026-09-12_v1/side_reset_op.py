"""過去連鎖NEXTの寿命を区別する片側再同期候補 (未接続・未認証)。

2026-09-12差分: 過去NEXTのみの非None拒否を修復。真のactive/settle/窓は維持し、
原resetと同じNone化を片側認識cacheに加える。旧helper/sourceは変更しない。

位置づけ (実v12で採用しない):
  - 所有は本rootのみ。src/ scripts/ 既存root へは配線しない。GOは出さない。
  - **上流資格の前提**: 呼出元は原 lease を保持し、対象sideの旧NEXT bindingが
    原 archive 資格 (空FIFO/旧owner保持) を満たしていることを既に確認済みで
    あること。本helperはその資格を**検証も付与もしない** (`qualification`
    文字列を報告へ写すだけ)。実lease配線・NEXT epoch前進・厳格joinは親担当。
  - 本helperは物理配置/current/FIFO の資格を一切主張しない
    (`physical_placement_verified` / `next_consumption_authority` /
     `fifo_qualification_checked` はすべて False)。
  - 原 `RecognitionPipeline.reset` / 原 `NextEnqueueController.reset` は呼ばない。
    `record_reset` も呼ばない (原 SM.reset hook が一回記録する)。
  - 汎用callbackを受け取らない。操作は下の固定表だけ (all-or-nothing ではない:
    途中例外は fail-stop として呼出元へ伝える。§SideRecognitionResetFailStop)。

退役の意味づけ (重要):
  - OJAMA_FALL の**認識state**(raw9 streak origin / 色swap streak origin /
    warmup / override 窓) は reset 対象として扱う。これは「おじゃまを消費した」
    「お邪魔処理が完了した」という主張では**ない**。共有おじゃま会計
    (`_ojama_fall_accounting_tracker`) には一切書き戻さない。
  - 未処理の旧認識 origin は `retired_unprocessed` として before/after に
    保存して返すだけで、原 physics の消費済みには**しない**。
"""
from __future__ import annotations

import math
from typing import Any

SIDES: tuple[str, str] = ("1P", "2P")

# 原メソッドをそのまま呼ぶ部品 (引数固定)。
NATIVE_RESETS: tuple[tuple[str, str], ...] = (("_gen_{s}", "reset"), ("_drift_{s}", "reset"))
# 認識cache (原 RP.reset と同じ .clear() / None)。
CACHE_CLEARS: tuple[str, ...] = ("_cnn_history_{s}", "_stable_cnn_history_{s}")
CACHE_NONE: tuple[str, ...] = ("_prev_confirmed_{s}", "_prev_stable_confirmed_{s}",
                               "_chain_start_next_{s}")
# 旧認識 origin (セル単位の起点時刻)。OJAMA_FALL 認識state を含む。
ORIGIN_CLEARS: tuple[str, ...] = ("_ojama_raw9_streak_start_{s}",
                                  "_ojama_color_swap_streak_{s}")
# 推定窓 / warmup / override 窓。値は原 RP.reset と同一。
FIXED_VALUES: tuple[tuple[str, Any], ...] = (
    ("_chain_estimate_result_{s}", None),
    ("_chain_estimate_trigger_{s}", 0.0),
    ("_chain_estimate_end_{s}", 0.0),
    ("_chain_estimate_low_confidence_{s}", False),
    ("_chain_estimate_last_board_{s}", None),
    ("_chain_estimate_stale_since_{s}", None),
    ("_chain_verify_pending_{s}", None),
    ("_tier1_warmup_remaining_{s}", 0),
    ("_ojama_tier1_warmup_remaining_{s}", 0),
    ("_ojama_override_exit_until_{s}", 0.0),
)
# 許可外なので触らない (対照テストで不変を確認する)。会計/ゲーム権利側。
NEVER_TOUCHED: tuple[str, ...] = (
    "_pending_tsumo_{s}", "_last_consumed_color_{s}", "_landing_pending_{s}",
    "_first_move_sec_{s}", "_tsumo_count_{s}", "_all_clear_pending_{s}",
    "_chain_tracker_{s}", "_score_tracker_{s}", "_formula_accum_{s}",
)
# 連鎖窓が開いている間は許可表に手当てが無い (触ると許可外)。拒否する。
ACTIVE_CHAIN_OBJECTS: tuple[str, ...] = (
    "_active_chain_{s}", "_last_chain_event_for_settle_{s}")
ACTIVE_CHAIN_UNTIL: tuple[str, ...] = (
    "_chain_until_{s}", "_chain_event_max_until_{s}", "_chain_exit_until_{s}")


class SideRecognitionResetRejected(RuntimeError):
    """前提未達。このとき本helperは一切変更していない。"""

    def __init__(self, reason: str, report: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason, self.report = reason, report


class SideRecognitionResetFailStop(RuntimeError):
    """適用途中の例外。rollback済みとも無変更とも主張しない (fail-stop)。"""

    def __init__(self, report: dict[str, Any], cause: BaseException) -> None:
        super().__init__("partial_side_reset:" + str(report.get("failed_step")))
        self.report, self.cause = report, cause


def _sfx(side: str) -> str:
    """side を厳格に検査して属性 suffix を返す。"""
    if type(side) is not str or side not in SIDES:
        raise SideRecognitionResetRejected("invalid_side", {"side": repr(side)})
    return side.lower()


def _get(pipe: Any, name: str) -> Any:
    """原 pipeline の属性。欠落を黙って None にしない (未知fieldも触らない)。"""
    if not hasattr(pipe, name):
        raise AttributeError(f"pipeline has no attribute {name!r}")
    return getattr(pipe, name)


def _render(value: Any) -> Any:
    """before/after を保存可能 (JSON化可能) な形にする。値は捨てない。"""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {"type": type(value).__name__, "len": len(value),
                "items": sorted((repr(k), repr(v)) for k, v in value.items())}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {"type": type(value).__name__, "len": len(value),
                "items": [repr(item) for item in value]}
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def _names(side_suffix: str) -> list[tuple[str, str]]:
    """(field名, 分類) の固定一覧。許可表の外は一つも含まない。"""
    out: list[tuple[str, str]] = []
    for template, _method in NATIVE_RESETS:
        out.append((template.format(s=side_suffix), "native_component"))
    for template in CACHE_CLEARS + CACHE_NONE:
        out.append((template.format(s=side_suffix), "recognition_cache"))
    for template in ORIGIN_CLEARS:
        out.append((template.format(s=side_suffix), "retired_unprocessed"))
    for template, _value in FIXED_VALUES:
        out.append((template.format(s=side_suffix), "estimate_window"))
    return out


def snapshot_recognition(pipe: Any, side: str) -> dict[str, Any]:
    """旧認識の保存可能な snapshot (読むだけ、副作用なし)。"""
    suffix = _sfx(side)
    machine = _get(pipe, f"_sm_{suffix}")
    context = machine.context
    fields = {name: _render(_get(pipe, name)) for name, _kind in _names(suffix)}
    return {"side": side, "fields": fields,
            "kinds": {name: kind for name, kind in _names(suffix)},
            "sm_state": context.state.name,
            "sm_context_id": id(context),
            "sm_confirmed": _render(context.confirmed_board),
            "sm_pending_count": context.pending_count}


def check_preconditions(pipe: Any, side: str, *, now_sec: float) -> list[dict[str, Any]]:
    """許可表で手当てできない残留 (連鎖窓) を列挙する。空なら適用可。"""
    suffix = _sfx(side)
    if type(now_sec) not in (int, float) or not math.isfinite(now_sec) or now_sec < 0:
        raise SideRecognitionResetRejected("nonfinite_or_negative_clock",
                                           {"now_sec": repr(now_sec)})
    blockers: list[dict[str, Any]] = []
    for template in ACTIVE_CHAIN_OBJECTS:
        name = template.format(s=suffix)
        if _get(pipe, name) is not None:
            blockers.append({"field": name, "kind": "active_chain",
                             "value": _render(_get(pipe, name))})
    for template in ACTIVE_CHAIN_UNTIL:
        name = template.format(s=suffix)
        value = _get(pipe, name)
        if type(value) not in (int, float) or not math.isfinite(value):
            blockers.append({"field": name, "kind": "invalid_window",
                             "value": _render(value)})
        elif float(value) > now_sec:
            blockers.append({"field": name, "kind": "active_chain",
                             "value": _render(value)})
    # 過去NEXTは単独ではactiveの証拠ではない。真の残留があれば退役しない。
    latch = f"_chain_start_next_{suffix}"
    if blockers and _get(pipe, latch) is not None:
        blockers.append({"field": latch, "kind": "chain_start_latch_not_retirable",
                         "value": _render(_get(pipe, latch))})
    return blockers


def _apply_fixed_operations(pipe: Any, suffix: str) -> list[str]:
    """固定表の操作を順に実行し、実行済み step を返す。例外は呼出元で保存する。"""
    done: list[str] = []
    machine = _get(pipe, f"_sm_{suffix}")
    # 原 BoardStateMachine.reset。既存hookがここで record_reset を一回記録する
    # (本helperからは呼ばない)。keep_match_state=True で試合state は保つ。
    machine.reset(keep_match_state=True)
    done.append(f"_sm_{suffix}.reset(keep_match_state=True)")
    for template, method in NATIVE_RESETS:
        name = template.format(s=suffix)
        getattr(_get(pipe, name), method)()
        done.append(f"{name}.{method}()")
    for template in CACHE_CLEARS + ORIGIN_CLEARS:
        name = template.format(s=suffix)
        _get(pipe, name).clear()   # object identity は保つ (差替えない)
        done.append(f"{name}.clear()")
    for template in CACHE_NONE:
        name = template.format(s=suffix)
        _get(pipe, name)           # 欠落なら AttributeError で止める
        setattr(pipe, name, None)
        done.append(f"{name}=None")
    for template, value in FIXED_VALUES:
        name = template.format(s=suffix)
        _get(pipe, name)
        setattr(pipe, name, value)
        done.append(f"{name}={value!r}")
    return done


def reset_recognition_side(pipe: Any, side: str, *, now_sec: float,
                           qualification: str, dry_run: bool = False) -> dict[str, Any]:
    """対象sideの認識stateだけを固定操作で再同期する (私有・未接続・未認証)。

    Args:
        pipe: 原 RecognitionPipeline。
        side: "1P" / "2P" (厳格)。
        now_sec: 連鎖窓判定に使う現在時刻。真の試合時計は読むだけ。
        qualification: 上流(親)が保持する lease/archive 資格の識別子。
            本helperは**検証しない**。空文字や非strは受け取らない。
        dry_run: True なら適用せず判定と before だけ返す。

    Raises:
        SideRecognitionResetRejected: 前提未達 (このとき一切変更しない)。
        SideRecognitionResetFailStop: 適用途中の例外 (部分変更、元例外を保存)。
    """
    suffix = _sfx(side)
    if type(qualification) is not str or not qualification.strip():
        raise SideRecognitionResetRejected("missing_upstream_qualification",
                                           {"side": side})
    if type(dry_run) is not bool:
        raise SideRecognitionResetRejected("dry_run_type", {"side": side})
    blockers = check_preconditions(pipe, side, now_sec=now_sec)
    before = snapshot_recognition(pipe, side)
    if blockers:
        raise SideRecognitionResetRejected("unhandled_residue",
                                           {"side": side, "blockers": blockers,
                                            "before": before})
    report = {"side": side, "now_sec": now_sec, "before": before,
              "upstream_qualification": qualification,
              "qualification_verified_by_helper": False,
              "record_reset_called_by_helper": False,
              "next_epoch_advanced_by_helper": False,
              "fifo_qualification_checked": False,
              "physical_placement_verified": False,
              "next_consumption_authority": False,
              "ojama_processing_completed": False,
              "origin_marked_physics_consumed": False,
              "production_permission": False}
    if dry_run:
        return dict(report, applied=False, steps=[], after=before)
    try:
        steps = _apply_fixed_operations(pipe, suffix)
    except BaseException as error:  # noqa: BLE001 - 種別に依らず fail-stop
        raise SideRecognitionResetFailStop(
            dict(report, applied=False, partial_change_possible=True,
                 rolled_back=False, failed_step=repr(error),
                 after=snapshot_recognition(pipe, side)), error) from error
    return dict(report, applied=True, steps=steps,
                after=snapshot_recognition(pipe, side))

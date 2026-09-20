"""学習用状態表から、比較条件A/B/Cの数値入力を決定論的に作る。"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.board import Board


FEATURE_TABLE_VERSION = "event-indicator-features/v1"
SIDES = ("p1", "p2")
BOARD_CELL_COUNT = 78
LATE_EXCHANGE_MIN_OCCUPIED = 36
COLOR_VALUES = frozenset({1, 2, 3, 4, 5})
CHAIN_MECHANISMS = ("baseline", "landing")
EVENT_HISTORY_TYPES = (
    "chain_started", "attack_finalized", "garbage_sent", "garbage_cancelled",
    "garbage_fall_completed", "all_clear_gained", "all_clear_consumed",
)
A_CONTEXT_BASES = (
    "score_log", "tsumo_log", "all_clear_pending", "stable_confidence",
    "next_pair_same", "next_first_match_ratio", "next_second_match_ratio",
    "double_next_pair_same", "double_next_first_match_ratio",
    "double_next_second_match_ratio", "mechanism_baseline", "mechanism_landing",
    "mechanism_other",
)
B_SIDE_BASES = (
    "pending_garbage_log", "chain_active", "all_clear_event_state",
    "provisional_generated_log", "provisional_score_log",
    "provisional_chain_count_log",
)
META_KEYS = (
    "schema_version", "state_id", "source_video_id", "source_group_id",
    "partition_fold", "online_segment_index", "available_frame", "available_ms",
    "a_input_usable", "b_input_usable", "c_input_usable",
    "a_reason_codes", "b_reason_codes", "c_reason_codes",
    "game_key", "p1_won", "sample_weight", "evaluation_scope_usable",
    "training_usable",
)


class EventIndicatorFeatureError(ValueError):
    """入力表の対応や値が安全な特徴量生成条件を満たさない。"""


@dataclass(frozen=True, slots=True)
class IndicatorFeatureBundle:
    """全条件共通の行と、条件ごとの入力列一覧。"""

    rows: tuple[dict[str, Any], ...]
    feature_names: Mapping[str, tuple[str, ...]]


def build_indicator_feature_bundle(
    states: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    event_links: Sequence[Mapping[str, Any]],
    indicator_registry: Mapping[str, Callable[[Board], Any]],
) -> IndicatorFeatureBundle:
    """同じstate・正解・重みを保ったA/B/Cの入力列を作る。"""

    label_map = _unique_by_state(labels, "正解")
    link_map = _links_by_state(event_links)
    cache: dict[tuple[int, ...], dict[str, float | None]] = {}
    rows: list[dict[str, Any]] = []
    history: dict[tuple[str, int], defaultdict[tuple[str, str], int]] = {}
    ordered = sorted(states, key=_state_sort_key)
    for state in ordered:
        state_id = str(state["state_id"])
        if state_id not in label_map:
            raise EventIndicatorFeatureError(f"正解行がありません: {state_id}")
        row = _metadata_row(state, label_map[state_id])
        a_features = _a_features(state, indicator_registry, cache)
        b_features = _b_features(state)
        c_features = _c_features(state, link_map.get(state_id, ()), history)
        row.update(a_features)
        row.update(b_features)
        row.update(c_features)
        rows.append(row)
    names = _feature_name_groups(rows)
    return IndicatorFeatureBundle(tuple(rows), names)


def _unique_by_state(
    rows: Sequence[Mapping[str, Any]], label: str,
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        state_id = str(row.get("state_id", ""))
        if not state_id or state_id in output:
            raise EventIndicatorFeatureError(f"{label}state_idが空または重複です")
        output[state_id] = row
    return output


def _links_by_state(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("model_input_allowed") is False:
            continue
        grouped[str(row.get("state_id", ""))].append(row)
    return {
        key: tuple(sorted(value, key=lambda item: int(item["event_sequence"])))
        for key, value in grouped.items()
    }


def _state_sort_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("source_video_id", "")),
        int(row.get("online_segment_index", 0)),
        int(row.get("through_sequence", -1)),
    )


def _metadata_row(
    state: Mapping[str, Any], label: Mapping[str, Any],
) -> dict[str, Any]:
    if any(state.get(key) != label.get(key) for key in (
        "state_id", "source_video_id", "source_group_id", "partition_fold",
    )):
        raise EventIndicatorFeatureError("状態と正解の識別情報が一致しません")
    return {
        "schema_version": FEATURE_TABLE_VERSION,
        "state_id": state["state_id"], "source_video_id": state["source_video_id"],
        "source_group_id": state["source_group_id"],
        "partition_fold": state["partition_fold"],
        "online_segment_index": state["online_segment_index"],
        "available_frame": state["available_frame"], "available_ms": state["available_ms"],
        "a_input_usable": bool(label.get("a_input_usable", False)),
        "b_input_usable": bool(label.get("b_input_usable", False)),
        "c_input_usable": bool(label.get("c_input_usable", False)),
        "a_reason_codes": list(state.get("a_reason_codes", ())),
        "b_reason_codes": list(state.get("b_reason_codes", ())),
        "c_reason_codes": list(state.get("c_reason_codes", ())),
        "game_key": label.get("game_key"), "p1_won": label.get("p1_won"),
        "sample_weight": label.get("sample_weight"),
        "evaluation_scope_usable": bool(label.get("evaluation_scope_usable", False)),
        "training_usable": bool(label.get("training_usable", False)),
    }


def _a_features(
    state: Mapping[str, Any], registry: Mapping[str, Callable[[Board], Any]],
    cache: dict[tuple[int, ...], dict[str, float]],
) -> dict[str, float | None]:
    side_values: dict[str, dict[str, float | None]] = {}
    for side in SIDES:
        grid = _grid_tuple(state[f"a_{side}_grid"])
        values = dict(_cached_board_indicators(grid, registry, cache))
        values.update(_board_count_values(state, side))
        values.update(_context_values(state, side, grid))
        side_values[side] = values
    features: dict[str, float | None] = {}
    for name in sorted(side_values["p1"]):
        _add_paired(features, f"a_{name}", side_values["p1"][name], side_values["p2"][name])
    return features


def _cached_board_indicators(
    grid: tuple[int, ...], registry: Mapping[str, Callable[[Board], Any]],
    cache: dict[tuple[int, ...], dict[str, float | None]],
) -> Mapping[str, float | None]:
    if grid not in cache:
        board = Board.from_list([list(grid[index:index + 6]) for index in range(0, 78, 6)])
        cache[grid] = {
            f"indicator_{name}": _indicator_score(registry[name](board).score, name)
            for name in sorted(registry)
        }
    return cache[grid]


def _indicator_score(value: Any, name: str) -> float | None:
    """指標の未定義NaNは欠測へ変換し、無限値は入力異常として拒否する。"""

    number = float(value)
    if math.isnan(number):
        return None
    if not math.isfinite(number):
        raise EventIndicatorFeatureError(f"指標が有限値ではありません: {name}")
    return number


def _grid_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EventIndicatorFeatureError("盤面が一次元配列ではありません")
    grid = tuple(int(cell) for cell in value)
    if len(grid) != BOARD_CELL_COUNT:
        raise EventIndicatorFeatureError("盤面セル数が78ではありません")
    return grid


def _board_count_values(
    state: Mapping[str, Any], side: str,
) -> dict[str, float]:
    return {
        "color_ratio": _ratio(state[f"a_{side}_color_count"], BOARD_CELL_COUNT),
        "garbage_ratio": _ratio(state[f"a_{side}_garbage_count"], BOARD_CELL_COUNT),
        "occupied_ratio": _ratio(state[f"a_{side}_occupied_count"], BOARD_CELL_COUNT),
        "unknown_ratio": _ratio(state[f"a_{side}_unknown_count"], BOARD_CELL_COUNT),
    }


def _context_values(
    state: Mapping[str, Any], side: str, grid: tuple[int, ...],
) -> dict[str, float | None]:
    values: dict[str, float | None] = {
        "score_log": _optional_log(state.get(f"a_{side}_score")),
        "tsumo_log": _optional_log(state.get(f"a_{side}_tsumo_count")),
        "all_clear_pending": _optional_bool(state.get(f"a_{side}_all_clear_pending")),
        "stable_confidence": _optional_bool(state.get(f"a_{side}_stable_confidence")),
    }
    values.update(_next_context_values(state, side, grid))
    mechanism = state.get(f"a_{side}_chain_mechanism")
    for known in CHAIN_MECHANISMS:
        values[f"mechanism_{known}"] = None if mechanism is None else float(mechanism == known)
    values["mechanism_other"] = None if mechanism is None else float(mechanism not in CHAIN_MECHANISMS)
    return values


def _next_context_values(
    state: Mapping[str, Any], side: str, grid: tuple[int, ...],
) -> dict[str, float | None]:
    color_total = sum(cell in COLOR_VALUES for cell in grid)
    output: dict[str, float | None] = {}
    for prefix, first_key, second_key in (
        ("next", "next_first", "next_second"),
        ("double_next", "double_next_first", "double_next_second"),
    ):
        first = _optional_color(state.get(f"a_{side}_{first_key}"))
        second = _optional_color(state.get(f"a_{side}_{second_key}"))
        output[f"{prefix}_pair_same"] = None if first is None or second is None else float(first == second)
        output[f"{prefix}_first_match_ratio"] = _color_match_ratio(grid, first, color_total)
        output[f"{prefix}_second_match_ratio"] = _color_match_ratio(grid, second, color_total)
    return output


def _b_features(state: Mapping[str, Any]) -> dict[str, float | None]:
    side_values: dict[str, dict[str, float | None]] = {side: {} for side in SIDES}
    for side in SIDES:
        side_values[side] = {
            "pending_garbage_log": _optional_log(_preferred_pending(state, side)),
            "chain_active": _optional_bool(state.get(f"b_{side}_chain_active")),
            "all_clear_event_state": _optional_bool(state.get(f"b_{side}_all_clear_event_state")),
            "provisional_generated_log": _optional_log(state.get(f"b_{side}_provisional_generated")),
            "provisional_score_log": _optional_log(state.get(f"b_{side}_provisional_score")),
            "provisional_chain_count_log": _optional_log(state.get(f"b_{side}_provisional_chain_count")),
        }
    features: dict[str, float | None] = {}
    for name in B_SIDE_BASES:
        _add_paired(features, f"b_{name}", side_values["p1"][name], side_values["p2"][name])
    balance = _observed_attack_balance_log(state)
    features["b_observed_attack_balance_log_diff"] = balance
    features["b_late_observed_attack_balance_log_diff"] = (
        balance if _max_occupied_count(state) >= LATE_EXCHANGE_MIN_OCCUPIED else 0.0
    )
    return features


def _max_occupied_count(state: Mapping[str, Any]) -> int:
    counts = [int(state[f"a_{side}_occupied_count"]) for side in SIDES]
    if any(value < 0 or value > BOARD_CELL_COUNT for value in counts):
        raise EventIndicatorFeatureError("盤面個数が範囲外です")
    return max(counts)


def _observed_attack_balance_log(state: Mapping[str, Any]) -> float | None:
    """観測済みの進行中・送付済み攻撃の差を符号付きで返す。"""

    causal = state.get("b_causal_observed_attack_balance")
    if "b_causal_observed_attack_balance" in state:
        if causal is None:
            return None
        net = float(causal)
        return math.copysign(math.log1p(abs(net)), net) if net else 0.0

    p1_outgoing = _known_outgoing_amount(state, "p1", recipient="p2")
    p2_outgoing = _known_outgoing_amount(state, "p2", recipient="p1")
    if p1_outgoing is None or p2_outgoing is None:
        return None
    net = p1_outgoing - p2_outgoing
    return math.copysign(math.log1p(abs(net)), net) if net else 0.0


def _known_outgoing_amount(
    state: Mapping[str, Any], side: str, *, recipient: str,
) -> float | None:
    """送付済み残量と、まだ確定していない攻撃量を同じ向きで足す。"""

    pending = _preferred_pending(state, recipient)
    provisional = state.get(f"b_{side}_provisional_generated")
    if pending is None:
        return None
    if provisional is None:
        if bool(state.get(f"b_{side}_chain_active")):
            return None
        provisional = 0
    pending_value = _nonnegative_number(pending)
    provisional_value = _nonnegative_number(provisional)
    return pending_value + provisional_value


def _preferred_pending(state: Mapping[str, Any], side: str) -> Any:
    """因果会計がある新しい表では、推定落下を含む旧残量を使わない。"""

    key = f"b_{side}_causal_pending_garbage"
    return state.get(key) if key in state else state.get(f"b_{side}_pending_garbage")


def _nonnegative_number(value: Any) -> float:
    number = float(value)
    if number < 0 or not math.isfinite(number):
        raise EventIndicatorFeatureError("攻撃量が非負の有限値ではありません")
    return number


def _c_features(
    state: Mapping[str, Any], links: Sequence[Mapping[str, Any]],
    histories: dict[tuple[str, int], defaultdict[tuple[str, str], int]],
) -> dict[str, float]:
    history_key = (str(state["source_video_id"]), int(state["online_segment_index"]))
    history = histories.setdefault(history_key, defaultdict(int))
    current: defaultdict[tuple[str, str], int] = defaultdict(int)
    for link in links:
        event_type = str(link.get("event_type", ""))
        side = str(link.get("side", ""))
        if event_type in EVENT_HISTORY_TYPES and side in SIDES:
            history[(event_type, side)] += 1
            current[(event_type, side)] += 1
    features: dict[str, float] = {}
    for event_type in EVENT_HISTORY_TYPES:
        _add_history_pair(features, event_type, history, current)
    return features


def _add_history_pair(
    output: dict[str, float], event_type: str,
    history: Mapping[tuple[str, str], int], current: Mapping[tuple[str, str], int],
) -> None:
    p1 = float(math.log1p(history.get((event_type, "p1"), 0)))
    p2 = float(math.log1p(history.get((event_type, "p2"), 0)))
    _add_paired(output, f"c_history_{event_type}_log", p1, p2, add_missing=False)
    output[f"c_current_{event_type}_p1"] = float(current.get((event_type, "p1"), 0) > 0)
    output[f"c_current_{event_type}_p2"] = float(current.get((event_type, "p2"), 0) > 0)


def _add_paired(
    output: dict[str, float | None], prefix: str,
    p1: float | None, p2: float | None, *, add_missing: bool = True,
) -> None:
    output[f"{prefix}_p1"] = p1
    output[f"{prefix}_p2"] = p2
    both = p1 is not None and p2 is not None
    output[f"{prefix}_diff"] = None if not both else p1 - p2
    output[f"{prefix}_max"] = None if not both else max(p1, p2)
    output[f"{prefix}_sum"] = None if not both else p1 + p2
    if add_missing:
        output[f"{prefix}_p1_missing"] = float(p1 is None)
        output[f"{prefix}_p2_missing"] = float(p2 is None)


def _feature_name_groups(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    names = sorted(key for key in rows[0] if key not in META_KEYS) if rows else []
    a_names = tuple(name for name in names if name.startswith("a_"))
    b_only = tuple(name for name in names if name.startswith("b_"))
    c_only = tuple(name for name in names if name.startswith("c_"))
    return {"A": a_names, "B": a_names + b_only, "C": a_names + b_only + c_only}


def _ratio(value: Any, denominator: int) -> float:
    number = int(value)
    if number < 0 or number > denominator:
        raise EventIndicatorFeatureError("盤面個数が範囲外です")
    return float(number / denominator)


def _optional_log(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if number < 0 or not math.isfinite(number):
        raise EventIndicatorFeatureError("非負の観測値ではありません")
    return float(math.log1p(number))


def _optional_bool(value: Any) -> float | None:
    return None if value is None else float(bool(value))


def _optional_color(value: Any) -> int | None:
    if value is None:
        return None
    color = int(value)
    return color if color in COLOR_VALUES else None


def _color_match_ratio(
    grid: tuple[int, ...], color: int | None, color_total: int,
) -> float | None:
    if color is None:
        return None
    return float(sum(cell == color for cell in grid) / max(1, color_total))

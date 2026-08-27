"""Gate 4 条件1〜5を、同じ実表示行とWIN★正解で横並び集計する。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


ROOT = Path("data/verify/gate4_formal_dense_2026-08-26")
CONDITION5_ROOT = Path("data/verify/gate4_condition5_2026-08-26")
TRUTH_PATH = ROOT / "win_panel_truth.tsv"
PAIR_ANALYZER = Path(__file__).with_name(
    "_analyze_pm100_display_pair_2026-08-26.py")
CONDITION_DIRS = (
    ("1", "OFF基準", ROOT / "cond1_off_baseline"),
    ("2", "ヒステリシス", ROOT / "cond2_hysteresis_only"),
    ("3", "規模比較", ROOT / "cond3_scale_compare_only"),
    ("4", "ヒステリシス+規模比較", ROOT / "cond4_a_plus_b"),
    ("5", "交換エピソード", CONDITION5_ROOT / "cond5_exchange_episode_v12"),
)
OPTIONAL_CONDITION_DIRS = (
    ("5H", "交換エピソード+ヒステリシス", CONDITION5_ROOT / "cond5_hysteresis_v6"),
)


def _load_pair_analyzer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gate4_pair_analyzer", PAIR_ANALYZER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"比較器を読み込めない: {PAIR_ANALYZER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _final_metrics(
    module: ModuleType,
    baseline: list[Any],
    current: list[Any],
    truths: dict[tuple[str, int], str],
) -> tuple[int, int, int]:
    base_final = module._final_displays(baseline)
    current_final = module._final_displays(current)
    common = sorted(set(base_final) & set(current_final))
    known = [key for key in common if key in truths]
    wrong = 0
    weakened = 0
    sign_changes = 0
    for key in common:
        sign_changes += int(np.sign(base_final[key]) != np.sign(current_final[key]))
        if key not in truths:
            continue
        want = 1.0 if truths[key] == "1P" else -1.0
        wrong += int(np.sign(current_final[key]) != want)
        weakened += int(
            abs(base_final[key]) >= module.FINAL_WEAK_TH
            and abs(current_final[key]) < module.FINAL_WEAK_TH)
    if len(known) != len(truths):
        missing = sorted(set(truths) - set(known))
        raise ValueError(f"WIN★正解を評価できない試合がある: {missing}")
    return wrong, weakened, sign_changes


def _format_row(
    condition: str,
    name: str,
    metrics: dict[str, object],
    final: tuple[int, int, int],
) -> str:
    total = float(metrics["total"])
    sticky_pct = 100.0 * float(metrics["stick"]) / total
    reverse_pct = 100.0 * float(metrics["wrong"]) / total
    wrong, weakened, sign_changes = final
    values = (
        condition, name, f"{total:.1f}", f"{sticky_pct:.2f}",
        f"{reverse_pct:.2f}", str(len(metrics["flip_games"])),
        str(metrics["flip_events"]), str(metrics["swings"]), str(wrong),
        str(weakened), str(sign_changes), str(metrics["gap_bad"]),
    )
    return "\t".join(values)


def _validate_complete(module: ModuleType, baseline: list[Any], current: list[Any]) -> None:
    if len(current) != len(baseline):
        raise ValueError(
            f"区間数が不足: 基準={len(baseline)} 対象={len(current)}")
    module._assert_paired(baseline, current)


def main() -> int:
    module = _load_pair_analyzer()
    baseline = module._load_dir(CONDITION_DIRS[0][2])
    truths = module._load_panel_truth(TRUTH_PATH)
    header = (
        "condition\tname\tdisplay_sec\tsticky_pct\treverse_pct\t"
        "flip_games\tflip_events\tswings\tfinal_wrong\tlethal_weakened\t"
        "off_sign_changes\tdense_gap_bad"
    )
    print(header)
    available = list(CONDITION_DIRS)
    for item in OPTIONAL_CONDITION_DIRS:
        if len(list(item[2].glob("seg*_display.npz"))) == len(baseline):
            available.append(item)
    for condition, name, path in available:
        current = module._load_dir(path)
        _validate_complete(module, baseline, current)
        metrics = module._metrics(current)
        final = _final_metrics(module, baseline, current, truths)
        print(_format_row(condition, name, metrics, final))
    return 0


if __name__ == "__main__":
    sys.exit(main())

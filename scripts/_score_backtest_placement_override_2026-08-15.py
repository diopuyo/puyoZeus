"""placement_override 全域バックテスト (拡張代表サンプル) の A/B 比較 (2026-08-15)。

`scripts/_backtest_placement_override_full_2026-08-15.py` が出力した
data/indicators_v2/backtest_placement_override_2026-08-15/{a,b}/*.npz と
{a,b}_trace/*.npz を読み込み、以下を構成a (本番採用構成) / 構成b
(a + placement_override修正版) で比較する:

  1. 品質ゲート (scripts/phase_l_video_quality_gate.py を re-import して再利用)
  2. OJAMA_FALL 滞在時間分布 + 0.35秒未満振動遷移数 (状態トレースから算出)
  3. 盤面churn (隣接STABLE snapshot間のセル変化量分布)
  4. 重力違反率 (src.self_supervised.physical_consistency.check_gravity_rule)
  5. 幻連鎖疑い (chain_trigger_sec ありなのに score 増分 0 の遷移数)

出力: data/verify/backtest_placement_override_2026-08-15/summary.md
      + 個別 csv 一式

使い方:
    PYTHONPATH=. ./venv/bin/python -m \\
        scripts._score_backtest_placement_override_2026-08-15
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.board import Board  # noqa: E402
from src.self_supervised.physical_consistency import check_gravity_rule  # noqa: E402
import scripts.phase_l_video_quality_gate as qgate  # noqa: E402

IN_ROOT: Path = _ROOT / "data" / "indicators_v2" / "backtest_placement_override_2026-08-15"
OUT_DIR: Path = _ROOT / "data" / "verify" / "backtest_placement_override_2026-08-15"
CONFIGS: tuple[str, ...] = ("a", "b")

# 振動判定の実時間閾値 (scripts/_diag_scene1_oscillation_recheck_2026-08-15.py
# と同一の既定値、場面1の実測振動周期0.15-0.3秒より広めの安全側カットオフ)。
OSCILLATION_DT_THRESHOLD_SEC: float = 0.35
OJAMA_FALL_STATE_NAME: str = "ojama_fall"


@dataclass
class TraceStats:
    """1 構成分の状態トレース集計結果。"""

    n_oscillations: dict[str, int]
    ojama_fall_dwell_sec: list[float]


def _load_npz_paths(config: str) -> list[Path]:
    return sorted((IN_ROOT / config).glob("*.npz"))


def _load_trace_paths(config: str) -> list[Path]:
    return sorted((IN_ROOT / f"{config}_trace").glob("*.npz"))


def compute_trace_stats(config: str) -> TraceStats:
    """状態トレースから振動回数・OJAMA_FALL滞在時間分布を計算する。"""
    n_osc: dict[str, int] = {"1P": 0, "2P": 0}
    dwell: list[float] = []
    for path in _load_trace_paths(config):
        d = np.load(path, allow_pickle=True)
        state_names = list(d["state_names"])
        ojama_code = state_names.index(OJAMA_FALL_STATE_NAME)
        for side_code, side_label in ((0, "1P"), (1, "2P")):
            mask = d["side"] == side_code
            t = d["t_sec"][mask]
            st = d["state"][mask]
            if len(t) == 0:
                continue
            order = np.argsort(t)
            t, st = t[order], st[order]
            n_osc[side_label] += _count_oscillations(t, st, ojama_code)
            dwell.extend(_ojama_fall_dwell_durations(t, st, ojama_code))
    return TraceStats(n_oscillations=n_osc, ojama_fall_dwell_sec=dwell)


def _count_oscillations(t: np.ndarray, st: np.ndarray, ojama_code: int) -> int:
    """OJAMA_FALL を離脱した直後 (< 閾値秒) に再突入した回数を数える (真の往復振動)。

    2026-08-15 バグ修正: 当初は「隣接サンプル間の任意の遷移で dt<閾値」を数えて
    いたが、フレームサンプリング間隔自体が既に閾値未満 (~30fps=0.033秒) のため、
    通常の1回限りの OJAMA_FALL 突入/離脱まで無差別に「振動」として過大計上して
    いた (config bで悪化して見える逆転が発覚、自己検収で発見)。正しくは
    「離脱→短時間内の再突入」(scene1実測の0.15-0.3秒周期パターン) のみを数える。
    """
    n = 0
    last_exit_t: float | None = None
    for i in range(1, len(st)):
        left_ojama = st[i - 1] == ojama_code and st[i] != ojama_code
        entered_ojama = st[i - 1] != ojama_code and st[i] == ojama_code
        if left_ojama:
            last_exit_t = t[i]
        elif entered_ojama:
            if (
                last_exit_t is not None
                and (t[i] - last_exit_t) < OSCILLATION_DT_THRESHOLD_SEC
            ):
                n += 1
            last_exit_t = None
    return n


def _ojama_fall_dwell_durations(
    t: np.ndarray, st: np.ndarray, ojama_code: int,
) -> list[float]:
    """OJAMA_FALL の連続区間長 (秒) の一覧を返す。"""
    durations: list[float] = []
    run_start: float | None = None
    for i in range(len(st)):
        if st[i] == ojama_code:
            if run_start is None:
                run_start = t[i]
        else:
            if run_start is not None:
                durations.append(t[i - 1] - run_start)
                run_start = None
    if run_start is not None:
        durations.append(t[-1] - run_start)
    return durations


def compute_churn_and_gravity(config: str) -> dict[str, Any]:
    """盤面churn分布・重力違反率を計算する (npz grids から)。"""
    cell_diffs: list[int] = []
    n_snapshots = 0
    n_gravity_violations = 0
    n_fake_chain = 0
    for path in _load_npz_paths(config):
        d = np.load(path, allow_pickle=True)
        grids = d["grids"]
        t_sec = d["t_sec"]
        side = d["side"]
        video_id = d["video_id"]
        score = d["score"] if "score" in d else np.full(len(grids), -1)
        chain_trigger = (
            d["chain_trigger_sec"] if "chain_trigger_sec" in d
            else np.full(len(grids), np.nan)
        )
        n_snapshots += len(grids)
        for g in grids:
            ok, violations = check_gravity_rule(_grid_to_board(g))
            if not ok:
                n_gravity_violations += len(violations)
        keys = {(vid, s) for vid, s in zip(video_id, side)}
        for vid, s in keys:
            m = (video_id == vid) & (side == s)
            order = np.argsort(t_sec[m])
            sub_grids = grids[m][order]
            sub_score = score[m][order]
            sub_trigger = chain_trigger[m][order]
            for i in range(1, len(sub_grids)):
                diff = int(np.count_nonzero(sub_grids[i] != sub_grids[i - 1]))
                cell_diffs.append(diff)
                had_chain = not np.isnan(sub_trigger[i]) if sub_trigger.dtype.kind == "f" else False
                score_delta = int(sub_score[i]) - int(sub_score[i - 1])
                if had_chain and score_delta <= 0:
                    n_fake_chain += 1
    return {
        "n_snapshots": n_snapshots,
        "cell_diffs": cell_diffs,
        "n_gravity_violations": n_gravity_violations,
        "n_fake_chain_suspects": n_fake_chain,
    }


def _grid_to_board(grid: np.ndarray) -> Board:
    """npz の生グリッドを Board へ包む (from_list の厳格 validation を避け、
    既存 collect_boards_lean.py と同じ private attribute 直接代入方式を使う)。
    """
    b = Board()
    b._grid = grid.astype(np.uint8)
    return b


def _non_empty_npz_paths(npz_dir: Path) -> dict[str, Path]:
    """0 snapshot (STABLE盤面が1件も無いチャンク) を除外した video_id->path を返す。

    2分チャンクという短い窓のため、対戦が行われていない/両側とも
    STABLE化しなかった区間がまれに存在し grids.shape==(0,) になる
    (phase_l_video_quality_gate.py は元々フル尺前提でこのケースを
    想定していないため、ここで先に弾くread-onlyの互換処理)。
    """
    out: dict[str, Path] = {}
    for video_id, path in qgate._collect_npz_paths(npz_dir).items():
        d = np.load(path, allow_pickle=True)
        if d["grids"].ndim == 3 and d["grids"].shape[0] > 0:
            out[video_id] = path
        else:
            print(f"[gate] skip empty chunk: {path.name}")
    return out


def run_quality_gate(config: str) -> str:
    """既存の phase_l_video_quality_gate ロジックを再利用して品質ゲートを実行する。"""
    npz_dir = IN_ROOT / config
    npz_paths = _non_empty_npz_paths(npz_dir)
    library_rates = {
        vid: qgate.compute_color_col_rates(qgate.load_video_arrays(p))
        for vid, p in npz_paths.items()
    }
    results = []
    for video_id, path in npz_paths.items():
        arrays = qgate.load_video_arrays(path)
        results.append(qgate.evaluate_video(arrays, library_rates))
    out_dir = OUT_DIR / f"quality_gate_{config}"
    qgate.write_scorecard(results, out_dir, filename="scorecard.tsv")
    n_fail = sum(1 for r in results if r.verdict == "FAIL")
    n_warn = sum(1 for r in results if r.verdict == "WARN")
    n_review = sum(1 for r in results if r.verdict == "REVIEW")
    n_pass = len(results) - n_fail - n_warn - n_review
    return (
        f"{len(results)}チャンク中 PASS={n_pass} WARN={n_warn} "
        f"REVIEW={n_review} FAIL={n_fail}"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# placement_override 全域バックテスト (拡張代表サンプル) 結果\n"]
    summary: dict[str, dict[str, Any]] = {}
    for config in CONFIGS:
        trace_stats = compute_trace_stats(config)
        churn = compute_churn_and_gravity(config)
        gate_summary = run_quality_gate(config)
        diffs = np.array(churn["cell_diffs"]) if churn["cell_diffs"] else np.array([0])
        summary[config] = {
            "gate": gate_summary,
            "n_snapshots": churn["n_snapshots"],
            "n_gravity_violations": churn["n_gravity_violations"],
            "n_fake_chain_suspects": churn["n_fake_chain_suspects"],
            "churn_median": float(np.median(diffs)),
            "churn_p95": float(np.percentile(diffs, 95)),
            "churn_max": int(diffs.max()),
            "n_oscillations_1p": trace_stats.n_oscillations["1P"],
            "n_oscillations_2p": trace_stats.n_oscillations["2P"],
            "n_dwell_events": len(trace_stats.ojama_fall_dwell_sec),
            "dwell_median_sec": (
                float(np.median(trace_stats.ojama_fall_dwell_sec))
                if trace_stats.ojama_fall_dwell_sec else float("nan")
            ),
            "dwell_p95_sec": (
                float(np.percentile(trace_stats.ojama_fall_dwell_sec, 95))
                if trace_stats.ojama_fall_dwell_sec else float("nan")
            ),
        }
        n_osc_total = (
            summary[config]["n_oscillations_1p"] + summary[config]["n_oscillations_2p"]
        )
        n_entries = summary[config]["n_dwell_events"]
        # 振動「率」(= 往復振動数 / OJAMA_FALL突入総数)。大連鎖後の複数波の
        # 正規のおじゃま降下 (真の再突入) が母数に含まれるため、率で正規化しないと
        # 突入回数自体が多い構成が不利に見える (2026-08-15 自己検収で発見)。
        summary[config]["oscillation_rate"] = (
            n_osc_total / n_entries if n_entries > 0 else float("nan")
        )
        print(f"[{config}] {summary[config]}")

    lines.append("| 指標 | a (本番構成) | b (a+placement_override) |")
    lines.append("|---|---|---|")
    for key in (
        "gate", "n_snapshots", "n_gravity_violations", "n_fake_chain_suspects",
        "churn_median", "churn_p95", "churn_max",
        "n_oscillations_1p", "n_oscillations_2p", "n_dwell_events",
        "oscillation_rate", "dwell_median_sec", "dwell_p95_sec",
    ):
        lines.append(f"| {key} | {summary['a'][key]} | {summary['b'][key]} |")
    out_path = OUT_DIR / "summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()

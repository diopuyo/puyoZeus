"""段1の固定集計。選別は動画単位対bootstrapの行重みLLで行う。"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scripts.gfe_phase_stage1_20261002 import (
    BOOTSTRAPS, BOOTSTRAP_SEED, OUTPUT, PHASES, STAGE_LIMIT, VARIANTS,
    metric, save_json,
)


def paired_ll(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
              groups: np.ndarray, video_count: int) -> dict[str, Any]:
    """動画を復元抽出し、各動画の行数を維持したLL差を返す。"""
    epsilon = np.finfo(np.float64).eps
    left, right = [np.clip(p, epsilon, 1-epsilon) for p in (baseline, candidate)]
    delta = -y * (np.log(right)-np.log(left)) - (1-y) * (np.log1p(-right)-np.log1p(-left))
    counts = np.bincount(groups, minlength=video_count)
    sums = np.bincount(groups, weights=delta, minlength=video_count)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAPS)
    for iteration in range(BOOTSTRAPS):
        sample = rng.integers(0, video_count, video_count)
        values[iteration] = sums[sample].sum() / counts[sample].sum()
    return dict(delta=float(delta.mean()), ci95=np.quantile(values, [.025, .975]).tolist())


def scope_report(part: pd.DataFrame) -> dict[str, Any]:
    """同じ行とseedを束ね、動画抽出は全構成で対にする。"""
    groups, videos = pd.factorize(part.video_id, sort=True)
    y = part.label.to_numpy()
    metrics = {v: metric(y, part[v].to_numpy()) for v in VARIANTS}
    differences = {v: paired_ll(y, part.V0.to_numpy(), part[v].to_numpy(), groups, len(videos))
                   for v in VARIANTS if v != "V0"}
    return dict(rows=int(part.row_id.nunique()), games=int(part.game_id.nunique()),
                videos=len(videos), predictions=len(part), metrics=metrics, differences=differences)


def evaluate(combined: pd.DataFrame) -> dict[str, Any]:
    """U・現行fill・既存記録P_timeを区別して全表を返す。"""
    scopes = {"overall": scope_report(combined)}
    for column in ("U", "current_fill", "phase"):
        for stage, name in enumerate(PHASES):
            key = f"{column}_{name}"
            scopes[key] = scope_report(combined[combined[column] == stage])
            print(f"集計完了: {key}", flush=True)
    decisions = {}
    for variant in VARIANTS[1:]:
        means = [scopes[f"U_{name}"]["differences"][variant]["delta"] for name in PHASES]
        upper = scopes["overall"]["differences"][variant]["ci95"][1]
        checks = dict(overall_upper_lt_zero=upper < 0, all_U_means_le_limit=all(x <= STAGE_LIMIT for x in means))
        decisions[variant] = dict(checks=checks, passed=all(checks.values()), U_mean_deltas=means)
    return dict(scopes=scopes, decisions=decisions)


def write_summary(report: dict[str, Any]) -> None:
    """母数と全指標をMarkdownへ出力し、採否判断とは区別する。"""
    lines = ["# G_fe 局面比較 段1", "", f"状態: {report['status']}", "",
             f"母数: {report['rows']:,}行 / {report['games']:,}試合 / {report['videos']}動画。",
             f"実測実行時間: {report['elapsed_seconds']:.2f}秒。", "",
             "V0=59列、V1=60列、V1b/V3=54列、V2=192列。",
             "V2は基準段階を省かず、46指標×3段階=138列を追加。",
             "LRは標準化なし、C=1、lbfgs、max_iter=1000、tol=1e-4、random_state=0。",
             "Uの相手欠測は自側段階と欠測フラグで扱う。",
             "ECEは10等幅区間。CIは行重みLL差の動画単位対bootstrap（10,000回、seed 0）。", ""]
    if "scopes" in report:
        lines += ["|区分|行/試合/動画|構成|LL|AUC|ECE|ΔLL|95%区間|", "|---|---|---|---:|---:|---:|---:|---|"]
        for scope, values in report["scopes"].items():
            for variant, m in values["metrics"].items():
                d = values["differences"].get(variant)
                diff = f"{d['delta']:+.9f}|[{d['ci95'][0]:+.9f}, {d['ci95'][1]:+.9f}]" if d else "—|—"
                lines.append(f"|{scope}|{values['rows']}/{values['games']}/{values['videos']}|{variant}|"
                             f"{m['log_loss']:.9f}|{m['auc']:.9f}|{m['ece']:.9f}|{diff}|")
        lines += ["", "段1条件の機械判定（採否は統括）:", "```json",
                  __import__("json").dumps(report["decisions"], ensure_ascii=False, indent=2), "```"]
    else:
        lines += ["V0不一致のためV1/V1b/V2/V3は未実行。構成変更・再試行は行っていない。", "",
                  "|区分|再現LL|既存LL|ΔLL|再現AUC|既存AUC|ΔAUC|一致|", "|---|---:|---:|---:|---:|---:|---:|---|"]
        for scope, check in report["reproduction"].items():
            a, e, d = check["actual"], check["expected"], check["delta"]
            lines.append(f"|{scope}|{a['log_loss']:.12f}|{e['log_loss']:.12f}|{d['log_loss']:+.12f}|"
                         f"{a['auc']:.12f}|{e['auc']:.12f}|{d['auc']:+.12f}|{check['passed']}|")
    (OUTPUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(OUTPUT / "METRICS.json", report)

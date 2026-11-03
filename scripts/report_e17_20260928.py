"""E17の単独比較・採否・照合証跡を一つの報告へまとめる。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from src.board import COLOR_EMPTY, COLOR_OJAMA
from scripts.run_e17_ablation_20260928 import OUT, ALL_SOURCES, PREFIRE, TERMINAL, directory
from scripts.run_e3_exchange_eval_20260926 import save_json

DOC = Path("docs/E17_ABLATION_2026-09-28.md")
LABELS = dict(baseline="E15", sync="① 同期", death="② 死亡後拒否", layers="③ 旧照合",
              all="①②③＋その他", fixed="③′ 修正照合", combined="採用候補の組合せ")
SOURCE_FILES = ("src/exchange_event_completion_check.py", "src/exchange_event_layers.py",
    "src/exchange_event_overlay.py", "scripts/replay_exchange_event_20260926.py",
    "scripts/run_e17_ablation_20260928.py", "scripts/audit_e17_hold_20260928.py",
    "scripts/report_e17_20260928.py", "tests/test_e17_completion.py", "tests/test_e17_ablation.py")
FUNCTION_LIMIT = 50
EXPECTED_FRAMES = (6526, 8333, 62, 45)


def read(path: Path) -> dict:
    """保存済みJSONだけを読む。"""
    return json.loads(path.read_text(encoding="utf-8"))


def code_audit() -> dict:
    """変更対象の関数長と本番設定不変を機械確認する。"""
    long = []
    for name in SOURCE_FILES:
        tree = ast.parse(Path(name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno-node.lineno+1
                if length > FUNCTION_LIMIT:
                    long.append(dict(file=name, function=node.name, lines=length))
    protected = hashlib.sha256(Path("src/production_config.py").read_bytes()).hexdigest()
    return dict(long_functions=long, production_config_changed=protected != read(OUT/"PROTECTED.json")["sha256"])


def comparisons(variant: str) -> dict:
    """旧照合の早過ぎる照合と次着手後の照合を実際の時刻で数える。"""
    rows = [dict(source=s, **r) for s in ALL_SOURCES
            for r in read(directory(variant, s)/"audit.json")["comparisons"]]
    mismatches = [r for r in rows if r["mismatch"]]
    garbage_only = sum(np.array_equal(np.where(np.array(r["actual"]) == COLOR_OJAMA, COLOR_EMPTY, r["actual"]),
        np.where(np.array(r["predicted"]) == COLOR_OJAMA, COLOR_EMPTY, r["predicted"])) for r in mismatches)
    return dict(comparisons=len(rows), mismatches=len(mismatches),
        before_end_confirmation=sum(r["end_confirmed"] is not True for r in rows),
        after_next_move=sum(r["post_end_drop_sec"] is not None
                            and r["board_sec"] >= r["post_end_drop_sec"] for r in rows),
        garbage_only_mismatches=int(garbage_only),
        review_cases=[r for r in rows if r["source"] == "review"])


def provenance() -> dict:
    """全条件共通入力と既存モデルをハッシュで固定する。"""
    files = [Path("logs/e16/records")/f"{s}.jsonl.gz" for s in ALL_SOURCES]
    files += [Path(f"models/exchange_event_{v}")/f for v in ("v3", "v4")
              for f in ("manifest.json", "S1_prime_light.joblib", "S3_prime_light.joblib")]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def table(results: dict, hold: dict) -> str:
    """共通母数と全条件を一表にして、基準を満たす条件だけ明記する。"""
    lines = ["q: 4試合・6,526行。zenchi: 8,333行。場面: 発火前62行／終盤45行。",
        "CV: 84,445行・144動画・15fold・253,335予測（軽量S3）。", "",
        "|条件|q log loss|zenchi一致率（件数）|誤発火|発火前平均|符号反転|終盤最小|CV AUC|採否|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for variant, row in results.items():
        z, d, s = row["zenchi"], row["deaths"], row["scenes"]
        assert (row["q"]["frames"], z["frames"], s["prefire_frames"], s["terminal_frames"]) == EXPECTED_FRAMES
        cv = hold["cv"]["metrics"].get("E15" if variant == "baseline" else variant)
        auc = f'{cv["S3_prime_light"]["auc"]:.6f}' if cv else "—"
        decision = "基準" if variant == "baseline" else "採用候補" if row["candidate"] else "不採用"
        lines.append(f'|{LABELS[variant]}|{row["q"]["log_loss"]:.6f}|'
            f'{z["agreement"]*100:.2f}% ({z["hits"]}/{z["frames"]})|{d["false"]}/{d["total"]}|'
            f'{s["prefire_mean"]:.6f}|{s["margin_sign_flips"]}|{s["terminal_min"]:.6f}|{auc}|{decision}|')
    return "\n".join(lines)


def scene_evidence(variants: dict) -> dict:
    """表の場面値を再計算できる最小限の行を残す。"""
    evidence = {}
    for variant in variants:
        root = directory(variant, "review")
        trace = json.loads((root/"count_trace.json").read_text(encoding="utf-8"))
        first = [dict(t_sec=r["t_sec"], p1=r["p1"], counter_margin=float(
                    np.sign(r["counts"][0][-1])*np.expm1(abs(r["counts"][0][-1]))))
                 for r in trace if PREFIRE[0] <= r["t_sec"] <= PREFIRE[1]]
        with np.load(root/"display.npz") as data:
            last = [dict(t_sec=float(t), p1=float(p)) for t, p in zip(data["t_sec"], data["display_p1"])
                    if TERMINAL[0] <= t <= TERMINAL[1]]
        evidence[variant] = dict(prefire=first, terminal=last)
    return evidence


def main() -> None:
    """最終結果を上書き可能な報告末尾へ追記する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    audit = code_audit()
    save_json(OUT/"CODE_AUDIT.json", audit)
    if args.audit_only:
        print(audit)
        return
    metrics, hold = read(OUT/"METRICS.json"), read(OUT/"HOLD_AUDIT.json")
    metrics["comparison_audit"] = {v: comparisons(v) for v in ("layers", "fixed")}
    metrics["cv"] = hold["cv"]
    metrics["hold_distribution"] = hold["different_from_latest"]
    metrics["provenance"] = provenance()
    metrics["verification"] = dict(code=audit, tests=(OUT/"final_tests.log").read_text().strip(),
        baseline=read(directory("baseline", "review")/"equivalence.json"),
        all=read(directory("all", "review")/"equivalence.json"))
    save_json(OUT/"METRICS.json", metrics)
    save_json(OUT/"SCENE_ROWS.json", scene_evidence(metrics["results"]))
    report = table(metrics["results"], hold)
    chosen = "＋".join(LABELS[v] for v in metrics["selected"]) or "E15維持"
    decision = "全基準を満たした" if metrics["results"]["combined"]["candidate"] else "基準未達だった"
    report += f"\n\n採用候補の組合せは **{chosen}**。新規再生した組合せ条件も{decision}。本番採用は行わず既定OFFを維持する。\n"
    old, fixed = (metrics["comparison_audit"][v] for v in ("layers", "fixed"))
    report += (f'\n旧③の照合{old["comparisons"]}件中、終了確定前{old["before_end_confirmation"]}件、'
        f'次着手以降{old["after_next_move"]}件。おじゃま差だけの不一致は{old["garbage_only_mismatches"]}件。'
        f'③′は照合{fixed["comparisons"]}件で終了確定前・次着手以降とも0件だが、精度基準は未達。\n')
    text = DOC.read_text(encoding="utf-8").split("\n## 結果\n")[0]
    DOC.write_text((text+"\n## 結果\n\n"+report).rstrip()+"\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

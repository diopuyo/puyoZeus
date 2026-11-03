"""1動画ぶんの B3 学習行を生成して parquet + manifest を出力する。

使い方: python -m scripts.b3_generate_rows --video video_51 --out D:/puyo_analyzer/b3_data/rows_v1 \
          --timing-table D:/puyo_analyzer/b3_data/timing_v1/timing_table.json
完了票のある attempt だけを読む。完了マーカー <video>.done が最後に作られる (未完ファイルを採用しない)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.b3_column_spec import COLUMN_SPECS, SCHEMA_VERSION, Kind, render_markdown
from src.b3_rows import (label_vs_death_audit, HEARTBEAT_SEC, TRIGGER_BITS, finalize_weights, generate_rows, to_frame, with_mirrors)
from src.b3_source import B2_ROOT_DEFAULT, MATCH_END_GRACE_SEC, load_lean, obs_game_ranges, resolve_attempt, sha256_file, winners_by_overlap
from src.b3_timing import TimingTable

CODE_FILES = ("src/scoring.py", "src/b3_column_spec.py", "src/b3_timing.py", "src/b3_search.py", "src/b3_source.py",
              "src/b3_state.py", "src/b3_chain_prediction.py", "src/exchange_prefire_candidates.py",
              "src/b3_timeaxis.py", "src/b3_features.py", "src/b3_rows.py", "src/b3_score_ledger.py", "src/b3_score_ledger_port.py",
              "scripts/b3_generate_rows.py")
ROOT = Path(__file__).resolve().parents[1]


def evaluation_rows(record: Path, video: str, table: TimingTable, winners: dict,
                    mapping: dict, max_frame: int | None = None,
                    allow_unknown_winner: bool = False) -> tuple[list[dict], dict]:
    """既知勝者の生成履歴を維持し、評価用には未確定試合だけを別走査で補う。"""
    rows, info = generate_rows(record, video, table, max_frame=max_frame, winners=winners,
                               require_winner=True, match_mapping=mapping)
    unknown = {g: span for g, span in mapping.items() if winners.get(g) is None}
    if not allow_unknown_winner or not unknown:
        return rows, info
    extra, extra_info = generate_rows(record, video, table, max_frame=max_frame, winners=winners,
                                     require_winner=False, match_mapping=unknown)
    rows.extend(extra)
    rows.sort(key=lambda r: r["frame_idx"])
    info["rows"] = len(rows)
    info["skipped_no_winner_frames"] = 0
    info["leak_violations"] += extra_info["leak_violations"]
    info["stable_death"].update(extra_info["stable_death"])
    counts = info["excluded_outside_match_rows_by_game"]
    counts.update({g: n for g, n in extra_info["excluded_outside_match_rows_by_game"].items()
                   if int(g) in unknown})
    info["excluded_outside_match_rows"] = sum(counts.values())
    return rows, info


def evaluation_audit(rows: list[dict], stable_death: dict, allow_unknown_winner: bool) -> dict:
    """未確定試合をラベル照合から除外し、評価用では対象・除外母数も記録する。"""
    if not allow_unknown_winner:
        return label_vs_death_audit(rows, stable_death)
    known = [r for r in rows if np.isfinite(r["won_p1"])]
    audit = label_vs_death_audit(known, stable_death)
    total = len({r["game_idx"] for r in rows})
    audit.update(total_games=total, excluded_unknown_winner_games=total - audit["games"])
    return audit


def seal_outputs(df: pd.DataFrame, manifest: dict, out: Path, video: str) -> None:
    """教師と監査詳細を隔離し、通常出力には特徴と通過状態だけを残す。"""
    label_path = out / f"{video}.sealed_label.parquet"
    df[["row_key", "won_p1"]].to_parquet(label_path, engine="pyarrow", compression="zstd", index=False)
    private = {key: manifest.pop(key) for key in ("winners", "label_death_audit")}
    private["stable_death"] = manifest["frame_info"].pop("stable_death")
    (out / f"{video}.sealed_label.json").write_text(json.dumps(private, ensure_ascii=False), encoding="utf-8")
    df.drop(columns=["won_p1"], inplace=True)
    manifest["missing"]["nan_rate"].pop("won_p1", None)
    manifest.update(n_columns=len(df.columns), labels_file=label_path.name, label_death_audit_passed=True)


def code_sha() -> dict:
    """コード SHA: 生成器ファイルの内容ハッシュ + git HEAD。"""
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update((ROOT / f).read_bytes())
    try:
        head = subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        head = ""
    return {"files_sha256": h.hexdigest(), "git_head": head}


def ledger_info() -> dict:
    """台帳v2 の mode とコミット SHA (src/b3_score_ledger.py を最後に変えたコミット)。"""
    from src.b3_score_ledger import ScoreLedger
    sha = subprocess.run(["git", "-c", "safe.directory=*", "log", "-1", "--format=%H", "--", "src/b3_score_ledger.py"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    return {"mode": ScoreLedger().core.mode, "commit": sha}


def missing_summary(df) -> dict:
    """NaN/mask 集計 (母数つき)。"""
    n = len(df)
    nan_rate = {c: float(df[c].isna().mean()) for c in df.columns if df[c].dtype.kind in "fO" or str(df[c].dtype).startswith(("Int", "float"))}
    masks = {}
    for c in df.columns:
        if c.startswith("mk_"):
            vc = df[c].value_counts(dropna=False)
            masks[c] = {str(k): int(v) for k, v in vc.items()}
    return {"rows": n, "nan_rate": nan_rate, "mask_counts": masks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timing-table", type=Path)
    ap.add_argument("--max-frame", type=int)
    ap.add_argument("--b2-root", type=Path, default=B2_ROOT_DEFAULT)
    ap.add_argument("--seal-labels", action="store_true", help="教師と監査詳細を sealed_label ファイルへ隔離")
    ap.add_argument("--allow-unknown-winner", action="store_true", help="評価用に勝者未確定試合も生成する")
    a = ap.parse_args()
    t0 = time.time()
    p = resolve_attempt(a.video, root=a.b2_root)
    table = TimingTable.from_json(a.timing_table.read_text(encoding="utf-8")) if a.timing_table else TimingTable()
    obs_sha = sha256_file(p.record / "observations.jsonl.gz")
    receipt = json.loads(p.receipt.read_text(encoding="utf-8"))
    if receipt.get("observations_sha256") and receipt["observations_sha256"] != obs_sha:
        raise SystemExit(f"{a.video}: observations の SHA が完了票と不一致")
    lean = load_lean(p.record)
    winners, win_info = winners_by_overlap(lean, obs_game_ranges(p.record, a.max_frame))
    rows, info = evaluation_rows(p.record, a.video, table, winners, win_info, a.max_frame,
                                 a.allow_unknown_winner)
    if info["leak_violations"]:
        raise SystemExit(f"{a.video}: available_at リーク {info['leak_violations']} 行")
    audit = evaluation_audit(rows, info["stable_death"], a.allow_unknown_winner)
    if audit['mismatch'] > max(1, audit['judged'] // 20):
        if a.seal_labels:
            raise SystemExit(f"{a.video}: 窒息監査不合格（封印のため詳細非表示）")
        raise SystemExit(f"{a.video}: ラベルと窒息盤面の不一致 {audit}")
    finalize_weights(rows)
    full = with_mirrors(rows)
    df = to_frame(full)
    if a.allow_unknown_winner:
        df = pd.concat([df, df.won_p1.isna().astype("int8").rename("winner_unknown")], axis=1)
    a.out.mkdir(parents=True, exist_ok=True)
    trig = {k: int(sum(1 for r in rows if r["trigger_mask"] & b)) for k, b in TRIGGER_BITS.items()}
    games = sorted({r["game_idx"] for r in rows})
    from src.b3_search import search_manifest
    manifest = {
        "search": search_manifest(),
        "schema_version": SCHEMA_VERSION, "video_id": a.video, "attempt": p.attempt,
        "inputs": {"observations_sha256": obs_sha, "events_sha256": sha256_file(p.record / "events.jsonl"),
                   "lean_sha256": sha256_file(p.record / "lean.npz")},
        "code": code_sha(), "score_ledger": ledger_info(), "score_ledger_v2_present": (ROOT / "src/b3_score_ledger.py").exists(), "column_spec_sha256": hashlib.sha256(render_markdown().encode()).hexdigest(),
        "n_columns": len(COLUMN_SPECS), "timing_table": {"fit_scope": table.fit_scope, "meta": table.meta,
        "sha256": hashlib.sha256(table.to_json().encode()).hexdigest()}, "heartbeat_sec": HEARTBEAT_SEC,
        "games_with_rows": [int(g) for g in games], "winners": {str(k): v for k, v in winners.items()}, "label_death_audit": audit, "winner_mapping": {str(k): v for k, v in win_info.items()},
        "rows_original": len(rows), "rows_total_with_mirror": len(full), "frame_info": info, "trigger_row_counts": trig,
        "match_end_grace_sec": MATCH_END_GRACE_SEC,
        "weight_sum_per_game": {str(g): float(df[df.game_idx == g].weight.sum()) for g in games},
        "missing": missing_summary(df), "elapsed_sec": time.time() - t0, "max_frame": a.max_frame,
        "notes": ["会計台帳は特徴・教師に不使用 (r_ledger_* は R 再現専用)", "a→b→a 後半は未実装 (ta2_* は DEFERRED)",
                  "timing table は全データ推定で fold-safe ではない"],
    }
    if a.seal_labels:
        seal_outputs(df, manifest, a.out, a.video)
    if a.allow_unknown_winner:
        manifest.update(n_columns=len(df.columns), allow_unknown_winner=True,
                        winner_unknown_games=len({r["game_idx"] for r in rows if not np.isfinite(r["won_p1"])}))
    tmp = a.out / f"{a.video}.parquet.tmp"
    df.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
    tmp.replace(a.out / f"{a.video}.parquet")
    (a.out / f"{a.video}.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (a.out / f"{a.video}.done").write_text("done\n", encoding="utf-8")
    print(f"{a.video}: rows {len(rows)} (x2) elapsed {manifest['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()

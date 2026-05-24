"""Phase E ダッシュボード: 評価観点を 1 枚の md に集約.

集約観点:
    1. 教師データ品質
       - 動画数 / 試合数 / サンプル数 / ラベル分布
       - 動画別 行数 / phase 別行数
    2. モデル評価
       - video holdout / random / leave-one-video-out (LOOV) 精度
       - Phase 別 (start / mid / end) LOOV 精度
       - 5-fold CV 安定性
    3. 認識精度メトリクス
       - 動画別 両側 STABLE 率 / 連鎖検出 / 浮きぷよ
       - 異常動画リスト
    4. 特徴量評価
       - VIF (多重共線性)
       - RF importance / LR coef ランク
       - 削除推奨リスト
    5. 試合境界 / 勝者検出の信頼性
       - 動画別 検出試合数 / UNKNOWN 比率
    6. 重みセット比較 (旧 vs 新方針)
       - LEARNED_WEIGHTS_PHASE_J vs PHASE_E

利用例:
    PYTHONPATH=. ./venv/bin/python -m scripts.phase_e_dashboard \
        --csv data/training/match_features_phase_e_v01-40.csv \
        --review-tsv data/verify/phase_e_recognition_review_v01-40.tsv \
        --learn-json data/verify/learned_weights_phase_e_phase_aware_v01-40.json \
        --vif-json data/verify/multicollinearity_phase_e.json \
        --eval-json data/verify/indicator_evaluation_phase_e.json \
        --out data/verify/phase_e_dashboard.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.console_init import init_console, to_windows_path  # noqa: E402
init_console()


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def section_data_quality(rows: list[dict]) -> list[str]:
    md: list[str] = []
    md.append("## 1. 教師データ品質")
    md.append("")
    by_video: Counter = Counter(r["video_id"] for r in rows)
    by_phase: Counter = Counter(r["time_phase"] for r in rows)
    by_label: Counter = Counter(int(r["label"]) for r in rows)
    md.append(
        f"- 総サンプル数: **{len(rows):,}**, 動画数: **{len(by_video)}**"
    )
    n1p = by_label.get(1, 0)
    n2p = by_label.get(-1, 0)
    md.append(
        f"- ラベル分布: 1P 勝利 (+1) = {n1p:,} ({n1p/len(rows):.1%}) / "
        f"2P 勝利 (-1) = {n2p:,} ({n2p/len(rows):.1%})"
    )
    md.append("")
    md.append("### 1-1. Phase (time_phase) 別")
    md.append("")
    md.append("| phase | サンプル数 |")
    md.append("|:---|---:|")
    for phase, n in sorted(by_phase.items(), key=lambda kv: -kv[1]):
        md.append(f"| {phase} | {n:,} |")
    md.append("")
    md.append("### 1-2. 動画別")
    md.append("")
    md.append("| video | サンプル数 |")
    md.append("|:---|---:|")
    for vid, n in sorted(by_video.items()):
        md.append(f"| v{vid} | {n} |")
    md.append("")
    return md


def section_recognition(review_rows: list[dict]) -> list[str]:
    md: list[str] = []
    md.append("## 2. 認識精度メトリクス (動画別)")
    md.append("")
    if not review_rows:
        md.append("_(レビューデータなし)_")
        return md

    md.append(
        "- 各動画の試合 1 区間 (matches.tsv idx=1) または動画冒頭 90 秒 "
        "を pipeline 通し"
    )
    md.append("- `both_stable_rate` = 両側 STABLE 状態の frame 比率")
    md.append("- `cnn_floating` = 生 CNN 出力での浮きぷよ件数 / frame")
    md.append("")
    sorted_rows = sorted(
        review_rows, key=lambda r: -float(r["both_stable_rate"]),
    )
    md.append("| video | both_stable | chain_p1+p2 | cnn_float (avg) | 備考 |")
    md.append("|:---|---:|---:|---:|:---|")
    for r in sorted_rows:
        v = int(r["video_id"])
        bs = float(r["both_stable_rate"])
        cp1 = int(r["chain_events_p1"])
        cp2 = int(r["chain_events_p2"])
        cnn_f = (
            float(r.get("cnn_floating_per_frame_p1", 0))
            + float(r.get("cnn_floating_per_frame_p2", 0))
        ) / 2
        note = ""
        if bs > 0.95:
            note = "⚠️ 試合区間外の可能性 (intro?)"
        elif bs < 0.22:
            note = "⚠️ アクション過多 / 認識不安定"
        md.append(
            f"| v{v:02d} | {bs:.1%} | {cp1+cp2} | {cnn_f:.3f} | {note} |"
        )
    md.append("")
    return md


def section_model_eval(learn_json: dict, eval_json: dict | None) -> list[str]:
    md: list[str] = []
    md.append("## 3. モデル評価 (Phase Aware LR L2)")
    md.append("")
    phases = learn_json.get("phases", {})
    md.append(
        f"- 入力 CSV: `{learn_json.get('csv', '?')}`"
    )
    md.append(
        f"- 削減後特徴量数: {learn_json.get('n_features', '?')} "
        f"(削除 {len(learn_json.get('dropped', []))} 個)"
    )
    md.append("")
    md.append(
        "| phase | LOOV mean | LOOV std | サンプル数 | 学習 C |"
    )
    md.append("|:---|---:|---:|---:|---:|")
    overall: list[float] = []
    for name in ("start", "mid", "end"):
        r = phases.get(name)
        if not r:
            continue
        md.append(
            f"| {name} | {r['loov_mean']:.3f} | {r['loov_std']:.3f} | "
            f"{r['n_total']} | {r['final_C']} |"
        )
        overall.append(r["loov_mean"])
    if overall:
        md.append(
            f"| **average** | **{sum(overall)/len(overall):.3f}** | — | — | — |"
        )
    md.append("")
    if eval_json:
        md.append("### 3-1. 削減シナリオ別 (E-3 から)")
        md.append("")
        md.append("| シナリオ | 残特徴量 | LR (v03) | RF (v03) | LOOV mean |")
        md.append("|:---|---:|---:|---:|---:|")
        for s in eval_json.get("scenarios", []):
            md.append(
                f"| {s['name']} | {s['n_features']} | "
                f"{s['lr_video_holdout']:.3f} | "
                f"{s['rf_video_holdout']:.3f} | {s['loov_mean']:.3f} |"
            )
        md.append("")
    return md


def section_features(eval_json: dict | None,
                     vif_json: dict | None) -> list[str]:
    md: list[str] = []
    md.append("## 4. 特徴量評価")
    md.append("")
    if eval_json:
        rf_rank = eval_json.get("rf_importance", {})
        lr_rank = eval_json.get("lr_coef_abs", {})
        vif = eval_json.get("vif", {}) or {}
        md.append("### 4-1. 重要度ランキング (上位 10)")
        md.append("")
        md.append("| 順位 | RF importance | LR |coef| | VIF |")
        md.append("|---:|:---|:---|:---|")
        feats_by_rf = sorted(
            rf_rank.items(), key=lambda kv: -kv[1],
        )[:10]
        feats_by_lr = sorted(
            lr_rank.items(), key=lambda kv: -kv[1],
        )[:10]
        for i in range(10):
            rf_n, rf_v = feats_by_rf[i] if i < len(feats_by_rf) else ("—", 0)
            lr_n, lr_v = feats_by_lr[i] if i < len(feats_by_lr) else ("—", 0)
            vif_n_str = ""
            if isinstance(rf_n, str) and rf_n in vif:
                v = vif[rf_n]
                vif_n_str = (
                    f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
                )
            md.append(
                f"| {i+1} | {rf_n} ({rf_v:.4f}) | "
                f"{lr_n} ({lr_v:.4f}) | {vif_n_str} |"
            )
        md.append("")
    if vif_json:
        md.append("### 4-2. VIF 多重共線性 (削除推奨)")
        md.append("")
        feat_vif = vif_json.get("feat_vif")
        if not feat_vif:
            vif_dict = vif_json.get("vif_per_feature", {})
            if not vif_dict:
                vif_dict = {
                    n: v for n, v in vif_json.items()
                    if isinstance(v, (int, float))
                }
            feat_vif = sorted(
                vif_dict.items(),
                key=lambda kv: -(kv[1] if isinstance(kv[1], (int, float))
                                 else 1e18),
            )[:8]
        md.append("| 順位 | 特徴量 | VIF | 判定 |")
        md.append("|---:|:---|---:|:---|")
        for i, (n, v) in enumerate(feat_vif[:8], 1):
            if not isinstance(v, (int, float)):
                vstr, label = "inf", "深刻 (線形従属)"
            elif v >= 10:
                vstr, label = f"{v:.2f}", "深刻"
            elif v >= 5:
                vstr, label = f"{v:.2f}", "要警戒"
            else:
                vstr, label = f"{v:.2f}", "OK"
            md.append(f"| {i} | {n} | {vstr} | {label} |")
        md.append("")
    return md


def section_match_detection(boundary_root: Path,
                            winners_root: Path) -> list[str]:
    md: list[str] = []
    md.append("## 5. 試合境界 / 勝者検出の信頼性")
    md.append("")
    md.append("| video | 試合数 (matches) | 1P | 2P | UNKNOWN |")
    md.append("|:---|---:|---:|---:|---:|")
    for vdir in sorted(boundary_root.glob("video_*")):
        try:
            v = int(vdir.name.split("_")[1])
        except ValueError:
            continue
        matches_tsv = vdir / "matches.tsv"
        if not matches_tsv.exists():
            continue
        with matches_tsv.open(encoding="utf-8") as f:
            n_matches = max(0, sum(1 for _ in f) - 1)
        winner_path = winners_root / f"match_winners_v{v:02d}.tsv"
        n1p = n2p = nuk = 0
        if winner_path.exists():
            for r in load_tsv(winner_path):
                w = r.get("winner", "")
                if w == "1P":
                    n1p += 1
                elif w == "2P":
                    n2p += 1
                elif w == "UNKNOWN":
                    nuk += 1
        md.append(f"| v{v:02d} | {n_matches} | {n1p} | {n2p} | {nuk} |")
    md.append("")
    return md


def section_weight_compare() -> list[str]:
    md: list[str] = []
    md.append("## 6. 重みセット比較")
    md.append("")
    md.append(
        "| 重みセット | 入力データ | 特徴量数 | overall LOOV | end LOOV | 改善幅 |"
    )
    md.append("|:---|:---|---:|---:|---:|---:|")
    md.append("| LEARNED_WEIGHTS_GLOBAL (旧) | v01-03 / 1,390 | 16 | — | 0.885 | base |")
    md.append("| LEARNED_WEIGHTS_PHASE_J (旧) | v01-19 / ~1,500 | 21 | 0.655 | 0.744 | +5.4 |")
    md.append("| LEARNED_WEIGHTS_PHASE_E v1 (2026-05-05 旧 csv) | v01-19 / 4,893 | 16 | 0.644 | 0.840 | +6.6 |")
    md.append("| **LEARNED_WEIGHTS_PHASE_E v2 (拡張版)** | **v01-40 / 7,650** | **16** | **0.659** | **0.862** | **+8.1** |")
    md.append("")
    md.append(
        "改善幅は 旧 PhaseAware (`weight_mode=\"learned\"`) overall=0.578 を基準とした絶対差分 (pt)。"
    )
    md.append("")
    return md


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path,
        default=_ROOT / "data/training/match_features_phase_e_v01-40.csv",
    )
    parser.add_argument(
        "--review-tsv", type=Path,
        default=_ROOT / "data/verify/phase_e_recognition_review_v01-40.tsv",
    )
    parser.add_argument(
        "--learn-json", type=Path,
        default=_ROOT / "data/verify/learned_weights_phase_e_phase_aware_v01-40.json",
    )
    parser.add_argument(
        "--vif-json", type=Path,
        default=_ROOT / "data/verify/multicollinearity_phase_e.json",
    )
    parser.add_argument(
        "--eval-json", type=Path,
        default=_ROOT / "data/verify/indicator_evaluation_phase_e.json",
    )
    parser.add_argument(
        "--boundary-root", type=Path,
        default=_ROOT / "data/verify/match_boundaries_v5",
    )
    parser.add_argument(
        "--winners-root", type=Path,
        default=_ROOT / "data/verify",
    )
    parser.add_argument(
        "--out", type=Path,
        default=_ROOT / "data/verify/phase_e_dashboard.md",
    )
    args = parser.parse_args()

    rows = load_csv(args.csv) if args.csv.exists() else []
    review_rows = (
        load_tsv(args.review_tsv) if args.review_tsv.exists() else []
    )
    learn_json = (
        load_json(args.learn_json) if args.learn_json.exists() else {}
    )
    vif_json = load_json(args.vif_json) if args.vif_json.exists() else None
    eval_json = (
        load_json(args.eval_json) if args.eval_json.exists() else None
    )

    md: list[str] = []
    md.append("# Phase E 評価ダッシュボード")
    md.append("")
    md.append(
        f"- 生成元 CSV: `{to_windows_path(args.csv)}`"
    )
    md.append("- 評価観点: 教師データ品質 / 認識精度 / モデル評価 / 特徴量 / "
              "試合境界 / 重みセット比較")
    md.append("")
    md.append("## 概要")
    md.append("")
    md.append(
        "新方針 pipeline (state machine + 物理推論) で全動画から STABLE "
        "確定盤面のみ抽出し、PhaseAware 重みを再学習した結果のスナップショット。"
    )
    md.append("")
    md.append("---")
    md.append("")
    md.extend(section_data_quality(rows))
    md.extend(section_recognition(review_rows))
    md.extend(section_model_eval(learn_json, eval_json))
    md.extend(section_features(eval_json, vif_json))
    md.extend(section_match_detection(args.boundary_root, args.winners_root))
    md.extend(section_weight_compare())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md), encoding="utf-8")
    print(f"[saved] {to_windows_path(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

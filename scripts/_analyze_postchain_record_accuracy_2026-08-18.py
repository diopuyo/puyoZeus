"""_diag_postchain_record_accuracy_2026-08-18.py の出力を分析する (2026-08-18)。

核心の質問: 「連鎖後 (chain) / おじゃま着弾後 (ojama) に記録される盤面は、
通常時 (normal) と同等に正しいか」。

自動判定 (ground truth 不要の自己無矛盾性チェック):
  同一 side の連続する2つの「記録された盤面」(=本番 npz に載る値) を比較し、
  同一セルが 非空値→別の非空値 に変化していたら物理的にありえない
  (色が消去なしに別の色へ変わることはない、ぷよは空セルにしか新規着地
  しない)。これは記録Nか記録N+1のどちらかが誤りという確定的な signature
  であり、ground truth なしで「記録の誤り」を検出できる。

このスクリプトは:
  1. 全 flip (非空→別の非空) を列挙し、隣接する記録の kind (chain/ojama/normal)
     で分類する。
  2. kind 別の「記録が flip に関与した比率」(chain-adjacent 記録が通常より
     誤りやすいか) を集計する。
  3. 各 flip について、frames.jsonl から該当セルの値の時系列を追い、
     誤り値が何秒間残ったか (=残存時間) を測る。
  4. 目視レビュー用に、疑わしい flip の上位 N件 + 正常な代表例を抽出し
     JSON で書き出す (crop スクリプトの入力)。

コードは変更しない (診断専用)。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

_TAG = "2026-08-18"
_LOG_DIR = Path("logs")

COLOR_NAMES = {0: "空", 1: "赤", 2: "青", 3: "緑", 4: "黄", 5: "紫", 9: "お邪魔", 10: "UNK"}


def _load_moves() -> list[dict]:
    path = _LOG_DIR / f"_diag_postchain_record_accuracy_{_TAG}_ALL_moves.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _group_by_side(moves: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m in moves:
        groups[(m["video"], m["side"])].append(m)
    for key in groups:
        groups[key].sort(key=lambda m: m["t_sec"])
    return groups


def _find_flips(groups: dict[tuple[str, str], list[dict]]) -> list[dict]:
    """隣接記録間の非空→別非空 flip を全て列挙する。"""
    flips: list[dict] = []
    for (video, side), seq in groups.items():
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            ga, gb = a["grid"], b["grid"]
            for r in range(13):
                for c in range(6):
                    va, vb = ga[r][c], gb[r][c]
                    if va != 0 and vb != 0 and va != vb:
                        flips.append({
                            "video": video, "side": side, "r": r, "c": c,
                            "val_a": va, "val_b": vb,
                            "t_a": a["t_sec"], "t_b": b["t_sec"],
                            "frame_a": a["frame_idx"], "frame_b": b["frame_idx"],
                            "kind_a": a["kind"], "kind_b": b["kind"],
                            "chain_end_a": a.get("chain_end_t_sec"),
                            "ojama_end_a": a.get("ojama_end_t_sec"),
                        })
    return flips


def _residual_duration(flip: dict) -> "float | None":
    """該当セルの誤り値が frames.jsonl 上で何秒残ったかを測る。"""
    tag_candidates = [
        p.stem.replace(f"_diag_postchain_record_accuracy_{_TAG}_", "").replace("_frames", "")
        for p in _LOG_DIR.glob(f"_diag_postchain_record_accuracy_{_TAG}_*_frames.jsonl")
    ]
    video = flip["video"]
    if video not in tag_candidates:
        return None
    path = _LOG_DIR / f"_diag_postchain_record_accuracy_{_TAG}_{video}_frames.jsonl"
    r, c, side = flip["r"], flip["c"], flip["side"]
    val_a = flip["val_a"]
    first_t = None
    last_wrong_t = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["side"] != side or rec["grid"] is None:
                continue
            if rec["t_sec"] < flip["t_a"] - 0.001:
                continue
            if rec["t_sec"] > flip["t_b"] + 0.001:
                break
            v = rec["grid"][r][c]
            if v == val_a:
                if first_t is None:
                    first_t = rec["t_sec"]
                last_wrong_t = rec["t_sec"]
    if first_t is None or last_wrong_t is None:
        return None
    return round(last_wrong_t - first_t, 3)


def summarize(moves: list[dict], flips: list[dict]) -> dict:
    total_by_kind: dict[str, int] = defaultdict(int)
    for m in moves:
        total_by_kind[m["kind"]] += 1

    involved_by_kind: dict[str, set] = defaultdict(set)
    for fl in flips:
        key_a = (fl["video"], fl["side"], fl["t_a"])
        key_b = (fl["video"], fl["side"], fl["t_b"])
        involved_by_kind[fl["kind_a"]].add(key_a)
        involved_by_kind[fl["kind_b"]].add(key_b)

    rate_by_kind = {
        k: {
            "total_records": total_by_kind[k],
            "records_involved_in_flip": len(involved_by_kind[k]),
            "rate_pct": (
                round(100.0 * len(involved_by_kind[k]) / total_by_kind[k], 2)
                if total_by_kind[k] else None
            ),
        }
        for k in ("normal", "chain", "ojama")
    }
    return {
        "total_moves": len(moves),
        "total_flips": len(flips),
        "by_kind": rate_by_kind,
    }


def main() -> None:
    moves = _load_moves()
    groups = _group_by_side(moves)
    flips = _find_flips(groups)
    for fl in flips:
        fl["residual_sec"] = _residual_duration(fl)

    summary = summarize(moves, flips)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_path = _LOG_DIR / f"_analyze_postchain_record_accuracy_{_TAG}_flips.json"
    out_path.write_text(json.dumps(flips, ensure_ascii=False, indent=2), encoding="utf-8")
    summ_path = _LOG_DIR / f"_analyze_postchain_record_accuracy_{_TAG}_summary.json"
    summ_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out_path}")
    print(f"-> {summ_path}")

    # kind 別の残存時間分布 (中央値・最大)
    import statistics
    for kind_pair_name, pred in (
        ("chain側が疑わしいflip", lambda f: f["kind_a"] == "chain" and f["kind_b"] == "normal"),
        ("ojama側が疑わしいflip", lambda f: f["kind_a"] == "ojama" and f["kind_b"] == "normal"),
        ("normal同士のflip(基準線)", lambda f: f["kind_a"] == "normal" and f["kind_b"] == "normal"),
    ):
        durs = [f["residual_sec"] for f in flips if pred(f) and f["residual_sec"] is not None]
        if durs:
            print(f"{kind_pair_name}: n={len(durs)} 中央値={statistics.median(durs):.3f}s "
                  f"最大={max(durs):.3f}s")
        else:
            print(f"{kind_pair_name}: n=0")


if __name__ == "__main__":
    main()

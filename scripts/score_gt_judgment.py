"""人手判定 JSON を採点して、動画ごとのセル正解率の見積もりを出す。

video_38 で確定した方式をそのまま使う (2026-09-19):

    本物の誤り率 = (本物と判定された数) / (判定した数)
    推定される誤りセル数 = 代理指標が挙げた食い違い数 x 本物の誤り率
    セル正解率 = 1 - 推定される誤りセル数 / 代理指標の基準セル数

「判定できない」の扱いで3通りの見積もりを出す:

  - 最も厳しい: 判定できない = すべて本物の誤り
  - 判定できた分で外挿: 判定できない = 母数から除く
  - 最も甘い: 判定できない = すべて代理指標の誤検出

代理指標 (CNN == HSV かつ != COLOR_UNKNOWN) は落下中のぷよ・テロップ文字を
盤面のぷよと数えるため、実際より厳しく出る。抽出も食い違い優先なので、
ここでの誤り率は動画全体より高く出る。

使い方:
    py -3 scripts/score_gt_judgment.py --self-check
    py -3 scripts/score_gt_judgment.py --video 39 --judged <judged.json> [--out <receipt.json>]
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]

# 判定の値
VERDICT_RECORDED = "recorded"  # 記録が正しい = 代理指標の誤検出
VERDICT_SCREEN = "screen"  # 画面が正しい = 本物の誤り
VERDICT_UNKNOWN = "unknown"  # 判定できない

# video_38 の既知の結果。新しい台はまずこれを再現できることを示す。
SELF_CHECK_RECEIPT = ROOT / "data" / "verify" / "g3_cell_accuracy_2026-09-19" / "RECEIPT.json"
SELF_CHECK_JUDGED = ROOT / "data" / "verify" / "g3_cell_accuracy_2026-09-19" / "judged_by_user.json"
SELF_CHECK_TOLERANCE = 1e-4


def _load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    """判定 JSON を読む。{"judged": [...]} でも素の配列でも受ける。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("judged", "cells", "rows"):
            if key in data:
                return list(data[key])
        raise ValueError(f"{path}: judged 配列が見つからない (keys={list(data)})")
    return list(data)


def _cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["side"], row["frame"], row["row"], row["col"])


def tally(rows: list[dict[str, Any]]) -> dict[str, int]:
    """判定の内訳を数える。既知以外の値は例外にする (黙って落とさない)。"""
    counts = {VERDICT_RECORDED: 0, VERDICT_SCREEN: 0, VERDICT_UNKNOWN: 0}
    for row in rows:
        verdict = row.get("verdict")
        if verdict not in counts:
            raise ValueError(f"未知の判定値 {verdict!r}: {row}")
        counts[verdict] += 1
    return counts


def estimate_accuracy(counts: dict[str, int], n_basis: int, n_mismatch: int) -> dict[str, float]:
    """3通りの見積もりを返す。単位はパーセント。"""
    n_judged = sum(counts.values())
    n_decided = counts[VERDICT_RECORDED] + counts[VERDICT_SCREEN]
    if n_judged == 0 or n_basis == 0:
        raise ValueError("判定数または基準セル数が0では見積もれない")

    def acc(real_rate: float) -> float:
        return round(100.0 * (1.0 - (n_mismatch * real_rate) / n_basis), 4)

    strict = (counts[VERDICT_SCREEN] + counts[VERDICT_UNKNOWN]) / n_judged
    lenient = counts[VERDICT_SCREEN] / n_judged
    decided = (counts[VERDICT_SCREEN] / n_decided) if n_decided else 0.0
    return {
        "最も厳しい(判定できない=全部本物)": acc(strict),
        "判定できた分で外挿": acc(decided),
        "最も甘い(判定できない=全部誤検出)": acc(lenient),
    }


def verify_coverage(rows: list[dict[str, Any]], picks_path: pathlib.Path) -> dict[str, Any]:
    """判定 JSON が抽出したセルを過不足なく覆っているかを確かめる。"""
    picks = json.loads(picks_path.read_text(encoding="utf-8"))
    expected: set[tuple[Any, ...]] = set()
    for board in picks["picks"]:
        for row, col, _screen, _recorded in board["cells"]:
            expected.add((board["side"], board["frame"], row, col))
    got = {_cell_key(r) for r in rows}
    return {
        "抽出したセル": len(expected),
        "判定したセル": len(rows),
        "重複": len(rows) - len(got),
        "判定漏れ": sorted(expected - got),
        "抽出外": sorted(got - expected),
    }


def score_video(video: str, judged_path: pathlib.Path) -> dict[str, Any]:
    kit_dir = ROOT / "logs" / "diag_gt" / f"kit_video_{video}"
    picks_path = kit_dir / "picks.json"
    picks = json.loads(picks_path.read_text(encoding="utf-8"))
    detail = pathlib.Path(picks["source_detail"])
    receipt_path = (detail if detail.is_absolute() else ROOT / detail).parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    dc = receipt["denominators_and_counts"]

    rows = _load_rows(judged_path)
    coverage = verify_coverage(rows, picks_path)
    counts = tally(rows)
    n_basis = int(dc["npz_basis_cellframes_total"])
    n_mismatch = int(dc["npz_mismatch_cellframes_total"])
    n_recorded_boards = sum(int(v) for v in dc["steps_recorded_to_npz"].values())

    by_reason: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_reason.setdefault(
            row.get("reason", "不明"),
            {"見た": 0, VERDICT_RECORDED: 0, VERDICT_SCREEN: 0, VERDICT_UNKNOWN: 0},
        )
        bucket["見た"] += 1
        bucket[row["verdict"]] += 1

    n_judged = sum(counts.values())
    n_decided = counts[VERDICT_RECORDED] + counts[VERDICT_SCREEN]
    return {
        "schema": "g3_cell_accuracy/v1",
        "動画": f"video_{video}",
        "受領票": str(receipt_path.relative_to(ROOT)).replace("\\", "/"),
        "抽出": str(picks_path.relative_to(ROOT)).replace("\\", "/"),
        "被覆確認": coverage,
        "母数": {
            "記録された盤面": n_recorded_boards,
            "うち食い違いのあった盤面": int(picks["n_boards_with_mismatch"]),
            "代理指標の基準セル": n_basis,
            "代理指標が挙げた食い違い": n_mismatch,
            "user が判定したセル": n_judged,
        },
        "判定の内訳": {
            "記録が正しい(代理指標の誤検出)": counts[VERDICT_RECORDED],
            "画面が正しい(本物の誤り)": counts[VERDICT_SCREEN],
            "判定できない": counts[VERDICT_UNKNOWN],
        },
        "選び方別": by_reason,
        "本物の誤りの割合": {
            "値": round(counts[VERDICT_SCREEN] / n_decided, 4) if n_decided else None,
            "分子": counts[VERDICT_SCREEN],
            "分母": n_decided,
        },
        "推定セル正解率": estimate_accuracy(counts, n_basis, n_mismatch),
        "合格線": 99.5,
    }


def self_check() -> bool:
    """video_38 の既知の結果を再現できるかを先に示す。"""
    if not SELF_CHECK_RECEIPT.exists():
        print(f"NG: 自己確認の元データが無い ({SELF_CHECK_RECEIPT})")
        return False
    known = json.loads(SELF_CHECK_RECEIPT.read_text(encoding="utf-8"))
    rows = _load_rows(SELF_CHECK_JUDGED)
    counts = tally(rows)
    got = estimate_accuracy(
        counts,
        int(known["母数"]["代理指標の基準セル"]),
        int(known["母数"]["代理指標が挙げた食い違い"]),
    )
    ok = True
    print("自己確認 (video_38 の既知の結果を再現できるか):")
    print(f"  判定の内訳: {counts}  (既知の母数 {known['母数']['user が判定したセル']} セル)")
    for key, want in known["推定セル正解率"].items():
        have = got[key]
        hit = abs(have - float(want)) <= SELF_CHECK_TOLERANCE
        ok &= hit
        print(f"  {'OK ' if hit else 'NG '} {key}: 既知 {want} / 再現 {have}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", help="動画名 (38 / 39 / c74 / c80 / c138)")
    ap.add_argument("--judged", type=pathlib.Path, help="人手判定 JSON")
    ap.add_argument("--out", type=pathlib.Path, help="受領票の保存先 (既存があれば書かない)")
    ap.add_argument("--self-check", action="store_true", help="video_38 の再現だけ行う")
    args = ap.parse_args()

    if args.self_check:
        return 0 if self_check() else 1

    if not args.video or not args.judged:
        ap.error("--video と --judged の両方が要る")

    if not self_check():
        print("NG: 既知の結果を再現できないので採点しない")
        return 1

    receipt = score_video(args.video, args.judged)
    print()
    print(json.dumps(receipt, ensure_ascii=False, indent=2))

    if args.out:
        if args.out.exists():
            print(f"\nNG: {args.out} が既にある。上書きしない")
            return 1
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n保存: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

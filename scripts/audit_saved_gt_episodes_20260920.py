"""保存済みの食い違い区間と実NPZ明細を、同一run内だけで突き合わせる。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

VIDEOS = ("video_38", "video_39", "video_c74", "video_c80", "video_c138")
UPPER_ROWS = range(1, 5)
CHAIN_STATES = frozenset(("CHAIN", "GRAVITY_SETTLE"))
TARGET_WRITERS = ("_apply_gravity_column", "clear_floating_above_gap")
KNOWN_CELLS = {20917: (3, (8, 9, 10, 11)), 32245: (2, (1, 2, 4, 5))}


def rows(path: Path) -> Iterator[dict[str, Any]]:
    """大きい原票を一行ずつ読み、欠落や破損は例外にする。"""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def state_group(states: dict[str, int]) -> str:
    """区間の状態集合を分類する。記録瞬間の状態と混同しない。"""
    active = set(states)
    if active <= CHAIN_STATES:
        return "連鎖状態のみ"
    if active & CHAIN_STATES:
        return "連鎖状態を含む混在区間"
    return "連鎖状態なし・テロップ判定不能"


def contiguous_cells(cells: list[list[int]]) -> set[tuple[int, int]]:
    """実記録で、同一列の隣接二セル以上が空になる食い違いを拾う。"""
    empty = {(r, c) for r, c, observed, written in cells if written == 0 and observed != 0}
    return {(r, c) for r, c in empty if (r-1, c) in empty or (r+1, c) in empty}


def read_details(root: Path) -> tuple[dict[tuple, dict], Counter]:
    """実NPZの食い違いセルをキーにして重複と行数を検査する。"""
    details: dict[tuple, dict] = {}
    counts: Counter = Counter()
    for row in rows(root / "npz_detail.jsonl"):
        assert row["n_mismatch"] == len(row["cells"])
        vertical = contiguous_cells(row["cells"])
        counts["mismatch_boards"] += 1
        for r, c, observed, written in row["cells"]:
            key = (row["side"], r, c, observed, written, row["frame"])
            assert key not in details, key
            details[key] = dict(state=row["state"], vertical=(r, c) in vertical, matches=[])
            counts["npz_mismatch_cells"] += 1
            counts["upper_npz_mismatch_cells"] += r in UPPER_ROWS
            counts["vertical_npz_mismatch_cells"] += (r, c) in vertical
    return details, counts


def attach_episode(ep: dict, index: dict, counters: Counter, groups: Counter) -> None:
    """補正前区間と実記録の値・区間を同一run内で照合する。"""
    counters["episodes"] += 1
    counters["episode_cellframes"] += ep["n"]
    counters["recorded_episode_cellframes_before_fix"] += ep["n_recorded"]
    counters["episodes_with_record"] += ep["n_recorded"] > 0
    assert sum(ep["states"].values()) == ep["n"]
    assert sum(ep["writers"].values()) == ep["n"]
    writers = tuple(sorted(ep["writers"]))
    group = (" | ".join(writers), state_group(ep["states"]), str(ep["n_recorded"] > 0))
    groups[group] += 1
    key = (ep["side"], ep["row"], ep["col"], ep["observed"], ep["written"])
    for frame, detail in index.get(key, ()):
        if ep["f0"] <= frame <= ep["f1"]:
            detail["matches"].append(dict(writers=writers, states=ep["states"],
                f0=ep["f0"], f1=ep["f1"], n_recorded=ep["n_recorded"]))


def detail_summary(details: dict, counts: Counter) -> tuple[list, list]:
    """単一書き手だけ確定し、複数書き手の時点別帰属は未測定に残す。"""
    cross: Counter = Counter()
    known = []
    for key, item in details.items():
        matches = item["matches"]
        assert len(matches) <= 1, (key, matches)
        writers = matches[0]["writers"] if matches else ()
        writer = writers[0] if len(writers) == 1 else "混在・時点別書き手未測定" if writers else "対応区間なし"
        upper = key[1] in UPPER_ROWS
        group = state_group(matches[0]["states"]) if matches else "対応区間なし"
        cross[(writer, item["state"], group, str(upper), str(item["vertical"]))] += 1
        counts["npz_cells_unique_writer"] += len(writers) == 1
        for target in TARGET_WRITERS:
            counts[target + "_certain"] += len(writers) == 1 and target in writer
            counts[target + "_possible"] += any(target in w for w in writers)
        col, rr = KNOWN_CELLS.get(key[-1], (-1, ()))
        if key[0] == "2P" and key[2] == col and key[1] in rr:
            known.append(dict(key=list(key), **item))
    return [dict(writer=k[0], recorded_state=k[1], episode_state_group=k[2],
                 upper_rows=k[3], vertical=k[4], count=v) for k, v in sorted(cross.items())], known


def audit(root: Path) -> dict[str, Any]:
    """一動画を集計し、元受領票の母数と一致させる。"""
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    details, counts = read_details(root) if (root / "npz_detail.jsonl").exists() else ({}, Counter())
    counts["detail_cells_available"] = len(details)
    index: dict = defaultdict(list)
    for key, value in details.items():
        index[key[:-1]].append((key[-1], value))
    groups: Counter = Counter()
    for ep in rows(root / "episodes.jsonl"):
        attach_episode(ep, index, counts, groups)
    assert counts["episodes"] == receipt["episodes_written"] == receipt["episodes_total"]
    expected = receipt["denominators_and_counts"]
    if (root / "npz_detail.jsonl").exists():
        assert counts["npz_mismatch_cells"] == expected["npz_mismatch_cellframes_total"]
    else:
        counts["npz_mismatch_cells"] = expected["npz_mismatch_cellframes_total"]
        counts["upper_npz_mismatch_cells"] = sum(int(v) for k, v in
            receipt["proxy_rates"]["npz_by_row"].items() if int(k) in UPPER_ROWS)
    cross, known = detail_summary(details, counts)
    counts["npz_basis_cells"] = expected["npz_basis_cellframes_total"]
    counts["basis_cellframes"] = expected["basis_cellframes_total"]
    return dict(source=str(root), counts=dict(counts), npz_cross=cross, known_cells=known,
        episode_cross=[dict(writers=k[0], state_group=k[1], recorded=k[2], episodes=v)
                       for k, v in sorted(groups.items())],
        source_sha256={name: file_sha(root/name)
                       for name in ("receipt.json", "episodes.jsonl", "npz_detail.jsonl")
                       if (root/name).exists()})


def file_sha(path: Path) -> str:
    """原票を読んだ後に必ずファイルを閉じる。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """全長5動画と人手判定用video38を分け、排他出力する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full = {v: audit(Path("logs/diag_gt/four_videos") / v) for v in VIDEOS[1:]}
    full["video_38"] = audit(Path("logs/diag_ab/five_videos_recordfix/on_video_38"))
    human38 = audit(Path("logs/diag_gt/video_38"))
    # video38全長は記録明細が存在せず、先頭500秒の別runから補完しない。
    assert sum(v["counts"]["upper_npz_mismatch_cells"] for v in full.values()) == 3659
    assert len(full["video_c80"]["known_cells"]) == 8
    result = dict(diagnostic_not_quality_pass=True, full=full, human_video38=human38,
        limitations=["食い違いは人手真値でない", "states/writersは区間内の周辺集計で同時刻の組を復元できない",
                     "全消しテロップはstates/writersだけでは確定できない", "n_recordedは補正後の誤り数ではない"])
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({v: r["counts"] for v, r in full.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()

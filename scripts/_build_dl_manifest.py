"""plC(idx7-54) + tierA(30本) を video_c34.. に割当てて DL マニフェストを生成する。

_plC_full.tsv は yt-dlp --print の出力で、区切りがリテラル 2 文字 '\\t'(バックスラッシュ+t)。
pending_tierA_missing.tsv は実タブ区切りの正規 TSV。両者を吸収して重複除去する。
"""
from __future__ import annotations

import csv
from pathlib import Path

LIT_TAB = "\\t"  # ファイル中のリテラル区切り(2文字)
ID_MAX_LEN = 15  # YouTube ID は 11 桁想定。異常に長い=分割失敗として除外
START_IDX = 34   # 既存 c1-c33 の続き


def _load_plc(path: Path) -> list[tuple[str, str]]:
    """plC の全エントリ (id, title) を返す。区切りはリテラル '\\t'。"""
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(LIT_TAB)
        title = parts[2] if len(parts) > 2 else ""
        rows.append((parts[0].strip(), title))
    return rows


def _load_tier_a(path: Path) -> list[tuple[str, str]]:
    """tierA の (id, title) を返す。実タブ TSV。"""
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for p in reader:
            if len(p) >= 2:
                rows.append((p[1].strip(), p[3] if len(p) > 3 else ""))
    return rows


def main() -> None:
    base = Path("data")
    plc = _load_plc(base / "_plC_full.tsv")[6:]  # idx7-54 = 未取得48本
    tier_a = _load_tier_a(base / "pending_tierA_missing.tsv")
    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for vid, title in plc + tier_a:
        if not vid or len(vid) > ID_MAX_LEN or vid in seen:
            continue
        seen.add(vid)
        uniq.append((vid, title))
    out = base / "_dl_expand.tsv"
    with out.open("w", encoding="utf-8") as f:
        for i, (vid, title) in enumerate(uniq, start=START_IDX):
            f.write(f"video_c{i}\t{vid}\t{title}\n")
    print(f"plC未取得 {len(plc)} + tierA {len(tier_a)} -> 重複除去後 {len(uniq)} 本")
    print(f"範囲: video_c{START_IDX} .. video_c{START_IDX + len(uniq) - 1}")


if __name__ == "__main__":
    main()

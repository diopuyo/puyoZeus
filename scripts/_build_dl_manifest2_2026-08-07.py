"""Phase L step2 (DL拡大) 用マニフェスト生成 (2026-08-07)。

data/_new_pl_candidates_2026-08-07.tsv (8プレイリスト分、リテラル '\\t' 区切り、
_plC_full.tsv と同形式) から、S級・A級を補強する目的で選定ルールに沿って
51本目標で候補を選び、data/_dl_expand2_2026-08-07.tsv (name/id/title、実タブ
区切り、_dl_expand.tsv と同形式) を生成する。name は video_c96 から連番
(c95 が既存最終)。

選定ルール (2026-08-07 user確定):
1. r2_sleague 全部 (目標14本)
2. r1_sleague 全部 (目標6本)
3. r3_alevel を playlist_index 昇順で15本 (除外分は次indexで補充)
4. r2_alevel を playlist_index 昇順で15本 (除外分は次indexで補充)
5. 追加1本 (plD idx7 lWmBXteEbeg、マスター扱い)

共通フィルタ: duration >= MIN_DURATION_SEC、既存video_id除外
(data/video_tier_index_2026-08-07.tsv 全体と突合)、選定内ID重複除去。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_TSV = PROJECT_ROOT / "data" / "_new_pl_candidates_2026-08-07.tsv"
TIER_INDEX_TSV = PROJECT_ROOT / "data" / "video_tier_index_2026-08-07.tsv"
OUT_TSV = PROJECT_ROOT / "data" / "_dl_expand2_2026-08-07.tsv"

LIT_TAB = "\\t"  # 候補ファイルの区切り: リテラル2文字 (バックスラッシュ+t)
START_C_INDEX = 96  # 既存 c1-c95 の続き
MIN_DURATION_SEC = 300.0  # 共通フィルタ: 5分未満はPV等とみなし除外

# 「全部取り」枠 (プレイリスト全量が対象、除外分の補充元が無い)
FULL_TAKE_TAGS: tuple[tuple[str, int], ...] = (
    ("r2_sleague", 14),
    ("r1_sleague", 6),
)
# 「上位N本+同枠内で補充」枠 (playlist_index 昇順に歩いて目標数まで拾う)
WALK_TAKE_TAGS: tuple[tuple[str, int], ...] = (
    ("r3_alevel", 15),
    ("r2_alevel", 15),
)
# 追加1本 (data/_pending_dl_candidates_2026-08-07.tsv plD idx7 由来)
EXTRA_ONE = ("lWmBXteEbeg", "【マスター進出決定トーナメント・Bブロック】3 vsにゃんきち  "
             "30先(解説：ヨダソウマ)【第2回新おいうリーグ】#ぷよぷよ #ぷよぷよeスポーツ")


@dataclass
class Candidate:
    """候補動画1本分。"""

    pl_tag: str
    playlist_index: int
    video_id: str
    duration_sec: float | None
    title: str


def load_candidates() -> list[Candidate]:
    """data/_new_pl_candidates_2026-08-07.tsv を読み込む (リテラル '\\t' 区切り)。"""
    rows: list[Candidate] = []
    for line in CANDIDATES_TSV.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(LIT_TAB)
        if len(parts) != 5:
            continue
        pl_tag, idx_s, vid, dur_s, title = parts
        duration = None if dur_s == "NA" else float(dur_s)
        rows.append(Candidate(pl_tag, int(idx_s), vid, duration, title))
    return rows


def load_existing_ids() -> set[str]:
    """data/video_tier_index_2026-08-07.tsv の video_id 全体を返す。"""
    ids: set[str] = set()
    with TIER_INDEX_TSV.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            vid = row.get("video_id", "")
            if vid:
                ids.add(vid)
    return ids


def _passes_filter(cand: Candidate, seen_ids: set[str]) -> bool:
    """共通フィルタ (duration >= MIN_DURATION_SEC、既存/選定済ID除外)。"""
    if cand.duration_sec is None or cand.duration_sec < MIN_DURATION_SEC:
        return False
    return cand.video_id not in seen_ids


def select_full_take(
    rows: list[Candidate], tag: str, target: int, seen_ids: set[str],
) -> tuple[list[Candidate], int]:
    """タグ全量を playlist_index 昇順にフィルタし、選定数と不足数を返す。"""
    pool = sorted((r for r in rows if r.pl_tag == tag), key=lambda r: r.playlist_index)
    picked: list[Candidate] = []
    for cand in pool:
        if _passes_filter(cand, seen_ids):
            picked.append(cand)
            seen_ids.add(cand.video_id)
    shortfall = max(0, target - len(picked))
    return picked, shortfall


def select_walk_take(
    rows: list[Candidate], tag: str, target: int, seen_ids: set[str],
) -> tuple[list[Candidate], int]:
    """タグを playlist_index 昇順に歩き、target本に達するまで拾う (自動補充)。"""
    pool = sorted((r for r in rows if r.pl_tag == tag), key=lambda r: r.playlist_index)
    picked: list[Candidate] = []
    for cand in pool:
        if len(picked) >= target:
            break
        if _passes_filter(cand, seen_ids):
            picked.append(cand)
            seen_ids.add(cand.video_id)
    shortfall = max(0, target - len(picked))
    return picked, shortfall


def select_extra_one(seen_ids: set[str]) -> tuple[Candidate | None, bool]:
    """追加1本 (lWmBXteEbeg) を評価する。既存重複なら None (0本追加) を返す。"""
    vid, title = EXTRA_ONE
    is_duplicate = vid in seen_ids
    if is_duplicate:
        return None, True
    seen_ids.add(vid)
    return Candidate("extra_master", 0, vid, None, title), False


def write_manifest(entries: list[Candidate]) -> None:
    """data/_dl_expand2_2026-08-07.tsv (name/id/title) を書き出す。"""
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", encoding="utf-8", newline="\n") as f:
        for i, cand in enumerate(entries, start=START_C_INDEX):
            f.write(f"video_c{i}\t{cand.video_id}\t{cand.title}\n")


def main() -> int:
    rows = load_candidates()
    seen_ids = load_existing_ids()
    report_lines: list[str] = []

    all_picked: list[Candidate] = []
    for tag, target in FULL_TAKE_TAGS:
        picked, shortfall = select_full_take(rows, tag, target, seen_ids)
        all_picked.extend(picked)
        report_lines.append(
            f"[{tag}] 目標{target} 選定{len(picked)} 不足{shortfall}",
        )
    for tag, target in WALK_TAKE_TAGS:
        picked, shortfall = select_walk_take(rows, tag, target, seen_ids)
        all_picked.extend(picked)
        report_lines.append(
            f"[{tag}] 目標{target} 選定{len(picked)} 不足{shortfall}",
        )
    extra, was_dup = select_extra_one(seen_ids)
    if extra is not None:
        all_picked.append(extra)
    report_lines.append(
        f"[extra_master] lWmBXteEbeg 既存重複={was_dup} 追加本数="
        f"{0 if was_dup else 1}",
    )

    write_manifest(all_picked)
    for line in report_lines:
        print(line)
    print(f"[manifest] 合計選定={len(all_picked)}本 -> {OUT_TSV}")
    print(f"[manifest] 範囲: video_c{START_C_INDEX} .. "
          f"video_c{START_C_INDEX + len(all_picked) - 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

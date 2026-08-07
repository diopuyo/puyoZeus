"""Phase L 全動画 regen 準備: 統合ティアインデックス生成スクリプト。

以下の各ソースを統合し、動画単位の tier (S級/マスター/チャレンジャー/A級/
その他大会系/UNCONFIRMED) と on_disk 状態を1枚の TSV にまとめる。

- data/phase_e_dl_index.tsv    : v29-v95 (マスター級以上、第2回新おいうリーグ)
- data/pending_tierA_missing.tsv: s1-s3 のタイトル引き (plA=S級決定戦)
- data/pl_new_missing.tsv       : c4-c33 (チャレンジャー、行順に逐次割当)
- data/_dl_expand.tsv           : c34-c95 (video_name/id/title 明示対応)
- data/_dl_expand2_2026-08-07.tsv: c96-c144 (Phase L step2、S級/A級補強、
  video_name/id/title 明示対応、_dl_expand.tsv と同形式)
- scripts/_dl_tier_avc1.sh      : s1-s3, c1-c3 の video_id ハードコード元
  (c1-c3 はタイトル記録がどのファイルにも存在しないため、背景調査で確認済み
  のチャレンジャー枠として直接指定する)

出力:
- data/video_tier_index_2026-08-07.tsv         (全動画の台帳、重複含む)
- data/video_tier_index_dup_report_2026-08-07.txt (ID重複検出レポート)
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
PHASE_E_INDEX_TSV = PROJECT_ROOT / "data" / "phase_e_dl_index.tsv"
PENDING_TIERA_TSV = PROJECT_ROOT / "data" / "pending_tierA_missing.tsv"
PL_NEW_MISSING_TSV = PROJECT_ROOT / "data" / "pl_new_missing.tsv"
DL_EXPAND_TSV = PROJECT_ROOT / "data" / "_dl_expand.tsv"
DL_EXPAND2_TSV = PROJECT_ROOT / "data" / "_dl_expand2_2026-08-07.tsv"

OUT_TSV = PROJECT_ROOT / "data" / "video_tier_index_2026-08-07.tsv"
DUP_REPORT_TXT = PROJECT_ROOT / "data" / "video_tier_index_dup_report_2026-08-07.txt"

VIDEO_NAME_ZERO_PAD = 2  # video_29 等 v系ファイル名の桁数 (video_idx を0埋め)
PL_NEW_MISSING_FIRST_C_INDEX = 4  # data/pl_new_missing.tsv 先頭行 -> video_c4

# tier キーワード分類の優先順位 (順序に意味がある: マスター動画の解説者名
# 「〇〇プロ」が「プロ」キーワードに誤爆するため、具体ティアを
# 「その他大会系」より必ず先に判定すること)
TIER_KEYWORD_ORDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("S級", ("S級",)),
    ("マスター", ("マスター",)),
    ("チャレンジャー", ("チャレンジャー",)),
    ("A級", ("A級",)),
    ("その他大会系", ("GP", "プロ", "オフライン", "決勝", "大会")),
)
TIER_UNCONFIRMED = "UNCONFIRMED"

# CLAUDE.md「使用可: A級/マスター/チャレンジャー/S級の動画のみ」に対応する
# 学習可能tier白リスト (2026-08-07 user指示で regen 既定に適用)。
# 「その他大会系」はキーワード「決勝/大会」で拾えるが、実体はB/C/D級混在の
# 複合トーナメント (例: B2・C1級複合決勝トーナメント) であり、上級者対戦
# スコープの非該当。ここに含めない。
PHASE_L_TIER_WHITELIST: tuple[str, ...] = ("S級", "マスター", "チャレンジャー", "A級")

# c1-c3: video_id は scripts/_dl_tier_avc1.sh にハードコードされるが、
# 対応タイトルはどのTSVにも存在しない (全リポジトリ検索で確認済み)。
# 背景調査でチャレンジャー枠と確認済みのため、tier のみ直接指定する。
C1_C3_IDS: dict[str, str] = {
    "video_c1": "kETyIUk_Vb8",
    "video_c2": "Lw6Z8Nzguo4",
    "video_c3": "fIIDFPCD2w0",
}
# s1-s3: video_id も同スクリプトにハードコード。タイトルは
# pending_tierA_missing.tsv の plA 行 (S級決定戦) から ID 一致で引けるため
# ハードコードしない (classify_tier に委ねる)。
S1_S3_IDS: dict[str, str] = {
    "video_s1": "04Lb9BZCpP0",
    "video_s2": "UpnGj22itdA",
    "video_s3": "GDfVPnyrfwU",
}

# データ源の優先順位 (重複 video_id 時にどちらを採用するか)。値が小さいほど
# 優先。根拠: c系再DLは全て avc1(H.264) 強制指定で、旧 v系
# (phase_e_dl_index.tsv) はコーデック不明 (=再DLの動機そのもの、
# scripts/_dl_tier_avc1.sh 冒頭コメント参照)。c系内では明示 id/title 対応の
# _dl_expand.tsv / _dl_expand2_2026-08-07.tsv を、行番号からの逐次割当である
# pl_new_missing.tsv より優先する (前者は取り違えリスクが小さい)。
# _dl_expand2 は _dl_expand の後追い (Phase L step2) なので、既存 c34-c95 と
# 同一IDの候補が混入した場合は旧側 (_dl_expand.tsv) を優先し二重DLを避ける
# (2026-08-07、追加1本 lWmBXteEbeg が既存 video_c95 と重複した実例で確認)。
SOURCE_PRIORITY: dict[str, int] = {
    "data/_dl_expand.tsv": 0,
    "data/_dl_expand2_2026-08-07.tsv": 1,
    "scripts/_dl_tier_avc1.sh": 2,
    "data/pl_new_missing.tsv": 3,
    "data/phase_e_dl_index.tsv": 4,
    "": 9,  # UNCONFIRMED (記録なし)
}


@dataclass
class VideoRecord:
    """1動画分のティア台帳エントリ。"""

    video_name: str
    video_id: str
    duration_sec: str  # 欠損許容のため文字列で保持 (空文字 = 不明)
    title: str
    tier: str
    source_file: str
    on_disk: bool = False


def classify_tier(title: str) -> str:
    """タイトル文字列からキーワード分類で tier を決める。

    TIER_KEYWORD_ORDER の順に判定し、最初に一致したキーワード群の tier を
    返す。どれにも一致しなければ UNCONFIRMED。
    """
    for tier_name, keywords in TIER_KEYWORD_ORDER:
        if any(kw in title for kw in keywords):
            return tier_name
    return TIER_UNCONFIRMED


def discover_on_disk_names() -> set[str]:
    """data/frames/ 内の video_*.mp4 の stem 集合を返す。"""
    return {p.stem for p in FRAMES_DIR.glob("video_*.mp4")}


def load_phase_e_records() -> list[VideoRecord]:
    """data/phase_e_dl_index.tsv (v29-v95) を読み込む。"""
    records: list[VideoRecord] = []
    if not PHASE_E_INDEX_TSV.exists():
        return records
    with PHASE_E_INDEX_TSV.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            idx = int(row["video_idx"])
            name = f"video_{idx:0{VIDEO_NAME_ZERO_PAD}d}"
            title = row["title"]
            records.append(VideoRecord(
                video_name=name, video_id=row["video_id"],
                duration_sec=row["duration"], title=title,
                tier=classify_tier(title),
                source_file="data/phase_e_dl_index.tsv",
            ))
    return records


def load_pending_tierA_lookup() -> dict[str, tuple[str, str]]:
    """pending_tierA_missing.tsv から video_id -> (title, duration) 辞書を作る。"""
    lookup: dict[str, tuple[str, str]] = {}
    if not PENDING_TIERA_TSV.exists():
        return lookup
    with PENDING_TIERA_TSV.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            lookup[row["id"]] = (row["title"], row["duration"])
    return lookup


def build_hardcoded_s_c123_records() -> list[VideoRecord]:
    """s1-s3 (タイトル引き可) と c1-c3 (タイトル不明) の台帳行を作る。"""
    lookup = load_pending_tierA_lookup()
    records: list[VideoRecord] = []
    for name, vid in S1_S3_IDS.items():
        title, duration = lookup.get(vid, ("", ""))
        records.append(VideoRecord(
            video_name=name, video_id=vid, duration_sec=duration,
            title=title, tier=classify_tier(title),
            source_file="scripts/_dl_tier_avc1.sh",
        ))
    for name, vid in C1_C3_IDS.items():
        records.append(VideoRecord(
            video_name=name, video_id=vid, duration_sec="", title="",
            tier="チャレンジャー",  # 背景調査で確認済み (タイトル記録なし)
            source_file="scripts/_dl_tier_avc1.sh",
        ))
    return records


def load_pl_new_missing_records() -> list[VideoRecord]:
    """data/pl_new_missing.tsv (c4-c33、行順に逐次割当) を読み込む。"""
    records: list[VideoRecord] = []
    if not PL_NEW_MISSING_TSV.exists():
        return records
    with PL_NEW_MISSING_TSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for offset, row in enumerate(rows):
        c_idx = PL_NEW_MISSING_FIRST_C_INDEX + offset
        title = row["title"]
        records.append(VideoRecord(
            video_name=f"video_c{c_idx}", video_id=row["id"],
            duration_sec=row["dur"], title=title,
            tier=classify_tier(title),
            source_file="data/pl_new_missing.tsv",
        ))
    return records


def _load_name_id_title_tsv(path: Path, source_label: str) -> list[VideoRecord]:
    """ヘッダなし・3列 (video_name, id, title) の TSV を読む共通ローダ。

    data/_dl_expand.tsv / data/_dl_expand2_2026-08-07.tsv が同形式。
    """
    records: list[VideoRecord] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 3:
                continue
            name, vid, title = row[0], row[1], row[2]
            records.append(VideoRecord(
                video_name=name, video_id=vid, duration_sec="",
                title=title, tier=classify_tier(title),
                source_file=source_label,
            ))
    return records


def load_dl_expand_records() -> list[VideoRecord]:
    """data/_dl_expand.tsv (c34-c95、video_name/id/title 明示対応) を読む。"""
    return _load_name_id_title_tsv(DL_EXPAND_TSV, "data/_dl_expand.tsv")


def load_dl_expand2_records() -> list[VideoRecord]:
    """data/_dl_expand2_2026-08-07.tsv (c96-、Phase L step2) を読む。"""
    return _load_name_id_title_tsv(
        DL_EXPAND2_TSV, "data/_dl_expand2_2026-08-07.tsv",
    )


def build_all_records() -> dict[str, VideoRecord]:
    """全ソースを統合し video_name -> VideoRecord の辞書を作る。

    on_disk 未確認の動画向けに UNCONFIRMED の空行も補完する
    (v01-v28 等、記録がどのソースにも存在しない動画)。
    """
    merged: dict[str, VideoRecord] = {}
    all_loaders = (
        load_phase_e_records, build_hardcoded_s_c123_records,
        load_pl_new_missing_records, load_dl_expand_records,
        load_dl_expand2_records,
    )
    for loader in all_loaders:
        for rec in loader():
            if rec.video_name in merged:
                print(
                    f"[warn] video_name 重複 (別ソースで再定義): "
                    f"{rec.video_name} ({merged[rec.video_name].source_file} "
                    f"-> {rec.source_file})", file=sys.stderr,
                )
            merged[rec.video_name] = rec

    on_disk_names = discover_on_disk_names()
    for name in on_disk_names - merged.keys():
        merged[name] = VideoRecord(
            video_name=name, video_id="", duration_sec="", title="",
            tier=TIER_UNCONFIRMED, source_file="",
        )
    for rec in merged.values():
        rec.on_disk = rec.video_name in on_disk_names
    return merged


def dedupe_by_video_id(
    records: dict[str, VideoRecord],
) -> tuple[set[str], list[list[str]]]:
    """video_id が同一の video_name 群を検出し、採用優先順で主系列を選ぶ。

    戻り値: (除外対象 video_name 集合, 重複グループ [primary, *losers] の一覧)。
    video_id が空の行は対象外 (unknown 同士は重複とみなさない)。

    採用優先順は「on_disk (実ファイル有無) を最優先」+ SOURCE_PRIORITY
    (c系優先) を次点で用いる。実測で c38/c39 (data/_dl_expand.tsv 側、
    優先度最高) が現在 data/frames/ に存在せず、対応する旧 v51/v52
    (data/phase_e_dl_index.tsv 側) は存在するケースが確認された。
    on_disk を無視すると regen 可能な動画を誤って除外してしまうため、
    ソース優先度より先に on_disk の有無で判定する。
    """
    by_id: dict[str, list[str]] = {}
    for name, rec in records.items():
        if not rec.video_id:
            continue
        by_id.setdefault(rec.video_id, []).append(name)

    excluded: set[str] = set()
    dup_groups: list[list[str]] = []
    for names in by_id.values():
        if len(names) <= 1:
            continue
        ranked = sorted(
            names,
            key=lambda n: (
                not records[n].on_disk,
                SOURCE_PRIORITY.get(records[n].source_file, 9),
                n,
            ),
        )
        primary, *losers = ranked
        dup_groups.append([primary, *losers])
        excluded.update(losers)
    return excluded, dup_groups


def write_index_tsv(records: dict[str, VideoRecord]) -> None:
    """全動画台帳を data/video_tier_index_2026-08-07.tsv に書き出す。"""
    fieldnames = [
        "video_name", "video_id", "duration_sec", "title", "tier",
        "source_file", "on_disk",
    ]
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(fieldnames)
        for name in sorted(records):
            rec = records[name]
            writer.writerow([
                rec.video_name, rec.video_id, rec.duration_sec, rec.title,
                rec.tier, rec.source_file, "1" if rec.on_disk else "0",
            ])


def write_dup_report(
    records: dict[str, VideoRecord], dup_groups: list[list[str]],
) -> None:
    """ID重複検出レポートを標準出力と DUP_REPORT_TXT の両方に出す。"""
    lines = [f"[dup] 検出された video_id 重複グループ数: {len(dup_groups)}", ""]
    for group in sorted(dup_groups, key=lambda g: g[0]):
        primary, *losers = group
        vid = records[primary].video_id
        lines.append(f"video_id={vid}")
        lines.append(f"  採用: {primary} (source={records[primary].source_file})")
        for loser in losers:
            lines.append(
                f"  除外: {loser} (source={records[loser].source_file})",
            )
        lines.append("")
    text = "\n".join(lines)
    print(text)
    DUP_REPORT_TXT.write_text(text, encoding="utf-8")


def main() -> int:
    records = build_all_records()
    excluded, dup_groups = dedupe_by_video_id(records)
    write_index_tsv(records)
    write_dup_report(records, dup_groups)
    n_on_disk = sum(1 for r in records.values() if r.on_disk)
    n_confirmed_on_disk = sum(
        1 for r in records.values()
        if r.on_disk and r.tier != TIER_UNCONFIRMED
    )
    n_target = sum(
        1 for name, r in records.items()
        if r.on_disk and r.tier != TIER_UNCONFIRMED and name not in excluded
    )
    print(f"[index] 台帳総行数={len(records)} on_disk={n_on_disk} "
          f"on_disk&tier確定={n_confirmed_on_disk} 重複除外後regen対象={n_target}")
    print(f"[save] {OUT_TSV}")
    print(f"[save] {DUP_REPORT_TXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

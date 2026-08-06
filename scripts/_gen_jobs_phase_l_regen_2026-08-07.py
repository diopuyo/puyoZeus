"""Phase L 全動画 regen ジョブ一覧生成 (2026-08-07)。

scripts/build_video_tier_index.py の統合台帳から regen 対象
(on_disk かつ tier確定 かつ video_id重複除去後) を選び、
scripts/_collect_lean_1t.py 呼び出しコマンドを1行1本ずつ書き出す。

新標準構成 (第4機構修正A' 含む) を既定フラグとして採用する。
既定出力: scripts/_jobs_phase_l_regen_2026-08-07.txt

第二波(2026-08-07 追加): --exclude-jobs <file> --out <file> を optional で
渡すと、既存ジョブファイルに含まれる動画を除外した差分のみを別ファイルに
書き出せる (第一波 scripts/_jobs_phase_l_regen_2026-08-07.txt は走行中のため
無引数実行では一切変更しない、後方互換)。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_video_tier_index import (  # noqa: E402
    PHASE_L_TIER_WHITELIST,
    TIER_UNCONFIRMED,
    VideoRecord,
    build_all_records,
    dedupe_by_video_id,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_JOBS_TXT = PROJECT_ROOT / "scripts" / "_jobs_phase_l_regen_2026-08-07.txt"
OUT_NPZ_DIR = "data/indicators_v2/boards_lean_phase_l_2026-08-07"

# 新標準構成 (第4機構修正A' 含む、2026-08-07 確定)。
# フレーム間引き禁止 (--max-sec 0 --sample-interval 0) は前回全量regen
# (2026-07-31) からの確定知見を踏襲する。
PHASE_L_REGEN_FLAGS: tuple[str, ...] = (
    "--enable-chain-tracker",
    "--with-next",
    "--enable-effect-gate",
    "--enable-burst-guard-v2",
    "--enable-transition-merge-guard",
    "--burst-gate-open-threshold", "0.954",
    "--enable-hidden-row-burst-guard",
    "--enable-match-transition-debounce",
    "--max-sec", "0",
    "--sample-interval", "0",
)


def build_job_command(video_name: str) -> str:
    """1動画分の regen コマンド文字列を組み立てる。

    出力 npz 名は既存 lean_regen 系の慣習 (video_ プレフィクス除去) に揃える
    (例: video_c10 -> c10.npz)。
    """
    suffix = video_name.removeprefix("video_")
    video_path = f"$HOME/frames/{video_name}.mp4"
    out_npz = f"{OUT_NPZ_DIR}/{suffix}.npz"
    flags = " ".join(PHASE_L_REGEN_FLAGS)
    return (
        "PYTHONPATH=. ./venv/bin/python -u -m scripts._collect_lean_1t "
        f"--video {video_path} --out-npz {out_npz} {flags}"
    )


def select_target_video_names() -> list[str]:
    """regen 対象 video_name を確定する。

    条件: on_disk かつ tier が PHASE_L_TIER_WHITELIST (S級/マスター/
    チャレンジャー/A級、CLAUDE.md使用可スコープ) に含まれる かつ
    video_id重複解消済。「その他大会系」(B/C/D級混在の複合トーナメント)
    は 2026-08-07 user指示で既定除外する。
    """
    records: dict[str, VideoRecord] = build_all_records()
    excluded_dup, _ = dedupe_by_video_id(records)
    assert TIER_UNCONFIRMED not in PHASE_L_TIER_WHITELIST  # 安全確認
    return sorted(
        name for name, rec in records.items()
        if rec.on_disk and rec.tier in PHASE_L_TIER_WHITELIST
        and name not in excluded_dup
    )


def extract_video_names_from_jobs_file(path: Path) -> set[str]:
    """既存ジョブファイルの `--video <path>` 引数から video_name 集合を返す。

    第二波の差分抽出用 (第一波ジョブファイルに既に含まれる動画を除外する)。
    ファイル不在時は空集合 (除外なし)。
    """
    pattern = re.compile(r"--video\s+(\S+)")
    names: set[str] = set()
    if not path.exists():
        return names
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pattern.search(line)
        if m:
            names.add(Path(m.group(1)).stem)
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI引数を解析する。両方 optional、既定は従来動作 (後方互換)。"""
    parser = argparse.ArgumentParser(description="Phase L regen ジョブ一覧生成")
    parser.add_argument(
        "--exclude-jobs", type=Path, default=None,
        help="このジョブファイルに含まれる動画を除外して差分のみ出力する "
             "(optional、既定=除外なし)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="出力先パス (optional、既定 = scripts/_jobs_phase_l_regen_"
             "2026-08-07.txt)",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    targets = select_target_video_names()
    if args.exclude_jobs is not None:
        excluded_names = extract_video_names_from_jobs_file(args.exclude_jobs)
        targets = [n for n in targets if n not in excluded_names]
        print(f"[gen_jobs] --exclude-jobs={args.exclude_jobs} "
              f"除外={len(excluded_names)}件")

    out_path = args.out if args.out is not None else OUT_JOBS_TXT
    lines = [build_job_command(name) for name in targets]
    content = "\n".join(lines) + ("\n" if lines else "")
    out_path.write_text(content, encoding="utf-8")
    print(f"[gen_jobs] targets={len(targets)} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase L 全動画 regen ジョブ一覧生成 (2026-08-07)。

scripts/build_video_tier_index.py の統合台帳から regen 対象
(on_disk かつ tier確定 かつ video_id重複除去後) を選び、
scripts/_collect_lean_1t.py 呼び出しコマンドを1行1本ずつ書き出す。

新標準構成 (第4機構修正A' 含む) を既定フラグとして採用する。
出力: scripts/_jobs_phase_l_regen_2026-08-07.txt
"""
from __future__ import annotations

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


def main() -> int:
    targets = select_target_video_names()
    lines = [build_job_command(name) for name in targets]
    OUT_JOBS_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[gen_jobs] targets={len(targets)} -> {OUT_JOBS_TXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

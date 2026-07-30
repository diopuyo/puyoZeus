"""全フレーム基準データ収集 (2026-07-30) のジョブを幅優先に並べ替える。

coordinator指示 (2026-07-30): この基準データの用途は「最適化後の認識が正しいか
の汎化検証」であり、同一動画内の5試合は互いに相関が強く汎化の証拠にならない。
66動画×1試合(幅)を先に揃える方が、13動画×5試合(深さ)より検証価値が高い。

処理:
  1. data/verify/allframes_ref_2026-07-30/selected_games.csv を読む
     (video_id でグループ化済み、各グループ内は game_abs_idx 昇順
     = その動画内での時系列順位 1..NUM_GAMES_PER_VIDEO)。
  2. 出力 npz (data/indicators_v2/boards_lean_allframes_ref_2026-07-30/
     c{n}_g{game_abs_idx}.npz) の存在で完了済みを判定し、除外する
     (完了済みジョブは再実行しない)。
  3. 「ラウンド」= 各動画内の時系列順位 (1本目, 2本目, ...) で束ね、
     ラウンド1 (全動画の1本目) → ラウンド2 (全動画の2本目) → ... の順に
     ジョブを並べる (幅優先)。
  4. _collect_lean_1t 用ジョブ定義を LF 固定で書き出す
     (2026-07-30 CRLF事故の教訓、Windows python 実行時の write_text 既定
     newline変換で全ジョブ即死した実績があるため newline="\\n" を必ず明示)。
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SELECTED_CSV = PROJECT_ROOT / "data/verify/allframes_ref_2026-07-30/selected_games.csv"
OUT_NPZ_DIR_REL = "data/indicators_v2/boards_lean_allframes_ref_2026-07-30"
OUT_NPZ_DIR = PROJECT_ROOT / OUT_NPZ_DIR_REL
JOBS_OUT_PATH = PROJECT_ROOT / "scripts/_jobs_allframes_ref_2026-07-30_breadth.txt"
EXT4_FRAMES_DIR = "$HOME/frames"


def load_selected_rows(csv_path: Path) -> list[dict[str, Any]]:
    """selected_games.csv を読み込み、型変換して返す (元の行順を保持)。"""
    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "video_id": row["video_id"],
                "game_abs_idx": int(row["game_abs_idx"]),
                "start_sec": float(row["start_sec"]),
                "dur_sec": float(row["dur_sec"]),
            })
    return rows


def group_by_video_with_rank(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """video_id ごとにグループ化し、各グループ内での時系列順位 (rank) を付与する。

    selected_games.csv は build_selection() で video_id ごとに game_abs_idx
    昇順で書き出されているため、出現順そのものが時系列順位になる。
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(r["video_id"], []).append(r)
    for vid, games in grouped.items():
        for rank, g in enumerate(games, start=1):
            g["rank"] = rank
    return grouped


def is_done(video_id: str, game_abs_idx: int) -> bool:
    """出力npzが既に存在するか (完了済み判定)。"""
    n = video_id.replace("video_c", "")
    return (OUT_NPZ_DIR / f"c{n}_g{game_abs_idx}.npz").exists()


def build_breadth_first_order(
    grouped: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """ラウンド1 (全動画の1本目) → ラウンド2 → ... の順で未完了ジョブを並べる。"""
    max_rank = max((len(games) for games in grouped.values()), default=0)
    ordered: list[dict[str, Any]] = []
    for rank in range(1, max_rank + 1):
        for vid in sorted(grouped.keys()):
            games = grouped[vid]
            if rank > len(games):
                continue
            g = games[rank - 1]
            assert g["rank"] == rank
            if is_done(g["video_id"], g["game_abs_idx"]):
                continue
            ordered.append(g)
    return ordered


def write_jobs_txt(rows: list[dict[str, Any]], out_path: Path) -> None:
    """_collect_lean_1t 用ジョブ定義を LF 固定で書き出す (CRLF事故防止)。"""
    lines = []
    for r in rows:
        n = r["video_id"].replace("video_c", "")
        video_path = f"{EXT4_FRAMES_DIR}/video_c{n}.mp4"
        out_npz = f"{OUT_NPZ_DIR_REL}/c{n}_g{r['game_abs_idx']}.npz"
        cmd = (
            "PYTHONPATH=. ./venv/bin/python -u -m scripts._collect_lean_1t "
            f"--video {video_path} --out-npz {out_npz} "
            f"--start-sec {r['start_sec']} --max-sec {r['dur_sec']} "
            "--sample-interval 0 --enable-chain-tracker --with-next"
        )
        lines.append(cmd)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    """幅優先ジョブ定義を生成するエントリポイント。"""
    rows = load_selected_rows(SELECTED_CSV)
    grouped = group_by_video_with_rank(rows)
    ordered = build_breadth_first_order(grouped)
    write_jobs_txt(ordered, JOBS_OUT_PATH)

    n_videos = len(grouped)
    round1 = [g for g in ordered if g["rank"] == 1]
    round1_sec = sum(g["dur_sec"] for g in round1)
    total_sec = sum(g["dur_sec"] for g in ordered)
    print(f"[reorder] 対象動画数: {n_videos}")
    print(f"[reorder] 残りジョブ数 (完了済み除外後): {len(ordered)} / 330")
    print(f"[reorder] ラウンド1 (幅優先1周目) 残り: {len(round1)}本, "
          f"{round1_sec:.1f}秒")
    print(f"[reorder] 全残り総動画秒数: {total_sec:.1f}秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

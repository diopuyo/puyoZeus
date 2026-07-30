"""全フレーム基準データ収集 (2026-07-30) の対象試合を決定的に選定し、
_collect_lean_1t のジョブ定義ファイルを生成するスクリプト。

背景: frame間引き収集が盤面を壊す(project_frame_sampling_corrupts_boards_2026-07-30)
ことが確定したため、間引き耐性改修の検証台となる「全フレーム収集の基準データ」を
combined66 (c20+m20+m30 の66動画) から作る。単位は秒数の任意窓ではなく
「完結した1試合」(winners_panel_diff の games 配列) とする。

選定基準 (すべて日本語コメントで明示、マジックナンバーは定数化):
  - SKIP_FIRST_N_GAMES: 動画冒頭の試合は「まだ試合が始まっていない」
    可能性が高い(user指摘、c60 games[0] が異常に長く confidence=asymmetric
    だった実例で裏付け済み)ため、game_abs_idx がこの値未満の試合は除外する。
  - confidence == "strict" のみ (asymmetric は勝敗パネル差分が非対称=読み取り
    不安定を意味するため除外)。
  - 試合長 [DURATION_MIN_SEC, DURATION_MAX_SEC] の範囲内のみ (極端に短い/長い
    試合は境界誤検出の疑いが強い、全66動画3362 strict試合の実測分布から決定)。
  - 上記を満たす試合の中から NUM_GAMES_PER_VIDEO 本を「game_abs_idx で昇順に
    並べた上で全区間に渡り等間隔」に決定的に選ぶ(乱数不使用、恣意性排除)。

出力:
  - 選定結果レポート CSV (data/verify/allframes_ref_2026-07-30/selected_games.csv)
  - _collect_lean_1t 用ジョブ定義 (scripts/_jobs_allframes_ref_2026-07-30.txt)
  - 選べなかった動画があれば標準エラーに明記して報告できるようにする
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# --- 定数 (マジックナンバー禁止のため明示) ---------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMBINED66_CSV = (
    PROJECT_ROOT
    / "data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv"
)
WINNERS_PANEL_DIR = PROJECT_ROOT / "data/verify/winners_panel_diff_2026-07-26"
REPORT_DIR = PROJECT_ROOT / "data/verify/allframes_ref_2026-07-30"
JOBS_TXT_PATH = PROJECT_ROOT / "scripts/_jobs_allframes_ref_2026-07-30.txt"
OUT_NPZ_DIR = "data/indicators_v2/boards_lean_allframes_ref_2026-07-30"
# ext4 コピー先 (9p I/O ボトルネック回避、既存 c20/m20/m30 収集ジョブの慣例踏襲)
EXT4_FRAMES_DIR = "$HOME/frames"

SKIP_FIRST_N_GAMES = 3  # 動画冒頭の試合を機械的に除外する本数 (user指示 2026-07-30)
CONFIDENCE_REQUIRED = "strict"  # asymmetric は除外
DURATION_MIN_SEC = 30.0  # strict 試合分布 p5=30秒 (実測、下記 analyze 参照)
DURATION_MAX_SEC = 120.0  # strict 試合分布の大半(95.3%)をカバーしp99=118秒未満を除外
NUM_GAMES_PER_VIDEO = 5  # user指示 (3本→5本に変更 2026-07-30)


def load_combined66_video_ids(csv_path: Path) -> list[str]:
    """labeled_win_combined66.csv の video_id 列からユニーク一覧を得る (昇順)。"""
    video_ids: set[str] = set()
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_ids.add(row["video_id"])
    return sorted(video_ids)


def select_eligible_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """冒頭除外 + strict限定 + 試合長範囲で候補を絞り込む (game_abs_idx昇順維持)。"""
    eligible = []
    for g in games:
        if g["game_abs_idx"] < SKIP_FIRST_N_GAMES:
            continue
        if g["confidence"] != CONFIDENCE_REQUIRED:
            continue
        dur = g["end_sec"] - g["start_sec"]
        if not (DURATION_MIN_SEC <= dur <= DURATION_MAX_SEC):
            continue
        eligible.append(g)
    return sorted(eligible, key=lambda g: g["game_abs_idx"])


def pick_evenly_spaced(eligible: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """候補リストから両端を含む等間隔で k 本を決定的に選ぶ (乱数不使用)。

    m 件から k 件を選ぶ際、インデックス round(i*(m-1)/(k-1)) (i=0..k-1) を使う。
    これは中央付近だけに偏らず動画全体の試合進行(序盤〜終盤)をカバーする。
    """
    m = len(eligible)
    if m <= k:
        return eligible
    indices = sorted(
        {round(i * (m - 1) / (k - 1)) for i in range(k)}
    )
    # 丸め衝突で k 未満になった場合に備えて重複しない次点を機械的に補充する
    idx_set = set(indices)
    candidate = 0
    while len(idx_set) < k and candidate < m:
        idx_set.add(candidate)
        candidate += 1
    return [eligible[i] for i in sorted(idx_set)[:k]]


def build_selection(video_ids: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """全動画について選定を実行し、(選定結果行, 不足動画一覧) を返す。"""
    rows: list[dict[str, Any]] = []
    shortfall: list[str] = []
    for vid in video_ids:
        panel_path = WINNERS_PANEL_DIR / f"{vid}.json"
        with panel_path.open(encoding="utf-8") as f:
            panel = json.load(f)
        eligible = select_eligible_games(panel["games"])
        if len(eligible) < NUM_GAMES_PER_VIDEO:
            shortfall.append(
                f"{vid}: eligible={len(eligible)} (< {NUM_GAMES_PER_VIDEO})"
            )
        selected = pick_evenly_spaced(eligible, NUM_GAMES_PER_VIDEO)
        for g in selected:
            rows.append(
                {
                    "video_id": vid,
                    "game_abs_idx": g["game_abs_idx"],
                    "start_sec": g["start_sec"],
                    "end_sec": g["end_sec"],
                    "dur_sec": round(g["end_sec"] - g["start_sec"], 3),
                    "confidence": g["confidence"],
                }
            )
    return rows, shortfall


def write_report_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    """選定結果を CSV レポートとして書き出す。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_id", "game_abs_idx", "start_sec", "end_sec", "dur_sec", "confidence",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jobs_txt(rows: list[dict[str, Any]], out_path: Path) -> None:
    """_collect_lean_1t 用のジョブ定義 (1行1コマンド) を書き出す。

    --sample-interval 0 (全フレーム) を明示指定する (既定値だが混乱を避けるため)。
    --enable-chain-tracker / --with-next も明示指定する (2026-07-30 追加)。
    coordinator判断 (2026-07-30): 本番 collect_indicators_v2.py:714 が
    enable_chain_tracker=True かつ load_next_detector=True で動いているため、
    基準データ (正解データ) もこれに揃える。呼び出し側の設定不一致こそが
    今夜の一連の欠陥の根っこであり、基準データが本番と違う設定で作られていたら
    意味がない、という判断による。1動画 (video_c56) での3条件比較検証
    (OFF/chain_trackerのみ/両方ON) で snapshot数 236→214→275、
    chain_trigger_sec 非NaN率 0.0%→6.5%→7.6% と両方ONが最良だったことも
    この判断を支持する。
    入力動画は ext4 コピー先 ($HOME/frames) から読む (9p I/O ボトルネック回避)。
    """
    lines = []
    for r in rows:
        n = r["video_id"].replace("video_c", "")
        video_path = f"{EXT4_FRAMES_DIR}/video_c{n}.mp4"
        out_npz = f"{OUT_NPZ_DIR}/c{n}_g{r['game_abs_idx']}.npz"
        cmd = (
            "PYTHONPATH=. ./venv/bin/python -u -m scripts._collect_lean_1t "
            f"--video {video_path} --out-npz {out_npz} "
            f"--start-sec {r['start_sec']} --max-sec {r['dur_sec']} "
            "--sample-interval 0 --enable-chain-tracker --with-next "
            # 意図的なライブラリ既定上書き (2026-07-30, coordinator指示):
            # collect_boards_lean.py の normalize_fps_30 既定は True に変更
            # 済み (60fps stride-2 化、user承認済み)。しかしこの基準データは
            # 「全フレームの正解」であることが定義そのものであり、stride化
            # すると実効30fpsになって全フレームでなくなる。既に収集済みの
            # 全件が全フレームのため、ここで既定に従うと世代混在
            # (フレーム密度が違う npz が同一データセットに混ざる) を自分で
            # 作ってしまう。よって --no-normalize-fps-30 を必ず明示する。
            "--no-normalize-fps-30"
        )
        lines.append(cmd)
    # newline="\n" を明示 (2026-07-30 CRLF事故の修正: Windows python で
    # write_text の既定 newline 変換により \r\n化し、bash の read -r で
    # 末尾引数に \r が混入して argparse が "unrecognized arguments" で
    # 全ジョブ即死する事故が実際に発生した。feedback_crlf_churn_before_push
    # の教訓通り、生成物は必ず LF 固定で書き出す)。
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    """選定 + レポート + ジョブファイル生成のエントリポイント。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()  # 引数なし (固定定数運用、将来 optional 引数追加のみ許容)

    video_ids = load_combined66_video_ids(COMBINED66_CSV)
    rows, shortfall = build_selection(video_ids)

    write_report_csv(rows, REPORT_DIR / "selected_games.csv")
    write_jobs_txt(rows, JOBS_TXT_PATH)

    total_video_sec = sum(r["dur_sec"] for r in rows)
    print(f"[select] combined66 動画数: {len(video_ids)}")
    print(f"[select] 選定試合数: {len(rows)} (期待値 {len(video_ids) * NUM_GAMES_PER_VIDEO})")
    print(f"[select] 総動画秒数: {total_video_sec:.1f} 秒")
    if shortfall:
        print(f"[select][WARN] 5本に満たない動画 ({len(shortfall)}件):", file=sys.stderr)
        for line in shortfall:
            print(f"  {line}", file=sys.stderr)
    else:
        print("[select] 全動画で NUM_GAMES_PER_VIDEO 本を選定できた (不足なし)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

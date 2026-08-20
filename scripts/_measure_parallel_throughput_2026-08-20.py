"""収集の実スループットを測る (2026-08-20、並列数の最適点を探すため)。

14並列のベースライン実測 (21:00 時点、経過138分):
  全体 8,090 frames/min / 1プロセス 578 frames/min / 完了4本

物理8コア(論理16)に14プロセスは 1プロセスあたり0.57物理コア相当で
詰め込みすぎの疑いがある。並列数を落とすと 1プロセスあたりが速くなるが
本数が減るので、**総スループットが上がるかは理論では決まらない**
(損益分岐: 10並列なら 8,090/10 = 809 frames/min を超えれば改善)。

公平に比べるため、測るのは常に「全体 frames/min」にする。1プロセスあたりの
数字は並列数が違えば当然変わるので、比較指標にしてはいけない。

進行中フレームは per_video ログの最新 `t=` × fps で推定する。ログ出力の
頻度が動画ごとに違うため過小評価になりうるので、**完了本数のフレーム数
(status.tsv から確定値) を必ず足す**。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

cv2.setNumThreads(1)

_STATUS = PROJECT_ROOT / "data" / "verify" / "regen_2026-08-20_model50v2" / "status.tsv"
_LOGDIR = PROJECT_ROOT / "logs" / "regen_model50v2_2026-08-20_per_video"
_FRAMES = PROJECT_ROOT / "data" / "frames"


def _frames_of(tid: str) -> tuple[float, float]:
    """(総フレーム数, fps) を返す。"""
    c = cv2.VideoCapture(str(_FRAMES / f"video_{tid}.mp4"))
    n = c.get(cv2.CAP_PROP_FRAME_COUNT)
    f = c.get(cv2.CAP_PROP_FPS) or 30.0
    c.release()
    return n, f


def main() -> int:
    """指定の起動時刻からの経過で全体スループットを算出する。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="起動時刻 HH:MM:SS (今日)")
    ap.add_argument("--skip-done", type=int, default=0,
                    help="起動時に既完了だった本数 (その分のフレームは除外する)")
    args = ap.parse_args()

    h, m, sec = (int(x) for x in args.start.split(":"))
    today = dt.date.today()
    start = dt.datetime.combine(today, dt.time(h, m, sec))
    el = (dt.datetime.now() - start).total_seconds()
    if el <= 0:
        print("[error] 起動時刻が未来")
        return 1

    running = subprocess.run(
        ["pgrep", "-af", "_collect_lean_1t"], capture_output=True, text=True,
    ).stdout
    active = sorted(set(re.findall(r"video_([a-z0-9]+)\.mp4", running)))

    # 完了分 (確定値)。--skip-done で起動前に完了していた分を除く
    done_ids: list[str] = []
    if _STATUS.exists():
        with _STATUS.open(encoding="utf-8") as f:
            done_ids = [r["target_id"] for r in csv.DictReader(f, delimiter="\t")]
    new_done = done_ids[args.skip_done:]
    done_frames = sum(_frames_of(t)[0] for t in new_done)

    # 進行中 (ログからの推定、過小評価になりうる)
    prog = 0.0
    for tid in active:
        lg = _LOGDIR / f"{tid}.log"
        if not lg.exists():
            continue
        mm = re.findall(r"t=([0-9]+\.[0-9]+)", lg.read_text(errors="ignore"))
        if not mm:
            continue
        _, fps = _frames_of(tid)
        prog += float(mm[-1]) * fps

    total = done_frames + prog
    print(f"経過 {el/60:.0f}分 / 稼働 {len(active)} 本 / 期間中の完了 {len(new_done)} 本")
    print(f"処理フレーム (完了{done_frames:,.0f} + 進行中{prog:,.0f}) = {total:,.0f}")
    rate = total / el * 60
    print(f"**全体スループット {rate:,.0f} frames/min**")
    print(f"  (参考) 1プロセスあたり {rate/max(1,len(active)):,.0f} frames/min")
    print()
    print("比較対象: 14並列のベースライン **8,090 frames/min**")
    if rate > 8090:
        print(f"  → 改善 {rate/8090:.2f}倍")
    else:
        print(f"  → 悪化または同等 {rate/8090:.2f}倍")
    remain = 8393358 - sum(_frames_of(t)[0] for t in done_ids) - prog
    if rate > 0:
        fin = dt.datetime.now() + dt.timedelta(minutes=remain / rate)
        print(f"  残り {remain:,.0f} frames → 完了見込み {fin.strftime('%m/%d %H:%M')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

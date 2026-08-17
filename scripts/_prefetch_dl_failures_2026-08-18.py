"""148再収集で DL に失敗した動画を、走行を止めずに裏で先回りDLする (2026-08-18)。

## なぜ必要か

`scripts/_regen148_recollect_2026-08-18.py` の DL は YouTube から 403 Forbidden を
高頻度で受ける。リトライ4回 (間8秒) でほとんどは回復するが、4回とも外すと
`collect_status=SKIP_DL_FAIL` として記録され、その動画は欠測になる。
2026-08-18 の実測では失敗率が 12.5% -> 17.6% と推移しており、148本なら
20〜26本が欠ける計算。

欠測分はパイプラインを再実行すれば再挑戦されるが、その回収実行でも
「DL (数分) + 収集 (約3時間)」が丸ごと必要になる。本スクリプトは走行中に
**DL だけを先回りして済ませておく**ことで、回収実行を収集だけにする。

パイプライン側の `download_video` は出力先が既に存在すれば即座に成功を返すため、
`data/frames/` に置いておくだけで回収実行がそのまま速くなる (実装への変更は不要)。

## 設計

- 本スクリプトは **DL のみ** を行う。収集は一切走らせない (CPUは収集14並列で
  飽和しているため、CPUをほぼ使わない DL だけに徹する)
- 1本ずつ直列、間に待機を挟む。403 はレート由来の可能性があるため、
  パイプライン本体 (待機8秒) よりゆっくり回す
- 走行中のパイプラインとは status.tsv を読むだけの関係で、書き込みは一切しない
- 既に `data/frames/` にある動画は skip する (パイプラインが収集後に削除した
  ものを再DLしてしまわないよう、status.tsv 上で SKIP_DL_FAIL の分だけを対象)

## 使い方

    wsl -d Ubuntu -- bash -c "cd /mnt/c/.../puyo_analyzer && \
      setsid -f ./venv/bin/python -u scripts/_prefetch_dl_failures_2026-08-18.py \
      > logs/prefetch_dl_failures_2026-08-18.log 2>&1 < /dev/null"
"""
from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_TSV = PROJECT_ROOT / "data" / "verify" / "regen_2026-08-11_manifest.tsv"
STATUS_TSV = PROJECT_ROOT / "data" / "verify" / "regen_2026-08-18" / "status.tsv"
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
FFMPEG_LOCATION = PROJECT_ROOT / "venv" / "bin"
NODE24_PATH = "/home/ryouj/.nvm/versions/node/v24.19.0/bin/node"

# パイプライン本体と同一のフォーマット指定 (AV1 は OpenCV でデコード不可)。
YT_DLP_FORMAT = (
    "bv*[vcodec^=avc1][height<=1080]+ba/"
    "b[ext=mp4][vcodec^=avc1][height<=1080]/"
    "b[height<=1080][vcodec!*=av01]/b[ext=mp4]"
)

# 本体 (4回/8秒) よりゆっくり・粘り強く。403 がレート由来なら間隔を空けるほど当たる。
DL_RETRY_COUNT = 6
DL_RETRY_SLEEP_SEC = 45.0
# 1本DLし終えてから次を始めるまでの間隔。
BETWEEN_VIDEOS_SLEEP_SEC = 30.0
# status.tsv に新しい失敗が現れるのを待つポーリング間隔。
POLL_SLEEP_SEC = 300.0
TOTAL_TARGETS = 148


def load_manifest() -> dict[str, tuple[str, str]]:
    """target_id -> (video_filename, video_id) を返す。"""
    out: dict[str, tuple[str, str]] = {}
    with MANIFEST_TSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[row["target_id"]] = (row["video_filename"], row["video_id"])
    return out


def load_failed_ids() -> tuple[list[str], int]:
    """SKIP_DL_FAIL な target_id の一覧と、status.tsv の総行数を返す。"""
    failed: list[str] = []
    total = 0
    if not STATUS_TSV.exists():
        return failed, total
    with STATUS_TSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            total += 1
            if row.get("collect_status") == "SKIP_DL_FAIL":
                failed.append(row["target_id"])
    return failed, total


def download_one(target_id: str, video_filename: str, video_id: str) -> bool:
    """1本DLする。成功したら True。"""
    out_path = FRAMES_DIR / video_filename
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[prefetch][{target_id}] 既に存在するため skip", flush=True)
        return True

    url = f"https://www.youtube.com/watch?v={video_id}"
    for attempt in range(1 + DL_RETRY_COUNT):
        cmd = [
            str(PROJECT_ROOT / "venv" / "bin" / "python"), "-m", "yt_dlp",
            "--ffmpeg-location", str(FFMPEG_LOCATION),
            "--js-runtimes", f"node:{NODE24_PATH}",
            "-f", YT_DLP_FORMAT,
            "--remux-video", "mp4", "--no-playlist", "--no-progress",
            "-o", str(out_path), url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[prefetch][{target_id}] OK (attempt={attempt})", flush=True)
            return True
        for p in FRAMES_DIR.glob(f"{video_filename}*part"):
            p.unlink(missing_ok=True)
        print(
            f"[prefetch][{target_id}] 失敗 (attempt={attempt}): "
            f"{proc.stderr[-200:]}", flush=True,
        )
        if attempt < DL_RETRY_COUNT:
            time.sleep(DL_RETRY_SLEEP_SEC)
    return False


def main() -> int:
    manifest = load_manifest()
    done: set[str] = set()
    give_up: set[str] = set()
    print("[prefetch] 開始", flush=True)

    while True:
        failed, total = load_failed_ids()
        todo = [t for t in failed if t not in done and t not in give_up]

        if not todo:
            # パイプラインが148本すべてを status.tsv に記録し終え、かつ
            # 未処理の失敗が無ければ役目は終わり。
            if total >= TOTAL_TARGETS:
                print(
                    f"[prefetch] 完了 (status {total}/{TOTAL_TARGETS}行、"
                    f"回収成功={len(done)} 断念={len(give_up)})", flush=True,
                )
                return 0
            time.sleep(POLL_SLEEP_SEC)
            continue

        for target_id in todo:
            if target_id not in manifest:
                print(f"[prefetch][{target_id}] manifest に無い、skip", flush=True)
                give_up.add(target_id)
                continue
            video_filename, video_id = manifest[target_id]
            if not video_id:
                # origin=derived_c96 等、URL から取れないもの。
                print(f"[prefetch][{target_id}] video_id 無し、skip", flush=True)
                give_up.add(target_id)
                continue
            if download_one(target_id, video_filename, video_id):
                done.add(target_id)
            else:
                give_up.add(target_id)
                print(f"[prefetch][{target_id}] 断念", flush=True)
            time.sleep(BETWEEN_VIDEOS_SLEEP_SEC)


if __name__ == "__main__":
    raise SystemExit(main())

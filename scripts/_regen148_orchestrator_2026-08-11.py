"""148 動画 学習データ再生成パイプライン 本体 (2026-08-11、user承認済み)。

data/indicators_v2/boards_lean_phase_l_2026-08-07/ の 148 npz を、現行の
採用認識構成 (src/production_config.py + 幻盤面ガード新規追加) で再生成する。

設計 (詳細は data/verify/regen_2026-08-11_manifest.tsv 生成元の
scripts/_gen_regen_manifest_2026-08-11.py と合わせて参照):
- ダウンロードは 1 本ずつ直列。 収集キュー (maxsize=1) が空くまで先行 DL は
  待機する = 「収集3並列 + 先行DL1本」で同時保持動画を概ね 4 本に抑える。
- 収集 (collect_boards_lean) は最大 3 並列 (docs/CYCLE_FINDINGS.md の
  GPU 制約由来ルールを流用、収集自体はCPU処理だが並列上限として踏襲)。
- 収集成功判定は npz 存在 + grids 行数 > 0。
- 動画削除は origin="download"/"derived_c96" のみ、収集成功後に行う。
  origin="preexisting" (元々 data/frames/ にあった 47 本) は絶対に削除しない。
- data/verify/regen_2026-08-11_status.tsv に逐次追記 (再開可能: 既に
  collect_status=OK な target_id はスキップする)。
"""
from __future__ import annotations

import csv
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.production_config import collect_flags  # noqa: E402

MANIFEST_TSV = PROJECT_ROOT / "data" / "verify" / "regen_2026-08-11_manifest.tsv"
STATUS_TSV = PROJECT_ROOT / "data" / "verify" / "regen_2026-08-11_status.tsv"
NEW_NPZ_DIR = PROJECT_ROOT / "data" / "indicators_v2" / "boards_lean_phase_l_2026-08-11"
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
LOG_DIR = PROJECT_ROOT / "logs" / "regen_2026-08-11_per_video"
FFMPEG_LOCATION = PROJECT_ROOT / "venv" / "bin"

MAX_COLLECT_PARALLEL = 10       # CPU処理のみのため引き上げ (wave1/2はMAXPAR=14実績。§3.1の上限3はGPU制約由来で本パイプライン非該当、2026-08-11)
DOWNLOAD_QUEUE_SIZE = 2         # 先行DLは1本まで (同時保持を約4本に抑える)
TOTAL_HOLD_SLOTS = MAX_COLLECT_PARALLEL + DOWNLOAD_QUEUE_SIZE  # 同時保持上限 (4本)
MIN_FREE_DISK_GB = 30.0         # これを下回ったらDLを一時停止する安全弁
DL_RETRY_COUNT = 1              # DL失敗時の自動再試行回数

# 同時保持本数の実効上限。ThreadPoolExecutor.submit() は即時リターンし内部
# キューに無制限に積めてしまうため、これが無いと DL スレッドが収集完了を
# 待たずに全 144 本を連続DLしてしまう (ready_q の maxsize=1 だけでは
# 「submit 済みだが未完了」を区別できずスロットリングにならない)。
# acquire は DL/準備の直前、release は collect_one 完了時 (成否問わず) に行う。
_HOLD_SEMAPHORE = threading.Semaphore(TOTAL_HOLD_SLOTS)

# AV1 は OpenCV でデコード不可 (memory 教訓、2026-08 各所 DL script で確認済)。
# avc1(H.264) を優先し、無ければ 1080p 以下にフォールバックする。
YT_DLP_FORMAT = (
    "bv*[vcodec^=avc1][height<=1080]+ba/"
    "b[ext=mp4][vcodec^=avc1][height<=1080]/"
    "b[height<=1080][vcodec!*=av01]/b[ext=mp4]"
)

STATUS_HEADER = (
    "target_id\tvideo_id\ttier\torigin\tdl_status\tdl_seconds\t"
    "video_bytes\tcollect_status\trows\tcollect_seconds\tdeleted\tfinished_at\n"
)

_status_lock = threading.Lock()


@dataclass
class Target:
    target_id: str
    video_filename: str
    video_id: str
    tier: str
    origin: str  # preexisting / derived_c96 / download


def load_manifest() -> list[Target]:
    targets: list[Target] = []
    with MANIFEST_TSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            targets.append(Target(
                target_id=row["target_id"], video_filename=row["video_filename"],
                video_id=row["video_id"], tier=row["tier"], origin=row["origin"],
            ))
    return targets


def load_done_ids() -> set[str]:
    """再開用: 既に collect_status=OK な target_id を返す。"""
    done: set[str] = set()
    if not STATUS_TSV.exists():
        return done
    with STATUS_TSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("collect_status") == "OK":
                done.add(row["target_id"])
    return done


def append_status(fields: dict[str, str]) -> None:
    with _status_lock:
        is_new = not STATUS_TSV.exists()
        STATUS_TSV.parent.mkdir(parents=True, exist_ok=True)
        with STATUS_TSV.open("a", encoding="utf-8", newline="\n") as f:
            if is_new:
                f.write(STATUS_HEADER)
            f.write(
                "\t".join(fields.get(k, "") for k in (
                    "target_id", "video_id", "tier", "origin", "dl_status",
                    "dl_seconds", "video_bytes", "collect_status", "rows",
                    "collect_seconds", "deleted", "finished_at",
                )) + "\n"
            )


def _free_disk_gb() -> float:
    usage = shutil.disk_usage(str(PROJECT_ROOT))
    return usage.free / (1024 ** 3)


def download_video(t: Target) -> tuple[bool, float]:
    """yt-dlp で t.video_filename を data/frames/ に取得する。成否と所要秒を返す。"""
    out_path = FRAMES_DIR / t.video_filename
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, 0.0
    url = f"https://www.youtube.com/watch?v={t.video_id}"
    start = time.monotonic()
    for attempt in range(1 + DL_RETRY_COUNT):
        cmd = [
            str(PROJECT_ROOT / "venv" / "bin" / "python"), "-m", "yt_dlp",
            "--ffmpeg-location", str(FFMPEG_LOCATION),
            # 2026-08-11: JSランタイム必須化 (無いと一部動画が403) への対処。
            # WSL の nvm 管理 node を明示指定する
            "--js-runtimes", "node:/home/ryouj/.nvm/versions/node/v20.20.1/bin/node",
            "-f", YT_DLP_FORMAT,
            "--remux-video", "mp4", "--no-playlist", "--no-progress",
            "-o", str(out_path), url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if out_path.exists() and out_path.stat().st_size > 0:
            return True, time.monotonic() - start
        # 失敗した部分ファイルを掃除してから再試行 (yt-dlp の .part / .f137.part 等)
        for p in FRAMES_DIR.glob(f"{t.video_filename}*part"):
            p.unlink(missing_ok=True)
        print(f"[dl][retry={attempt}] {t.target_id} 失敗: {proc.stderr[-300:]}", flush=True)
    return False, time.monotonic() - start


def prepare_c96_splits(targets: list[Target], done_ids: set[str]) -> None:
    """video_c96 から c96s1/s2/s3 を切り出す (未生成分・未完了分のみ)。"""
    needed = [
        t for t in targets
        if t.origin == "derived_c96" and t.target_id not in done_ids
        and not (FRAMES_DIR / t.video_filename).exists()
    ]
    if not needed:
        return
    src = FRAMES_DIR / "video_c96.mp4"
    if not src.exists():
        print(f"[c96-split][ERROR] ソース不在: {src}", file=sys.stderr)
        return
    seg_tsv = PROJECT_ROOT / "data" / "verify" / "c96_split_2026-08-08" / "series_segments.tsv"
    segments: dict[str, tuple[str, str]] = {}
    with seg_tsv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            segments[f"c96s{row['idx']}"] = (row["clip_start_sec"], row["clip_end_sec"])

    import imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    for t in needed:
        start, end = segments[t.target_id]
        out_path = FRAMES_DIR / t.video_filename
        print(f"[c96-split] {t.target_id}: {start} -> {end}", flush=True)
        subprocess.run(
            [ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-ss", start, "-to", end, "-i", str(src), "-c", "copy", str(out_path)],
            check=False,
        )


def _npz_row_count(npz_path: Path) -> int:
    with np.load(npz_path) as d:
        return int(d["grids"].shape[0])


def collect_one(t: Target) -> None:
    """1 動画を収集する (ThreadPoolExecutor worker)。

    _HOLD_SEMAPHORE の release は必ず行う (成否問わず、DLスレッドの
    デッドロック防止のため try/finally で保証する)。
    """
    try:
        _collect_one_body(t)
    finally:
        _HOLD_SEMAPHORE.release()


def _collect_one_body(t: Target) -> None:
    """collect_one の本体 (release 保証のため分離、1関数50行制約対応)。"""
    video_path = FRAMES_DIR / t.video_filename
    out_npz = NEW_NPZ_DIR / f"{t.target_id}.npz"
    NEW_NPZ_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cmd = (
        [str(PROJECT_ROOT / "venv" / "bin" / "python"), "-u", "-m",
         "scripts._collect_lean_1t",
         "--video", str(video_path), "--out-npz", str(out_npz)]
        + collect_flags().split()
        + ["--with-next", "--enable-phantom-board-guard",
           "--max-sec", "0", "--sample-interval", "0"]
    )
    log_path = LOG_DIR / f"{t.target_id}.log"
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    collect_seconds = time.monotonic() - start

    rows = 0
    collect_status = "FAIL"
    if proc.returncode == 0 and out_npz.exists():
        try:
            rows = _npz_row_count(out_npz)
            collect_status = "OK" if rows > 0 else "FAIL_EMPTY"
        except Exception as e:  # noqa: BLE001
            print(f"[collect][{t.target_id}] npz 読み込み失敗: {e}", file=sys.stderr)
            collect_status = "FAIL_NPZ_READ"

    deleted = "N"
    if collect_status == "OK" and t.origin in ("download", "derived_c96"):
        try:
            video_path.unlink(missing_ok=True)
            deleted = "Y"
        except OSError as e:  # noqa: BLE001
            print(f"[collect][{t.target_id}] 動画削除失敗: {e}", file=sys.stderr)

    append_status({
        "target_id": t.target_id, "video_id": t.video_id, "tier": t.tier,
        "origin": t.origin, "dl_status": "SKIP_NA",
        "dl_seconds": "0", "video_bytes": "",
        "collect_status": collect_status, "rows": str(rows),
        "collect_seconds": f"{collect_seconds:.1f}", "deleted": deleted,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(
        f"[collect][{t.target_id}] status={collect_status} rows={rows} "
        f"sec={collect_seconds:.1f} deleted={deleted}", flush=True,
    )


def downloader_loop(
    targets: list[Target], done_ids: set[str], ready_q: "queue.Queue[Target]",
) -> None:
    """DL担当スレッド: 直列DL + _HOLD_SEMAPHORE で同時保持を4本に抑える。

    acquire は「この target を保持し始める」直前 (DL開始前) に行い、
    release は collect_one 完了時 (成否問わず) に行う。DL失敗時のみ
    ここで即 release する (収集フェーズに進まないため)。
    """
    for t in targets:
        if t.target_id in done_ids:
            continue
        if t.origin == "download":
            while _free_disk_gb() < MIN_FREE_DISK_GB:
                print(f"[dl] 空き容量不足 ({_free_disk_gb():.1f}GB) のため待機", flush=True)
                time.sleep(60)
        _HOLD_SEMAPHORE.acquire()
        if t.origin == "download":
            ok, dl_seconds = download_video(t)
            if not ok:
                append_status({
                    "target_id": t.target_id, "video_id": t.video_id,
                    "tier": t.tier, "origin": t.origin, "dl_status": "FAIL",
                    "dl_seconds": f"{dl_seconds:.1f}", "video_bytes": "0",
                    "collect_status": "SKIP_DL_FAIL", "rows": "0",
                    "collect_seconds": "0", "deleted": "N",
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                print(f"[dl][{t.target_id}] DL失敗、収集スキップ", flush=True)
                _HOLD_SEMAPHORE.release()
                continue
            print(f"[dl][{t.target_id}] OK ({dl_seconds:.1f}s)", flush=True)
        # preexisting / derived_c96 は既に用意済み (derived は事前ステップで生成済み)。
        # semaphore は保持したまま collect_one に引き継ぎ、そちらで release する。
        ready_q.put(t)  # キューが満杯 (=収集側が詰まっている) ならここでブロック
    ready_q.put(None)  # type: ignore[arg-type]  # 終端シグナル


def main() -> int:
    targets = load_manifest()
    done_ids = load_done_ids()
    print(f"[orchestrator] 総数={len(targets)} 既完了={len(done_ids)}", flush=True)

    prepare_c96_splits(targets, done_ids)

    ready_q: "queue.Queue[Target | None]" = queue.Queue(maxsize=DOWNLOAD_QUEUE_SIZE)
    dl_thread = threading.Thread(
        target=downloader_loop, args=(targets, done_ids, ready_q), daemon=True,
    )
    dl_thread.start()

    from concurrent.futures import ThreadPoolExecutor

    # 実際の同時実行数抑制は _HOLD_SEMAPHORE (acquire=DL開始前/release=collect完了時)
    # が担うため、ここでは受け取った分をそのまま submit するだけでよい。
    in_flight: list = []
    with ThreadPoolExecutor(max_workers=MAX_COLLECT_PARALLEL) as pool:
        while True:
            item = ready_q.get()
            if item is None:
                break
            in_flight.append(pool.submit(collect_one, item))
        for f in in_flight:
            f.result()

    dl_thread.join()
    print("[orchestrator] ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

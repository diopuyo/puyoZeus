"""148再収集の番人 — DLがレート制限で全滅しても放置で完走させる (2026-08-18)。

## なぜ必要か

YouTube のレート制限により、パイプライン本体の DL が連続で 403 を返す時間帯がある
(2026-08-18 実測: c103 まで順調 → c104 以降8本以上が連続失敗)。本体は失敗した
target を `SKIP_DL_FAIL` として記録して先に進むだけなので、制限中は残り全部を
取りこぼしたまま `ALL_DONE` で終了してしまう。

`scripts/_prefetch_dl_failures_2026-08-18.py` (先回りDL) が制限の合間を縫って
動画を `data/frames/` に貯めるので、**本体を再実行すれば貯まった分を14並列で
収集できる**。本スクリプトはその再実行を自動化し、長時間放置でも148本が
埋まるまで回し続ける。

yt-dlp は既に最新 (2026.07.04 が PyPI 最新) であり、ツール更新では解決しない。
403 は時間で解けるのを待つしかない、という前提に立った設計。

## 動作

1. 本体プロセスが走っていれば何もしない (完了を待つ)
2. 本体が終了していて、未収集の target のうち**動画がローカルに存在するもの**が
   1本以上あれば、本体を再起動する (`collect_status=OK` の分は本体側がスキップ)
3. 全148本が OK になったら終了
4. 動画が1本も貯まっていなければ、先回りDLの回収を待って様子見に戻る

再起動が空回りしないよう、最低間隔 `MIN_RESTART_INTERVAL_SEC` を空ける。

## 使い方

    wsl -d Ubuntu -- bash -c "cd /mnt/c/.../puyo_analyzer && \
      setsid -f ./venv/bin/python -u scripts/_watchdog_regen148_2026-08-18.py \
      > logs/watchdog_regen148_2026-08-18.log 2>&1 < /dev/null"
"""
from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_TSV = PROJECT_ROOT / "data" / "verify" / "regen_2026-08-11_manifest.tsv"
STATUS_TSV = PROJECT_ROOT / "data" / "verify" / "regen_2026-08-18" / "status.tsv"
FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
RUNNER_SH = PROJECT_ROOT / "scripts" / "_run_regen148_recollect_2026-08-18.sh"

TOTAL_TARGETS = 148
POLL_SLEEP_SEC = 300.0
# 再起動の空回り防止。本体は起動〜収集完了まで数時間かかるため長めでよい。
MIN_RESTART_INTERVAL_SEC = 1800.0
# 本体プロセスの検出パターン。
ORCH_PATTERN = "_regen148_recollect_2026-08-18"


def load_manifest() -> dict[str, str]:
    """target_id -> video_filename。"""
    out: dict[str, str] = {}
    with MANIFEST_TSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[row["target_id"]] = row["video_filename"]
    return out


def load_collected_ids() -> set[str]:
    """collect_status=OK の target_id。"""
    ok: set[str] = set()
    if not STATUS_TSV.exists():
        return ok
    with STATUS_TSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("collect_status") == "OK":
                ok.add(row["target_id"])
    return ok


def orchestrator_running() -> bool:
    proc = subprocess.run(
        ["pgrep", "-f", ORCH_PATTERN], capture_output=True, text=True,
    )
    return bool(proc.stdout.strip())


def ready_to_collect(manifest: dict[str, str], collected: set[str]) -> list[str]:
    """未収集かつ動画がローカルに存在する target_id。"""
    ready: list[str] = []
    for target_id, video_filename in manifest.items():
        if target_id in collected:
            continue
        path = FRAMES_DIR / video_filename
        if path.exists() and path.stat().st_size > 0:
            ready.append(target_id)
    return ready


def restart_orchestrator() -> None:
    subprocess.run(
        ["setsid", "-f", "bash", str(RUNNER_SH)],
        stdin=subprocess.DEVNULL, cwd=str(PROJECT_ROOT), check=False,
    )


def main() -> int:
    manifest = load_manifest()
    last_restart = 0.0
    print("[watchdog] 開始", flush=True)

    while True:
        collected = load_collected_ids()
        if len(collected) >= TOTAL_TARGETS:
            print(f"[watchdog] 全{TOTAL_TARGETS}本の収集完了、終了", flush=True)
            return 0

        if orchestrator_running():
            time.sleep(POLL_SLEEP_SEC)
            continue

        ready = ready_to_collect(manifest, collected)
        remaining = TOTAL_TARGETS - len(collected)
        if not ready:
            print(
                f"[watchdog] 本体停止中だが収集可能な動画なし "
                f"(収集済={len(collected)}/{TOTAL_TARGETS}、先回りDLの回収待ち)",
                flush=True,
            )
            time.sleep(POLL_SLEEP_SEC)
            continue

        since = time.monotonic() - last_restart
        if last_restart and since < MIN_RESTART_INTERVAL_SEC:
            time.sleep(POLL_SLEEP_SEC)
            continue

        print(
            f"[watchdog] 本体を再起動 (収集済={len(collected)}/{TOTAL_TARGETS}、"
            f"残={remaining}、うち動画あり={len(ready)}本)", flush=True,
        )
        restart_orchestrator()
        last_restart = time.monotonic()
        # 起動直後は本体が立ち上がるまで待つ。
        time.sleep(POLL_SLEEP_SEC)


if __name__ == "__main__":
    raise SystemExit(main())

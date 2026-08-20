"""50本 再収集 v2 (2026-08-20、境界修正3フラグの配線是正後)。

`_regen_model50_2026-08-19.py` の 50本収集は、境界修正3フラグ
(--enable-lockdown-score-numeric-release / --enable-lockdown-score-moving-release
/ --enable-boundary-newmatch-evidence) が `collect_flags()` に登録漏れしていた
ため一度も発火せず、20本時点で停止した。コミット 0595ed9 で登録を是正し、
本番オーケストレータ経由の2本実収集 (39/38) で配線を実証済み:

  - 39 は手書きフラグ版 (boards_lean_lockfix_2026-08-19) と全指標が完全一致
    (5,418行 / 欠損19.3% / 59試合 / ラッチ1.3%) = 本番経路にフラグが届いている
  - 38 (手書き版なしの新規動画) で 行数 2,565→5,422 / 欠損 16.6%→0.2% /
    ラッチ 12.9%→2.5%

そのため旧 `boards_lean_model50_2026-08-19` (20本) は無効データとして破棄し、
本スクリプトで 50本を取り直す。パイプラインは実績ある
`_regen148_orchestrator_2026-08-11.py` をパス上書きだけで流用する
(ロジックは複製しない)。

manifest の origin (2026-08-20 の変更点):
  - 動画が data/frames に残っている 30本 = "preexisting" にして**削除させない**
    (収集品質の再確認が起きやすい時期のため。ディスクは 130G 空きで余裕)
  - 手元にない 20本 = "download" のまま (収集成功後に削除、ストレージ節約規約)

Cookie: 旧セッションの scratchpad にあったマスターは参照できないため、
`_cookie_args` はマスター不在時に空リストを返す実装に依存して Cookie なしで
DL を試みる。年齢制限等で失敗した動画は status.tsv に FAIL が残るので、
その時点で user に Cookie を依頼する。
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_ORCH_PATH = PROJECT_ROOT / "scripts" / "_regen148_orchestrator_2026-08-11.py"
_spec = importlib.util.spec_from_file_location("_regen148_orchestrator_impl", _ORCH_PATH)
orch = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = orch
_spec.loader.exec_module(orch)

_NODE24_PATH = "/home/ryouj/.nvm/versions/node/v24.19.0/bin/node"
_DL_RETRY_COUNT = 5
_DL_RETRY_SLEEP_SEC = 45.0


def _download_video(t):  # noqa: ANN001, ANN201 (orch.Target 型を再利用)
    """orch.download_video の複製+修正版 (node24パス差し替え+リトライ強化)。

    差分は `_regen148_recollect_2026-08-18.py` / `_regen_model50_2026-08-19.py`
    と同一 (node v24 の js-runtime 指定 + リトライ5回 + 待機45秒)。Cookie は
    現セッションではマスターを持たないため付けない。
    """
    out_path = orch.FRAMES_DIR / t.video_filename
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, 0.0
    url = f"https://www.youtube.com/watch?v={t.video_id}"
    start = time.monotonic()
    for attempt in range(1 + _DL_RETRY_COUNT):
        cmd = [
            str(PROJECT_ROOT / "venv" / "bin" / "python"), "-m", "yt_dlp",
            "--ffmpeg-location", str(orch.FFMPEG_LOCATION),
            "--js-runtimes", f"node:{_NODE24_PATH}",
            "-f", orch.YT_DLP_FORMAT,
            "--remux-video", "mp4", "--no-playlist", "--no-progress",
            "-o", str(out_path), url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if out_path.exists() and out_path.stat().st_size > 0:
            return True, time.monotonic() - start
        for p in orch.FRAMES_DIR.glob(f"{t.video_filename}*part"):
            p.unlink(missing_ok=True)
        print(f"[dl][retry={attempt}] {t.target_id} 失敗: {proc.stderr[-300:]}", flush=True)
        if attempt < _DL_RETRY_COUNT:
            time.sleep(_DL_RETRY_SLEEP_SEC)
    return False, time.monotonic() - start


if __name__ == "__main__":
    base = PROJECT_ROOT / "data" / "verify" / "regen_2026-08-20_model50v2"
    orch.MANIFEST_TSV = base / "manifest.tsv"
    orch.STATUS_TSV = base / "status.tsv"
    orch.NEW_NPZ_DIR = (
        PROJECT_ROOT / "data" / "indicators_v2" / "boards_lean_model50v2_2026-08-20"
    )
    orch.LOG_DIR = PROJECT_ROOT / "logs" / "regen_model50v2_2026-08-20_per_video"
    orch.download_video = _download_video
    # 並列14が実測最適 (memory project_collect_indicators_v2_perf_2026-07-20、
    # 10->14 で1本あたりの時間はほぼ変わらずスループットだけ上がる)。
    # 並列数は環境変数で上書きできる (2026-08-20、飽和検証のため)。
    # 物理8コア(論理16)に14プロセスは詰め込みすぎで、1プロセスあたりの
    # 実効コアが0.57しかない。落とせば総スループットが上がるかを実測する。
    import os as _os
    orch.MAX_COLLECT_PARALLEL = int(_os.environ.get('COLLECT_PARALLEL', '14'))
    orch.TOTAL_HOLD_SLOTS = orch.MAX_COLLECT_PARALLEL + orch.DOWNLOAD_QUEUE_SIZE
    orch._HOLD_SEMAPHORE = threading.Semaphore(orch.TOTAL_HOLD_SLOTS)

    if "--smoke-check" in sys.argv:
        from src.production_config import collect_flags
        cf = collect_flags()
        print("[smoke] MANIFEST =", orch.MANIFEST_TSV)
        print("[smoke] NPZ_DIR  =", orch.NEW_NPZ_DIR)
        print("[smoke] 並列     =", orch.MAX_COLLECT_PARALLEL)
        print("[smoke] dl override =", orch.download_video is _download_video)
        for n in ("--enable-lockdown-score-numeric-release",
                  "--enable-lockdown-score-moving-release",
                  "--enable-boundary-newmatch-evidence",
                  "--enable-move-segmented-recording",
                  "--enable-physics-persistence-filter"):
            print(f"[smoke] {'OK ' if n in cf else 'NG '}{n}")
        raise SystemExit(0)

    raise SystemExit(orch.main())

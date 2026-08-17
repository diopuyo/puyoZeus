"""W12根治P2: forecast真値列が欠損している63本npzの再収集 (2026-08-16)。

`scripts/_regen148_orchestrator_2026-08-11.py` (148動画再生成パイプライン、
実績のある DL直列+収集並列+再開可能status.tsv 設計) をそのまま流用する。
差分はパス定数のみ:
- 対象: 63本 (`data/verify/w12_recollect_2026-08-16/targets.tsv`、
  `scripts/_gen_w12_recollect_targets_2026-08-16.py` で機械的に特定)
- 出力npz: `data/indicators_v2/boards_lean_w12_2026-08-16/` (既存85本の
  `boards_lean_phase_l_2026-08-11/` には一切触れない、上書き事故防止)
- status: `data/verify/w12_recollect_2026-08-16/status.tsv` (再開可能)
- ログ: `logs/regen_w12_2026-08-16_per_video/`

削除規約 (origin="preexisting" は絶対に削除しない)・幻盤面ガードON・
production_config.collect_flags() 経由の現行採用構成は元スクリプトの
ロジックをそのまま継承するため変更していない (import 後にモジュール
グローバルを上書きするだけ、ロジック本体はコピーしない)。
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ファイル名にハイフンを含む (`_regen148_orchestrator_2026-08-11.py`) ため
# 通常の import 文/`-m` 実行では読み込めない (無効な識別子)。実装コードを
# 複製せず同一ファイルをそのまま動的ロードして再利用する。
_ORCH_PATH = PROJECT_ROOT / "scripts" / "_regen148_orchestrator_2026-08-11.py"
_spec = importlib.util.spec_from_file_location("_regen148_orchestrator_impl", _ORCH_PATH)
orch = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
# dataclass の内部処理が sys.modules[cls.__module__] を参照するため、
# exec_module 前に登録しておく必要がある (動的ロードの既知の落とし穴)。
sys.modules[_spec.name] = orch
_spec.loader.exec_module(orch)

# 2026-08-16 追加修正: node v20.20.1 が yt-dlp-ejs の JS Challenge Provider に
# 「unsupported」判定される事態が新規発生 (2026-08-11時点では node v20.20.1 で
# 通っていたが、yt-dlp/YouTube側のJS課題要件が強化された模様、
# `--js-runtimes node:<path> -v` の debug 出力 `JS runtimes: node-20.20.1
# (unsupported)` で確認)。nvm で node v24.19.0 (LTS) を追加インストール済み。
# あわせて 403/500 の間欠的失敗 (同一URLで再試行すると成功する事例を確認済み)
# に対し DL_RETRY_COUNT を 1->4 に引き上げ、再試行間に短い待機を入れる。
_NODE24_PATH = "/home/ryouj/.nvm/versions/node/v24.19.0/bin/node"
_DL_RETRY_COUNT_W12 = 4
_DL_RETRY_SLEEP_SEC = 8.0


def _download_video_node24(t):  # noqa: ANN001, ANN201 (orch.Target 型を再利用)
    """orch.download_video の複製+修正版 (node24パス差し替え+リトライ強化)。

    差分: (1) --js-runtimes の node パスを v24.19.0 に変更 (2) リトライ回数を
    4 に引き上げ (3) 再試行間に `_DL_RETRY_SLEEP_SEC` 秒待機を追加 (間欠的な
    403/500 はCDNエッジの一時的な問題である可能性が高く、間隔を空けての
    再試行が有効なことを手動検証で確認済み)。それ以外のロジック
    (部分ファイル掃除・戻り値の型) は元関数と同一。
    """
    out_path = orch.FRAMES_DIR / t.video_filename
    if out_path.exists() and out_path.stat().st_size > 0:
        return True, 0.0
    url = f"https://www.youtube.com/watch?v={t.video_id}"
    start = time.monotonic()
    for attempt in range(1 + _DL_RETRY_COUNT_W12):
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
        if attempt < _DL_RETRY_COUNT_W12:
            time.sleep(_DL_RETRY_SLEEP_SEC)
    return False, time.monotonic() - start


if __name__ == "__main__":
    # パス定数のみ上書き (関数はモジュールグローバルを実行時に参照するため有効)。
    orch.MANIFEST_TSV = PROJECT_ROOT / "data" / "verify" / "w12_recollect_2026-08-16" / "targets.tsv"
    orch.STATUS_TSV = PROJECT_ROOT / "data" / "verify" / "w12_recollect_2026-08-16" / "status.tsv"
    orch.NEW_NPZ_DIR = PROJECT_ROOT / "data" / "indicators_v2" / "boards_lean_w12_2026-08-16"
    orch.LOG_DIR = PROJECT_ROOT / "logs" / "regen_w12_2026-08-16_per_video"
    orch.download_video = _download_video_node24
    raise SystemExit(orch.main())

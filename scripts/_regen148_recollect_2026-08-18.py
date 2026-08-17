"""148動画 全再収集 (2026-08-18 新本番構成、user承認済み)。

2026-08-18 に8フラグ全群 (W25会計整合フィルタ+固着対策/境界マルチシグナル系
b-1/b-2/③/stable-persistence-gate/winner-panel-crosscheck) が本採用され
`src/production_config.py` に登録済み (コミット01c3e37)。`collect_flags()` が
新構成を返すようになったため、148動画を新構成で再収集する。

`scripts/_regen148_orchestrator_2026-08-11.py` (148動画再生成パイプライン
本体、DL直列+収集並列+再開可能status.tsv、148/148 OK の実績あり) を
`scripts/_regen148_w12_recollect_2026-08-16.py` と同じパターンで流用する:
同一ファイルを動的ロードし、モジュールグローバル (パス定数・DL関数) だけを
上書きする。ロジック本体は一切コピーしない。

差分はパス定数のみ:
- 対象: `data/verify/regen_2026-08-11_manifest.tsv` の148本全部 (流用、
  今回は絞り込みなし)
- 出力npz: `data/indicators_v2/boards_lean_phase_l_2026-08-18/` (新規。
  既存の 2026-08-07 / 2026-08-11 / w12_2026-08-16 には一切触れない、
  上書き事故防止)
- status: `data/verify/regen_2026-08-18/status.tsv` (再開可能)
- ログ: `logs/regen_2026-08-18_per_video/`

削除規約 (origin="preexisting" は絶対に削除しない)・幻盤面ガードON・
production_config.collect_flags() 経由の現行採用構成 (8フラグ全群込み) は
元スクリプトのロジックをそのまま継承するため変更していない。

DL関数は w12 スクリプトと同じ node v24.19.0 + リトライ強化版を流用する
(node v20.20.1 が yt-dlp-ejs の JS Challenge Provider に unsupported 判定
される既知問題、2026-08-16 w12 再収集で node24+リトライ4回により86件中
安定してOK確認済み。144/148が origin="download" のため本問題は今回の
成否に直結する)。
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

# node v20.20.1 が yt-dlp-ejs の JS Challenge Provider に「unsupported」
# 判定される問題 (2026-08-16 w12 再収集で発見・node24へ切替済み) を継承する。
# あわせて 403/500 の間欠的失敗 (同一URLで再試行すると成功する事例を確認済み)
# に対し DL_RETRY_COUNT を 1->4 に引き上げ、再試行間に短い待機を入れる。
_NODE24_PATH = "/home/ryouj/.nvm/versions/node/v24.19.0/bin/node"
_DL_RETRY_COUNT = 4
_DL_RETRY_SLEEP_SEC = 8.0


def _download_video_node24(t):  # noqa: ANN001, ANN201 (orch.Target 型を再利用)
    """orch.download_video の複製+修正版 (node24パス差し替え+リトライ強化)。

    差分: (1) --js-runtimes の node パスを v24.19.0 に変更 (2) リトライ回数を
    4 に引き上げ (3) 再試行間に `_DL_RETRY_SLEEP_SEC` 秒待機を追加。
    それ以外のロジック (部分ファイル掃除・戻り値の型) は元関数と同一
    (`_regen148_w12_recollect_2026-08-16.py` と同一実装)。
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


def _print_config_smoke_check() -> None:
    """起動前スモーク: 上書き後のモジュールグローバルを目視確認する。

    パイプライン本体は起動しない (main Claude が別途起動する)。既存の
    2026-08-07/2026-08-11/w12_2026-08-16 npz ディレクトリと衝突していない
    ことをここで確認できる。
    """
    print("[smoke] MANIFEST_TSV =", orch.MANIFEST_TSV)
    print("[smoke] STATUS_TSV   =", orch.STATUS_TSV)
    print("[smoke] NEW_NPZ_DIR  =", orch.NEW_NPZ_DIR)
    print("[smoke] LOG_DIR      =", orch.LOG_DIR)
    print("[smoke] download_video overridden =", orch.download_video is _download_video_node24)


if __name__ == "__main__":
    # パス定数のみ上書き (関数はモジュールグローバルを実行時に参照するため有効)。
    orch.MANIFEST_TSV = PROJECT_ROOT / "data" / "verify" / "regen_2026-08-11_manifest.tsv"
    orch.STATUS_TSV = PROJECT_ROOT / "data" / "verify" / "regen_2026-08-18" / "status.tsv"
    orch.NEW_NPZ_DIR = PROJECT_ROOT / "data" / "indicators_v2" / "boards_lean_phase_l_2026-08-18"
    orch.LOG_DIR = PROJECT_ROOT / "logs" / "regen_2026-08-18_per_video"
    orch.download_video = _download_video_node24

    if "--smoke-check" in sys.argv:
        _print_config_smoke_check()
        raise SystemExit(0)

    raise SystemExit(orch.main())

"""video_30 を配信ソフトへ流す台の入力仕様 (定数と純関数だけ・副作用なし)。

流し直しの起動口 (run_live_x_video30.py) と突き合わせ (compare_live_video30_review.py) と試験が
同じ値を読む単一の置き場。値の根拠は <work>/review_v418_20261022 の manifest.json と
deliverables/verification.json (どちらも読み取りのみ)。

映像の渡し方:
- 入力は 1280x720・60fps のゲーム画面そのもの。配信ソフトの VideoFileSource は 1920x1080 でない画像を
  INTER_AREA で 1920x1080 へ拡大する (src/phase_j/live_source.py)。レビューの取り込みも同じ経路なので、
  切り抜き・色補正・時刻のずらしは要らない (21c の OBS 録画だけが 0.852 倍の切り抜きと 601→709 を要した)。
- 配信ソフトは 60fps を stride 2 (実効 30Hz) で処理する。レビューは stride 1 (60Hz) で取り込んでいる。
"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


from dataclasses import dataclass
from pathlib import Path

from src.fps_normalize import resolve_normalize_fps_30_stride

REVIEW_ROOT = Path(_public_dev_text('<work>/review_v418_20261022'))
VIDEO = REVIEW_ROOT/'source/video30_decode.mp4'
# 計算日 2026-10-22。元動画 (YouTube HtCsJm9uwyA) の先頭 284 秒を可逆 H.264 にした 17,040 コマのファイル。
VIDEO_SHA256 = 'b1e07ee2c0c818a73db1e7dfae875e04e04bc6cfd133e426ba275d4a578ef6e8'
VIDEO_FPS = 60.0
VIDEO_FRAMES = 17040
VIDEO_SIZE = (1280, 720)

REVIEW_CSV = REVIEW_ROOT/'deliverables/video30_frames.csv'
REVIEW_CSV_SHA256 = '1654f12f0c7e4503e73857e42ae5b8c7ec92513e80caa0e6c14680ae254f5fde'  # deliverables/sha256.json
REVIEW_FIRST_FRAME = 11040
REVIEW_FRAME_COUNT = 6000

# 取り込みの先頭 = manifest.json の capture_start。レビューの認識・較正・状態機械の履歴の起点と同じコマにする。
# 試合 1 の最初の手 187.93 秒の 12.9 秒前。レビューの取り込みでは検証 → 較正 (進捗 15→100%) が先頭の数コマで終わり、
# 試合境界 (game 0→1) は 184.7 秒ごろに出る。つまり較正も試合開始の検出も、この位置から十分に間に合う。
FIRST_PROCESSED_SEC = 175.0
END_SEC = 284.0                 # manifest.json の capture_end = ファイルの終端
WARMUP_SEC = 1.0                # src/phase_j/launcher.py の VIDEO_WARMUP_SEC (試験で一致を固定)


@dataclass(frozen=True)
class MatchRange:
    """レビュー動画の 1 試合。区間は両端を含む (verification.json の frames と一致)。"""

    index: int
    first_move_sec: float
    last_scoring_sec: float

    def frames(self, fps: float = VIDEO_FPS) -> tuple[int, int]:
        """(最初の手のコマ, 最後の得点のコマ)。"""
        return round(self.first_move_sec*fps), round(self.last_scoring_sec*fps)


MATCHES = (MatchRange(1, 187.93333333333334, 221.6), MatchRange(2, 232.76666666666668, 278.76666666666665))
EXPECTED_MATCH_FRAMES = {1: 2021, 2: 2761}   # verification.json の probabilities.matches[].frames


def stride_for(fps: float = VIDEO_FPS) -> int:
    """配信ソフトの入口が選ぶ間引き幅 (60fps なら 2)。"""
    return resolve_normalize_fps_30_stride(fps)


def first_frame(first_sec: float = FIRST_PROCESSED_SEC, fps: float = VIDEO_FPS) -> int:
    return round(first_sec*fps)


def live_start_sec(first_sec: float = FIRST_PROCESSED_SEC, fps: float = VIDEO_FPS,
                   warmup: float = WARMUP_SEC) -> float:
    """puyo_live.json の start_sec。配信ソフトは int((start_sec-warmup)*fps) コマ目から認識を始める。

    評価・公開が始まるのは int(start_sec*fps) コマ目から (live_bridge.observe の write_frame)。
    """
    start = first_sec+warmup
    if int(max(0., (start-warmup)*fps)) != first_frame(first_sec, fps):
        raise ValueError(f'start_sec={start} が先頭コマ {first_frame(first_sec, fps)} に対応しない (浮動小数の丸め)')
    return start


def write_frame(start_sec: float, fps: float = VIDEO_FPS) -> int:
    """評価・公開が始まるコマ (これより前は認識だけ)。"""
    return int(start_sec*fps)


def end_frame(end_sec: float = END_SEC, fps: float = VIDEO_FPS, total: int = VIDEO_FRAMES) -> int:
    """配信ソフトの走査の終端 (含まない)。generate() と同じく int(end_sec*fps) と総コマ数の小さい方。"""
    return min(total, int(end_sec*fps))


def processed_frames(first: int, end: int, stride: int) -> range:
    """配信ソフトが認識するコマ番号 (VideoFileSource は first から stride おきに end 未満)。"""
    return range(first, end, stride)


def workload(first_sec: float = FIRST_PROCESSED_SEC, end_sec: float = END_SEC,
             fps: float = VIDEO_FPS) -> dict[str, float | int]:
    """費用を決める単位: stride 後の処理コマ数 (復号コマ数ではない)。実時間条件の壁時計は映像の長さ。"""
    first, end, stride = first_frame(first_sec, fps), end_frame(end_sec, fps), stride_for(fps)
    processed = len(processed_frames(first, end, stride))
    return dict(first_frame=first, end_frame=end, stride=stride, processed_frames=processed,
                decoded_frames=end-first, video_seconds=(end-first)/fps,
                evaluated_from_frame=write_frame(live_start_sec(first_sec, fps), fps))

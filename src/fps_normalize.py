"""動画fpsを実効30fpsへ正規化するための stride 決定 (2026-07-30)。

## 背景
`board_state_machine.py` / `recognition_pipeline.py` の多くのフレーム数定数
(例: `STABLE_WARMUP_FRAMES=12`「0.4秒@30fps」、`STABLE_CNN_HISTORY_FRAMES=18`
「0.6秒 = 30fps×0.6s」、`GRAVITY_SETTLE_MIN_FRAMES=8`「@30fps 0.27秒」) は
「30fps で1フレーム進む = 実時間 1/30 秒」という前提でコメントされている。
60fps 動画をそのまま全フレーム処理すると、これらの定数が指す実時間が
設計意図の半分になってしまう (例: 0.4秒のつもりが0.2秒しか待たない)。

「60fps を 2フレームおきに処理する(実効30fps)」ことで、30fps 動画を
全フレーム処理するのと同じ時間解像度に揃えられる、という user 方針
(2026-07-30) に基づき、動画の実fpsから「実効30fpsになる stride」を
決定する純粋関数をここに定義する。

## stateless 原則 (CLAUDE.md)
本モジュールは状態を一切持たない純粋関数のみで構成する。呼び出し側
(collect_boards_lean.py / collect_indicators_v2.py 等) が既存の
`--sample-interval-frames` 引数にこの関数の戻り値をそのまま渡せる設計。

## 既知の副作用 (要確認事項、2026-07-30 記録)
`recognition_pipeline.py` の `BASELINE_BROKEN_CONSEC_FRAMES=60`(「1秒」と
コメント)は pipeline.update() の呼出し回数ベースのカウンタであり、
60fps ネイティブ実行を前提に「60回呼ばれたら1秒」という設計になっている
(他の大半の定数と異なり60fps前提)。本 stride (60fps→stride=2) を適用すると
呼出し回数が半減するため、この自己修復ガードが作動するまでの実時間が
1秒→2秒に伸びる (副作用、致命ではないが要 flag)。
"""
from __future__ import annotations

# 正規化先の目標fps。board_state_machine.py 等のフレーム数定数がこの fps を
# 前提にコメントされているため、これを基準に stride を決める (user方針2026-07-30)。
NORMALIZE_FPS_30_TARGET: float = 30.0

# stride の下限 (fps<=0 等の異常値や 30fps 以下の動画では間引かない = 1)。
MIN_NORMALIZE_STRIDE: int = 1


def resolve_normalize_fps_30_stride(fps: float) -> int:
    """動画の実fpsから「実効30fpsになる stride (フレーム間引き幅)」を返す。

    30fps なら 1 (間引きなし)、60fps なら 2、59.94fps (NTSC) も 2、
    120fps なら 4 のように `round(fps / NORMALIZE_FPS_30_TARGET)` を返す。
    30fps 未満の動画では stride が 0 または負になり得るため
    MIN_NORMALIZE_STRIDE (=1) に丸める (間引かない = 安全側)。

    Args:
        fps: 動画の実fps (例: cv2.VideoCapture.get(cv2.CAP_PROP_FPS))。
            0 以下の異常値は MIN_NORMALIZE_STRIDE にフォールバックする。

    Returns:
        stride。呼び出し側の `--sample-interval-frames` にそのまま渡せる。
    """
    if fps <= 0:
        return MIN_NORMALIZE_STRIDE
    stride = round(fps / NORMALIZE_FPS_30_TARGET)
    return max(MIN_NORMALIZE_STRIDE, stride)

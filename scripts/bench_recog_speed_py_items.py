"""live_recog_speed_py の項目ごとの「省ける 1 回あたりの費用」を、実コマで交互に測る台 (診断用)。

コマ全体の速度の ABA は、PC1 が他の作業 (OBS・別の再生) と取り合うと 1 本ごとに +-3〜20ms 揺れ、
項目の効果 (0.5〜2ms) を読めない。そこで、項目が省く処理そのものを、同じ process・同じ実コマで
「従来 / 新」を交互に何百回も回して測る (揺れは両方へ同じように乗り、中央値で外れ値を除く)。
1 コマあたりの削減 = 1 回あたりの差 x 1 コマあたりの回数 (回数は出力一致の再生の件数 STATS から)。

  python -B scripts/bench_recog_speed_py_items.py --app <候補 app> --out <json>
実行は配布 python (puyo_core 入り)。cwd は候補 app になる。1 本の process・直列・BelowNormal。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPEATS = 300           # 交互に回す回数 (従来・新を各 REPEATS 回)
WARM = 20               # 先頭の捨て回数 (キャッシュ・遅延 import)
FRAMES = 12             # 使う実コマ数 (連鎖中を含む窓から)
MS = 1000.0


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def alternate(old: Callable[[], Any], new: Callable[[], Any], repeats: int = REPEATS) -> dict[str, float]:
    """old と new を 1 回ずつ交互に呼び、各 repeats 回の時間 (ms) の平均・中央値・差を返す。"""
    old_times, new_times = [], []
    for index in range(repeats + WARM):
        started = time.perf_counter()
        old()
        middle = time.perf_counter()
        new()
        finished = time.perf_counter()
        if index >= WARM:
            old_times.append((middle - started) * MS)
            new_times.append((finished - middle) * MS)
    return dict(old_mean=sum(old_times) / len(old_times), new_mean=sum(new_times) / len(new_times),
                old_p50=percentile(old_times, .5), new_p50=percentile(new_times, .5),
                saved_mean=sum(old_times) / len(old_times) - sum(new_times) / len(new_times),
                saved_p50=percentile(old_times, .5) - percentile(new_times, .5), repeats=repeats)


def set_items(flag: bool, items: str = '') -> None:
    """旗と項目の環境変数を切り替える (src.recog_speed_py.enabled は呼ぶたびに環境を読む)。"""
    if flag:
        os.environ['PUYO_LIVE_RECOG_SPEED_PY'] = '1'
        os.environ['PUYO_LIVE_RECOG_SPEED_PY_ITEMS'] = items
    else:
        os.environ.pop('PUYO_LIVE_RECOG_SPEED_PY', None)
        os.environ.pop('PUYO_LIVE_RECOG_SPEED_PY_ITEMS', None)


def bench_hsv(session: Any, frames: list) -> dict[str, Any]:
    """read_board_hsv_only 1 回 / 使い回しの命中 1 回 (状態の照合・複製・窓への記録を含む)。"""
    from src.image_reader import DEFAULT_P1_REGION
    from src.recog_speed_py import read_hsv_only
    reader = session.pipe._reader
    reader._live_hsv_boards, reader._live_raw_board_sec = {}, 0.
    image = frames[0]
    set_items(True, 'hsv_memo')
    read_hsv_only(reader, image, DEFAULT_P1_REGION)           # 先に 1 回読み、以降は命中
    result = alternate(lambda: reader.read_board_hsv_only(image, DEFAULT_P1_REGION),
                       lambda: read_hsv_only(reader, image, DEFAULT_P1_REGION))
    set_items(False)
    return result


def bench_slide(frames: list) -> dict[str, Any]:
    """NextSlideDetector.update 1 回 (1 側)。連続したコマ列を、従来 / 前コマのグレー使い回しで交互に回す。"""
    from src.next_slide_detector import NextSlideDetector
    old_detector, new_detector = NextSlideDetector('1P'), NextSlideDetector('1P')
    pairs = list(zip(frames, frames[1:]))
    state = {'i': 0}

    def step(detector: Any, flag: bool) -> None:
        set_items(flag, 'slide_gray')
        prev, curr = pairs[state['i'] % len(pairs)]
        detector.update(prev, curr)
    old_times, new_times = [], []
    for index in range(REPEATS + WARM):
        state['i'] = index          # 連続した対 (前コマ = 前回の現コマ)。コマ列の端で 1 周するときだけ使い回せない
        started = time.perf_counter()
        step(old_detector, False)
        middle = time.perf_counter()
        step(new_detector, True)
        finished = time.perf_counter()
        if index >= WARM:
            old_times.append((middle - started) * MS)
            new_times.append((finished - middle) * MS)
    set_items(False)
    return dict(old_mean=sum(old_times) / len(old_times), new_mean=sum(new_times) / len(new_times),
                old_p50=percentile(old_times, .5), new_p50=percentile(new_times, .5),
                saved_mean=sum(old_times) / len(old_times) - sum(new_times) / len(new_times),
                saved_p50=percentile(old_times, .5) - percentile(new_times, .5), repeats=len(old_times))


def bench_images(frames: list) -> dict[str, Any]:
    """発火前画像リング: append 1 回 (全画面 copy / 盤面 2 領域の copy)・classify 1 回 (1 セルずつ / 一括)。"""
    from src.phase_j.live_obs_prepare import create_v49f_capture
    ring = create_v49f_capture().images
    image = frames[0]

    def append_old() -> None:
        set_items(False)
        ring.append(0., image)

    def append_new() -> None:
        set_items(True, 'image_crops')
        ring.append(0., image)
    appended = alternate(append_old, append_new)
    legacy = ring._classify_per_cell
    from src.image_reader import DEFAULT_P1_REGION
    classified = alternate(lambda: legacy(image, DEFAULT_P1_REGION, 0),
                           lambda: ring._classify_batched(image, DEFAULT_P1_REGION, 0), repeats=100)
    set_items(False)
    return dict(append=appended, classify_one_side=classified,
                native_median=bool(ring.classifiers[0]._enable_native_hsv))


def bench_scores(session: Any, frames: list) -> dict[str, Any]:
    """表示得点の再読 (read_side x 2) と、B3 の得点の読み直し (ocr.read)。共有時はどちらも 0。"""
    from src.exchange_event_overlay import displayed_scores_from_pipeline
    image = frames[0]
    owner = session.pipe
    capture = owner._b3_capture            # recognize() が作った実運用の B3 観測 (同じ ScoreOcr の設定・テンプレート)
    nothing: Callable[[], Any] = lambda: None   # noqa: E731
    return dict(display_reread=alternate(lambda: displayed_scores_from_pipeline(owner, image), nothing),
                b3_score_reread_per_read=alternate(lambda: capture.ocr.read(image), nothing))


def bench_pair(frames: list) -> dict[str, Any]:
    """終了の組の 4 領域: 毎コマの複製 / view だけ (複製は worker が 0.5 秒に 1 回)。"""
    from src.match_end_pair import PAIR_ROIS
    image = frames[0]
    return alternate(lambda: {k: image[y:y + h, x:x + w].copy() for k, (x, y, w, h) in PAIR_ROIS.items()},
                     lambda: {k: image[y:y + h, x:x + w] for k, (x, y, w, h) in PAIR_ROIS.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start', type=float, default=654.0, help='連鎖中を含む窓の開始秒 (d04 の 654〜660 秒は連鎖)')
    args = parser.parse_args()
    app = args.app.resolve()
    sys.path.insert(0, str(app))
    os.chdir(app)
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    import scripts.verify_recog_speed_py as gate
    out = args.out.parent / (args.out.stem + '_work')
    out.mkdir(parents=True, exist_ok=True)
    os.environ['PUYO_CALIBRATION_DIR'] = str(out / 'calibration')
    gate.distribution_env(out, [])
    from src.phase_j.dshow_color import correct_601_as_709
    frames = [correct_601_as_709(raw) for raw in gate.decode_frames(gate.DEFAULT_VIDEO, args.start, FRAMES)]
    session = gate.build_session(out)
    for position, image in enumerate(frames):         # 実コマで認識器を温める (分類器の較正・履歴を実運用に近づける)
        from src.phase_j import live_process
        from src.phase_j.live_source import CapturedFrame
        now = time.perf_counter()
        live_process.recognize(session, CapturedFrame(position, position / gate.FPS, now, now, image))
    results = dict(frames=len(frames), start=args.start, repeats=REPEATS,
                   hsv_only=bench_hsv(session, frames), slide_gray=bench_slide(frames),
                   images=bench_images(frames), scores=bench_scores(session, frames), pair_crops=bench_pair(frames))
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

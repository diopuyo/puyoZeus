"""117分動画のレンダが実時間12倍の主犯を確定する計装 (2026-08-21)。

方針 (指示準拠):
    - 本体 (scripts/visualize_advantage_overlay.py) は編集しない。
    - `vao.main()` を実際の CLI と同じ引数 (sys.argv 差し替え) で呼び、
      内部の主要関数をモジュール属性ごと monkeypatch してタイミングを取る
      (late-binding: Python は関数呼び出し時にモジュール名前空間を都度
      引くため、呼び出し前に `vao._score_advantage = wrapped` のように
      属性を差し替えれば呼び出し元コードは無変更のまま計装できる)。
    - cProfile は使わない。time.perf_counter のみ (memory
      project_speed_4to26fps_2026-07-31 の教訓4)。

計装対象 (5分解、受け入れ条件=総壁時間の85%以上を説明):
    1. 認識          RecognitionPipeline.update (クラスメソッド)
    2. 応手の計算     scripts.mc_counter_estimator.estimate_counter_distribution
                      (呼び出し回数・分布[中央値/p90/最大]も記録)
    3. 有利不利の判定 HeavyAdvCache.update (model_adv/threat/ukeyasusa/
                      saturated_chain_count を包含)
    4. パネル描画     _draw_panel_layout
    5. 動画の書き出し cv2.VideoWriter.write (クラスメソッド、グローバルpatch)

追加計装 (coordinator 17:55 追加指示):
    - 応手の呼び出しごとの所要分布 (中央値/p90/最大)
    - 応手の総呼び出し回数と 0.5秒間引き (COUNTER_RECOMPUTE_INTERVAL_SEC) の
      理論値との突合 (CounterReachTracker.update() 呼び出し前の内部状態
      [_last_t_sec/_last_budget_sec/_last_result] を読み、本体のソースコードの
      条件分岐をそのまま "予測" することで、呼び出しが (a) budget<=0 即0リターン
      (b) 時間間引きショートカット (c) 実評価 (cache参照) のどれになるかを
      呼び出し前に判定する。ロジックは vao.CounterReachTracker.update
      docstring 付近の条件式の写しであり、本体は変更しない)
    - 盤面キャッシュ (visualize_advantage_overlay.py:2401 の self._cache) の
      ヒット率 (dict サブクラスで __contains__ をカウントする非破壊 monkeypatch)

使い方 (A/Bは同一プロセス内で連続実行、間を空けない):
    python -m scripts._diag_render_breakdown_2026-08-21 \
        --sequence baseline,no-counter,positive-control
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

cv2.setNumThreads(1)

import scripts.mc_counter_estimator as mc_counter  # noqa: E402
import scripts.visualize_advantage_overlay as vao  # noqa: E402
import scripts.visualize_recognition as vrecog  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

VIDEO = "data/frames/video_zenchi_c0BQoMJwwQU.mp4"
MODEL_DIR = "data/verify/retrain_model62_2026-08-21"


def _loadavg() -> str:
    try:
        return Path("/proc/loadavg").read_text().strip()
    except OSError:
        return "N/A (proc/loadavgが無い環境)"


class _Bucket:
    """1計装対象の呼び出しごとの所要時間(ms)リストを溜める。"""

    def __init__(self) -> None:
        self.durations_ms: list[float] = []

    def add(self, dt_sec: float) -> None:
        self.durations_ms.append(dt_sec * 1000.0)

    @property
    def n(self) -> int:
        return len(self.durations_ms)

    @property
    def total_ms(self) -> float:
        return sum(self.durations_ms)

    def percentile(self, p: float) -> float:
        if not self.durations_ms:
            return float("nan")
        s = sorted(self.durations_ms)
        idx = min(len(s) - 1, int(len(s) * p))
        return s[idx]


class _CountingCache(dict):
    """CounterReachTracker._cache 差し替え用。__contains__ を hit/miss として計上する。

    dict の挙動そのものは変えない (単純に super() を呼ぶだけ)。
    """

    def __init__(self, stats: dict) -> None:
        super().__init__()
        self._stats = stats

    def __contains__(self, key) -> bool:  # noqa: ANN001
        present = super().__contains__(key)
        self._stats["hits" if present else "misses"] += 1
        return present


def _wrap_function(module, name: str, bucket: _Bucket, depth_counter: list[int] | None = None):
    orig = getattr(module, name)

    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        if depth_counter is not None:
            depth_counter[0] += 1
        try:
            r = orig(*args, **kwargs)
        finally:
            if depth_counter is not None:
                depth_counter[0] -= 1
        bucket.add(time.perf_counter() - t0)
        return r

    setattr(module, name, wrapped)
    return orig


def _wrap_method(cls, name: str, bucket: _Bucket, depth_counter: list[int] | None = None):
    """cls.name を計装する。depth_counter を渡すと呼び出し中は [0] を+1し、
    ネスト検知用に使える (例: RecognitionPipeline.update 内で呼ばれる
    cv2.resize を "ループ直下のresize" バケットから除外するため)。
    """
    orig = getattr(cls, name)

    def wrapped(self, *args, **kwargs):
        t0 = time.perf_counter()
        if depth_counter is not None:
            depth_counter[0] += 1
        try:
            r = orig(self, *args, **kwargs)
        finally:
            if depth_counter is not None:
                depth_counter[0] -= 1
        bucket.add(time.perf_counter() - t0)
        return r

    setattr(cls, name, wrapped)
    return orig


def _wrap_counter_tracker_update(throttle_stats: dict, cache_stats: dict):
    """CounterReachTracker.update の呼び出し前状態から分岐先を予測して集計する。

    本体 (vao.CounterReachTracker.update) のソースを読んだ条件式の写し
    (budget_transitioned / 時間間引き判定) を「呼ぶ前」に評価するだけで、
    本体の実行そのものには一切介入しない (副作用ゼロの観測)。
    """
    orig = vao.CounterReachTracker.update

    def wrapped(self, b1, b2, budget_sec=0.0, **kwargs):  # noqa: ANN001
        t_sec = kwargs.get("t_sec")
        defender_side = kwargs.get("defender_side")
        throttle_stats["total_calls"] += 1
        if defender_side is not None:
            throttle_stats["defender_only_calls"] += 1
            return orig(self, b1, b2, budget_sec, **kwargs)
        budget_transitioned = (budget_sec <= 0.0) != (self.last_budget_sec <= 0.0)
        if budget_sec <= 0.0:
            throttle_stats["zero_budget"] += 1
        elif (
            t_sec is not None and not budget_transitioned
            and self._last_result is not None and self._last_t_sec is not None
            and (t_sec - self._last_t_sec) < vao.COUNTER_RECOMPUTE_INTERVAL_SEC
        ):
            throttle_stats["time_throttled"] += 1
        else:
            throttle_stats["evaluated"] += 1
        return orig(self, b1, b2, budget_sec, **kwargs)

    vao.CounterReachTracker.update = wrapped
    return orig


def _wrap_cv2_resize_outside(depth_counter: list[int], bucket: _Bucket):
    """cv2.resize のうち、RecognitionPipeline.update の呼び出し中 (depth_counter[0]>0)
    ではないものだけを計上する (bucket1 との二重計上を避けるため)。
    generate() のループ本体で直接呼ばれる recog_frame/display_frame/raw_native の
    リサイズがここに入る。
    """
    orig = cv2.resize

    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        r = orig(*args, **kwargs)
        dt = time.perf_counter() - t0
        if depth_counter[0] == 0:
            bucket.add(dt)
        return r

    cv2.resize = wrapped
    return orig


def _wrap_cv2_capture_read(bucket: _Bucket):
    orig = cv2.VideoCapture.read

    def wrapped(self, *args, **kwargs):
        t0 = time.perf_counter()
        r = orig(self, *args, **kwargs)
        bucket.add(time.perf_counter() - t0)
        return r

    cv2.VideoCapture.read = wrapped
    return orig


def _wrap_counter_tracker_init(cache_stats: dict):
    """CounterReachTracker.__init__ 直後に self._cache を計数dictへ差し替える。"""
    orig = vao.CounterReachTracker.__init__

    def wrapped(self, *a, **kw):  # noqa: ANN001
        orig(self, *a, **kw)
        self._cache = _CountingCache(cache_stats)

    vao.CounterReachTracker.__init__ = wrapped
    return orig


def run(config: str, start_sec: float, end_sec: float, out_dir: Path,
        show_recognition: bool = True, cache_font: bool = False,
        tag: str = "") -> dict:
    """指定 config で generate() を実行し、内訳を計装・報告する。戻り値は集計dict。

    show_recognition: False で納品構成 (認識色overlayなし) を再現する
        (user指定「中央は元映像のみ」、coordinator依頼④)。
    cache_font: True で `_font()` を size 単位の lru_cache でラップする
        (coordinator依頼②「フォント描画キャッシュでどれだけ削れるか」の実測用。
        `_font` は size のみに依存する純関数 [同じ size は同じ ImageFont を返す]
        ため、キャッシュしても出力は bit-identical — 検証観点で安全)。
    """
    b_recog = _Bucket()
    b_counter = _Bucket()
    b_heavy_adv = _Bucket()
    b_panel = _Bucket()
    b_encode = _Bucket()
    b_recog_overlay = _Bucket()
    b_decode = _Bucket()
    b_loop_resize = _Bucket()
    recog_depth = [0]
    throttle_stats = {
        "total_calls": 0, "defender_only_calls": 0, "zero_budget": 0,
        "time_throttled": 0, "evaluated": 0,
    }
    cache_stats = {"hits": 0, "misses": 0}

    orig_update = _wrap_method(RecognitionPipeline, "update", b_recog, recog_depth)
    orig_resize = _wrap_cv2_resize_outside(recog_depth, b_loop_resize)
    orig_cap_read = _wrap_cv2_capture_read(b_decode)
    orig_estimate = _wrap_function(
        mc_counter, "estimate_counter_distribution", b_counter)
    orig_heavy = _wrap_method(vao.HeavyAdvCache, "update", b_heavy_adv)
    orig_panel = _wrap_function(vao, "_draw_panel_layout", b_panel)
    orig_write = _wrap_method(cv2.VideoWriter, "write", b_encode)
    orig_ct_update = _wrap_counter_tracker_update(throttle_stats, cache_stats)
    orig_ct_init = _wrap_counter_tracker_init(cache_stats)
    # --show-recognition (production probe と同条件) の認識色 overlay 描画。
    # generate() は毎回 `from scripts.visualize_recognition import
    # draw_cell_overlay as _draw_recog_cells` を実行時に評価する (関数内 import、
    # 呼び出しごとに再評価される) ため、vao.main() 呼び出し前にモジュール
    # 属性を差し替えれば効く。
    orig_draw_cells = _wrap_function(vrecog, "draw_cell_overlay", b_recog_overlay, recog_depth)
    orig_draw_state = _wrap_function(vrecog, "draw_state_label", b_recog_overlay, recog_depth)
    # generate() 冒頭 (フレームループ開始前) の一回限りセットアップ
    # (モデル読込・RecognitionPipeline.load_default 構築等) を分離計測する。
    # 短い窓ではこれが無視できないため (10秒窓で希釈率が低い)、
    # wall_total から除いた「ループ内訳」も別途出す。
    b_setup = _Bucket()
    orig_acquire_model = _wrap_function(vao, "_acquire_model", b_setup)

    # --- パネル描画サブ分解 (coordinator依頼②) ---
    # PIL の ImageDraw/ImageFont はクラスメソッドなので cv2.VideoWriter.write と
    # 同じ手法でグローバル計装できる。d.text/d.rectangle/d.line は
    # _draw_panel_layout 内でのみ呼ばれる想定 (他経路は --layout overlay 用の
    # _draw_overlay で、今回は --layout panel 固定なので混在しない)。
    b_font = _Bucket()
    b_text = _Bucket()
    b_shapes = _Bucket()
    b_canvas_io = _Bucket()  # Image.new/paste/fromarray/np.array + cv2.resize/cvtColor
    import PIL.ImageDraw as _pil_draw
    orig_text = _pil_draw.ImageDraw.text
    orig_rectangle = _pil_draw.ImageDraw.rectangle
    orig_line = _pil_draw.ImageDraw.line

    def _timed_text(self, *a, **kw):
        t0 = time.perf_counter()
        r = orig_text(self, *a, **kw)
        b_text.add(time.perf_counter() - t0)
        return r

    def _timed_rect(self, *a, **kw):
        t0 = time.perf_counter()
        r = orig_rectangle(self, *a, **kw)
        b_shapes.add(time.perf_counter() - t0)
        return r

    def _timed_line(self, *a, **kw):
        t0 = time.perf_counter()
        r = orig_line(self, *a, **kw)
        b_shapes.add(time.perf_counter() - t0)
        return r

    _pil_draw.ImageDraw.text = _timed_text
    _pil_draw.ImageDraw.rectangle = _timed_rect
    _pil_draw.ImageDraw.line = _timed_line

    # vao._font (module-level 関数、`ImageFont.truetype(path, size)` を毎回
    # 呼ぶだけの薄いラッパー) を計装する。cache_font=True 時は size 単位の
    # lru_cache 版に差し替える (`_font` は size のみに依存する純関数なので
    # キャッシュしても出力は bit-identical、coordinator依頼②の実測用A/B)。
    orig_vao_font = vao._font
    if cache_font:
        import functools as _functools

        @_functools.lru_cache(maxsize=None)
        def _font_cached(size):  # noqa: ANN001
            return orig_vao_font(size)

        def _font_wrapped(size):  # noqa: ANN001
            t0 = time.perf_counter()
            r = _font_cached(size)
            b_font.add(time.perf_counter() - t0)
            return r

        vao._font = _font_wrapped
    else:
        def _font_wrapped(size):  # noqa: ANN001
            t0 = time.perf_counter()
            r = orig_vao_font(size)
            b_font.add(time.perf_counter() - t0)
            return r

        vao._font = _font_wrapped

    orig_image_new = vao.Image.new
    orig_image_fromarray = vao.Image.fromarray
    orig_cvtColor = cv2.cvtColor

    def _timed_image_new(*a, **kw):
        t0 = time.perf_counter()
        r = orig_image_new(*a, **kw)
        b_canvas_io.add(time.perf_counter() - t0)
        return r

    def _timed_fromarray(*a, **kw):
        t0 = time.perf_counter()
        r = orig_image_fromarray(*a, **kw)
        b_canvas_io.add(time.perf_counter() - t0)
        return r

    def _timed_cvtColor(*a, **kw):
        t0 = time.perf_counter()
        r = orig_cvtColor(*a, **kw)
        dt = time.perf_counter() - t0
        # RecognitionPipeline.update / draw_cell_overlay / draw_state_label 内の
        # cvtColor (それぞれのバケットで既に計上済み) と二重計上しないよう、
        # recog_depth[0]==0 のときだけ (=_draw_panel_layout 内の2箇所) 計上する。
        if recog_depth[0] == 0:
            b_canvas_io.add(dt)
        return r

    vao.Image.new = _timed_image_new
    vao.Image.fromarray = _timed_fromarray
    cv2.cvtColor = _timed_cvtColor

    out_path = out_dir / f"diag_{config}{tag}_{int(start_sec)}_{int(end_sec)}.mp4"
    argv = [
        "visualize_advantage_overlay",
        "--video", VIDEO,
        "--start-sec", str(start_sec),
        "--end-sec", str(end_sec),
        "--layout", "panel",
        "--no-force-in-match",
        "--model-dir", MODEL_DIR,
        "--out", str(out_path),
    ]
    if show_recognition:
        argv.append("--show-recognition")
    if config == "no-counter":
        argv.append("--no-counter-reach")
    orig_n_rollouts = vao.COUNTER_N_ROLLOUTS
    if config == "positive-control":
        # 陽性対照: n_rollouts を倍にして「応手」区間だけが伸びることを確認する。
        vao.COUNTER_N_ROLLOUTS = orig_n_rollouts * 2

    print(f"\n[loadavg 開始前 config={config}] {_loadavg()}")
    old_argv = sys.argv
    sys.argv = argv
    t_wall0 = time.perf_counter()
    try:
        vao.main()
    finally:
        sys.argv = old_argv
        vao.COUNTER_N_ROLLOUTS = orig_n_rollouts
        RecognitionPipeline.update = orig_update
        mc_counter.estimate_counter_distribution = orig_estimate
        vao.HeavyAdvCache.update = orig_heavy
        vao._draw_panel_layout = orig_panel
        cv2.VideoWriter.write = orig_write
        vao.CounterReachTracker.update = orig_ct_update
        vao.CounterReachTracker.__init__ = orig_ct_init
        vrecog.draw_cell_overlay = orig_draw_cells
        vrecog.draw_state_label = orig_draw_state
        vao._acquire_model = orig_acquire_model
        cv2.resize = orig_resize
        cv2.VideoCapture.read = orig_cap_read
        _pil_draw.ImageDraw.text = orig_text
        _pil_draw.ImageDraw.rectangle = orig_rectangle
        _pil_draw.ImageDraw.line = orig_line
        vao._font = orig_vao_font
        vao.Image.new = orig_image_new
        vao.Image.fromarray = orig_image_fromarray
        cv2.cvtColor = orig_cvtColor
    wall_total = time.perf_counter() - t_wall0
    print(f"[loadavg 終了後 config={config}] {_loadavg()}")

    buckets = [
        ("1_認識(RecognitionPipeline.update)", b_recog),
        ("2_応手の計算(estimate_counter_distribution)", b_counter),
        ("3_有利不利の判定(HeavyAdvCache.update)", b_heavy_adv),
        ("4_パネル描画(_draw_panel_layout)", b_panel),
        ("5_動画書き出し(VideoWriter.write)", b_encode),
        ("6_認識色overlay描画(draw_cell/state)", b_recog_overlay),
        ("7_decode(cap.read)", b_decode),
        ("8_ループ直下resize(recog_frame/display_frame等)", b_loop_resize),
    ]
    print(f"\n{'=' * 88}\n[config={config}] window={end_sec - start_sec:.0f}s "
          f"({start_sec:.0f}-{end_sec:.0f}s) 壁時間={wall_total:.2f}s\n{'=' * 88}")
    explained = 0.0
    for label, bk in buckets:
        pct = bk.total_ms / (wall_total * 1000.0) * 100.0 if wall_total > 0 else 0.0
        explained += bk.total_ms
        avg = bk.total_ms / bk.n if bk.n else 0.0
        p90 = bk.percentile(0.90)
        mx = max(bk.durations_ms) if bk.durations_ms else float("nan")
        print(f"  {label:<44} 呼数={bk.n:6d} 合計={bk.total_ms/1000.0:8.2f}s "
              f"平均={avg:7.2f}ms p90={p90:7.2f}ms 最大={mx:8.2f}ms 割合={pct:5.1f}%")
    explained_pct = explained / (wall_total * 1000.0) * 100.0 if wall_total > 0 else 0.0
    print(f"  {'説明できた割合合計':<44} {'':>13} {'':>10} {'':>9} {'':>9} {'':>10} "
          f"割合={explained_pct:5.1f}%  (受け入れ条件 >=85%)")
    loop_wall = wall_total - b_setup.total_ms / 1000.0
    explained_loop_pct = (
        explained / (loop_wall * 1000.0) * 100.0 if loop_wall > 0 else 0.0
    )
    print(f"  一回限りセットアップ(_acquire_model等): {b_setup.total_ms/1000.0:.2f}s "
          f"(壁時間の{b_setup.total_ms/(wall_total*1000.0)*100.0 if wall_total else 0:.1f}%)")
    print(f"  ループ内訳のみで見た説明できた割合: {explained_loop_pct:.1f}% "
          f"(loop壁時間={loop_wall:.2f}s)")
    real_x = (end_sec - start_sec) / wall_total if wall_total > 0 else float("nan")
    print(f"  実時間倍率: {real_x:.4f}x "
          f"(この窓の{end_sec-start_sec:.0f}秒処理に壁時間{wall_total:.1f}秒 "
          f"= 実時間の{(1.0/real_x) if real_x else float('nan'):.2f}倍かかる)")

    panel_total_ms = b_panel.total_ms
    sub_buckets = [
        ("font_load(_font)", b_font),
        ("text描画(d.text)", b_text),
        ("図形(d.rectangle/d.line)", b_shapes),
        ("canvas変換(Image.new/fromarray/cvtColor)", b_canvas_io),
    ]
    print(f"\n  --- 4_パネル描画の内部分解 (合計{panel_total_ms/1000.0:.2f}s の"
          f"うちどれだけか、cache_font={cache_font}) ---")
    sub_explained = 0.0
    for label, bk in sub_buckets:
        sub_explained += bk.total_ms
        pct_of_panel = bk.total_ms / panel_total_ms * 100.0 if panel_total_ms > 0 else 0.0
        avg = bk.total_ms / bk.n if bk.n else 0.0
        print(f"    {label:<40} 呼数={bk.n:6d} 合計={bk.total_ms/1000.0:7.2f}s "
              f"平均={avg:6.3f}ms パネル内割合={pct_of_panel:5.1f}%")
    other_ms = panel_total_ms - sub_explained
    print(f"    {'その他(_draw_panel_layout内の残り)':<40} {'':>13} "
          f"合計={other_ms/1000.0:7.2f}s パネル内割合="
          f"{other_ms/panel_total_ms*100.0 if panel_total_ms else 0:5.1f}%")

    print(f"\n  --- 応手 (CounterReachTracker.update) 呼び出し分類 ---")
    print(f"  {throttle_stats}")
    total_side_queries = cache_stats["hits"] + cache_stats["misses"]
    hit_rate = (cache_stats["hits"] / total_side_queries * 100.0) if total_side_queries else float("nan")
    print(f"  盤面キャッシュ: hits={cache_stats['hits']} misses={cache_stats['misses']} "
          f"ヒット率={hit_rate:.1f}% (miss数はestimate_counter_distribution呼数と一致するはず: "
          f"miss={cache_stats['misses']} vs 応手呼数={b_counter.n})")

    return {
        "config": config, "wall_total": wall_total, "real_x": real_x,
        "buckets": {label: bk for label, bk in buckets},
        "throttle_stats": dict(throttle_stats),
        "cache_stats": dict(cache_stats),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default="baseline,no-counter,positive-control",
                     help="カンマ区切りで config を連続実行 (間を空けない)")
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float, default=60.0)
    ap.add_argument("--out-dir", type=Path,
                     default=Path("data/verify/zenchi_render_diag_2026-08-21"))
    ap.add_argument("--no-show-recognition", action="store_true", default=False,
                     help="納品構成 (認識色overlayなし) を再現する")
    ap.add_argument("--cache-font", action="store_true", default=False,
                     help="_font() を lru_cache 版に差し替えて計測する (A/B)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = args.sequence.split(",")
    results = []
    for i, c in enumerate(configs):
        tag = f"_run{i}" if configs.count(c) > 1 else ""
        results.append(run(
            c, args.start_sec, args.end_sec, args.out_dir,
            show_recognition=not args.no_show_recognition,
            cache_font=args.cache_font, tag=tag,
        ))

    print(f"\n{'=' * 88}\n[まとめ] window={args.end_sec - args.start_sec:.0f}s\n{'=' * 88}")
    for r in results:
        print(f"  config={r['config']:<18} 壁時間={r['wall_total']:7.2f}s "
              f"実時間倍率={r['real_x']:.4f}x")
    if len(results) >= 2:
        base = results[0]
        for r in results[1:]:
            diff = r["wall_total"] - base["wall_total"]
            print(f"  {base['config']} -> {r['config']}: 壁時間差 {diff:+.2f}s "
                  f"({diff / base['wall_total'] * 100.0:+.1f}%)")


if __name__ == "__main__":
    main()

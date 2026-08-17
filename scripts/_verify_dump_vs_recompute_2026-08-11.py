"""タイムラインdump工事の検証: 1動画で dump モード vs 再計算モードの D0 突合。

指令書 (2026-08-11 タイムラインdump工事) の検証手順:
  --dump-timeline + --no-render 実行 -> dump から走査 -> 再計算モードの結果と
  同一レコード時刻で検出が一致することを確認 (完全一致でなくてよい、D0 のみ突合)。

対象: data/evaluation_videos/v29_match2_156s.mp4 (v29 の2試合目、156秒、1本のみ)。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import threadpoolctl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.scan_judgment_anomalies import (  # noqa: E402
    detect_d0, make_score_fn, scan_video, scan_video_from_dump,
)
from scripts.visualize_advantage_overlay import _train_model, generate  # noqa: E402

VIDEO = Path("data/evaluation_videos/v29_match2_156s.mp4")
RECOMPUTE_NPZ = Path("data/verify/timeline_dump_verify_2026-08-11/v29m2.npz")
DUMP_NPZ = Path("data/verify/timeline_dump_verify_2026-08-11/v29m2_dump.npz")
MATCH_TOLERANCE_SEC = 1.0  # 突合の許容時間差 (2モードで再計算タイミングが異なるため)
# 検証用に処理秒数を絞る (2026-08-11 実測: Phase L 全域 regen 10並列走行中の
# severe CPU contention下で、settled 毎の _score_advantage() 呼び出しが
# near_future_fire_power(ビームサーチ)/fire_stability(ビームサーチ)/
# expected_fire(モンテカルロN=48) を含むため 1回 3.3秒 (無контеンション時)
# かかることが判明 (scan_judgment_anomalies.py モジュール docstring と同一の
# コスト要因)。フル156秒だと t=8.6s 到達までに1238秒かかり非現実的だったため、
# 60秒 (序盤ノイズ+最初の数回の settled 更新を含む) に短縮する。
DUMP_MAX_SEC = 60.0


def _d0_hits(records) -> list[tuple[float, bool]]:
    """(t_sec, D0検出有無) のリストを返す。"""
    return [(r.t_sec, detect_d0(r) is not None) for r in records]


def main() -> int:
    with threadpoolctl.threadpool_limits(limits=2):
        print("[1/3] dump 生成 (--dump-timeline + render=False)...", flush=True)
        t0 = time.time()
        generate(
            VIDEO, Path("data/verify/timeline_dump_verify_2026-08-11/_unused.mp4"),
            max_sec=DUMP_MAX_SEC, sample_interval=0.15, render=False,
            dump_timeline_path=DUMP_NPZ,
        )
        dump_dt = time.time() - t0
        print(f"  dump生成 所要時間: {dump_dt:.1f}s", flush=True)

        print("[2/3] 再計算モード (scan_video) ...", flush=True)
        t0 = time.time()
        model = _train_model(None)
        score_fn = make_score_fn(model)
        recompute_records = scan_video(RECOMPUTE_NPZ, score_fn)
        recompute_dt = time.time() - t0
        print(f"  再計算 所要時間(学習込): {recompute_dt:.1f}s "
              f"records={len(recompute_records)}", flush=True)

        print("[3/3] dump 読み出しモード (scan_video_from_dump) ...", flush=True)
        t0 = time.time()
        dump_records = scan_video_from_dump(DUMP_NPZ)
        dump_scan_dt = time.time() - t0
        print(f"  dump走査 所要時間: {dump_scan_dt:.2f}s "
              f"records={len(dump_records)}", flush=True)

    recompute_d0 = _d0_hits(recompute_records)
    dump_d0 = _d0_hits(dump_records)
    dump_times = np.array([t for t, _ in dump_d0])
    dump_flags = np.array([f for _, f in dump_d0])

    n_recompute_d0 = sum(1 for _, f in recompute_d0 if f)
    n_dump_d0 = sum(1 for f in dump_flags if f)
    matched, agree = 0, 0
    for t_r, flag_r in recompute_d0:
        if dump_times.size == 0:
            continue
        idx = int(np.argmin(np.abs(dump_times - t_r)))
        if abs(dump_times[idx] - t_r) <= MATCH_TOLERANCE_SEC:
            matched += 1
            if bool(dump_flags[idx]) == flag_r:
                agree += 1

    print("\n=== 結果 ===")
    print(f"recompute: records={len(recompute_records)} D0検出={n_recompute_d0}")
    print(f"dump     : records={len(dump_records)} D0検出={n_dump_d0}")
    print(f"突合(±{MATCH_TOLERANCE_SEC}s以内に相手レコードあり)={matched}/{len(recompute_d0)}")
    if matched > 0:
        print(f"D0判定一致率(突合できたレコードのうち)={agree}/{matched} "
              f"({100.0 * agree / matched:.1f}%)")
    print(f"\n所要時間: dump生成={dump_dt:.1f}s / 再計算(学習込)={recompute_dt:.1f}s "
          f"/ dump走査={dump_scan_dt:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

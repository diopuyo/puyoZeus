"""`--sample-interval 0` (間引きなし) の再レンダ所要時間への影響を、短区間の
実レンダ (render=True、実エンコード込み) で新旧2条件を壁時計で比較する
(2026-08-22 自己検収、cProfile不使用、指示通り短区間のみ)。

## 背景
sample_interval は writer.write() (実フレーム書き出し) を一切間引かない
(:4519 の step は _draw_graph が読む history の記録頻度のみを制御)。
sample-interval 0 では記録点が約4〜5倍に増え、_draw_graph が history 全点を
毎フレーム Python ループで描画するため、試合が長いほど1試合内の描画コストが
二次関数的に増える可能性がある (history は試合境界でクリアされるため
無制限には増えない)。実測で確かめる。

## 測定方法
同一の120秒区間 (試合境界を含まない typical な区間) を
sample_interval=0.15 (現行既定) と 0 (採用フラグ) の両方で render=True
(実エンコード込み) で処理し、壁時計を比較する。cv2.setNumThreads(1) で
本番と同じ1プロセス1コア設定に揃える。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
cv2.setNumThreads(1)

import scripts.visualize_advantage_overlay as vao  # noqa: E402

VIDEO = PROJECT_ROOT / "data/frames/video_zenchi_c0BQoMJwwQU.mp4"
MODEL_DIR = PROJECT_ROOT / "data/verify/retrain_model62_2026-08-21"
OUT_DIR = PROJECT_ROOT / "logs/_verify_sample_interval_cost_2026-08-22"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_SEC = 830.0
END_SEC = 950.0     # 120秒区間 (kill_override検収と同一区間、境界1回のみで
                    # リセットストーム無しを確認済み。当初 400-520s を試したが
                    # 411.6-414.7s 付近でスコアOCRが0付近で暴れるリセット
                    # ストーム区間に当たったため変更、2026-08-22)
WARMUP_SEC = 30.0


def _run(sample_interval: float) -> float:
    """指定 sample_interval で実レンダを1本回し、壁時計秒を返す。"""
    label = f"si{sample_interval}"
    t0 = time.perf_counter()
    vao.generate(
        video=VIDEO,
        out=OUT_DIR / f"_dummy_{label}.mp4",
        max_sec=0.0,
        sample_interval=sample_interval,
        start_sec=START_SEC,
        end_sec=END_SEC,
        warmup_sec=WARMUP_SEC,
        model_dir=MODEL_DIR,
        force_in_match=False,
        enable_early_fire_reaction=True,
        enable_resolved_exchange_eval=True,
        enable_resolved_decisive_amplify=True,
        enable_resolved_live_defender=True,
        enable_resolved_live_defender_strict=True,
        layout="panel",
        panel_subtitle_h=0,
        render=True,  # 実エンコード込み (グラフ描画コストを含めて測る)
        dump_timeline_path=None,
    )
    return time.perf_counter() - t0


def main() -> None:
    wall_015 = _run(0.15)
    print(f"[sample_interval=0.15 (現行既定)] wall={wall_015:.1f}s")
    wall_0 = _run(0.0)
    print(f"[sample_interval=0    (採用フラグ)] wall={wall_0:.1f}s")
    delta = wall_0 - wall_015
    pct = (delta / wall_015 * 100.0) if wall_015 > 0 else float("nan")
    print(f"[差分] {delta:+.1f}s ({pct:+.1f}%)")
    video_sec = END_SEC - START_SEC
    print(f"[参考] 処理区間={video_sec:.0f}動画秒 "
          f"(0.15: {video_sec / wall_015:.4f} 動画秒/壁秒, "
          f"0: {video_sec / wall_0:.4f} 動画秒/壁秒)")


if __name__ == "__main__":
    main()

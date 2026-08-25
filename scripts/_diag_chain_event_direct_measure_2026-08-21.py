"""`ChainEvent` (r.p1.chain_event/r.p2.chain_event) の実際の非None区間を
認識パイプラインを実際に走らせて直接記録する (2026-08-21、推定でなく実測)。

前段 (`_diag_mutual_exchange_frequency_v2_2026-08-21.py`) は npz の疎な観測
行 + 保持時間の式 (0.3×N 等) から区間を **推定**していた。本スクリプトは
`RecognitionPipeline.update()` の戻り値 `PipelineResult.p1/p2.chain_event` を
毎フレーム直接読み、以下を記録する:
  - 1P/2P それぞれの chain_event 非None区間 (開始/終了/連鎖数)
  - 両者が同時に非None だった区間
  - `chain_end_pending timeout` ログ (src.ojama_accounting, 別系統の
    お邪魔会計 settle 検知タイムアウト。ChainEvent の保持タイマーとは別の
    機構だが、あわせて出現回数・時刻を記録する)

対象: video_zenchi_c0BQoMJwwQU.mp4 の3区間 (0-300 / 3300-3600 / 6733-7033、
30先動画プローブ既定区間)。render=False (動画書き出しなし、計算のみ)。
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import scripts.visualize_advantage_overlay as vao
from src.recognition_pipeline import RecognitionPipeline

VIDEO = Path("data/frames/video_zenchi_c0BQoMJwwQU.mp4")
SEGMENTS = [(0.0, 300.0), (3300.0, 3600.0), (6733.0, 7033.0)]


@dataclass
class FrameSample:
    t_sec: float
    p1_active: bool
    p1_chain_count: "int | None"
    p1_trigger_sec: "float | None"
    p2_active: bool
    p2_chain_count: "int | None"
    p2_trigger_sec: "float | None"
    p1_next: "tuple[int, int] | None" = None
    p2_next: "tuple[int, int] | None" = None
    p1_state: "str | None" = None
    p2_state: "str | None" = None


@dataclass
class _Recorder:
    samples: list = field(default_factory=list)
    pending_timeout_events: list = field(default_factory=list)  # (side, t_sec相当なし, msg)


_recorder = _Recorder()
_orig_pipeline_update = RecognitionPipeline.update


def _patched_pipeline_update(self, frame_idx, time_sec, frame):
    result = _orig_pipeline_update(self, frame_idx, time_sec, frame)
    ev1 = result.p1.chain_event
    ev2 = result.p2.chain_event
    _recorder.samples.append(FrameSample(
        t_sec=time_sec,
        p1_active=ev1 is not None,
        p1_chain_count=(ev1.chain_count if ev1 is not None else None),
        p1_trigger_sec=(ev1.trigger_sec if ev1 is not None else None),
        p2_active=ev2 is not None,
        p2_chain_count=(ev2.chain_count if ev2 is not None else None),
        p2_trigger_sec=(ev2.trigger_sec if ev2 is not None else None),
        # 内部の「直近観測ネクスト」値 (_is_game_event_chain_exit が比較する
        # のと同じ属性) をそのまま記録する。実フレームでの目視確認と
        # 突き合わせるための客観的な裏付け (2026-08-21 追加)。
        p1_next=getattr(self, "_last_seen_next_1p", None),
        p2_next=getattr(self, "_last_seen_next_2p", None),
        p1_state=(result.p1.state.name if result.p1.state is not None else None),
        p2_state=(result.p2.state.name if result.p2.state is not None else None),
    ))
    return result


class _PendingTimeoutCaptureHandler(logging.Handler):
    """`chain_end_pending timeout` warning ログを捕捉するハンドラ。"""

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "chain_end_pending timeout" in msg:
            _recorder.pending_timeout_events.append(msg)


def _build_episodes(samples: list, side: str) -> list[dict]:
    """フレームサンプルから side の連鎖エピソード (実測区間) を作る。

    同一エピソード内で chain_event オブジェクトの識別 (trigger_sec が
    同一のまま続くか) を陽性対照として検証する (docstring 参照)。
    """
    episodes: list[dict] = []
    cur = None
    for s in samples:
        active = s.p1_active if side == "1P" else s.p2_active
        trig = s.p1_trigger_sec if side == "1P" else s.p2_trigger_sec
        cnt = s.p1_chain_count if side == "1P" else s.p2_chain_count
        if active:
            if cur is None or cur["trigger_sec"] != trig:
                if cur is not None:
                    episodes.append(cur)
                cur = {"trigger_sec": trig, "chain_count": cnt,
                       "start_sec": s.t_sec, "end_sec": s.t_sec, "n_frames": 0}
            cur["end_sec"] = s.t_sec
            cur["n_frames"] += 1
        else:
            if cur is not None:
                episodes.append(cur)
                cur = None
    if cur is not None:
        episodes.append(cur)
    return episodes


def _overlaps(ep1: list[dict], ep2: list[dict]) -> list[float]:
    durs = []
    for e1 in ep1:
        for e2 in ep2:
            lo = max(e1["start_sec"], e2["start_sec"])
            hi = min(e1["end_sec"], e2["end_sec"])
            if hi >= lo:
                durs.append(hi - lo)
    return durs


def _positive_control_identity_check(episodes: list[dict]) -> bool:
    """陽性対照: 記録された各エピソードが `n_frames >= 1` かつ
    `chain_count` が正の整数であることを確認する (実測データの健全性、
    「同一 trigger_sec を共有する連続フレームを1エピソードとして正しく
    束ねられているか」の直接検証)。"""
    if not episodes:
        return True  # 該当なしは失敗ではない (該当区間が実際に無かった場合)
    ok = all(e["n_frames"] >= 1 and e["chain_count"] is not None
             and e["chain_count"] > 0 for e in episodes)
    print(f"  [positive-control] エピソード健全性 (n_frames>=1, chain_count>0 "
          f"全件): {'OK' if ok else 'NG'} (n={len(episodes)})")
    return ok


def main() -> None:
    segments = SEGMENTS
    if len(sys.argv) >= 3:
        # 個別区間だけ再計測したいとき用 (フレーム突合の再実行コスト削減、
        # 2026-08-21 追加、既定の3区間セットは変更しない)。
        segments = [(float(sys.argv[1]), float(sys.argv[2]))]
    handler = _PendingTimeoutCaptureHandler()
    logging.getLogger("src.ojama_accounting").addHandler(handler)
    logging.getLogger("src.ojama_accounting").setLevel(logging.WARNING)
    RecognitionPipeline.update = _patched_pipeline_update

    all_chain_counts_p1: list[int] = []
    all_chain_counts_p2: list[int] = []
    all_overlap_durs: list[float] = []
    total_frames = 0

    for start_sec, end_sec in segments:
        _recorder.samples.clear()
        _recorder.pending_timeout_events.clear()
        t0 = time.perf_counter()
        written = vao.generate(
            VIDEO, Path("/tmp/_unused_chain_event_measure.mp4"), 0.0, 0.15,
            start_sec=start_sec, end_sec=end_sec, render=False,
        )
        elapsed = time.perf_counter() - t0
        ep1 = _build_episodes(_recorder.samples, "1P")
        ep2 = _build_episodes(_recorder.samples, "2P")
        overlaps = _overlaps(ep1, ep2)
        ok1 = _positive_control_identity_check(ep1)
        ok2 = _positive_control_identity_check(ep2)
        all_chain_counts_p1.extend(e["chain_count"] for e in ep1)
        all_chain_counts_p2.extend(e["chain_count"] for e in ep2)
        all_overlap_durs.extend(overlaps)
        total_frames += len(_recorder.samples)
        print(f"\n=== 区間 {start_sec:.0f}-{end_sec:.0f}s (written={written}, "
              f"elapsed={elapsed:.1f}s, control={'OK' if ok1 and ok2 else 'NG'}) ===")
        print(f"  1Pエピソード数={len(ep1)} 2Pエピソード数={len(ep2)} "
              f"同時区間数={len(overlaps)}")
        if overlaps:
            print(f"  同時区間の持続時間: {sorted(overlaps)}")
        # 実フレーム突合用に episode 一覧を JSON へ保存 (side/開始/終了/連鎖数)。
        dump_path = Path(
            f"data/verify/chain_event_episodes_{int(start_sec)}_{int(end_sec)}_2026-08-21.json")
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps({
            "1P": ep1, "2P": ep2, "overlap_durations_sec": overlaps,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [saved] {dump_path}")
        # 生フレームサンプル (t_sec/active/next値) も保存する。エピソード境界の
        # 直前直後で next_pair が実際に変化したかを事後突合するための一次データ
        # (2026-08-21 追加、誤検知件数の判定に使う)。
        raw_path = Path(
            f"data/verify/chain_event_raw_samples_{int(start_sec)}_{int(end_sec)}_2026-08-21.json")
        raw_path.write_text(json.dumps([
            {"t_sec": s.t_sec, "p1_active": s.p1_active, "p1_next": s.p1_next,
             "p2_active": s.p2_active, "p2_next": s.p2_next,
             "p1_state": s.p1_state, "p2_state": s.p2_state}
            for s in _recorder.samples
        ], ensure_ascii=False), encoding="utf-8")
        print(f"  [saved] {raw_path}")
        print(f"  chain_end_pending timeout ログ件数: "
              f"{len(_recorder.pending_timeout_events)}")
        for msg in _recorder.pending_timeout_events[:10]:
            print(f"    {msg}")

    print("\n=== 全区間まとめ (実測、推定なし) ===")
    print(f"総フレーム数: {total_frames}")
    print(f"同時区間の総数: {len(all_overlap_durs)}")
    if all_overlap_durs:
        import numpy as np
        arr = np.array(all_overlap_durs)
        print(f"同時区間の持続時間: min={arr.min():.2f} median={np.median(arr):.2f} "
              f"p90={np.percentile(arr, 90):.2f} max={arr.max():.2f}")
    print(f"1P連鎖数の分布: {sorted(all_chain_counts_p1)}")
    print(f"2P連鎖数の分布: {sorted(all_chain_counts_p2)}")
    if all_chain_counts_p1 or all_chain_counts_p2:
        import numpy as np
        combined = all_chain_counts_p1 + all_chain_counts_p2
        print(f"連鎖数(両side結合)の代表値: 中央値={np.median(combined):.1f} "
              f"平均={np.mean(combined):.2f} 最頻値の候補={sorted(set(combined))}")


if __name__ == "__main__":
    main()

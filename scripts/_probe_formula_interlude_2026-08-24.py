"""Q-01 修正後の偽イベント再測定プローブ (2026-08-24)。

## 何を確かめるための測定か

Q-02 の判定器が残した唯一のブロッカーは「重複候補の増加」だった。
段の進行を除いた重複 (FLICKER = 連鎖数も素点も同じままの再トリガー) が
3 窓すべてで増えている:

    w1  10.6% → 15.6%   w2  18.2% → 25.8%   c62  44.0% → 48.1%
    (重複/物理連鎖でも 0.13→0.21 / 0.28→0.43 / 0.94→1.34 と増加)

**仮説**: この FLICKER 増加は Q-01 の症状そのものではないか。
`formula_read` イベントの `total_score` は段累積器の `total_power` から来るので、
Q-01 のバグ (右辺が減ると段を落とす / セッション破棄) で段が進まなければ
**同じ (連鎖数, 素点) が繰り返し出る** → FLICKER として数えられる。

Q-01 を直して段が正しく進むなら、それらは STEP_PROGRESS に変わり
FLICKER は減るはずである。この測定はその検証。

## 既存プローブとの違い

`scripts/_probe_formula_false_event_2026-08-24.py` (2026-08-24 先行測定) に対し

  - `enable_formula_step_interlude=True` を ON 構成に追加した
  - **出力先を別ディレクトリにした** (既存の測定結果を上書きしない)

以外は同一。OFF 構成・窓・動画・その他フラグはすべて先行測定と揃える
(`feedback_paired_comparison_fixed_population_2026-08-20`: 母集団を変えない)。

使い方:
  python scripts/_probe_formula_interlude_2026-08-24.py <mode> <t0> <t1> <tag>
出力: logs/_probe_formula_interlude_2026-08-24/probe_<tag>_<mode>.log
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

cv2.setNumThreads(1)

import scripts.visualize_advantage_overlay as vao  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

MODE = sys.argv[1]
assert MODE in ("off", "on"), MODE
T0 = float(sys.argv[2])
T1 = float(sys.argv[3])
TAG = sys.argv[4]
WARMUP = 30.0
VIDEO = PROJECT_ROOT / "data/frames/video_zenchi_c0BQoMJwwQU.mp4"
MODEL_DIR = PROJECT_ROOT / "data/verify/retrain_model62_2026-08-21"
OUT_DIR = PROJECT_ROOT / "logs/_probe_formula_interlude_2026-08-24"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 先行測定の 3 フラグに、Q-01 修正の幕間フラグを足したもの。
NEW_FLAGS = dict(
    enable_chain_formula_read_verify=True,
    enable_formula_chain_count_update=True,
    enable_slide_exit_no_min_display=True,
    enable_formula_step_interlude=True,
)

_orig_load_default = RecognitionPipeline.load_default.__func__


def _patched_load_default(cls, *args, **kwargs):
    if MODE == "on":
        kwargs.update(NEW_FLAGS)
    return _orig_load_default(cls, *args, **kwargs)


_orig_update = RecognitionPipeline.update
_last = {"ev1": None, "ev2": None, "s1": None, "s2": None}


def _patched_update(self, frame_idx, time_sec, frame):
    r = _orig_update(self, frame_idx, time_sec, frame)
    for key, ev in (("ev1", self._active_chain_1p),
                    ("ev2", self._active_chain_2p)):
        cur = None if ev is None else (
            round(ev.trigger_sec, 2), ev.mechanism, ev.chain_count,
            ev.total_score,
        )
        if _last[key] != cur:
            print(f"[ev] t={time_sec:.3f} {key} {cur}", flush=True)
            _last[key] = cur
    s1 = (self._score_tracker_1p.last_score
          if self._score_tracker_1p is not None else None)
    s2 = (self._score_tracker_2p.last_score
          if self._score_tracker_2p is not None else None)
    if _last["s1"] != s1:
        print(f"[score] t={time_sec:.3f} 1P {s1}", flush=True)
        _last["s1"] = s1
    if _last["s2"] != s2:
        print(f"[score] t={time_sec:.3f} 2P {s2}", flush=True)
        _last["s2"] = s2
    return r


def main() -> None:
    RecognitionPipeline.load_default = classmethod(_patched_load_default)
    RecognitionPipeline.update = _patched_update
    try:
        vao.generate(
            video=VIDEO, out=OUT_DIR / f"_dummy_{TAG}_{MODE}.mp4",
            max_sec=0.0, sample_interval=0.0,
            start_sec=max(0.0, T0 - WARMUP), end_sec=T1, warmup_sec=WARMUP,
            model_dir=MODEL_DIR, layout="panel", panel_subtitle_h=0,
            render=False, dump_timeline_path=None,
            enable_early_fire_reaction=True,
            enable_per_side_settled=True,
            disable_score_lead_bias=True,
            disable_pressure=True,
            enable_counter_reach=False,
            normalize_fps_30=True,
            use_production_recognition=True,
            resize_1080p=True,
            enable_resolved_live_defender_strict=True,
            enable_resolved_kill_override=True,
            enable_resolved_exchange_eval=True,
            enable_resolved_decisive_amplify=True,
            enable_resolved_live_defender=True,
            enable_slide_exit_min_display_guard=True,
            force_in_match=False,
        )
    finally:
        RecognitionPipeline.load_default = classmethod(_orig_load_default)
        RecognitionPipeline.update = _orig_update


if __name__ == "__main__":
    main()

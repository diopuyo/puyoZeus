"""Gate 3R: 本番認識構成でcap前gross会計を検収するv5プローブ。

既存v4を上書きせず、認識条件だけ本番構成へ揃え、pending純差分の代わりに
`GrossOjamaCounters`のカテゴリ別差分をExchangeEpisodeTrackerへ供給する。
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import cv2

from src.exchange_episode_tracker import classify_gross_counter_delta
from src.ojama_accounting import GrossOjamaCounters, OjamaAccountingTracker
from src.production_config import recognition_load_default_kwargs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V4_PATH = PROJECT_ROOT / "scripts/_gate3_episode_probe_v4_2026-08-25.py"
DEFAULT_VIDEO = PROJECT_ROOT / "data/frames/video_zenchi_c0BQoMJwwQU.mp4"
DEFAULT_OUT = PROJECT_ROOT / "data/verify/gate3_episode_v5_gross_2026-08-25/zenchi_exact"
DEFAULT_WARMUP_SEC: float = 60.0
NORMALIZED_FPS: float = 30.0


@dataclasses.dataclass
class _GrossAudit:
    """gross差分の実測集計。0には必ず検査side数を併記する。"""

    frame_count: int = 0
    inspected_side_count: int = 0
    nonzero_residual_side_count: int = 0
    residual_abs_total: float = 0.0
    residual_abs_max: float = 0.0
    boundary_wiped_total: int = 0
    clamp_loss_total: int = 0


def _load_v4() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gate3_probe_v4", V4_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"v4 probeをロードできません: {V4_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--t0", type=float, default=780.0)
    parser.add_argument("--t1", type=float, default=1080.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--warmup-sec", type=float, default=DEFAULT_WARMUP_SEC)
    parser.add_argument(
        "--no-normalize-fps-30", action="store_true",
        help="60fps動画を間引かず全フレーム処理する（比較用途）",
    )
    return parser.parse_args()


def _record_result(audit: _GrossAudit, result: Any) -> None:
    """1フレームの保存則残差・wipe・clampを母数付きで集計する。"""
    residuals = (result.conservation_residual_p1, result.conservation_residual_p2)
    audit.frame_count += 1
    audit.inspected_side_count += result.inspected_side_count
    audit.nonzero_residual_side_count += sum(abs(value) > 1e-9 for value in residuals)
    audit.residual_abs_total += sum(abs(value) for value in residuals)
    audit.residual_abs_max = max(audit.residual_abs_max, *(abs(v) for v in residuals))
    audit.boundary_wiped_total += (
        result.boundary_wiped_on_1p + result.boundary_wiped_on_2p
    )
    audit.clamp_loss_total += result.clamp_loss_on_1p + result.clamp_loss_on_2p


class _StrideCapture:
    """v4を変更せず、元動画を実効30fpsとして順次デコードする。"""

    def __init__(self, path: str, stride: int) -> None:
        self._cap = cv2.VideoCapture(path)
        self._stride = stride
        self._first_read = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cap, name)

    def get(self, prop_id: int) -> float:
        value = self._cap.get(prop_id)
        if prop_id in (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT,
                       cv2.CAP_PROP_POS_FRAMES):
            return value / self._stride
        return value

    def set(self, prop_id: int, value: float) -> bool:
        if prop_id == cv2.CAP_PROP_POS_FRAMES:
            self._first_read = True
            value *= self._stride
        return self._cap.set(prop_id, value)

    def read(self) -> tuple[bool, Any]:
        if not self._first_read:
            for _ in range(self._stride - 1):
                ok, _frame = self._cap.read()
                if not ok:
                    return False, None
        self._first_read = False
        return self._cap.read()


class _Cv2Proxy:
    """VideoCaptureだけをstride対応へ差し替えるcv2透過proxy。"""

    def __init__(self, stride: int) -> None:
        self._stride = stride

    def __getattr__(self, name: str) -> Any:
        return getattr(cv2, name)

    def VideoCapture(self, path: str) -> _StrideCapture:  # noqa: N802
        return _StrideCapture(path, self._stride)


def _install_production_config(module: ModuleType, stride: int) -> None:
    """v4のload_default呼出を、本番採用フラグかつforce_in_match=Falseへ揃える。"""
    original = module.RecognitionPipeline.load_default.__func__

    def load_default(cls: type, *args: Any, **kwargs: Any) -> Any:
        merged = recognition_load_default_kwargs()
        merged.update(kwargs)
        merged["force_in_match"] = False
        merged["enable_slide_exit_min_display_guard"] = True
        pipeline = original(cls, *args, **merged)
        if stride > 1:
            original_update = pipeline.update

            def update(frame_idx: int, time_sec: float, frame: Any) -> Any:
                return original_update(frame_idx * stride, time_sec, frame)

            pipeline.update = update
        return pipeline

    module.RecognitionPipeline.load_default = classmethod(load_default)


def _install_gross_supply(module: ModuleType, audit: _GrossAudit) -> None:
    """v4の旧pending差分分類だけをgross累積差分へ置換する。"""
    state: dict[str, Any] = {"tracker": None, "counters": {}}
    original_make_frame = module._make_pending_frame

    class RecordingTracker(OjamaAccountingTracker):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            state["tracker"] = self

    def make_frame(*args: Any, **kwargs: Any) -> Any:
        frame = original_make_frame(*args, **kwargs)
        tracker = state["tracker"]
        if tracker is None:
            raise RuntimeError("OjamaAccountingTrackerが未初期化です")
        state["counters"][frame.t_sec] = tracker.get_gross_counters(frame.t_sec)
        return frame

    def classify(prev: Any, curr: Any) -> Any:
        counters: dict[float, GrossOjamaCounters] = state["counters"]
        result = classify_gross_counter_delta(
            counters[prev.t_sec], counters[curr.t_sec],
            (prev.p1_uncapped, prev.p2_uncapped),
            (curr.p1_uncapped, curr.p2_uncapped), curr.game_idx,
        )
        _record_result(audit, result)
        return SimpleNamespace(
            settlement=result.settlement, wiped_sides=result.wiped_sides,
            unclassified_drop_p1=abs(result.conservation_residual_p1),
            unclassified_drop_p2=abs(result.conservation_residual_p2),
        )

    module.OjamaAccountingTracker = RecordingTracker
    module._make_pending_frame = make_frame
    module.classify_pending_uncapped_delta = classify


def main() -> None:
    args = _parse_args()
    module = _load_v4()
    audit = _GrossAudit()
    module.WARMUP_SEC = args.warmup_sec
    fps_probe = cv2.VideoCapture(str(args.video))
    source_fps = fps_probe.get(cv2.CAP_PROP_FPS)
    fps_probe.release()
    stride = max(1, round(source_fps / NORMALIZED_FPS))
    if args.no_normalize_fps_30:
        stride = 1
    if stride > 1:
        module.cv2 = _Cv2Proxy(stride)
    _install_production_config(module, stride)
    _install_gross_supply(module, audit)
    module._run_probe(args.video, args.t0, args.t1, args.out_dir)
    summary_path = args.out_dir / "gross_validation.json"
    summary_path.write_text(
        json.dumps(dataclasses.asdict(audit), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[gross saved] {summary_path}", flush=True)
    print(json.dumps(dataclasses.asdict(audit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""指摘16調査: bg_fp (背景フィンガープリント) 採取タイミングで
実際に何が写っていたかを計装する (計装専用、本番コード src/ は無変更)。

仮説: cycle 71h の「試合開始2秒間は puyo 数上限を実質無制限 (144) に緩和して
bg_fp 採取を強制する」ロジック (recognition_pipeline.py:3630-3646) が、
試合開始直後に既に設置済みの本物のぷよを「背景」として焼き込んでしまい、
そのセル位置で後続フレームの同色ぷよが NCC>=0.92 で背景と一致 → tier1 で
無条件 EMPTY 化される、という機構を検証する。

やること:
  1. src.background_fingerprint.capture_patch_pair_robust をラップし、
     呼び出しごとに (呼び出し時刻, 使われた5フレームの生画像) を記録する。
  2. RecognitionPipeline を実データ (review_demo_2026-08-12.mp4, 試合3開始
     付近) に対して走らせ、採取された bg フレームの P1/P2 該当セル領域を
     切り出して PNG 保存する (実画面証拠)。
  3. 採取フレームに実ぷよが写っているかを目視確認できるようにする。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import src.background_fingerprint as bgfp_module
from src.fps_normalize import resolve_normalize_fps_30_stride
from src.production_config import recognition_load_default_kwargs
from src.recognition_pipeline import RecognitionPipeline

VIDEO = Path("data/frames/review_demo_2026-08-12.mp4")
OUT_DIR = Path("data/verify/diag_issue16_2026-08-15/bgfp_capture")

# デモ生成スクリプト (_gen_demo_final3_2026-08-15.sh) が使った追加3フラグ。
DEMO_EXTRA_KWARGS = {
    "stable_majority_window": True,
    "enable_ojama_fall_entry_hardening": True,
    "enable_ojama_fall_scoped_exit": True,
}

_capture_log: list[dict] = []


def _wrap_capture_patch_pair_robust(orig_fn):
    """capture_patch_pair_robust をラップし、呼出し時の frames_list を記録する。"""

    def wrapped(frames_list, p1_rect, p2_rect):
        record = {
            "call_index": len(_capture_log),
            "n_frames": len(frames_list),
            "p1_rect": p1_rect,
            "p2_rect": p2_rect,
        }
        _capture_log.append(record)
        # 先頭フレームを保存 (5フレームバッファの実データ確認用)
        idx = record["call_index"]
        out_path = OUT_DIR / f"bgfp_call{idx:02d}_frame0.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), frames_list[0])
        # 最終フレームも保存 (5フレーム間でどう変化したか)
        out_path2 = OUT_DIR / f"bgfp_call{idx:02d}_frame4.png"
        cv2.imwrite(str(out_path2), frames_list[-1])
        return orig_fn(frames_list, p1_rect, p2_rect)

    return wrapped


def build_pipeline(use_demo_extra: bool) -> RecognitionPipeline:
    """demo 構成 (追加3フラグ込み) または本番採用構成のみで pipe を構築する。"""
    kwargs = dict(recognition_load_default_kwargs())
    if use_demo_extra:
        kwargs.update(DEMO_EXTRA_KWARGS)
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=False,
        **kwargs,
    )
    return pipe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-sec", type=float, default=95.0,
                    help="処理開始 (絶対秒、試合3開始より十分前から)")
    ap.add_argument("--end-sec", type=float, default=130.0)
    ap.add_argument("--demo-extra", action="store_true",
                    help="デモ限定3フラグ (stable_majority_window等) を追加する")
    ap.add_argument("--log-out", type=Path,
                    default=OUT_DIR / "bgfp_capture_log.json")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _capture_log.clear()

    # モンキーパッチ (計装専用、src/ ファイル自体は無変更)
    orig = bgfp_module.capture_patch_pair_robust
    bgfp_module.capture_patch_pair_robust = _wrap_capture_patch_pair_robust(orig)
    # recognition_pipeline.py は capture_patch_pair_robust をモジュール内で
    # `from src.background_fingerprint import capture_patch_pair_robust` として
    # import 済 (トップレベル) のため、そちらの参照も差し替える必要がある。
    import src.recognition_pipeline as rp_module
    rp_module.capture_patch_pair_robust = bgfp_module.capture_patch_pair_robust

    try:
        pipe = build_pipeline(args.demo_extra)
        cap = cv2.VideoCapture(str(VIDEO))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = resolve_normalize_fps_30_stride(fps)
        start_frame = int(args.start_sec * fps)
        end_frame = int(args.end_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        events: list[dict] = []
        for fi in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if (fi - start_frame) % stride != 0:
                continue
            t = fi / fps
            recog_frame = (
                frame if frame.shape[:2] == (1080, 1920)
                else cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
            )
            r = pipe.update(fi, t, recog_frame)
            events.append({
                "t": round(t, 3), "fi": fi,
                "score1": r.p1.score, "score2": r.p2.score,
                "state1": str(r.p1.state), "state2": str(r.p2.state),
                "n_bgfp_calls_so_far": len(_capture_log),
            })
        cap.release()

        # bg_fp が実際に採取された時刻を events から逆算 (n_bgfp_calls_so_far の増加点)
        capture_times: list[float] = []
        prev_n = 0
        for e in events:
            if e["n_bgfp_calls_so_far"] > prev_n:
                capture_times.append(e["t"])
                prev_n = e["n_bgfp_calls_so_far"]

        summary = {
            "demo_extra": args.demo_extra,
            "n_bgfp_calls": len(_capture_log),
            "bgfp_call_capture_times_sec": capture_times,
            "capture_log": _capture_log,
            "events_sample": events[:20],
        }
        args.log_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        print(f"bg_fp capture 回数: {len(_capture_log)}")
        print(f"capture 時刻(秒): {capture_times}")
        print(f"ログ保存先: {args.log_out}")
    finally:
        bgfp_module.capture_patch_pair_robust = orig
        rp_module.capture_patch_pair_robust = orig


if __name__ == "__main__":
    main()

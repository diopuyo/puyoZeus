"""配布物から「ゲーム画面の切り出し画像 (models/ui_templates)」を外せるかの A/B 検証道具 (2026-09-30)。

user 決定 (2026-09-30): ゲーム画面の切り出し画像は再配布しない。実行時に本当に要るものだけ暫定で残す。
サブコマンド (全て読み取り専用。src/ は変更しない。低優先度・単一 process で短区間だけ回す):

  scan-telop  動画を粗い間隔で走査し、TelopDetector が可視と判定する区間を探す (A/B に使える窓の発見)
  zero        score_zero を「画像テンプレート」と「ScoreOcr の 8 桁全て 0」で同一フレームに対し比較
  zero-ab     score_zero をテンプレート/ScoreOcr にして認識パイプラインを回し確定盤面を全セル比較
  telop-ab    テロップ検出あり/なし (テンプレート不在) で認識パイプラインを回し確定盤面を全セル比較
  imread-trace パイプライン生成+短区間で cv2.imread された models/ui_templates 配下のファイルを列挙
              (packaging/audit_sitecustomize.py は Python の open() しか見ない。cv2.imread は C++ 側の
               ファイル読みで記録されないため、その監査だけでは実行時使用を判定できない)

実行は資産のある作業ディレクトリ (models/ を持つ本体) で:
    cd <asset_root> && python <この worktree>/scripts/ab_ui_template_removal_20260930.py zero ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

FRAME_SIZE = (1920, 1080)
TEMPLATE_DIR_MARK = "ui_templates"
SEED = 20260930
SCAN_STRIDE_SEC_DEFAULT = 5.0
NATIVE_FPS_STRIDE_THRESHOLD = 45.0  # これ以上の fps は本番と同じく 1/2 に間引く
STATE_IN_MATCH = "in_match"
CATEGORY_IN, CATEGORY_PRE, CATEGORY_POST, CATEGORY_NONE = "in_match", "pre_match_intro", "result_screen", "no_match_seen"


def open_video(path: str) -> tuple[Any, float]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise SystemExit(f"動画を開けない: {path}")
    return capture, float(capture.get(cv2.CAP_PROP_FPS) or 30.0)


def to_1080p(frame: np.ndarray) -> np.ndarray:
    if frame.shape[:2] != FRAME_SIZE[::-1]:
        return cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    return frame


def read_window(path: str, start_sec: float, end_sec: float, stride: int = 1):
    """[start,end) を順次読みし (フレーム番号, 秒, 1080p フレーム) を返す。"""
    capture, fps = open_video(path)
    first = int(round(start_sec * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    index = first
    try:
        while index / fps < end_sec:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if (index - first) % stride == 0:
                yield index, index / fps, to_1080p(frame)
            index += 1
    finally:
        capture.release()


# ---------------------------------------------------------------- scan-telop
def scan_telop(args: argparse.Namespace) -> None:
    from src.telop_detector import TelopDetector
    detector = TelopDetector.load_default()
    report: dict[str, Any] = {"templates": sorted(detector._templates), "videos": {}}
    for path in args.videos:
        capture, fps = open_video(path)
        duration = float(capture.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
        visible: list[float] = []
        best = 0.0
        samples = 0
        for sec in np.arange(0.0, duration, args.stride_sec):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(sec) * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            samples += 1
            result = detector.detect(to_1080p(frame))
            best = max(best, result.score)
            if result.is_visible:
                visible.append(round(float(sec), 1))
        capture.release()
        report["videos"][Path(path).name] = dict(duration_sec=round(duration, 1), samples=samples,
                                                 visible_samples=len(visible), visible_secs=visible, best_score=round(best, 3))
        print(Path(path).name, report["videos"][Path(path).name]["visible_samples"], "/", samples, "best", round(best, 3), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- zero
def collect_zero_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    from src.match_state import MatchStateDetector
    from src.score_ocr import ScoreOcr
    from src.score_zero import OcrScoreZeroDetector, ScoreZeroDetector
    template = ScoreZeroDetector.load_default()
    ocr = ScoreOcr.load_default()
    by_ocr = OcrScoreZeroDetector(ocr)
    match_detector = MatchStateDetector.load_default()
    rows: list[dict[str, Any]] = []
    for index, sec, frame in read_window(args.video, args.start, args.end, args.stride):
        a, b = template.detect(frame), by_ocr.detect(frame)
        rows.append(dict(frame=index, t=round(sec, 3), state=match_detector.detect(frame).state.value,
                         tpl=a.both_zero, ocr=b.both_zero, tpl1=a.is_1p_zero, tpl2=a.is_2p_zero,
                         ocr1=b.is_1p_zero, ocr2=b.is_2p_zero, ncc1=round(a.score_1p, 3), ncc2=round(a.score_2p, 3),
                         conf1=round(b.score_1p, 3), conf2=round(b.score_2p, 3)))
    return rows


def categorize(rows: list[dict[str, Any]]) -> list[str]:
    """フレームを 試合中 / 試合前(紹介) / リザルト に分ける。

    試合中 = MatchStateDetector が in_match。それ以外は「直前の in_match 終了からの距離」と
    「次の in_match 開始までの距離」の近い方で リザルト(直前が近い) / 試合前(次が近い) とする。
    区間内に in_match が全く無ければ no_match_seen。"""
    n = len(rows)
    inside = [r["state"] == STATE_IN_MATCH for r in rows]
    if not any(inside):
        return [CATEGORY_NONE] * n
    prev_gap, gap = [n] * n, n
    for i in range(n):
        gap = 0 if inside[i] else gap + 1
        prev_gap[i] = gap if any(inside[:i + 1]) else n
    next_gap, gap = [n] * n, n
    for i in range(n - 1, -1, -1):
        gap = 0 if inside[i] else gap + 1
        next_gap[i] = gap if any(inside[i:]) else n
    labels = []
    for i in range(n):
        if inside[i]:
            labels.append(CATEGORY_IN)
        else:
            labels.append(CATEGORY_POST if prev_gap[i] <= next_gap[i] else CATEGORY_PRE)
    return labels


def summarize_zero(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = categorize(rows)
    table: dict[str, dict[str, int]] = {}
    disagree: list[dict[str, Any]] = []
    for row, label in zip(rows, labels):
        cell = table.setdefault(label, dict(frames=0, agree=0, tpl_only=0, ocr_only=0, both_true=0))
        cell["frames"] += 1
        cell["agree"] += int(row["tpl"] == row["ocr"])
        cell["both_true"] += int(row["tpl"] and row["ocr"])
        cell["tpl_only"] += int(row["tpl"] and not row["ocr"])
        cell["ocr_only"] += int(row["ocr"] and not row["tpl"])
        if row["tpl"] != row["ocr"]:
            disagree.append(dict(row, category=label))
    return dict(total_frames=len(rows), by_category=table, disagreements=disagree)


def run_zero(args: argparse.Namespace) -> None:
    rows = collect_zero_rows(args)
    summary = summarize_zero(rows)
    summary.update(video=args.video, start=args.start, end=args.end, stride=args.stride)
    Path(args.out).write_text(json.dumps(dict(summary, rows=rows), ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "disagreements"}, ensure_ascii=False))
    print("disagreements:", len(summary["disagreements"]))


# ---------------------------------------------------------------- telop-ab / imread-trace
def load_pipeline(config_path: Path):
    from src.recognition_pipeline import RecognitionPipeline
    random.seed(SEED)
    np.random.seed(SEED)
    return RecognitionPipeline.load_default(**json.loads(config_path.read_text(encoding="utf-8")))


def grids_of(result: Any) -> tuple[bytes | None, bytes | None]:
    def grid(side: Any) -> bytes | None:
        return None if side.confirmed_board is None else side.confirmed_board.grid_bytes()
    return grid(result.p1), grid(result.p2)


def run_pipeline(args: argparse.Namespace, telop_present: bool = True, zero_from_ocr: bool = False) -> list[dict[str, Any]]:
    from src.score_zero import OcrScoreZeroDetector
    from src.telop_detector import TelopDetector
    pipe = load_pipeline(Path(args.config))
    if zero_from_ocr:  # 本番と同じ経路: score_zero を ScoreOcr 由来へ (flag ON と等価)
        pipe._score_zero_detector = OcrScoreZeroDetector(pipe._score_ocr)
    if not telop_present:  # 空ディレクトリ = 配布物にテロップ画像が無い状態と同じ (テンプレート 0 件)
        with tempfile.TemporaryDirectory() as empty:
            pipe._telop_detector = TelopDetector.load_default(template_dir=Path(empty))
    capture, fps = open_video(args.video)
    capture.release()
    stride = 2 if fps >= NATIVE_FPS_STRIDE_THRESHOLD else 1
    rows = []
    for index, sec, frame in read_window(args.video, args.start, args.end, stride):
        result = pipe.update(index, sec, frame)
        g1, g2 = grids_of(result)
        rows.append(dict(frame=index, t=sec, g1=g1, g2=g2, s1=str(result.p1.state), s2=str(result.p2.state),
                         telop=bool(pipe._last_telop_visible)))
    return rows


def compare_runs(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    """確定盤面をフレーム×サイドで比較。セル差は (フレーム,サイド,行,列,A値,B値) を列挙。"""
    cells, diff_cells, diff_frames, state_diffs = 0, [], set(), 0
    for ra, rb in zip(a, b):
        for side in ("g1", "g2"):
            ga, gb = ra[side], rb[side]
            if ga is None and gb is None:
                continue
            if ga is None or gb is None:
                diff_frames.add(ra["frame"])
                diff_cells.append((ra["frame"], side, "None-vs-board"))
                continue
            arr_a, arr_b = np.frombuffer(ga, np.uint8), np.frombuffer(gb, np.uint8)
            cells += arr_a.size
            bad = np.nonzero(arr_a != arr_b)[0]
            for pos in bad:
                diff_cells.append((ra["frame"], side, int(pos), int(arr_a[pos]), int(arr_b[pos])))
            if bad.size:
                diff_frames.add(ra["frame"])
        state_diffs += int(ra["s1"] != rb["s1"]) + int(ra["s2"] != rb["s2"])
    return dict(frames=len(a), compared_cells=cells, differing_cells=len(diff_cells),
                differing_frames=len(diff_frames), state_diffs=state_diffs, first_diffs=diff_cells[:30])


def run_telop_ab(args: argparse.Namespace) -> None:
    present = run_pipeline(args, True)
    absent = run_pipeline(args, False)
    out = dict(video=args.video, start=args.start, end=args.end,
               telop_visible_frames_present=sum(r["telop"] for r in present),
               telop_visible_frames_absent=sum(r["telop"] for r in absent),
               present_vs_absent=compare_runs(present, absent))
    if args.control:  # A/A: 同一構成を 2 回。これが 0 差でないと A/B の差は決定性の欠如で説明され得る
        out["control_present_vs_present"] = compare_runs(present, run_pipeline(args, True))
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, default=str))


def run_zero_ab(args: argparse.Namespace) -> None:
    """score_zero をテンプレート方式 / ScoreOcr 方式にして認識パイプラインを回し、確定盤面を比較する。"""
    template = run_pipeline(args)
    ocr = run_pipeline(args, zero_from_ocr=True)
    out = dict(video=args.video, start=args.start, end=args.end, template_vs_ocr=compare_runs(template, ocr))
    if args.control:
        out["control_template_vs_template"] = compare_runs(template, run_pipeline(args))
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, default=str))


def run_imread_trace(args: argparse.Namespace) -> None:
    original, seen = cv2.imread, []

    def traced(path: str, *rest: Any) -> Any:
        if TEMPLATE_DIR_MARK in str(path):
            seen.append(str(path).replace("\\", "/"))
        return original(path, *rest)

    cv2.imread = traced  # type: ignore[assignment]
    try:
        pipe = load_pipeline(Path(args.config))
        loaded = sorted(set(seen))
        frames = 0
        for index, sec, frame in read_window(args.video, args.start, args.end, 1):
            pipe.update(index, sec, frame)
            frames += 1
    finally:
        cv2.imread = original  # type: ignore[assignment]
    out = dict(loaded_at_init=loaded, all_after_run=sorted(set(seen)), frames=frames)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    scan = sub.add_parser("scan-telop")
    scan.add_argument("videos", nargs="+")
    scan.add_argument("--stride-sec", type=float, default=SCAN_STRIDE_SEC_DEFAULT)
    for name in ("zero", "telop-ab", "zero-ab", "imread-trace"):
        p = sub.add_parser(name)
        p.add_argument("--video", required=True)
        p.add_argument("--start", type=float, required=True)
        p.add_argument("--end", type=float, required=True)
        if name == "zero":
            p.add_argument("--stride", type=int, default=1)
        else:
            p.add_argument("--config", required=True, help="recognition_config.json (配布版が保存した認識設定)")
        if name in ("telop-ab", "zero-ab"):
            p.add_argument("--control", action="store_true")
    for p in sub.choices.values():
        p.add_argument("--out", required=True)
    args = parser.parse_args()
    {"scan-telop": scan_telop, "zero": run_zero, "telop-ab": run_telop_ab, "zero-ab": run_zero_ab, "imread-trace": run_imread_trace}[args.cmd](args)


if __name__ == "__main__":
    main()

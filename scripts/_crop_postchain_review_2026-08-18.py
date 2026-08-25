"""連鎖後/おじゃま着弾後の記録誤り疑いを、user目視レビュー用に3点セット画像化する
(2026-08-18)。

_analyze_postchain_record_accuracy_2026-08-18.py が書き出した flips.json から、
「連鎖/おじゃま隣接記録が絡む flip」を優先して選び、各ケースについて
  (a) 状態遷移終了 (chain_end/ojama_end) 直後のフレーム
  (b) 問題の記録が取られた瞬間のフレーム (認識グリッドを重畳)
  (c) 数秒後 (次の記録、訂正されているはずの状態) のフレーム (認識グリッドを重畳)
の3枚を横に並べた1枚の画像を書き出す。

コードは変更しない (診断専用)。 動画は cv2.VideoCapture のフレーム番号シークで
取得する (数秒〜数十秒程度の近距離シークなので mp4 の GOP 構造下でも実用上の
ずれは小さい。 本スクリプトはレビュー資料生成が目的でありフレーム精度の
厳密性は要求しない)。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_TAG = "2026-08-18"
_LOG_DIR = _ROOT / "logs"
_OUT_DIR = _ROOT / "data" / "verify" / "postchain_record_review_2026-08-18"

VIDEO_PATHS = {
    "video36_118_340": _ROOT / "data" / "frames" / "video_36.mp4",
    "video52_129_330": _ROOT / "data" / "frames" / "video_52.mp4",
    "c100_570_660": _ROOT / "data" / "frames" / "video_c100.mp4",
}

COLOR_NAMES = {0: "空", 1: "赤", 2: "青", 3: "緑", 4: "黄", 5: "紫", 9: "邪", 10: "?"}
COLOR_BGR = {
    0: (60, 60, 60), 1: (0, 0, 230), 2: (230, 120, 0), 3: (0, 180, 0),
    4: (0, 220, 220), 5: (200, 0, 200), 9: (200, 200, 200), 10: (0, 0, 0),
}

# BoardRegion (src/image_reader.py DEFAULT_P1/P2_REGION と同一値をハードコード
# する。診断スクリプトが本番コードに依存しすぎないよう定数のみ複写。
REGION = {
    "1P": dict(x=282, y=160, width=384, height=720),
    "2P": dict(x=1258, y=160, width=384, height=720),
}


def _grab_frame(cap: cv2.VideoCapture, fps: float, frame_idx: int) -> "np.ndarray | None":
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    if frame.shape[:2] != (1080, 1920):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame


def _crop_board(frame: np.ndarray, side: str) -> np.ndarray:
    r = REGION[side]
    return frame[r["y"]:r["y"] + r["height"], r["x"]:r["x"] + r["width"]].copy()


def _overlay_grid(
    board_img: np.ndarray, grid: "list | None", highlight: "tuple[int, int] | None" = None,
) -> np.ndarray:
    """認識グリッド値をセル中央に小さく描画する (grid は 13行 x 6列、行0=隠し段)。"""
    vis = board_img.copy()
    h, w = vis.shape[:2]
    cell_h, cell_w = h / 12.0, w / 6.0
    for r in range(13):
        y = int(r * cell_h)
        cv2.line(vis, (0, y), (w, y), (90, 90, 90), 1)
    for c in range(7):
        x = int(c * cell_w)
        cv2.line(vis, (x, 0), (x, h), (90, 90, 90), 1)
    if grid is not None:
        for r in range(1, 13):  # row0(隠し段)は画面外なので描画しない
            for c in range(6):
                v = grid[r][c]
                y0 = int((r - 1) * cell_h)
                x0 = int(c * cell_w)
                cx, cy = x0 + int(cell_w * 0.5), y0 + int(cell_h * 0.5)
                cv2.putText(
                    vis, COLOR_NAMES.get(v, "?"), (x0 + 4, y0 + int(cell_h) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BGR.get(v, (255, 255, 255)), 2,
                    cv2.LINE_AA,
                )
    if highlight is not None:
        hr, hc = highlight
        y0 = int((hr - 1) * cell_h)
        x0 = int(hc * cell_w)
        cv2.rectangle(
            vis, (x0, y0), (x0 + int(cell_w), y0 + int(cell_h)), (0, 0, 255), 4,
        )
    return vis


def _label_panel(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.zeros((36, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def build_case(flip: dict, cap_cache: dict) -> "tuple[np.ndarray, str] | None":
    video = flip["video"]
    side = flip["side"]
    r, c = flip["r"], flip["c"]
    video_path = VIDEO_PATHS.get(video)
    if video_path is None or not video_path.exists():
        return None
    if video not in cap_cache:
        cap = cv2.VideoCapture(str(video_path))
        cap_cache[video] = (cap, cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap, fps = cap_cache[video]

    end_t = flip.get("chain_end_a") or flip.get("ojama_end_a")
    end_frame_idx = int(round(end_t * fps)) if end_t is not None else flip["frame_a"]

    panels = []
    fr_end = _grab_frame(cap, fps, end_frame_idx)
    if fr_end is not None:
        img = _overlay_grid(_crop_board(fr_end, side), None)
        panels.append(_label_panel(img, f"t={end_t if end_t is not None else '?'}s 状態遷移終了直後"))

    fr_a = _grab_frame(cap, fps, flip["frame_a"])
    if fr_a is not None:
        # frame_a時点のgridはflip["val_a"]がその値、他セルは不明なのでNoneグリッド上に該当セルだけ手動注記
        img = _overlay_grid(_crop_board(fr_a, side), None, highlight=(r, c))
        cv2.putText(
            img, f"認識:{COLOR_NAMES.get(flip['val_a'],'?')}", (8, img.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
        )
        panels.append(_label_panel(img, f"t={flip['t_a']}s 記録された瞬間 (kind={flip['kind_a']})"))

    fr_b = _grab_frame(cap, fps, flip["frame_b"])
    if fr_b is not None:
        img = _overlay_grid(_crop_board(fr_b, side), None, highlight=(r, c))
        cv2.putText(
            img, f"認識:{COLOR_NAMES.get(flip['val_b'],'?')}", (8, img.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA,
        )
        panels.append(_label_panel(img, f"t={flip['t_b']}s 次の記録 (kind={flip['kind_b']})"))

    if not panels:
        return None
    max_h = max(p.shape[0] for p in panels)
    panels = [
        cv2.copyMakeBorder(p, 0, max_h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for p in panels
    ]
    combined = np.hstack(panels)
    r1, c1 = r, c
    row_disp = r1  # 行番号はそのまま (0=隠し段, 1〜12=可視12行、盤面上から数える表示ではなく内部行index)
    fname = (
        f"{video}_{side}_r{r1}c{c1}_t{flip['t_a']}_{flip['val_a']}to{flip['val_b']}.png"
    )
    return combined, fname


def main(limit_chain: int = 10, limit_ojama: int = 10, limit_clean: int = 5) -> None:
    flips = json.loads(
        (_LOG_DIR / f"_analyze_postchain_record_accuracy_{_TAG}_flips.json").read_text(encoding="utf-8"),
    )
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    def sort_key(f):
        return -(f.get("residual_sec") or 0.0)

    chain_flips = sorted(
        [f for f in flips if f["kind_a"] == "chain" and f["kind_b"] != "chain"],
        key=sort_key,
    )[:limit_chain]
    ojama_flips = sorted(
        [f for f in flips if f["kind_a"] == "ojama" and f["kind_b"] != "ojama"],
        key=sort_key,
    )[:limit_ojama]

    selected = chain_flips + ojama_flips
    cap_cache: dict = {}
    manifest: list[dict] = []
    for fl in selected:
        result = build_case(fl, cap_cache)
        if result is None:
            continue
        img, fname = result
        out_path = _OUT_DIR / fname
        cv2.imwrite(str(out_path), img)
        manifest.append({
            "path": str(out_path).replace("/", "\\"),
            "video": fl["video"], "side": fl["side"],
            "cell": f"r{fl['r']}c{fl['c']}",
            "kind_a": fl["kind_a"], "kind_b": fl["kind_b"],
            "val_a": fl["val_a"], "val_b": fl["val_b"],
            "residual_sec": fl.get("residual_sec"),
            "t_a": fl["t_a"], "t_b": fl["t_b"],
        })
        print(f"[ok] {out_path}")
    for cap, _fps in cap_cache.values():
        cap.release()

    manifest_path = _OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {manifest_path} ({len(manifest)} cases)")


if __name__ == "__main__":
    main()

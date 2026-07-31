"""試合開始直後の A(warmup30)/D(warmup0) フレーム比較画像を生成する診断スクリプト。

背景: user が「認識が早くなった」と評価した D (校正ON+warmupなし) が、実は
warmup=30 の A よりも「試合開始時に前試合の残骸(幽霊セル)が少ないから」
そう見えたのではないか、という仮説を画像で検証する (2026-07-30)。

A/D は同一試合 c56_g3 の同一区間 (絶対 288.0〜362.0秒) から生成された動画で、
どちらも動画先頭 = 相対 t=0.0s なので、フレーム番号がそのまま同一時刻に対応する
(この前提は本スクリプト内で fps・尺・フレーム数を実測して確認する)。

出力: data/verify/ad_start_compare_2026-07-30/ 配下に
  - t{ms:04d}ms_full.png       … 画面全体 (勝率パネル含む) の左右比較
  - t{ms:04d}ms_board_zoom.png … 盤面領域のみ拡大した左右比較
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- 対象動画 (どちらも既存生成物、再生成しない) ---
VIDEO_A = Path(
    "data/verify/review4_2026-07-29/advantage_c56_g3_full_score0to0_h264.mp4"
)  # A = 校正OFF + warmup 30秒
VIDEO_D = Path(
    "data/verify/review4_2026-07-29/advantage_c56_g3_calibON_full_score0to0_h264.mp4"
)  # D = 校正ON + warmupなし (userが「早くなった」と評価した方)
OUT_DIR = Path("data/verify/ad_start_compare_2026-07-30")

# 抽出したい相対時刻 (秒)。開始直後を密に取る。
TARGET_TIMES_SEC: tuple[float, ...] = (0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0)

# visualize_advantage_overlay.py のキャンバス定数 (再掲、値は同一に保つ)。
CANVAS_W, CANVAS_H = 1280, 1110
GAME_Y0, GAME_Y1 = 240, 960  # ゲーム画面領域 (盤面+勝率パネルの下)
# scripts/visualize_recognition.py の ROI をキャンバス座標系 (縮小後+TOP_H加算後)
# へ換算した範囲。state ラベル文字も入るよう上に少し余裕を持たせる。
BOARD_ZOOM_Y0, BOARD_ZOOM_Y1 = 285, 840
BOARD_ZOOM_X0, BOARD_ZOOM_X1 = 170, 1105

FONT_CANDIDATES = (r"C:\Windows\Fonts\meiryo.ttc", "/mnt/c/Windows/Fonts/meiryo.ttc")


def _font(size: int) -> ImageFont.ImageFont:
    """meiryo フォントを取得 (無ければ default、visualize_advantage_overlay.py の流用)。"""
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def probe_video(path: Path) -> tuple[float, int, float]:
    """fps・総フレーム数・尺(秒) を実測して返す (同一場面前提の検証用)。"""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    dur = nframes / fps if fps else float("nan")
    return fps, nframes, dur


def extract_target_frames(
    video_path: Path, target_indices: list[int]
) -> dict[int, np.ndarray]:
    """動画から指定フレーム番号の画像を逐次読み込みで正確に抽出する。

    cv2 の CAP_PROP_POS_FRAMES によるシークは h264 で数フレームずれることが
    あるため、対象フレームまで順番に読み進める確実な方式を採る (対象は最大でも
    8秒=480フレームなので低コスト)。
    """
    cap = cv2.VideoCapture(str(video_path))
    result: dict[int, np.ndarray] = {}
    target_set = set(target_indices)
    max_idx = max(target_indices)
    idx = 0
    while idx <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in target_set:
            result[idx] = frame.copy()
        idx += 1
    cap.release()
    return result


def _bgr_to_pil(frame: np.ndarray) -> Image.Image:
    """OpenCV BGR ndarray を PIL RGB Image に変換する。"""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _make_header(width: int, height: int, left_label: str, right_label: str,
                  half_w: int) -> Image.Image:
    """左右比較画像の上部に貼るラベル帯 (白背景+黒文字) を作る。"""
    header = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(header)
    d.text((16, 8), left_label, font=_font(22), fill=(0, 0, 0))
    d.text((half_w + 16, 8), right_label, font=_font(22), fill=(0, 0, 0))
    d.line([(half_w, 0), (half_w, height)], fill=(0, 0, 0), width=2)
    return header


def build_side_by_side(img_a: Image.Image, img_d: Image.Image, t_rel: float,
                        title: str) -> Image.Image:
    """A/D 2枚を横に並べ、上部にラベル帯を付けた比較画像を作る。"""
    gap = 6
    w, h = img_a.size
    header_h = 40
    canvas = Image.new("RGB", (w * 2 + gap, h + header_h), (255, 255, 255))
    label_a = f"A (warmup30, calibOFF)  t={t_rel:+.1f}s  [{title}]"
    label_d = f"D (warmup0, calibON)  t={t_rel:+.1f}s  [{title}]"
    header = _make_header(w * 2 + gap, header_h, label_a, label_d, w + gap // 2)
    canvas.paste(header, (0, 0))
    canvas.paste(img_a, (0, header_h))
    canvas.paste(img_d, (w + gap, header_h))
    return canvas


def build_full_image(frame: np.ndarray) -> Image.Image:
    """ゲーム画面領域(盤面+勝率パネル下部)を切り出す(等倍、全体の見た目確認用)。"""
    crop = frame[GAME_Y0:GAME_Y1, 0:CANVAS_W]
    return _bgr_to_pil(crop)


def build_board_zoom_image(frame: np.ndarray, scale: float = 2.5) -> Image.Image:
    """盤面領域(state ラベル込み)を切り出して拡大する(セル色読み取り用)。"""
    crop = frame[BOARD_ZOOM_Y0:BOARD_ZOOM_Y1, BOARD_ZOOM_X0:BOARD_ZOOM_X1]
    h, w = crop.shape[:2]
    up = cv2.resize(crop, (int(w * scale), int(h * scale)),
                     interpolation=cv2.INTER_LANCZOS4)
    return _bgr_to_pil(up)


def process_all(times_sec: tuple[float, ...], fps: float) -> None:
    """全対象時刻について A/D 比較画像 (全体版+盤面拡大版) を生成し保存する。"""
    indices = [round(t * fps) for t in times_sec]
    frames_a = extract_target_frames(VIDEO_A, indices)
    frames_d = extract_target_frames(VIDEO_D, indices)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for t, idx in zip(times_sec, indices):
        if idx not in frames_a or idx not in frames_d:
            print(f"[WARN] t={t}s (frame {idx}) が A または D で取得できず skip")
            continue
        fa, fd = frames_a[idx], frames_d[idx]
        ms = int(round(t * 1000))
        full_img = build_side_by_side(
            build_full_image(fa), build_full_image(fd), t, "全体")
        full_path = OUT_DIR / f"t{ms:04d}ms_full.png"
        full_img.save(full_path)
        zoom_img = build_side_by_side(
            build_board_zoom_image(fa), build_board_zoom_image(fd), t, "盤面拡大")
        zoom_path = OUT_DIR / f"t{ms:04d}ms_board_zoom.png"
        zoom_img.save(zoom_path)
        print(f"[ok] t={t:.1f}s (frame {idx}) -> {full_path.name}, {zoom_path.name}")


def main() -> None:
    """A/D の尺・fps・フレーム数を実測して同一場面前提を検証し、比較画像を生成する。"""
    fps_a, n_a, dur_a = probe_video(VIDEO_A)
    fps_d, n_d, dur_d = probe_video(VIDEO_D)
    print(f"A: fps={fps_a:.4f} nframes={n_a} dur={dur_a:.3f}s")
    print(f"D: fps={fps_d:.4f} nframes={n_d} dur={dur_d:.3f}s")
    if (fps_a, n_a) != (fps_d, n_d):
        print("[WARN] A/D の fps またはフレーム数が一致しない -> "
              "相対時刻での同一場面比較の前提が崩れている可能性")
    process_all(TARGET_TIMES_SEC, fps_a)


if __name__ == "__main__":
    main()

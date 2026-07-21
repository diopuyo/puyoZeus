"""1P 手数ごとの断面 + 有利不利の根拠を1枚の画像にする (診断用)。

各 1P 手 (tsumo 増分) の STABLE 断面で:
  - 両者の盤面グリッド
  - 有利不利スコア / 勝率
  - 各指標の 1P値 / 2P値 / 差 / 効く方向 / 値の説明
を1枚に描画し、data/indicators_v2/moves/ に move_NNN.png として保存。
最後にサムネイル montage も作る。

使い方 (v29 game B):
    python -m scripts.plot_move_diagnostics \
        --video data/frames/video_29.mp4 --video-id video_29 \
        --start-sec 202 --end-sec 283 --warmup-sec 16 --exclude-video video_29
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.indicators_v2 as iv  # noqa: E402
from src.board import Board  # noqa: E402
from src.board_state_machine import BoardState  # noqa: E402
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402
from scripts.collect_indicators_v2 import _SideTracker, _drive_ojama  # noqa: E402
from scripts.visualize_advantage_overlay import (  # noqa: E402
    _train_model, _side_feats, FEATURES,
)

FONT_PATH_CANDS = (r"C:\Windows\Fonts\meiryo.ttc", "/mnt/c/Windows/Fonts/meiryo.ttc")
EVEN = 5.0
CELL_COLORS = {  # 認識色 -> RGB
    0: (45, 45, 55), 1: (220, 70, 70), 2: (70, 120, 220), 3: (70, 190, 90),
    4: (235, 215, 70), 5: (185, 90, 205), 9: (150, 150, 150), 10: (100, 70, 70),
}
# 指標メタ: (列名, 日本語, 効く符号(1P高が1P有利=+1), 単位, 説明)
META: tuple[tuple, ...] = (
    ("board_ojama_count", "盤面お邪魔数", -1, "個", "自盤面のお邪魔。多いほど圧迫=不利"),
    ("death_margin", "窒息余裕", +1, "", "窒息(3列目上)までの余裕。大きいほど安全"),
    ("max_column_height", "最大列高", -1, "段", "一番高い列。高いほど窒息に近く不利"),
    ("current_max_chain", "現在最大連鎖", +1, "連鎖", "今すぐ組める最大連鎖数。多いほど有利"),
    ("ojama_forecast", "お邪魔予告", -1, "個", "自分に降る予告お邪魔。多いほど不利"),
    ("ojama_net_balance", "お邪魔純収支", +1, "", "相殺後の攻め収支。プラスほど1P攻勢"),
    ("board_color_puyo_total", "色ぷよ総数", +1, "個", "自盤面の色ぷよ=土台の厚み"),
    ("conn_pair_count", "2連結数", +1, "組", "2つ繋がった色の数=連鎖の種"),
    ("conn_triple_count", "3連結数", -1, "組", "3連結=発火寸前。過多は不安定気味"),
    ("column_bumpiness", "列の凸凹", -1, "", "列高のばらつき。大きいほど組みにくい"),
    ("dig_resistance", "掘り耐性", +1, "", "お邪魔を掘り返す力。高いほど受け強い"),
    ("death_margin_neighbor", "窒息余裕(近傍)", +1, "", "窒息列周辺の余裕"),
)


def _font(sz: int) -> ImageFont.ImageFont:
    for p in FONT_PATH_CANDS:
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def _raw(board: Board, col: str, net: int, forecast: int) -> float:
    """表示用の生値 (お邪魔収支/予告は snap 由来)。"""
    if col == "ojama_net_balance":
        return float(iv.ojama_net_balance(net).raw)
    if col == "ojama_forecast":
        return float(iv.ojama_forecast(forecast).raw)
    if col in ("conn_pair_count", "conn_triple_count"):
        co, _ = iv.connectivity_observation(board)
        return float(co.pair_count if col == "conn_pair_count" else co.triple_count)
    fn = getattr(iv, col)
    return float(fn(board).raw)


def _render_board(d: "ImageDraw.ImageDraw", board: Board, ox: int, oy: int,
                  cell: int, label: str) -> None:
    """盤面グリッドを描画。"""
    g = board._grid  # (13, 6)
    d.text((ox, oy - 24), label, font=_font(18), fill=(255, 255, 255))
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            col = CELL_COLORS.get(int(g[r, c]), (60, 60, 60))
            x, y = ox + c * cell, oy + r * cell
            d.rectangle([x, y, x + cell - 1, y + cell - 1], fill=col,
                        outline=(25, 25, 30))


def _contributions(model, diff: dict[str, float]) -> tuple[float, dict[str, float]]:
    """モデル忠実な寄与。各指標の差を0(互角)に戻した時の勝率変化(pt, 1P視点)。

    Returns: (base_p1, {col: 寄与pt})。寄与pt>0 = その指標が1Pの勝率を押し上げている。
    """
    x = np.array([[diff[c] for c in FEATURES]], dtype=float)
    base = float(model.predict_proba(x)[0, 1])
    contrib: dict[str, float] = {}
    for i, c in enumerate(FEATURES):
        xn = x.copy(); xn[0, i] = 0.0
        pn = float(model.predict_proba(xn)[0, 1])
        contrib[c] = (base - pn) * 100.0
    return base, contrib


def _diag_image(move_no: int, t_rel: float, b1: Board, b2: Board,
                snap, model) -> Image.Image:
    """1手分の診断画像を作る (根拠 = モデルの各指標寄与)。"""
    net1, fo1 = snap.net_balance_capped, snap.forecast_p1
    net2, fo2 = -snap.net_balance_capped, snap.forecast_p2
    f1 = _side_feats(b1, net1, fo1); f2 = _side_feats(b2, net2, fo2)
    diff = {c: f1[c] - f2[c] for c in FEATURES}
    p1, contrib = _contributions(model, diff)
    adv = (p1 - 0.5) * 200.0
    img = Image.new("RGB", (1180, 760), (20, 22, 30))
    d = ImageDraw.Draw(img)
    verdict = ("互角" if abs(adv) < EVEN else f"{'1P' if adv > 0 else '2P'}有利")
    d.text((20, 12), f"1P手数 {move_no}  (t=+{t_rel:.1f}s)", font=_font(26),
           fill=(255, 255, 255))
    d.text((20, 46), f"有利不利 {adv:+.0f}  ({verdict})   勝率 1P {p1 * 100:.0f}% / "
           f"2P {(1 - p1) * 100:.0f}%", font=_font(24),
           fill=(150, 200, 255) if adv > 0 else (255, 180, 180))
    _render_board(d, b1, 24, 110, 22, "1P 盤面")
    _render_board(d, b2, 190, 110, 22, "2P 盤面")
    tx, ty = 350, 96
    heads = ["指標", "1P", "2P", "差", "寄与(勝率pt)", "説明"]
    xs = [tx, tx + 150, tx + 212, tx + 274, tx + 336, tx + 452]
    for x, h in zip(xs, heads):
        d.text((x, ty), h, font=_font(16), fill=(255, 255, 120))
    ty += 26
    # 寄与の絶対値が大きい順 = 根拠として効いている順に並べる
    rows = sorted(META, key=lambda m: -abs(contrib.get(m[0], 0.0)))
    for col, jp, _sgn, unit, desc in rows:
        v1, v2 = _raw(b1, col, net1, fo1), _raw(b2, col, net2, fo2)
        cp = contrib.get(col, 0.0)
        arrow = "→1P" if cp > 0.3 else ("→2P" if cp < -0.3 else "・")
        cc = (150, 200, 255) if cp > 0.3 else (
            (255, 180, 180) if cp < -0.3 else (170, 170, 170))
        d.text((xs[0], ty), jp, font=_font(15), fill=(230, 230, 230))
        d.text((xs[1], ty), f"{v1:.1f}{unit}", font=_font(14), fill=(210, 210, 210))
        d.text((xs[2], ty), f"{v2:.1f}{unit}", font=_font(14), fill=(210, 210, 210))
        d.text((xs[3], ty), f"{v1 - v2:+.1f}", font=_font(14), fill=(210, 210, 210))
        d.text((xs[4], ty), f"{arrow} {cp:+.1f}", font=_font(14), fill=cc)
        d.text((xs[5], ty), desc, font=_font(13), fill=(180, 180, 180))
        ty += 27
    d.text((20, 726), "※ 寄与 = その指標差を互角(0)に戻した時の1P勝率変化。＋=1Pに有利方向。"
           "有利不利=tier1軽量モデル(試作)", font=_font(13), fill=(150, 150, 150))
    return img


def _collect_and_render(a) -> list[Path]:
    """1P 手ごとに診断画像を保存し、パス一覧を返す。"""
    model = _train_model(a.exclude_video)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    proc = int(max(0.0, a.start_sec - a.warmup_sec) * fps)
    end = int(a.end_sec * fps) if a.end_sec > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, proc)
    pipe = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True)
    pipe.set_video_id(a.video_id)
    tr = OjamaAccountingTracker(); tr.reset()
    tp1, tp2 = _SideTracker(), _SideTracker()
    ps1 = ps2 = BoardState.MENU
    b1 = b2 = None
    prev_tsumo = -1
    move_no = 0
    paths: list[Path] = []
    step = max(1, int(round(0.1 * fps)))
    for fi in range(proc, end):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (1080, 1920):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        t = fi / fps
        r = pipe.update(fi, t, frame)  # 会計のため毎フレーム更新
        if r.p1.state == BoardState.STABLE and r.p1.confirmed_board is not None:
            b1 = r.p1.confirmed_board
        if r.p2.state == BoardState.STABLE and r.p2.confirmed_board is not None:
            b2 = r.p2.confirmed_board
        snap = _drive_ojama(tr, r.p1, r.p2, ps1, ps2, t,
                            tracker_p1=tp1, tracker_p2=tp2, pipeline=pipe)
        ps1, ps2 = r.p1.state, r.p2.state
        ts1 = pipe.tsumo_count("1P")
        new_move = ts1 != prev_tsumo and r.p1.state == BoardState.STABLE
        if t >= a.start_sec and new_move and b1 is not None and b2 is not None:
            move_no += 1
            prev_tsumo = ts1
            p = out_dir / f"move_{move_no:03d}.png"
            _diag_image(move_no, t - a.start_sec, b1, b2, snap, model).save(p)
            paths.append(p)
        elif ts1 != prev_tsumo:
            prev_tsumo = ts1
    cap.release()
    print(f"[moves] {len(paths)} 手を保存 -> {out_dir}")
    return paths


def _montage(paths: list[Path], out: Path, cols: int = 5) -> None:
    """診断画像のサムネイル一覧を作る。"""
    if not paths:
        return
    thumbs = [Image.open(p).resize((295, 190)) for p in paths]
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 300, rows * 195), (10, 10, 12))
    for i, th in enumerate(thumbs):
        sheet.paste(th, ((i % cols) * 300 + 2, (i // cols) * 195 + 2))
    sheet.save(out)
    print("montage:", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--start-sec", type=float, default=0.0)
    ap.add_argument("--end-sec", type=float, default=0.0)
    ap.add_argument("--warmup-sec", type=float, default=16.0)
    ap.add_argument("--exclude-video", default=None)
    ap.add_argument("--out-dir", default="data/indicators_v2/moves")
    a = ap.parse_args()
    paths = _collect_and_render(a)
    _montage(paths, Path(a.out_dir) / "_montage.png")


if __name__ == "__main__":
    main()

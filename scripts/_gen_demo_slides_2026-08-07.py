"""YouTube動画台本用の図解スライド画像を8枚生成するスクリプト (2026-08-07)

出力: data/verify/youtube_demo_2026-08-07/slides/01〜08.png (1920x1080)
規約: 型ヒント必須、1関数50行以内 (CLAUDE.md)。使い捨てスクリプトのため
      LEARNED_WEIGHTS_* 等の本番資産には一切触れない (副作用なし)。
"""
from __future__ import annotations

import os
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

# --- 定数定義 (マジックナンバー禁止規約に従い集約) ---
CANVAS_W = 1920
CANVAS_H = 1080
BG_COLOR = (20, 26, 46)
WHITE = (245, 247, 250)
GRAY = (170, 178, 200)
BLUE = (77, 163, 255)
RED = (255, 90, 90)
YELLOW = (255, 215, 90)
BOX_FILL = (30, 38, 64)
BOX_EDGE = (80, 95, 140)

FONT_DIR = "/mnt/c/Windows/Fonts"
FONT_BOLD = os.path.join(FONT_DIR, "meiryob.ttc")
FONT_REGULAR = os.path.join(FONT_DIR, "meiryo.ttc")

OUT_DIR = "data/verify/youtube_demo_2026-08-07/slides"
CREDIT_TEXT = "puyoZeus"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """日本語対応フォントを読み込む (meiryo/meiryob)。"""
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(path, size)


def new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """背景色済みの1920x1080キャンバスを作る。"""
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_COLOR)
    draw = ImageDraw.Draw(img)
    return img, draw


def draw_credit(draw: ImageDraw.ImageDraw) -> None:
    """右下に小さくクレジットを描画。"""
    f = font(22)
    bbox = draw.textbbox((0, 0), CREDIT_TEXT, font=f)
    w = bbox[2] - bbox[0]
    draw.text((CANVAS_W - w - 36, CANVAS_H - 46), CREDIT_TEXT, font=f, fill=GRAY)


def draw_title_bar(draw: ImageDraw.ImageDraw, title: str, subtitle: str = "") -> None:
    """スライド共通の上部見出しバーを描画。"""
    draw.text((80, 60), title, font=font(56, bold=True), fill=WHITE)
    if subtitle:
        draw.text((80, 132), subtitle, font=font(28), fill=GRAY)
    draw.line([(80, 190), (CANVAS_W - 80, 190)], fill=BOX_EDGE, width=2)


def rounded_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    edge_color: tuple[int, int, int] = BOX_EDGE,
    fill_color: tuple[int, int, int] = BOX_FILL,
    width: int = 3,
    radius: int = 24,
) -> None:
    """角丸ボックスを描画する共通ヘルパー。"""
    draw.rounded_rectangle(xy, radius=radius, fill=fill_color, outline=edge_color, width=width)


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    f: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int] = WHITE,
    y_offset: int = 0,
) -> None:
    """box (x0,y0,x1,y1) の中央にテキストを描く。"""
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=f)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = (x0 + x1) / 2 - w / 2
    cy = (y0 + y1) / 2 - h / 2 - bbox[1] + y_offset
    draw.text((cx, cy), text, font=f, fill=fill)


def arrow(draw: ImageDraw.ImageDraw, p0: tuple[int, int], p1: tuple[int, int],
          color: tuple[int, int, int] = BOX_EDGE, width: int = 4, head: int = 14) -> None:
    """矢印線を描画する (直線+矢じり)。"""
    import math

    draw.line([p0, p1], fill=color, width=width)
    angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for da in (2.6, -2.6):
        hx = p1[0] - head * math.cos(angle - da)
        hy = p1[1] - head * math.sin(angle - da)
        draw.line([p1, (hx, hy)], fill=color, width=width)


# ============================================================
# 01: タイトル
# ============================================================
def slide_01_title() -> Image.Image:
    img, draw = new_canvas()
    main = "ぷよぷよ対戦をAIが解析する"
    sub = "盤面認識 99.5% × 45指標 × 勝率予測"
    f_main = font(84, bold=True)
    f_sub = font(40)
    bbox = draw.textbbox((0, 0), main, font=f_main)
    w = bbox[2] - bbox[0]
    draw.text(((CANVAS_W - w) / 2, 430), main, font=f_main, fill=WHITE)
    bbox2 = draw.textbbox((0, 0), sub, font=f_sub)
    w2 = bbox2[2] - bbox2[0]
    draw.text(((CANVAS_W - w2) / 2, 560), sub, font=f_sub, fill=BLUE)
    draw.line([(CANVAS_W / 2 - 220, 540), (CANVAS_W / 2 + 220, 540)], fill=YELLOW, width=4)
    draw_credit(draw)
    return img


# ============================================================
# 02: 状態機械図
# ============================================================
def slide_02_state_machine() -> Image.Image:
    img, draw = new_canvas()
    draw_title_bar(draw, "認識の状態機械", "4つの状態を巡回しながら盤面を確定する")

    box_w, box_h = 420, 190
    positions = {
        "stable": (CANVAS_W // 2 - box_w // 2, 300),
        "tsumo": (1280, 620),
        "chain": (CANVAS_W // 2 - box_w // 2, 780),
        "ojama": (220, 620),
    }
    labels = {
        "stable": "安定 (STABLE)",
        "tsumo": "ツモ落下中",
        "chain": "連鎖中",
        "ojama": "おじゃま落下中",
    }
    for key, (x, y) in positions.items():
        is_stable = key == "stable"
        rounded_box(
            draw, (x, y, x + box_w, y + box_h),
            edge_color=BLUE if is_stable else BOX_EDGE,
            width=6 if is_stable else 3,
        )
        centered_text(draw, (x, y, x + box_w, y + box_h), labels[key], font(38, bold=True), y_offset=-24)
        note = "CNNを信じるのはここだけ" if is_stable else "物理シミュレーションで推定"
        centered_text(draw, (x, y, x + box_w, y + box_h), note, font(22),
                      fill=YELLOW if is_stable else GRAY, y_offset=28)

    order = ["stable", "tsumo", "chain", "ojama"]
    centers = {
        k: (positions[k][0] + box_w / 2, positions[k][1] + box_h / 2) for k in order
    }
    for i in range(len(order)):
        a, b = centers[order[i]], centers[order[(i + 1) % len(order)]]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = max((dx ** 2 + dy ** 2) ** 0.5, 1.0)
        shrink = box_h / 2 + 10
        pa = (a[0] + dx / length * shrink, a[1] + dy / length * shrink)
        pb = (b[0] - dx / length * shrink, b[1] - dy / length * shrink)
        arrow(draw, pa, pb, color=BLUE, width=4)

    draw_credit(draw)
    return img


# ============================================================
# 03: ハイブリッド認識フロー
# ============================================================
def slide_03_hybrid_flow() -> Image.Image:
    img, draw = new_canvas()
    draw_title_bar(draw, "ハイブリッド認識フロー", "2系統の多数決で片方の誤りを守る")

    boxes = {
        "frame": (100, 420, 380, 590),
        "hsv": (560, 260, 840, 430),
        "cnn": (560, 600, 840, 770),
        "vote": (1060, 420, 1340, 590),
        "board": (1560, 420, 1840, 590),
        "calib": (560, 900, 840, 1000),
    }
    labels = {
        "frame": "動画フレーム",
        "hsv": "HSV色判定",
        "cnn": "セルCNN",
        "vote": "多数決",
        "board": "確定盤面",
        "calib": "オンラインHSV較正",
    }
    for key, xy in boxes.items():
        rounded_box(draw, xy)
        centered_text(draw, xy, labels[key], font(30, bold=True))

    def c(key: str) -> tuple[float, float]:
        x0, y0, x1, y1 = boxes[key]
        return (x0 + x1) / 2, (y0 + y1) / 2

    arrow(draw, (boxes["frame"][2], c("frame")[1]), (boxes["hsv"][0], c("hsv")[1]), color=BLUE)
    arrow(draw, (boxes["frame"][2], c("frame")[1]), (boxes["cnn"][0], c("cnn")[1]), color=BLUE)
    arrow(draw, (boxes["hsv"][2], c("hsv")[1]), (boxes["vote"][0], c("vote")[1] - 40), color=BLUE)
    arrow(draw, (boxes["cnn"][2], c("cnn")[1]), (boxes["vote"][0], c("vote")[1] + 40), color=BLUE)
    arrow(draw, (boxes["vote"][2], c("vote")[1]), (boxes["board"][0], c("board")[1]), color=BLUE)
    arrow(draw, (c("calib")[0], boxes["calib"][1]), (c("hsv")[0], boxes["hsv"][3]), color=YELLOW)

    draw_credit(draw)
    return img


# ============================================================
# 04: 3つの敵
# ============================================================
def slide_04_three_enemies() -> Image.Image:
    img, draw = new_canvas()
    draw_title_bar(draw, "盤面認識 3つの敵")

    items = [
        ("1", "圧縮ノイズ", "色の境界がにじむ", RED),
        ("2", "連鎖演出", "半透明レイヤーで色が別物に", RED),
        ("3", "残像", "消えたぷよが残って見える", RED),
    ]
    y = 300
    row_h = 240
    for num, title, desc, color in items:
        box = (140, y, CANVAS_W - 140, y + row_h - 40)
        rounded_box(draw, box)
        draw.text((190, y + 30), num, font=font(96, bold=True), fill=color)
        draw.text((330, y + 30), title, font=font(48, bold=True), fill=WHITE)
        draw.text((330, y + 100), desc, font=font(30), fill=GRAY)
        y += row_h

    draw_credit(draw)
    return img


# ============================================================
# 05: 事故カタログ
# ============================================================
def slide_05_accident_catalog() -> Image.Image:
    img, draw = new_canvas()
    draw_title_bar(draw, "測定器の事故カタログ (実話)")

    items = [
        "2つのAIが同じ誤りに合意 → 検出不能",
        "改善機能が本番に配線されてなかった",
        "フレーム間引きが盤面を壊した",
        "AUC近似バグで幻の改善",
        "確定不能フレームを混ぜて測定",
        "20分打ち切りでラベル欠落",
    ]
    y = 226
    row_step = 104
    for i, text in enumerate(items, start=1):
        row = (140, y, CANVAS_W - 140, y + row_step - 20)
        rounded_box(draw, row, radius=18)
        draw.text((175, y + 16), f"{i}", font=font(36, bold=True), fill=YELLOW)
        draw.text((240, y + 20), text, font=font(30), fill=WHITE)
        y += row_step

    tip_box = (140, y + 16, CANVAS_W - 140, y + 96)
    rounded_box(draw, tip_box, edge_color=BLUE, width=4)
    centered_text(draw, tip_box, "数字が良くなった時ほど、測定器から疑う", font(32, bold=True), fill=BLUE)

    draw_credit(draw)
    return img


# ============================================================
# 06: 自己無矛盾性の罠
# ============================================================
def slide_06_self_consistency_trap() -> Image.Image:
    img, draw = new_canvas()
    draw_title_bar(draw, "自己無矛盾性の罠", "一致率は正しさではない")

    a_box = (200, 300, 620, 430)
    b_box = (740, 300, 1160, 430)
    result_box = (1280, 300, 1720, 430)
    rounded_box(draw, a_box)
    rounded_box(draw, b_box)
    rounded_box(draw, result_box, edge_color=RED, width=5)
    centered_text(draw, a_box, "AI-A「ここは空」", font(28, bold=True))
    centered_text(draw, b_box, "AI-B「ここは空」", font(28, bold=True))
    centered_text(draw, result_box, "一致 ✓ (でも誤り)", font(28, bold=True), fill=RED)
    arrow(draw, (a_box[2], (a_box[1] + a_box[3]) // 2), (b_box[0], (b_box[1] + b_box[3]) // 2), color=BOX_EDGE)
    arrow(draw, (b_box[2], (b_box[1] + b_box[3]) // 2), (result_box[0], (result_box[1] + result_box[3]) // 2), color=BOX_EDGE)
    draw.text((200, 460), "※ 実際にはそのマスにぷよがある = 誤り", font=font(24), fill=GRAY)

    label_box = (200, 660, 900, 790)
    truth_box = (1020, 660, 1720, 790)
    rounded_box(draw, label_box, edge_color=BLUE, width=4)
    rounded_box(draw, truth_box, edge_color=YELLOW, width=4)
    centered_text(draw, label_box, "人手ラベル 3,672セル (物差し)", font(30, bold=True), fill=BLUE)
    centered_text(draw, truth_box, "真の精度 99.54%", font(34, bold=True), fill=YELLOW)
    arrow(draw, (label_box[2], (label_box[1] + label_box[3]) // 2),
          (truth_box[0], (truth_box[1] + truth_box[3]) // 2), color=YELLOW, width=5)

    draw_credit(draw)
    return img


# ============================================================
# 07: AI駆動開発体制
# ============================================================
def slide_07_ai_team() -> Image.Image:
    img, draw = new_canvas()
    draw_title_bar(draw, "AI駆動開発体制")

    human_box = (CANVAS_W // 2 - 460, 260, CANVAS_W // 2 + 460, 380)
    rounded_box(draw, human_box, edge_color=YELLOW, width=5)
    centered_text(draw, human_box, "人間", font(36, bold=True), fill=YELLOW, y_offset=-22)
    centered_text(draw, human_box, "要件定義・採否判断・ドメイン知識", font(24), fill=GRAY, y_offset=22)

    # 絵文字はmeiryo.ttcにグリフが無く文字化け(tofu)するため、
    # 役割ごとの色付き丸マーカーをPIL描画で代替する。
    labels = ["設計", "実装", "テスト", "分析"]
    marker_colors = [BLUE, YELLOW, RED, BLUE]
    box_w, box_h = 360, 220
    gap = 60
    marker_r = 14
    total_w = 4 * box_w + 3 * gap
    start_x = (CANVAS_W - total_w) // 2
    y2 = 520
    for i, (label, marker_color) in enumerate(zip(labels, marker_colors)):
        x = start_x + i * (box_w + gap)
        box = (x, y2, x + box_w, y2 + box_h)
        rounded_box(draw, box, edge_color=BLUE, width=4)
        mcx = (x + x + box_w) // 2
        mcy = y2 + 46
        draw.ellipse(
            (mcx - marker_r, mcy - marker_r, mcx + marker_r, mcy + marker_r),
            fill=marker_color,
        )
        centered_text(draw, box, label, font(34, bold=True), y_offset=8)
        centered_text(draw, box, "AIエージェント", font(20), fill=GRAY, y_offset=52)
        arrow(draw, ((human_box[0] + human_box[2]) // 2, human_box[3]),
              ((x + x + box_w) // 2, y2), color=YELLOW, width=3)

    gate_box = (140, 840, CANVAS_W - 140, 940)
    rounded_box(draw, gate_box, edge_color=RED, width=4)
    centered_text(draw, gate_box, "品質ゲート: テスト4,273本 + 数値ゲート + 全域無悪化チェック",
                  font(28, bold=True), fill=RED)

    draw_credit(draw)
    return img


# ============================================================
# 08: 数字まとめ
# ============================================================
def slide_08_numbers() -> Image.Image:
    img, draw = new_canvas()
    draw_title_bar(draw, "数字でみるプロジェクト")

    cells = [
        ("認識精度", "99.54%"),
        ("処理速度", "31fps"),
        ("学習動画", "146本"),
        ("指標", "45種類"),
        ("テスト", "4,273本"),
        ("コード", "12.5万行\n(テスト6.7万含む)"),
    ]
    cols, rows = 3, 2
    margin_x, margin_y = 140, 260
    gap = 40
    cell_w = (CANVAS_W - 2 * margin_x - (cols - 1) * gap) // cols
    cell_h = 300
    for i, (label, value) in enumerate(cells):
        r, c = divmod(i, cols)
        x = margin_x + c * (cell_w + gap)
        y = margin_y + r * (cell_h + gap)
        box = (x, y, x + cell_w, y + cell_h)
        rounded_box(draw, box, edge_color=BLUE, width=4)
        draw.text((x + 32, y + 30), label, font=font(30), fill=GRAY)
        value_font = font(56, bold=True)
        lines = value.split("\n")
        vy = y + 120
        for line in lines:
            draw.text((x + 32, vy), line, font=value_font if len(lines) == 1 else font(40, bold=True), fill=WHITE)
            vy += 60

    draw_credit(draw)
    return img


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    slides = [
        ("01_title.png", slide_01_title),
        ("02_state_machine.png", slide_02_state_machine),
        ("03_hybrid_flow.png", slide_03_hybrid_flow),
        ("04_three_enemies.png", slide_04_three_enemies),
        ("05_accident_catalog.png", slide_05_accident_catalog),
        ("06_self_consistency_trap.png", slide_06_self_consistency_trap),
        ("07_ai_team.png", slide_07_ai_team),
        ("08_numbers.png", slide_08_numbers),
    ]
    for name, fn in slides:
        img = fn()
        assert img.size == (CANVAS_W, CANVAS_H)
        out_path = os.path.join(OUT_DIR, name)
        img.save(out_path)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

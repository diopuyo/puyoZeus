# -*- coding: utf-8 -*-
"""YouTubeサムネ生成 (2026-08-11 user指示)"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1280, 720
PUYO = [(230, 57, 70), (38, 110, 230), (67, 170, 60), (240, 180, 20), (150, 60, 200)]
F = "/mnt/c/Windows/Fonts/YuGothB.ttc"

# 背景: 評価パネル付き中盤フレーム
bg = Image.open("data/verify/youtube_demo_2026-08-07/release/panel_layout_midframe.png").convert("RGB")
bg = bg.resize((W, H), Image.LANCZOS)

# 下部に向かって暗くなるグラデーション (文字のコントラスト確保)
grad = Image.new("L", (1, H), 0)
for y in range(H):
    v = 0
    if y > H * 0.45:
        v = int(200 * ((y - H * 0.45) / (H * 0.55)) ** 1.2)
    grad.putpixel((0, y), min(v, 200))
grad = grad.resize((W, H))
black = Image.new("RGB", (W, H), (0, 0, 0))
bg = Image.composite(black, bg, grad)

d = ImageDraw.Draw(bg)

def outlined(draw, xy, text, font, fill, outline, ow, anchor="lm"):
    x, y = xy
    for dx in range(-ow, ow + 1, 2):
        for dy in range(-ow, ow + 1, 2):
            if dx * dx + dy * dy <= ow * ow:
                draw.text((x + dx, y + dy), text, font=font, fill=outline, anchor=anchor)
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)

# 1段目: ぷよぷよ盤面評価AI (白+紺フチ、大)
f1 = ImageFont.truetype(F, 96)
outlined(d, (48, 500), "ぷよぷよ盤面評価AI", f1,
         fill=(255, 255, 255), outline=(15, 25, 60), ow=12)

# 2段目: puyoZeus (1文字ずつぷよ5色+白フチ+黒フチ、特大)
f2 = ImageFont.truetype(F, 150)
text = "puyoZeus"
widths = [d.textlength(ch, font=f2) for ch in text]
x = 44
ybase = 630
for i, (ch, w) in enumerate(zip(text, widths)):
    for dx in range(-12, 13, 3):
        for dy in range(-12, 13, 3):
            if dx * dx + dy * dy <= 144:
                d.text((x + dx, ybase + dy), ch, font=f2, fill=(255, 255, 255), anchor="lm")
    for dx in range(-5, 6, 2):
        for dy in range(-5, 6, 2):
            d.text((x + dx, ybase + dy), ch, font=f2, fill=(20, 20, 20), anchor="lm")
    d.text((x, ybase), ch, font=f2, fill=PUYO[i % 5], anchor="lm")
    x += w

out = "data/verify/slides_2026-08-10/youtube_thumbnail.png"
bg.save(out)
print("saved:", out)

# -*- coding: utf-8 -*-
"""タイトルスライド生成: 白基調 + 派手なレタリング (2026-08-10 user指示)"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math, random

W, H = 1920, 1080
PUYO = [(230, 57, 70), (38, 110, 230), (67, 170, 60), (240, 180, 20), (150, 60, 200)]

random.seed(7)
bg = Image.new("RGB", (W, H), (255, 255, 255))
bd = ImageDraw.Draw(bg)
for i in range(30):
    c = PUYO[i % 5]
    pale = tuple(int(255 - (255 - v) * 0.13) for v in c)
    r = random.randint(40, 150)
    x, y = random.randint(-50, W + 50), random.randint(-50, H + 50)
    if (240 < y < 840 and 200 < x < 1720) or y > H - 260:
        continue
    bd.ellipse([x - r, y - r, x + r, y + r], fill=pale)
img = bg.filter(ImageFilter.GaussianBlur(6))
d = ImageDraw.Draw(img)

F = "/mnt/c/Windows/Fonts/YuGothB.ttc"
f_sub = ImageFont.truetype(F, 88)
f_main = ImageFont.truetype(F, 210)
f_dev = ImageFont.truetype(F, 130)

def outlined(draw, xy, text, font, fill, outline, ow, anchor="mm"):
    x, y = xy
    for dx in range(-ow, ow + 1, 2):
        for dy in range(-ow, ow + 1, 2):
            if dx * dx + dy * dy <= ow * ow:
                draw.text((x + dx, y + dy), text, font=font, fill=outline, anchor=anchor)
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)

# 集中線 (先に描いて文字を上に載せる)
for ang in range(0, 360, 15):
    if 60 < ang < 120 or 240 < ang < 300:
        continue
    rad = math.radians(ang)
    cx, cy = W // 2, H // 2
    d.line([cx + math.cos(rad) * 780, cy + math.sin(rad) * 570,
            cx + math.cos(rad) * 1400, min(cy + math.sin(rad) * 980, H - 170)],
           fill=(255, 213, 70), width=6)

outlined(d, (W // 2, 250), "ぷよぷよ対戦判定ツール", f_sub,
         fill=(25, 40, 90), outline=(255, 255, 255), ow=10)

text = "puyoZeus"
widths = [d.textlength(ch, font=f_main) for ch in text]
total = sum(widths)
x = (W - total) / 2
ybase = 460
shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
sd = ImageDraw.Draw(shadow)
xx = x
for ch, w in zip(text, widths):
    sd.text((xx + 12, ybase + 16), ch, font=f_main, fill=(0, 0, 0, 120), anchor="lm")
    xx += w
shadow = shadow.filter(ImageFilter.GaussianBlur(8))
img.paste(Image.new("RGB", (W, H), (40, 40, 40)), (0, 0), shadow)
d = ImageDraw.Draw(img)
xx = x
for i, (ch, w) in enumerate(zip(text, widths)):
    for dx in range(-9, 10, 3):
        for dy in range(-9, 10, 3):
            if dx * dx + dy * dy <= 81:
                d.text((xx + dx, ybase + dy), ch, font=f_main, fill=(255, 255, 255), anchor="lm")
    for dx in range(-4, 5, 2):
        for dy in range(-4, 5, 2):
            d.text((xx + dx, ybase + dy), ch, font=f_main, fill=(30, 30, 30), anchor="lm")
    d.text((xx, ybase), ch, font=f_main, fill=PUYO[i % 5], anchor="lm")
    xx += w

dev = Image.new("RGBA", (1500, 340), (0, 0, 0, 0))
dd = ImageDraw.Draw(dev)
outlined(dd, (750, 170), "絶賛開発中！！", f_dev,
         fill=(235, 80, 20), outline=(255, 255, 255), ow=12)
dev = dev.rotate(3, expand=True, resample=Image.BICUBIC)
img.paste(dev, ((W - dev.width) // 2, 620), dev)

out = "data/verify/slides_2026-08-10/slide_01_title.png"
img.save(out)
print("saved:", out)

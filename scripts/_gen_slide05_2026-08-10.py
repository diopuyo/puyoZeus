# -*- coding: utf-8 -*-
"""今後についてスライド (2026-08-10 user指示)"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

W, H = 1920, 1080
PUYO = [(230, 57, 70), (38, 110, 230), (67, 170, 60), (240, 180, 20), (150, 60, 200)]
NAVY = (25, 40, 90)
GRAY = (130, 135, 145)
F = "/mnt/c/Windows/Fonts/YuGothB.ttc"

random.seed(23)
bg = Image.new("RGB", (W, H), (255, 255, 255))
bd = ImageDraw.Draw(bg)
for i in range(30):
    c = PUYO[i % 5]
    pale = tuple(int(255 - (255 - v) * 0.13) for v in c)
    r = random.randint(40, 140)
    x, y = random.randint(-50, W + 50), random.randint(-50, H + 50)
    if (140 < y < 880 and 100 < x < 1820) or y > H - 260:
        continue
    bd.ellipse([x - r, y - r, x + r, y + r], fill=pale)
img = bg.filter(ImageFilter.GaussianBlur(6))
d = ImageDraw.Draw(img)

f_title = ImageFont.truetype(F, 92)
f_head = ImageFont.truetype(F, 54)
f_body = ImageFont.truetype(F, 38)
f_small = ImageFont.truetype(F, 33)

def outlined(draw, xy, text, font, fill, outline, ow, anchor="mm"):
    x, y = xy
    for dx in range(-ow, ow + 1, 2):
        for dy in range(-ow, ow + 1, 2):
            if dx * dx + dy * dy <= ow * ow:
                draw.text((x + dx, y + dy), text, font=font, fill=outline, anchor=anchor)
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)

outlined(d, (W // 2, 120), "今後について", f_title,
         fill=NAVY, outline=(255, 255, 255), ow=10)

def row_card(y0, y1, accent, head, lines, line_colors=None):
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.rounded_rectangle([130 + 8, y0 + 10, 1790 + 8, y1 + 10], 28, fill=(0, 0, 0, 55))
    img.paste(Image.new("RGB", (W, H), (0, 0, 0)), (0, 0), sh.filter(ImageFilter.GaussianBlur(7)))
    dd = ImageDraw.Draw(img)
    dd.rounded_rectangle([130, y0, 1790, y1], 28, fill=(255, 255, 255), outline=accent, width=7)
    # 左の見出し帯
    dd.rounded_rectangle([130, y0, 470, y1], 28, fill=accent)
    dd.rectangle([440, y0, 470, y1], fill=accent)
    # 見出しが帯に収まるようフォントを自動縮小
    fs = 54
    while fs > 30:
        fh = ImageFont.truetype(F, fs)
        if dd.textlength(head, font=fh) <= 300:
            break
        fs -= 2
    dd.text((300, (y0 + y1) // 2), head, font=ImageFont.truetype(F, fs), fill=(255, 255, 255), anchor="mm")
    # 本文
    n = len(lines)
    for i, line in enumerate(lines):
        cy = (y0 + y1) // 2 + (i - (n - 1) / 2) * 52
        color = (line_colors or [NAVY] * n)[i]
        dd.text((520, cy), line, font=f_body if len(line) < 46 else f_small,
                fill=color, anchor="lm")

# ① 判定の強化 (赤)
row_card(210, 420, PUYO[0], "判定の強化",
         ["少し自信過剰な値を出すことがある → 較正で実態に合わせる",
          "「色ぷよが盤面に多いほど強い」という人間同士の試合では",
          "定石とも言える部分を、まだ学びきれていない"],
         [NAVY, NAVY, PUYO[0]])

# ② アプリ化 (青)
row_card(450, 660, PUYO[1], "アプリ化",
         ["OBS に乗せてリアルタイムで動かす",
          "認識は既に 1秒31フレーム で読み取り可能",
          "有利不利判定は 1秒に2〜3回 の更新を予定"],
         [NAVY, PUYO[1], PUYO[1]])

# ③ 学習が進めば… (緑)
row_card(690, 880, PUYO[2], "学習が進めば…",
         ["人間の感覚 vs 統計データ —",
          "将棋AIやセイバーメトリクスが辿ってきた道を行けるのでは"],
         [NAVY, PUYO[2]])

out = "data/verify/slides_2026-08-10/slide_05_future.png"
img.save(out)
print("saved:", out)

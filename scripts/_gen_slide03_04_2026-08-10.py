# -*- coding: utf-8 -*-
"""評価の仕方スライド2枚 (2026-08-10 user指示)"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

W, H = 1920, 1080
PUYO = [(230, 57, 70), (38, 110, 230), (67, 170, 60), (240, 180, 20), (150, 60, 200)]
NAVY = (25, 40, 90)
GRAY = (130, 135, 145)
F = "/mnt/c/Windows/Fonts/YuGothB.ttc"

def base_bg(seed):
    random.seed(seed)
    bg = Image.new("RGB", (W, H), (255, 255, 255))
    bd = ImageDraw.Draw(bg)
    for i in range(30):
        c = PUYO[i % 5]
        pale = tuple(int(255 - (255 - v) * 0.13) for v in c)
        r = random.randint(40, 140)
        x, y = random.randint(-50, W + 50), random.randint(-50, H + 50)
        if (150 < y < 860 and 120 < x < 1800) or y > H - 260:
            continue
        bd.ellipse([x - r, y - r, x + r, y + r], fill=pale)
    return bg.filter(ImageFilter.GaussianBlur(6))

def outlined(draw, xy, text, font, fill, outline, ow, anchor="mm"):
    x, y = xy
    for dx in range(-ow, ow + 1, 2):
        for dy in range(-ow, ow + 1, 2):
            if dx * dx + dy * dy <= ow * ow:
                draw.text((x + dx, y + dy), text, font=font, fill=outline, anchor=anchor)
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)

def card(img, x0, y0, x1, y1, accent, header, header_fs=64):
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.rounded_rectangle([x0 + 10, y0 + 12, x1 + 10, y1 + 12], 36, fill=(0, 0, 0, 60))
    img.paste(Image.new("RGB", (W, H), (0, 0, 0)), (0, 0), sh.filter(ImageFilter.GaussianBlur(8)))
    dd = ImageDraw.Draw(img)
    dd.rounded_rectangle([x0, y0, x1, y1], 36, fill=(255, 255, 255), outline=accent, width=8)
    dd.rounded_rectangle([x0, y0, x1, y0 + 130], 36, fill=accent)
    dd.rectangle([x0, y0 + 95, x1, y0 + 130], fill=accent)
    dd.text(((x0 + x1) // 2, y0 + 65), header, font=ImageFont.truetype(F, header_fs),
            fill=(255, 255, 255), anchor="mm")
    return dd

f_title = ImageFont.truetype(F, 92)
f_body = ImageFont.truetype(F, 46)
f_body_s = ImageFont.truetype(F, 40)
f_big = ImageFont.truetype(F, 72)
f_badge = ImageFont.truetype(F, 56)

# ---------- 1枚目: アプローチは大きくふたつ ----------
img = base_bg(13)
d = ImageDraw.Draw(img)
outlined(d, (W // 2, 130), "評価の仕方 — アプローチは大きくふたつ", f_title,
         fill=NAVY, outline=(255, 255, 255), ow=10)

CT, CB = 260, 800
# 左: 最強AI方式 (グレー・不採用)
c1 = card(img, 130, CT, 930, CB, GRAY, "① 最強AIに従う")
c1.text((530, CT + 230), "めっちゃ強いAIを作って", font=f_body, fill=NAVY, anchor="mm")
c1.text((530, CT + 300), "そいつの評価に従う", font=f_body, fill=NAVY, anchor="mm")
c1.text((530, CT + 410), "(今の将棋AIはこの方式)", font=f_body_s, fill=GRAY, anchor="mm")

# 右: データ逆算方式 (赤・採用)
c2 = card(img, 990, CT, 1790, CB, PUYO[0], "② 勝敗データから逆算")
c2.text((1390, CT + 230), "大量の対戦データの", font=f_body, fill=NAVY, anchor="mm")
c2.text((1390, CT + 300), "勝敗から逆算する", font=f_body, fill=NAVY, anchor="mm")

# 採用バッジ (右カードに斜め)
badge = Image.new("RGBA", (500, 160), (0, 0, 0, 0))
bd2 = ImageDraw.Draw(badge)
bd2.rounded_rectangle([10, 30, 490, 130], 24, fill=(230, 57, 70), outline=(255, 255, 255), width=6)
bd2.text((250, 80), "こちらを採用！", font=f_badge, fill=(255, 255, 255), anchor="mm")
badge = badge.rotate(-6, expand=True, resample=Image.BICUBIC)
img.paste(badge, (1180, CB - 90), badge)

img.save("data/verify/slides_2026-08-10/slide_03_approach.png")
print("saved: slide_03_approach.png")

# ---------- 2枚目: モットー ----------
img = base_bg(17)
d = ImageDraw.Draw(img)
outlined(d, (W // 2, 130), "評価の仕方 — 指標づくりの考え方", f_title,
         fill=NAVY, outline=(255, 255, 255), ow=10)

# 上: 人間バイアスは補助程度
d.rounded_rectangle([200, 250, 1720, 400], 30, fill=(255, 255, 255), outline=GRAY, width=6)
d.text((W // 2, 300), "「この盤面は 1P 有利」のような人間の感覚は", font=f_body, fill=NAVY, anchor="mm")
d.text((W // 2, 360), "補助程度にしか使わない", font=f_body, fill=GRAY, anchor="mm")

# 中央: モットーバナー
sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
sd = ImageDraw.Draw(sh)
sd.rounded_rectangle([170, 470, 1760, 680], 36, fill=(0, 0, 0, 70))
img.paste(Image.new("RGB", (W, H), (0, 0, 0)), (0, 0), sh.filter(ImageFilter.GaussianBlur(10)))
d = ImageDraw.Draw(img)
d.rounded_rectangle([160, 460, 1750, 670], 36, fill=NAVY)
d.text((W // 2, 525), "モットー", font=f_body_s, fill=(255, 213, 70), anchor="mm")
d.text((W // 2, 605), "「形は手段、機能が本質」", font=f_big, fill=(255, 255, 255), anchor="mm")

# 下: セイバーメトリクス
d.text((W // 2, 750), "目標は 野球のセイバーメトリクス のような", font=f_body, fill=NAVY, anchor="mm")
d.text((W // 2, 820), "統計が作り出す指標づくり", font=f_big, fill=PUYO[0], anchor="mm")

img.save("data/verify/slides_2026-08-10/slide_04_motto.png")
print("saved: slide_04_motto.png")

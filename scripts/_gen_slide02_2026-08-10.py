# -*- coding: utf-8 -*-
"""目的スライド生成: 観戦補助+統計補助の2本柱 (2026-08-10 user指示)"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

W, H = 1920, 1080
PUYO = [(230, 57, 70), (38, 110, 230), (67, 170, 60), (240, 180, 20), (150, 60, 200)]
NAVY = (25, 40, 90)

random.seed(11)
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
img = bg.filter(ImageFilter.GaussianBlur(6))
d = ImageDraw.Draw(img)

F = "/mnt/c/Windows/Fonts/YuGothB.ttc"
f_title = ImageFont.truetype(F, 96)
f_head = ImageFont.truetype(F, 76)
f_body = ImageFont.truetype(F, 46)
f_tag = ImageFont.truetype(F, 40)

def outlined(draw, xy, text, font, fill, outline, ow, anchor="mm"):
    x, y = xy
    for dx in range(-ow, ow + 1, 2):
        for dy in range(-ow, ow + 1, 2):
            if dx * dx + dy * dy <= ow * ow:
                draw.text((x + dx, y + dy), text, font=font, fill=outline, anchor=anchor)
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)

outlined(d, (W // 2, 165), "このプロジェクトの目的", f_title,
         fill=NAVY, outline=(255, 255, 255), ow=10)

def rounded_card(x0, y0, x1, y1, accent):
    # 影
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.rounded_rectangle([x0 + 10, y0 + 12, x1 + 10, y1 + 12], 36, fill=(0, 0, 0, 60))
    blurred = sh.filter(ImageFilter.GaussianBlur(8))
    img.paste(Image.new("RGB", (W, H), (0, 0, 0)), (0, 0), blurred)
    dd = ImageDraw.Draw(img)
    dd.rounded_rectangle([x0, y0, x1, y1], 36, fill=(255, 255, 255), outline=accent, width=8)
    dd.rounded_rectangle([x0, y0, x1, y0 + 150], 36, fill=accent)
    dd.rectangle([x0, y0 + 110, x1, y0 + 150], fill=accent)
    return dd

CARD_TOP, CARD_BOT = 300, 830
# 左カード: 観戦補助 (青)
c1 = rounded_card(130, CARD_TOP, 930, CARD_BOT, PUYO[1])
c1.text((530, CARD_TOP + 75), "観戦補助", font=f_head, fill=(255, 255, 255), anchor="mm")
c1.text((530, CARD_TOP + 250), "ぷよぷよ観戦を", font=f_body, fill=NAVY, anchor="mm")
c1.text((530, CARD_TOP + 320), "もっとわかりやすく", font=f_body, fill=NAVY, anchor="mm")
c1.text((530, CARD_TOP + 430), "有利不利の指数を", font=f_body, fill=PUYO[1], anchor="mm")
c1.text((530, CARD_TOP + 500), "リアルタイム表示", font=f_body, fill=PUYO[1], anchor="mm")

# 右カード: 統計補助 (緑)
c2 = rounded_card(990, CARD_TOP, 1790, CARD_BOT, PUYO[2])
c2.text((1390, CARD_TOP + 75), "統計補助", font=f_head, fill=(255, 255, 255), anchor="mm")
c2.text((1390, CARD_TOP + 250), "人が気づけていない", font=f_body, fill=NAVY, anchor="mm")
c2.text((1390, CARD_TOP + 350), "勝敗を決める要素を", font=f_body, fill=PUYO[2], anchor="mm")
c2.text((1390, CARD_TOP + 420), "データから見つける", font=f_body, fill=PUYO[2], anchor="mm")

out = "data/verify/slides_2026-08-10/slide_02_purpose.png"
img.save(out)
print("saved:", out)

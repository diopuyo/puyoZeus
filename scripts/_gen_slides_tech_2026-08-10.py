# -*- coding: utf-8 -*-
"""技術パートスライド4枚 (2026-08-10 user指示)"""
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

W, H = 1920, 1080
PUYO = [(230, 57, 70), (38, 110, 230), (67, 170, 60), (240, 180, 20), (150, 60, 200)]
NAVY = (25, 40, 90)
GRAY = (130, 135, 145)
F = "/mnt/c/Windows/Fonts/YuGothB.ttc"

def base(seed):
    random.seed(seed)
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
    return bg.filter(ImageFilter.GaussianBlur(6))

def outlined(draw, xy, text, font, fill, outline, ow, anchor="mm"):
    x, y = xy
    for dx in range(-ow, ow + 1, 2):
        for dy in range(-ow, ow + 1, 2):
            if dx * dx + dy * dy <= ow * ow:
                draw.text((x + dx, y + dy), text, font=font, fill=outline, anchor=anchor)
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)

f_title = ImageFont.truetype(F, 88)
f_head = ImageFont.truetype(F, 50)
f_body = ImageFont.truetype(F, 40)
f_small = ImageFont.truetype(F, 34)
f_big = ImageFont.truetype(F, 64)

def shadow_rect(img, box, radius=28, alpha=55):
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.rounded_rectangle([box[0] + 8, box[1] + 10, box[2] + 8, box[3] + 10], radius, fill=(0, 0, 0, alpha))
    img.paste(Image.new("RGB", (W, H), (0, 0, 0)), (0, 0), sh.filter(ImageFilter.GaussianBlur(7)))

# ============ T1: 盤面をどう読むか ============
img = base(31); d = ImageDraw.Draw(img)
outlined(d, (W // 2, 120), "技術① 盤面をどう読んでいるか", f_title, NAVY, (255, 255, 255), 10)
shadow_rect(img, (140, 220, 1780, 400)); d = ImageDraw.Draw(img)
d.rounded_rectangle([140, 220, 1780, 400], 28, fill=(255, 255, 255), outline=PUYO[0], width=7)
d.text((W // 2, 280), "画面はほとんどの時間、何かが動いている", font=f_head, fill=NAVY, anchor="mm")
d.text((W // 2, 350), "動いている最中に読むと必ず間違える", font=f_head, fill=PUYO[0], anchor="mm")
# 状態チップ
states = [("安定", PUYO[2]), ("ツモ落下", PUYO[1]), ("連鎖中", PUYO[0]), ("おじゃま落下", PUYO[4]), ("演出", GRAY)]
cx = 200
d.text((160, 470), "「今どの状態か」を常に判定:", font=f_small, fill=NAVY, anchor="lm")
cx = 200
for name, col in states:
    wch = int(d.textlength(name, font=f_small)) + 56
    d.rounded_rectangle([cx, 510, cx + wch, 580], 22, fill=col)
    d.text((cx + wch // 2, 545), name, font=f_small, fill=(255, 255, 255), anchor="mm")
    cx += wch + 26
shadow_rect(img, (140, 640, 1780, 880)); d = ImageDraw.Draw(img)
d.rounded_rectangle([140, 640, 1780, 880], 28, fill=NAVY)
d.text((W // 2, 705), "盤面が静止した瞬間だけ、読み取りを確定", font=f_big, fill=(255, 255, 255), anchor="mm")
d.text((W // 2, 790), "連鎖中は直前の確定盤面を凍結し、物理シミュレーションで補う", font=f_body,
       fill=(255, 213, 70), anchor="mm")
img.save("data/verify/slides_2026-08-10/slide_t1_state_machine.png")
print("saved: t1")

# ============ T2: 色の認識 ============
img = base(37); d = ImageDraw.Draw(img)
outlined(d, (W // 2, 120), "技術② 色の認識は2方式の併用", f_title, NAVY, (255, 255, 255), 10)
def vcard(x0, x1, accent, head, lines, colors):
    shadow_rect(img, (x0, 230, x1, 660))
    dd = ImageDraw.Draw(img)
    dd.rounded_rectangle([x0, 230, x1, 660], 32, fill=(255, 255, 255), outline=accent, width=7)
    dd.rounded_rectangle([x0, 230, x1, 350], 32, fill=accent)
    dd.rectangle([x0, 320, x1, 350], fill=accent)
    dd.text(((x0 + x1) // 2, 290), head, font=f_head, fill=(255, 255, 255), anchor="mm")
    for i, (line, col) in enumerate(zip(lines, colors)):
        # カード幅に収まるようフォントを自動縮小
        fs = 40
        while fs > 24 and dd.textlength(line, font=ImageFont.truetype(F, fs)) > (x1 - x0) - 60:
            fs -= 2
        dd.text(((x0 + x1) // 2, 430 + i * 75), line, font=ImageFont.truetype(F, fs), fill=col, anchor="mm")
vcard(140, 930, PUYO[1], "色を数値で見る (HSV)",
      ["速い", "でも ぷよの光沢 (白いテカリ) を", "「空きマス」と誤読しやすい"],
      [PUYO[2], NAVY, PUYO[0]])
vcard(990, 1780, PUYO[4], "小さなニューラルネット",
      ["光沢に強い", "でも 学習していないパターン に弱い", "(空の盤面を赤ぷよだらけと誤認した事故も)"],
      [PUYO[2], PUYO[0], GRAY])
shadow_rect(img, (140, 720, 1780, 880)); d = ImageDraw.Draw(img)
d.rounded_rectangle([140, 720, 1780, 880], 28, fill=NAVY)
d.text((W // 2, 775), "得意分野が真逆なので組み合わせる", font=f_head, fill=(255, 255, 255), anchor="mm")
d.text((W // 2, 840), "人手で正解を付けた盤面との突き合わせで 精度 99.54%", font=f_body,
       fill=(255, 213, 70), anchor="mm")
img.save("data/verify/slides_2026-08-10/slide_t2_color.png")
print("saved: t2")

# ============ T3: ルールを検証に使う ============
img = base(41); d = ImageDraw.Draw(img)
outlined(d, (W // 2, 120), "技術③ ゲームのルールが検証装置になる", f_title, NAVY, (255, 255, 255), 10)
rules = [
    ("幽霊連鎖", "13段目のぷよは4つ繋がっても消えない", PUYO[4]),
    ("おじゃまの降り方", "6列に均等 + 端数はランダム", PUYO[1]),
    ("物理整合", "空中に浮いたぷよ / 物理計算と合わない連鎖中の盤面 = 誤認識", PUYO[2]),
]
y = 230
for head, body, col in rules:
    shadow_rect(img, (140, y, 1780, y + 130))
    dd = ImageDraw.Draw(img)
    dd.rounded_rectangle([140, y, 1780, y + 130], 26, fill=(255, 255, 255), outline=col, width=6)
    dd.rounded_rectangle([140, y, 560, y + 130], 26, fill=col)
    dd.rectangle([530, y, 560, y + 130], fill=col)
    dd.text((350, y + 65), head, font=f_head, fill=(255, 255, 255), anchor="mm")
    fs_b = 40
    while fs_b > 26 and dd.textlength(body, font=ImageFont.truetype(F, fs_b)) > 1140:
        fs_b -= 2
    dd.text((610, y + 65), body, font=ImageFont.truetype(F, fs_b), fill=NAVY, anchor="lm")
    y += 155
d = ImageDraw.Draw(img)
d.text((W // 2, 740), "確定ルールに反する認識結果は、その場で誤りと分かる", font=f_head, fill=NAVY, anchor="mm")
shadow_rect(img, (140, 790, 1780, 890)); d = ImageDraw.Draw(img)
d.rounded_rectangle([140, 790, 1780, 890], 24, fill=PUYO[0])
d.text((W // 2, 840), "AIはルールを平気で取り違える → ルールの確定は必ず人間がやる", font=f_body,
       fill=(255, 255, 255), anchor="mm")
img.save("data/verify/slides_2026-08-10/slide_t3_rules.png")
print("saved: t3")

# ============ T4: 開発の規模 ============
img = base(43); d = ImageDraw.Draw(img)
outlined(d, (W // 2, 120), "開発の規模と体制", f_title, NAVY, (255, 255, 255), 10)
stats = [
    ("開発期間", "約2.5ヶ月", PUYO[1]),
    ("コード総量", "約34万行", PUYO[0]),
    ("テスト", "4,400件+", PUYO[2]),
    ("学習データ", "148試合動画", PUYO[4]),
]
x = 150
for head, val, col in stats:
    shadow_rect(img, (x, 230, x + 390, 440))
    dd = ImageDraw.Draw(img)
    dd.rounded_rectangle([x, 230, x + 390, 440], 28, fill=(255, 255, 255), outline=col, width=7)
    dd.text((x + 195, 295), head, font=f_small, fill=GRAY, anchor="mm")
    dd.text((x + 195, 375), val, font=ImageFont.truetype(F, 58), fill=col, anchor="mm")
    x += 415
d = ImageDraw.Draw(img)
shadow_rect(img, (140, 500, 1780, 660)); d = ImageDraw.Draw(img)
d.rounded_rectangle([140, 500, 1780, 660], 28, fill=(255, 255, 255), outline=NAVY, width=7)
d.text((W // 2, 550), "特徴: 本体 6万行 に対して 検証コードが19万行", font=f_head, fill=NAVY, anchor="mm")
d.text((W // 2, 620), "「作る」より「測る」に力を割いているプロジェクト", font=f_body, fill=PUYO[0], anchor="mm")
shadow_rect(img, (140, 700, 1780, 880)); d = ImageDraw.Draw(img)
d.rounded_rectangle([140, 700, 1780, 880], 28, fill=NAVY)
d.text((W // 2, 755), "開発はAI駆動 — AIが管理役、担当AIたちが実装・測定を分業", font=f_body,
       fill=(255, 255, 255), anchor="mm")
d.text((W // 2, 825), "ただし ぷよぷよのルールの確定だけは人間 (前科があるので)", font=f_body,
       fill=(255, 213, 70), anchor="mm")
img.save("data/verify/slides_2026-08-10/slide_t4_scale.png")
print("saved: t4")

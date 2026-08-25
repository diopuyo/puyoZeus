# -*- coding: utf-8 -*-
"""開発規模2枚 (コード内訳+AI体制図) (2026-08-10 user指示)"""
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
        if (140 < y < 900 and 100 < x < 1820) or y > H - 240:
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

def shadow_rect(img, box, radius=28, alpha=55):
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.rounded_rectangle([box[0] + 8, box[1] + 10, box[2] + 8, box[3] + 10], radius, fill=(0, 0, 0, alpha))
    img.paste(Image.new("RGB", (W, H), (0, 0, 0)), (0, 0), sh.filter(ImageFilter.GaussianBlur(7)))

f_title = ImageFont.truetype(F, 84)
f_head = ImageFont.truetype(F, 46)
f_body = ImageFont.truetype(F, 38)
f_small = ImageFont.truetype(F, 31)
f_num = ImageFont.truetype(F, 44)

# ============ T4: コード構成の内訳 ============
img = base(43); d = ImageDraw.Draw(img)
outlined(d, (W // 2, 110), "開発の規模 — コードは3種類に分かれる", f_title, NAVY, (255, 255, 255), 10)
d.text((W // 2, 195), "約2.5ヶ月 / 総量 約34万行 / 学習データ148試合動画", font=f_body, fill=GRAY, anchor="mm")

rows = [
    ("本体コード", "6.0万行", 60, PUYO[1],
     "認識・指標・有利不利判定 — 製品として動く部分"),
    ("テストコード", "6.8万行", 68, PUYO[2],
     "4,400件+が常に全パス — 本体より多い。壊してはいけない仕様の永続記憶"),
    ("検証スクリプト", "19.2万行", 192, PUYO[0],
     "測定器・実験・バックテスト — 精度や効果を疑って測るための使い捨ての道具"),
]
BAR_MAX = 1000
y = 270
for name, val, amt, col, desc in rows:
    shadow_rect(img, (140, y, 1780, y + 150), 24)
    dd = ImageDraw.Draw(img)
    dd.rounded_rectangle([140, y, 1780, y + 150], 24, fill=(255, 255, 255), outline=col, width=6)
    dd.text((175, y + 45), name, font=f_head, fill=NAVY, anchor="lm")
    # バー
    bw = int(BAR_MAX * amt / 192)
    dd.rounded_rectangle([560, y + 22, 560 + bw, y + 68], 14, fill=col)
    dd.text((560 + bw + 24, y + 45), val, font=f_num, fill=col, anchor="lm")
    dd.text((175, y + 110), desc, font=f_small, fill=GRAY, anchor="lm")
    y += 175

shadow_rect(img, (140, y + 5, 1780, y + 115), 24); d = ImageDraw.Draw(img)
d.rounded_rectangle([140, y + 5, 1780, y + 115], 24, fill=NAVY)
d.text((W // 2, y + 60), "「作る」1 に対して「試す・測る」4 — 測定中心の開発", font=f_head,
       fill=(255, 213, 70), anchor="mm")
img.save("data/verify/slides_2026-08-10/slide_t4_scale.png")
print("saved: t4")

# ============ T5: AI駆動の開発体制 ============
img = base(47); d = ImageDraw.Draw(img)
outlined(d, (W // 2, 110), "開発体制 — AIチームによる分業", f_title, NAVY, (255, 255, 255), 10)

def box(x0, y0, x1, y1, fill_col, head, sub, head_col=(255, 255, 255), sub_col=(255, 255, 255), ol=None):
    shadow_rect(img, (x0, y0, x1, y1), 22)
    dd = ImageDraw.Draw(img)
    dd.rounded_rectangle([x0, y0, x1, y1], 22, fill=fill_col, outline=ol or fill_col, width=5)
    cx = (x0 + x1) // 2
    avail = (x1 - x0) - 44
    fs = 46
    while fs > 26 and dd.textlength(head, font=ImageFont.truetype(F, fs)) > avail:
        fs -= 2
    dd.text((cx, y0 + (52 if sub else (y1 - y0) // 2)), head,
            font=ImageFont.truetype(F, fs), fill=head_col, anchor="mm")
    if sub:
        for i, s in enumerate(sub):
            fs2 = 31
            while fs2 > 18 and dd.textlength(s, font=ImageFont.truetype(F, fs2)) > avail:
                fs2 -= 1
            dd.text((cx, y0 + 105 + i * 42), s, font=ImageFont.truetype(F, fs2), fill=sub_col, anchor="mm")

# 人間 (最上段)
box(560, 200, 1360, 330, (255, 255, 255), "人間 (開発者)",
    ["ぷよぷよのルール・戦術の確定 / 採否の最終決定 / 映像の目視レビュー"],
    head_col=NAVY, sub_col=PUYO[0], ol=NAVY)
# 矢印
d = ImageDraw.Draw(img)
d.line([960, 330, 960, 400], fill=NAVY, width=8)
d.polygon([(940, 395), (980, 395), (960, 425)], fill=NAVY)
# 管理役AI
box(560, 425, 1360, 555, NAVY, "管理役の AI",
    ["指令書を書き、報告を裁き、人間に判断を上げる"])
d = ImageDraw.Draw(img)
for cx in (280, 620, 960, 1300, 1640):
    d.line([960, 555, cx, 630], fill=NAVY, width=6)
    d.polygon([(cx - 16, 626), (cx + 16, 626), (cx, 652)], fill=NAVY)

agents = [
    ("アーキ", "設計・方針の審査", PUYO[4]),
    ("コーダ", "実装+互換維持", PUYO[1]),
    ("テスター", "回帰の防止", PUYO[2]),
    ("アナリスト", "数値の検収", PUYO[3]),
    ("ライブラリアン", "既存資産の調査", PUYO[0]),
]
x = 110
for name, role, col in agents:
    box(x, 655, x + 340, 800, col, name, [role])
    x += 355

shadow_rect(img, (140, 830, 1780, 925), 22); d = ImageDraw.Draw(img)
d.rounded_rectangle([140, 830, 1780, 925], 22, fill=(255, 255, 255), outline=NAVY, width=6)
_flow = "流れ: 実装 → テスト → 全域バックテスト → 人間レビュー。夜間も測定が自走する24時間運転"
_fs = 31
while _fs > 20 and d.textlength(_flow, font=ImageFont.truetype(F, _fs)) > 1560:
    _fs -= 1
d.text((W // 2, 877), _flow, font=ImageFont.truetype(F, _fs), fill=NAVY, anchor="mm")
img.save("data/verify/slides_2026-08-10/slide_t5_agents.png")
print("saved: t5")

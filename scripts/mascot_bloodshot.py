"""触角の目玉を時間で充血させる試作 (0=なし, 1=30分, 2=45分, 3=60分)。"""
import numpy as np
from PIL import Image, ImageDraw
# 目玉の白目部分 (触角の先、画面上半分の暖色寄りの白)
def sclera_mask(a: np.ndarray) -> np.ndarray:
    r, g, b, al = (a[..., k].astype(int) for k in range(4))
    m = (al > 200) & (r > 215) & (g > 185) & ((r - b) > 18)
    m[175:, :] = False
    return m
STAGES = {0: (0.0, 0, 0), 1: (0.12, 3, 7), 2: (0.24, 6, 9), 3: (0.38, 10, 11)}  # (赤み, 血管の本数, 太さ=10倍描画での画素数 → 0.7/0.9/1.1 画素) (user 10/7)
ANGLES = [200, 160, 240, 120, 280, 180, 220, 140, 260, 100]  # 黒目から見た血管の方向 (度、0=右・時計回り) — 黒目の反対側から
def iris_center(a: np.ndarray) -> tuple[float, float] | None:
    r, g, b, al = (a[..., k].astype(int) for k in range(4))
    m = (al > 200) & (r > 200) & (g > 120) & (g < 200) & (b < 90)
    m[175:, :] = False
    ys, xs = np.nonzero(m)
    return (xs.mean(), ys.mean()) if len(xs) else None
def apply(frame: Image.Image, stage: int) -> Image.Image:
    tint, n, w = STAGES[stage]
    a = np.array(frame.convert('RGBA'))
    m = sclera_mask(a)
    c = iris_center(a)
    if stage == 0 or not m.any() or c is None:
        return frame.convert('RGBA')
    red = np.array([235, 80, 80])
    a[m, :3] = (a[m, :3] * (1 - tint) + red * tint).astype(np.uint8)
    ys, xs = np.nonzero(m)
    sx, sy = xs.mean(), ys.mean()
    base = np.degrees(np.arctan2(sy - c[1], sx - c[0]))  # 黒目→白目の中心の向き
    SS = 10  # 10倍の大きさで描いて縮め、0.1画素刻みの太さにする (user 10/7)
    veins = Image.new('L', (frame.size[0] * SS, frame.size[1] * SS), 0)
    d = ImageDraw.Draw(veins)
    for k in range(n):
        ang = np.radians(base + (ANGLES[k] - 180) * 0.55)
        # 外周 (白目の端) を探し、黒目へ向かって描く
        far = None
        for t in range(4, 90):
            x, y = int(c[0] + t * np.cos(ang)), int(c[1] + t * np.sin(ang))
            if 0 <= y < a.shape[0] and 0 <= x < a.shape[1] and m[y, x]:
                far = (x, y, t)
        if far is None:
            continue
        x, y, t = far
        length = t * 0.55
        pts = [(x, y)]
        for q in (0.35, 0.7, 1.0):
            jit = (6 if q < 1 else 0) * (1 if k % 2 else -1)
            px = x - q * length * np.cos(ang) + jit * np.sin(ang) * q
            py = y - q * length * np.sin(ang) - jit * np.cos(ang) * q
            pts.append((px, py))
        d.line([(px * SS, py * SS) for px, py in pts], fill=255, width=w)
        # 枝
        bx, by = pts[1]
        d.line([(bx * SS, by * SS), ((bx - 7 * np.cos(ang + 0.8)) * SS, (by - 7 * np.sin(ang + 0.8)) * SS)], fill=255, width=max(1, w - 1))
    alpha = np.array(veins.resize(frame.size, Image.Resampling.BOX)).astype(float) / 255.0 * m
    alpha = np.clip(alpha * 1.15, 0, 1)[..., None]
    a[..., :3] = (a[..., :3] * (1 - alpha) + np.array([190, 25, 40]) * alpha).astype(np.uint8)
    return Image.fromarray(a)

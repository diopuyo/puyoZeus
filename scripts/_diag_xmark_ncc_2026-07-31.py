"""×印のUIマスクNCCスコアを誤認時(満杯)と通常時で比較実測する。

仮説: ぷよが×印の直下まで積まれると×印の見え方が変わり (背景が
ぷよになる/×印が縮む等)、NCC が閾値 0.75 を外れて素通しになる。
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
from src.ui_mask import UiMaskMatcher

m = UiMaskMatcher.load_default()
print(f"閾値 = {m._threshold}")
# (動画, region, 誤認フレーム, 通常時フレーム群)
cases = [
    ("video_c11", DEFAULT_P1_REGION, 138171, (120000, 125000, 130000)),
    ("video_c23", DEFAULT_P2_REGION, 74554, (60000, 65000, 70000)),
]
for vid, reg, bad_fr, ok_frs in cases:
    cap = cv2.VideoCapture(f"data/frames/{vid}.mp4")
    def score_at(fr):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(fr))
        ok, f = cap.read()
        if not ok: return None
        if f.shape[:2] != (1080,1920):
            f = cv2.resize(f,(1920,1080),interpolation=cv2.INTER_AREA)
        x1,y1,x2,y2 = reg.cell_sample_rect(1,2)
        r = m.match(f[y1:y2, x1:x2])
        return r
    r = score_at(bad_fr)
    print(f"\n{vid} 誤認フレーム f{bad_fr}: score={r.score:.3f} is_ui={r.is_ui} tmpl={r.template_name}")
    for fr in ok_frs:
        r = score_at(fr)
        if r: print(f"  通常時 f{fr}: score={r.score:.3f} is_ui={r.is_ui}")
    cap.release()

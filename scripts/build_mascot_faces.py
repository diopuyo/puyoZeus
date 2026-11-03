"""生成した目・口だけを元の6コマへ合成し、透過WebPと見本を作る。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / 'src/phase_j/brand'
SHEET = ROOT / 'scripts/assets/mascot_faces/generated_sheet.png'
SHEET_V2 = ROOT / 'scripts/assets/mascot_faces/generated_sheet_v2.png'
EXPORT = Path('D:/puyo_analyzer/brand/mascot_faces')
BANDS = (10, 30, 50, 70, 90)
TILE = 627
SIDE = 320
FEATURE_THRESHOLD = 170
REPAIR_RADIUS = 3
BOXES = ((210, 360, 300, 447), (370, 360, 460, 447), (288, 420, 388, 495))
CENTERS = ((121, 233), (190, 233), (156, 252))
SIZES = {10: ((23, 15), (23, 15), (22, 12)),
         30: ((16, 30), (16, 30), (30, 12)),
         70: ((22, 16), (22, 16), (26, 26)),
         90: ((19, 30), (19, 30), (30, 28))}
FACE_BOUNDS = (104, 211, 205, 272)
# 70 は目を元のまま残し、口だけ差し替える (user 10/7「あおりの方は目は普通にして」)
MOUTH_ONLY = {70, 90}
MOUTH_BOUNDS = (136, 243, 177, 272)
# 70・90 は目を元の黒目のまま、目のカーブに沿った線を入れて「ガンギマリ」に (70=1本、90=2本、user 10/7)
STARE_LINES = {70: 1, 90: 2}
# 口は 70 と 90 で入れ替え (user 10/7): 70 = 大きく開いた笑顔 (旧90)、90 = 舌を出したにやり (旧70)
MOUTH_SOURCE = {70: 90, 90: 70}
# 90 の別パターン: 星の目 (旧90の目) + 90 の口。線なし / 線2本 (user 10/7)
# 合成の材料 (目, 口, 線の本数)。目 'orig' は元の黒目のまま。既定 (user 10/7): 70 = 黒目+線1、90 = 星の目+線2
RECIPES = {70: ('orig', 90, 1), 90: ('orig', 70, 2)}
# オプション「ゼウスくんのドーパミン」(user 10/7): 低 / 普通 (既定) / 高。70・90 の顔だけ変わる
DOPAMINE = {
    'low': {70: ('orig', 'v1:70', 0), 90: ('v1:70', 90, 0)},  # 線なし、70 = 黒目+笑顔、90 = 😊 の目 (細めた目)+にっこり
    'normal': {70: ('orig', 90, 1), 90: ('orig', 70, 2)},   # 70 = 黒目+線1、90 = 黒目+線2+舌出し
    'high': {70: ('orig', 90, 2), 90: (90, 70, 4)},         # 70 = 黒目+線2、90 = 星の目+線4+舌出し
}
EXTRA: dict = {}  # 90 の別パターンは作らない (user 10/7: オプションにしなくていい)
STARE_COLOR = (35, 16, 24, 255)
STARE_WIDTH = 3
V2_BOXES = {
    10: ((210, 340, 300, 460), (375, 340, 460, 460), (295, 425, 380, 480)),
    70: ((210, 375, 300, 435), (380, 370, 460, 435), (295, 410, 380, 480)),
}
V2_SIZES = {10: ((26, 36), (26, 36), (27, 17)),
            70: ((25, 19), (25, 20), (31, 22))}
TEAR_COLOR_GAP = 30


def expression_layer_v2(sheet: Image.Image, band: int, all_parts: bool = False) -> Image.Image:
    """生成した眉・涙も目と一緒に抜き出し、顔の範囲内だけに合成する。"""
    y = 0 if band == 10 else TILE
    tile = sheet.crop((0, y, TILE, y + TILE))
    layer = Image.new('RGBA', (SIDE, SIDE))
    parts = list(zip(V2_BOXES[band], CENTERS, V2_SIZES[band]))
    if band in MOUTH_ONLY and not all_parts:
        parts = parts[2:]
    for box, center, size in parts:
        rgb = np.array(tile.crop(box).convert('RGB'))
        dark = (rgb[:, :, 0] < FEATURE_THRESHOLD) & (rgb[:, :, 1] < FEATURE_THRESHOLD)
        blue = (rgb[:, :, 0] < FEATURE_THRESHOLD) & (
            rgb[:, :, 2].astype(int) - rgb[:, :, 0] > TEAR_COLOR_GAP)
        mask = (dark | blue).astype('uint8') * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
        part = Image.fromarray(np.dstack((rgb, mask)))
        part = part.crop(part.getbbox()).resize(size, Image.Resampling.LANCZOS)
        layer.alpha_composite(part, (center[0] - size[0] // 2, center[1] - size[1] // 2))
    return layer


def feature(tile: Image.Image, box: tuple[int, ...], size: tuple[int, int]) -> Image.Image:
    """顔の淡青色の地を除外し、目の星・口の内側は保持する。"""
    rgb = np.array(tile.crop(box).convert('RGB'))
    mask = ((rgb[:, :, 0] < FEATURE_THRESHOLD) & (rgb[:, :, 1] < FEATURE_THRESHOLD)).astype('uint8') * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask.fill(0)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    rgba = Image.fromarray(np.dstack((rgb, mask)))
    return rgba.crop(rgba.getbbox()).resize(size, Image.Resampling.LANCZOS)


def expression_layer(sheet: Image.Image, band: int, all_parts: bool = False) -> Image.Image:
    """各目・口の中心を原画と揃え、キャラクター全体の描き直しを避ける。"""
    index = (10, 30, 70, 90).index(band)
    x, y = index % 2 * TILE, index // 2 * TILE
    tile = sheet.crop((x, y, x + TILE, y + TILE))
    layer = Image.new('RGBA', (SIDE, SIDE))
    parts = list(zip(BOXES, CENTERS, SIZES[band]))
    if band in MOUTH_ONLY and not all_parts:
        parts = parts[2:]
    for box, center, size in parts:
        part = feature(tile, box, size)
        layer.alpha_composite(part, (center[0] - size[0] // 2, center[1] - size[1] // 2))
    return layer


def part_layer(sheet: Image.Image, sheet_v2: Image.Image, src_band: int, which: str) -> Image.Image:
    """生成シートの指定帯から目 (eyes: 左右の目) か口 (mouth) の部品だけを、元の位置に合成した層。
    src_band を 'v1:70' のように書くと、v2 がある帯でも v1 シートの部品を使う。"""
    force_v1 = isinstance(src_band, str)
    if force_v1:
        src_band = int(src_band.split(':')[1])
    if src_band in V2_BOXES and not force_v1:
        y = 0 if src_band == 10 else TILE
        tile = sheet_v2.crop((0, y, TILE, y + TILE))
        parts = list(zip(V2_BOXES[src_band], CENTERS, V2_SIZES[src_band]))
        extract = v2_part
    else:
        index = (10, 30, 70, 90).index(src_band)
        x, y = index % 2 * TILE, index // 2 * TILE
        tile = sheet.crop((x, y, x + TILE, y + TILE))
        parts = list(zip(BOXES, CENTERS, SIZES[src_band]))
        extract = feature
    parts = parts[:2] if which == 'eyes' else parts[2:]
    layer = Image.new('RGBA', (SIDE, SIDE))
    for box, center, size in parts:
        part = extract(tile, box, size)
        layer.alpha_composite(part, (center[0] - size[0] // 2, center[1] - size[1] // 2))
    return layer


def v2_part(tile: Image.Image, box: tuple[int, ...], size: tuple[int, int]) -> Image.Image:
    """v2 シートの部品 (眉・涙を含む) を切り出す。"""
    rgb = np.array(tile.crop(box).convert('RGB'))
    dark = (rgb[:, :, 0] < FEATURE_THRESHOLD) & (rgb[:, :, 1] < FEATURE_THRESHOLD)
    blue = (rgb[:, :, 0] < FEATURE_THRESHOLD) & (rgb[:, :, 2].astype(int) - rgb[:, :, 0] > TEAR_COLOR_GAP)
    mask = (dark | blue).astype('uint8') * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
    part = Image.fromarray(np.dstack((rgb, mask)))
    return part.crop(part.getbbox()).resize(size, Image.Resampling.LANCZOS)


def recipe_layer(sheet: Image.Image, sheet_v2: Image.Image, recipe: tuple) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """(目, 口, 線の本数) から顔の層と、元画像で消す範囲を作る。"""
    eyes, mouth, lines = recipe
    layer = part_layer(sheet, sheet_v2, mouth, 'mouth')
    if eyes != 'orig':
        layer.alpha_composite(part_layer(sheet, sheet_v2, eyes, 'eyes'))
    if lines:
        layer = stare_lines(layer, lines)
    return layer, (MOUTH_BOUNDS if eyes == 'orig' else FACE_BOUNDS)


def stare_lines(layer: Image.Image, count: int) -> Image.Image:
    """両目の外側に、目のカーブに沿った弧を count 本ずつ描く (ガンギマリ)。外側ほど短い。"""
    draw = ImageDraw.Draw(layer)
    (lx, ly), (rx, ry) = CENTERS[0], CENTERS[1]
    for k in range(count):
        half_w, half_h, spread = 16 + 5 * k, 17 + 4 * k, 38 - 6 * k
        draw.arc((lx - half_w, ly - half_h, lx + half_w, ly + half_h), 180 - spread, 180 + spread,
                 fill=STARE_COLOR, width=STARE_WIDTH)
        draw.arc((rx - half_w, ry - half_h, rx + half_w, ry + half_h), -spread, spread,
                 fill=STARE_COLOR, width=STARE_WIDTH)
    return layer


def clean_face(frame: Image.Image, bounds: tuple[int, int, int, int] = FACE_BOUNDS) -> Image.Image:
    """元の小さな目・口の画素だけを周囲の淡青色で補間する。"""
    rgba = np.array(frame)
    left, top, right, bottom = bounds
    region = rgba[top:bottom, left:right, :3]
    mask = ((region[:, :, 0] < FEATURE_THRESHOLD) & (region[:, :, 1] < FEATURE_THRESHOLD)).astype('uint8') * 255
    mask = cv2.dilate(mask, np.ones((REPAIR_RADIUS, REPAIR_RADIUS), dtype='uint8'))
    repaired = cv2.inpaint(region, mask, REPAIR_RADIUS, cv2.INPAINT_TELEA)
    rgba[top:bottom, left:right, :3] = repaired
    return Image.fromarray(rgba)


def preview(frames: dict[int, Image.Image], destination: Path) -> None:
    """実際に同梱する先頭コマを横並びで確認できるようにする。"""
    result = Image.new('RGB', (1600, 410), '#172233')
    draw = ImageDraw.Draw(result)
    font = ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc', 19)
    labels = ('10｜大幅不利', '30｜不利', '50｜中央・元の顔', '70｜有利', '90｜大幅有利')
    ranges = ('1P：10%未満', '1P：10%以上30%未満', '1P：30%以上70%未満', '1P：70%以上90%未満', '1P：90%以上')
    for index, (band, label) in enumerate(zip(BANDS, labels)):
        result.paste(frames[band], (index * 320, 24), frames[band])
        draw.text((index * 320 + 60, 353), label, font=font, fill='white')
        draw.text((index * 320 + 35, 382), ranges[index], font=font, fill='white')
    result.save(destination)


def build(export: Path) -> None:
    """時間・ループ・透過・顔以外の画素を元画像から引き継ぐ。"""
    source, sheet = Image.open(BRAND / 'overlay_mascot.webp'), Image.open(SHEET)
    sheet_v2 = Image.open(SHEET_V2)
    if sheet.size != (TILE * 2, TILE * 2) or sheet_v2.size != (TILE * 2, TILE * 2):
        raise ValueError('生成元シートは1254×1254pxで指定してください')
    frames, durations = [], []
    for index in range(source.n_frames):
        source.seek(index)
        frames.append(source.convert('RGBA').copy())
        durations.append(source.info['duration'])
    export.mkdir(parents=True, exist_ok=True)
    first = {}
    for band in BANDS:
        path = BRAND / f'mascot_{band}.webp'
        if band == 50:
            shutil.copyfile(BRAND / 'overlay_mascot.webp', path)
        else:
            if band in RECIPES:
                layer, bounds = recipe_layer(sheet, sheet_v2, RECIPES[band])
            else:
                layer = expression_layer_v2(sheet_v2, band) if band in V2_BOXES else expression_layer(sheet, band)
                bounds = FACE_BOUNDS
            changed = [Image.alpha_composite(clean_face(frame, bounds), layer) for frame in frames]
            changed[0].save(path, save_all=True, append_images=changed[1:],
                            duration=durations, loop=source.info['loop'], lossless=True, exact=True)
        shutil.copyfile(path, export / f'{band}.webp')
        first[band] = Image.open(path).convert('RGBA')
    for level, recipes in DOPAMINE.items():
        for band, recipe in recipes.items():
            EXTRA[f'{band}_{level}'] = recipe
    for name, recipe in EXTRA.items():
        layer, bounds = recipe_layer(sheet, sheet_v2, recipe)
        changed = [Image.alpha_composite(clean_face(frame, bounds), layer) for frame in frames]
        path = BRAND / f'mascot_{name}.webp'
        changed[0].save(path, save_all=True, append_images=changed[1:],
                        duration=durations, loop=source.info['loop'], lossless=True, exact=True)
        shutil.copyfile(path, export / f'{name}.webp')
    preview(first, export / 'mascot_expressions_preview.png')
    shutil.copyfile(export / 'mascot_expressions_preview.png', ROOT / 'docs/mascot_expressions_preview.png')


def build_bloodshot() -> None:
    """元の表情資産を再利用し、全コマの触角の白目だけを加工する。"""
    from scripts.mascot_bloodshot import apply
    names = ['overlay_mascot'] + [f'mascot_{band}' for band in BANDS]
    names += [f'mascot_{band}_{level}' for band in (70, 90) for level in DOPAMINE]
    for name in names:
        source = Image.open(BRAND / (name + '.webp'))
        frames, durations = [], []
        for index in range(source.n_frames):
            source.seek(index)
            frames.append(source.convert('RGBA').copy())
            durations.append(source.info['duration'])
        for stage in (1, 2, 3):
            changed = [apply(frame, stage) for frame in frames]
            changed[0].save(BRAND / f'{name}_b{stage}.webp', save_all=True,
                append_images=changed[1:], duration=durations, loop=0, lossless=True, exact=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, default=EXPORT)
    parser.add_argument('--bloodshot-only', action='store_true')
    args = parser.parse_args()
    if not args.bloodshot_only:
        build(args.export)
    build_bloodshot()

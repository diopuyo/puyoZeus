"""大きな日本語表示と、過去を書き換えない2本の勝率グラフ。"""
from __future__ import annotations

import json
import math
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Callable
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scripts.exchange_review_timeline import read_csv
from scripts.exchange_review_io import save
from src.b3_review_panel import default_font
from src.b3_review_video import ffmpeg_exe

WIDTH, VIDEO_HEIGHT, BAND_HEIGHT = 1280, 720, 720
FONT, SMALL, LINE = 32, 27, 47
COLORS = [(240, 240, 250), (255, 180, 70)]
BG, TEXT = (18, 24, 34), (240, 244, 250)
GRAPH = (100, 425, 1230, 665)
AXIS_SECONDS, LIMIT_BYTES = 180., 30_000_000
KEYS = ('b3_probability', 'exchange_probability')
MOBILE_WIDTH, MOBILE_FPS, MOBILE_CRF = 960, 30, 30


def number(v: float | None, percent: bool = False) -> str:
    return '—' if v is None else (f'{v * 100:.1f}%' if percent else f'{v:.1f}')


def lines(row: dict, start: float) -> list[str]:
    result = [f"動画80・第7試合  {row['t_sec'] - start:.1f}秒  1Pの勝率",
              f"現行 B3   {number(row['b3_probability'], True)}",
              f"撃ち合い反映（1秒版）   {number(row['exchange_probability'], True)}",
              f"{row['receiver']}の返せる量   {number(row['predicted_return'])}個",
              f"判断：{row['policy']}   [{row['route']}]   " + ('打切り・暫定' if row['truncated'] else '')]
    for side in ('1P', '2P'):
        amount, land = row.get(f'{side}_actual_amount'), row.get(f'{side}_landing')
        policy = {'main': '本線', 'counter': '対応'}.get(row.get(f'{side}_actual_policy'), '不明')
        result.append(f"実際 {side} 返した量 {number(amount)}個 ({policy}) / 確認着地 {number(land)}個")
    result.append('白 B3 / 橙 撃ち合い反映 ｜ 印：赤 発火・緑 着地・紫 再帰到着')
    return result


@lru_cache(maxsize=2048)
def text_tile(text: str, size: int, color: tuple) -> Image.Image:
    font = default_font(size)
    if font.getlength(text) > WIDTH - 40:
        raise ValueError(f'表示幅超過: {text}')
    tile = Image.new('RGB', (WIDTH - 40, LINE), BG)
    ImageDraw.Draw(tile).text((0, 0), text, font=font, fill=color)
    return tile


def draw_band(row: dict, start: float, history: list, marks: list,
              line_builder: Callable = lines, colors: list = COLORS) -> tuple[np.ndarray, dict]:
    img = Image.new('RGB', (WIDTH, BAND_HEIGHT), BG)
    texts = line_builder(row, start)
    for i, text in enumerate(texts):
        color = colors[0] if i != 2 else colors[1]
        img.paste(text_tile(text, SMALL if i == len(texts) - 1 else FONT, color), (20, 8 + i * LINE))
    draw, font = ImageDraw.Draw(img), default_font(SMALL)
    left, top, right, bottom = GRAPH
    x = lambda t: left + (right - left) * (t - start) / AXIS_SECONDS
    y = lambda p: bottom - p * (bottom - top)
    for tick in (0., .5, 1.):
        draw.line((left, y(tick), right, y(tick)), fill=(65, 75, 90))
        draw.text((5, y(tick) - SMALL / 2), f'{tick * 100:.0f}%', font=font, fill=TEXT)
    for sec in range(0, int(AXIS_SECONDS) + 1, 30):
        draw.text((min(right - 55, x(start + sec) - 15), bottom + 6), f'{sec}秒', font=font, fill=TEXT)
    for j, color in enumerate(colors):
        points = [(x(t), y(v[j])) for t, v in history if v[j] is not None]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        if points:
            px, py = points[-1]
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
    draw.line((x(row['t_sec']), top, x(row['t_sec']), bottom), fill=(240, 220, 120), width=2)
    for stamp, kind in marks:
        px = x(stamp)
        color = {'発火': (255, 90, 90), '着地': (90, 240, 120), '再帰': (200, 130, 255),
                 '暫定': (255, 180, 70), '最終': (80, 220, 240)}[kind]
        draw.ellipse((px - 4, top - 13, px + 4, top - 5), fill=color)
    receipt = dict(frame=row['frame_index'], texts=texts, graph_last=history[-1],
                   graph_count=len(history), marks=marks.copy())
    return np.asarray(img)[:, :, ::-1], receipt


def encoder(path: Path, fps: float, pc: bool) -> subprocess.Popen:
    cmd = [ffmpeg_exe(), '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s',
           f'{WIDTH}x{VIDEO_HEIGHT + BAND_HEIGHT}', '-r', str(fps), '-i', '-']
    if pc:
        cmd += ['-vf', 'scale=1920:2160']
    else:
        cmd += ['-vf', f'scale={MOBILE_WIDTH}:-2,fps={MOBILE_FPS}:round=up']
    cmd += ['-c:v', 'libx264', '-preset', 'fast']
    cmd += (['-b:v', '1350k', '-maxrate', '1800k', '-bufsize', '2700k'] if pc
            else ['-crf', str(MOBILE_CRF)])
    cmd += ['-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def event_marks(manifest: dict, shadow: list[dict]) -> list:
    result = [(e['input_sec'], '発火') for e in manifest['events']]
    result += [(e['background_ready_sec'], '再帰') for e in manifest['events']
               if e.get('background', {}).get('tree')]
    result += [(max(r['t_sec'], r.get('confirmed_sec', r['t_sec'])), '着地')
               for r in shadow if r.get('event') == 'landing' and r.get('game_idx') == 7]
    return sorted(result)


def verify_row(row: dict, expected: dict, receipt: dict, history: list, start: float,
               keys: tuple = KEYS, line_builder: Callable = lines) -> int:
    """全コマでCSVの値を照合し、グラフへ渡した全履歴の追加値も検査する。"""
    assert row == expected
    assert receipt['texts'] == line_builder(expected, start)
    assert history[-1] == (expected['t_sec'], [expected[k] for k in keys])
    assert receipt['graph_count'] == expected['frame_index'] + 1
    stamps = [v for k, v in expected.items() if v is not None
              and (k.endswith('input_sec') or k.endswith('ready_sec'))]
    stamps.extend(t for t, _ in receipt['marks'])
    assert all(t <= expected['t_sec'] for t in stamps), '未来参照'
    assert all(t <= expected['t_sec'] for t, _ in history), 'グラフの未来参照'
    return len(stamps) + len(history)


def render(output: Path, manifest: dict, rows: list[dict], shadow: list[dict],
           keys: tuple = KEYS, colors: list = COLORS, line_builder: Callable = lines,
           marks_override: list | None = None, extra_checks: dict | None = None,
           band_builder: Callable = draw_band, enforce_publication_budget: bool = True) -> None:
    # 前回の元映像領域を再利用する。共有原本の再転送・再認識を省く。
    cap = cv2.VideoCapture(str(output.parent / 'video_80_g7/pc.mp4'))
    encoders = [encoder(output / name, manifest['fps'], pc)
                for name, pc in [('pc.mp4', True), ('mobile_00.mp4', False)]]
    history, marks = [], []
    pending_marks = event_marks(manifest, shadow) if marks_override is None else marks_override
    start, checks, mark_cursor = manifest['match']['start_sec'], 0, 0
    delays = [(math.ceil(e['background_ready_sec'] * manifest['fps']) / manifest['fps'] - e['input_sec'])
              for e in manifest['events'] if e.get('background', {}).get('tree')]
    bounds = [e.get('publication_budget_sec', 1.) for e in manifest['events']
              if e.get('background', {}).get('tree')]
    if enforce_publication_budget:
        assert all(delay <= bound for delay, bound in zip(delays, bounds)), '再帰の表示が公開期限を超過'
    csv_rows = iter(read_csv(output / 'frames.csv'))
    with (output / 'panel_receipts.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            ok, frame = cap.read()
            assert ok, '元映像不足'
            history.append((row['t_sec'], [row[k] for k in keys]))
            while mark_cursor < len(pending_marks) and pending_marks[mark_cursor][0] <= row['t_sec']:
                marks.append(pending_marks[mark_cursor])
                mark_cursor += 1
            band, receipt = band_builder(row, start, history, marks, line_builder, colors)
            checks += verify_row(row, next(csv_rows), receipt, history, start, keys, line_builder)
            rendered = np.vstack([cv2.resize(frame[:1080], (WIDTH, VIDEO_HEIGHT)), band])
            for enc in encoders:
                enc.stdin.write(rendered.tobytes())
            stream.write(json.dumps(receipt, ensure_ascii=False) + '\n')
            if row['frame_index'] in (0, 1800):
                cv2.imwrite(str(output / f"preview_{row['frame_index']}.png"), rendered)
            if row['frame_index'] % 1200 == 0:
                print(f"描画 {row['frame_index']}/{len(rows)}", flush=True)
    cap.release()
    for enc in encoders:
        enc.stdin.close()
        assert enc.wait() == 0
    finish(output, rows, checks, delays, keys, extra_checks)


def finish(output: Path, rows: list, checks: int, delays: list, keys: tuple,
           extra_checks: dict | None) -> None:
    sizes = {name: (output / name).stat().st_size for name in ('pc.mp4', 'mobile_00.mp4')}
    assert all(size <= LIMIT_BYTES for size in sizes.values())
    save(output / 'selfcheck.json', dict(frames=len(rows), panel_csv_mismatch=0,
        graph_csv_mismatch=0, future_references=0, causal_checks=checks, sizes=sizes,
        recursive_publications=len(delays), maximum_recursive_delay_sec=max(delays, default=0),
        verification='全コマの実描画文字列・グラフ入力とCSVを照合。圧縮動画のOCR検査は省略。',
        graph_history=f'各コマの公開した{len(keys)}値だけをappend。過去の代入更新なし。',
        **(extra_checks or {})))
    print('完了', sizes, flush=True)

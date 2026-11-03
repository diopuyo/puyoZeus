"""撃ち合いレビュー専用の入出力。保存入力と事後ラベルを明確に分ける。"""
from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path
from typing import Iterator, Any
from src.board import (BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA,
                       COLOR_UNKNOWN, COLOR_RED, COLOR_PURPLE)

REVIEW_INDICES = (92, 32, 51, 80, 61, 34)
FIRE_KINDS = ('fire', 'counter', 'main')
CONSUMPTION_LIMIT = 0.6
FPS = 30.0
HASH_CHUNK_SIZE = 1024 * 1024
FILES = dict(shadow='out/spool/exchange_projection_shadow.jsonl',
             inputs='out/inputs.jsonl.gz', exchanges='out/supervised-exchanges.pickle',
             b3='out/b3_updates.jsonl', display='out/display.npz',
             starts='out/spool/game_starts.pickle', publications='out/spool/publications.pickle')


def jsonl(path: Path) -> Iterator[dict]:
    """壊れた途中ファイルは成功扱いにせず例外にする。"""
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def pickles(path: Path) -> Iterator[Any]:
    """自分の走行で生成した信頼済み pickle の連続レコードを読む。"""
    with path.open('rb') as stream:
        while stream.peek(1):
            yield pickle.load(stream)  # レコード途中のEOFは未完了として例外にする。


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_SIZE), b''):
            digest.update(chunk)
    return digest.hexdigest()


def paths(run: Path) -> dict[str, Path]:
    """ed DETAILS と実ファイルで確認した配置。曖昧な再帰 glob は使わない。"""
    result = {key: run / name for key, name in FILES.items()}
    status = run / 'status.json'
    if status.exists():
        state = json.loads(status.read_text(encoding='utf-8'))
        if state.get('state') != 'complete' or state.get('exit_code') != 0:
            raise FileNotFoundError(f'ed の走行が未完了です: {status}')
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('ed の必要ファイルが未完了: ' + ', '.join(missing))
    return result


def fires(rows: list[dict]) -> list[dict]:
    """動画内の (試合, 側, chain_episode_id) で最初の通知だけを採る。"""
    seen, result = set(), []
    for row in sorted(rows, key=lambda r: float(r.get('t_sec', 0))):
        if row.get('kind') != 'event' or row.get('event') not in FIRE_KINDS:
            continue
        episode = row.get('evidence', {}).get('chain_episode_id')
        if episode is None:
            if row.get('status') == 'skipped':
                continue  # 投影不能行にはedでも印が無い。選定側で件数を記録して試合を除外する。
            raise ValueError('chain_episode_id が無い旧影ログです。ed の再計算用出力が必要です')
        key = (row['game_idx'], row['attacker'], episode)
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def grid(value: str | list) -> list[int]:
    if isinstance(value, str):
        glyphs = {'.': COLOR_EMPTY, 'o': COLOR_OJAMA, '?': COLOR_UNKNOWN}
        result = [glyphs[c] if c in glyphs else int(c) for c in value.replace('/', '')]
    else:
        result = [int(v) for row in value for v in row] if isinstance(value[0], list) else list(value)
    if len(result) != BOARD_ROWS * BOARD_COLS:
        raise ValueError('盤面は13行6列です')
    return result


def consumption(before: list[int], after: list[int], placements: int = 0) -> float | None:
    """前後の色数と設置数による消費率。範囲外をクランプして隠さない。"""
    count = lambda board: sum(COLOR_RED <= v <= COLOR_PURPLE for v in board)
    initial = count(before)
    ratio = (initial + 2 * placements - count(after)) / initial if initial else None
    return ratio if ratio is not None and 0 <= ratio <= 1 else None


def match_bounds(starts: list[dict], end_sec: float) -> dict[int, tuple[float, float]]:
    """開始記録から次の試合開始まで全区間。最初の途中試合も明示できる。"""
    ordered = sorted({int(r['game']): float(r['t_sec']) for r in starts}.items(), key=lambda x: x[1])
    return {game: (start, ordered[i + 1][1] if i + 1 < len(ordered) else end_sec)
            for i, (game, start) in enumerate(ordered)}

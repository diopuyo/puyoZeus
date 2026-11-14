"""試合終了の組 (やった・ばたんきゅー) の画像なし判定器を学び、学習に使わない動画で先生と比べる。

配布の設定 (live_match_end_pair) では、終了の組を MatchEndPairDetector が切り抜き画像との NCC で照合する。
画像なし方式ではその画像が無く、組は一度も成立しなかった (11/14 発見)。ここでは配布と同じ照合
(「やった」は文字の部分だけ = PUYO_MATCH_END_YATTA_TEXT_TEMPLATE=1) を先生にして、
領域ごとに「閾値以上か」を決定木の森に学ばせる。左右は同じ画像を照合するので、やった・ばたんきゅーの
2 つの森を左右の領域で共有する。
使い方 (配布の python、cwd は切り抜き画像のある app):
  collect <動画id> <動画> <出力>  /  train <出力> <元の ui.joblib> <動画id...>  /  evaluate <出力> <ui.joblib> <動画id> <動画>
"""
from __future__ import annotations

import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

KEYS = ('yatta_1p', 'yatta_2p', 'batan_1p', 'batan_2p')
SCAN_FRAMES = 15            # 学習の標本は 15 コマごと (30fps で 0.5 秒、60fps の動画は 0.25 秒ごとで細かめ)
SCAN_SECONDS = 0.5          # 評価は配信の組照合と同じ 0.5 秒ごと (時刻で決める)
DENSE_FRAMES = 3            # 告知が見える前後は 0.1 秒ごとに足す (暗くなる途中・出始めの境目を学ぶ)
DENSE_AFTER_FRAMES = 45     # 告知らしいコマの後ろ 1.5 秒は細かく取る
NEAR_SCORE = 0.30           # 先生の値がこれ以上なら告知の近く (閾値 0.50 の手前から細かく取る)
HARD_NEGATIVE_SCORE = 0.25  # 閾値の下でも紛らわしい陰性は全部残す
NEGATIVE_KEEP_EVERY = 4     # それ以外の陰性は 4 回に 1 回だけ残す (行数を抑える)
TREE_COUNT = 64             # 既存の UI の森 (train_image_free_fix) と同じ
SEED = 20261114
MIN_POSITIVES = 50
FRAME_SIZE = (1920, 1080)


def teacher() -> Any:
    """配布と同じ設定の組照合 (切り抜き画像あり)。"""
    os.environ['PUYO_IMAGE_FREE_ALL'] = '0'
    os.environ['PUYO_MATCH_END_YATTA_TEXT_TEMPLATE'] = '1'
    from src.match_end_detector import MatchEndDetector
    from src.match_end_pair import MatchEndPairDetector
    pair = MatchEndPairDetector(MatchEndDetector.load_default()._templates)
    if sorted(pair._templates) != ['match_end_batan', 'match_end_yatta']:
        raise RuntimeError(f'先生の画像が足りません: {sorted(pair._templates)}')
    return pair


def model_key(key: str) -> str:
    """左右で共有する森の名前 (pair_yatta / pair_batan)。"""
    return 'pair_' + key.split('_')[0]


def crops(frame: np.ndarray) -> dict[str, np.ndarray]:
    from src.match_end_pair import PAIR_ROIS
    return {key: frame[y:y + h, x:x + w] for key, (x, y, w, h) in PAIR_ROIS.items()}


DECODER = Path('C:/puyo1_tools/soak/vcam_venv/Scripts/python.exe')   # 動画を開ける cv2 (配布の cv2 は ffmpeg 無し)
HEADER_BYTES = 8
FRAME_BYTES = FRAME_SIZE[0] * FRAME_SIZE[1] * 3


def frames(path: str, step: int = DENSE_FRAMES) -> Iterator[tuple[int, np.ndarray]]:
    """別 process (動画を開ける cv2) で先頭から順に読み、step コマごとのコマを 1080p で受け取る。
    特徴と先生の照合は、この process (配布と同じ cv2・numpy) で計算する。"""
    import subprocess
    process = subprocess.Popen([str(DECODER), '-B', str(Path(__file__).resolve()), 'decode', path, str(step)],
                               stdout=subprocess.PIPE)
    try:
        while len(header := process.stdout.read(HEADER_BYTES)) == HEADER_BYTES:
            raw = process.stdout.read(FRAME_BYTES)
            if len(raw) != FRAME_BYTES:
                raise RuntimeError(f'復号の途中で切れました: {path}')
            yield int.from_bytes(header, 'little'), np.frombuffer(raw, np.uint8).reshape(FRAME_SIZE[1], FRAME_SIZE[0], 3)
    finally:
        process.stdout.close()
        process.terminate()
        process.wait()
    if process.returncode not in (0, 1):
        raise RuntimeError(f'復号の process が失敗: {path} {process.returncode}')


def decode(path: str, step: int) -> None:
    """動画を開ける側。step コマごとのコマを (番号 8 バイト + 1080p BGR) で標準出力へ流す。"""
    import msvcrt
    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f'動画を開けません: {path}')
    out, index = sys.stdout.buffer, 0
    while cap.grab():
        if index % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                raise RuntimeError(f'復号失敗: {path}:{index}')
            if frame.shape[1::-1] != FRAME_SIZE:
                frame = cv2.resize(frame, FRAME_SIZE)
            out.write(index.to_bytes(HEADER_BYTES, 'little'))
            out.write(np.ascontiguousarray(frame).tobytes())
        index += 1
    out.flush()


def observe(pair: Any, frame: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
    """4 領域の (領域, 特徴, 先生の NCC)。"""
    from src.image_free_ui_ensemble import task_features
    return [(key, task_features('pair_' + key, crop), pair._score_crop(crop, key))
            for key, crop in crops(frame).items()]


def keep(score: float, scan: int) -> bool:
    from src.match_end_pair import PAIR_NCC_THRESHOLD
    return score >= HARD_NEGATIVE_SCORE or score >= PAIR_NCC_THRESHOLD or scan % NEGATIVE_KEEP_EVERY == 0


def collect(video_id: str, path: str, output: Path) -> None:
    """0.5 秒ごとに先生で採点し、告知の近くは前後を 0.1 秒ごとに足して、特徴と先生の値を保存する。"""
    pair, rows = teacher(), []
    recent: deque[tuple[int, np.ndarray]] = deque(maxlen=SCAN_FRAMES // DENSE_FRAMES)
    dense_until, done = -1, set()
    for index, frame in frames(path):
        is_scan = index % SCAN_FRAMES == 0
        if not is_scan and index > dense_until:
            recent.append((index, frame))
            continue
        observed = observe(pair, frame)
        done.add(index)
        rows.extend((key, feature, score, index) for key, feature, score in observed
                    if not is_scan or keep(score, index // SCAN_FRAMES) or index <= dense_until)
        if max(score for _, _, score in observed) >= NEAR_SCORE:
            for earlier, image in recent:                # 出始めの前のコマも取る
                if earlier not in done:
                    rows.extend((k, f, s, earlier) for k, f, s in observe(pair, image))
                    done.add(earlier)
            dense_until = index + DENSE_AFTER_FRAMES
        recent.clear()
    save_rows(output / f'{video_id}.npz', rows)
    print(video_id, 'rows', len(rows), 'frames', len(done), flush=True)


def save_rows(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, keys=np.array([r[0] for r in rows]),
                        features=np.stack([r[1] for r in rows]).astype(np.float32),
                        scores=np.array([r[2] for r in rows], dtype=np.float32),
                        frames=np.array([r[3] for r in rows], dtype=np.int64))


def fit(output: Path, base_model: Path, video_ids: list[str]) -> None:
    """やった・ばたんきゅーの森を学び、元の UI の判定器に足して保存する (元の判定器は変えない)。"""
    import joblib
    from sklearn.ensemble import ExtraTreesClassifier
    from src.image_free_ui_ensemble import UIEnsemble
    from src.match_end_pair import PAIR_NCC_THRESHOLD
    parts = [np.load(output / f'{video_id}.npz') for video_id in video_ids]
    keys = np.concatenate([p['keys'] for p in parts])
    features = np.concatenate([p['features'] for p in parts])
    labels = np.concatenate([p['scores'] for p in parts]) >= PAIR_NCC_THRESHOLD
    model, receipt = joblib.load(base_model), dict(videos=video_ids, tasks={})
    for name in ('pair_yatta', 'pair_batan'):
        rows = np.array([model_key(k) == name for k in keys])
        target = [json.dumps(bool(v)) for v in labels[rows]]
        classifier = ExtraTreesClassifier(n_estimators=TREE_COUNT, random_state=SEED, class_weight='balanced',
                                          max_features='sqrt', min_samples_leaf=2, n_jobs=1)
        classifier.fit(features[rows], target)
        positives = int(labels[rows].sum())
        receipt['tasks'][name] = dict(samples=int(rows.sum()), positives=positives,
                                      negatives=int(rows.sum()) - positives,
                                      depths=[t.get_depth() for t in classifier.estimators_])
        if min(positives, int(rows.sum()) - positives) < MIN_POSITIVES:
            raise RuntimeError(f'陽性か陰性が足りません: {name} {receipt["tasks"][name]}')
        model['tasks'][name] = UIEnsemble(name, classifier)
    joblib.dump(model, output / 'ui.joblib', compress=3)
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: {kk: v[kk] for kk in ('samples', 'positives')} for k, v in receipt['tasks'].items()}))


def decisions(scores: dict[str, float]) -> tuple[bool, str | None]:
    from src.match_end_pair import MatchEndPairDetector
    result = MatchEndPairDetector._result(None, scores, scores)
    return result.detected, result.winner


def video_fps(path: str) -> float:
    """動画の fps (動画を開ける側の cv2 で読む)。60fps の動画もあるので、0.5 秒ごとは時刻で決める。"""
    import subprocess
    out = subprocess.run([str(DECODER), '-B', str(Path(__file__).resolve()), 'fps', path],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip().splitlines()[-1])


def evaluate(output: Path, model_path: Path, video_id: str, path: str) -> None:
    """配信と同じ 0.5 秒ごとに、先生と画像なしの領域ごとの採否・組の判定を全区間で並べて保存する。"""
    pair = teacher()
    os.environ['PUYO_IMAGE_FREE_UI_MODEL'] = str(model_path)
    from src.image_free_ui import learned_score
    fps = video_fps(path)
    step = round(fps * SCAN_SECONDS)
    rows = []
    for index, frame in frames(path, step):
        regions = crops(frame)
        old = {key: pair._score_crop(crop, key) for key, crop in regions.items()}
        new = {key: learned_score(model_key(key), crop) for key, crop in regions.items()}
        rows.append(dict(frame=index, t=index / fps, old=old, new=new, old_pair=decisions(old), new_pair=decisions(new)))
    (output / f'eval_{video_id}.json').write_text(json.dumps(rows), encoding='utf-8')
    print(video_id, 'scans', len(rows), flush=True)


def eval_collect(output: Path, video_id: str, path: str) -> None:
    """評価用の動画を配信と同じ 0.5 秒ごとに読み、4 領域の特徴と先生の NCC を保存する (判定器は後で当てる)。"""
    pair, fps = teacher(), video_fps(path)
    rows = []
    for index, frame in frames(path, round(fps * SCAN_SECONDS)):
        rows.extend((key, feature, score, index) for key, feature, score in observe(pair, frame))
    save_rows(output / f'eval_{video_id}.npz', rows)
    (output / f'eval_{video_id}.fps').write_text(str(fps), encoding='ascii')
    print(video_id, 'eval rows', len(rows), flush=True)


def eval_apply(output: Path, model_path: Path, video_id: str) -> None:
    """保存した特徴に判定器を当て、先生と画像なしの領域ごとの値・組の判定を並べる (evaluate と同じ形で保存)。"""
    import joblib
    from src.match_end_pair import PAIR_NCC_THRESHOLD
    from src.image_free_ui_ensemble import DECISION_MARGIN
    data, fps = np.load(output / f'eval_{video_id}.npz'), float((output / f'eval_{video_id}.fps').read_text())
    tasks = joblib.load(model_path)['tasks']
    keys, frames_, scores, features = data['keys'], data['frames'], data['scores'], data['features']
    learned = np.empty(len(keys))
    for name in ('pair_yatta', 'pair_batan'):
        rows = np.array([model_key(k) == name for k in keys])
        labels = [json.loads(str(v)) for v in tasks[name].classifier.predict(features[rows])]
        learned[rows] = [PAIR_NCC_THRESHOLD + (DECISION_MARGIN if v else -DECISION_MARGIN) for v in labels]
    table: dict[int, dict] = {}
    for key, index, old, new in zip(keys, frames_, scores, learned):
        row = table.setdefault(int(index), dict(frame=int(index), t=int(index) / fps, old={}, new={}))
        row['old'][str(key)], row['new'][str(key)] = float(old), float(new)
    out = [dict(row, old_pair=decisions(row['old']), new_pair=decisions(row['new'])) for _, row in sorted(table.items())]
    (output / f'eval_{video_id}.json').write_text(json.dumps(out), encoding='utf-8')
    print(video_id, 'scans', len(out), flush=True)


def main() -> None:
    command, *args = sys.argv[1:]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if command not in ('decode', 'fps'):
        cv2.setNumThreads(1)
    if command == 'decode':
        decode(args[0], int(args[1]))
    elif command == 'fps':
        print(cv2.VideoCapture(args[0]).get(cv2.CAP_PROP_FPS))
    elif command == 'collect':
        collect(args[0], args[1], Path(args[2]))
    elif command == 'train':
        fit(Path(args[0]), Path(args[1]), args[2:])
    elif command == 'evalcollect':
        eval_collect(Path(args[0]), args[1], args[2])
    elif command == 'evalapply':
        eval_apply(Path(args[0]), Path(args[1]), args[2])
    elif command == 'evaluate':
        evaluate(Path(args[0]), Path(args[1]), args[2], args[3])
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()

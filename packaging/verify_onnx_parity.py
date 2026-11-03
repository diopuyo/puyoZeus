"""合否基準 (a): CNN の全セル argmax が torch 版と ONNX 版で完全一致するか (実動画フレームで測る)。

    <配布版 python> packaging/verify_onnx_parity.py <設定 JSON> <出力 JSON>

設定 JSON: {"models_dir": ".../models", "onnx_dir": ".../models/onnx",
            "videos": [{"name":"zenchi","path":"...mp4","windows":[[2580,3427]],"samples":1200}, ...],
            "models": ["cnn_phase_b_large_v2.pt", ...]}
母集団: 標本フレームごとに 1P/2P の可視 12 行 x 6 列 = 144 セルの実パッチ (本番と同じ切り出し)。
同一パッチを torch (既定) と ONNX (環境変数 PUYO_CNN_BACKEND=onnx) の両方へ通し、
argmax 不一致セル数・確率の最大絶対差・上位 2 クラス確率差 (マージン) の小さい境界例を数える。
合否は事前登録 (docs/PHASE_J_ONNX_PREREGISTRATION_2026-09-30.md): argmax 不一致 0 件のみ合格。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FRAME_SIZE = (1920, 1080)
NEAR_TIE_MARGINS = (1e-3, 1e-4, 1e-5, 1e-6)
EXAMPLES = 5
END_GUARD_SEC = 3.0


def load_classifier(path: Path) -> Any:
    """recognition_pipeline と同じ規則 (第 1 conv の出力数) で small/large を選んで重みを載せる。"""
    import torch
    from src.patch_classifier import CnnPatchClassifier, CnnPatchClassifierLarge
    state = torch.load(str(path), map_location='cpu', weights_only=True)
    cnn = (CnnPatchClassifierLarge if state[next(iter(state))].shape[0] == 32 else CnnPatchClassifier)()
    cnn._model.load_state_dict(state)
    cnn._model.eval()
    return cnn


def cell_patches(frame: np.ndarray) -> list[np.ndarray]:
    """1P/2P の可視 12x6 セルの実パッチ (predict_proba_grid と同じ切り出し)。"""
    from src.image_reader import DEFAULT_P1_REGION, DEFAULT_P2_REGION
    from src.patch_classifier import BOARD_COLS, HIDDEN_ROWS, VISIBLE_ROWS
    height, width = frame.shape[:2]
    patches: list[np.ndarray] = []
    for region in (DEFAULT_P1_REGION, DEFAULT_P2_REGION):
        for vrow in range(VISIBLE_ROWS):
            for col in range(BOARD_COLS):
                x1, y1, x2, y2 = region.cell_sample_rect(vrow + HIDDEN_ROWS, col)
                x1 = max(0, min(int(x1), width - 1))
                x2 = max(x1 + 1, min(int(x2), width))
                y1 = max(0, min(int(y1), height - 1))
                y2 = max(y1 + 1, min(int(y2), height))
                patches.append(frame[y1:y2, x1:x2])
    return patches


def sample_frame_indices(fps: float, windows: list[list[float]], samples: int) -> list[int]:
    """窓の和集合の上で等間隔に samples 個のフレーム番号 (重複除去)。"""
    lengths = [end - start for start, end in windows]
    positions = np.linspace(0, sum(lengths), samples, endpoint=False)
    frames: list[int] = []
    for position in positions:
        for (start, _), length in zip(windows, lengths):
            if position < length:
                frames.append(int(round((start + position) * fps)))
                break
            position -= length
    return sorted(set(frames))


def iter_frames(path: Path, indices: list[int]):
    """indices (昇順) のフレームを順次デコードで 1 枚ずつ返す (全部をメモリに置かない)。"""
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
    position = indices[0]
    for target in indices:
        while position < target:
            capture.grab()
            position += 1
        ok, frame = capture.read()
        position += 1
        if not ok:
            raise RuntimeError(f'フレーム {target} を読めません: {path}')
        if frame.shape[1::-1] != FRAME_SIZE:
            frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        yield target, frame
    capture.release()


class Tally:
    """1 モデル x 1 動画の集計。"""

    def __init__(self) -> None:
        self.frames = self.cells = self.mismatched = 0
        self.max_diff = 0.0
        self.margins: list[float] = []
        self.examples: list[dict] = []
        self.seconds = {'torch': 0.0, 'onnx': 0.0}

    def add(self, index: int, torch_probs: np.ndarray, onnx_probs: np.ndarray) -> None:
        bad = np.flatnonzero(torch_probs.argmax(1) != onnx_probs.argmax(1))
        self.frames += 1
        self.cells += len(torch_probs)
        self.mismatched += len(bad)
        self.max_diff = max(self.max_diff, float(np.abs(torch_probs - onnx_probs).max()))
        top2 = np.sort(torch_probs, axis=1)[:, -2:]
        self.margins.extend((top2[:, 1] - top2[:, 0]).tolist())
        for cell in bad[:EXAMPLES]:
            self.examples.append(dict(frame=index, cell=int(cell), torch=torch_probs[cell].tolist(),
                                      onnx=onnx_probs[cell].tolist()))

    def report(self) -> dict[str, Any]:
        margins = np.array(self.margins)
        return dict(frames=self.frames, cells=self.cells, argmax_mismatch_cells=self.mismatched,
                    max_abs_prob_diff=self.max_diff, min_top2_margin_torch=float(margins.min()),
                    near_ties={f'margin<{m:g}': int((margins < m).sum()) for m in NEAR_TIE_MARGINS},
                    cnn_ms_per_frame={k: round(v * 1000 / self.frames, 3) for k, v in self.seconds.items()},
                    cnn_ms_per_patch={k: round(v * 1000 / self.cells, 5) for k, v in self.seconds.items()},
                    examples=self.examples[:EXAMPLES * 4])


def timed_probs(cnn: Any, patches: list[np.ndarray], backend: str | None, tally: Tally, key: str) -> np.ndarray:
    from src import cnn_onnx
    if backend:
        os.environ[cnn_onnx.BACKEND_ENV] = backend
    else:
        os.environ.pop(cnn_onnx.BACKEND_ENV, None)
    started = time.perf_counter()
    probs = cnn.predict_proba_batch(patches)
    tally.seconds[key] += time.perf_counter() - started
    os.environ.pop(cnn_onnx.BACKEND_ENV, None)
    return probs


def run_video(video: dict, classifiers: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    probe = cv2.VideoCapture(video['path'])
    fps = probe.get(cv2.CAP_PROP_FPS)
    duration = probe.get(cv2.CAP_PROP_FRAME_COUNT) / fps - END_GUARD_SEC  # 末尾付近は読めないことがある (実測: mia8)
    probe.release()
    windows = [[s, min(e, duration)] for s, e in video['windows'] if s < duration]
    indices = sample_frame_indices(fps, windows, video['samples'])
    tallies = {name: Tally() for name in classifiers}
    for index, frame in iter_frames(Path(video['path']), indices):
        patches = cell_patches(frame)
        for name, cnn in classifiers.items():
            torch_probs = timed_probs(cnn, patches, None, tallies[name], 'torch')
            onnx_probs = timed_probs(cnn, patches, 'onnx', tallies[name], 'onnx')
            tallies[name].add(index, torch_probs, onnx_probs)
    return dict(fps=fps, sampled_frames=len(indices), wall_seconds=round(time.perf_counter() - started, 1),
                models={name: tally.report() for name, tally in tallies.items()})


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    models_dir, onnx_dir = Path(config['models_dir']), Path(config['onnx_dir'])
    from src import cnn_onnx
    original = cnn_onnx.session_for
    cnn_onnx.session_for = lambda digest, root=onnx_dir, optimization=None: original(digest, root, optimization)
    if config.get('torch_threads'):  # 本番は cpu_threads=1。ONNX 側 (intra_op=1) と条件を揃えた時間比較用
        import torch
        torch.set_num_threads(int(config['torch_threads']))
    classifiers = {name: load_classifier(models_dir / name) for name in config['models']}
    report: dict[str, Any] = dict(videos={}, config=config)
    for video in config['videos']:
        report['videos'][video['name']] = run_video(video, classifiers)
        Path(sys.argv[2]).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({v: {m: {k: r[k] for k in ('frames', 'cells', 'argmax_mismatch_cells', 'max_abs_prob_diff')}
                             for m, r in d['models'].items()} for v, d in report['videos'].items()},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

"""CNN パッチ分類器の ONNX 推論バックエンド (既定 OFF、torch 版は残す)。

切替: 環境変数 `PUYO_CNN_BACKEND=onnx` のときだけ `CnnPatchClassifier` が torch の代わりに ONNX Runtime
(CPU) で logits を計算する。未設定なら従来どおり (bit-identical、この module は import すらされない)。

設計 (fail-silent の排除):
- ONNX ファイルは「重みの内容ハッシュ」で引く (`models/onnx/<digest>.onnx`)。どの .pt をどう読んだか
  (load / load_state_dict) に依らず、実際に載っている重みと 1 対 1 で対応する。
- 対応する ONNX が無ければ例外 (黙って torch へ戻さない)。戻すと「ONNX で動いている」という前提の
  測定・配布物が実は torch で動いていた、という事故になる。
- 前処理 (リサイズ・色空間・円形マスク) と softmax は既存コードのまま。差し替えるのは畳み込み本体の logits だけ。
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

BACKEND_ENV = 'PUYO_CNN_BACKEND'
BACKEND_ONNX = 'onnx'
ONNX_DIR = Path('models/onnx')
INDEX_NAME = 'index.json'
INPUT_NAME, OUTPUT_NAME = 'patch', 'logits'
OPSET = 17
DIGEST_CHARS = 16
LARGE_FIRST_CONV_CHANNELS = 32  # recognition_pipeline と同じ small/large 判別 (state_dict の第 1 conv 出力数)
INTRA_OP_THREADS = 1  # 既存の cpu_threads=1 (認識は 1 スレッド) と揃える
_SESSIONS: dict[str, Any] = {}
_CALLS = {'calls': 0, 'patches': 0}  # 実際に ONNX で計算した回数 (使われたことの証拠。終了時に stderr へ出す)


class OnnxBackendError(RuntimeError):
    """ONNX バックエンドが要求されたのに使えない (対応ファイル無し・ONNX Runtime 無し等)。"""


def backend_requested() -> bool:
    return os.environ.get(BACKEND_ENV, '').strip().lower() == BACKEND_ONNX


def state_digest(state: dict[str, Any]) -> str:
    """state_dict (テンソル辞書) の内容ハッシュ。名前・dtype・形状・生バイトを名前順に連結する。"""
    digest = hashlib.sha256()
    for name in sorted(state):
        array = state[name].detach().cpu().contiguous().numpy()
        digest.update(f'{name}|{array.dtype}|{array.shape}'.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()[:DIGEST_CHARS]


def onnx_file(digest: str, root: Path = ONNX_DIR) -> Path:
    return root / f'{digest}.onnx'


def session_for(digest: str, root: Path = ONNX_DIR, optimization: str | None = None) -> Any:
    """digest に対応する ONNX Runtime セッション (プロセス内でキャッシュ)。無ければ OnnxBackendError。"""
    key = f'{root.resolve()}|{digest}|{optimization}'
    if key in _SESSIONS:
        return _SESSIONS[key]
    path = onnx_file(digest, root)
    if not path.is_file():
        raise OnnxBackendError(f'{BACKEND_ENV}=onnx ですが、重み {digest} に対応する ONNX がありません: {path}'
                               ' (packaging/export_onnx.py で生成してください)')
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise OnnxBackendError('onnxruntime が未導入です (pip install onnxruntime)') from error
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = INTRA_OP_THREADS
    if optimization:
        options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, optimization)
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    if not _SESSIONS:
        atexit.register(report_usage)
    _SESSIONS[key] = session
    return session


def report_usage() -> None:
    """プロセス終了時に ONNX 利用実績を stderr へ (ログから「本当に ONNX で動いた」ことを後で確認できる)。"""
    print(f'[cnn_onnx] pid={os.getpid()} sessions={len(_SESSIONS)} calls={_CALLS["calls"]} patches={_CALLS["patches"]}',
          file=sys.stderr, flush=True)


def run_logits(session: Any, batch: np.ndarray) -> np.ndarray:
    """(N, C, H, W) float32 → (N, クラス数) の logits。"""
    _CALLS['calls'] += 1
    _CALLS['patches'] += len(batch)
    return session.run([OUTPUT_NAME], {INPUT_NAME: np.ascontiguousarray(batch, dtype=np.float32)})[0]


def export_state(state: dict[str, Any], out_dir: Path) -> tuple[str, Path]:
    """state_dict から分類器を構築して ONNX へ書き出す。(digest, パス) を返す。
    small/large は既存の判別規則 (第 1 conv の出力チャンネル数) に従う。"""
    import torch
    from src.patch_classifier import (CnnPatchClassifier, CnnPatchClassifierLarge, PATCH_RESIZE_H, PATCH_RESIZE_W)
    first = next(iter(state))
    large = state[first].shape[0] == LARGE_FIRST_CONV_CHANNELS
    classifier = CnnPatchClassifierLarge() if large else CnnPatchClassifier()
    classifier._model.load_state_dict(state)
    classifier._model.eval()
    digest = state_digest(classifier._model.state_dict())
    out_dir.mkdir(parents=True, exist_ok=True)
    path = onnx_file(digest, out_dir)
    dummy = torch.zeros(1, classifier.INPUT_CHANNELS, PATCH_RESIZE_H, PATCH_RESIZE_W)
    torch.onnx.export(classifier._model, dummy, str(path), input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
                      dynamic_axes={INPUT_NAME: {0: 'n'}, OUTPUT_NAME: {0: 'n'}}, opset_version=OPSET,
                      do_constant_folding=True, dynamo=False)
    return digest, path


def write_index(out_dir: Path, entries: dict[str, dict[str, Any]]) -> Path:
    """digest → 由来 (.pt 名・torch/onnxruntime の版・opset) の台帳。"""
    path = out_dir / INDEX_NAME
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=1, sort_keys=True), encoding='utf-8')
    return path

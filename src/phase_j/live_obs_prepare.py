"""21c本人試験だけの同値最適化。モデル・観測条件・更新間隔は触らない。"""
from __future__ import annotations

import importlib
from functools import lru_cache
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

ENV = 'PUYO_OBS_PREPARE_EXACT'
MODULES = {
    'evaluation': ('src.prefire_v49d', 'src.phase_j.live_debug_panel',
                   'src.exchange_integrity_memory'),
    'event-evaluation': ('src.phase_j.live_b3', 'src.b3_stage2_trace'),
    'recognition': ('src.phase_j.live_v49f_capture', 'src.row1_publication', 'src.live_delta_cells',
                    'src.phase_j.live_score_share', 'src.prefire_origin_stream',
                    'src.b3_live_queues', 'src.phase_j.live_observation_diagnostics', 'src.board_rules'),
}
ROOT_CACHE_SIZE = 8


@lru_cache(maxsize=ROOT_CACHE_SIZE)
def resolve_at(directory: str, root: str) -> Path:
    """検収済み配布の不変モデルディレクトリを、作業場所ごとに一度だけ解決する。"""
    path = Path(root)
    return (path if path.is_absolute() else Path(directory)/path).resolve()


def resolved_model_root(root: Path) -> Path:
    return resolve_at(os.getcwd(), str(root))


def prepare_imports(role: str) -> None:
    """初回通知で必要なモジュールを、通知の開始前に読み込む。状態更新は呼ばない。"""
    if os.environ.get(ENV) != '1':
        return
    started = perf_counter()
    for name in MODULES.get(role, ()):
        importlib.import_module(name)
    print('OBS_PREPARE '+json.dumps(dict(role=role,
        imports=MODULES.get(role, ()), wall_ms=(perf_counter()-started)*1000)), flush=True)


def prepare_recognition(pipe: Any) -> None:
    """固定HSV分類器に既存の同値Rust経路を指定する。較正側の設定は触らない。"""
    if os.environ.get(ENV) != '1':
        return
    started = perf_counter()
    prepare_imports('recognition')
    runtime = getattr(pipe, '_placement_reconcile', None)
    if runtime is not None:
        enabled = runtime.hsv.enable_native_hsv()
        print('OBS_PLACEMENT_NATIVE '+json.dumps(dict(enabled=enabled)), flush=True)
    if os.environ.get('PUYO_LIVE_V49F') == '1':
        pipe._v49f_prepared_capture = create_v49f_capture()
    prepare_sessions(pipe)
    print('OBS_RECOGNITION_READY '+json.dumps(dict(wall_ms=(perf_counter()-started)*1000)), flush=True)


def create_v49f_capture() -> Any:
    """画像履歴をまだ持たない初期状態を作り、固定HSVの既存同値経路を選ぶ。"""
    from .live_v49f_capture import CaptureV49F
    capture = CaptureV49F()
    if os.environ.get(ENV) == '1':
        for classifier in capture.images.classifiers:
            if not classifier._vote_mode:
                classifier.enable_native_hsv()
    return capture


def prepare_sessions(pipe: Any) -> None:
    """推論せず、固定重みのONNXセッション作成だけを取得開始前へ移す。"""
    from src import cnn_onnx
    if not cnn_onnx.backend_requested():
        return
    classifiers = [getattr(pipe._reader._classifier, '_cnn', None),
                   getattr(getattr(pipe, '_placement_reconcile', None), 'cnn', None)]
    for classifier in classifiers:
        if classifier is not None:
            digest = getattr(classifier, '_onnx_digest', None)
            if digest is None:
                digest = classifier._onnx_digest = cnn_onnx.state_digest(classifier._model.state_dict())
            cnn_onnx.session_for(digest)

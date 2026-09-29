"""D1の短区間再生。既存G3計装を読込み、出力だけを追加する。"""
from __future__ import annotations
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from typing import Any
import cv2
import torch
from src.recognition_pipeline import RecognitionPipeline
from scripts.enrich_e26_midchain import frame_at
from scripts.enrich_e16_records_20260928 import ZENCHI_VIDEO
from scripts._d1_inventory import OUT

ASSETS = Path('/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer')
START, END, LOG_START, FPS, STRIDE = 2549.05, 2752., 2746., 60, 2


class Captured(Exception):
    """描画処理へ入る前に設定だけを取り出す。"""


def pipeline() -> RecognitionPipeline:
    """既存描画関数の既定認識設定をそのまま使う。"""
    from scripts import visualize_advantage_overlay as viz
    original, config = RecognitionPipeline.load_default, {}
    def capture(**kwargs: Any) -> None:
        config.update(kwargs)
        raise Captured()
    RecognitionPipeline.load_default = capture
    viz._acquire_model = lambda *a, **k: None
    try:
        viz.generate(video=ZENCHI_VIDEO, out=OUT/'unused.mp4', max_sec=END, sample_interval=0, render=False)
    except Captured:
        pass
    finally:
        RecognitionPipeline.load_default = original
    (OUT/'trace_config.json').write_text(json.dumps(config, default=str))
    return original(**config)


def instrument(stream: Any) -> Any:
    """G3のBoard書込み追跡を再利用し、各sideの生観測を保存する。"""
    spec = importlib.util.spec_from_file_location('d1_existing_g3', ASSETS/'scripts/diag_board_write_paths.py')
    module = importlib.util.module_from_spec(spec)
    old_path = sys.path[:]
    spec.loader.exec_module(module)
    sys.path[:] = old_path
    module._install_provenance_hooks()
    original = RecognitionPipeline._step_side
    signature = inspect.signature(original)
    def step(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        cnn = a['cnn_board']._grid.copy()
        result = original(self, *args, **kwargs)
        if a['time_sec'] >= LOG_START:
            board = result.confirmed_board
            region = self._reader._p1_region if a['side'] == '1P' else self._reader._p2_region
            hsv = self._reader.read_board_hsv_only(a['frame_bgr'], region)._grid
            row = dict(t=a['time_sec'], frame=a['frame_idx'], side=a['side'], state=result.state.name,
                cnn=cnn.tolist(), hsv=hsv.tolist(), board=board._grid.tolist() if board else None,
                writers=getattr(board, '_dwp', None), written_frames=getattr(board, '_dwpf', None))
            stream.write(json.dumps(row)+'\n')
        return result
    RecognitionPipeline._step_side = step
    return module


def main() -> None:
    """前試合境界から短区間を再生し、元記録との一致を後で確認する。"""
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    with (OUT/'scene_trace.jsonl').open('w') as stream:
        g3 = instrument(stream)
        pipe = pipeline()
        cap = cv2.VideoCapture(str(ZENCHI_VIDEO))
        for index in range(round(START*FPS), round(END*FPS), STRIDE):
            g3._CUR_FRAME[0] = index
            pipe.update(index, index/FPS, frame_at(cap, index))
            if index % (FPS*5) == 1:
                print(index/FPS, flush=True)
        cap.release()
    (OUT/'scene_sites.json').write_text(json.dumps(g3._SITE_NAME, indent=2))


if __name__ == '__main__':
    main()

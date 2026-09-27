"""同一設定の元動画2回とNV12二水準を認識し、補助のtsumo_countも記録する。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

import scripts.verify_live_b4_degradation as b4
from scripts.run_live_pipeline_20260928 import save_json

OUTPUT = Path('logs/live_b5_recognition')
DETAILS = b4.frame_details


def details(frame: np.ndarray, result: Any, pipe: Any) -> dict:
    """B4の条件を維持し、非単調なtsumo_countと試合内外だけ加える。"""
    row = DETAILS(frame, result, pipe)
    row['turns'] = np.array([pipe.tsumo_count(side) for side in ('1P', '2P')])
    row['active'] = np.array(result.is_match_active)
    row['scores'] = np.array([side.score if side.score is not None else -1
                              for side in (result.p1, result.p2)])
    return row


def run_one(name: str, video: Path, config: dict, clipped: bool) -> None:
    """各回で認識器と乱数を初期化し、結果を独立保存する。"""
    with patch.object(b4, 'frame_details', details):
        result = b4.recognize(video, config, clipped)
    np.savez_compressed(OUTPUT / f'{name}.npz',
                        **{key: value for key, value in result.items() if key != 'calibration'})
    save_json(OUTPUT / f'{name}_calibration.json', result['calibration'])
    print(name, 'completed', flush=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(b4.CONFIG.read_text())
    for name in ('original_a', 'original_b'):
        run_one(name, b4.DEFAULT_VIDEO, config, False)
    for crf in b4.QUALITY_LEVELS:
        video = OUTPUT / f'crf{crf}.mp4'
        command = b4.transcode(b4.DEFAULT_VIDEO, video, crf)
        save_json(OUTPUT / f'crf{crf}_command.json', command)
        run_one(f'crf{crf}', video, config, True)
    # 比較画像を書き出すまで変換動画を保持し、分析側で削除する。
    (OUTPUT / 'complete.txt').write_text('0', encoding='utf-8')


if __name__ == '__main__':
    main()

"""既知の短区間で旧教師・前候補・修正候補を並べる段2の再生台。"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import json
import os
from pathlib import Path

import cv2

from scripts.compare_image_free_all import mode, read_config, save_json
from scripts.train_image_free_all import teachers

KNOWN_C68 = (14554, 19761, 25481, 29634, 8776, 9392, 17056, 20000,
             7151, 14566, 17137)
WINDOW_MARGIN = 10
DEATH_THRESHOLD = 0.55


def dead_latch_starts(output: Path) -> list[int]:
    """既存の確定チャンクから新だけ死亡確定した場面を抽出する。"""
    manifest = json.loads((output/'checkpoint.json').read_text())
    starts, previous = [], -2
    for name in manifest['chunks']:
        with gzip.open(output/name, 'rt', encoding='utf-8') as stream:
            for row in json.load(stream)['rows']:
                old, new = row['old'], row['new']
                if (new.get('display_source') == 'confirmed_death'
                        and old.get('display_source') != 'confirmed_death'):
                    if row['frame'] > previous + 1:
                        starts.append(row['frame'])
                    previous = row['frame']
    return starts


def discrete(readers: dict, frame: object) -> dict:
    """スコアの表現差と離散検出差を混同しない。"""
    return dict(death=[readers['death']._score(frame, None, idx) >= DEATH_THRESHOLD for idx in range(2)],
        match_end=readers['match_end'].detect(frame).detected,
        warning=[dict(icons=[asdict(icon) for icon in result.icons], total_count=result.total_count)
                 for result in readers['ojama'].detect(frame)],
        chain=[readers['chain'].read_side(frame, side).chain_count for side in ('1P', '2P')])


def prepare_readers(models: list[str], reuse: bool) -> dict:
    readers = {}
    for index in ([2] if reuse else range(len(models))):
        os.environ['PUYO_IMAGE_FREE_UI_MODEL'] = models[index]
        with mode(index != 0):
            readers[index] = teachers()
    return readers


def baseline_rows(path: Path | None, video: str, centers: list[int]) -> dict:
    if path is None:
        return {}
    baseline = json.loads((path/(video+'.json')).read_text())
    if baseline['video'] != video or baseline['centers'] != centers:
        raise ValueError('再利用する段2の対象区間が異なります')
    return {row['frame']: row for row in baseline['rows']}


def verify_video(video: dict, centers: list[int], models: list[str], output: Path,
                 baseline: Path | None = None) -> dict:
    saved = baseline_rows(baseline, video['id'], centers)
    readers = prepare_readers(models, bool(saved))
    cap, rows, counts = cv2.VideoCapture(video['path']), [], Counter()
    for center in centers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, center-WINDOW_MARGIN))
        for index in range(max(0, center-WINDOW_MARGIN), center+WINDOW_MARGIN+1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"復号失敗: {video['id']}:{index}")
            frame = cv2.resize(frame, (1920, 1080))
            values = [saved[index]['old'], saved[index]['before']] if saved else []
            for reader_index, reader in readers.items():
                os.environ['PUYO_IMAGE_FREE_UI_MODEL'] = models[reader_index]
                with mode(reader_index != 0):
                    values.append(discrete(reader, frame))
            for field in values[0]:
                for label, value in zip(('before', 'after'), values[1:]):
                    counts[f'{label}_{field}'] += int(values[0][field] != value[field])
            rows.append(dict(frame=index, center=center, old=values[0], before=values[1], after=values[2]))
        print(video['id'], center, dict(counts), flush=True)
    cap.release()
    result = dict(video=video['id'], centers=centers, frames=len(rows), counts=dict(counts),
        unique_frames=len({row['frame'] for row in rows}),
        reused_baseline=str(baseline) if baseline is not None else None,
        candidate_image_reads=0, diagnostic_not_quality_pass=True, rows=rows)
    save_json(output/(video['id']+'.json'), result)
    return {k: v for k, v in result.items() if k != 'rows'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    os.chdir(config['asset_root'])
    cv2.setNumThreads(1)
    output = Path(config['output'])/config.get('stage2_name', 'stage2')
    output.mkdir(parents=True, exist_ok=True)
    models = [config['base_ui_model'], config['base_ui_model'],
              config.get('candidate_ui_model', str(Path(config['output'])/'ui.joblib'))]
    centers = dict(c68=list(KNOWN_C68), video_d04=dead_latch_starts(Path(config['previous_output'])/'video_d04'))
    baseline = Path(config['reuse_stage2']) if config.get('reuse_stage2') else None
    results = [verify_video(video, centers[video['id']], models, output, baseline)
               for video in config['videos'] if video['id'] in centers]
    save_json(output/'receipt.json', dict(videos=results, diagnostic_not_quality_pass=True))


if __name__ == '__main__':
    main()

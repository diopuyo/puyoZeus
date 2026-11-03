"""記録の最初の静止評価を同じM0へ通し、OS数値差の発生層を保存する。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


def first_static(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """実overlayと同じ両側STABLE履歴から、最初のstatic入力を再構成する。"""
    from src.exchange_event_record import read_records, static_key
    from src.board_state_machine import BoardState
    latest: list[Any] = [None, None]
    snapshot = None
    pending = None
    for row in read_records(path):
        if row['kind'] == 'update':
            result, snapshot = row['args'][:2]
            for index, side in enumerate((result.p1, result.p2)):
                if side.state == BoardState.STABLE and side.confirmed_board is not None:
                    latest[index] = side
            # fallback付きupdateはstatic呼出の後に記録されるので、その行まで待つ。
            if pending is not None:
                boards = tuple(side.confirmed_board for side in latest)
                assert static_key(boards, snapshot) == pending['key']
                queues = [[*(side.next_pair or (0, 0)), *(side.dnext_pair or (0, 0))] for side in latest]
                return np.stack([board._grid for board in boards]), np.array(queues), pending['input']['m0_probability_1p']
        elif row['kind'] == 'static' and pending is None:
            pending = row
    raise ValueError('静止入力がありません')


def probe(boards: np.ndarray, queues: np.ndarray, output: Path, threads: int | None) -> dict:
    """重み・入力を固定し、推論層の出力を観測するだけで演算を変えない。"""
    from src.exchange_event_m0 import FileM0Predictor
    predictor = FileM0Predictor(Path('models/exchange_event_v3/M0'))
    if threads is not None:
        torch.set_num_threads(threads)
    tensors: dict[str, np.ndarray] = dict(boards=boards, queues=queues)
    order: list[str] = []

    def hook(name: str) -> Any:
        def capture(module: Any, inputs: Any, value: Any) -> None:
            if isinstance(value, torch.Tensor):
                key = f'{name}#{sum(key.startswith(name + "#") for key in tensors)}'
                tensors[key] = value.detach().cpu().numpy().copy()
                order.append(key)
        return capture

    for name, module in predictor.model.named_modules():
        if name:
            module.register_forward_hook(hook(name))
    probability = predictor(boards, queues)
    np.savez_compressed(output.with_suffix('.npz'), **tensors)
    checkpoint = Path('models/exchange_event_v1/M0/model.pt')
    return dict(probability=probability, torch=torch.__version__, threads=torch.get_num_threads(),
                model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                layer_order=order, mkldnn=torch.backends.mkldnn.enabled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('app', type=Path)
    parser.add_argument('record', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--threads', type=int)
    args = parser.parse_args()
    app, record, output = args.app.resolve(), args.record.resolve(), args.output.resolve()
    sys.path.insert(0, str(app))
    os.chdir(app)
    boards, queues, recorded = first_static(record)
    report = probe(boards, queues, output, args.threads)
    report.update(recorded=recorded, reproduces_record=report['probability'] == recorded)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

"""保存実画素で計装前後を比較する台。各条件を独立プロセスで実行する。

引数: ソースroot 素材root 認識設定 出力先 guard(0/1) audit(0/1)
cwd は配布版 app。B3同期表示を測定し、外部R死亡証明は与えない。
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from typing import Any

PRIORITY = 0x4000
SETS = ('video_65_1054.0', 'video_65_2327.0', 'video_79_1758.0')


def setup(root: Path) -> None:
    sys.path[:0] = [str(root), str(Path.cwd())]
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), PRIORITY)
    os.environ['PUYO_B3_JUDGMENT'] = '1'
    os.environ['PUYO_LIVE_CPU_THREADS'] = '1'
    from src.phase_j.live_cpu import apply_runtime
    apply_runtime('recognition', lower_priority=False)


def run_set(directory: Path, config: dict, output: Path, auditing: bool) -> list[dict]:
    import cv2
    from src.recognition_pipeline import RecognitionPipeline
    from src.phase_j.live_audit import RecognitionAudit
    from src.phase_j.live_b3 import LiveB3, close_capture
    from src.phase_j import live_process
    from src.phase_j.live_source import CapturedFrame
    output.mkdir(parents=True, exist_ok=True)
    pipe = RecognitionPipeline.load_default(**config)
    audit = RecognitionAudit(output/'recognition.npz' if auditing else None)
    b3 = LiveB3(Path('models/b3/model.joblib'), Path('models/b3/timing_table.json'), output)
    reference = SimpleNamespace(b3_reference=dict(source=None, probability=None, physical=[]))
    rows: list[dict] = []
    real_recognize = live_process.recognize
    real_predict = b3._predict
    features: list[dict] = []

    def observed_predict(row: dict, notice: Any) -> None:
        features.append(dict(row))
        real_predict(row, notice)

    b3._predict = observed_predict

    def observed(owner: Any, frame: Any) -> Any:
        notice = real_recognize(owner, frame)
        b3.update(notice, reference)
        rows.append(dict(frame=frame.index, result=notice.result_bytes,
            published=pickle.dumps([s.confirmed_board for s in (notice.result().p1, notice.result().p2)]),
            internal=pickle.dumps([owner._sm_1p.context.confirmed_board, owner._sm_2p.context.confirmed_board]),
            prev_confirmed=pickle.dumps([owner._prev_confirmed_1p, owner._prev_confirmed_2p]),
            evaluation=pickle.dumps(notice.evaluation_result()), counts=notice.pipeline.counts,
            b3_input=pickle.dumps(notice.b3), features=pickle.dumps(features[-1:]),
            display=pickle.dumps(b3.display.shown), raw=b3.display.raw))
        return notice

    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    source = SimpleSource([CapturedFrame(m['frame_index_0based'], m['t_sec'], 0., 0.,
        cv2.imread(str(directory/m['file']))) for m in manifest])
    live_process.recognize = observed
    try:
        live_process.send_notices(SimpleNamespace(put=lambda value: None),
            SimpleNamespace(is_set=lambda: False), pipe, source, audit=audit)
    finally:
        live_process.recognize = real_recognize
        b3.close()
        close_capture(pipe)
    return rows


class SimpleSource:
    """取得時刻に左右されない、保存済みの全フレーム。"""

    dropped = 0
    dropped_times: list[float] = []

    def __init__(self, frames: list) -> None:
        self.frames = frames

    def __iter__(self) -> Any:
        return iter(self.frames)


def main() -> None:
    if sys.argv[1] == 'compare':
        compare(Path(sys.argv[2]))
        return
    root, frames, config_file, output = map(Path, sys.argv[1:5])
    setup(root)
    config = json.loads(config_file.read_text(encoding='utf-8'))
    config['enable_row1_col2_guard'] = bool(int(sys.argv[5]))
    rows = []
    for name in SETS:
        rows.extend(run_set(frames/name, config, output/name, bool(int(sys.argv[6]))))
        print(name, len(rows), flush=True)
    (output/'rows.pickle').write_bytes(pickle.dumps(rows))
    print(json.dumps(dict(frames=len(rows), nonnull_display=sum(r['raw'] is not None for r in rows))))


def compare(directory: Path) -> None:
    """同一条件の計装前後を、ハッシュだけでなく全バイトで照合する。"""
    results = []
    for guard in (0, 1):
        for audit in (0, 1):
            suffix = f'g{guard}_a{audit}'
            before, after = [pickle.loads((directory/f'{version}_{suffix}'/'rows.pickle').read_bytes())
                             for version in ('base', 'new')]
            assert len(before) == len(after) and before
            assert [r['frame'] for r in before] == [r['frame'] for r in after]
            mismatches = {key: sum(a[key] != b[key] for a, b in zip(before, after))
                          for key in ('result', 'published', 'internal', 'display')}
            results.append(dict(guard=guard, audit=audit, frames=len(before),
                nonnull_display=sum(r['raw'] is not None for r in after), mismatches=mismatches))
    summary = dict(baseline='f185c1f6a75033ed4c80fc0bca2802df1a5a71fb', conditions=results,
        scope='保存実画素3組、冷起動、B3同期表示、外部R死亡証明なし。eb2全試合再門ではない。')
    (directory/'comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    assert not any(n for r in results for n in r['mismatches'].values())


if __name__ == '__main__':
    main()

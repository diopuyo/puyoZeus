"""統合版の品質門 (判定不変) と段別速度を、同一画素の再投入で測る台。

実 live の経路を再現する: 取り込み画素 → 色補正 (configured_corrector) → live_process.recognize。
条件は環境変数で与え、製品の既定値は変更しない。1条件=1プロセス (直列、BelowNormal)。
  run     : 1条件を実行し、frame別の判定ハッシュ・盤面・CNN入力行別logits・段別時間を保存
  compare : 基準と候補を全件照合 (母数付き)
  speed   : 保存済みの段別時間を ABA 順に要約
"""
from __future__ import annotations

import sys as _public_release_sys
from pathlib import Path as _PublicReleasePath
_public_release_root = str(_PublicReleasePath(__file__).resolve().parents[0])
if _public_release_root not in _public_release_sys.path:
    _public_release_sys.path.insert(0, _public_release_root)
from src.public_dev_paths import resolve_text as _public_dev_text


import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs/live_integ'
SPEED2 = _public_dev_text('<work>/speed2')
ASSETS = Path(_public_dev_text('<work>/assets/baseline_b/app'))
DECODER = 'C:/puyo1_tools/soak/vcam_venv/Scripts/python.exe'
REC_CONFIG = 'C:/puyo1_tools/soak/run_k/output/recognition_config.json'
SHAPE = (1080, 1920, 3)
FRAME_BYTES = SHAPE[0] * SHAPE[1] * SHAPE[2]
FPS = 30.0
MS = 1000.0
WARMUP = 60
BELOW_NORMAL = 0x4000
ROW_HASH_BYTES = 16
BOARD_FIELDS = ('raw_cnn_board', 'cnn_board', 'confirmed_board')
SIDES = ('p1', 'p2')
RESULT_FIELDS = ('frame', 't_sec', 'result_bytes', 'pipeline', 'dropped_before', 'b3')
BOARD_SHAPE = (13, 6)


class DecodedFrames:
    """専用デコーダの画素を無加工で逐次返す (背圧で全画像をRAMに貯めない)。"""

    def __init__(self, video: str, start: float, count: int) -> None:
        self.video, self.start, self.count = video, start, count

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Any:
        import numpy as np
        process = subprocess.Popen([DECODER, '-B', str(Path(__file__).resolve()), '--decode',
            self.video, str(self.start), str(self.count)], stdout=subprocess.PIPE,
            creationflags=BELOW_NORMAL)
        try:
            for _ in range(self.count):
                raw = process.stdout.read(FRAME_BYTES)
                if len(raw) != FRAME_BYTES:
                    raise RuntimeError('デコードが指定枚数より前に終了')
                yield np.frombuffer(raw, dtype=np.uint8).reshape(SHAPE)
        finally:
            process.stdout.close()
            process.terminate()
            process.wait()


def decode_main() -> None:
    import cv2
    import msvcrt
    cv2.setNumThreads(1)
    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    capture = cv2.VideoCapture(sys.argv[2])
    capture.set(cv2.CAP_PROP_POS_MSEC, float(sys.argv[3]) * MS)
    try:
        for _ in range(int(sys.argv[4])):
            ok, image = capture.read()
            if not ok:
                break
            if image.shape != SHAPE:
                image = cv2.resize(image, (SHAPE[1], SHAPE[0]), interpolation=cv2.INTER_AREA)
            sys.stdout.buffer.write(image.tobytes())
    finally:
        capture.release()


class Recorder:
    """frame別の判定ハッシュ・盤面・段別時間・CNN入力行別logitsを貯める。"""

    def __init__(self, want_logits: bool) -> None:
        self.want_logits = want_logits
        self.records: list[tuple] = []
        self.boards: list[Any] = []
        self.stable: list[bool] = []
        self.color_ms: list[float] = []
        self.board_ms: list[float] = []
        self.recognize_ms: list[float] = []
        self.rows: dict[bytes, bytes] = {}
        self.row_total = 0
        self.row_inconsistent = 0
        self.calls = 0
        self.board_acc = 0.0

    def add_logits(self, name: str, batch: Any, output: Any) -> None:
        import numpy as np
        self.calls += 1
        for row, value in zip(batch, output):
            key = name.encode() + hashlib.blake2b(np.ascontiguousarray(row).tobytes(),
                                                  digest_size=ROW_HASH_BYTES).digest()
            data = value.tobytes()
            self.row_total += 1
            self.row_inconsistent += self.rows.setdefault(key, data) != data

    def add_notice(self, notice: Any) -> None:
        import numpy as np
        result = pickle.loads(notice.result_bytes)
        self.records.append(tuple(hashlib.sha256(pickle.dumps(getattr(notice, f))).hexdigest()
                                  for f in RESULT_FIELDS))
        grid = np.full((len(SIDES), len(BOARD_FIELDS)) + BOARD_SHAPE, -1, dtype=np.int16)
        for si, side in enumerate(SIDES):
            for bi, name in enumerate(BOARD_FIELDS):
                board = getattr(getattr(result, side), name)
                if board is not None:
                    grid[si, bi] = board._grid
        self.boards.append(grid)
        self.stable.append(result.p1.state.name == result.p2.state.name == 'STABLE')


def install_hooks(recorder: Recorder) -> None:
    """盤面読取の時間とCNN入力行別のlogitsを、製品コードを変えずに記録する。"""
    from src import cnn_onnx
    from src.image_reader import ImageReader
    read = ImageReader.read_both_boards

    def timed(self: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return read(self, *args, **kwargs)
        finally:
            recorder.board_acc += time.perf_counter() - started
    ImageReader.read_both_boards = timed
    if recorder.want_logits:
        run = cnn_onnx.run_logits

        def logits(session: Any, batch: Any) -> Any:
            output = run(session, batch)
            recorder.add_logits(Path(session._model_path).name, batch, output)
            return output
        cnn_onnx.run_logits = logits


def make_run_core(recorder: Recorder, first_index: int) -> Any:
    def run_core(frames: Any, indexes: list[int], pipe: Any, stages: bool) -> None:
        from src.phase_j import live_process
        from src.phase_j.dshow_color import correct_601_as_709
        from src.phase_j.live_source import CapturedFrame
        try:
            from src.phase_j.live_color_buffer import configured_corrector
            corrector = configured_corrector() or correct_601_as_709
        except ImportError:
            corrector = correct_601_as_709  # m (dc24ee44) には補正bufferが無い
        guard = load_guard()
        for position, raw in enumerate(frames):
            guard()
            recorder.board_acc = 0.0
            started = time.perf_counter()
            image = corrector(raw)
            corrected = time.perf_counter()
            index = first_index + position
            notice = live_process.recognize(pipe, CapturedFrame(index, index / FPS, corrected, corrected, image))
            finished = time.perf_counter()
            recorder.color_ms.append((corrected - started) * MS)
            recorder.board_ms.append(recorder.board_acc * MS)
            recorder.recognize_ms.append((finished - corrected) * MS)
            recorder.add_notice(notice)
            del notice, image
            sys.modules['bench_live2'].NOTICES.clear()
    return run_core


def load_guard() -> Any:
    """基準側の展開ソースにscriptsが無くても使えるよう、ファイルから直接読む。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location('live_parallel_probe', ROOT / 'scripts/live_parallel_probe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.guard


def run_condition(args: argparse.Namespace) -> None:
    load_guard()()
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL)
    sys.path[:0] = [str(args.source_root), SPEED2]
    import bench_live2 as bench
    recorder = Recorder(args.logits)
    first = round(args.start * FPS)
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    build = bench.build_pipe

    def build_pipe(*values: Any, **kwargs: Any) -> Any:
        import cv2
        os.chdir(ASSETS)
        cv2.setNumThreads(1)
        install_hooks(recorder)
        return build(*values, **kwargs)
    bench.build_pipe = build_pipe
    bench.run_core = make_run_core(recorder, first)
    frames = DecodedFrames(args.video, args.start, args.frames)
    bench.load_frames = lambda *unused: (frames, list(range(first, first + args.frames)))
    sys.argv = ['bench', '--app', str(args.source_root), '--frames', 'unused', '--n',
        str(args.frames - WARMUP), '--warmup', str(WARMUP), '--layer', 'core', '--rec-config', REC_CONFIG,
        '--work', str(out / 'work'), '--out', str(out / 'bench_summary.json'),
        '--calibration-dir', str(out / 'calibration')]
    for item in args.env:
        sys.argv += ['--env', item]
    for item in ('PUYO_CNN_BACKEND=onnx', 'PUYO_LIVE_CPU_THREADS=1', 'PUYO_PERFORMANCE_CORES=1',
                 'PUYO_B3_SCORE_MATMUL=1', 'PUYO_B3_WARNING_PARALLEL=1', 'PUYO_B3_WARNING_WORKERS=4',
                 'PUYO_B3_SKIP_ICON_READING=1', 'PUYO_B3_LEDGER_MODE=1', 'PUYO_B3_LEDGER_LANDING=1',
                 'PUYO_B3_SHARE_SCORE_READ=1'):
        sys.argv += ['--env', item]
    started = time.time()
    bench.main()
    save_run(out, recorder, args, started)


def save_run(out: Path, recorder: Recorder, args: argparse.Namespace, started: float) -> None:
    import numpy as np
    np.save(out / 'boards.npy', np.stack(recorder.boards))
    (out / 'timing.json').write_text(json.dumps(dict(color_ms=recorder.color_ms,
        board_ms=recorder.board_ms, recognize_ms=recorder.recognize_ms)), encoding='utf-8')
    with (out / 'records.pickle').open('wb') as stream:
        pickle.dump(dict(records=recorder.records, stable=recorder.stable, rows=recorder.rows,
            row_total=recorder.row_total, row_inconsistent=recorder.row_inconsistent,
            calls=recorder.calls), stream)
    (out / 'condition.json').write_text(json.dumps(dict(vars(args), source_root=str(args.source_root),
        frames_done=len(recorder.records), started=started, ended=time.time()), default=str), encoding='utf-8')
    print(args.tag, 'frames', len(recorder.records), flush=True)


def load_run(tag: str) -> tuple[dict, Any]:
    import numpy as np
    with (OUT / tag / 'records.pickle').open('rb') as stream:
        return pickle.load(stream), np.load(OUT / tag / 'boards.npy')


def compare_runs(base_tag: str, cand_tag: str) -> dict[str, Any]:
    """frame別ハッシュ・盤面・CNN入力行別logitsを全件照合し、母数を併記する。"""
    import numpy as np
    base, base_boards = load_run(base_tag)
    cand, cand_boards = load_run(cand_tag)
    count = min(len(base['records']), len(cand['records']))
    field_diff = {name: sum(a[i] != b[i] for a, b in zip(base['records'], cand['records']))
                  for i, name in enumerate(RESULT_FIELDS)}
    stable = [s and t for s, t in zip(base['stable'], cand['stable'])]
    boards = {}
    for bi, name in enumerate(BOARD_FIELDS):
        left, right = base_boards[:count, :, bi], cand_boards[:count, :, bi]
        differ = left != right
        boards[name] = dict(cells=int(left.size), diff_cells=int(differ.sum()),
            diff_frames=int(differ.reshape(count, -1).any(axis=1).sum()))
    stable_mask = np.array([a and b for a, b in zip(base['stable'], cand['stable'])])
    stable_diff = int((base_boards[:count][stable_mask] != cand_boards[:count][stable_mask]).sum())
    rows = compare_rows(base, cand)
    return dict(base=base_tag, candidate=cand_tag, frames_base=len(base['records']),
        frames_candidate=len(cand['records']), frames_compared=count,
        frame_field_mismatch=field_diff, frames_any_mismatch=sum(a != b for a, b in
            zip(base['records'], cand['records'])), boards=boards,
        both_stable_frames=int(stable_mask.sum()), both_stable_board_diff_cells=stable_diff,
        both_stable_board_cells=int(base_boards[:count][stable_mask].size), logits=rows)


def compare_rows(base: dict, cand: dict) -> dict[str, Any]:
    """候補のCNN入力行すべてが基準にもあり、logitsがバイト一致することを確かめる。"""
    unknown = sum(key not in base['rows'] for key in cand['rows'])
    different = sum(key in base['rows'] and base['rows'][key] != value
                    for key, value in cand['rows'].items())
    return dict(base_calls=base['calls'], candidate_calls=cand['calls'],
        base_rows=base['row_total'], candidate_rows=cand['row_total'],
        base_unique_inputs=len(base['rows']), candidate_unique_inputs=len(cand['rows']),
        candidate_inputs_missing_in_base=unknown, logits_bytes_mismatch=different,
        base_self_inconsistent=base['row_inconsistent'], candidate_self_inconsistent=cand['row_inconsistent'])


def stats(values: list[float]) -> dict[str, float]:
    import numpy as np
    array = np.array(values)
    return dict(mean=float(array.mean()), p50=float(np.percentile(array, 50)),
                p95=float(np.percentile(array, 95)))


def speed_runs(tags: list[str]) -> dict[str, Any]:
    """暖機60枚と終端1枚を除いた段別時間 (ms) を、与えた順に要約する。"""
    result = {}
    for tag in tags:
        timing = json.loads((OUT / tag / 'timing.json').read_text(encoding='utf-8'))
        steady = {k: v[WARMUP:-1] for k, v in timing.items()}
        result[tag] = dict(n=len(steady['color_ms']), **{k: stats(v) for k, v in steady.items()})
    return result


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--tag', required=True)
    run.add_argument('--source-root', type=Path, default=ROOT)
    run.add_argument('--video', default='C:/puyo1_tools/soak/media/video_d04.mp4')
    run.add_argument('--start', type=float, default=650.0)
    run.add_argument('--frames', type=int, default=900)
    run.add_argument('--env', action='append', default=[])
    run.add_argument('--logits', action='store_true')
    cmp_ = sub.add_parser('compare')
    cmp_.add_argument('base')
    cmp_.add_argument('candidates', nargs='+')
    cmp_.add_argument('--out', type=Path, required=True)
    spd = sub.add_parser('speed')
    spd.add_argument('tags', nargs='+')
    spd.add_argument('--out', type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    sys.path.insert(0, str(ROOT))
    args = parse()
    if args.command == 'run':
        run_condition(args)
    elif args.command == 'compare':
        rows = [compare_runs(args.base, tag) for tag in args.candidates]
        args.out.write_text(json.dumps(rows, indent=1), encoding='utf-8')
        print(json.dumps(rows, indent=1))
    else:
        data = speed_runs(args.tags)
        args.out.write_text(json.dumps(data, indent=1), encoding='utf-8')
        print(json.dumps(data, indent=1))


if __name__ == '__main__':
    decode_main() if '--decode' in sys.argv else main()

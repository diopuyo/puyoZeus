"""本番を変更せず、推論境界と発火直後の予測を別ファイルに計装する。"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts._audit_firepower_1003_inventory import OUT, ROOT, resources

NAMES = tuple(f'c{i}' for i in range(1, 7)) + ('zenchi',) + tuple(f's{i}' for i in range(7))


def install(folder: Path) -> list[Any]:
    """入力を複写するだけのラッパー。元関数の結果をそのまま返す。"""
    from src.exchange_event_evaluator import FileExchangeModels
    from src.exchange_event_overlay import ExchangeEventOverlay
    handles = [gzip.open(folder / f'{n}.jsonl.gz', 'wt') for n in ('features', 'observations')]
    handles.append(gzip.open(folder / 'shown.jsonl.gz', 'wt'))
    install_display_capture(handles[-1])
    state: dict[str, Any] = {}
    predict, update = FileExchangeModels.predict_source_probability, ExchangeEventOverlay.update
    install_missing_capture(state)

    def captured_predict(self: Any, name: str, x: Any) -> float:
        p = predict(self, name, x)
        row = dict(state, model=name, x=x.tolist(), p=p,
                   static_r1s=self.landing_models is not None)
        handles[0].write(json.dumps(row) + '\n')
        return p

    def captured_update(self: Any, *args: Any, **kwargs: Any) -> Any:
        result, snapshot, finalization, sec, game = args[:5]
        state.update(t=sec, game=game, states=[result.p1.state.name, result.p2.state.name])
        answer = update(self, *args, **kwargs)
        if round(sec * 30) % 1800 == 0:
            print('progress', folder.name, sec, resources(), flush=True)
        chains = self.tracker.current.chains if self.tracker.current else []
        row = dict(state, chains=[asdict(c) for c in chains],
                   score_elapsed=self.tracker._score_elapsed,
                   observed=[vars(s.chain_event) if s.chain_event else None for s in (result.p1, result.p2)],
                   scores=list(args[6]), formulas=list(args[5]),
                   drops=[snapshot.total_dropped_to_p1, snapshot.total_dropped_to_p2],
                   source=self.tracker.source, p=self.tracker.probability)
        # 大きな盤面配列は予測誤差の集計には不要。
        for c in row['chains']:
            c.pop('predicted_final_board', None)
        handles[1].write(json.dumps(row, default=str) + '\n')
        return answer

    FileExchangeModels.predict_source_probability = captured_predict
    ExchangeEventOverlay.update = captured_update
    return handles


def install_display_capture(stream: Any) -> None:
    """CSVの平滑前確率とは別に、画面へ返された平滑後確率を記録する。"""
    from scripts import visualize_advantage_overlay as visual
    original = visual._exchange_display

    def captured(*args: Any, **kwargs: Any) -> Any:
        value = original(*args, **kwargs)
        sec = args[4] if len(args) > 4 else kwargs.get('t_sec')
        stream.write(json.dumps(dict(t_sec=sec, shown_p1=value[1])) + '\n')
        return value

    visual._exchange_display = captured


def install_missing_capture(state: dict[str, Any]) -> None:
    """0補完前のD欠測位置を保存する。"""
    import numpy as np
    from src import exchange_event_evaluator as evaluator
    original = evaluator.build_features

    def captured(event: Any, thresholds: Any) -> Any:
        answer = original(event, thresholds)
        item = event.firing if isinstance(event, evaluator.ExchangeEndInput) else event
        item = item.static if isinstance(item, evaluator.FiringInput) else item
        state['d_missing'] = np.flatnonzero(np.isnan(item.d_features)).tolist()
        return answer

    evaluator.build_features = captured


def worker(name: str) -> None:
    """本番CLIをそのまま呼び、監査先以外には書かない。"""
    from scripts import replay_exchange_event_20260926 as replay
    folder = OUT / 'replay' / name
    folder.mkdir(parents=True, exist_ok=True)
    before = display_hashes(folder)
    if before:
        (folder / 'status.json').unlink(missing_ok=True)
        (folder / 'EXTRACTED.json').unlink(missing_ok=True)
    (folder / 'RESOURCE.json').write_text(json.dumps(resources()))
    handles = install(folder)
    sys.argv = ['replay', str(ROOT / 'collect/records' / f'{name}.jsonl.gz'),
                '--out', str(folder), '--production-exchange-event']
    try:
        replay.main()
    finally:
        for stream in handles:
            stream.close()
    if before:
        after = display_hashes(folder)
        (folder / 'INSTRUMENTATION_EQUIVALENCE.json').write_text(json.dumps(
            dict(before=before, after=after, identical=before == after)))
        assert before == after, '計装追加前後の表示が不一致'


def display_hashes(folder: Path) -> dict[str, str]:
    """npzの圧縮日時によらず、全表示列のバイトを照合する。"""
    import numpy as np
    path = folder / 'display.npz'
    if not path.exists():
        return {}
    with np.load(path) as data:
        return {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in data.items()}


def main() -> None:
    """区間を逐次実行し、各区間開始前にもメモリ制約を確認する。"""
    if len(sys.argv) > 1 and sys.argv[1] == '--complete-capture':
        complete_capture()
        return
    if len(sys.argv) > 1 and sys.argv[1] == '--repair-initial':
        repair_initial()
        return
    if len(sys.argv) > 1:
        worker(sys.argv[1])
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        resources()
        if (OUT / 'replay' / name / 'status.json').exists():
            continue
        print('start', name, flush=True)
        subprocess.run([sys.executable, '-B', '-m', 'scripts._audit_firepower_1003_replay', name], check=True,
                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        print('done', name, flush=True)


def repair_initial() -> None:
    """追加計装前に開始した4区間だけを逐次再測定する。"""
    for name in NAMES[:4]:
        resources()
        subprocess.run([sys.executable, '-B', '-m', 'scripts._audit_firepower_1003_replay', name], check=True)


def complete_capture() -> None:
    """全区間終了後、計装追加前の区間だけを再測定する。"""
    while not all((OUT / 'replay' / n / 'status.json').exists() for n in NAMES):
        time.sleep(5)
    for name in NAMES:
        folder = OUT / 'replay' / name
        if (folder / 'shown.jsonl.gz').exists():
            continue
        resources()
        print('recapture', name, flush=True)
        subprocess.run([sys.executable, '-B', '-m', 'scripts._audit_firepower_1003_replay', name], check=True)
        (folder / 'EXTRACTED.json').unlink(missing_ok=True)
    (OUT / 'CAPTURE_COMPLETE.json').write_text(json.dumps(resources()))


if __name__ == '__main__':
    main()

"""multilanding 全応手探索の呼出しごとの壁時計を、保存記録の再生で計装する (段2)。

本番構成 (E36b = run_pending_expiry_full の再生) をそのまま走らせ、次の関数の呼出しごとの
所要と入力規模を JSONL に残す。判定・出力は一切変えない (関数を素通しで包むだけ)。
  - ExchangeLandingProjection._evaluate  (通知 1 回の全体)
  - evaluate_multilanding / prove_multilanding / cached_proof
  - PostCounterDeathBound.evaluate       (E35 の事後上界)
  - future_send                          (キャッシュ不命中のみ = 近未来探索の実行)
  - Search.responses / ChainSimulator.simulate の累計 (内訳)
使い方: python -m scripts.profile_multilanding_speed --source S --out logs/multilanding_speed/base
出力: <out>/profile_<source>.jsonl  と、通常再生の出力 <out>/on/...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from scripts import run_c65_guard_full as base
from scripts import run_pending_expiry_full as expiry

TARGET_TOLERANCE_SEC = 0.02  # 通知時刻の一致許容 (秒)
TOP_FUNCTIONS = 45           # cProfile の表示行数
SLOW_LOG_SEC = 0.0  # 全呼出しを記録する (分布の母数を落とさない)
OCCUPIED_ZERO = 0

ROWS: list[dict] = []
STACK: list[str] = []


def board_fill(board: Any) -> int:
    """盤面の非空セル数 (探索規模の目安)。"""
    return int(np.count_nonzero(board._grid != OCCUPIED_ZERO))


def timed(name: str, fn: Callable, describe: Callable[[tuple, dict, Any], dict]) -> Callable:
    """素通しの計時ラッパ。親呼出しの名前を一緒に残す。"""
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        parent = STACK[-1] if STACK else ''
        STACK.append(name)
        started = perf_counter()
        try:
            result = fn(*args, **kwargs)
        finally:
            elapsed = perf_counter() - started
            STACK.pop()
        if elapsed >= SLOW_LOG_SEC:
            ROWS.append(dict(fn=name, parent=parent, sec=elapsed, **describe(args, kwargs, result)))
        return result
    wrapper.__wrapped__ = fn
    return wrapper


def describe_prove(args: tuple, kwargs: dict, result: dict) -> dict:
    """prove_multilanding の入力規模と結果。"""
    board = args[0]
    # 単体ベンチ用に入力を全て残す (盤面は 13x6 のリスト)。elapsed は 6 桁目まで保持。
    inputs = dict(grid=board._grid.tolist(), queue=[int(v) for v in args[1]], incoming_in=args[2],
                  hands_in=args[3], elapsed_in=float(args[4]), credit_in=int(args[6]) if len(args) > 6
                  else int(kwargs.get('credit', 0)))
    responses = [r.get('responses', 0) for rounds in result.get('rounds', []) for r in rounds]
    digest = hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest()
    return dict(inputs=inputs, result_sha=digest, fill=board_fill(board), incoming=args[2], hands=args[3], queue_len=len(args[1]),
                nodes=result.get('nodes', 0), reason=result.get('reason'), dead=result.get('dead'),
                rounds=len(result.get('rounds', [])), responses_max=max(responses, default=0),
                responses_sum=sum(responses))


def describe_eval(args: tuple, kwargs: dict, result: Any) -> dict:
    """evaluate_multilanding: 通知時刻と各側の証明結果の要約。"""
    audits = result.get('multi_landing', []) if isinstance(result, dict) else []
    return dict(t_sec=float(args[5]), dead_sides=list(result.get('dead_sides', [])),
                reasons=[a.get('reason') for a in audits])


def describe_send(args: tuple, kwargs: dict, result: Any) -> dict:
    """future_send 不命中: 盤面規模と手数。"""
    raw, shape, dtype = args[0], args[1], args[2]
    grid = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return dict(fill=int(np.count_nonzero(grid != OCCUPIED_ZERO)), hands=args[4],
                inputs=dict(grid=grid.tolist(), queue=[int(v) for v in args[3]], hands_in=args[4],
                            elapsed_in=float(args[5])))


def install_hooks() -> None:
    """モジュール属性を計時ラッパへ差し替える (本番コードはディスク上不変)。"""
    from src import exchange_event_landing as landing
    from src import exchange_event_multilanding as multi
    from src import exchange_single_death_proof as single
    multi.prove_multilanding = timed('prove_multilanding', multi.prove_multilanding, describe_prove)
    multi.cached_proof = timed('cached_proof', multi.cached_proof, lambda a, k, r: dict(
        dead=r.get('dead'), nodes=r.get('nodes', 0)))
    single.cached_proof = multi.cached_proof
    multi.evaluate_multilanding = timed('evaluate_multilanding', multi.evaluate_multilanding, describe_eval)
    original = landing.future_send

    def send_miss(*args: Any, **kwargs: Any) -> float:
        before = original.cache_info().misses
        started = perf_counter()
        value = original(*args, **kwargs)
        if original.cache_info().misses > before:
            ROWS.append(dict(fn='future_send_miss', parent=STACK[-1] if STACK else '',
                             sec=perf_counter() - started, **describe_send(args, kwargs, value)))
        return value
    landing.future_send = send_miss
    landing.ExchangeLandingProjection._evaluate = timed(
        'projection_evaluate', landing.ExchangeLandingProjection._evaluate,
        lambda a, k, r: dict(t_sec=float(a[-1]) if isinstance(a[-1], (int, float)) else None))


def install_weighted_hook() -> None:
    """隠し段確率の候補列挙 (weighted_landing) の所要と、候補数・重複数を残す。"""
    from src import exchange_hidden_row_probability as weighted
    from src import exchange_event_landing as landing
    seen: dict[str, list] = dict(incoming=[], totals=[], scenarios=0)
    original_scenarios = weighted.scenarios
    original_inputs = landing.ExchangeLandingProjection._probability_inputs

    def scenarios(engine: Any, chains: list) -> list:
        result = original_scenarios(engine, chains)
        seen['scenarios'] = len(result)
        seen['totals'] = sorted({tuple(weighted.totals(chains, s).tolist()) for s, _ in result})
        return result

    def inputs(self: Any, overlay: Any, snapshot: Any, latest: tuple, incoming: list, *rest: Any) -> Any:
        seen['incoming'].append(tuple(incoming))
        return original_inputs(self, overlay, snapshot, latest, incoming, *rest)

    def describe(args: tuple, kwargs: dict, result: Any) -> dict:
        row = dict(scenarios=seen['scenarios'], distinct_totals=len(seen['totals']),
                   probability_inputs_calls=len(seen['incoming']),
                   distinct_incoming=len(set(seen['incoming'])))
        seen.update(incoming=[], totals=[], scenarios=0)
        return row
    weighted.scenarios = scenarios
    landing.ExchangeLandingProjection._probability_inputs = inputs
    weighted.weighted_landing = timed('weighted_landing', weighted.weighted_landing, describe)


def install_cprofile(targets: list[float], out: Path) -> None:
    """指定した通知時刻 (±TARGET_TOLERANCE_SEC) の `_evaluate` だけを cProfile で包み、上位を保存する。"""
    import cProfile
    import io
    import pstats
    from src import exchange_event_landing as landing
    inner = landing.ExchangeLandingProjection._evaluate

    def profiled(self: Any, *args: Any, **kwargs: Any) -> Any:
        stamp = args[-1]
        if not any(abs(float(stamp) - t) < TARGET_TOLERANCE_SEC for t in targets):
            return inner(self, *args, **kwargs)
        profile = cProfile.Profile()
        started = perf_counter()
        profile.enable()
        try:
            return inner(self, *args, **kwargs)
        finally:
            profile.disable()
            stream = io.StringIO()
            pstats.Stats(profile, stream=stream).sort_stats('cumulative').print_stats(TOP_FUNCTIONS)
            path = out / f'cprofile_{float(stamp):.3f}.txt'
            path.write_text(f'wall={perf_counter() - started:.3f}s' + chr(10) + stream.getvalue())
    landing.ExchangeLandingProjection._evaluate = profiled


def install_post_counter_hook() -> None:
    """d5_runtime.install が読み込んだ E35 (0f28a04版) の evaluate を包む。"""
    module = sys.modules['src.exchange_post_counter_bound']
    cls = module.PostCounterDeathBound
    cls.evaluate = timed('post_counter_evaluate', cls.evaluate,
                         lambda a, k, r: dict(t_sec=float(a[5])))


def replay_record(record: Path, dest: Path, variant: str = 'exact') -> None:
    """任意の記録を E36b と同じ構成 (モデル v3・live_count) で再生する。variant は gate と同じ名前。"""
    from scripts import run_multilanding_speed_gate as gate
    from scripts import replay_exchange_event_20260926 as replay_module
    from scripts import d5_runtime
    dest.mkdir(parents=True, exist_ok=True)
    d5_runtime.OUT = dest.parent.parent / 'runtime'
    d5_runtime.install()
    result = replay_module.replay(record, dest, Path('models/exchange_event_v3'), True, None,
                                  **gate.options(variant))
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True,
                        help='保存記録の名前。--record 指定時は出力ファイル名にだけ使う')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cprofile-at', type=str, default='',
                        help='この通知時刻 (秒、カンマ区切り) の _evaluate だけを cProfile で保存する')
    parser.add_argument('--head-modules', action='store_true',
                        help='E35 の証明器を固定版 (0f28a04) でなく現行 src (E35b 高速版) で読む')
    parser.add_argument('--variant', default='exact', choices=('exact', 'bounded', 'bounded2'),
                        help='--record の再生オプション (run_multilanding_speed_gate と同じ名前)')
    parser.add_argument('--record', type=Path, default=None,
                        help='リアルタイム実行が保存した inputs.jsonl.gz など任意の記録を直接再生する')
    args = parser.parse_args()
    assert args.record is not None or args.source in base.SOURCES
    expiry.configure()
    base.REPLAY_OUT = args.out
    base.COLLECT_OUT = expiry.COLLECT_OUT  # 記録は保存済みの本番記録を読む
    from scripts import d5_runtime
    original_install = d5_runtime.install

    def install_and_hook() -> None:
        if args.head_modules:
            import src.exchange_post_counter_bound  # noqa: F401  現行 src をそのまま使う
        else:
            original_install()
        install_post_counter_hook()
    d5_runtime.install = install_and_hook
    install_hooks()
    install_weighted_hook()
    if args.cprofile_at:
        args.out.mkdir(parents=True, exist_ok=True)
        install_cprofile([float(v) for v in args.cprofile_at.split(',')], args.out)
    started = perf_counter()
    try:
        if args.record is None:
            base.replay(args.source)
        else:
            replay_record(args.record, args.out / 'on' / args.source, args.variant)
    finally:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f'profile_{args.source}.jsonl'
        with path.open('w') as stream:
            for row in ROWS:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps(dict(rows=len(ROWS), wall_sec=perf_counter() - started)), flush=True)


if __name__ == '__main__':
    main()

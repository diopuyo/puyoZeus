"""コマ落ちに強い状態判定 (live_frame_drop_robust) の箇所別 OFF (既定は何も外さない)。

live_frame_drop_robust は 1 本のフラグで全箇所 (R01・R02 (R03 込み)・R04・R06・R10・R15・B07・R35) を ON にする。
箇所ごとの採否が測定で分かれたとき、再組立てなしに puyo_live.json の "live_frame_drop_rules_off" で
外す箇所だけを OFF と同じ数え方へ戻せるようにする (例: ["R06", "R10", "R15"])。
実装は、認識 process の起動時に、外す箇所の判定メソッドだけを「OFF と同じ数え方」の関数へ差し替える。
  - 未指定 (空) なら何も差し替えない (呼び出しも無く、従来の live_frame_drop_robust のまま)。
  - 差し替え先の名前が無ければ AttributeError で止める (箇所の名前が変わったのに黙って効かない、を防ぐ)。
  - 差し替えた箇所は FRAME_DROP_ROBUST の印 (run.log・標準出力) と live_trial_flags.json に残る。
外した箇所の診断の数え (audit) は書かれない (母数 0 = その箇所は働かなかった)。
"""
from __future__ import annotations

from typing import Any, Callable

from src.phase_j.live_speed import FRAME_DROP_RULES


def replace(target: Any, name: str, new: Callable[..., Any]) -> None:
    """target の属性 name を new へ差し替える。無い名前なら AttributeError (黙って新設しない)。"""
    if not hasattr(target, name):
        raise AttributeError(f'{getattr(target, "__name__", target)} に {name} がありません (箇所の名前が変わった?)')
    setattr(target, name, new)


def _plain_append(self: Any, cnn_board: Any, media_tick: Any) -> None:
    """R02 を外したとき: OFF と同じ積み方 (件数が大きさを超えたら最古を 1 件捨てる)。"""
    history = self._ctx.non_stable_cnn_history
    history.append(cnn_board.copy())
    if len(history) > self._non_stable_history_size:
        history.pop(0)


def _off_state_machine(rules: tuple[str, ...]) -> None:
    """src.board_state_machine の R01・R02・R04。"""
    import src.board_state_machine as module
    machine = module.BoardStateMachine
    if 'R01' in rules:
        original = machine._pending_reached
        replace(machine, '_pending_reached', lambda self, media_tick: original(self, None))
    if 'R02' in rules:
        replace(machine, '_resume_vote_weights', lambda self, signals: None)
        replace(machine, '_append_non_stable_history_ticked', _plain_append)
    if 'R04' in rules:
        replace(module, '_recovery_count_reached', lambda ctx, cell, count, need, tick: count >= need)


def _off_detectors(rules: tuple[str, ...]) -> None:
    """R06 (src.state_detectors) と R10 (src.ojama_visual_detector)。"""
    if 'R06' in rules:
        import src.state_detectors as module
        replace(module.GravitySettleDetector, '_settle_reached_ticked',
                lambda self, stamp: self._stable_consec >= module.GRAVITY_SETTLE_MIN_FRAMES)
    if 'R10' in rules:
        import src.ojama_visual_detector as ojama
        original = ojama.OjamaVisualDetector._board_settled
        replace(ojama.OjamaVisualDetector, '_board_settled', lambda self, stamp: original(self, None))


def _off_pipeline_and_queues(rules: tuple[str, ...]) -> None:
    """R15・R35 (src.recognition_pipeline) と B07 (src.b3_live_queues)。"""
    if 'R15' in rules or 'R35' in rules:
        import src.recognition_pipeline as module
        if 'R35' in rules:
            replace(module.RecognitionPipeline, '_r35_missed_placement', lambda self, side, time_sec, via: None)
        if 'R15' in rules:
            replace(module.RecognitionPipeline, '_r15_warmup_step', lambda self: 1)
    if 'B07' in rules:
        import src.b3_live_queues as queues
        replace(queues.LiveQueues, 'move_window_frame', lambda self, pipe, frame: None)


def apply_rules_off(rules: tuple[str, ...]) -> tuple[str, ...]:
    """rules の箇所を OFF と同じ数え方へ戻す (空なら何もしない)。差し替えた箇所を返す。

    認識 process (spawn の子) で 1 回だけ呼ぶ。クラス・モジュールの属性を差し替えるので、同じ process へ
    もう一度呼んでも結果は同じ (冪等)。元へ戻す口は作らない (試験は monkeypatch で退避する)。
    """
    unknown = [rule for rule in rules if rule not in FRAME_DROP_RULES]
    if unknown:
        raise ValueError(f'知らない箇所: {unknown} (候補 {list(FRAME_DROP_RULES)})')
    if not rules:
        return ()
    _off_state_machine(rules)
    _off_detectors(rules)
    _off_pipeline_and_queues(rules)
    return tuple(rules)

"""Gate 3-2b 実データ検証プローブ (2026-08-24 初版、2026-08-25 更新)。

## 何を測るか

`src/exchange_episode_tracker.py` (台帳・解決器・tracker) はここまで
合成データのテストしか通っていない。実際の動画で保存則が成り立つのか、
まだ1件も測っていない。本番の表示経路
(`scripts/visualize_advantage_overlay.py`) へ配線する前に、実データで測る。

**本番ファイルは1つも変更しない。** `scripts/visualize_advantage_overlay.py`
は import すらしない (読むだけの参考にとどめた)。

## 【2026-08-25 更新】mechanism='baseline' 是正後の再測定

`ObservationKind.BASELINE` を `CHAIN_SETTLED` (値は使わない) と
`SCORE_FINALIZE` (score OCR 確定差分、値の権威) に分割した設計変更を受け、
本プローブに `observe_generation()` の供給を追加した
(`OjamaAccountSnapshot.total_generated_by_p1/p2` の前フレーム差分)。
併せて以下をコマンドライン引数化し、複数動画・複数窓で使い回せるようにした
(既存の `data/verify/gate3_episode_2026-08-24/` は絶対に上書きしない。
既定の出力先も新しいディレクトリに変更した):

- `--video`: 対象動画パス
- `--t0` / `--t1`: 対象窓 (秒)
- `--out-dir`: 出力先ディレクトリ

## 既存資産の流用 (指示により報告)

探した結果、以下を **そのまま import して流用**した。ゼロから書いたのは
「フレームループの本体と ExchangeEpisodeTracker への変換」の部分のみ。

- `scripts/collect_indicators_v2.py::_drive_ojama` —
  `OjamaAccountingTracker` を `on_state_transition` / `on_tsumo_settled`
  (tsumo_count 増分駆動) で毎フレーム駆動し `OjamaAccountSnapshot` を
  返す既存関数。そのまま import した。
- `scripts/collect_indicators_v2.py::_SideTracker` — tsumo_count 駆動用の
  補助トラッカー (`game_idx` / `prev_score` / `prev_tsumo` を保持)。
  そのまま import した。
- `scripts/collect_indicators_v2.py::_update_game_idx` — score 大幅減少
  (`SCORE_RESET_THRESHOLD`) による試合境界検知。そのまま import した。
- `scripts/collect_indicators_v2.py::collect()` の `cv2.VideoCapture` ループ
  構造 (fps 取得 → seek → `pipeline.update` → `_drive_ojama` → 状態保持
  更新という順序) を**参考にした** (CSV 出力・指標計算等の不要部分は
  持ち込まず、本プローブに必要な部分だけを独立ループとして書いた)。
- `scripts/_probe_formula_interlude_2026-08-24.py` の「ChainEvent の
  (trigger_sec, mechanism, chain_count, total_score) が変化した瞬間だけ
  記録する」パターンを参考にした (同一値の重複 push を避けるため)。

`scripts/visualize_advantage_overlay.py` 本体はモンキーパッチ経由で使わず、
`collect_indicators_v2.py` 側の軽量な部品だけを使う設計にした。理由:
`vao.generate` は描画・多数の他 tracker を含む巨大関数であり、本プローブが
必要とする「pipeline.update + お邪魔会計 drain」だけを取り出すには
`collect_indicators_v2.py` の部品のほうが過不足なく合致するため。

## 制約

- 変更してよいのは本ファイルのみ。
- 既存の差分・ログ・成果物 (`data/verify/gate3_episode_2026-08-24/` 等) を
  reset / checkout / stash / 削除 / 上書きしない。

使い方 (例、2026-08-25 zenchi 再測定):
  python scripts/_gate3_episode_probe_2026-08-24.py \\
      --video data/frames/video_zenchi_c0BQoMJwwQU.mp4 --t0 780.0 --t1 1080.0 \\
      --out-dir data/verify/gate3_episode_fixed_2026-08-25/zenchi
出力: <out-dir>/diagnostics.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

cv2.setNumThreads(1)

from scripts.collect_indicators_v2 import (  # noqa: E402
    DEFAULT_FPS,
    TARGET_H,
    TARGET_W,
    _drive_ojama,
    _SideTracker,
    _update_game_idx,
)
from src.board_state_machine import BoardState  # noqa: E402
from src.chain_detector import ChainEvent  # noqa: E402
from src.exchange_episode_tracker import (  # noqa: E402
    ChainEventObservation,
    ExchangeEpisodeTracker,
    GenerationObservation,
    SettlementObservation,
)
from src.chain_id_resolver import FINALIZED_SOURCE_SCORE_OCR_DIFF  # noqa: E402
from src.exchange_ledger import FINALIZE_DOWNWARD_TOLERANCE  # noqa: E402
from src.ojama_accounting import OjamaAccountingTracker  # noqa: E402
from src.recognition_pipeline import RecognitionPipeline  # noqa: E402

# 2026-08-25 追加: コーディネーター指摘の残件1〜3を計装で確定するための
# トレース (診断専用。会計そのものには一切使わない)。
_TRACE_FIELDS: tuple[str, ...] = (
    "total_generated_by_p1", "total_generated_by_p2",
    "total_offset_by_p1", "total_offset_by_p2",
    "total_dropped_to_p1", "total_dropped_to_p2",
    "pending_p1", "pending_p2",
    # 2026-08-25 追加: 残件2の cap 起因説を検証するため、cap 前の並行帳簿
    # (`src/ojama_accounting.py:222-223`) もトレースに含める (読み取りのみ)。
    "pending_p1_uncapped", "pending_p2_uncapped",
)

# 状態機械のウォームアップ (既存プローブ _probe_formula_interlude_2026-08-24.py
# と同じ 30 秒)。ウォームアップ中も chain_event/snapshot の追跡は続けるが、
# tracker への観測供給は t0 以降のみ行う (境界での偽の差分計上を防ぐ)。
WARMUP_SEC: float = 30.0

# 掛け算式の段を正しく取るために必須の構成 (タスク指定)。
FORMULA_FLAGS: dict[str, bool] = dict(
    enable_chain_formula_read_verify=True,
    enable_formula_chain_count_update=True,
    enable_slide_exit_no_min_display=True,
    enable_formula_step_interlude=True,
)

_SIDE_LABELS: tuple[str, str] = ("1P", "2P")


def _parse_args() -> argparse.Namespace:
    """CLI 引数 (2026-08-25 追加。動画・窓・出力先を使い回せるようにする)。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video", type=str,
        default=str(PROJECT_ROOT / "data/frames/video_zenchi_c0BQoMJwwQU.mp4"),
    )
    parser.add_argument("--t0", type=float, default=780.0)
    parser.add_argument("--t1", type=float, default=1080.0)
    parser.add_argument(
        "--out-dir", type=str,
        default=str(PROJECT_ROOT / "data/verify/gate3_episode_2026-08-24"),
    )
    return parser.parse_args()


def _chain_event_key(ev: ChainEvent | None) -> tuple | None:
    """ChainEvent の同一性キー (変化検知用。既存プローブと同じ4値)。"""
    if ev is None:
        return None
    return (round(ev.trigger_sec, 3), ev.mechanism, ev.chain_count, ev.total_score)


def _to_chain_observation(
    side_label: str, ev: ChainEvent, t_sec: float, game_idx: int, elapsed_sec: float,
    authoritative_ojama: float | None,
) -> ChainEventObservation:
    """ChainEvent から tracker が要求する最小情報を抜き出す。

    `authoritative_ojama` (2026-08-25 追加): 同一フレームで観測された
    score OCR 確定生成量の差分 (`_diff_generation` 参照)。無ければ `None`
    (自己検算はスキップとして計上される)。
    """
    return ChainEventObservation(
        side=side_label, t_sec=t_sec, mechanism=ev.mechanism or "",
        chain_count=ev.chain_count, total_score=ev.total_score,
        ojama_sent=ev.ojama_sent, game_idx=game_idx, elapsed_sec=elapsed_sec,
        authoritative_ojama=authoritative_ojama,
    )


def _diff_settlement(prev_snap, snap, t_sec: float, game_idx: int):
    """前フレームとの累積カウンタ差分から SettlementObservation を作る。

    累積カウンタは通常単調増加だが、試合境界等で減少する可能性があるため
    負の差分は 0 にクリップする (黙って無視はしない、呼び出し側でログする)。
    """
    d_c1 = snap.total_offset_by_p1 - prev_snap.total_offset_by_p1
    d_c2 = snap.total_offset_by_p2 - prev_snap.total_offset_by_p2
    d_l1 = snap.total_dropped_to_p1 - prev_snap.total_dropped_to_p1
    d_l2 = snap.total_dropped_to_p2 - prev_snap.total_dropped_to_p2
    negatives = [name for name, v in (
        ("canceled_by_1p", d_c1), ("canceled_by_2p", d_c2),
        ("landed_on_1p", d_l1), ("landed_on_2p", d_l2),
    ) if v < 0.0]
    if negatives:
        print(f"[warn] t={t_sec:.2f} 負の差分をクリップ: {negatives}", flush=True)
    d_c1, d_c2, d_l1, d_l2 = (max(0.0, v) for v in (d_c1, d_c2, d_l1, d_l2))
    if d_c1 == 0.0 and d_c2 == 0.0 and d_l1 == 0.0 and d_l2 == 0.0:
        return None
    return SettlementObservation(
        t_sec=t_sec, game_idx=game_idx,
        canceled_by_1p=float(d_c1), canceled_by_2p=float(d_c2),
        landed_on_1p=float(d_l1), landed_on_2p=float(d_l2),
    )


def _snapshot_trace_row(t_sec: float, snap, tsumo_1p: int, tsumo_2p: int) -> dict:
    """診断トレース 1 行 (2026-08-25 追加、残件2 の切り分け用)。"""
    row = {"t_sec": round(t_sec, 3), "tsumo_1p": tsumo_1p, "tsumo_2p": tsumo_2p}
    for field in _TRACE_FIELDS:
        row[field] = getattr(snap, field)
    return row


def _diff_generation(prev_snap, snap) -> dict[str, int]:
    """前フレームとの `total_generated_by_p1/p2` 差分 (2026-08-25 追加)。

    **クリップしない。** `observe_generation()` 自身が負の差分を
    `negative_generation_delta_count` として数える設計 (2026-08-25
    コーディネーター指摘) のため、ここで隠さずそのまま渡す。
    """
    return {
        "1P": snap.total_generated_by_p1 - prev_snap.total_generated_by_p1,
        "2P": snap.total_generated_by_p2 - prev_snap.total_generated_by_p2,
    }


def _diagnose_self_check_targets(episode_tracker: ExchangeEpisodeTracker) -> dict:
    """残件1 (自己検算が発火しない) の根因を計装で確定する (2026-08-25 追加)。

    `finalized_source==score_ocr_diff` で確定した chain ごとに、
    `_context_by_t.get(closed_at_sec)` が見つかるか・`authoritative_ojama`
    が設定されているかを直接調べる (read-only の private アクセス、
    `ojama_tracker._elapsed(t)` と同じ既存慣習)。
    """
    resolved = episode_tracker._resolver.resolved()  # noqa: SLF001
    n_score_ocr_diff = 0
    n_context_found = 0
    n_auth_present = 0
    per_chain: list[dict] = []
    for rc in resolved:
        if rc.finalized_source != FINALIZED_SOURCE_SCORE_OCR_DIFF:
            continue
        n_score_ocr_diff += 1
        ctx = episode_tracker._context_by_t.get(rc.closed_at_sec)  # noqa: SLF001
        found = ctx is not None
        has_auth = found and ctx.authoritative_ojama is not None
        n_context_found += int(found)
        n_auth_present += int(has_auth)
        per_chain.append({
            "chain_id": rc.chain_id, "side": rc.side,
            "closed_at_sec": rc.closed_at_sec,
            "context_found": found, "authoritative_ojama_present": has_auth,
        })
    return {
        "n_score_ocr_diff_chains": n_score_ocr_diff,
        "n_context_found": n_context_found,
        "n_authoritative_ojama_present": n_auth_present,
        "per_chain": per_chain,
    }


def _dump_rejected_chains(episode_tracker: ExchangeEpisodeTracker) -> list[dict]:
    """残件4 (`rejected_divergence_amount_total=0.0` の解釈) を確定するための
    生値ダンプ (2026-08-25 追加)。`simulate_fallback` で確定した chain の
    `provisional_score`/`finalized_score` をそのまま出す。
    """
    resolved = episode_tracker._resolver.resolved()  # noqa: SLF001
    return [
        {
            "chain_id": rc.chain_id, "side": rc.side,
            "growth_observed": rc.growth_observed,
            "provisional_score": rc.provisional_score,
            "finalized_score": rc.finalized_score,
        }
        for rc in resolved
        if rc.was_finalized and rc.finalized_source != FINALIZED_SOURCE_SCORE_OCR_DIFF
    ]


def _dump_all_resolved_chains(episode_tracker: ExchangeEpisodeTracker) -> list[dict]:
    """全 resolved chain の生値ダンプ (2026-08-25 追加、コーディネーター指摘の

    修正案B検証用)。SUPERSEDED で閉じた chain とその直後に同じ side で開いた
    chain が同一の物理連鎖の分裂かどうかを、段数・時刻・スコアの連続性で
    目視・機械検査するために必要な最小限の情報を出す。
    """
    resolved = episode_tracker._resolver.resolved()  # noqa: SLF001
    return [
        {
            "chain_id": rc.chain_id, "side": rc.side,
            "opened_at_sec": rc.opened_at_sec, "closed_at_sec": rc.closed_at_sec,
            "step_count": rc.step_count, "provisional_score": rc.provisional_score,
            "finalized_score": rc.finalized_score, "was_finalized": rc.was_finalized,
            "force_cut": rc.force_cut, "close_reason": rc.close_reason.name,
            "growth_observed": rc.growth_observed,
            "finalized_source": rc.finalized_source,
        }
        for rc in resolved
    ]


def _run_probe(video: Path, t0: float, t1: float, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[ERROR] cannot open: {video}", file=sys.stderr)
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    start_sec = max(0.0, t0 - WARMUP_SEC)
    start_frame = int(start_sec * fps)
    end_frame = int(t1 * fps)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))

    pipeline = RecognitionPipeline.load_default(
        stable_frame_count=3, load_score_ocr=True, enable_chain_tracker=True,
        temporal_smoothing=1, load_next_detector=True, force_in_match=True,
        **FORMULA_FLAGS,
    )
    ojama_tracker = OjamaAccountingTracker()
    ojama_tracker.reset()
    tracker_p1 = _SideTracker()
    tracker_p2 = _SideTracker()
    prev_state_p1 = BoardState.MENU
    prev_state_p2 = BoardState.MENU
    episode_tracker = ExchangeEpisodeTracker(enabled=True)

    last_chain_key: dict[str, tuple | None] = {"1P": None, "2P": None}
    prev_snap = None
    n_observed_frames = 0
    n_chain_observations = 0
    n_settlement_observations = 0
    n_generation_observations = 0
    n_negative_generation_diffs = 0
    # 2026-08-25 追加 (残件1〜3 の計装)。
    n_auth_assigned: dict[str, int] = {"1P": 0, "2P": 0}
    trace_rows: list[dict] = []
    last_trace_key: tuple | None = None
    # 2026-08-25 追加: 修正案B検証用の生 ChainEvent 観測ログ (診断専用)。
    raw_chain_obs: list[dict] = []

    fi = start_frame
    while fi < end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[:2] != (TARGET_H, TARGET_W):
            frame = cv2.resize(
                frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA,
            )
        t_sec = fi / fps
        if fi % 1800 == 0:
            print(f"[progress] fi={fi} t={t_sec:.2f}", flush=True)
        result = pipeline.update(fi, t_sec, frame)
        snap = _drive_ojama(
            ojama_tracker, result.p1, result.p2, prev_state_p1, prev_state_p2, t_sec,
            tracker_p1=tracker_p1, tracker_p2=tracker_p2, pipeline=pipeline,
        )
        _update_game_idx(tracker_p1, result.p1.score)
        _update_game_idx(tracker_p2, result.p2.score)
        prev_state_p1 = result.p1.state
        prev_state_p2 = result.p2.state

        game_idx = max(tracker_p1.game_idx, tracker_p2.game_idx)
        if tracker_p1.game_idx != tracker_p2.game_idx:
            print(
                f"[warn] t={t_sec:.2f} game_idx 不一致: "
                f"p1={tracker_p1.game_idx} p2={tracker_p2.game_idx}", flush=True,
            )
        # 既存本番コード (scripts/visualize_advantage_overlay.py) と同じ
        # private アクセスの慣習 (`tracker._elapsed(t)`)。read-only、
        # src/ojama_accounting.py は変更しない。
        elapsed_sec = ojama_tracker._elapsed(t_sec)  # noqa: SLF001

        gen_diff = _diff_generation(prev_snap, snap) if prev_snap is not None else None

        # 2026-08-25 追加: 残件2 (着弾がほぼ観測されない) の切り分け用トレース。
        # 会計に使う値は 1 つも変えない (診断専用の読み取りのみ)。
        if t_sec >= t0:
            tsumo_1p = pipeline.tsumo_count("1P")
            tsumo_2p = pipeline.tsumo_count("2P")
            trace_key = (
                snap.total_generated_by_p1, snap.total_generated_by_p2,
                snap.total_offset_by_p1, snap.total_offset_by_p2,
                snap.total_dropped_to_p1, snap.total_dropped_to_p2,
                snap.pending_p1, snap.pending_p2, tsumo_1p, tsumo_2p,
                # 2026-08-25 追加: uncapped 側だけが変化する瞬間 (cap 起因の
                # 消失を追う残件2 の検証) もトレース行として残す。
                snap.pending_p1_uncapped, snap.pending_p2_uncapped,
            )
            if trace_key != last_trace_key:
                trace_rows.append(_snapshot_trace_row(t_sec, snap, tsumo_1p, tsumo_2p))
                last_trace_key = trace_key

        for side_label, side_result in (("1P", result.p1), ("2P", result.p2)):
            ev = side_result.chain_event
            key = _chain_event_key(ev)
            changed = key != last_chain_key[side_label]
            last_chain_key[side_label] = key
            if changed and t_sec >= t0 and ev is not None:
                auth = (
                    float(gen_diff[side_label])
                    if gen_diff is not None and gen_diff[side_label] != 0
                    else None
                )
                if auth is not None:
                    n_auth_assigned[side_label] += 1
                episode_tracker.observe(_to_chain_observation(
                    side_label, ev, t_sec, game_idx, elapsed_sec, auth,
                ))
                n_chain_observations += 1
                # 2026-08-25 追加: 修正案B (連鎖の過剰分割) 検証用の生観測ログ。
                # resolver へ渡す直前の (mechanism, chain_count, total_score) を
                # そのまま記録する (会計には使わない、診断専用)。
                raw_chain_obs.append({
                    "t_sec": round(t_sec, 3), "side": side_label,
                    "mechanism": ev.mechanism, "chain_count": ev.chain_count,
                    "total_score": ev.total_score,
                })

        if gen_diff is not None and t_sec >= t0:
            for side_label in _SIDE_LABELS:
                delta = gen_diff[side_label]
                if delta == 0:
                    continue
                if delta < 0:
                    n_negative_generation_diffs += 1
                    print(
                        f"[warn] t={t_sec:.2f} side={side_label} "
                        f"生成累積カウンタが減少: delta={delta}", flush=True,
                    )
                episode_tracker.observe_generation(GenerationObservation(
                    side=side_label, t_sec=t_sec, game_idx=game_idx,
                    generated_delta=delta,
                ))
                n_generation_observations += 1

        if prev_snap is not None and t_sec >= t0:
            settlement = _diff_settlement(prev_snap, snap, t_sec, game_idx)
            if settlement is not None:
                episode_tracker.observe_settlement(settlement)
                n_settlement_observations += 1
        prev_snap = snap
        if t_sec >= t0:
            n_observed_frames += 1
        fi += 1

    cap.release()
    episode_tracker.finish()
    diag = episode_tracker.diagnostics()

    # 台帳レベルの累積カウンタ (retired/duplicate_generated_suppressed/
    # finalize_rejected) は `ExchangeEpisodeDiagnostics` に出ていないため、
    # `scripts/_gate3_episode_probe_2026-08-24.py` 自身の既存慣習
    # (`ojama_tracker._elapsed(t)` 等の private アクセス) にならい、
    # 台帳の `snapshot()` を直接読む (read-only。src は変更しない)。
    ledger_snapshot = episode_tracker._ledger.snapshot()  # noqa: SLF001
    closed = episode_tracker._ledger.closed_episodes()  # noqa: SLF001
    closed_status_counts: dict[str, int] = {}
    for summary in closed:
        closed_status_counts[summary.status.name] = (
            closed_status_counts.get(summary.status.name, 0) + 1
        )
    d2_divergence_count = len(diag.d2.divergences)
    d2_gate_held_ratio = (
        diag.d2.gate_held_count / d2_divergence_count if d2_divergence_count else None
    )
    d2_abs_over_tolerance_count = sum(
        1 for d in diag.d2.divergences if abs(d) > FINALIZE_DOWNWARD_TOLERANCE
    )
    d2_abs_over_tolerance_ratio = (
        d2_abs_over_tolerance_count / d2_divergence_count if d2_divergence_count else None
    )
    self_check_diag = _diagnose_self_check_targets(episode_tracker)
    rejected_chain_dump = _dump_rejected_chains(episode_tracker)
    all_chains_dump = _dump_all_resolved_chains(episode_tracker)

    report = {
        "video": str(video),
        "window": {"t0": t0, "t1": t1, "warmup_sec": WARMUP_SEC},
        "n_observed_frames": n_observed_frames,
        "n_chain_observations": n_chain_observations,
        "n_settlement_observations": n_settlement_observations,
        "n_generation_observations": n_generation_observations,
        "n_negative_generation_diffs_seen_in_probe": n_negative_generation_diffs,
        "n_auth_assigned": n_auth_assigned,
        "self_check_diagnosis": self_check_diag,
        "rejected_chain_dump": rejected_chain_dump,
        "all_chains_dump": all_chains_dump,
        "raw_chain_obs": raw_chain_obs,
        "trace": trace_rows,
        "diagnostics": dataclasses.asdict(diag),
        "ledger_extra": {
            "retired_chain_count": ledger_snapshot.retired_chain_count,
            "retired_unreconciled": ledger_snapshot.retired_unreconciled,
            "duplicate_generated_suppressed_count":
                ledger_snapshot.duplicate_generated_suppressed_count,
            "duplicate_generated_suppressed_amount":
                ledger_snapshot.duplicate_generated_suppressed_amount,
            "finalize_rejected_count": ledger_snapshot.finalize_rejected_count,
            "finalize_rejected_amount": ledger_snapshot.finalize_rejected_amount,
        },
        "n_closed_episodes": len(closed),
        "closed_episode_status_counts": closed_status_counts,
        "d2_divergence_count": d2_divergence_count,
        "d2_gate_held_ratio": d2_gate_held_ratio,
        "d2_abs_over_tolerance_count": d2_abs_over_tolerance_count,
        "d2_abs_over_tolerance_ratio": d2_abs_over_tolerance_ratio,
    }
    out_path = out_dir / "diagnostics.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[saved] {out_path}", flush=True)


if __name__ == "__main__":
    args = _parse_args()
    _run_probe(Path(args.video), args.t0, args.t1, Path(args.out_dir))

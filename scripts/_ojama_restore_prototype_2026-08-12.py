"""おじゃま収支2列 (ojama_net_balance / ojama_forecast) の近似復元プロトタイプ。

## 背景 (2026-08-12 user採用: docs/INDICATOR_REORG_PROPOSAL_2026-08-12.md 提案0-3)
scripts/build_labeled_win_from_npz.py (npz→学習CSV変換) は npz が STABLE 重複除去済み
スナップショットのみ (フレーム間の BoardState 遷移列を保持しない) のため、
OjamaAccountingTracker (src/ojama_accounting.py) が要求する frame-level 駆動
(on_state_transition / on_tsumo_settled) を再現できず、この2列を出力できない。

本スクリプトは npz に既に保存されている `score` 列 (collect_boards_lean.py が
常時保存、後方互換フィールド) だけを使って、この2列を「近似復元」する。

## 復元アルゴリズムの設計方針
- 実装は 100% 既存の純関数 (src.ojama_accounting.score_to_ojama /
  cancel_own_pending_then_send_surplus, src.scoring.compute_effective_rate) を
  再利用する (マジックナンバー・会計ロジックの二重実装を避ける、CLAUDE.md規約)。
- 「1つの新規 STABLE snapshot = 1回の tsumo 着地イベント」と仮定する
  (根拠: 得点はぷよ消去でのみ増加するため、同一 side の連続する2つの
  STABLE snapshot 間で得点が変化したなら、その差分は連鎖の合計得点そのもの。
  変化しなければ「連鎖なしの着地」)。
  - 得点差分 >= CHAIN_TOTAL_MIN_SCORE (40): 連鎖発生。着地による自分の
    pending drain → score_to_ojama で生成量 gen を算出 →
    cancel_own_pending_then_send_surplus で相殺+相手への繰越。
  - それ未満 (0 含む): 連鎖なしの着地。pending drain のみ。
  - 負の差分: score OCR 誤読疑い (試合境界は別途 reset で処理済のはず)。
    drain のみ行い chain 計算はスキップする (過剰計上防止、診断ログに記録)。
- 試合境界: 単一 side の score が SCORE_RESET_THRESHOLD 以上減少したら
  その side の leftover/forecast を 0 にリセットする (OjamaAccountingTracker.
  _reset_side_boundary と同じ、side 独立)。MENU 遷移による reset は
  npz に BoardState が残っていないため検出不可 (既知の非対応、レポートに明記)。
- マージンタイム: 経過秒 = 現在 t_sec - 試合開始時刻 (この試合境界内で
  最初に有効 score が観測された t_sec、両 side 共有)。
  from_first_move=False (score_to_ojama の既定と同じ、96秒起点)。

## 既知の非対応 (正直な記録)
- MENU 遷移による試合境界検出 (npz に BoardState が無いため不可能)。
- 連鎖中の「本当の settle」判定 (K_SETTLE_FRAMES 20フレーム連続不変) は
  再現不可。STABLE snapshot 間の得点差分をそのまま chain_total とみなす
  近似 (dedup snapshot の間には通常ちょうど1回の着地しかないという前提に依存)。
- 全消しボーナスの持ち越し (次連鎖時に加算) は本近似では再現しない
  (score 差分に全消しボーナスが混入していれば chain_total に自動的に含まれる
  ため実害は限定的、ただし持ち越しタイミングのズレは検証されていない)。

## 使い方
    PYTHONPATH=. ./venv/bin/python -m scripts._ojama_restore_prototype_2026-08-12 \\
        --npz-dir data/indicators_v2/boards_lean_regen_2026-07-31 \\
        --truth-csv data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv \\
        --out-dir data/verify/ojama_restore_proto_2026-08-12
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ojama_accounting import (  # noqa: E402
    CHAIN_TOTAL_MIN_SCORE,
    SCORE_RESET_THRESHOLD,
    THEORY_DROP_PER_TURN,
    ON_FIELD_CAP,
    cancel_own_pending_then_send_surplus,
)
from src.scoring import OJAMA_RATE_STANDARD, score_to_ojama  # noqa: E402

# ============================
# 定数
# ============================

# 検証窓 (combined66 c-series 収集時の3窓定義、scripts/_gen_jobs_labeled_win_c20_2026-07-26.sh
# と完全一致させる。ground truth 側がこの窓ごとに OjamaAccountingTracker を
# 独立起動している(=各窓冒頭でリセット)ため、復元側もここでコールドスタートする
# 必要がある(合わせないと前後の履行が汚染し比較不能になる、2026-08-12 実測で確認)。
VALIDATION_WINDOWS: tuple[tuple[float, float, str], ...] = (
    (0.0, 300.0, "base"),
    (300.0, 900.0, "gap"),
    (1200.0, 1560.0, "mid"),
)

# 得点欠損 (score OCR 失敗) の npz 上のセンチネル値
SCORE_NONE_SENTINEL: int = -1

# ネクスト未検出/未取得 (--with-next 未指定収集) の npz 上のセンチネル値
# (scripts/collect_boards_lean.py NEXT_COLOR_UNKNOWN と同じ)
NEXT_COLOR_UNKNOWN: int = -1

# 近傍マッチ許容誤差 [秒] (2つの独立収集run間のフレームサンプリングずれを吸収)
MATCH_TOLERANCE_SEC: float = 1.5

# tsumo_count 列の未取得センチネル値 (scripts/collect_boards_lean.py
# TSUMO_COUNT_UNKNOWN と同一の契約、v3 追加、2026-08-12)。
TSUMO_COUNT_UNKNOWN: int = -1


# ============================
# 復元シミュレータ (stateless イベント → 状態遷移)
# ============================

@dataclass
class _SimSideState:
    """1 side分の近似会計状態 (score_to_ojama / cancel_* の入出力を保持)。"""
    leftover: int = 0
    forecast: int = 0
    prev_score: int | None = None


@dataclass
class _RestoreDiagnostics:
    """異常検知カウンタ (レポート用)。"""
    negative_delta_count: int = 0
    reset_count: int = 0
    chain_count: int = 0
    plain_settle_count: int = 0


def reconstruct_ojama_sequence(
    events: list[tuple[float, str, int | None, tuple[int, int] | None]],
) -> tuple[list[dict], _RestoreDiagnostics]:
    """(t_sec, side, score, next_pair) の時系列から ojama収支2列を近似復元する。

    2026-08-12 改良: 「dedup済み STABLE snapshot は必ず1回のtsumo着地に対応する」
    という当初仮説は誤りと実測で判明 (c11 実測 t=277.8-282.1、next_pair不変のまま
    grid だけ変化する snapshot が9行連続、production の tsumo_count は1回も
    増分していないはずの区間で forecast が誤って毎回drainされてしまう)。
    このため「次ネクストペアが前回と異なる = 実際に1回着地した」を着地イベントの
    ゲートとして使う (--with-next 収集時のみ有効、次ネクスト未検出/未取得
    ((-1,-1)) の行は直前の既知ペアを保持したまま判定をスキップする)。

    Args:
        events: (t_sec, side, score, next_pair) のリスト。side は "1P"/"2P"。
            score / next_pair は None (認識欠損) を許容する。**呼び出し前に
            t_sec 昇順ソート済みであること**。
        既知の非対応: next_pair が全区間 None (--with-next 未収集) の場合は
            全行を「着地イベント無し」として扱い drain/chain とも発生しない
            (安全側だが forecast が0のまま停滞し精度が大きく劣化する。
            --with-next 必須指標であることの根拠)。

    Returns:
        (復元済み行のリスト, 診断カウンタ)。各行は
        {t_sec, side, pred_ojama_net_balance_raw, pred_ojama_forecast_raw}。
    """
    sides: dict[str, _SimSideState] = {"1P": _SimSideState(), "2P": _SimSideState()}
    last_next: dict[str, tuple[int, int] | None] = {"1P": None, "2P": None}
    match_start_sec: float | None = None
    diag = _RestoreDiagnostics()
    out: list[dict] = []

    for t_sec, side, score, next_pair in events:
        other_label = "2P" if side == "1P" else "1P"
        s = sides[side]
        o = sides[other_label]
        if score is not None and match_start_sec is None:
            match_start_sec = t_sec
        is_placement = (
            next_pair is not None
            and last_next[side] is not None
            and next_pair != last_next[side]
        )
        if next_pair is not None:
            last_next[side] = next_pair
        if score is not None and s.prev_score is not None:
            if s.prev_score - score >= SCORE_RESET_THRESHOLD:
                # 試合境界 (このsideのscoreが大幅減少): このsideのみリセット
                # (OjamaAccountingTracker._reset_side_boundary と同じ side 独立仕様)。
                s.leftover = 0
                s.forecast = 0
                match_start_sec = t_sec
                diag.reset_count += 1
            elif is_placement:
                delta = score - s.prev_score
                if delta < 0:
                    # OCR誤読疑い: drainのみ行いchain計算はスキップ
                    _drain(s)
                    diag.negative_delta_count += 1
                elif delta >= CHAIN_TOTAL_MIN_SCORE:
                    _drain(s)
                    elapsed = max(0.0, t_sec - (match_start_sec or t_sec))
                    res = score_to_ojama(
                        score=delta, prev_leftover=s.leftover,
                        elapsed_sec=elapsed, rate_base=OJAMA_RATE_STANDARD,
                    )
                    s.leftover = res.leftover_score
                    gen = res.ojama_count
                    s.forecast, o.forecast = cancel_own_pending_then_send_surplus(
                        gen, s.forecast, o.forecast,
                    )
                    diag.chain_count += 1
                else:
                    # 連鎖なしの着地 (delta 0〜CHAIN_TOTAL_MIN_SCORE未満)
                    _drain(s)
                    diag.plain_settle_count += 1
            # is_placement=False (次ネクスト不変) の行は「見かけ上grid変化だが
            # 実着地なし」として何もしない (drainもchainも発生させない)。
        if score is not None:
            s.prev_score = score
        own_capped = min(s.forecast, ON_FIELD_CAP)
        other_capped = min(o.forecast, ON_FIELD_CAP)
        out.append({
            "t_sec": t_sec, "side": side,
            "pred_ojama_net_balance_raw": float(other_capped - own_capped),
            "pred_ojama_forecast_raw": float(max(0, s.forecast)),
        })
    return out, diag


def _drain(s: _SimSideState) -> None:
    """1回の着地による pending drain (on_tsumo_settled 相当)。"""
    drain = min(THEORY_DROP_PER_TURN, s.forecast)
    s.forecast -= drain


# ============================
# v3: tsumo_count 増分ゲート版 (2026-08-12 追加)
#
# v2 (reconstruct_ojama_sequence、次ネクストペア変化ゲート) は65動画検証で
# pooled相関0.33〜0.38 と不合格だった。根本原因は「dedup済み STABLE snapshot
# は1着地に対応しない」(video_c11 t=277.8-282.1秒の9行で next1不変・grid変化
# を実測) であり、次ネクスト自体も NextDetector 非依存でないため取り漏れが
# ある。RecognitionPipeline.tsumo_count(side) は TSUMO_FALL→STABLE 物理遷移
# でのみ増分する stateless getter (NextDetector 非依存、--with-next 不要) の
# ため、より確実な着地イベントの代理指標になる。
# v1/v2 (reconstruct_ojama_sequence 本体) は変更しない (最小差分方針、
# 2026-08-12)。実データでの v3 精度検証は tsumo_count 列入り npz が
# 1本できてから行う (本追加はユニットテストレベルの動作確認まで)。
# ============================

def _apply_landing_delta(
    s: _SimSideState,
    o: _SimSideState,
    delta: int,
    elapsed_sec: float,
    diag: _RestoreDiagnostics,
) -> None:
    """1回の着地イベントによる drain→(必要なら)chain 会計を適用する (v3専用)。

    reconstruct_ojama_sequence (v2) 内のインライン処理と等価な会計ロジック
    だが、v2 側は既存動作の無変更を優先し本ヘルパへの切替は行わない
    (2026-08-12、最小差分方針)。reconstruct_ojama_sequence_tsumo_count_gate
    (v3) 専用のヘルパ。
    """
    _drain(s)
    if delta < 0:
        diag.negative_delta_count += 1
        return
    if delta < CHAIN_TOTAL_MIN_SCORE:
        diag.plain_settle_count += 1
        return
    res = score_to_ojama(
        score=delta, prev_leftover=s.leftover,
        elapsed_sec=elapsed_sec, rate_base=OJAMA_RATE_STANDARD,
    )
    s.leftover = res.leftover_score
    s.forecast, o.forecast = cancel_own_pending_then_send_surplus(
        res.ojama_count, s.forecast, o.forecast,
    )
    diag.chain_count += 1


def _is_landing_by_tsumo_count(
    tsumo_count: int | None,
    last_tsumo_count: int | None,
) -> bool:
    """tsumo_count の増分を着地イベントとして判定する (v3 ゲート、2026-08-12)。

    Args:
        tsumo_count: この snapshot の tsumo_count(side) 値。None または
            TSUMO_COUNT_UNKNOWN (-1) は未取得として着地でないと判定する。
        last_tsumo_count: 直前に観測済みの既知 tsumo_count 値。None は
            まだ一度も観測していない (この side での初回行は着地扱いしない
            = 試合開始直後の初期配置を誤って着地と数えない)。

    Returns:
        bool: 着地イベントと判定するか。
    """
    if tsumo_count is None or tsumo_count == TSUMO_COUNT_UNKNOWN:
        return False
    if last_tsumo_count is None:
        return False
    return tsumo_count > last_tsumo_count


def _make_output_row(
    t_sec: float, side: str, s: "_SimSideState", o: "_SimSideState",
) -> dict:
    """1 side の現在状態から出力行 (キャップ済み収支2列) を作る (v3専用)。"""
    own_capped = min(s.forecast, ON_FIELD_CAP)
    other_capped = min(o.forecast, ON_FIELD_CAP)
    return {
        "t_sec": t_sec, "side": side,
        "pred_ojama_net_balance_raw": float(other_capped - own_capped),
        "pred_ojama_forecast_raw": float(max(0, s.forecast)),
    }


def reconstruct_ojama_sequence_tsumo_count_gate(
    events: list[tuple[float, str, int | None, int | None]],
) -> tuple[list[dict], _RestoreDiagnostics]:
    """(t_sec, side, score, tsumo_count) からおじゃま収支2列を近似復元する (v3)。

    v2 (reconstruct_ojama_sequence、次ネクストペア変化ゲート) の代替版。
    dedup済み STABLE snapshot が同一着地を指して複数回連続しても、
    tsumo_count が不変なら着地と判定しない (v2 の弱点だった next_pair 欠損
    行での判定不能が原理的に発生しない: tsumo_count は NextDetector 非依存
    で --with-next 無指定でも常時取得できる)。

    Args:
        events: (t_sec, side, score, tsumo_count) のリスト。**呼び出し前に
            t_sec 昇順ソート済みであること**。score/tsumo_count は None
            (認識欠損/未取得) を許容する。

    Returns:
        (復元済み行のリスト, 診断カウンタ)。行スキーマは reconstruct_ojama_
        sequence と同一 ({t_sec, side, pred_ojama_net_balance_raw,
        pred_ojama_forecast_raw})。
    """
    sides: dict[str, _SimSideState] = {"1P": _SimSideState(), "2P": _SimSideState()}
    last_tsumo_count: dict[str, int | None] = {"1P": None, "2P": None}
    match_start_sec: float | None = None
    diag = _RestoreDiagnostics()
    out: list[dict] = []

    for t_sec, side, score, tsumo_count in events:
        other_label = "2P" if side == "1P" else "1P"
        s, o = sides[side], sides[other_label]
        if score is not None and match_start_sec is None:
            match_start_sec = t_sec
        is_placement = _is_landing_by_tsumo_count(tsumo_count, last_tsumo_count[side])
        if tsumo_count is not None and tsumo_count != TSUMO_COUNT_UNKNOWN:
            last_tsumo_count[side] = tsumo_count
        if score is not None and s.prev_score is not None:
            if s.prev_score - score >= SCORE_RESET_THRESHOLD:
                # 試合境界 (このsideのscoreが大幅減少): このsideのみリセット。
                s.leftover, s.forecast = 0, 0
                match_start_sec = t_sec
                diag.reset_count += 1
            elif is_placement:
                # tsumo_count 不変の行 (見かけ上grid変化だが実着地なし) は
                # drain/chain とも発生させない (v2 の巻き込みバグ根治点)。
                elapsed = max(0.0, t_sec - (match_start_sec or t_sec))
                _apply_landing_delta(s, o, score - s.prev_score, elapsed, diag)
        if score is not None:
            s.prev_score = score
        out.append(_make_output_row(t_sec, side, s, o))
    return out, diag


# ============================
# npz 読み込み
# ============================

def load_npz_events(
    npz_path: Path,
) -> dict[str, list[tuple[float, str, int | None, tuple[int, int] | None]]]:
    """npz から video_id 別に (t_sec, side, score, next_pair) イベント列を作る (t_sec昇順)。

    next1_a/next1_b が npz に存在しない (--with-next 未収集) 場合は全行
    next_pair=None として扱う (呼び出し側の diag で着地イベント検出0件が
    分かるようにする、無言で劣化させない)。
    """
    d = np.load(str(npz_path), allow_pickle=True)
    video_ids = d["video_id"]
    t_secs = d["t_sec"]
    sides = d["side"]
    scores = d["score"]
    has_next = "next1_a" in d and "next1_b" in d
    next_as = d["next1_a"] if has_next else None
    next_bs = d["next1_b"] if has_next else None
    by_video: dict[str, list[tuple[float, str, int | None, tuple[int, int] | None]]] = {}
    for i in range(len(video_ids)):
        vid = str(video_ids[i])
        sc = int(scores[i])
        next_pair: tuple[int, int] | None = None
        if has_next:
            na, nb = int(next_as[i]), int(next_bs[i])
            if na != NEXT_COLOR_UNKNOWN and nb != NEXT_COLOR_UNKNOWN:
                next_pair = (na, nb)
        by_video.setdefault(vid, []).append((
            float(t_secs[i]), str(sides[i]),
            None if sc == SCORE_NONE_SENTINEL else sc,
            next_pair,
        ))
    for vid in by_video:
        by_video[vid].sort(key=lambda r: r[0])
    return by_video


def load_npz_events_tsumo_count_gate(
    npz_path: Path,
) -> dict[str, list[tuple[float, str, int | None, int | None]]]:
    """npz から video_id 別に (t_sec, side, score, tsumo_count) イベント列を作る (v3用)。

    tsumo_count 列が npz に存在しない (2026-08-12 以前の旧世代収集) 場合は
    全行 tsumo_count=None として扱う (load_npz_events の next_pair 欠損時と
    同じ設計: 呼び出し側で着地イベント検出0件になることで無言劣化を避ける)。
    """
    d = np.load(str(npz_path), allow_pickle=True)
    video_ids = d["video_id"]
    t_secs = d["t_sec"]
    sides = d["side"]
    scores = d["score"]
    has_tsumo_count = "tsumo_count" in d
    tsumo_counts = d["tsumo_count"] if has_tsumo_count else None
    by_video: dict[str, list[tuple[float, str, int | None, int | None]]] = {}
    for i in range(len(video_ids)):
        vid = str(video_ids[i])
        sc = int(scores[i])
        tc: int | None = None
        if has_tsumo_count:
            tc_raw = int(tsumo_counts[i])
            tc = None if tc_raw == TSUMO_COUNT_UNKNOWN else tc_raw
        by_video.setdefault(vid, []).append((
            float(t_secs[i]), str(sides[i]),
            None if sc == SCORE_NONE_SENTINEL else sc,
            tc,
        ))
    for vid in by_video:
        by_video[vid].sort(key=lambda r: r[0])
    return by_video


# ============================
# 検証 (窓ごとコールドスタート + 近傍マッチ + 指標算出)
# ============================

def restore_with_validation_windows(
    events: list[tuple[float, str, int | None]],
) -> pd.DataFrame:
    """VALIDATION_WINDOWS ごとに独立コールドスタートで復元し、1本のDataFrameにまとめる。

    ground truth (combined66) が3窓を独立プロセスで収集した (=各窓冒頭で
    OjamaAccountingTracker.reset() 相当) ことに合わせるため、復元側も同じ
    窓境界でリセットする (2026-08-12 実測: 揃えないと前後の履行が汚染して
    比較不能になることを確認済み)。
    """
    frames: list[pd.DataFrame] = []
    for start, end, label in VALIDATION_WINDOWS:
        window_events = [e for e in events if start <= e[0] < end]
        if not window_events:
            continue
        rows, diag = reconstruct_ojama_sequence(window_events)
        df = pd.DataFrame(rows)
        df["window"] = label
        df.attrs[f"diag_{label}"] = diag
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["t_sec", "side", "pred_ojama_net_balance_raw",
                                      "pred_ojama_forecast_raw", "window"])
    return pd.concat(frames, ignore_index=True)


def match_to_truth(
    pred_df: pd.DataFrame, truth_df: pd.DataFrame, video_id: str,
) -> pd.DataFrame:
    """side別に最近傍 t_sec マッチングして予測値と真値を対応付ける。"""
    matched: list[dict] = []
    for side in ("1P", "2P"):
        p = pred_df[pred_df["side"] == side].sort_values("t_sec")
        t = truth_df[truth_df["side"] == side].sort_values("t_sec")
        if p.empty or t.empty:
            continue
        p_t = p["t_sec"].to_numpy()
        for _, trow in t.iterrows():
            idx = np.searchsorted(p_t, trow["t_sec"])
            candidates = [i for i in (idx - 1, idx) if 0 <= i < len(p_t)]
            if not candidates:
                continue
            best = min(candidates, key=lambda i: abs(p_t[i] - trow["t_sec"]))
            if abs(p_t[best] - trow["t_sec"]) > MATCH_TOLERANCE_SEC:
                continue
            prow = p.iloc[best]
            matched.append({
                "video_id": video_id, "side": side, "t_sec": trow["t_sec"],
                "gt_net": trow["ojama_net_balance_raw"],
                "pred_net": prow["pred_ojama_net_balance_raw"],
                "gt_forecast": trow["ojama_forecast_raw"],
                "pred_forecast": prow["pred_ojama_forecast_raw"],
            })
    return pd.DataFrame(matched)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: pd.Series, b: pd.Series) -> float:
    return _corr(a.rank().to_numpy(), b.rank().to_numpy())


def summarize(pairs: pd.DataFrame) -> dict:
    """1動画分の対応ペアから精度指標をまとめる。"""
    if pairs.empty:
        return {"n": 0}
    net_sign_agree = float(np.mean(
        np.sign(pairs["gt_net"]) == np.sign(pairs["pred_net"])
    ))
    return {
        "n": len(pairs),
        "net_pearson": _corr(pairs["gt_net"].to_numpy(), pairs["pred_net"].to_numpy()),
        "net_spearman": _spearman(pairs["gt_net"], pairs["pred_net"]),
        "net_mae": float(np.mean(np.abs(pairs["gt_net"] - pairs["pred_net"]))),
        "net_sign_agree": net_sign_agree,
        "forecast_pearson": _corr(
            pairs["gt_forecast"].to_numpy(), pairs["pred_forecast"].to_numpy(),
        ),
        "forecast_spearman": _spearman(pairs["gt_forecast"], pairs["pred_forecast"]),
        "forecast_mae": float(np.mean(np.abs(pairs["gt_forecast"] - pairs["pred_forecast"]))),
        "both_zero_rate": float(np.mean(
            (pairs["gt_forecast"] == 0) & (pairs["pred_forecast"] == 0)
        )),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--truth-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--videos", nargs="*", default=None,
                     help="対象 video_id (例 video_c10)。省略時は npz-dir 内で truth と重複する全動画")
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(a.truth_csv, usecols=[
        "video_id", "t_sec", "side", "ojama_net_balance_raw", "ojama_forecast_raw",
    ])

    npz_files = {p.stem: p for p in a.npz_dir.glob("*.npz")}
    target_videos = a.videos or sorted(
        vid for vid in truth["video_id"].unique()
        if vid.replace("video_", "") in npz_files
    )

    all_pairs: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for vid in target_videos:
        npz_key = vid.replace("video_", "")
        npz_path = npz_files.get(npz_key)
        if npz_path is None:
            print(f"[skip] {vid}: npz not found")
            continue
        by_video = load_npz_events(npz_path)
        events = by_video.get(vid)
        if events is None:
            print(f"[skip] {vid}: npz内に video_id 不一致")
            continue
        pred_df = restore_with_validation_windows(events)
        truth_sub = truth[
            (truth["video_id"] == vid)
            & (truth["t_sec"] < VALIDATION_WINDOWS[-1][1])
        ]
        pairs = match_to_truth(pred_df, truth_sub, vid)
        if pairs.empty:
            print(f"[warn] {vid}: マッチ0件")
            continue
        all_pairs.append(pairs)
        stats = summarize(pairs)
        stats["video_id"] = vid
        summary_rows.append(stats)
        print(f"[{vid}] n={stats['n']} net_pearson={stats['net_pearson']:.3f} "
              f"net_mae={stats['net_mae']:.2f} forecast_pearson={stats['forecast_pearson']:.3f} "
              f"forecast_mae={stats['forecast_mae']:.2f}")

    if all_pairs:
        pooled = pd.concat(all_pairs, ignore_index=True)
        pooled.to_csv(a.out_dir / "ojama_restore_pairs.csv", index=False)
        pooled_stats = summarize(pooled)
        print("\n=== プール全体 ===")
        for k, v in pooled_stats.items():
            print(f"  {k}: {v}")
        pd.DataFrame(summary_rows).to_csv(a.out_dir / "ojama_restore_per_video.csv", index=False)
    else:
        print("[FATAL] マッチ0件、検証不能")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

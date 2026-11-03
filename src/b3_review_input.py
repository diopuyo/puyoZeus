"""B3 レビュー動画の入力形式 (docs/B3_REVIEW_INPUT.md)。

学習・評価側の出力 (b3_rows の列名 + 予測値) を、レビュー表示用の平坦な時系列
(1 行 = 1 評価時点、CSV) へ変換する口と、CSV の読み書き・検査を持つ。
描画やモデルには依存しない (stateless)。
"""
from __future__ import annotations

import csv
import math
from bisect import bisect_right
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

SCHEMA_VERSION = "b3_review_v1"
SIDES = ("p1", "p2")
# 欠測理由コード (docs/B3_COLUMN_SPEC.md の Mask と同一。表示用名称のみ持つ)。
MISSING_NAMES = {
    0: "OK", 1: "旧版に無い", 2: "観測不能", 3: "未到達", 4: "境界不明", 5: "片側未観測",
    6: "予測不能", 7: "探索打切り", 8: "該当なし", 9: "未実装(先送り)", 10: "認識矛盾", 11: "得点履歴不完全",
}
STATES = ("static", "exchange", "none")
DEATH_STATES = ("none", "suspect", "proven_dead")

# 共通列 (順序固定)。確率は P(1P 勝ち) in [0,1]。
COMMON_STR = ("video_id", "mode", "state_reason", "missing_note")
PROB_FIELDS = ("p_static", "p_exchange", "p_selected", "p_display", "p_ref", "p_ref_display")
COMMON_FIELDS = ("t_sec", "frame_idx", "video_id", "game_idx", "phase", *PROB_FIELDS, "mode",
                 "state_reason", "missing_code", "missing_note", "sl_rate", "won_p1")
SIDE_STR = ("death_state", "death_basis")
SIDE_FLOAT = ("H_med", "H_fast", "Q_med", "Q_fast", "I_sl", "I_pred", "I_icon", "M",
              "recv", "send", "a_rem", "b_busy", "window", "req_h", "done_h", "trunc",
              "hand_med", "hand_fast", "active", "sl_pending", "sl_generated", "sl_offset",
              "sl_dropped", "sl_carry", "icon_corr", "icon_raw", "mk_ta", "mk_sl")
INPUT_FIELDS = (*COMMON_FIELDS,
                *(f"{s}_{n}" for s in SIDES for n in (*SIDE_FLOAT, *SIDE_STR)))
# 出力 (フレーム別 CSV) は入力 + 計算列。
DERIVED_FIELDS = ("frame_out", "p_diff_vs_ref", "p1_ledger_icon_gap", "p2_ledger_icon_gap",
                  "p1_missing_name", "p2_missing_name", "missing_name")
CSV_FIELDS = (*INPUT_FIELDS, *DERIVED_FIELDS)

# b3_rows 列名 (語幹; 実列は <語幹>_p1 / _p2) → 入力 (p1_*) の対応。
ROWS_TO_SIDE = {
    "ta_H_med": "H_med", "ta_H_fast": "H_fast", "ta_q_med": "Q_med", "ta_q_fast": "Q_fast",
    "ta_i_sl": "I_sl", "ta_i_pred": "I_pred", "ta_i_icon": "I_icon", "ta_m_sl_med": "M",
    "ta_recv_pred_med": "recv", "ta_send_pred_med": "send", "ta_a_rem_sec": "a_rem",
    "ta_b_busy": "b_busy", "ta_window": "window", "ta_req_h": "req_h", "ta_done_h": "done_h",
    "ta_trunc": "trunc", "ta_hand_med_sec": "hand_med", "ta_hand_fast_sec": "hand_fast",
    "ta_active": "active", "sl_pending": "sl_pending", "sl_generated": "sl_generated",
    "sl_offset": "sl_offset", "sl_dropped": "sl_dropped", "sl_carry": "sl_carry",
    "icon_corr_total": "icon_corr", "icon_raw_total": "icon_raw", "mk_ta": "mk_ta",
    "mk_sl": "mk_sl",
}
ROWS_TO_COMMON = ("t_sec", "frame_idx", "video_id", "game_idx", "phase", "sl_rate", "won_p1")
# 評価側が渡す予測値 dict の共通キー。側別は p1_death_state 等。
PRED_COMMON = (*PROB_FIELDS, "mode", "state_reason", "missing_code", "missing_note")


def _num(value: Any) -> Optional[float]:
    """空・NaN・非数は None (表示 '--')。0 とは区別する。"""
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def empty_row() -> dict[str, Any]:
    """全列 None の入力行 (欠測の初期値)。"""
    return {name: None for name in INPUT_FIELDS}


def from_b3_rows(row: Mapping[str, Any], pred: Mapping[str, Any]) -> dict[str, Any]:
    """b3_rows の 1 行 (列名は B3_COLUMN_SPEC) + 予測値 dict → レビュー入力 1 行。

    列に無い値は None のまま (捏造しない)。M は台帳 v2 版 (ta_m_sl_med) を優先し、
    無ければ予測完走版 (ta_m_pred_med) へ落とす (どちらも無ければ None)。
    """
    out = empty_row()
    for key in ROWS_TO_COMMON:
        out[key] = row.get(key)
    for side in SIDES:
        for src, dst in ROWS_TO_SIDE.items():
            out[f"{side}_{dst}"] = row.get(f"{src}_{side}")
        if out[f"{side}_M"] is None:
            out[f"{side}_M"] = row.get(f"ta_m_pred_med_{side}")
        out[f"{side}_death_state"] = pred.get(f"{side}_death_state", "none")
        out[f"{side}_death_basis"] = pred.get(f"{side}_death_basis")
    for key in PRED_COMMON:
        out[key] = pred.get(key)
    return out


def from_b3_rows_iter(rows: Iterable[Mapping[str, Any]],
                      preds: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """行と予測を同数で対応させる。長さ不一致は黙って切らず例外にする。"""
    rows, preds = list(rows), list(preds)
    if len(rows) != len(preds):
        raise ValueError(f"行数不一致: rows={len(rows)} preds={len(preds)}")
    return [from_b3_rows(r, p) for r, p in zip(rows, preds)]


def _coerce(name: str, value: Any) -> Any:
    """CSV 文字列を型へ。str 列は空文字を None に。"""
    if name in COMMON_STR or name.endswith(SIDE_STR):
        return None if value in (None, "") else str(value)
    return _num(value)


def validate_rows(rows: list[dict[str, Any]]) -> list[str]:
    """入力の健全性を検査し、問題の一覧を返す (空なら合格)。

    t_sec の存在と昇順・確率の範囲 [0,1]・状態語彙・死亡語彙を見る。
    """
    problems: list[str] = []
    prev = -math.inf
    for i, row in enumerate(rows):
        t = _num(row.get("t_sec"))
        if t is None:
            problems.append(f"行{i}: t_sec 欠落")
            continue
        if t < prev:
            problems.append(f"行{i}: t_sec が降順 ({t} < {prev})")
        prev = t
        for key in PROB_FIELDS:
            p = _num(row.get(key))
            if p is not None and not 0.0 <= p <= 1.0:
                problems.append(f"行{i}: {key}={p} が [0,1] 外")
        if row.get("mode") not in (None, *STATES):
            problems.append(f"行{i}: mode 語彙外 {row.get('mode')!r}")
        for side in SIDES:
            ds = row.get(f"{side}_death_state")
            if ds not in (None, *DEATH_STATES):
                problems.append(f"行{i}: {side}_death_state 語彙外 {ds!r}")
    return problems


def load_review_input(path: Path) -> list[dict[str, Any]]:
    """入力 CSV を読む。未知の列は無視、足りない列は None。検査に落ちれば例外。"""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = [{n: _coerce(n, r.get(n)) for n in INPUT_FIELDS} for r in csv.DictReader(f)]
    problems = validate_rows(rows)
    if problems:
        raise ValueError("レビュー入力が不正: " + "; ".join(problems[:10]))
    return rows


def save_review_input(path: Path, rows: list[dict[str, Any]]) -> None:
    """入力 CSV を保存 (None は空欄)。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INPUT_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({n: ("" if row.get(n) is None else row[n]) for n in INPUT_FIELDS})


def missing_name(code: Any) -> str:
    """欠測コード → 日本語名。語彙外は '不明(code)'、None は '--'。"""
    c = _num(code)
    return "--" if c is None else MISSING_NAMES.get(int(c), f"不明({int(c)})")


def ledger_icon_gap(row: Mapping[str, Any], side: str) -> Optional[float]:
    """台帳 v2 の予告 − 画面アイコン補正値 (どちらか欠測、または両方 0 なら None)。"""
    sl, ic = _num(row.get(f"{side}_sl_pending")), _num(row.get(f"{side}_icon_corr"))
    if sl is None or ic is None or (sl == 0 and ic == 0):
        return None
    return sl - ic


def derive(row: Mapping[str, Any], frame_out: int) -> dict[str, Any]:
    """入力行 + 計算列 (出力 CSV 1 行)。画面と CSV の値を同じ関数から作る。"""
    out = {n: row.get(n) for n in INPUT_FIELDS}
    ps, pr = _num(row.get("p_display")), _num(row.get("p_ref_display"))
    out.update(frame_out=frame_out, p_diff_vs_ref=None if ps is None or pr is None else ps - pr,
               p1_ledger_icon_gap=ledger_icon_gap(row, "p1"),
               p2_ledger_icon_gap=ledger_icon_gap(row, "p2"),
               p1_missing_name=missing_name(row.get("p1_mk_ta")),
               p2_missing_name=missing_name(row.get("p2_mk_ta")),
               missing_name=missing_name(row.get("missing_code")))
    return out


def row_at(rows: list[dict[str, Any]], times: list[float], t: float) -> Optional[dict[str, Any]]:
    """時刻 t 以前で最後の行 (二分探索)。先頭より前は None (表示しない)。"""
    i = bisect_right(times, t) - 1
    return rows[i] if i >= 0 else None

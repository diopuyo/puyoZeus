"""(甲)修正 enable_placement_color_cnn_check 効果ゼロの原因切り分け診断 (2026-07-25)。

背景: commit c14309a (案(iii)) を監査器 (scripts/physics_violation_audit.py) で
A/B 比較したところ、OFF/ON で全592違反・全類型・全動画が完全に同数だった。
read-only (src 非改変)、monkeypatch 計装のみで下記2仮説を切り分ける。

H1: distrust フラグが一度も発火していない
    (`_flag_landing_distrust_cells` 呼び出し時点の cnn_board が着地セルで
    UNKNOWN/EMPTY ばかり → フラグ条件 cnn_v in _VALID_PUYO_COLORS が
    成立しない)。
H2: 発火しても監査指標に効かない構造
    (案(iii) は P7 の判定のみを変えるが、実際の色フリッカは P7 より前
    (P2 誤書き→P5 訂正、~8フレーム) で完結しているため P7 側をどう
    直しても件数は変わらない)。

計装対象 (src/recognition_pipeline.py):
    1. モジュール関数 `_flag_landing_distrust_cells` をラップし、
       呼び出し回数・非空フラグ回数・候補セル (pv=EMPTY/UNKNOWN かつ
       inferred=有色) の cnn_v 内訳 (有効色一致/有効色不一致/UNKNOWN/EMPTY/
       おじゃま/その他) を計数する。
    2. `RecognitionPipeline._update_landing_votes` をラップし、
       finalization (elapsed>=LANDING_VOTE_FRAMES) 時点で distrust セルが
       実際に「NEXT色バイアス経路を迂回し生CNN多数決フォールバックへ
       分岐した」回数 (=発火回数) と、その分岐が無ければ選ばれていたはずの
       NEXT色バイアス勝者 (counterfactual) と実際の確定値が異なった回数
       (=P7判定が変わった回数) を計数する。

Usage:
    PYTHONPATH=. python -m scripts._tmp_diag_c14309a_effect_2026-07-25 --smoke
    PYTHONPATH=. python -m scripts._tmp_diag_c14309a_effect_2026-07-25 \
        --video 30 --start-sec 225.0 --max-sec 90.0
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import cv2

# スレッド制限 (熱暴走防止、feedback_thermal_safety_mandatory 準拠)。並列しない。
for _env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_env_key, "3")

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

import src.recognition_pipeline as rp  # noqa: E402
from src.board import BOARD_COLS, BOARD_ROWS, COLOR_EMPTY, COLOR_OJAMA, COLOR_UNKNOWN  # noqa: E402
from src.placement_inferrer import _VALID_PUYO_COLORS  # noqa: E402
from scripts.recognition_physics_review import _capture_frames  # noqa: E402

# ============================
# 定数
# ============================
OUTPUT_DIR: Path = PROJ_ROOT / "data" / "verify" / "c14309a_effect_diag"

# スモーク窓: video_30 の実測済み安全区間 (physics_violation_audit.py と同一値)。
SMOKE_VIDEO_STEM: str = "30"
SMOKE_START_SEC: float = 225.0
SMOKE_MAX_SEC: float = 21.0

# user指定の本窓 (video_30、225-315秒、90秒級)。
DEFAULT_VIDEO_STEM: str = "30"
DEFAULT_START_SEC: float = 225.0
DEFAULT_MAX_SEC: float = 90.0


@dataclass
class _Counters:
    """診断計数 (単一プロセス・同期実行前提、共有可変状態)。"""

    # --- _flag_landing_distrust_cells 計装 ---
    n_flag_calls: int = 0
    n_flag_nonempty: int = 0
    n_flag_total_cells: int = 0
    n_candidate_cells: int = 0
    cnn_v_valid_match: int = 0      # 有効色 かつ inferred と一致 (=フラグされない)
    cnn_v_valid_mismatch: int = 0   # 有効色 かつ inferred と不一致 (=フラグ対象)
    cnn_v_empty: int = 0
    cnn_v_unknown: int = 0
    cnn_v_ojama: int = 0
    cnn_v_other: int = 0

    # --- _update_landing_votes 計装 (finalization 時点) ---
    n_distrust_branch_fired: int = 0        # distrust セルが fallback へ迂回した回数
    n_would_use_next_bias: int = 0          # 迂回しなければ NEXT バイアスが勝っていたはずの回数
    n_p7_judgment_changed: int = 0          # 上記のうち実際に確定値が変わった回数
    n_p7_judgment_same: int = 0             # 上記のうち確定値が結果的に同じだった回数
    n_no_diff_possible: int = 0             # NEXT バイアス自体が不成立 (どのみち fallback) だった回数

    # --- early-confirm 迂回の参考計数 (finalization 前, 蓄積期間中) ---
    n_early_bypass_events: int = 0          # distrust によって早期確定 gate を素通りした回数 (frame単位)

    # --- 追加計装: _start_landing_vote 自体が entry を生成しているか ---
    # (タイミングずれ仮説: NEXT検知フレーム(landing_pending起動) が
    # TSUMO_FALL→STABLE 遷移フレームと一致しないと prev_confirmed==final_board
    # となり cells_with_expected が常に空になる、という別経路の疑いを検証する)。
    n_start_vote_calls: int = 0
    n_start_vote_empty_diff: int = 0        # cells_with_expected が空 (entry生成なし)
    n_start_vote_nonempty_diff: int = 0     # entry生成あり (cells_with_expected非空)
    n_start_vote_distrust_present: int = 0  # distrust_cells非空で呼ばれた回数

    detail_rows: list = field(default_factory=list)


def _make_flag_wrapper(orig, counters: _Counters):
    """`_flag_landing_distrust_cells` (モジュール関数) をラップする。"""

    @functools.wraps(orig)
    def wrapped(inferred, prev_confirmed, cnn_board):
        result = orig(inferred, prev_confirmed, cnn_board)
        counters.n_flag_calls += 1
        if result:
            counters.n_flag_nonempty += 1
            counters.n_flag_total_cells += len(result)
        # 候補セル (= 関数内部の一次フィルタ条件を満たすセル) の cnn_v 内訳を
        # 独立に再計算する (src の実装は変更しない、観測のみ)。
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                pv = int(prev_confirmed.get(r, c))
                iv = int(inferred.get(r, c))
                if not (
                    pv in (COLOR_EMPTY, COLOR_UNKNOWN)
                    and iv not in (COLOR_EMPTY, COLOR_UNKNOWN, COLOR_OJAMA)
                ):
                    continue
                counters.n_candidate_cells += 1
                cnn_v = int(cnn_board.get(r, c))
                if cnn_v in _VALID_PUYO_COLORS:
                    if cnn_v != iv:
                        counters.cnn_v_valid_mismatch += 1
                    else:
                        counters.cnn_v_valid_match += 1
                elif cnn_v == COLOR_EMPTY:
                    counters.cnn_v_empty += 1
                elif cnn_v == COLOR_UNKNOWN:
                    counters.cnn_v_unknown += 1
                elif cnn_v == COLOR_OJAMA:
                    counters.cnn_v_ojama += 1
                else:
                    counters.cnn_v_other += 1
        return result

    return wrapped


def _would_use_next_bias(
    nc_obs: list[int], min_count: int, min_ratio: float,
) -> tuple[bool, int | None]:
    """distrust が無ければ NEXT色バイアス経路が採用されていたか (counterfactual)。"""
    if len(nc_obs) < min_count:
        return False, None
    counter = Counter(nc_obs)
    winner, count = counter.most_common(1)[0]
    ratio = count / len(nc_obs)
    if ratio >= min_ratio:
        return True, winner
    return False, None


def _make_start_vote_wrapper(orig, counters: _Counters):
    """`RecognitionPipeline._start_landing_vote` をラップする (タイミングずれ仮説検証)。"""

    @functools.wraps(orig)
    def wrapped(
        self, side, frame_idx, prev_confirmed, final_board,
        next_colors=None, distrust_cells=None,
    ):
        pending_before = len(
            self._pending_landing_vote_1p if side == "1P"
            else self._pending_landing_vote_2p,
        )
        result = orig(
            self, side, frame_idx, prev_confirmed, final_board,
            next_colors=next_colors, distrust_cells=distrust_cells,
        )
        pending_after = len(
            self._pending_landing_vote_1p if side == "1P"
            else self._pending_landing_vote_2p,
        )
        counters.n_start_vote_calls += 1
        if pending_after > pending_before:
            counters.n_start_vote_nonempty_diff += 1
        else:
            counters.n_start_vote_empty_diff += 1
        if distrust_cells:
            counters.n_start_vote_distrust_present += 1
        return result

    return wrapped


def _make_update_votes_wrapper(orig, counters: _Counters):
    """`RecognitionPipeline._update_landing_votes` (bound method) をラップする。"""

    @functools.wraps(orig)
    def wrapped(self, side, frame_idx, cnn_board, confirmed_board, frame_bgr=None):
        pending = (
            self._pending_landing_vote_1p if side == "1P"
            else self._pending_landing_vote_2p
        )
        # orig() が破壊的にミュートする前に、判定に必要な情報だけ浅く複製する。
        pre_snapshots: list[dict] = []
        for entry in pending:
            elapsed = frame_idx - entry["start"]
            distrust_cells = entry.get("distrust_cells", set())
            pre_snapshots.append({
                "will_finalize": elapsed >= self.LANDING_VOTE_FRAMES,
                "distrust_cells": set(distrust_cells),
                "confirmed_cells": set(entry.get("confirmed_cells", set())),
                "next_color_votes": {
                    k: list(v) for k, v in entry.get("next_color_votes", {}).items()
                },
                "cells": list(entry["cells"]),
            })
            # early-confirm 迂回の参考計数 (蓄積期間中、distrust セルが
            # 早期確定 gate 判定自体をスキップした回数)。
            if not pre_snapshots[-1]["will_finalize"] and distrust_cells:
                counters.n_early_bypass_events += len(
                    distrust_cells - pre_snapshots[-1]["confirmed_cells"],
                )

        result = orig(self, side, frame_idx, cnn_board, confirmed_board, frame_bgr)

        for pre in pre_snapshots:
            if not pre["will_finalize"] or not pre["distrust_cells"]:
                continue
            for (r, c, _expected) in pre["cells"]:
                if (r, c) not in pre["distrust_cells"]:
                    continue
                if (r, c) in pre["confirmed_cells"]:
                    continue  # 早期確定済 (distrust セルには通常起きないはずのガード)
                counters.n_distrust_branch_fired += 1
                nc_obs = pre["next_color_votes"].get((r, c), [])
                would_use, winner = _would_use_next_bias(
                    nc_obs, self.LANDING_VOTE_NEXT_MIN_COUNT,
                    self.LANDING_VOTE_NEXT_MIN_RATIO,
                )
                if not would_use:
                    counters.n_no_diff_possible += 1
                    continue
                counters.n_would_use_next_bias += 1
                actual_value = int(result.get(r, c)) if result is not None else None
                changed = actual_value is not None and actual_value != winner
                if changed:
                    counters.n_p7_judgment_changed += 1
                else:
                    counters.n_p7_judgment_same += 1
                counters.detail_rows.append({
                    "side": side, "frame_idx": frame_idx, "cell": [r, c],
                    "counterfactual_next_bias_winner": winner,
                    "actual_value": actual_value,
                    "changed": changed,
                })
        return result

    return wrapped


@contextmanager
def _install_hooks(counters: _Counters):
    """monkeypatch を一時適用し、with を抜けると必ず復元する。"""
    orig_flag = rp._flag_landing_distrust_cells
    orig_start_vote = rp.RecognitionPipeline._start_landing_vote
    orig_update = rp.RecognitionPipeline._update_landing_votes
    rp._flag_landing_distrust_cells = _make_flag_wrapper(orig_flag, counters)
    rp.RecognitionPipeline._start_landing_vote = _make_start_vote_wrapper(
        orig_start_vote, counters,
    )
    rp.RecognitionPipeline._update_landing_votes = _make_update_votes_wrapper(
        orig_update, counters,
    )
    try:
        yield
    finally:
        rp._flag_landing_distrust_cells = orig_flag
        rp.RecognitionPipeline._start_landing_vote = orig_start_vote
        rp.RecognitionPipeline._update_landing_votes = orig_update


def _format_report(counters: _Counters) -> str:
    """人間可読レポートに整形する。"""
    lines = [
        "==== (甲)修正 効果ゼロ 原因切り分け診断 (2026-07-25) ====",
        "",
        "--- H1向け: _flag_landing_distrust_cells 発火状況 ---",
        f"  呼び出し回数 (=着地イベント数, prev_confirmed非None): {counters.n_flag_calls}",
        f"  非空フラグ回数 (>=1セルflagged):                     {counters.n_flag_nonempty}",
        f"  フラグされた総セル数:                                 {counters.n_flag_total_cells}",
        f"  候補セル総数 (pv=EMPTY/UNKNOWN かつ inferred=有色):    {counters.n_candidate_cells}",
        f"    うち cnn_v=有効色 かつ inferred一致 (フラグされない): {counters.cnn_v_valid_match}",
        f"    うち cnn_v=有効色 かつ inferred不一致 (フラグ対象):  {counters.cnn_v_valid_mismatch}",
        f"    うち cnn_v=EMPTY (#47ガードでフラグされない):        {counters.cnn_v_empty}",
        f"    うち cnn_v=UNKNOWN (#47ガードでフラグされない):      {counters.cnn_v_unknown}",
        f"    うち cnn_v=おじゃま (#47ガードでフラグされない):     {counters.cnn_v_ojama}",
        f"    うち cnn_v=その他:                                   {counters.cnn_v_other}",
        "",
        "--- 追加検証: _start_landing_vote がそもそも entry を生成しているか ---",
        "  (タイミングずれ仮説: NEXT検知フレームがTSUMO_FALL→STABLE遷移フレームと"
        "ズレると prev_confirmed==final_board となり毎回 cells_with_expected が"
        "空になり得る)",
        f"  _start_landing_vote 呼び出し回数:                       {counters.n_start_vote_calls}",
        f"    うち cells_with_expected が空 (entry生成なし):         {counters.n_start_vote_empty_diff}",
        f"    うち cells_with_expected が非空 (entry生成あり):       {counters.n_start_vote_nonempty_diff}",
        f"    うち distrust_cells非空で呼ばれた回数:                 {counters.n_start_vote_distrust_present}",
        "",
        "--- H2向け: _update_landing_votes distrust分岐の実効性 ---",
        f"  早期確定gate迂回イベント数 (蓄積期間中, 参考値):        {counters.n_early_bypass_events}",
        f"  発火回数 (finalization時にfallbackへ迂回した回数):      {counters.n_distrust_branch_fired}",
        f"    うち counterfactual (distrust無し) でもNEXTバイアス不成立"
        f" (=どのみち差が出ない):                                 {counters.n_no_diff_possible}",
        f"    うち counterfactualでNEXTバイアスが成立していた回数:  {counters.n_would_use_next_bias}",
        f"      → 実際に確定値が変わった回数 (P7判定変化):          {counters.n_p7_judgment_changed}",
        f"      → 結果的に確定値が同じだった回数:                   {counters.n_p7_judgment_same}",
        "",
        "--- 判定 ---",
    ]
    if counters.n_flag_nonempty == 0:
        lines.append(
            "  H1確定: distrustフラグが一度も発火していない "
            "(非空フラグ回数=0)。",
        )
    elif (
        counters.n_start_vote_distrust_present > 0
        and counters.n_start_vote_nonempty_diff == 0
    ):
        lines.append(
            "  H1/H2いずれとも異なる第3因: distrustフラグは発火するが、"
            "_start_landing_vote が一度も非空 entry を生成していない "
            "(cells_with_expected が常に空)。原因は NEXT検知フレーム"
            "(landing_pending起動点) が TSUMO_FALL→STABLE 遷移フレームと"
            "一致しないタイミングずれ: prev_confirmed は _step_side 冒頭で"
            "毎フレーム再取得される (sm.update() 直前の confirmed_board) ため、"
            "着地が完了した後のフレームで _start_landing_vote が呼ばれると"
            "prev_confirmed==final_board となり diff が消える。"
            "この場合、P7 (_update_landing_votes) 自体が distrust の有無に"
            "関わらず実質死んでいる可能性が高い (要 Step 追加検証)。",
        )
    elif counters.n_distrust_branch_fired == 0:
        lines.append(
            "  H1寄り: フラグ自体は発火するが、_update_landing_votesの"
            "finalization分岐まで到達した事例が0件"
            "(着地投票entry自体が生成されないか、対象cellが早期確定/他経路で"
            "先に消費されている可能性)。",
        )
    elif counters.n_p7_judgment_changed == 0:
        lines.append(
            "  H2確定: distrust分岐は発火するが、P7の最終判定 (確定色) は"
            "一度も変化していない "
            "(NEXT色バイアス条件が不成立で元々fallbackと同じ経路を通っていた、"
            "または偶然fallback勝者=NEXTバイアス勝者だった)。",
        )
    else:
        lines.append(
            f"  H1/H2いずれも部分否定: フラグは発火し({counters.n_flag_nonempty}回)、"
            f"P7判定も{counters.n_p7_judgment_changed}回変化している。"
            "監査器側で効果が見えない場合は監査器のcolor_flicker検出条件"
            "(P7より前で完結する変化を捕捉できない等)側の問題を疑う必要がある。",
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="(甲)修正 効果ゼロ 原因切り分け診断 (read-only, 同期実行)",
    )
    ap.add_argument("--smoke", action="store_true", help="スモーク窓 (21秒) のみ実行")
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--start-sec", type=float, default=None, dest="start_sec")
    ap.add_argument("--max-sec", type=float, default=None, dest="max_sec")
    return ap.parse_args()


def main() -> None:
    cv2.setNumThreads(1)  # 熱対策・並列しない
    args = _parse_args()
    if args.smoke:
        stem, start_sec, max_sec = SMOKE_VIDEO_STEM, SMOKE_START_SEC, SMOKE_MAX_SEC
    elif args.video is not None:
        stem = args.video
        start_sec = args.start_sec if args.start_sec is not None else DEFAULT_START_SEC
        max_sec = args.max_sec if args.max_sec is not None else DEFAULT_MAX_SEC
    else:
        stem, start_sec, max_sec = DEFAULT_VIDEO_STEM, DEFAULT_START_SEC, DEFAULT_MAX_SEC

    print(
        f"[{time.strftime('%H:%M:%S')}] [{stem}] 診断開始 "
        f"start={start_sec:.1f}s dur={max_sec:.1f}s "
        f"(enable_placement_color_cnn_check=True 固定)", flush=True,
    )
    t0 = time.time()
    counters = _Counters()
    with _install_hooks(counters):
        _capture_frames(
            stem, start_sec, max_sec,
            enable_placement_color_cnn_check=True,
        )
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] 処理完了 ({elapsed:.1f}s)", flush=True)

    report = _format_report(counters)
    print(report)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUTPUT_DIR / f"{stem}_{int(start_sec)}_{int(max_sec)}_counters.json"
    out_txt = OUTPUT_DIR / f"{stem}_{int(start_sec)}_{int(max_sec)}_report.txt"
    out_json.write_text(
        json.dumps(counters.__dict__, ensure_ascii=False, indent=2, default=list),
        encoding="utf-8",
    )
    out_txt.write_text(report, encoding="utf-8")
    print(f"\n出力: {out_json}\n出力: {out_txt}")


if __name__ == "__main__":
    main()

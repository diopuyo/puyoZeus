"""Phase L 学習データ 動画別品質ゲート (2026-08-07)。

user指示「その辺の検知を厳格にするように」を受けた正式工程。
2026-08-07 夜に学習外動画 olRyxDGacbg で発見した系統誤認 2 種
(a) 未知キャラ背景→空セルが色に化ける (空クラス未学習)
(b) おじゃま→黄 誤認
を踏まえ、regen 済み boards_lean npz を入口検査し、動画単位で
系統誤りを検出→隔離するためのゲートスクリプト。

入力: `data/indicators_v2/boards_lean_phase_l_2026-08-07/*.npz`
      (collect_boards_lean.py 出力、新標準構成 = 第4機構修正A' 含む)
出力: scorecard.tsv (動画別 PASS/WARN/FAIL 一覧)
      + <video>.md (動画別詳細、根拠数値付き)

検査項目 (実装可能性調査済、可否は各関数の docstring 参照):
  1. 試合開始直後の空盤面正解率 (最重要、npz の t_sec + game_idx で高速判定)
  2. 列別・色別の系統偏り (row0-4、ライブラリ population z-score)
  3. おじゃま整合検査 (診断専用、chain 検知recall不足のためゲート化は見送り)
  4. 既存ゲート指標 I1 (列別 UNKNOWN 率) / C1 (平均ぷよ数) の per-video 適用

合格基準の根拠 (過学習防止則: シーン逆算禁止、定数はライブラリ実測から導出):
  - 空盤面 FAIL 閾値 0.5% は user指定の厳格値をそのまま採用。ライブラリ
    84 本の実測分布 (median 0.28%, p90 0.45%, p99 3.4%) 上では概ね p90
    相当であり、「厳格に」という指示と整合する (少数の既存動画も
    FAIL 対象になり得るが、それはゲート本来の目的=既存データの
    再点検であり不具合ではない)。
  - 系統偏り FAIL 閾値 z>5.0 はライブラリ population 統計から算出した
    z-score の多重比較 (6 列 × 5 色 = 30 セル) を踏まえた保守的カットオフ
    (帰無仮説下での期待 max|z| は理論上 3.0 前後、z>5 はほぼ確実に真の
    外れ値)。特定動画を狙って逆算した値ではない。

使い方:
    python scripts/phase_l_video_quality_gate.py --all
    python scripts/phase_l_video_quality_gate.py --video c10
    python scripts/phase_l_video_quality_gate.py --video olRyxDGacbg_10min \\
        --target-npz data/verify/phase_l_quality_gate_2026-08-07/negative_control/olRyxDGacbg_10min.npz
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from src.board import BOARD_COLS, COLOR_EMPTY, COLOR_UNKNOWN  # noqa: E402

# ============================
# 定数定義 (マジックナンバー禁止規約)
# ============================

# デフォルト npz ディレクトリ (ライブラリ population 基準にも使う)
DEFAULT_NPZ_DIR = Path("data/indicators_v2/boards_lean_phase_l_2026-08-07")
DEFAULT_OUT_DIR = Path("data/verify/phase_l_quality_gate_2026-08-07")

# --- 検査1: 試合開始直後の空盤面 ---
# scripts/_finetune_olRyxDGacbg_2026-08-07_extract_seed_v2.py の採取窓を
# 踏襲 (+1.0s 〜 +3.0s)。試合開始 = game_idx が変わった時点の最小 t_sec。
MATCH_START_OFFSET_LO_SEC: float = 1.0
MATCH_START_OFFSET_HI_SEC: float = 3.0
# 下段3行 (row10-12) は序盤の1手目設置が混入するため評価対象から除外
# (seed_v2 と同じ理由: 目的とは無関係な誤検知を避ける)。
MATCH_START_ROW_LO: int = 1  # HIDDEN_ROWS と同じ (隠し段は評価除外)
MATCH_START_ROW_HI_EXCLUSIVE: int = 10
# FAIL 閾値: 非空率 0.5% 超 (user指定の厳格値、根拠は module docstring 参照)
MATCH_START_NONEMPTY_FAIL_RATE: float = 0.005
# WARNING 閾値: ライブラリ84本の実測 p75 相当 (2026-08-07実測、NaN除く82本
# median=0.28%, p75=0.37%)。「FAILの半分」等の機械的値でなく実測分布から
# 導出 (過学習防止則: シーン逆算禁止、母集団の上位1/4を注意喚起帯とする)。
MATCH_START_NONEMPTY_WARN_RATE: float = 0.0037
# REVIEW 閾値 (2026-08-08 coordinator指摘対応): 検査1の評価対象セル数が
# この値未満なら「試合開始+1〜3秒窓に該当STABLEスナップショットが無い」
# = 判定不能とみなし REVIEW にする。0除算回避のための nan→PASS 誤魔化しは
# 誤PASS事故 (olRyxDGacbg 先頭10分クリップ、game境界0件→NaN→PASS化) の
# 直接原因だったため廃止する。
MATCH_START_MIN_EVAL_CELLS: int = 1

# --- 検査1 頑健化 (2026-08-11、パターンB=検査器誤検知の対策) ---
# 「各 game_idx の最小 t_sec = 試合開始」という代理値は、score リセット
# 検知の誤発火 (OCR誤読・勝敗確定直後の残存表示の巻き込み等) により実際は
# 試合中盤を指してしまうことがある (実画面 c57 t=1278s 満杯盤面で確定)。
# 見分け方: 真の試合開始なら両 side とも score は 0 近傍のはず。誤発火なら
# score は既存の(非ゼロの)値のまま、あるいは別の非ゼロ値になる。
#
# 根拠 (2026-08-11 実測、data/indicators_v2/boards_lean_phase_l_2026-08-07/
# 148 動画ライブラリ全 game_idx×side の有効 score (>0) エントリ 13,226 件):
# score<=13 に 13,161 件 (99.5%) が密集し、score∈[14,73] は完全に空白
# (該当0件、境界の1件のみ score=14)、次の実測値は score=74 から。
# 過学習防止則 (シーン逆算禁止) に従い、この空白域の中間にカットオフを置く
# (特定動画から逆算した値ではなく population のギャップから導出)。
MATCH_START_ANCHOR_SCORE_MAX: int = 50
# npz "score" 列の未読み取り sentinel (collect_boards_lean.py 準拠)。
# 未読み取り (-1) は「非試合画面でスコア UI 自体が存在しない」パターンA
# (本物の学習データ汚染) の代表的な特徴でもあるため除外対象にしない
# (厳格判定を維持、Aまで通してしまう過修正を避ける)。
SCORE_UNREADABLE: int = -1

# --- 検査2: 列別・色別の系統偏り ---
# row0 (隠し段) 〜 row4 (上から4行目の可視段) を「上部帯」として評価する。
SYSTEMIC_BIAS_ROW_LO: int = 0
SYSTEMIC_BIAS_ROW_HI_EXCLUSIVE: int = 5
# 色ぷよ (COLOR_UNKNOWN・COLOR_EMPTY・おじゃまは対象外、色化け検知が目的)
SYSTEMIC_BIAS_COLORS: tuple[int, ...] = (1, 2, 3, 4, 5)
# z-score カットオフ (根拠: module docstring)
SYSTEMIC_BIAS_Z_WARN: float = 3.5
SYSTEMIC_BIAS_Z_FAIL: float = 5.0
# population std がゼロ近傍のときの除算保護
SYSTEMIC_BIAS_STD_FLOOR: float = 1e-6

# --- 検査3 (診断専用、ゲート化しない): おじゃま整合 ---
# chain_mechanism フラグの recall 不足 (実測: 減少イベントの 93% が
# 同時刻±1entry に chain フラグを持たない) を 2026-08-07 に確認済。
# 現状ではこの数値だけで FAIL 判定すると既存ライブラリ動画も大量に
# 誤検知するため、報告専用の診断値として出力する (ゲートには使わない)。
OJAMA_DIAG_WINDOW_ENTRIES: int = 1

# --- 検査4: 既存ゲート指標 I1 / C1 の per-video 適用 ---
# 値の出典: scripts/measure_stable_cell_acc.py の同名定数
# (PER_COL_UNKNOWN_WARNING/CRITICAL, AVG_PUYO_COUNT_CRITICAL) をそのまま
# 踏襲する。npz には raw_cnn/raw_hsv が無いため D1 (postprocess_corruption)
# は算出不可 (scripts/_gate_check_2026-08-06.py の docstring 既知の限界と同じ)。
I1_COL_UNKNOWN_WARNING: float = 0.15
I1_COL_UNKNOWN_CRITICAL: float = 0.30
C1_AVG_PUYO_COUNT_WARNING: float = 5.0


# ============================
# データクラス
# ============================

@dataclass
class VideoArrays:
    """1 動画分の npz 生配列を保持する軽量コンテナ。"""

    video_id: str
    grids: np.ndarray  # (n, 13, 6) int8
    t_sec: np.ndarray
    side: np.ndarray
    game_idx: np.ndarray
    chain_mechanism: np.ndarray = field(default_factory=lambda: np.array([]))
    # 2026-08-11 追加 (検査1 頑健化用、後方互換のため末尾・デフォルト付き)。
    # score 列が無い npz (旧形式) では空配列のままとなり、
    # compute_match_start_nonempty は従来通りの判定にフォールバックする。
    score: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class VideoGateResult:
    """1 動画分のゲート判定結果。"""

    video_id: str
    match_start_nonempty_rate: float
    match_start_n_cells: int
    systemic_bias_max_z: float
    systemic_bias_worst_combo: str
    col_unknown_max_rate: float
    avg_puyo_count: float
    ojama_diag_decrease_no_chain_rate: float
    verdict: str  # "PASS" / "WARN" / "FAIL"
    reasons: list[str] = field(default_factory=list)
    # 2026-08-11 追加 (末尾・デフォルト付きで後方互換維持): 検査1で
    # 「試合開始でない」と判定して除外した game_idx×side 窓数 (診断用)。
    match_start_excluded_windows: int = 0


# ============================
# npz 読み込み
# ============================

def load_video_arrays(npz_path: Path) -> VideoArrays:
    """boards_lean npz を読み込み VideoArrays を返す。

    npz は collect_boards_lean.py 出力形式を前提とする
    (grids/t_sec/side/game_idx/chain_mechanism キー必須)。
    """
    d = np.load(npz_path, allow_pickle=True)
    return VideoArrays(
        video_id=npz_path.stem,
        grids=d["grids"],
        t_sec=d["t_sec"],
        side=d["side"],
        game_idx=d["game_idx"],
        chain_mechanism=d["chain_mechanism"] if "chain_mechanism" in d else np.array([]),
        score=d["score"] if "score" in d else np.array([]),
    )


# ============================
# 検査1: 試合開始直後の空盤面
# ============================

def _anchor_score(arrays: VideoArrays, side_game_mask: np.ndarray, start_sec: float) -> int | None:
    """side_game_mask 内で start_sec に最も近い score 値を返す。

    score 列が無い npz (旧形式) や該当エントリが無い場合は None
    (= 判定不能、頑健化チェックをスキップして従来通り評価する) を返す。
    """
    if arrays.score.size == 0 or not side_game_mask.any():
        return None
    tt = arrays.t_sec[side_game_mask]
    idx_nearest = int(np.argmin(np.abs(tt - start_sec)))
    return int(arrays.score[side_game_mask][idx_nearest])


def _is_untrustworthy_start(anchor_score: int | None) -> bool:
    """anchor_score が「本当の試合開始でない」ことを示すか判定する。

    真の試合開始は score が 0 近傍のはず。score が読めている
    (SCORE_UNREADABLE でない) のに MATCH_START_ANCHOR_SCORE_MAX を
    超える場合のみ「代理値がズレている」と判定する。
    SCORE_UNREADABLE (-1) や score 列欠如 (None) はパターンA
    (非試合画面でスコアUI自体が無い) の可能性を残すため除外しない
    (= 厳格判定を維持し、過修正を避ける)。
    """
    if anchor_score is None or anchor_score == SCORE_UNREADABLE:
        return False
    return anchor_score > MATCH_START_ANCHOR_SCORE_MAX


def compute_match_start_nonempty(arrays: VideoArrays) -> tuple[float, int, int]:
    """試合開始+1〜3秒の STABLE スナップショットにおける非空セル率を返す。

    各 game_idx の最小 t_sec を試合開始の代理値とし、その +1.0〜+3.0 秒
    窓に入る側別スナップショットを rows [MATCH_START_ROW_LO,
    MATCH_START_ROW_HI_EXCLUSIVE) で評価する。窓に該当エントリが無い
    game_idx/side は分母に加算しない (dedup npz のため「差分が無い=
    直前と同じ空盤面が継続」を意味し、誤検知の心配がないケース)。

    2026-08-11 追加 (パターンB=検査器誤検知対策): 代理値の起点で score が
    読めているのに 0 近傍でない (`_is_untrustworthy_start`) 場合、その
    game_idx×side の窓は「試合開始でない」とみなして評価対象から除外する
    (分子・分母どちらにも加算しない)。

    戻り値: (非空率, 評価対象セル総数, 除外した窓数)。
        セル総数 0 の場合は (nan, 0, 除外窓数)。
    """
    nonempty_total = 0
    cell_total = 0
    excluded_windows = 0
    for g in np.unique(arrays.game_idx):
        game_mask = arrays.game_idx == g
        if not game_mask.any():
            continue
        start_sec = arrays.t_sec[game_mask].min()
        lo = start_sec + MATCH_START_OFFSET_LO_SEC
        hi = start_sec + MATCH_START_OFFSET_HI_SEC
        for s in ("1P", "2P"):
            m = game_mask & (arrays.side == s) & (arrays.t_sec >= lo) & (arrays.t_sec <= hi)
            if not m.any():
                continue
            side_game_mask = game_mask & (arrays.side == s)
            anchor_score = _anchor_score(arrays, side_game_mask, start_sec)
            if _is_untrustworthy_start(anchor_score):
                excluded_windows += 1
                continue
            sub = arrays.grids[m][:, MATCH_START_ROW_LO:MATCH_START_ROW_HI_EXCLUSIVE, :]
            nonempty_total += int((sub != COLOR_EMPTY).sum())
            cell_total += int(sub.size)
    if cell_total == 0:
        return float("nan"), 0, excluded_windows
    return nonempty_total / cell_total, cell_total, excluded_windows


# ============================
# 検査2: 列別・色別の系統偏り
# ============================

def compute_color_col_rates(arrays: VideoArrays) -> np.ndarray:
    """rows [SYSTEMIC_BIAS_ROW_LO, HI) の col×color 出現率を返す。

    戻り値 shape: (BOARD_COLS, len(SYSTEMIC_BIAS_COLORS))。
    """
    band = arrays.grids[:, SYSTEMIC_BIAS_ROW_LO:SYSTEMIC_BIAS_ROW_HI_EXCLUSIVE, :]
    out = np.zeros((BOARD_COLS, len(SYSTEMIC_BIAS_COLORS)))
    for col in range(BOARD_COLS):
        col_band = band[:, :, col]
        denom = col_band.size
        for ci, color in enumerate(SYSTEMIC_BIAS_COLORS):
            out[col, ci] = (col_band == color).sum() / denom if denom > 0 else float("nan")
    return out


def build_population_stats(
    library_rates: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """ライブラリ全体の col×color 出現率から population mean/std を返す。"""
    mat = np.stack(list(library_rates.values()))  # (n_video, col, color)
    return mat.mean(axis=0), mat.std(axis=0)


def leave_one_out_z(
    target_rates: np.ndarray,
    library_rates: dict[str, np.ndarray],
    target_video_id: str,
) -> np.ndarray:
    """target を population から除外した leave-one-out z-score 行列を返す。

    target_video_id がライブラリに含まれない場合 (= 単発検証対象) は
    通常の population 統計をそのまま使う (LOO と等価な扱い)。
    """
    others = {k: v for k, v in library_rates.items() if k != target_video_id}
    if not others:
        return np.full_like(target_rates, float("nan"))
    mean, std = build_population_stats(others)
    return (target_rates - mean) / np.maximum(std, SYSTEMIC_BIAS_STD_FLOOR)


def summarize_worst_bias(z_matrix: np.ndarray) -> tuple[float, str]:
    """z 行列から最大絶対値とその (col, color) ラベルを返す。"""
    flat_idx = int(np.nanargmax(np.abs(z_matrix)))
    col, ci = np.unravel_index(flat_idx, z_matrix.shape)
    color = SYSTEMIC_BIAS_COLORS[ci]
    return float(z_matrix[col, ci]), f"col={col},color={color}"


# ============================
# 検査3 (診断専用): おじゃま整合
# ============================

def compute_ojama_diagnostic(arrays: VideoArrays) -> float:
    """おじゃま個数が chain フラグなしで減少したスナップショット遷移の比率。

    注意: 診断専用メトリクス。2026-08-07 実測で chain_mechanism フラグの
    recall 不足 (減少イベントの ~93% が同時刻近傍にフラグを持たない) を
    確認済のため、ゲート判定には使わない (FAIL/WARN に反映しない)。
    将来 chain 検知精度が改善した際の再評価用に数値だけ残す。
    """
    if arrays.chain_mechanism.size == 0:
        return float("nan")
    ojama_count = (arrays.grids == 9).reshape(arrays.grids.shape[0], -1).sum(axis=1)
    n_decrease = 0
    n_decrease_no_chain = 0
    for s in ("1P", "2P"):
        for g in np.unique(arrays.game_idx):
            m = (arrays.side == s) & (arrays.game_idx == g)
            if m.sum() < 2:
                continue
            order = np.argsort(arrays.t_sec[m])
            oj = ojama_count[m][order]
            cm = arrays.chain_mechanism[m][order]
            for i in range(1, len(oj)):
                if oj[i] >= oj[i - 1]:
                    continue
                n_decrease += 1
                lo = max(0, i - OJAMA_DIAG_WINDOW_ENTRIES)
                if not np.any(cm[lo:i + 1] != ""):
                    n_decrease_no_chain += 1
    return n_decrease_no_chain / n_decrease if n_decrease > 0 else float("nan")


# ============================
# 検査4: 既存ゲート指標 I1 / C1
# ============================

def compute_col_unknown_max_rate(arrays: VideoArrays) -> float:
    """列別 COLOR_UNKNOWN 比率の最大値を返す (I1 系メトリクス)。"""
    rates = [float(np.mean(arrays.grids[:, :, c] == COLOR_UNKNOWN)) for c in range(BOARD_COLS)]
    return max(rates) if rates else float("nan")


def compute_avg_puyo_count(arrays: VideoArrays) -> float:
    """STABLE スナップショット1枚あたりの平均ぷよ数を返す (C1 メトリクス)。

    EMPTY/UNKNOWN を除く全セル (色ぷよ+おじゃま) の平均個数。
    """
    if arrays.grids.shape[0] == 0:
        return float("nan")
    puyo_mask = (arrays.grids != COLOR_EMPTY) & (arrays.grids != COLOR_UNKNOWN)
    per_snapshot = puyo_mask.reshape(arrays.grids.shape[0], -1).sum(axis=1)
    return float(per_snapshot.mean())


# ============================
# 総合判定
# ============================

def _judge_match_start(rate: float, n_cells: int) -> tuple[str, list[str]]:
    """検査1の判定を返す。

    2026-08-08 修正: n_cells が MATCH_START_MIN_EVAL_CELLS 未満 (= 試合開始
    +1〜3秒窓に該当 STABLE スナップショットが無い、game_idx 境界不足の疑い)
    の場合は PASS でなく REVIEW を返す。以前は nan→PASS だったため
    olRyxDGacbg 先頭10分クリップ (game境界0件) を誤PASSさせた事故があった。
    """
    if n_cells < MATCH_START_MIN_EVAL_CELLS or np.isnan(rate):
        return "REVIEW", [
            f"match_start_nonempty_rate 判定不能 (評価セル数={n_cells}、"
            "試合開始+1〜3秒窓に該当STABLEスナップショット無し。"
            "game_idx境界不足 or score OCR破綻の疑い、要目視確認)"
        ]
    if rate > MATCH_START_NONEMPTY_FAIL_RATE:
        return "FAIL", [f"match_start_nonempty_rate={rate:.4%} > {MATCH_START_NONEMPTY_FAIL_RATE:.2%}"]
    if rate > MATCH_START_NONEMPTY_WARN_RATE:
        return "WARN", [f"match_start_nonempty_rate={rate:.4%} > {MATCH_START_NONEMPTY_WARN_RATE:.2%} (warn)"]
    return "PASS", []


def _judge_systemic_bias(max_z: float, combo: str) -> tuple[str, list[str]]:
    if np.isnan(max_z):
        return "PASS", []
    if abs(max_z) > SYSTEMIC_BIAS_Z_FAIL:
        return "FAIL", [f"systemic_bias max|z|={max_z:.2f} ({combo}) > {SYSTEMIC_BIAS_Z_FAIL}"]
    if abs(max_z) > SYSTEMIC_BIAS_Z_WARN:
        return "WARN", [f"systemic_bias max|z|={max_z:.2f} ({combo}) > {SYSTEMIC_BIAS_Z_WARN} (warn)"]
    return "PASS", []


def _judge_i1_c1(col_unknown_max: float, avg_puyo: float) -> tuple[str, list[str]]:
    reasons: list[str] = []
    verdict = "PASS"
    if not np.isnan(col_unknown_max) and col_unknown_max > I1_COL_UNKNOWN_CRITICAL:
        verdict = "FAIL"
        reasons.append(f"I1 col_unknown_max={col_unknown_max:.2%} > {I1_COL_UNKNOWN_CRITICAL:.0%}")
    elif not np.isnan(col_unknown_max) and col_unknown_max > I1_COL_UNKNOWN_WARNING:
        verdict = "WARN"
        reasons.append(f"I1 col_unknown_max={col_unknown_max:.2%} > {I1_COL_UNKNOWN_WARNING:.0%} (warn)")
    if not np.isnan(avg_puyo) and avg_puyo < C1_AVG_PUYO_COUNT_WARNING:
        verdict = "FAIL" if verdict == "FAIL" else "WARN"
        reasons.append(f"C1 avg_puyo_count={avg_puyo:.1f} < {C1_AVG_PUYO_COUNT_WARNING}")
    return verdict, reasons


# REVIEW (2026-08-08 追加): 検査1が判定不能な場合の第3区分。
# 「判定不能=良好」ではないため WARN より重く扱うが、確定 FAIL より軽い
# (人間の目視確認待ちという意味合い)。
_VERDICT_RANK = {"PASS": 0, "WARN": 1, "REVIEW": 2, "FAIL": 3}


def _combine_verdicts(verdicts: list[str]) -> str:
    return max(verdicts, key=lambda v: _VERDICT_RANK[v])


def evaluate_video(
    arrays: VideoArrays,
    library_rates: dict[str, np.ndarray],
) -> VideoGateResult:
    """1 動画分の全検査を実行し VideoGateResult を返す。"""
    ms_rate, ms_n, ms_excluded = compute_match_start_nonempty(arrays)
    ms_verdict, ms_reasons = _judge_match_start(ms_rate, ms_n)

    target_rates = compute_color_col_rates(arrays)
    z_matrix = leave_one_out_z(target_rates, library_rates, arrays.video_id)
    bias_max_z, bias_combo = summarize_worst_bias(z_matrix)
    bias_verdict, bias_reasons = _judge_systemic_bias(bias_max_z, bias_combo)

    col_unknown_max = compute_col_unknown_max_rate(arrays)
    avg_puyo = compute_avg_puyo_count(arrays)
    i1c1_verdict, i1c1_reasons = _judge_i1_c1(col_unknown_max, avg_puyo)

    ojama_diag = compute_ojama_diagnostic(arrays)

    verdict = _combine_verdicts([ms_verdict, bias_verdict, i1c1_verdict])
    reasons = ms_reasons + bias_reasons + i1c1_reasons

    return VideoGateResult(
        video_id=arrays.video_id,
        match_start_nonempty_rate=ms_rate,
        match_start_n_cells=ms_n,
        systemic_bias_max_z=bias_max_z,
        systemic_bias_worst_combo=bias_combo,
        col_unknown_max_rate=col_unknown_max,
        avg_puyo_count=avg_puyo,
        ojama_diag_decrease_no_chain_rate=ojama_diag,
        verdict=verdict,
        reasons=reasons,
        match_start_excluded_windows=ms_excluded,
    )


# ============================
# 出力
# ============================

_SCORECARD_HEADER = (
    "video_id\tverdict\tmatch_start_nonempty_rate\tsystemic_bias_max_z\t"
    "systemic_bias_worst_combo\tcol_unknown_max_rate\tavg_puyo_count\t"
    "ojama_diag_decrease_no_chain_rate\tmatch_start_excluded_windows\treasons"
)


def _format_scorecard_row(r: VideoGateResult) -> str:
    reasons_str = "; ".join(r.reasons) if r.reasons else "-"
    return "\t".join([
        r.video_id, r.verdict,
        f"{r.match_start_nonempty_rate:.6f}", f"{r.systemic_bias_max_z:.3f}",
        r.systemic_bias_worst_combo, f"{r.col_unknown_max_rate:.6f}",
        f"{r.avg_puyo_count:.2f}", f"{r.ojama_diag_decrease_no_chain_rate:.4f}",
        str(r.match_start_excluded_windows),
        reasons_str,
    ])


def write_scorecard(
    results: list[VideoGateResult], out_dir: Path, filename: str = "scorecard.tsv",
) -> Path:
    """全動画分の scorecard を書き出す。

    Args:
        filename: 出力ファイル名。デフォルト "scorecard.tsv" (--all 実行時)。
            2026-08-08 追加: --video 単発実行時にライブラリ全体の
            scorecard.tsv を誤って1行に上書きする事故 (olRyxDGacbg 陰性対照
            regen 完了時に発生) の再発防止として、呼出元 (main) は単発モード
            では別名を渡す。backwards compat: 省略時は従来どおり scorecard.tsv。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    lines = [_SCORECARD_HEADER] + [_format_scorecard_row(r) for r in results]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_video_detail_md(r: VideoGateResult, out_dir: Path) -> Path:
    """動画別詳細 md を書き出す。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{r.video_id}.md"
    lines = [
        f"# Phase L 品質ゲート: {r.video_id}",
        "",
        f"**判定: {r.verdict}**",
        "",
        "## 検査1: 試合開始直後の空盤面",
        f"- 非空率: {r.match_start_nonempty_rate:.4%} (評価セル数 {r.match_start_n_cells})",
        f"- FAIL閾値: {MATCH_START_NONEMPTY_FAIL_RATE:.2%} / WARN閾値: {MATCH_START_NONEMPTY_WARN_RATE:.2%}",
        f"- 除外した窓数 (score非ゼロ=代理値ズレ疑い、2026-08-11追加): "
        f"{r.match_start_excluded_windows}",
        "",
        "## 検査2: 列別・色別の系統偏り (row0-4, library z-score)",
        f"- 最大 |z|: {r.systemic_bias_max_z:.2f} ({r.systemic_bias_worst_combo})",
        f"- FAIL閾値: z>{SYSTEMIC_BIAS_Z_FAIL} / WARN閾値: z>{SYSTEMIC_BIAS_Z_WARN}",
        "",
        "## 検査4: 既存ゲート指標 I1/C1",
        f"- I1 col_unknown_max_rate: {r.col_unknown_max_rate:.4%}",
        f"- C1 avg_puyo_count: {r.avg_puyo_count:.2f}",
        "",
        "## 検査3 (診断専用、ゲート非対象): おじゃま整合",
        f"- ojama_diag_decrease_no_chain_rate: {r.ojama_diag_decrease_no_chain_rate:.4f}",
        "  (chain検知recall不足のため参考値のみ、PASS/FAILに影響しない)",
        "",
        "## 判定理由",
    ] + ([f"- {reason}" for reason in r.reasons] if r.reasons else ["- (該当なし)"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ============================
# CLI
# ============================

def _collect_npz_paths(npz_dir: Path) -> dict[str, Path]:
    """npz_dir 配下の *.npz を {video_id: path} で返す (video_id=stem)。"""
    return {p.stem: p for p in sorted(npz_dir.glob("*.npz"))}


def _build_library_rates(npz_dir: Path) -> dict[str, np.ndarray]:
    """population 基準用: npz_dir 内の全動画の col×color 出現率を返す。"""
    library_rates: dict[str, np.ndarray] = {}
    for video_id, path in _collect_npz_paths(npz_dir).items():
        arrays = load_video_arrays(path)
        library_rates[video_id] = compute_color_col_rates(arrays)
    return library_rates


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI 引数を解析する。"""
    parser = argparse.ArgumentParser(description="Phase L 動画別品質ゲート")
    parser.add_argument("--video", type=str, default=None, help="単一動画のみ検査 (video_id)")
    parser.add_argument("--all", action="store_true", help="--npz-dir 配下の全 npz を検査する")
    parser.add_argument("--npz-dir", type=Path, default=DEFAULT_NPZ_DIR, help="ライブラリ npz ディレクトリ")
    parser.add_argument(
        "--target-npz", type=Path, default=None,
        help="--video 対象の npz パスを明示指定する (--npz-dir 外の候補動画用)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="出力先ディレクトリ")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if not args.all and not args.video:
        print("[gate] --all か --video のいずれかを指定してください", file=sys.stderr)
        return 1

    library_rates = _build_library_rates(args.npz_dir)
    print(f"[gate] library population = {len(library_rates)} 動画 (from {args.npz_dir})")

    results: list[VideoGateResult] = []
    if args.all:
        for video_id, path in _collect_npz_paths(args.npz_dir).items():
            arrays = load_video_arrays(path)
            results.append(evaluate_video(arrays, library_rates))
    else:
        path = args.target_npz if args.target_npz is not None else args.npz_dir / f"{args.video}.npz"
        arrays = load_video_arrays(path)
        arrays.video_id = args.video
        results.append(evaluate_video(arrays, library_rates))

    for r in results:
        write_video_detail_md(r, args.out_dir)
    # 2026-08-08 修正: --video 単発実行時はライブラリ全体の scorecard.tsv を
    # 上書きしない (別名ファイルに書く)。--all のときのみ scorecard.tsv。
    scorecard_filename = "scorecard.tsv" if args.all else f"single_{args.video}.tsv"
    scorecard_path = write_scorecard(results, args.out_dir, filename=scorecard_filename)

    n_fail = sum(1 for r in results if r.verdict == "FAIL")
    n_warn = sum(1 for r in results if r.verdict == "WARN")
    n_review = sum(1 for r in results if r.verdict == "REVIEW")
    n_pass = len(results) - n_fail - n_warn - n_review
    print(
        f"[gate] 完了: {len(results)} 動画 (FAIL={n_fail} WARN={n_warn} "
        f"REVIEW={n_review} PASS={n_pass}) -> {scorecard_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

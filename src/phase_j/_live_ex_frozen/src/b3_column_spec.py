"""B3 学習行の列仕様台帳 (名前・単位・正規化・欠測理由・available_at・側属性・対応列・schema版)。

行生成器 (b3_rows) と文書 (docs/B3_COLUMN_SPEC.md) の単一情報源。
側交換 (1P<->2P) は列の種類 (kind) ごとに明示的に変換し、一括反転しない
(CLAUDE.md 2026-08-10 恒久指示)。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from typing import Mapping

import numpy as np

SCHEMA_VERSION = "b3_rows_v1"
SIDES: tuple[str, str] = ("p1", "p2")
# 非飽和正規化 z/(a+z) の既定 a。学習 fold 内で再決定する (生値は別列で保持)。
NONSAT_A_DEFAULT = 3.0
SIGNED_CENTER = 0.5
ICON_TYPES: tuple[str, ...] = ("small", "large", "rock", "star", "moon", "crown")
ICON_UNIT: Mapping[str, int] = {"small": 1, "large": 6, "rock": 30, "star": 180, "moon": 360, "crown": 720}
K_LEVELS: tuple[int, ...] = (1, 2, 3, 4, 5)
# 既存 FULL_MODEL_GRID_REGISTRY (27) + 連結3種 (build 側と同順)。
IND_NAMES: tuple[str, ...] = (
    "board_color_puyo_total", "board_puyo_total", "max_column_height", "column_bumpiness",
    "death_margin", "death_margin_neighbor", "center_bulge_color", "center_bulge_ojama",
    "board_ojama_count", "color_diversity_evenness", "buried_hole_count", "current_max_chain",
    "dig_resistance", "ukeyasusa", "sub_chain_count", "immediate_fire_power", "chain_efficiency",
    "min_puyos_to_ignite", "second_chain_potential", "main_linked_pair_count",
    "isolated_pair_count", "main_linked_ratio", "ignition_point_count", "multi_color_ignition",
    "simultaneous_pop_richness", "saturation_chain_upper", "chain_articulation_point_count",
    "conn_pair_count", "conn_triple_count", "conn_max_group_size",
)
RAW_COUNT_INDICATORS: frozenset[str] = frozenset(
    {"conn_pair_count", "conn_triple_count", "conn_max_group_size"})
TA_VARIANTS: tuple[str, ...] = ("pred_med", "pred_fast", "sl_med", "sl_fast", "icon_med", "icon_fast")


class Kind(str, Enum):
    """側交換時の変換規則。"""
    META = "meta"            # 側交換で不変 (キー・時刻等)
    COMMON = "common"        # 側交換で不変 (位相・共通時刻)
    SIDE_ABS = "side_abs"    # 絶対量: p1<->p2 列を入れ替える
    DIFF_RAW = "diff_raw"    # 生差分 (p1-p2): 符号反転
    DIFF_CENTERED = "diff_centered"  # 0.5+0.5*(p1-p2): 1-x
    LABEL = "label"          # won_p1: 1-x
    WEIGHT = "weight"        # 重み: 不変 (複製側で半分は生成時に反映)


class Mask(IntEnum):
    """欠測理由コード。値が NaN の列には必ず非0の理由が付く。"""
    OK = 0
    LEGACY_SCHEMA_ABSENT = 1
    UNOBSERVABLE = 2
    NOT_YET_AVAILABLE = 3
    BOUNDARY_UNKNOWN = 4
    ONE_SIDE_UNOBSERVED = 5
    UNPREDICTABLE = 6
    SEARCH_TRUNCATED = 7
    NOT_APPLICABLE = 8
    DEFERRED = 9
    RECOGNITION_CONFLICT = 10
    HISTORY_INCOMPLETE = 11


@dataclass(frozen=True)
class ColumnSpec:
    """1列の仕様。"""
    name: str
    kind: Kind
    dtype: str            # f8 / f4 / i4 / i1 / str
    unit: str
    norm: str             # 正規化の説明 (none / 0-1 / nonsat 等)
    mask: str             # 欠測理由を持つ mask 列名 ("" = NaN なし)
    avail: str            # available_at の元 (av_* 列の語幹 or "frame" 等)
    group: str
    counterpart: str      # 側交換の相手列 ("" = なし)
    desc: str
    role: str = "feature"  # feature / meta / label / R_only / audit
    schema: str = SCHEMA_VERSION


def nonsat(x: np.ndarray | float, a: float = NONSAT_A_DEFAULT) -> np.ndarray:
    """z=log1p(x), f=z/(a+z)。x>=0。NaN は NaN のまま (0 にしない)。"""
    arr = np.asarray(x, dtype=np.float64)
    z = np.log1p(np.maximum(arr, 0.0))
    return np.where(np.isnan(arr), np.nan, z / (a + z))


def signed_nonsat(m: np.ndarray | float, a: float = NONSAT_A_DEFAULT) -> np.ndarray:
    """g(m)=0.5+0.5*sign(m)*f(|m|)。m=0 は 0.5。NaN 保持。"""
    arr = np.asarray(m, dtype=np.float64)
    out = SIGNED_CENTER + SIGNED_CENTER * np.sign(arr) * nonsat(np.abs(arr), a)
    return np.where(np.isnan(arr), np.nan, out)


def _S(stem: str) -> tuple[str, str]:
    return (f"{stem}_p1", f"{stem}_p2")


def _side(stem: str, dtype: str, unit: str, norm: str, mask: str, avail: str, group: str,
          desc: str, role: str = "feature") -> list[ColumnSpec]:
    a, b = _S(stem)
    m1, m2 = (_S(mask) if mask else ("", ""))
    if avail in ("frame", "av_frame"):
        av1 = av2 = "av_frame"
    else:
        av1, av2 = (avail + "_p1", avail + "_p2") if avail else ("", "")
    return [ColumnSpec(a, Kind.SIDE_ABS, dtype, unit, norm, m1, av1, group, b, desc, role),
            ColumnSpec(b, Kind.SIDE_ABS, dtype, unit, norm, m2, av2, group, a, desc, role)]


def _mask_cols(stem: str, group: str) -> list[ColumnSpec]:
    a, b = _S(stem)
    d = f"{group} の欠測理由 (Mask)"
    return [ColumnSpec(a, Kind.SIDE_ABS, "i1", "code", "none", "", "", group, b, d, "audit"),
            ColumnSpec(b, Kind.SIDE_ABS, "i1", "code", "none", "", "", group, a, d, "audit")]


def _common(name: str, dtype: str, unit: str, norm: str, mask: str, avail: str, group: str,
            desc: str, role: str = "feature") -> ColumnSpec:
    return ColumnSpec(name, Kind.COMMON, dtype, unit, norm, mask, avail, group, "", desc, role)


def _meta_cols() -> list[ColumnSpec]:
    m = lambda n, d, u, desc, k=Kind.META, r="meta": ColumnSpec(n, k, d, u, "none", "", "", "meta", "", desc, r)
    cols = [m("row_key", "str", "-", "video|game|frame|orientation で一意"),
            m("video_id", "str", "-", "動画ID"), m("game_idx", "i4", "-", "試合番号 (B2 記録)"),
            m("frame_idx", "i4", "-", "行を生成した認識フレーム番号"),
            m("t_sec", "f8", "秒", "評価時点 (行は available_at<=t_sec の情報のみ使う)"),
            m("orientation", "i1", "-", "0=原向き 1=1P/2P交換複製"),
            m("trigger_mask", "i4", "bit", "更新契機ビット (b3_rows.TRIGGER_BITS)"),
            m("held_sec", "f8", "秒", "次の行までの保持秒 (行重みの根拠)", r="label"),
            ColumnSpec("weight", Kind.WEIGHT, "f8", "1/試合", "none", "", "", "meta", "",
                       "行重み。試合内合計1(原向き0.5+複製0.5)。未来の行に依存するため特徴ではない", "label"),
            ColumnSpec("won_p1", Kind.LABEL, "f4", "0/1", "none", "", "", "meta", "",
                       "教師: 1Pが最終勝者なら1 (lean.npz won 由来。特徴生成には渡さない)", "label"),
            m("av_max", "f8", "秒", "この行で使った全情報の available_at 最大 (<=t_sec が不変条件)", r="audit"),
            m("av_frame", "f8", "秒", "フレーム観測の available_at", r="audit")]
    cols += _side("lean_row", "i4", "index", "none", "", "", "meta", "lean.npz の対応する直前の盤面行 (-1=無し)", "meta")
    for stem in ("av_board", "av_queue", "av_fire", "av_formula", "av_icon"):
        cols += _side(stem, "f8", "秒", "none", "", "", "meta", f"{stem} の available_at (NaN=未使用)", "audit")
    return cols


def _common_cols() -> list[ColumnSpec]:
    c = [_common("phase", "i1", "0/1/2", "none", "mk_phase", "board", "phase",
                 "user定義局面: 各側の色1-5+おじゃま9の総量(row0込み,UNKNOWN除外)の両側max。<=18序盤/19-47中盤/>=48終盤"),
         _common("phase_total_max", "f4", "個", "none", "mk_phase", "board", "phase", "位相の元になる両側 max 総量"),
         _common("mk_phase", "i1", "code", "none", "", "", "phase", "位相の欠測理由 (片側盤面無=5, UNKNOWN含む=2)", "audit"),
         _common("elapsed_first_move", "f8", "秒", "none", "", "frame", "time", "両側のうち早い最初の1手からの経過秒 (G20)"),
         _common("rate_now", "f4", "点/個", "none", "", "frame", "time", "有効マージンレート (from_first_move 起点)"),
         _common("mode_exchange", "i1", "0/1", "none", "", "frame", "mode", "撃ち合い(1)/静止(0)。連鎖中・OJAMA_FALL・着弾直後があれば1 (勝敗を見ない)"),
         _common("mode_bits", "i1", "bit", "none", "", "frame", "mode", "1=連鎖中 2=OJAMA_FALL 4=着弾直後 8=予告あり（B3_LEDGER_MODE_DEFAULTで台帳へ切替、既定は補正アイコン）"),
         _common("n_known_colors", "i1", "個", "none", "", "board", "colors", "試合内で因果的に観測済みの色数 (<4 のとき探索は5色)")]
    return c


def _board_cols() -> list[ColumnSpec]:
    c: list[ColumnSpec] = []
    g = "board"
    for stem, unit, desc in (("cells_color", "個", "色ぷよ(1-5)数 row0込み"), ("cells_ojama", "個", "盤面おじゃま(9)数"),
                             ("cells_unknown", "個", "UNKNOWN(10)数 (総量に数えない)"),
                             ("cells_total", "個", "色+おじゃま総量 (位相の元)"),
                             ("max_height", "段", "最大列高さ"), ("board_age", "秒", "確定盤面の年齢")):
        c += _side(stem, "f4", unit, "none", "mk_board", "av_board", g, desc)
    c += _side("is_dead", "i1", "0/1", "none", "mk_board", "av_board", g, "DEATH_ROW=1 の窒息盤面。火力を0にせず mask/flag で渡す")
    c += _mask_cols("mk_board", g)
    for n in IND_NAMES:
        raw = n in RAW_COUNT_INDICATORS
        c += _side(f"ind_{n}", "f4", "個" if raw else "0-1", "none(D parity; 生個数)" if raw else "0-1(既存指標)",
                   "mk_ind", "av_board", "ind", f"既存静止指標 {n} (R1s の D 列の元)")
    c += _mask_cols("mk_ind", "ind")
    return c


def _search_cols() -> list[ColumnSpec]:
    c: list[ColumnSpec] = []
    for k in K_LEVELS:
        c += _side(f"k{k}_ojama", "f8", "個", "none", "mk_k", "av_board", "kfire",
                   f"B3専用探索: 理想ツモ総{k + 2}組以内の最大発火量(おじゃま個数, 因果4色/NEXT既知)")
        c += _side(f"k{k}_nf", "f4", "0-1", f"z/(a+z) a={NONSAT_A_DEFAULT}", "mk_k", "av_board", "kfire", f"k{k} の非飽和正規化")
    c += _side("search_done_h", "i1", "手", "none", "mk_k", "av_board", "kfire", "探索済み総手数 (打切り記録)")
    c += _mask_cols("mk_k", "kfire")
    c += _side("queue_age", "f4", "秒", "none", "mk_queue", "av_queue", "queue", "NEXT既知情報の年齢")
    c += _mask_cols("mk_queue", "queue")
    c += _side("score", "f8", "点", "none", "mk_score", "av_frame", "score", "画面得点 (非null時)")
    c += _mask_cols("mk_score", "score")
    return c


def _icon_cols() -> list[ColumnSpec]:
    c: list[ColumnSpec] = []
    g = "icon"
    c += _side("icon_raw_total", "f8", "個", "none", "mk_icon", "av_icon", g, "[参考列] 画面アイコン読みの総数 (生値・最新フレーム)")
    c += _side("icon_corr_total", "f8", "個", "none", "mk_icon", "av_icon", g,
               "[照合用] 補正済み: 試合中ゲート+時間窓中央値(過大誤読抑制)+大額の上限整合")
    for t in ICON_TYPES:
        c += _side(f"icon_raw_{t}", "f4", "個", "none", "mk_icon", "av_icon", g, f"{t} アイコン個数 (生値)")
        c += _side(f"icon_corr_{t}", "f4", "個", "none", "mk_icon", "av_icon", g, f"{t} アイコン個数 (補正値)")
    c += _side("icon_corr_nf", "f4", "0-1", f"z/(a+z) a={NONSAT_A_DEFAULT}", "mk_icon", "av_icon", g, "補正済み総数の非飽和正規化")
    c += _side("icon_big_clipped", "i1", "0/1", "none", "mk_icon", "av_icon", g, "大額(星/月/王冠)を相手得点由来の上限で切った")
    c += _side("icon_ub", "f8", "個", "none", "mk_icon", "av_icon", g, "相手得点由来の予告上限 (整合チェック用)")
    c += _side("icon_omitted", "i1", "0/1", "none", "mk_icon", "av_icon", g,
               "[照合用] 表示が省略されている疑い (アイコン数が表示上限6に達している)")
    c += _mask_cols("mk_icon", g)
    c += _score_ledger_cols()
    for stem, unit, desc in (("land_icon_drop", "個", "直近着弾: 既定は同一OJAMA_FALL区間の補正アイコン減少。B3_LEDGER_LANDING_DEFAULT時は台帳の観測着弾控除量（名前は互換維持）"),
                             ("land_board_net", "個", "直近着弾: 盤面おじゃま純増(空きで頭打ちの下限値)"),
                             ("land_age", "秒", "直近着弾からの経過")):
        c += _side(stem, "f8", unit, "none", "mk_land", "frame", "land", desc)
    c += _side("land_count", "i4", "回", "none", "mk_land", "frame", "land", "この試合の着弾区間数")
    c += _mask_cols("mk_land", "land")
    return c


def _score_ledger_cols() -> list[ColumnSpec]:
    """得点ベース台帳 v2 (src/b3_score_ledger.py, 別エージェント作成) の出力。主入力。未完成の間は全行 mask。"""
    c: list[ColumnSpec] = []
    g = "score_ledger"
    for stem, desc in (("sl_pending", "予告おじゃま (未決済, 相殺後)"), ("sl_generated", "累計生成量 (その側が送った)"),
                       ("sl_offset", "累計相殺量"), ("sl_dropped", "累計着弾量 (その側へ落ちた)"),
                       ("sl_carry", "得点の70点刻み端数繰越"),
                       ("sl_observed_landing", "累計 観測着弾 (受け側確定盤面のおじゃま増分のうち予告を差し引いた分。盤面確定まで1-3秒遅れて反映)"),
                       ("sl_board_increase", "累計 盤面おじゃま増分 (観測、予告で説明できない分も含む)"),
                       ("sl_board_excess", "累計 予告を超えた盤面増分 (予告に無い着弾=台帳の取りこぼし/頭打ちの手掛かり)")):
        c += _side(stem, "f8", "点" if stem == "sl_carry" else "個", "none", "mk_sl", "av_ledger", g,
                   f"[主入力] 得点ベース台帳v2: {desc}")
    c += _side("sl_pending_nf", "f4", "0-1", f"z/(a+z) a={NONSAT_A_DEFAULT}", "mk_sl", "av_ledger", g, "sl_pending の非飽和正規化")
    c.append(_common("sl_rate", "f4", "点/個", "none", "mk_sl_c", "av_ledger", g, "[主入力] 台帳v2の換算レート (マージンタイム込み)"))
    c.append(_common("mk_sl_c", "i1", "code", "none", "", "", g, "sl_rate の欠測理由", "audit"))
    c.append(_common("av_ledger", "f8", "秒", "none", "", "", "meta", "台帳v2 snapshot の available_at", "audit"))
    c += _mask_cols("mk_sl", g)
    return c


def _ledger_cols() -> list[ColumnSpec]:
    """会計台帳 (比較基準Rの再現専用。特徴にも教師にも使わない=統括 2026-10-05)。"""
    c: list[ColumnSpec] = []
    for stem, desc in (("r_ledger_pending", "台帳の予告(未決済)"), ("r_ledger_forecast", "台帳の予告 forecast")):
        c += _side(stem, "f8", "個", "none", "", "frame", "r_ledger", desc + " [R再現専用]", "R_only")
    c.append(ColumnSpec("r_ledger_net_balance", Kind.DIFF_RAW, "f8", "個", "none", "", "frame", "r_ledger", "",
                        "台帳 net_balance_capped (1P視点符号) [R再現専用]", "R_only"))
    return c


def _chain_cols() -> list[ColumnSpec]:
    c: list[ColumnSpec] = []
    g = "chain"
    spec = (("chain_active", "i1", "0/1", "連鎖進行中(fire_start受理後で終了合図前)"),
            ("chain_n_steps", "i4", "段", "観測済み式の段数"),
            ("chain_pred_l", "f4", "段", "観測した式の段と整合する最大段数候補の総段数。不整合時は欠測"),
            ("chain_rem_stages", "f4", "段", "残り段数 max(L-観測段,0)"),
            ("chain_rem_sec", "f4", "秒", "残り時間 (段時間表+端点補正; 終了合図で0)"),
            ("chain_rem_sec_table", "f4", "秒", "残り時間 (連鎖数別実測テーブル版)"),
            ("chain_age", "f4", "秒", "fire_start の trigger からの経過"),
            ("chain_pred_score", "f8", "点", "観測済み得点+最大段数の同一候補の残り得点。不整合時は欠測"),
            ("chain_pred_ojama", "f8", "個", "予測完走おじゃま個数 (現在レート)"),
            ("endsig_stable", "i1", "0/1", "終了合図: CHAIN解除後に2フレーム継続"),
            ("endsig_next", "i1", "0/1", "終了合図: 連鎖後のNEXT移動 (STABLE/TSUMO_FALL中のslide)"),
            ("endsig_ojama", "i1", "0/1", "終了合図: 連鎖側へおじゃま落下(OJAMA_FALL)"))
    for stem, dt, unit, desc in spec:
        c += _side(stem, dt, unit, "none", "mk_chain", "av_fire", g, desc)
    c += _mask_cols("mk_chain", g)
    for stem, dtype, desc in (("chain_pred_final", "str", "同じ最大段数候補の完走盤面JSON。予測不能はnull"),
                              ("chain_pred_reason", "str", "予測不能理由。整合候補ありは空文字"),
                              ("chain_observed_score", "f8", "観測済み式の積の累積得点"),
                              ("av_chain", "f8", "予測に使用した最新観測のavailable_at")):
        c += _side(stem, dtype, "none", "none", "", "av_chain", g, desc, "audit")
    return c


def _episode_cols() -> list[ColumnSpec]:
    c: list[ColumnSpec] = []
    spec = (("ep_closed_n", "i4", "回", "mk_ep", "閉じた発火エピソード数 (CHAIN復帰なし1秒で閉じる)"),
            ("ep_age", "f4", "秒", "mk_ep", "直近の閉じたエピソードからの経過"),
            ("ep_denominator", "f4", "個", "mk_ep_b", "盤面差分と同じ最初のpre盤面の色ぷよ数。pre不在時は起点盤面"),
            ("ep_steps", "f4", "段", "mk_ep_f", "式の段数合計 (完全なものだけ)"),
            ("ep_consumed_formula", "f4", "個", "mk_ep_f", "式由来の消費色ぷよ数 (段完全時のみ)"),
            ("ep_ratio_formula", "f4", "0-1", "mk_ep_f", "消費割合(式)"),
            ("ep_consumed_board", "f4", "個", "mk_ep_b", "盤面差分由来の消費数 (閉じた後のみ)"),
            ("ep_ratio_board", "f4", "0-1", "mk_ep_b", "消費割合(盤面差分)"),
            ("ep_honsen60_formula", "f4", "0/1", "mk_ep_f", "本線: 消費>=60% (式)。NaN=不明層"),
            ("ep_honsen60_board", "f4", "0/1", "mk_ep_b", "本線: 消費>=60% (盤面差分)"),
            ("ep_saisoku_formula", "f4", "0/1", "mk_ep_f", "催促: 消費<60%かつ送り>4個 (式)"),
            ("ep_chain7plus", "f4", "0/1", "mk_ep_f", "7連鎖以上完走 (式の段数>=7)。<7かつ不完全は不明"),
            ("ep_sent_ojama", "f8", "個", "mk_ep_f", "式の得点合計のおじゃま換算"),
            ("pf_available", "i1", "0/1", "", "直近発火で過去リング由来の発火前盤面があるか"),
            ("pf_color", "f4", "個", "mk_pf", "発火前盤面(過去確定)の色ぷよ数"),
            ("pf_before_color", "f4", "個", "mk_pf", "発火開始時盤面の色ぷよ数"))
    for stem, dt, unit, mk, desc in spec:
        c += _side(stem, dt, unit, "none", mk, "av_fire", "episode", desc)
    for m in ("mk_ep", "mk_ep_f", "mk_ep_b", "mk_pf"):
        c += _mask_cols(m, "episode")
    return c


def _ta_cols() -> list[ColumnSpec]:
    """時間軸の打ち返し。語幹の側 = 攻撃側 (a)。受け側 b は反対側。"""
    c: list[ColumnSpec] = []
    g = "time_axis"
    plain = (("ta_active", "i1", "0/1", "攻撃側aが連鎖中(予測完走あり)"),
             ("ta_a_rem_sec", "f4", "秒", "aの残り連鎖時間"),
             ("ta_b_busy", "f4", "秒", "受け側bの操作不能残り (自連鎖中)"),
             ("ta_window", "f4", "秒", "bが置ける窓 max(a残-b操作不能,0)"),
             ("ta_hand_med_sec", "f4", "秒", "1手時間 中央値版 (高さ別表)"),
             ("ta_hand_fast_sec", "f4", "秒", "1手時間 下位10%版 (高さ別表)"),
             ("ta_H_med", "f4", "手", "置ける総手数 中央値版 floor(窓/d)+1"),
             ("ta_H_fast", "f4", "手", "置ける総手数 下位10%版"),
             ("ta_req_h", "f4", "手", "探索に要求した総手数"),
             ("ta_done_h", "f4", "手", "探索済み総手数"),
             ("ta_trunc", "i1", "0/1", "要求Hが探索上限で打切られた"),
             ("ta_q_med", "f8", "個", "B3専用探索: bのH_med組以内の最大発火量 (理想ツモ, おじゃま個数)"),
             ("ta_q_fast", "f8", "個", "B3専用探索: bのH_fast組以内の最大発火量"),
             ("ta_i_pred", "f8", "個", "bの受け量: aの予測完走おじゃま"),
             ("ta_i_sl", "f8", "個", "bの受け量: 台帳v2の予告 sl_pending_b (主入力。v2定義確定後に in-flight 連鎖の含有を再確認)"),
             ("ta_i_icon", "f8", "個", "[照合用] bの画面アイコン補正値+aの予測完走"))
    for stem, dt, unit, desc in plain:
        c += _side(stem, dt, unit, "none", "mk_ta", "av_fire", g, desc)
    c += _side("ta_q_med_nf", "f4", "0-1", f"z/(a+z) a={NONSAT_A_DEFAULT}", "mk_ta", "av_fire", g, "Q 中央値版の正規化")
    c += _side("ta_q_fast_nf", "f4", "0-1", f"z/(a+z) a={NONSAT_A_DEFAULT}", "mk_ta", "av_fire", g, "Q 下位10%版の正規化")
    for v in TA_VARIANTS:
        c += _side(f"ta_m_{v}", "f8", "個", "none", "mk_ta", "av_fire", g, f"余地 M=Q-I ({v})")
        c += _side(f"ta_mg_{v}", "f4", "0-1", f"0.5+0.5*sign*z/(a+z) a={NONSAT_A_DEFAULT}", "mk_ta", "av_fire", g, f"M の符号付き非飽和 ({v}); M=0→0.5")
        c += _side(f"ta_recv_{v}", "f8", "個", "none", "mk_ta", "av_fire", g, f"bの残受け max(I-Q,0) ({v})")
        c += _side(f"ta_send_{v}", "f8", "個", "none", "mk_ta", "av_fire", g, f"bの残送り max(Q-I,0) ({v})")
        c += _side(f"ta_unc72_{v}", "i1", "0/1", "none", "mk_ta", "av_fire", g, f"相殺不能の予告 残受け>=72個 ({v}) [特徴のみ・死亡証明には使わない]")
    c += _mask_cols("mk_ta", g)
    for stem in ("ta2_q_med", "ta2_m_med"):
        c += _side(stem, "f8", "個", "none", "mk_ta2", "av_fire", "time_axis2", f"一往復 a→b→a の {stem} [v1 未実装・全行mask=DEFERRED]")
    c += _mask_cols("mk_ta2", "time_axis2")
    return c


def _diff_cols() -> list[ColumnSpec]:
    c: list[ColumnSpec] = []
    for m in ("mk_df_ind", "mk_df_k", "mk_df_board", "mk_df_icon"):
        c.append(_common(m, "i1", "code", "none", "", "", "diff", f"{m}: 差分列の欠測理由 = 両側 mask の最大値 (側交換で不変)", "audit"))

    def add(name: str, kind: Kind, desc: str, mask: str) -> None:
        c.append(ColumnSpec(name, kind, "f4", "0-1" if kind is Kind.DIFF_CENTERED else "個", "0.5+0.5*(p1-p2)" if kind is Kind.DIFF_CENTERED else "none",
                            mask, "", "diff", "", desc))
    for n in IND_NAMES:
        if n in RAW_COUNT_INDICATORS:
            add(f"df_ind_{n}", Kind.DIFF_RAW, f"{n} の差 p1-p2", "mk_df_ind")
        else:
            add(f"df_ind_{n}", Kind.DIFF_CENTERED, f"{n} の中心化差分", "mk_df_ind")
    for k in K_LEVELS:
        add(f"df_k{k}_nf", Kind.DIFF_CENTERED, f"k{k} 正規化の中心化差分", "mk_df_k")
    add("df_cells_color", Kind.DIFF_RAW, "色ぷよ数差", "mk_df_board")
    add("df_cells_ojama", Kind.DIFF_RAW, "盤面おじゃま差", "mk_df_board")
    add("df_icon_corr_total", Kind.DIFF_RAW, "補正アイコン総数の差", "mk_df_icon")
    return c


def _extra_search_cols() -> list[ColumnSpec]:
    """EXTRA末尾追記。絶対量はSIDE_ABSで交換し、符号反転しない。"""
    c: list[ColumnSpec] = []
    for prefix, mask, avail in (("search", "mk_k", "av_board"), ("ta", "mk_ta", "av_fire")):
        trunc_mask = f"mk_{prefix}_trunc"
        c += _side(f"{prefix}_trunc_empty", "f4", "マス", "none", trunc_mask, avail, "search_extra",
                   "要求H未達の上限打切り時、保持ビームの最大空き(row0込み78マス)。非打切りはNaN", "audit")
        c += _side(f"{prefix}_trunc_empty_nf", "f4", "0-1", "空き/78", trunc_mask, avail, "search_extra",
                   "打切り時空きの正規化。非打切りはNaN", "audit")
        c += _side(f"{prefix}_full", "f4", "0/1", "0-1", mask, avail, "search_extra",
                   "要求H以内に保持ビームの全候補で合法な次手が無く終了(満杯)。打切りと区別", "audit")
        c += _mask_cols(trunc_mask, "search_extra")
    return c


def build_column_specs() -> list[ColumnSpec]:
    """全列を決定的な順で返す (名前重複は禁止)。"""
    cols = (_meta_cols() + _common_cols() + _board_cols() + _search_cols() + _icon_cols()
            + _ledger_cols() + _chain_cols() + _episode_cols() + _ta_cols() + _diff_cols() + _extra_search_cols())
    cols.append(ColumnSpec("search_version", Kind.META, "str", "version", "none", "", "frame",
                           "meta", "", "B3専用2個組・潜在火力探索の版", "meta"))
    cols = [replace(x, role="reference_only") if x.group == "icon" and x.role == "feature" else x for x in cols]
    cols += _side("ta_bound", "i1", "0/1", "0-1", "", "av_fire", "time_axis",
                  "予測不能時の楽観上限補完=1。側は攻撃側、Qは反対側の返せる量。側交換のみ、反転なし")
    cols += _side("ep_denominator_formula", "f4", "個", "none", "mk_ep_f", "av_fire", "episode",
                  "式全段の消去数＋同じ完走候補の残存色数（発火補完込み）。候補終端不在は起点個数")
    names = [x.name for x in cols]
    if len(set(names)) != len(names):
        raise ValueError("列名が重複している")
    return cols


COLUMN_SPECS: list[ColumnSpec] = build_column_specs()
SPEC_BY_NAME: dict[str, ColumnSpec] = {c.name: c for c in COLUMN_SPECS}
COLUMN_NAMES: list[str] = [c.name for c in COLUMN_SPECS]
# B3 第1版: 全動画推定の下位10%時間表に依存する列は保存だけ行う（追記D）。
EXCLUDED_FAST_COLUMNS: tuple[str, ...] = tuple(
    c.name for c in COLUMN_SPECS if c.group == "time_axis" and (
        "_fast" in c.name or c.name in {
            f"{stem}_{side}" for stem in ("ta_req_h", "ta_done_h", "ta_trunc") for side in SIDES}))


def validate_spec() -> None:
    """相手列・mask列の整合を検査する (schema/mask 整合)。"""
    for c in COLUMN_SPECS:
        if c.kind is Kind.SIDE_ABS:
            partner = SPEC_BY_NAME.get(c.counterpart)
            if partner is None or partner.counterpart != c.name or partner.dtype != c.dtype:
                raise ValueError(f"{c.name}: 側交換の相手列が不整合")
        if c.mask and c.mask not in SPEC_BY_NAME:
            raise ValueError(f"{c.name}: mask列 {c.mask} が台帳に無い")


def swap_sides(row: Mapping[str, object]) -> dict[str, object]:
    """台帳の種類に従って 1P<->2P を交換した行を返す (2回適用で元に戻る)。

    META/COMMON/WEIGHT は不変 (orientation と row_key は呼出側 mirror_row が更新)。
    """
    out = dict(row)
    for c in COLUMN_SPECS:
        if c.name not in row:
            continue
        v = row[c.name]
        if c.kind is Kind.SIDE_ABS:
            out[c.name] = row[c.counterpart]
        elif c.kind is Kind.DIFF_RAW:
            out[c.name] = -v if _num(v) else v
        elif c.kind in (Kind.DIFF_CENTERED, Kind.LABEL):
            out[c.name] = 1.0 - v if _num(v) else v
    return out


def _num(v: object) -> bool:
    return isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)


def render_markdown() -> str:
    """docs/B3_COLUMN_SPEC.md の本文を台帳から生成する。"""
    lines = [f"# B3 列仕様台帳 ({SCHEMA_VERSION})", "",
             "自動生成: `python -m scripts.b3_render_column_spec` (手編集しない。tests/test_b3_columns.py が一致を検査)。", "",
             f"総列数: {len(COLUMN_SPECS)}。非飽和正規化 `nonsat(x,a)=z/(a+z), z=log1p(x)`, a 既定 {NONSAT_A_DEFAULT} (学習foldで決定、生値は別列)。", "",
             "## 欠測理由コード (mk_* 列)", ""]
    lines += [f"- {m.value}: {m.name}" for m in Mask]
    lines += ["", "認識矛盾(10): 負の消費断片・材料超過は欠測。5色盤面は色依存指標・探索・時間軸を欠測とし、個数・形状は保持する。",
              "得点の減少観測は直前の採用値を保持しmask=10、台帳の増分基準も下げない。",
              "履歴不完全(11): 開始得点から履歴欠落が判明した試合は、両側のsl_pending/正規化値・sl_board_excessを欠測としmk_sl=11。時間軸の台帳予告と派生余力も欠測。観測開始後の累計は診断用に保持し、試合リセットまで解除しない。初回得点採用待ちはmk_sl=3。",
              "ta_bound=1のQは残り色ぷよ＋追加ツモの算術上限。ビーム探索値（発火前1発火値を含む）は上限に使わない。連鎖中の段番号不確実時は最大係数、起点欠落時は全78セルを使う。",
              "消費割合は式用/盤面用の分母を分離し、クリップしない。pf_before_colorは元の起点個数を保持。",
              "レートはscoring.compute_effective_rateへ統一し、毎回切り捨て（70→52→39→29→21→15→11→8→6→4→3→2→1）。"]
    lines += ["", "## 側交換 (kind) の規則", "",
              "- side_abs: p1/p2 の列を入れ替える (相手列=counterpart)", "- diff_raw: 符号反転",
              "- diff_centered: 1-x", "- label: won_p1 -> 1-won_p1", "- meta/common/weight: 不変 (位相・共通時刻・重み)", "",
              "## 予告の扱い (統括決定 2026-10-05)", "",
              "- 主入力 = 得点ベース台帳v2 (`sl_*`, src/b3_score_ledger.py の出力。未完成の間は全行 mask)。画面アイコン (`icon_*`) は台帳v2第4版では一切使われない参考列 (role=reference_only)。旧台帳 (lean の ojama_forecast 等) は特徴にも教師にも使わず `r_ledger_*` (role=R_only) に比較基準Rの再現専用で残す。", "",
              "## 列一覧", "", "| 列 | kind | dtype | 単位 | 正規化 | mask列 | available_at | 相手列 | role | 説明 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in COLUMN_SPECS:
        lines.append(f"| {c.name} | {c.kind.value} | {c.dtype} | {c.unit} | {c.norm} | {c.mask} | {c.avail} | {c.counterpart} | {c.role} | {c.desc} |")
    return "\n".join(lines) + "\n"


validate_spec()

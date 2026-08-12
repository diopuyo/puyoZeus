"""本番構成 (採用済みフラグ) の単一の情報源 (2026-08-08).

## なぜ必要か — 退行の真因
本プロジェクトは backwards compat のため **改善を必ず「フラグ追加 + 既定 OFF」で
入れる**規約になっている (CLAUDE.md)。 この規約自体は再現性のために正しい。
問題は **採用が決まった後に「どのフラグを付けるのが正解か」が一元化されていない**
ことだった。 正解がジョブファイルや個別スクリプトに直書きで散在するため、
新しくデモや評価を作るたびに手でフラグを並べることになり、 **過去の改善が
まるごと抜け落ちる**。

実際に 2026-08-08 のデモ生成で `--early-fire-reaction` を付け忘れ、
「連鎖中に連鎖力を判断する機能がなくなった」「有利不利が大雑把にしか動かない」
という退行が起きた (user 指摘)。 機能は 2026-07-29 の user レビュー指摘に
対応して実装済みだったが、 既定 OFF のまま誰も付けなければ存在しないのと同じ
だった。

## 使い方
    from src.production_config import advantage_overlay_flags
    cmd += " " + advantage_overlay_flags()

## 運用ルール
- **採用が決まったフラグは必ずここに追記する**。 追記しない限り「採用済み」とは
  見なさない。
- 各エントリに **採用日と根拠** (計測結果・レビュー) を必ず書く。 後から
  「なぜこれが有効なのか」を辿れるようにするため。
- 既定 OFF のまま残すフラグ (実験中・未採用) はここに入れない。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdoptedFlag:
    """採用済みフラグ 1 件。"""

    flag: str          # CLI フラグ文字列 (値を取る場合は "--name value" 形式)
    adopted: str       # 採用日 (YYYY-MM-DD)
    reason: str        # 採用根拠 (計測結果 / レビュー)


# ============================
# 認識 — 収集専用 (collect_boards_lean のみが必要とするフラグ)
# ============================
# RecognitionPipeline.load_default では既定 True だが collect_boards_lean だけ
# 既定 False のため、 収集時のみ明示指定が要る。 表示系スクリプトに渡すと
# 「unrecognized arguments」で落ちるので分けて管理する
# (2026-08-08 に実際に踏んだ)。
COLLECT_ONLY_ADOPTED: tuple[AdoptedFlag, ...] = (
    AdoptedFlag(
        "--enable-chain-tracker", "2026-07-30",
        "機能D単独では CHAIN 検知が実運用 0 件で、連鎖中の盤面凍結が働かない",
    ),
)

# ============================
# 認識 — 共通 (collect_boards_lean / visualize_recognition の両方が受け付ける)
# ============================
# Phase L の全動画 regen で実際に使っている構成と一致させること。
RECOGNITION_ADOPTED: tuple[AdoptedFlag, ...] = (
    AdoptedFlag(
        "--enable-effect-gate", "2026-08-03",
        "全消し/カットイン演出中の誤読を抑止",
    ),
    AdoptedFlag(
        "--enable-burst-guard-v2", "2026-08-05",
        "バーストガード再設計 Stage1。誤り 93 セル -> 33 セル (user条件付き承諾)",
    ),
    AdoptedFlag(
        "--enable-transition-merge-guard", "2026-08-05",
        "Stage1.5。NON-STABLE->STABLE 遷移 merge の物理的期待値フィルタ",
    ),
    AdoptedFlag(
        "--burst-gate-open-threshold 0.954", "2026-08-05",
        "緊急較正。factorial バックテストで決定した閾値",
    ),
    AdoptedFlag(
        "--enable-hidden-row-burst-guard", "2026-08-05",
        "Stage1.5b (§11)",
    ),
    AdoptedFlag(
        "--enable-match-transition-debounce", "2026-08-06",
        "長時間劣化修正 A'。Phase I 合格構成に含まれる",
    ),
)

# ============================
# 有利不利オーバーレイ (visualize_advantage_overlay)
# ============================
ADVANTAGE_ADOPTED: tuple[AdoptedFlag, ...] = (
    AdoptedFlag(
        "--early-fire-reaction", "2026-07-29",
        "user レビュー指摘1/2 対処。発火フレームで即座に速報を反映する。"
        "無効だと両者 STABLE まで判定が凍結し、大連鎖中に有利不利が逆転する "
        "(2026-08-08 実測: 無効=2P有利24/77% -> 有効=互角52% -> 1P有利60%)",
    ),
    # --platt-calibration は 2026-08-12 user承認で撤回 (採用 2026-08-04)。
    # 根拠: 自信過剰 (80%表示→実勝率64%) の真因は対称化バグで B-1 修正済み
    # (8/11)。修正後は素の出力が ECE 0.0125 で最良、旧モデル向け補正の適用は
    # 逆に歪める (逆効果を実測)。148 本再学習後に新たな自信過剰が出た場合は
    # 新モデルのデータで較正を作り直して再導入する (旧補正の使い回し禁止)。
    AdoptedFlag(
        "--per-side-settled", "2026-08-09",
        "片側でも STABLE なら再計算する。従来の両者同時 STABLE ゲートは実測で "
        "試合時間の 72.3%・最長 13.97 秒 評価を凍結させ、1P が撃ち切って空・"
        "2P が窒息寸前でも「互角54%」のままだった。有効化で t=66 が "
        "1P有利72 (勝率97%) になり主因も窒息余裕差に変わった",
    ),
    AdoptedFlag(
        "--no-score-lead-bias", "2026-08-09",
        "得点タイブレークを外す。user伝授「スコアはおじゃまを送る手段で、"
        "送った時点で意味を失う」。送ったぶんは予告/盤面で既に観測できるため"
        "二重計上になる。t=29 が 2P有利80%% -> 69%% に改善",
    ),
    AdoptedFlag(
        "--no-pressure", "2026-08-09",
        "圧力成分を外す。おじゃまを個数で数える設計だったが、user伝授の通り"
        "評価すべきは盤面能力の低下。外しても判定はほぼ変わらず(38%%->40%%)、"
        "情報が他成分と重複していたことが実測で確認された",
    ),
    AdoptedFlag(
        "--sample-interval 0", "2026-07-13",
        "毎フレーム更新。0.5 秒間引きだとおじゃま会計がスコア変化・連鎖終了を"
        "取りこぼし net/forecast=0 になる (ADVANTAGE_OVERLAY_2026-07-13 §2-3)",
    ),
)

# ============================
# 表示 (visualize_recognition の overlay 系)
# ============================
VISUALIZATION_ADOPTED: tuple[AdoptedFlag, ...] = (
    AdoptedFlag(
        "--chain-formula-simulate-verify", "2026-08-08",
        "無効だと連鎖数が固定 1 になる。有効で実測値 (9連鎖なら 9) が出る",
    ),
    AdoptedFlag(
        "--overlay-chain-hold-until-end", "2026-08-08",
        "user 要望「連鎖中はずっと chain であってほしい」。"
        "連鎖中の異常な離脱 20 回 -> 0 回",
    ),
)


# ============================
# 連鎖シミュレーション — 物理ルール採用 (CLI フラグでなく Python 既定値の
# 単一情報源。src.chain.ChainSimulator(exclude_hidden_row_from_pop=...) の
# 呼び出し側は本フラグを import して使うこと。個別に True を書き散らさない)
# ============================
CHAIN_SIM_ADOPTED: tuple[AdoptedFlag, ...] = (
    AdoptedFlag(
        "exclude_hidden_row_from_pop=True", "2026-08-10",
        "幽霊連鎖ルール (13段目/隠し段のぷよは4つ繋がっても消えない、"
        "user伝授2026-08-09、実装・単体テストはコミット991fa80で完了済)。"
        "全域バックテスト (boards_lean_phase_l_2026-08-07 148動画) の結果を"
        "docs/verify 配下に記録。詳細は data/verify/ghost_chain_backtest_2026-08-10/",
    ),
)

# 上記フラグの真偽値そのもの。本番構築箇所はこれを import して
# `ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)` の形で使う。
GHOST_CHAIN_RULE_ENABLED: bool = True


# ============================
# 有利不利オーバーレイ — 主因表示の除外リスト (2026-08-11、ロードマップ Phase 1-3)
# ============================
# 背景: scripts/visualize_advantage_overlay.py の `_score_advantage()` が
# 組み立てる「主因」欄は、単に **|差分| の大きい順** で上位3件を選んでいるだけで、
# 実際にモデルの予測 (p1/adv) へどれだけ寄与しているか (=真の attribution、
# scripts/_diag_adv_attribution_2026-08-09.py の `_attribution()` が計算する
# ablation ベースの寄与) は一切見ていない。 そのため **勝敗と無相関な指標**でも、
# たまたま差分の絶対値が大きいと主因1位に選ばれてしまう。
#
# 以下は 2026-08-09/11 の層別 AUC 実測 (labeled_win_combined66.csv、66動画・
# 73,416 ペア、全体/おじゃまフラット/おじゃま差大の3層。
# scripts/_verify_efficiency_vs_material_2026-08-09.py +
# scripts/_verify_attribution_exclusion_2026-08-11.py) で「主因として表示する
# 根拠が無い」と確定した指標。
#
# **重要: モデルの特徴量からは外さない** (予測 adv/p1 には一切影響しない、
# 表示だけの修正)。 特徴量そのものの除去は別途「全域無悪化ゲート」を伴う
# 検証が要るためスコープ外 (feedback_overfitting_awareness_2026-08-04)。
ATTRIBUTION_EXCLUDED_INDICATORS: tuple[str, ...] = (
    # AUC 0.5000 (全体/おじゃまフラット/おじゃま差大の全層で寸分違わず一致)。
    # 根本原因は学習データ labeled_win_combined66.csv 上で本列が 100% NaN
    # (未収集) であること (2026-08-11 実測、193,623 行全て NaN)。学習された
    # 重みは実質ゼロだが、推論時 (_fill_expected_fire_candidate) は実盤面から
    # 都度計算するため差分値自体は大きくなり得て、「差分が大きい順」の現行
    # ロジックだと無情報にもかかわらず主因1位に出る (2026-08-09 デモ実測で
    # 「期待火力K1差 +0.64」が主因1位表示された事例)。
    "expected_fire_k1",
    "expected_fire_k2",  # 同上 (AUC 0.5000、全層一致、学習データ上も100% NaN)
    # 過去に「信頼不可」と判定済み (memory
    # project_saturation_ceiling_untrustworthy_2026-07-22: 「理想ツモ天井は
    # 空き空間量と同じで無相関」)。 加えて 2026-08-11 再検証で学習データ上
    # current_max_chain (現在最大連鎖) と 100% 完全一致 (193,623 行、
    # 平均/分散/AUC 全て同値) と判明 — 「現在最大連鎖」と同じ信号を
    # 「飽和連鎖量」という別名で二重表示しているだけで独立情報が無い。
    "saturated_chain_count",
)


def _join(flags: tuple[AdoptedFlag, ...]) -> str:
    """フラグ文字列を空白区切りで連結する。"""
    return " ".join(f.flag for f in flags)


def recognition_flags() -> str:
    """認識の本番構成フラグを返す (収集・表示の両方が受け付けるもの)。"""
    return _join(RECOGNITION_ADOPTED)


def collect_flags() -> str:
    """collect_boards_lean 用の全フラグ (共通 + 収集専用) を返す。"""
    return _join(RECOGNITION_ADOPTED + COLLECT_ONLY_ADOPTED)


def advantage_overlay_flags() -> str:
    """有利不利オーバーレイの本番構成フラグを返す。"""
    return _join(ADVANTAGE_ADOPTED)


def visualization_flags() -> str:
    """認識オーバーレイ表示の本番構成フラグを返す。"""
    return _join(VISUALIZATION_ADOPTED)


def describe() -> str:
    """採用済みフラグの一覧を人が読める形で返す (生成物への記録用)。"""
    lines: list[str] = []
    for title, flags in (
        ("認識(共通)", RECOGNITION_ADOPTED),
        ("認識(収集専用)", COLLECT_ONLY_ADOPTED),
        ("有利不利", ADVANTAGE_ADOPTED),
        ("表示", VISUALIZATION_ADOPTED),
        ("連鎖シミュレーション", CHAIN_SIM_ADOPTED),
    ):
        lines.append(f"[{title}]")
        for f in flags:
            lines.append(f"  {f.flag}  (採用 {f.adopted}) — {f.reason}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())

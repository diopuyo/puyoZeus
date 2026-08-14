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
    AdoptedFlag(
        "--enable-ojama-fall-placement-override", "2026-08-15",
        "案2修正版 (2026-08-13導入/2026-08-15 evidence一発判定のchain-active"
        "除外修正、コミット9dc5e35、user承認)。OJAMA_FALL滞在中の場面1振動 "
        "(0.15-0.3秒周期でOJAMA_FALL<->STABLE往復、docs/DEMO_REVIEW_2026-08-13.md) "
        "を根治する。物差しv2 (55盤面、既存人手ラベル): 94.08%→95.63%、新規劣化"
        "46→25セル (5盤面。内訳: 48%=W9系一過性ドロップアウトへの偶発的巻き込み"
        "でplacement_override自体のロジック誤りではない、40%=slide_motion経路"
        "がヒステリシス対象外のまま残る既知の残存リスク、12%=既存の色誤認ノイズ"
        "で無関係、data/verify/yardstick_v2_2026-08-14/diag_c1p/review_sheet.html "
        "参照)。scene1再現チェック (logs/diag_scene1_oscillation_recheck_2026-08-15.json)"
        "で往復振動0件を個別確認済み (旧15回/秒規模から根絶)。"
        "拡張代表サンプル全域バックテスト (2026-08-15 user承認、フルサイズでなく"
        "序盤/中盤/終盤 各2分×16動画中15本有効 [video_c15/c19はファイル破損で"
        "全チャンク0件、無関係な既知データ品質問題]、2構成、"
        "data/verify/backtest_placement_override_2026-08-15/summary.md): "
        "OJAMA_FALL滞在時間 中央値0.300→0.133秒・p95 1.5→0.933秒 (短縮、狙い通り "
        "= 過去の「全盤面ぷよ数静止待ち」による張り付き時間の縮小)。盤面churn "
        "(隣接STABLE間セル変化量、中央値2.0で同一)・幻連鎖疑い件数 (266→265) は "
        "悪化なし。品質ゲートPASS/FAIL比率はほぼ同水準 (42チャンク中PASS17→20/"
        "FAIL14→15)。**残課題 (悪化側、要監視)**: (a) 往復振動0.35秒未満のカウント"
        "自体は増加 (224+241→432+564、突入数あたり率0.186→0.318) — 直接調査の"
        "結果、大連鎖直後の複数波おじゃま降下 (STABLE挟みつつ短時間で再度降下する"
        "正規の物理パターン) を巻き込んでいるためと判明 (実フレーム系列確認済み、"
        "場面1の単発往復振動パターンとは別種)。(b) 重力違反セル数が39→69に増加"
        "したが、うち30件差分の17件はc22_chunk1の1スナップショットに集中した"
        "外れ値 (単一frame内18セル同時、認識自体の一過性乱れの可能性が高い) で、"
        "除けば39→51 (+31%) 相当。早期STABLE復帰が物理未確定フレームを捉える"
        "リスクとして小さく残る。両残課題とも次回148動画再収集 (本フラグ込み) "
        "でのより大規模な再測定を推奨",
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
    AdoptedFlag(
        "--counter-reach", "2026-08-12",
        "打ち合い応手確率 (モンテカルロ、mc_counter_estimator経由) の正式採用 "
        "(指標大整理提案書 0-4)。三つ巴比較 (案D実データ学習 / 急所3修正シミュ / "
        "併用) で全位相 (序盤/中盤/終盤) 有意勝ち (rho=0.808, AUC=0.837、"
        "memory project_exchange_triple_comparison_results_2026-08-02)。"
        "過去に「採用」とコードコメントのみ先行し、本ファイル未登録・"
        "visualize_advantage_overlay.py の CLI 既定値も OFF のままという"
        "食い違いがあった (0-4 の確認依頼で発覚)。COUNTER_REACH_ENABLED_"
        "BY_DEFAULT=True で CLI 既定値も同時に ON 化する",
    ),
    AdoptedFlag(
        "--normalize-fps-30", "2026-08-12",
        "60fps 動画を stride-2 相当 (実効30fps) に間引く "
        "(src.fps_normalize.resolve_normalize_fps_30_stride、"
        "collect_boards_lean.py が 2026-07-30 から既定採用済みの正規化と"
        "同一関数)。従来オーバーレイのみ全フレーム処理のままで、認識状態機械の"
        "フレーム数定数 (STABLE_RECOVERY_MIN_FRAMES 等、30fps前提でコメント"
        "済み) が実時間半分で発火し STABLE 遷移が23%%過多になる非調整領域で"
        "動いていた (収集・学習データと違う認識意味論の不一致)。"
        "OVERLAY_NORMALIZE_FPS_30_ENABLED_BY_DEFAULT=True で CLI 既定値も"
        "同時に ON 化し、収集側と同じ意味論に揃える",
    ),
    AdoptedFlag(
        "--production-recognition", "2026-08-13",
        "本番採用の認識フラグ群 (RECOGNITION_ADOPTED: effect-gate/"
        "burst-guard-v2/transition-merge-guard/burst-gate-open-threshold "
        "0.954/hidden-row-burst-guard/match-transition-debounce) を "
        "load_default() へ自動適用する (recognition_load_default_kwargs() 経由)。"
        "根因調査 (2026-08-13) の副次発見: 従来 overlay はこれらを一切転送して"
        "おらず、デモ/レビュー動画が本番より劣化した認識で生成されていた "
        "(2026-08-08の--early-fire-reaction付け忘れ事故と同型)。"
        "OVERLAY_PRODUCTION_RECOGNITION_ENABLED_BY_DEFAULT=True で CLI 既定値も"
        "同時に ON 化する",
    ),
    AdoptedFlag(
        "--resize-1080p", "2026-08-13",
        "認識入力を1920x1080へ正規化してから RecognitionPipeline.update() に"
        "渡す (collect_boards_lean.py:1050 と同一の正規化)。根因調査 "
        "(2026-08-13) の副次発見: 従来 overlay は表示キャンバス用サイズ "
        "OUT_W/OUT_H(1280x720) へ直接縮小したフレームをそのまま認識にも渡して"
        "おり、BoardRegion の絶対px座標較正 (1920x1080前提) と不整合だった "
        "(CLAUDE.md「他解像度は1920x1080にリサイズしてから認識する」原則違反)。"
        "OVERLAY_RESIZE_1080P_ENABLED_BY_DEFAULT=True で CLI 既定値も同時に "
        "ON 化する (認識用と表示用のフレームは独立に生成、表示解像度は不変)",
    ),
)

# --counter-reach の CLI 既定値。visualize_advantage_overlay.py の argparse
# default / generate() の関数既定値はここを import して使う
# (CHAIN_SIM_ADOPTED の GHOST_CHAIN_RULE_ENABLED と同じパターン。
# 「採用済みなのに初期値OFF」という食い違いの再発防止、2026-08-12)。
COUNTER_REACH_ENABLED_BY_DEFAULT: bool = True

# --normalize-fps-30 の CLI 既定値/generate() 既定値 (2026-08-12 追加、上記
# ADVANTAGE_ADOPTED エントリ参照)。COUNTER_REACH_ENABLED_BY_DEFAULT と同じ
# パターンで単一情報源化する。
OVERLAY_NORMALIZE_FPS_30_ENABLED_BY_DEFAULT: bool = True

# --production-recognition の CLI 既定値/generate() 既定値 (2026-08-13 追加、
# 上記2定数と同じパターン)。True で RECOGNITION_ADOPTED (本番採用の認識
# フラグ群) を recognition_load_default_kwargs() 経由で自動適用する。
OVERLAY_PRODUCTION_RECOGNITION_ENABLED_BY_DEFAULT: bool = True

# --resize-1080p の CLI 既定値/generate() 既定値 (2026-08-13 追加)。True で
# 認識入力を 1920x1080 へ正規化してから RecognitionPipeline.update() に渡す
# (collect_boards_lean.py と同一正規化、CLAUDE.md「他解像度は1920x1080に
# リサイズしてから認識する」原則)。
OVERLAY_RESIZE_1080P_ENABLED_BY_DEFAULT: bool = True

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
    AdoptedFlag(
        "--production-recognition", "2026-08-13",
        "横展開監査 (docs/CROSS_CUTTING_AUDIT_2026-08-13.md P1) の配線漏れ是正。"
        "visualize_recognition.py が RECOGNITION_ADOPTED (バーストガード等6"
        "フラグ) を明示指定しない限り一切適用しておらず、レビュー動画が本番"
        "より劣化した認識で生成されていた (visualize_advantage_overlay.py の "
        "eacb1f3 と同型の事故)。recognition_load_default_kwargs() 経由で "
        "load_default() へ自動適用する。--no-production-recognition で無効化",
    ),
    AdoptedFlag(
        "--production-visualization", "2026-08-13",
        "同上の是正。上記2フラグ (chain-formula-simulate-verify/"
        "overlay-chain-hold-until-end) 自体を CLI 既定 ON にする配線 "
        "(resolve_production_config_overrides() 経由)。"
        "--no-production-visualization で無効化",
    ),
)

# measure_stable_cell_acc.py (物差し) にも同型の配線漏れがあった (P1)。
# --no-production-recognition の明示指定で過去測定と bit-identical な旧構成
# (各フラグ明示指定必須) を再現できる (物差しの継続性維持)。専用の
# AdoptedFlag タプルは持たず、RECOGNITION_ADOPTED を単一情報源として
# resolve_production_recognition_flags() (同ファイル内) が消費する
# (recognition_load_default_kwargs() と同じ変換ロジック)。


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


# ============================
# 指標大整理 (2026-08-12 user確定、docs/INDICATOR_REORG_PROPOSAL_2026-08-12.md
# 「決定記録」節) — scripts/build_labeled_win_from_npz.py への実装
# ============================
# 実体の定数群 (DIFF_REPLACE_OWN_COLUMNS 等) は同ファイル側に置く
# (npz→CSV変換ツール専用の分類のため)。ここには決定内容の記録のみ残す
# (「採用日+根拠を必須記録」規約、production_config.py が単一情報源)。
INDICATOR_REORG_DECISIONS: tuple[AdoptedFlag, ...] = (
    AdoptedFlag(
        "a-1: *_raw列8種+saturated_chain_count を削除", "2026-08-12",
        "*_raw は score の定数倍で完全重複 (記録・学習の両方から削除)。"
        "saturated_chain_count は current_max_chain と19万場面で完全一致 "
        "(作成時のバグで同じものを2回計算していた、"
        "ATTRIBUTION_EXCLUDED_INDICATORS の根拠と同一)。absorption_capacity "
        "は build_labeled_win_from_npz.py では元々未収集のため対応不要",
    ),
    AdoptedFlag(
        "b-1: center_bulge を center_bulge_color/_ojama に分解", "2026-08-12",
        "合成版は当てやすさ0.509でほぼ無価値だが、分解すると不利の大部分は"
        "おじゃま由来 (影響は色ぷよの12倍) と判明。色ぷよ由来分は小さいが"
        "本物の効果 (おじゃまゼロ33万場面でも検出)。indicators_v2.py の"
        "center_bulge() 本体は backwards compat のため変更なし、"
        "center_bulge_color/_ojama を新規追加",
    ),
    AdoptedFlag(
        "b-2: 相手との差 (diff_) 列への置き換え", "2026-08-12",
        "63動画実測: 11項目中10項目で「自分のみ」より「差」の方が当てやすい "
        "(連結最大サイズは終盤0.508→0.567)。例外2つ (色ぷよ総数・3個連結) は"
        "置き換えず own/diff の使い分けを列ごとに分類 (詳細は"
        "scripts/build_labeled_win_from_npz.py の DIFF_* 定数群)。"
        "色ぷよ総数は user指示8/12によりおじゃま総数とのペア特徴として"
        "own+diff+比率+交互作用の4列で表現 (単純な差では向きが逆転する謎の答え)",
    ),
)


@dataclass(frozen=True)
class RemovedIndicator:
    """死亡確定・削除決定済みの指標 1 件 (REORG_REMOVED_INDICATORS 用)。"""

    name: str        # 指標名 (collect_indicators_v2.INDICATOR_COLUMNS 等の表記に合わせる base 列名)
    confirmed: str    # 死亡・削除の確定日 (YYYY-MM-DD)
    reason: str       # 死因 (実測結果 または重複判定の根拠)


# ============================
# 指標大整理 — 削除台帳 (2026-08-13 新設、横展開監査 P2)
# ============================
# 背景: saturated_chain_count の削除決定 (a-1、INDICATOR_REORG_DECISIONS 参照、
# 2026-08-12) が3箇所中1箇所 (build_labeled_win_from_npz.py) にしか反映されて
# おらず、scripts/visualize_advantage_overlay.py の FEATURE_CANDIDATES /
# scripts/model_indicator_win.py の REDUNDANT_COLS には残ったままだった
# (「片方の経路だけ直して他方が古いまま」の型A事故、
# docs/CROSS_CUTTING_AUDIT_2026-08-13.md P2)。加えて過去に死亡確定したのに
# 公式の除外リストへ一度も登録されていなかった3件 (honsen_output/
# taiou_capacity/disturbance_rejection、docs/INDICATOR_PROPOSAL_ROUND2_
# 2026-08-13.md D節) をここで台帳化する。
#
# 使い方: 学習・モデル特徴量の候補リストを組み立てる箇所は
# `reorg_removed_indicator_names()` の集合を除外候補として参照すること
# (FEATURE_CANDIDATES / REDUNDANT_COLS がこのパターンに追従済み)。
REORG_REMOVED_INDICATORS: tuple[RemovedIndicator, ...] = (
    RemovedIndicator(
        "saturated_chain_count", "2026-08-12",
        "current_max_chain と19万場面で完全一致 (作成時のバグで同じものを"
        "2回計算していた)。a-1決定 (INDICATOR_REORG_DECISIONS 参照、"
        "ATTRIBUTION_EXCLUDED_INDICATORS と同根拠)",
    ),
    RemovedIndicator(
        "absorption_capacity", "2026-08-12",
        "board_puyo_total と完全重複。a-1決定 (INDICATOR_REORG_DECISIONS 参照)。"
        "scripts/model_indicator_win.py の REDUNDANT_COLS では既に先行して"
        "除外済みだった",
    ),
    RemovedIndicator(
        "honsen_output", "2026-07-17",
        "催促・条件1 (本線打ち合い収支)。中盤AUC 0.512 = current_max_chain の"
        "生値0.514と同等で無寄与と確定 (memory "
        "project_midgame_indicator_failures_2026-07-17)",
    ),
    RemovedIndicator(
        "taiou_capacity", "2026-07-20",
        "対応力 (相手催促を本線温存で上回る)。単純な受け容量 (当てやすさ0.52)"
        "に負け、blend不採用と確定 (docs/INDICATOR_CANDIDATES_2026-07-20.md "
        "分類4)",
    ),
    RemovedIndicator(
        "disturbance_rejection", "2026-08-13",
        "外乱除去比 (C4、returned÷incoming お邪魔)。docs/INDICATOR_CANDIDATES_"
        "2026-07-20.md では候補提示のみ (「既存会計から実装ゼロ」) のまま構想"
        "倒れで終わっている。同系統の潰し・相殺設計である ojama_disruption "
        "(催促・条件2) が478行全ゼロの完全失敗 (memory "
        "project_midgame_indicator_failures_2026-07-17) で決着済みのため、"
        "docs/INDICATOR_PROPOSAL_ROUND2_2026-08-13.md D節の判定に従い死亡"
        "確定として登録する (独立実測が無い旨は正直に記録)",
    ),
)


def reorg_removed_indicator_names() -> frozenset[str]:
    """REORG_REMOVED_INDICATORS の指標名集合を返す (除外リスト構築用)。"""
    return frozenset(r.name for r in REORG_REMOVED_INDICATORS)


@dataclass(frozen=True)
class PipelineGap:
    """collect_indicators_v2 (旧) → build_labeled_win_from_npz (新) の意図的な
    列欠落 1 件 (KNOWN_PIPELINE_GAPS 用)。"""

    column: str       # collect_indicators_v2.INDICATOR_COLUMNS 側の base 列名 (*_raw等を除く)
    confirmed: str     # 欠落を意図的と確認した日付 (YYYY-MM-DD)
    reason: str        # 欠落の理由 (設計上の構造的制約 または明示的な削除決定)


# ============================
# npz→CSV 変換ツールのレジストリ整合 — 既知の意図的ギャップ許容リスト
# (2026-08-13 新設、横展開監査 P1/P2 台帳監査の提案4本の1)
# ============================
# collect_indicators_v2.INDICATOR_COLUMNS (旧収集) と
# build_labeled_win_from_npz._final_fieldnames("full") (新変換) の差分のうち、
# **設計上意図的で危険でない**ものだけをここに列挙する。
#
# 重要: 2026-08-13 時点で判明している「意図的でない脱落11列」
# (main_linked_pair_count 等、docs/INDICATOR_PROPOSAL_ROUND2_2026-08-13.md
# A-1、user採否待ち) はここに **含めない**。含めてしまうと A-1 の問題が
# テストから見えなくなり「直したことにする」事故を再生産するため
# (tests/test_indicator_pipeline_registry_2026-08-13.py が11列の存在を
# 継続的に検出する)。
KNOWN_PIPELINE_GAPS: tuple[PipelineGap, ...] = (
    PipelineGap(
        "tsumo_count_rate", "2026-08-13",
        "累積手数カウンタ (フレーム間の state) が必要。npz は盤面グリッドの"
        "みを保持する grid-only ツールのため構造的に計算不可 "
        "(build_labeled_win_from_npz.py 冒頭「現状カバー範囲」参照)",
    ),
    PipelineGap(
        "margin_time_rate", "2026-08-13",
        "試合相対経過秒 (state) が必要。tsumo_count_rate と同じ構造的制約",
    ),
    PipelineGap(
        "chain_duration_sec", "2026-08-13",
        "連鎖の実時間長 (フレーム間の state) が必要。同上の構造的制約",
    ),
    # ojama_net_balance/ojama_forecast はタスク#8 (2026-08-13、
    # docs/CROSS_CUTTING_AUDIT_2026-08-13.md P4決着) で再接続済みのため本
    # 許容リストから削除した (収集側 collect_boards_lean.py が
    # OjamaAccountingTracker を実駆動して真値を npz に保存するようになった、
    # build_labeled_win_from_npz.py の OJAMA_TRUTH_COLUMNS 参照。許容リストが
    # 陳腐化したまま残すと再接続の事実がテストから見えなくなるため削除する
    # ルール、test_known_pipeline_gaps_entries_are_actually_absent 参照)。
    PipelineGap(
        "reach_fire_power", "2026-08-13",
        "next_pair/dnext_pair 依存。npz は --with-next 収集時のみ next1_a/b を"
        "保持し、本ツールでは next 依存指標は未実装 (冒頭「現状カバー範囲」"
        "に既存記載)",
    ),
    PipelineGap(
        "near_future_fire_k1", "2026-08-13", "reach_fire_power と同じ next_pair 依存の制約",
    ),
    PipelineGap(
        "near_future_fire_k2", "2026-08-13", "同上",
    ),
    PipelineGap(
        "near_future_fire_k3", "2026-08-13", "同上",
    ),
    PipelineGap(
        "near_future_fire_k4", "2026-08-13", "同上",
    ),
    PipelineGap(
        "near_future_fire_k5", "2026-08-13", "同上",
    ),
    PipelineGap(
        "fire_stability_k2", "2026-08-13", "同上 (near_future_fire_power と同じビーム machinery の副産物)",
    ),
    PipelineGap(
        "fire_stability_k4", "2026-08-13", "同上",
    ),
    PipelineGap(
        "fire_stability_k6", "2026-08-13", "同上",
    ),
    PipelineGap(
        "expected_fire_k1", "2026-08-13", "同上 next_pair 依存 (ランダム色ツモのモンテカルロ平均)",
    ),
    PipelineGap(
        "expected_fire_k2", "2026-08-13", "同上",
    ),
    PipelineGap(
        "expected_fire_k3", "2026-08-13", "同上",
    ),
    PipelineGap(
        "expected_fire_k4", "2026-08-13", "同上",
    ),
    PipelineGap(
        "saturated_chain_count", "2026-08-12",
        "a-1決定 (INDICATOR_REORG_DECISIONS 参照)。REORG_REMOVED_INDICATORS と"
        "同一根拠 (current_max_chain と完全一致のため削除)",
    ),
    PipelineGap(
        "absorption_capacity", "2026-08-12",
        "a-1決定。REORG_REMOVED_INDICATORS と同一根拠 (board_puyo_total と"
        "完全重複)",
    ),
    PipelineGap(
        "center_bulge", "2026-08-12",
        "b-1決定 (INDICATOR_REORG_DECISIONS 参照)。center_bulge_color/_ojama "
        "に分解済みのため合成列は出力しない (機能は分解後の2列として存続、"
        "削除ではない)",
    ),
)


def known_pipeline_gap_columns() -> frozenset[str]:
    """KNOWN_PIPELINE_GAPS の列名集合を返す (整合テスト用)。"""
    return frozenset(g.column for g in KNOWN_PIPELINE_GAPS)


def _join(flags: tuple[AdoptedFlag, ...]) -> str:
    """フラグ文字列を空白区切りで連結する。"""
    return " ".join(f.flag for f in flags)


def recognition_flags() -> str:
    """認識の本番構成フラグを返す (収集・表示の両方が受け付けるもの)。"""
    return _join(RECOGNITION_ADOPTED)


def recognition_load_default_kwargs() -> dict[str, "float | bool"]:
    """RECOGNITION_ADOPTED を RecognitionPipeline.load_default() 用 kwargs に変換する。

    2026-08-13 是正 (根因調査の副次発見): scripts/visualize_advantage_overlay.py が
    RECOGNITION_ADOPTED (本番採用の認識フラグ群) を load_default() へ一切
    転送しておらず、デモ/レビュー動画が本番より劣化した認識 (バーストガード等が
    無効) で生成されていた (2026-08-08 の --early-fire-reaction 付け忘れ事故と
    同型)。collect_boards_lean.py の argparse dest 名 (enable_effect_gate 等、
    同ファイル 769-1009 行) は RecognitionPipeline.load_default() のキーワード
    引数名と完全に一致しているため、 "--xxx-yyy" 形式のフラグ名を dest 名
    "xxx_yyy" へ機械的に変換するだけで両者と同じ呼出し経路になる (RECOGNITION_
    ADOPTED に新しいフラグを追記するだけで overlay 側も自動追従し、個別配線の
    抜け漏れを構造的に根絶する)。値付きフラグ ("--burst-gate-open-threshold
    0.954" 等) は float にパースし、値無しフラグ (store_true 系) は True にする。
    """
    kwargs: dict[str, "float | bool"] = {}
    for f in RECOGNITION_ADOPTED:
        parts = f.flag.split()
        name = parts[0].lstrip("-").replace("-", "_")
        kwargs[name] = True if len(parts) == 1 else float(parts[1])
    return kwargs


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
        ("指標大整理", INDICATOR_REORG_DECISIONS),
    ):
        lines.append(f"[{title}]")
        for f in flags:
            lines.append(f"  {f.flag}  (採用 {f.adopted}) — {f.reason}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())

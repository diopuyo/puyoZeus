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
    AdoptedFlag(
        "--enable-stable-persistence-gate", "2026-08-18",
        "(d) STABLE確定の持続確認 (収集限定、RecognitionPipeline には足さない意図的"
        "非対称配線、docs/BOUNDARY_MULTISIGNAL_DESIGN_2026-08-17.md §5)。"
        "`_should_emit` 直前で直近 STABLE_PERSISTENCE_WINDOW_SEC=0.25 秒の盤面ROI"
        "生ピクセルdiffが全て閾値未満のときだけ確定を許可し、連鎖アニメ中・送付"
        "フラッシュ重畳による静止誤認を除外する。閾値 STABLE_PERSISTENCE_DIFF_"
        "THRESHOLD=1.0 は実測分離ギャップから決定 (汚染側029最小値1.07、綺麗な"
        "21枚側の最大値0.858、0.858<1.0<1.07に収まるラウンド値、src/board_motion.py "
        "コメント参照)。シーン逆算でなく物理量の分離ギャップから固定 (過学習禁止"
        "規約準拠)。③試合外は静止画面のため本機構では検出不能 (差分ゼロ、"
        "(b) 系列が担当)",
    ),
    AdoptedFlag(
        "--enable-boundary-multisignal", "2026-08-18",
        "game_idx境界マルチシグナル検知 (`_SharedGameCounter`/`_reconcile_"
        "boundary_anomalies`、W20/W21根治)。W22救済込みで c109/c13/c96 実測 "
        "7/8=87.5% (docs/BOUNDARY_MULTISIGNAL_DESIGN_2026-08-17.md §0)。"
        "**注記**: 今回 (2026-08-18) の採用登録タイミングでの追加A/Bは未実施。"
        "根拠は上記W22時代の実測のみであり、148再収集時に併走測定を推奨"
        "(docs/BOUNDARY_MULTISIGNAL_DESIGN_2026-08-17.md §2「(a)既存境界マル"
        "チシグナルの本採用判断」は正式なA/B測定を前提としていたが、user承認"
        "『全群採用』により本実測をもって先行登録する)",
    ),
    AdoptedFlag(
        "--enable-winner-panel-crosscheck", "2026-08-18",
        "WIN★パネル数値差分による勝者判定クロスチェック (`src/win_panel.py`+"
        "`src/match_winner.py`、オフライン専用)。boundary-multisignal と同じ"
        "W22時代の実測 (7/8=87.5%) が根拠であり、**今回の追加A/Bは未実施**"
        "(同上の注記が適用される)",
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
    AdoptedFlag(
        "--enable-patch-fp-hsv-guard", "2026-08-17",
        "W13 根治 (user承認2026-08-17)。**tier1 patch-NCC の EMPTY 判定に HSV 色域 AND "
        "ガードを追加**する (`src/image_reader.py:1044-1115` `_is_empty_tier1`)。"
        "patch-NCC が閾値 (PATCH_NCC_EMPTY_THRESHOLD=0.92) 以上で背景と一致しても、"
        "現フレームのパッチを単独 HSV 分類器にかけて EMPTY/UNKNOWN 以外を返すなら EMPTY 化を却下する。"
        "**新規閾値のシーン逆算はしていない** — 2026-05 cycle17-19 で median 距離ベースの "
        "`is_empty_by_fp` に作られた AND ガードを、その後移行した patch-NCC 経路へ**移植したもの** "
        "(移植漏れがW13の一因だった)。"
        "【W13の症状】試合開始2.0秒の背景指紋強制採取 (鶏卵問題対策で puyo 数上限を実質無制限に緩和) が "
        "既に設置済みの実在ぷよを『背景』として焼き込み、以降そのセルが patch-NCC 一致で無条件 EMPTY 化 "
        "→ **列が8〜10秒まるごと消える** (userレビュー指摘16、本番採用構成でも再現)。"
        "【測定 (物差しv2、55盤面、user全数レビュー済みラベル)】W16教訓に従い測定構成を明記: "
        "ベース=c1p (現行本番採用構成)。OFF/ONで STABLE snapshot の間引き周期が変わり分母がズレるため "
        "2段階で公平化した。**stage1 共通突合 (n=48/55、3453セル): 95.66% → 96.84%**。"
        "**stage2 同一フレーム限定 (n=22/55、1584セル): 98.61% → 98.74%**。"
        "方向別内訳で**新規悪化 (元々正解だったセルを壊す) は0件**、W9/W10/color_to_ojama 系の "
        "既知誤り軸には一切干渉なし。"
        "【案1を撤回した根拠】同時に検討した案1 `--enable-highlight-override` "
        "(白ハイライトblob検出でtier1のEMPTY判定を却下) は stage2 で **98.17% とベースより悪化** し、"
        "13セル/2盤面の新規退行を出した。根因は W19 に登録: (a) 救済がセル単位で白blobの写り方が "
        "フレームごとにばらつくため1セル救済に失敗すると穴が開き、`clear_floating_above_gap` が "
        "その上を全部 EMPTY 化して**列ごと道連れ**にする (b) **highlight_override が他セルを救済して "
        "puyo 数差を埋めてしまい、自己修復 `_apply_baseline_broken_counter` の発火条件に入らず "
        "汚染 bg_fp が生き残った** = 修正Aが修正Bの症状を隠す新種の副作用連鎖。"
        "本フラグは**案1の悪化13セルを全13セルとも解消**する。"
        "また案1+2 併用構成は STABLE 盤面 npz 全2023スナップショットで**本フラグ単体とbit-identical** "
        "(案1は本フラグが既に救済済みのセルにしか作用せず追加効果ゼロ) のため、併用は採用しない。"
        "【残リスク】stage2 の n=22 は小標本で1セル変化が±0.06pt動く規模。方向は明確だが "
        "**148動画での広域バックテストは別途必須**。全体テスト 5,043 passed / 0 failed、"
        "既定OFF時は bit-identical を静的テストで担保",
    ),
    AdoptedFlag(
        "--enable-floating-gap-restore", "2026-08-17",
        "R2浮きぷよ是正 (user承認2026-08-17、5フラグ一括採用・構成F)。TSUMO_FALL/"
        "OJAMA_FALL→STABLE 遷移で「下が空・上に puyo」の物理矛盾を検出したら、"
        "上を消すのでなく遷移前 confirmed_board から色を復元する "
        "(docs/KNOWN_WEAKNESSES.md R2)。**単独では持続誤認±0** (構成A→B: 106→106) "
        "だが物理矛盾是正の保険としてF構成 (5フラグ一括) に同梱採用。"
        "根拠データ: data/verify/diag_r2_floating_gap_restore_2026-08-17/。"
        "測定は必ずF構成 (5フラグ揃い) で行っており本フラグ単体の効果を測った"
        "数値ではない",
    ),
    AdoptedFlag(
        "--enable-landing-color-guard", "2026-08-17",
        "W10観測補正継続ガード (user承認2026-08-17、5フラグ一括採用・構成F)。"
        "着地セル色の継続監視ガードで、着地直後の一時的な色観測ブレを"
        "追跡し続けて誤確定を防ぐ。構成B→C (+本フラグ) で持続誤認106→105"
        "(-1)。根拠: docs/KNOWN_WEAKNESSES.md W10節、コミット85e39b3で観測補正"
        "継続ガードの実測を記録、d1c60f9でCLI配線漏れ (collect_boards_lean.py) "
        "を是正済み。測定はF構成 (5フラグ揃い) での積み上げ値",
    ),
    AdoptedFlag(
        "--enable-override-color-guard", "2026-08-17",
        "cycle71n長期投票overrideの安全網 (user承認2026-08-17、5フラグ一括採用・"
        "構成F)。真因は W23 (_validate_next_history のever_seen飢餓状態) と"
        "判明したため、本フラグ自体の位置づけは根治でなく安全網 "
        "(persistent_misread_e.json 系統1ガード)。コミットa43d3fbで実装。"
        "測定はF構成 (5フラグ揃い) での積み上げ値であり、単独の増減は"
        "分離測定していない",
    ),
    AdoptedFlag(
        "--enable-ojama-column-stack-fix", "2026-08-17",
        "持続誤認26件の系統2根治 (user承認2026-08-17、5フラグ一括採用・構成F)。"
        "おじゃま配分ロジックが同一列に対して二重に書き込み衝突する不具合を"
        "根治する (c109 実例で確認)。構成D→E (+本フラグ+override-color-guard) "
        "で持続誤認105→104 (-1)。コミットa43d3fbで実装。"
        "測定はF構成 (5フラグ揃い) での積み上げ値",
    ),
    AdoptedFlag(
        "--enable-next-history-starvation-fix", "2026-08-17",
        "W23根治 (user承認2026-08-17、5フラグ一括採用・構成F)。"
        "_validate_next_history の ever_seen 飢餓状態 (NEXT履歴が長期間"
        "更新されず検証ロジックが機能不全になる不具合) を根治する。"
        "構成E→F (+本フラグ) で **stage2 同一フレーム限定セル正解率 "
        "98.80%→99.43%、持続誤認104→70 (-34)**。W23直接検証25件は100%解消"
        "(persistent_misread_e.json 記載の25件が全て解消を個別確認)。"
        "新規出現3件は別機構の露出であり本フラグの副作用ではない "
        "(docs/KNOWN_WEAKNESSES.md 615-648行に詳細記録)。コミット19dc93aで実装。"
        "【5フラグ一括の位置づけ】本エントリを含む上記4フラグ (floating-gap-"
        "restore/landing-color-guard/override-color-guard/ojama-column-stack-fix) "
        "とセットで構成F として測定・採用 (2026-08-17統一測定、data/verify/"
        "recognition_unified_2026-08-17/persistent_misread_{a..f}.json)。"
        "構成A (現本番、持続誤認106) を基準に B→C→D→E→F の順で積み上げ、"
        "最終構成Fで持続誤認70・stage2セル正解率99.43%・W10_red_purple誤読"
        "10→0完全解消・盤面完全一致33→35/51を実測。フルpytest 5,113 passed / "
        "0 failed、既定OFF時は bit-identical (静的テストで担保)",
    ),
    AdoptedFlag(
        "--enable-ojama-cnn-override-warmup", "2026-08-18",
        "W25第1〜2弾 (user承認2026-08-18、docs/KNOWN_WEAKNESSES.md W25節)。"
        "当初はcycle71n override専用のOJAMA_FALL→STABLE warmupとして実装したが"
        "効果ゼロと実測 (対象9セル解消0/9、コミット2fc990d)。真因再追跡でdrift"
        "再同期の暴発 (雲によるCNN 4↔9往復→DriftDetector needs_resync発火→"
        "confirmed_board全None化→バイパス) と判明し、本フラグをOJAMA_FALL entry"
        "起動+drift-resync抑制サイトに転用 (コミット8fe0759)。**実測**: 28チャンク"
        "でresync発火14→7、resetの発生3→0を直接実証、物差しv2はbit-identical"
        "(無害)。ただし9セル自体は本フラグだけでは未解消 (第3の独立経路が別途"
        "存在、enable-ojama-write-accounting-guardが根治)。**役割再定義**: "
        "根治実装後は会計整合フィルタのフェイルセーフ (W2破綻動画等の会計崩壊時"
        "の保険) として位置づけ直した (アーキ決定、docs/KNOWN_WEAKNESSES.md "
        "W25節 750-756行)",
    ),
    AdoptedFlag(
        "--enable-ojama-write-accounting-guard", "2026-08-18",
        "W25第3弾 (根治) + 固着対策 (user承認2026-08-18、docs/KNOWN_WEAKNESSES.md "
        "W25節)。CNN観測→状態機械入力直前の一元会計整合フィルタ: 非空色→9への"
        "直接遷移を、その列に会計上の未着弾クレジットが無い限り無条件拒否する"
        "(物理制約「おじゃまは空セルにのみ着弾」)。**実測 (コミット4290fc5)**: "
        "対象9セル9/9解消。stage1 (共通突合) 97.98%→**98.41% (+0.44pt)**、"
        "**stage2 (同一フレーム限定) 99.46%→99.27% (-0.18pt、n=23小標本)**は"
        "悪化方向だが正直に記載する。反映遅延の新規退行なし。新規悪化2件"
        "(c10_2P/c109_2P) は精査の結果W1型永久固着 (消去+着弾を両方見逃すと"
        "古い色に固着) の現実化と確定し、持続観測タイムアウト解除 "
        "`OJAMA_REJECT_TIMEOUT_SEC=1.5` を追加実装 (コミット9565e9b) して固着"
        "2件を解消 (+0.07〜0.27秒で自己修正、上限1.5秒の1/5以下)、雲9セル"
        "9/9は維持。**新規許容の明文化 (アーキ承認)**: おじゃま反映が最大1.5秒"
        "遅れうる。8フレーム基準 (feedback_placement_reflection_8frames) は"
        "ツモ設置対象の受け入れ基準であり、おじゃま着弾には元々適用対象外"
        "(棄却側論拠=雲は0.85〜1.0秒で晴れる実測 / 受理側論拠=陳腐化した"
        "持続観測メモリは新しい実観測に屈服すべきという構造的原則)。副産物: "
        "OjamaAccountingTracker の PENDING_ABS_CAP=216到達バグ (score OCR異常"
        "由来) を独立の既存問題として発見、要対処リスト入り。フルpytest "
        "5,255 passed / 0 failed、既定OFF時 bit-identical",
    ),
    AdoptedFlag(
        "--enable-match-end-persist-override", "2026-08-18",
        "境界RT系 (b-1)。user承認2026-08-18、docs/BOUNDARY_MULTISIGNAL_DESIGN_"
        "2026-08-17.md §3(b-1)。`match_end_locked` が MATCH_END_PERSIST_"
        "OVERRIDE_SEC以上連続Trueなら、chain_in_progress による抑制を上書きして "
        "effective_hard_off を有効化する持続タイマー。Step0診断"
        "(data/verify/diag_match_end_miss_2026-08-17/) で判明した真因「本物の"
        "決着パネルは勝者の連鎖アニメ中に表示され始め3秒超持続するが、既存の"
        "chain_in_progress ガード (2026-07-23導入) が瞬間誤爆対策のまま2.55秒間"
        "誤って打ち消していた」に対処する。持続時間で瞬間誤爆 (単発) と本物の"
        "決着 (3秒超) を弁別。030実写検証 (c21) 済み、既存回帰テスト "
        "(test_gravity_settle_in_progress_suppresses_match_end_locked_false_"
        "positive) は維持したまま新規検証を追加",
    ),
    AdoptedFlag(
        "--enable-post-match-lockdown-latch", "2026-08-18",
        "境界RT系 (b-2)。user承認2026-08-18、docs/BOUNDARY_MULTISIGNAL_DESIGN_"
        "2026-08-17.md §3(b-2)。ばたんきゅー/やった!検出をトリガーに「次の"
        "本物の試合開始が確認されるまで試合外とみなす」ラッチ。結果パネル・"
        "対戦カード紹介・次ラウンド待機画面を一括カバーし、ロックダウン5秒切れ"
        "後の再活性化 (対戦カード紹介中に is_match_active へ復帰する新規盲点、"
        "030_c21_2P_f57548実写確認) を防ぐ。`hard_match_off = score_zero_both "
        "or match_end_locked or self._post_match_lockdown_active` へ合流、"
        "score_actively_moving/chain_in_progress の保護は無変更。RT本体実装可"
        "(「盤面が無いと確定している区間の延長」のため指摘13リスクなし)。"
        "実測: data/verify/boundary_impl_verify_2026-08-18/final_verify_"
        "summary.json で rt_blocked_count=4/rt_total=5 (試合外RT遮断4/5)",
    ),
    AdoptedFlag(
        "--enable-result-screen-hardening", "2026-08-18",
        "境界RT系 (③)。user承認2026-08-18。score_actively_moving の装飾演出"
        "(ラウンド告知・対戦カード紹介等の非試合画面でスコア風の数字表示が動いて"
        "見える) への誤認を裏取りする列フィルタ強化。実測: data/verify/"
        "boundary_impl_verify_2026-08-18/final_verify_summary.json で "
        "column_filter_excluded_count=5/column_filter_total=5 (試合外の"
        "誤混入5/5を列フィルタで除外)。b-1/b-2 と同一の検証セット "
        "(c18/c20/c21 実写5アンカー) で確認済み",
    ),
    AdoptedFlag(
        "--enable-ojama-fall-color-swap-guard", "2026-08-18",
        "W26 (docs/KNOWN_WEAKNESSES.md W26節、user承認2026-08-18、コミット"
        "ca03275で実装済み・既定OFF)。連鎖発火の閃光 (白〜青) による色→別色"
        "誤読 (例: 青→緑、赤→黄) をOJAMA_FALL中に限定して拒否する。"
        "`src/ojama_write_accounting.py` の `filter_ojama_write_by_accounting` "
        "に `reject_color_swap` (既定 False) を追加し、`prev_stable_color` と "
        "`new_cnn_value` が共に色ぷよ (1〜5) かつ不一致の場合を追加棄却する。"
        "`enable_ojama_write_accounting_guard` (W25) とは独立のフラグ (単独"
        "稼働をテストで確認済み)。**実測 (scripts/_verify_color_swap_guard_"
        "2026-08-18.py、対戦区間t≥120sのみ)**: user報告の具体箇所 "
        "(video36 t=141-148s、2P、青→緑/赤→黄の実例) 22件→0件 (完全解消)、"
        "実画面フレームと突合済み。広域窓: video36(120-160s) 2P 406→114件 "
        "(-72%)、video52(120-160s) 2P 2→0件 (-100%)、c100(570-660s) 1P "
        "134→36件 (-73%)/2P 70→7件 (-90%)。固着チェック: 全ケースで"
        "flagged_stuck_count=0 (OJAMA_REJECT_TIMEOUT_SEC=1.5秒タイムアウトは"
        "未発動、観測された違反の最大持続は1.0秒未満)。フルpytest "
        "5,388 passed / 0 failed。**既知の限界 (3点、必ず認識のこと)**: "
        "①完全解消ではない (72〜90%削減)。本フィルタは cnn_board が state "
        "machine に入る**前**に適用されるため「現フレームが OJAMA_FALL か」を"
        "前フレームの確定 state でしか判定できず、OJAMA_FALL 突入の最初の"
        "1フレーム (約33ms@30fps) は保護対象外 (W25本体と共通のアーキ制約)。"
        "根治には pipeline のフレーム遅延アーキ変更が必要でスコープ外。"
        "②残存 violation の全量説明には至っていない (上記①は残差の一部説明"
        "に過ぎず、追加調査の余地あり)。③CHAIN中は対象外 (同種の違反が大量に"
        "出ている: video36 で 1P 104ep / 2P 131ep / 40秒だが、多段消去と重力"
        "補充の高速遷移のエイリアシングが未精査のため意図的にスコープ外。"
        "CHAIN中は一切発火しない設計をテストで確認済み)",
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
    AdoptedFlag(
        "--resolved-live-defender-strict", "2026-08-15",
        "指摘14 案1 (user承認2026-08-15、コミット53129bb)。決着ホールド中の "
        "受け側再評価 (_reevaluate_live_defender、指摘13で導入) の起動条件を "
        "状態機械ベースに厳格化する。従来は「ちょうど片側の chain_event が "
        "None」という XOR のみで受け側を「自由」と判定していたが、ChainEvent は "
        "trigger 検知時に1度発行され hold 秒後に None へ戻るパルス方式のため、"
        "「旧連鎖の hold 切れ〜新連鎖の trigger 検知」の settle gap にいる側も "
        "ev=None になり誤って自由扱いされていた。実測 (t=195.30): "
        "ev1_cc=9 (1P継続中) / ev2=None かつ state2=GRAVITY_SETTLE の 2P を "
        "自由な受け側と誤分類し、着弾前の綺麗な盤面をモデルへ渡して "
        "hold_p1 96.1%→81.1% へ退行 (真の飛来おじゃま589個=回避不能死なのに "
        "2P 18.9% を5.2秒表示)。修正 = defender 側の状態が "
        "_LIVE_DEFENDER_BUSY_STATES {CHAIN, GRAVITY_SETTLE} なら再評価を "
        "スキップし直前値を維持する。TSUMO_FALL/OJAMA_FALL は指摘13が意図した "
        "正当な自由行動を塞がないため意図的に非busy。A/B実測 "
        "(logs/_diag_issue14_flags_ab_v2_2026-08-15.log): 指摘14窓 "
        "(194.53-201秒) は退行が完全消滅し 96.1% を窓全体で維持。指摘13 正当窓 "
        "(234.87-245.5秒) は t=236.27 以降 baseline と完全一致・0.5秒ごとに "
        "連続変化 (凍結の再発なし)、冒頭0.9秒のみ 2P 77%→84.3% に変化 "
        "(defender が実際に busy だった瞬間を正しくスキップした結果。この値は "
        "指摘12で決着した84.3%と一致)",
    ),
    AdoptedFlag(
        "--resolved-kill-override", "2026-08-15",
        "指摘14 案2 (user承認2026-08-15、コミット8f8a577)。既存の致死上書き "
        "安全弁 kill_override を決着ホールド表示値にも配線する。従来は "
        "ライブ per-frame 経路にのみ配線されており、決着ホールド中は "
        "disp_adv/disp_p1 を丸ごと上書きする経路が通常経路を迂回するため "
        "pending/room 比が致死水準でも安全弁が絶対に発火しなかった "
        "(実測 589/50≈11.8 ≫ KILL_RATIO_FULL=1.5 で無発火)。材料は新規に "
        "増やさず既存の観測量のみ再利用 (pending=_incoming_total_p1/p2 = "
        "指摘11の着弾完了判定と同一値、room=board_room(b1)/board_room(b2))。"
        "二重計上防止はライブ経路と同じく kill_override を最終段として適用 "
        "(g=1 の完全上書き時は amplify 由来の寄与も自動的に上書きされる)。"
        "A/B実測 (logs/_diag_issue14_flags_ab_2026-08-15.log): 指摘14窓の "
        "全時刻で adv=+100/p1=99.3% (2P 19%→0.7%)。案1が誤爆機構そのものを "
        "断つのに対し、本フラグは致死場面の最終防波堤として併用する",
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


# ============================
# 学習データビルダー — 標準採用オプション (2026-08-18 新設)
# ============================
# scripts/build_labeled_win_from_npz.py が受け付ける CLI フラグのうち、
# 「学習データビルド時に常に付けるべき標準構成」として確定したものをここに
# 記録する (INDICATOR_REORG_DECISIONS と同様、決定記録のみ・単一情報源)。
LEARNING_DATA_BUILD_ADOPTED: tuple[AdoptedFlag, ...] = (
    AdoptedFlag(
        "--exclude-match-end-locked", "2026-08-18",
        "境界実装の仕上げ (user承認2026-08-18)。npz の match_end_locked==1 "
        "または post_match_lockdown_active==1 のフレームを学習データ csv から"
        "除外する (scripts/build_labeled_win_from_npz.py)。決着後の結果パネル・"
        "対戦カード紹介・次ラウンド待機画面が試合中と誤って学習データに混入する"
        "のを防ぐ (W20「勝敗演出の幻盤面」族の学習データ側対策)。実測: "
        "data/verify/boundary_impl_verify_2026-08-18/final_verify_summary.json "
        "で column_filter_excluded_count=5/column_filter_total=5 (試合外混入"
        "5/5を除外)。両列が存在しない旧npz (収集時に本フラグ非対応だったもの) "
        "では no-op (後方互換、tests/test_build_labeled_win_from_npz.py "
        "test_convert_one_npz_exclude_match_end_locked_noop_when_columns_"
        "absent で確認済み)。既定 False のため、次回の学習データビルド実行時に"
        "明示指定が必要 (本エントリはその明示指定を「標準オプション」として"
        "記録するもの)",
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
        ("学習データビルダー", LEARNING_DATA_BUILD_ADOPTED),
    ):
        lines.append(f"[{title}]")
        for f in flags:
            lines.append(f"  {f.flag}  (採用 {f.adopted}) — {f.reason}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())

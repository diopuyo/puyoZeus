# 評価指標リファレンス (45 個)

CLAUDE.md の補足。指標の完全定義。各指標は **0.0〜1.0 に正規化必須**。

## 設計思想 (確定方針)

1. **「形は手段、機能が本質」**
   GTR / サブマリン / 階段 等の形分類は二次的。火力・中盤厚み・お邪魔体制等の
   「機能・能力」を直接測る指標が一次。

2. **「観測軸を提供 → 学習で重要度を発見」**
   私が「これが重要」と決めない。戦略概念をデータ観測指標に翻訳し、重要度・
   閾値は学習結果から ranking で発見。

3. **End-to-end CNN base + 補助 indicator** (確定)
   最終的に CNN が直接「盤面 → 勝率」を学ぶ。indicator は auxiliary supervision target。
   AlphaGo/AlphaZero 系の確立アーキテクチャ。

## 階軸構造

```
一次軸: 機能能力指標 7 個 (Phase H1)
二次軸: 戦況・タイミング指標 8 個 (Phase H1)
三次軸: 形分類指標 5 個 (Phase F B-1 + Phase H1 1)
+ 既存 16 戦術指標 (8 戦術 + 8 拡張、Phase 2)
+ Phase J 拡張 5 個
+ Phase K 拡張 3 個 (CSV 除外、推論用)
+ Tier B 形質 3 個
+ B-4 (rotation_skill) 1 個
= 計 45 features (CSV 列)、+ 3 推論用 = 48 indicators
```

## 一次軸: 機能能力指標 7 個 (Phase H1、★ 本質)

形 (GTR/サブマリン/階段) でなく「達成された機能」を直接測定。

| 指標 | 内容 | 計算方針 |
|---|---|---|
| `ready_chain_count` | 即発火可能な連鎖数 | ChainSimulator(board).chain_count → MAX_EXPECTED_CHAIN で正規化 |
| `ignition_distance` | 何手で発火可能になるか | 1-step / 2-step 探索で発火寸前を判定、低いほど近い→高スコア |
| `current_fire_power` | 即発火した場合の ojama 数 | chain_result.score / OJAMA_DIVISOR / MAX_OJAMA_OFFSET |
| `maximum_fire_power` | 完成後の最大 ojama 数 (= 飽和連鎖量) | 全 puyo 最適配置推定 (heuristic) |
| `mid_game_response_capacity` | 中盤応答能力 | (sub_chain_quality + max_fire_power 残量) / 1催促消費 |
| `harassment_readiness` | 即催促打てる小連鎖の存在 | sub_chain_quality 拡張、即時威力で評価 |
| `ojama_defense_capacity` | お邪魔体制 (受けて掘れる量) | 仮想 ojama 10/20/30 落下後、本線回復可能性 |

これらは形分類を超えた「機能の到達度」を測定。形が変わっても、これら 7 指標が
同じなら勝率は同等のはず (= 仮説、Phase H1 学習で検証済 — ojama_defense rank 3、
upper_board_density rank 2)。

## 二次軸: 戦況・タイミング指標 8 個 (Phase H1)

機能達成度を文脈化する指標 (時間軸・対戦相互作用):

| 指標 | 内容 |
|---|---|
| `self_chain_duration_frames` | 自連鎖 frame 数 (落下マス × クイック反映) |
| `opp_chain_duration_frames` | 相手連鎖 frame 数 |
| `chain_duration_advantage` | 応答可能 puyo 数差 (相手連鎖中に自分が打てる手) |
| `harass_event_count_30s` | 直近 30s の催促回数 (自/相手) |
| `early_aggression_score` | +30s での攻撃量 (連鎖発火 + ojama 送信) |
| `counter_ignition_signal` | ojama 受け直後の本線発火パターン検出 |
| `post_all_clear_state` | 序盤全消し検出 + ボーナス使用判定 |
| `upper_board_density` | 上部 (10+ 段) puyo 数 (上級者 advantage 観測) |

注意: `harass_event_count_30s` / `early_aggression_score` / `counter_ignition_signal` は
state-holding 必要なため stub 実装 (中立値 0.5)。phase_e_collect で外部 wrapper から
値注入予定 (Phase L で本格化)。

## 三次軸: 形分類指標 5 個 (補助)

形は機能達成の手段。学習で「機能指標と独立価値あるか」を判定:

| 指標 | 内容 | 状態 |
|---|---|---|
| `form_gtr` | GTR 完成度 | B-1 既存 |
| `form_llr` | LLR 完成度 | B-1 既存 |
| `form_staircase` | 階段完成度 | B-1 既存 |
| `form_zabuton` | 座布団完成度 | B-1 既存 |
| `gtr_orientation` | 先折り (0) / 後折り (1) / 自由形 (2) | Phase H1 新 |

## 既存 16 指標 (Phase 2、戦術 + 拡張)

### 戦術 8 指標
1. **本線完成度** (`main_chain_maturity`): 連鎖数 ÷ 推定最大連鎖数
2. **伸ばし余地** (`extension_potential`): 1〜2 ツモ探索で本線伸長placement 比率 + 空きセル余裕
3. **副砲の質** (`sub_chain_quality`): 本線と独立した小連鎖の威力・催促・合体可能性
4. **催促耐性** (`harassment_resistance`): おじゃま 10〜30 個仮想落下後に本線が生存
5. **窒息リスク** (`death_risk`): 3 列目 (致命列) 高さ + 周辺列重み付け平均
6. **相殺力** (`offset_power`): 即時発火可能な合計おじゃま数 + 上乗せ能力
7. **セカンド構築力** (`second_chain_potential`): 本線発火後の残ぷよで第二波を組める余地
8. **フィールド効率** (`field_efficiency`): 連鎖参加ぷよ数 ÷ 全ぷよ数 (≥4 連結対象)

### 拡張 8 指標
9. **ネクスト受け入れ** (`next_acceptance`): next + dnext 仮置きで連鎖伸長量
10. **形評価** (`shape_score`): U 字理想形 (10,9,7,7,9,10) との偏差 [mayah / puyoai 由来]
11. **接ぷよ密度** (`touching_density`): 同色隣接ペア / 全 puyo 数 [meatfighter 由来]
12. **連鎖尾高さ** (`tail_height`): 1 連鎖目発火点高さ [takapt AI 由来]
13. **色分散** (`color_variance`): 各色重心からの平均 Manhattan 距離
14. **キーぷよ柔軟性** (`key_flexibility`): +1 puyo で連鎖が伸びる位置の比率
15. **副砲独立性** (`sub_chain_independence`): 本線除外後の副砲評価
16. **連鎖タイミング圧** (`chain_timing_pressure`): あと N 個で発火可能の近接度

## Phase J 拡張 5 個

| 指標 | 内容 |
|---|---|
| `incoming_ojama_pressure` | 相手から飛来する予告おじゃま個数 / MAX_OJAMA_OFFSET 正規化 |
| `opponent_chain_threat` | 相手の即発火可能連鎖威力 |
| `adjacent_height_diff` | 隣接列の高さ差分集計 |
| `high_connection_count` | 高 connectivity (≥3 連結) の数 |
| `required_puyo_to_fire` | 発火に必要な追加 puyo 数 |

注: `next_acceptance` は Phase J ではなく Phase 2 拡張、ここでは触れない。

## Phase K 拡張 3 個 (CSV 除外、推論用)

| 指標 | 内容 |
|---|---|
| `opponent_offset_power` | 相手の相殺力 (`opp_board.offset_power`) |
| `post_ojama_chain_health` | おじゃま受け後の連鎖健全度 |
| `isolated_puyo_count` | 孤立 puyo 数 |

`PHASE_K_EXTRA_INDICATOR_NAMES` で別タプル管理。多重共線性で学習悪化のため CSV 除外。
推論ロジック (ojama 整合性チェック等) で利用。

## Tier B 形質指標 3 個

| 指標 | 内容 |
|---|---|
| `planning_entropy` | 最頻色の偏り (色配置の計画性) |
| `structure_solidity` | 連結ぷよの密度・構造的健全さ |
| `base_flatness` | 底面 (row 12) の平坦度 |

各指標は numpy 化 + `_connected_components` キャッシュで高速化済 (Tier B 高速化、2026-05-06)。

## B-4 追加指標 1 個

| 指標 | 内容 |
|---|---|
| `rotation_skill` | 回し入れ追跡 (RotationTracker、上級者戦術) |

state-holding なので scripts 側でスコア注入。GBM permutation rank 9 で実効果確認 (Phase H1)。

## 集計 / モジュール構造

`src/indicators.py`:
- `ALL_INDICATOR_NAMES` (8): 戦術 8 個
- `EXTRA_INDICATOR_NAMES` (37): 拡張 8 + Phase J 5 + Tier B 3 + B-1 4 + B-4 1 + Phase H1 16
- `PHASE_K_EXTRA_INDICATOR_NAMES` (3): 推論用 (CSV 除外)
- `IndicatorSet` dataclass: 各指標を field として持つ
- `IndicatorCalculator.compute_all()`: 全 45 指標を一括計算
- `IndicatorCalculator.compute_all_probabilistic()`: ProbabilisticBoard 入力版 (Phase G C-1)

合計 **45 features** (CSV) + 3 推論用 = **48 indicators**。
Phase H2 時系列展開で 45 × 6 軸 + interaction 10 = **280 features**。

## 学習における指標重要度 (Phase H1 GBM permutation top 10、Top-tier H1)

```
1. incoming_ojama_pressure 0.125 (rank 1 維持)
2. field_efficiency 0.014    ← 新規上昇
3. base_flatness 0.012
4. next_acceptance 0.012
5. offset_power 0.011         ← 復活 (旧 BCD で reduce dropped)
6. harassment_resistance 0.010
7. mid_game_response_capacity 0.009 ← H1 機能能力
8. current_fire_power 0.008   ← H1 機能能力
9. high_connection_count 0.007
10. shape_score 0.007
```

H2 時系列版では `__hist_max`, `__hist_mean` 系の時間軸特徴量が top 10 中 7 個。

## Scorer 重みセット

`src/scorer.py` の `LEARNED_WEIGHTS_*` 群、`PhaseAwareScorer(weight_mode=...)` で切替:

| 重みセット | 用途 |
|---|---|
| `DEFAULT_WEIGHTS` | 経験的、16 指標フル (test_acc=0.523) |
| `LEARNED_WEIGHTS_PHASE_E_*` | Phase E (38 動画、PhaseAware overall 0.659) |
| `OPTIMAL_PHASE_WEIGHTS` | DEFAULT(start) + V3(mid) + GLOBAL(end) |
| `LEARNED_WEIGHTS_PHASE_E_TIERB_*` | Tier B 高速化版 |
| (Phase H1 以降は dashboard.md に保存、scorer.py 統合は Phase L で実施予定) |

過渡域では隣接 phase の重みを線形補間 (`PHASE_BLEND_WIDTH_SEC=10s`)。

## 重要な互換維持ルール

- **重み追加時は既存 `LEARNED_WEIGHTS_*` を破壊しない**
- **新指標追加時は `EXTRA_INDICATOR_NAMES` の末尾に追加** (順序保持)
- **API 互換**: 既存 indicator の `compute()` シグネチャに optional 引数のみ追加可、削除不可
- **観測指標は stateless 実装を原則** (state-holding は外部 wrapper)

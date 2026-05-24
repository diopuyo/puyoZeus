# Session Handoff: 2026-05-06 (Tier B 高速化 + B-1/B-2 + C-2/C-3)

**目的**: 学習精度天井 (0.66) を破る障害解消プロジェクト。
本日 (2026-05-06) は Tier B 計算量問題の解決、B-1 (I-J 形テンプレ)、B-2 (I-E 2-step)、
C-2 (W-γ おじゃま予告)、C-3 部分 (HarassmentResistance 動的化) を実装。

PC 再起動による中断時点: **2026-05-06 23:30 頃**。BC regen の残り作業を再起動後に再開する。

---

## 1. 本日の完了タスク

### Step 1: Tier B 形質指標の高速化 ✅
1. `_connected_components` を盤面 grid hash でキャッシュ化 (`_CC_CACHE`、上限 50,000)
   - StructureSolidity / HighConnection / IsolatedPuyo の重複計算排除
2. `BaseFlatnessIndicator` を numpy ベクトル化
3. `StructureSolidityIndicator` のおじゃま集計を numpy 化
4. `PlanningEntropyIndicator` を `numpy.bincount` 化 + 再有効化
5. EXTRA_INDICATOR_NAMES に `planning_entropy` 復帰
6. tests/test_indicators_tier_b.py 18 新テスト追加

### Step 2: 全 66 動画 shard 再生成 (Tier B 反映) ✅
- 完了 2026-05-06 13:18、所要 7h18
- 出力: `data/training/match_features_phase_e_v01-94_tierb.csv` (10,945 行 / 28 列)
- workers=8, fps=3, max-matches=0
- 対象 66 動画: 1-21,23-38,40,43,48-52,54,57,61,63-66,70,73-75,77-81,84,86,89,91-93

### Step 3: Tier B 学習評価 ✅
- LR Phase Aware LOOV: **0.637** (旧 0.659 から -2.2pt、Tier B は中立〜弱負)
- GBM Phase LOOV: **0.650**、end **0.897**
- GBM video holdout: 0.628
- GBM permutation importance で `base_flatness` rank 2 (0.037)、`structure_solidity` rank 7 (0.023)
- `src/scorer.py` に `LEARNED_WEIGHTS_PHASE_E_TIERB_*` 追加 (`weight_mode="phase_e_tierb"`)
- ダッシュボード: `data/verify/phase_e_dashboard_tierb.md`

### Step 4-1: B-1 (I-J 形テンプレ) + B-2 (I-E 2-step) ✅

**実装**:
- B-1: `src/form_templates.py` 新規 (GTR / LLR / 階段 / 座布団 4 テンプレ)
  - 等価クラス (A/B/C/D) で色対称性、1P/2P 両側 mirror 評価
  - `src/indicators.py` に 4 indicator 統合 (INDICATOR_FORM_GTR/LLR/STAIRCASE/ZABUTON)
  - 新テスト: tests/test_form_templates.py (14) + tests/test_indicators_form.py (11)
- B-2: `ExtensionPotentialIndicator._search_improvement_2step`
  - 1-step で発火しなかった盤面に対し 2-step (1 puyo 追加) 探索
  - 2-step 重み 0.5 で集計
  - ChainSimulator キャッシュ活用
- パフォーマンス: IndicatorCalculator 31ms → 67ms (2.16x)

**再 regen** (完了 2026-05-06 21:38、所要 8h08):
- 出力: `data/training/match_features_phase_e_v01-94_b12.csv` (10,945 行 / 32 列、+4 form 列)

**学習結果** (`data/verify/learned_weights_phase_e_phase_aware_b12.json`):

| 指標 | Tier B | B-1+B-2 | 旧 PHASE_E | Δ vs Tier B |
|---|---:|---:|---:|---:|
| LR Phase LOOV avg | 0.637 | **0.643** | 0.659 | +0.006 |
| GBM Phase LOOV avg | 0.650 | 0.646 | 0.669 | -0.004 |
| GBM video holdout | 0.628 | **0.653** | 0.683 | **+2.5pt** |
| start phase | 0.491 | **0.506** | 0.533 | +1.5pt |
| end phase | 0.851 | **0.860** | 0.862 | +0.9pt |

**GBM permutation importance top 10** (B-1+B-2 反映後):
1. base_flatness 0.0239 (Tier B)
2. main_chain_maturity 0.0232
3. color_variance 0.0187
4. extension_potential 0.0162
5. field_efficiency 0.0141
6. adjacent_height_diff 0.0128
7. **form_staircase 0.0126** (B-1!)
8. death_risk 0.0121
9. structure_solidity 0.0081 (Tier B)
10. **form_zabuton 0.0079** (B-1!)

形テンプレ 2 個 (form_staircase, form_zabuton) が GBM importance top 10 入り。
期待値 +0.02 には届かなかったが、改善方向は確認。

### Step 4-2: C-2 (W-γ おじゃま予告) + C-3 部分 (HarassmentResistance 動的化) ✅ 実装完了

**C-2 (W-γ)**:
- 新規 `src/ojama_predictor.py`: OjamaPredictor クラス
  - score_delta * 70^-1 で予告おじゃま個数を時系列追跡
  - 自/相手の同時発火による相殺、自盤面 OJAMA セル増による消費
- `scripts/phase_e_collect_indicator_dataset.py` に統合
  - 各 STABLE frame で `ojama_pred.update(...)` → `incoming_ojama=pending_for(side)` を渡す
  - 旧: incoming_ojama=0 固定で IncomingOjamaPressureIndicator 機能不全
  - 新: 動的 pending count で活性化
- 新テスト: tests/test_ojama_predictor.py (10)

**C-3 部分 (I-H)**:
- `HarassmentResistanceIndicator` に `incoming_ojama` 引数追加
- 指定時、デフォルト 10-30 範囲に加え、実 incoming 数で 2x 重みの追加評価
- IndicatorCalculator.compute_all で HARASSMENT のみ incoming_ojama 渡し
- 残り C-3: chain_timing_pressure 等の opponent context 拡張 (次回)

**276 全 indicator/scorer/form/ojama テストパス**

---

## 2. 中断時の状態 (2026-05-06 23:00 頃 PC 再起動)

### 進行中だった作業: Step 4-2 BC regen

- **regen 開始**: 2026-05-06 21:39
- **中断時刻**: 2026-05-06 23:00 頃 (PC 再起動)
- **進捗**: 1/66 shards 完了 (v03、22:31 時点)
- **出力先**: `data/training/match_features_phase_e_v01-94_bc.csv` (未完成)
- **shards**: `data/training/phase_e_shards/shard_v??.csv`
- **ログ**: `logs/phase_e_bc_regen.log`

### 再開可能性

`scripts/phase_e_collect_indicator_dataset.py::_process_video_shard` には
shard 存在チェックの skip ロジックがあるため、再起動後に同コマンドを再実行すれば
完了済 shard はスキップされ続きから再開する。

---

## 3. 再起動後の手順

### 3-1. BC regen 再開
```bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
PYTHONPATH=. ./venv/bin/python -m scripts.phase_e_collect_indicator_dataset \
  --videos 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,40,43,48,49,50,51,52,54,57,61,63,64,65,66,70,73,74,75,77,78,79,80,81,84,86,89,91,92,93 \
  --max-matches 0 --fps 3 --workers 8 \
  --out-csv data/training/match_features_phase_e_v01-94_bc.csv \
  > logs/phase_e_bc_regen.log 2>&1 &
```

完了見込み: 約 8 時間。

### 3-2. BC 学習 (regen 完了後)
```bash
PYTHONPATH=. ./venv/bin/python -m scripts.phase_e_learn_phase_aware \
  --csv data/training/match_features_phase_e_v01-94_bc.csv \
  --out data/verify/learned_weights_phase_e_phase_aware_bc.json
PYTHONPATH=. ./venv/bin/python -m scripts.learn_weights_lgbm \
  --csv data/training/match_features_phase_e_v01-94_bc.csv \
  --out data/verify/learned_weights_lgbm_bc.json
PYTHONPATH=. ./venv/bin/python -m scripts.learn_weights_v3 \
  --csv data/training/match_features_phase_e_v01-94_bc.csv \
  --out data/verify/learned_weights_v3_bc.json
```

### 3-3. 評価ポイント
- **incoming_ojama_pressure** が GBM permutation importance に登場するか (C-2 効果)
- **harassment_resistance** の重みが変化したか (C-3 効果)
- LOOV avg が **0.65 以上**になるか (B-1+B-2: 0.643)

---

## 4. 残タスク

### Step 4 残り
- B-3 (W-κ score OCR 動画別キャリブ): 現状 OCR OK のため後回し
- B-4 (W-δ 回し入れ追跡): フレーム間 cell 消失追跡で複雑
- C-3 残り: chain_timing_pressure に opponent_board context 追加

### Step 5 (Tier C 大規模)
- C-1 (W-β + I-A 確率分布対応): 16 指標の API 拡張、最大規模改修
  - STABLE 状態では効果限定的
  - UNKNOWN セルが多い場合のみ寄与
- C-2 (完了)
- C-3 (部分完了)

### 期待される改善目標
- 現在 (B-1+B-2): 0.643 LOOV
- BC 完了後 (C-2+C-3): 0.65-0.68 (期待)
- 最終目標: 0.78+ (Tier C 完了 + 新指標 + ML 改良)

---

## 5. 主要ファイル参照

### 新規追加
- `src/form_templates.py` (B-1)
- `src/ojama_predictor.py` (C-2)
- `tests/test_form_templates.py` (B-1)
- `tests/test_indicators_form.py` (B-1)
- `tests/test_indicators_tier_b.py` (Tier B 高速化)
- `tests/test_ojama_predictor.py` (C-2)

### 修正
- `src/indicators.py`:
  - `_connected_components` キャッシュ化
  - `_CC_CACHE` モジュール変数
  - `BaseFlatnessIndicator` numpy 化
  - `StructureSolidityIndicator` numpy 化
  - `PlanningEntropyIndicator` numpy 化 + 再有効化
  - `ExtensionPotentialIndicator._search_improvement_2step` (B-2)
  - `HarassmentResistanceIndicator` に `incoming_ojama` 引数 (C-3)
  - I-J 4 indicator クラス追加 (B-1)
  - EXTRA_INDICATOR_NAMES に Tier B 3 + B-1 4 = 7 追加
  - IndicatorSet に Tier B 3 + B-1 4 = 7 フィールド追加
- `scripts/phase_e_collect_indicator_dataset.py`: OjamaPredictor 統合 (C-2)
- `src/scorer.py`: `LEARNED_WEIGHTS_PHASE_E_TIERB_*` + `weight_mode="phase_e_tierb"` 追加
- `tests/test_generate_training_dataset.py`: FEATURE_NAMES 21 → 28 に更新
- `tests/test_cell_recovery_refiner.py`: saturation 境界値 80 → 79 修正

### データ出力
- `data/training/match_features_phase_e_v01-94_tierb.csv` (Tier B、10,945 行 / 28 列)
- `data/training/match_features_phase_e_v01-94_b12.csv` (B-1+B-2、10,945 行 / 32 列)
- `data/training/match_features_phase_e_v01-94_bc.csv` (BC、未完成、再開対象)
- `data/verify/learned_weights_phase_e_phase_aware_tierb.json`
- `data/verify/learned_weights_lgbm_tierb.json`
- `data/verify/learned_weights_v3_tierb.json`
- `data/verify/learned_weights_phase_e_phase_aware_b12.json`
- `data/verify/learned_weights_lgbm_b12.json`
- `data/verify/learned_weights_v3_b12.json`
- `data/verify/phase_e_dashboard_tierb.md`

---

**END OF HANDOFF**

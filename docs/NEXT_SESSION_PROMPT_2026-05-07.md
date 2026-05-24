# 次セッション開始プロンプト (2026-05-07)

以下を新セッションのプロンプトに貼り付けて使用:

---

```
puyo_analyzer プロジェクトの引き継ぎセッションです。
2026-05-06 の PC 再起動による中断からの再開。

【まず読む】
1. docs/SESSION_HANDOFF_2026-05-06.md (詳細引継ぎ、必読)
2. memory の project_handoff_2026-05-06_tier_b.md
3. data/verify/learning_impact_audit.md (障害ステータス)

【中断時状態】
- Step 4-1 (B-1 形テンプレ + B-2 I-E 2-step) 完了:
  LR Phase LOOV 0.643 (旧 Tier B 0.637 から +0.006)
  GBM video holdout 0.653 (+2.5pt)
  form_staircase, form_zabuton が GBM importance top 10 入り
- Step 4-2 (C-2 W-γ おじゃま予告 + C-3 部分) 実装完了、regen 1/66 で中断
  出力先: data/training/match_features_phase_e_v01-94_bc.csv
  shard 再開可能 (_process_video_shard に存在 skip ロジックあり)

【まずやること】

Step A: Step 4-2 BC regen を再開
========================================
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
PYTHONPATH=. ./venv/bin/python -m scripts.phase_e_collect_indicator_dataset \
  --videos 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,40,43,48,49,50,51,52,54,57,61,63,64,65,66,70,73,74,75,77,78,79,80,81,84,86,89,91,92,93 \
  --max-matches 0 --fps 3 --workers 8 \
  --out-csv data/training/match_features_phase_e_v01-94_bc.csv \
  > logs/phase_e_bc_regen.log 2>&1 &

完了見込み: 約 8 時間 (前回 1/66 shard 完了済、残 65)

Step B: BC 学習 (regen 完了後)
========================================
PYTHONPATH=. ./venv/bin/python -m scripts.phase_e_learn_phase_aware \
  --csv data/training/match_features_phase_e_v01-94_bc.csv \
  --out data/verify/learned_weights_phase_e_phase_aware_bc.json
PYTHONPATH=. ./venv/bin/python -m scripts.learn_weights_lgbm \
  --csv data/training/match_features_phase_e_v01-94_bc.csv \
  --out data/verify/learned_weights_lgbm_bc.json

評価ポイント:
- incoming_ojama_pressure が GBM permutation importance に登場するか (C-2 効果)
- harassment_resistance の重みが変化したか (C-3 効果)
- LOOV avg が 0.65 以上になるか (B-1+B-2: 0.643)

Step C: 残タスク
========================================
1. C-3 残り: chain_timing_pressure に opponent_board context 追加
2. B-4 (W-δ 回し入れ追跡): 上級者試合用、優先度中
3. C-1 (W-β + I-A 確率分布対応): 16 指標 API 拡張、最大規模
4. B-3 (W-κ score OCR 動画別): 後回し OK

【最終目標】
LOOV avg 0.643 → 0.78+ (Tier C 完了 + 新指標 + ML 改良)

各 Step 完了時に進捗を memory に記録、テストを必ず実行してから次へ。
自律運転 OK (memory に記録済)、長時間放置前提で問題なし。
```

---

## 補足: 検証コマンド (引継ぎ後の整合性確認)

```bash
# Tests 全件確認
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_indicators.py tests/test_indicators_advanced.py tests/test_indicators_extra.py tests/test_indicators_phase_j.py tests/test_indicators_phase_k.py tests/test_indicators_tier_b.py tests/test_indicators_form.py tests/test_form_templates.py tests/test_ojama_predictor.py tests/test_scorer.py tests/test_generate_training_dataset.py -q
# 期待: 全 224 tests pass

# 既存出力ファイル確認
ls data/training/match_features_phase_e_v01-94_*.csv
ls data/verify/learned_weights_*_b12.json data/verify/learned_weights_*_tierb.json
ls data/verify/phase_e_dashboard_tierb.md

# Tier B 結果再確認
cat data/verify/learned_weights_phase_e_phase_aware_b12.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
for p in ['start','mid','end']:
    print(p, d['phases'][p]['loov_mean'])
"
```

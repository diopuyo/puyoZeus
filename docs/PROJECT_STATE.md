# プロジェクト状態 (詳細)

CLAUDE.md の補足。ディレクトリ構成、Phase 進捗、学習結果累積、参照 memory 等の詳細情報。

## ディレクトリ構成

```
puyo_analyzer/
├── CLAUDE.md                          # コアルール (短縮版)
├── docs/
│   ├── PROJECT_STATE.md               # 本ファイル
│   ├── INDICATOR_REFERENCE.md         # 45 指標フル定義
│   ├── INDICATOR_ROADMAP.md           # Phase H1〜L 詳細ロードマップ
│   ├── IMAGE_RECOGNITION_OVERVIEW.md  # 認識スタック完全説明 (1,039 行)
│   └── (旧 SESSION_HANDOFF_*.md 群)
├── src/
│   ├── board.py                       # 盤面データ表現
│   ├── probabilistic_board.py         # 確率分布盤面 (Phase G C-1)
│   ├── recognition_pipeline.py        # 統合 pipeline (state machine 主軸)
│   ├── board_state_machine.py         # 6 状態遷移
│   ├── image_reader.py                # フレーム→盤面変換 HSV+CNN
│   ├── patch_classifier.py            # GatedCnnClassifier
│   ├── hidden_row_inferrer.py         # 隠し段確率推論 (W-α)
│   ├── online_hsv_calibrator.py       # 動画別 HSV 自動学習 (Phase I 統合予定)
│   ├── chain.py                       # 連鎖シミュレーション + Probabilistic 拡張
│   ├── indicators.py                  # 45 評価指標
│   ├── scorer.py                      # Scorer + PhaseAwareScorer
│   ├── ojama_predictor.py             # おじゃま予告予測 (Phase F C-2)
│   ├── rotation_tracker.py            # 回し入れ追跡 (Phase F B-4)
│   ├── form_templates.py              # 形テンプレ (B-1)
│   ├── timeseries_indicator_wrapper.py # 時系列展開 (Phase H2)
│   ├── cnn_embedding/                 # Phase H4.2
│   │   ├── board_cnn.py               # BoardCNN, SiameseBoardCNN
│   │   └── __init__.py
│   ├── self_supervised/               # Phase I 進行中
│   │   ├── pseudo_label.py
│   │   ├── cross_validator.py
│   │   ├── label_store.py
│   │   ├── online_fine_tuner.py
│   │   ├── score_validator.py
│   │   ├── next_validator.py
│   │   ├── chain_validator.py
│   │   ├── reveal_tracker.py          # Hidden row (項目 4)
│   │   ├── hidden_row_validator.py
│   │   └── hidden_row_fine_tuner.py
│   ├── analyzer.py / overlay.py / video_compositer.py / cli.py
│   ├── stream_overlay.py              # OBS 用 (未検証)
│   └── (他 patch_extraction, board_rules, ui_mask, physics_sanity 等)
├── tests/                             # 90+ ファイル / 1,400+ テスト
├── scripts/
│   ├── phase_e_collect_indicator_dataset.py  # 5 phase × 1 snapshot 採取
│   ├── phase_h2_collect_indicator_dataset.py # 全 STABLE frame + 時系列展開
│   ├── phase_h2_collect_board.py             # board npz 同時収集
│   ├── phase_h2_learn.py                     # 280 features 自動学習
│   ├── phase_h3_ablation.py / mixed_effects.py / ensemble.py
│   ├── phase_h4_1_train.py / phase_h4_2_train.py
│   ├── phase_e_dl_playlists.py               # yt-dlp DL
│   ├── filter_top_tier.py                    # ティア filter
│   ├── visualize_recognition.py              # 認識可視化動画
│   └── learn_weights_v3.py / learn_weights_lgbm.py (旧)
├── models/
│   ├── cnn_phase_b_v1.pt              # 現役 (state machine 主軸、99.61%)
│   ├── cnn_phase_u_v17b.pt / v16.pt   # 動画別選択用
│   ├── cnn_phase_u_v6.pt              # 99.45%
│   └── ui_templates/
└── data/
    ├── frames/                        # video_NN.mp4 永続
    ├── training/
    │   ├── match_features_phase_e_v01-94_h1.csv      # H1 (66 動画混在)
    │   ├── match_features_phase_e_v29-94_h1.csv      # Top-tier filter (39 動画)
    │   ├── match_features_phase_h2_quick_phased.csv  # H2 quick (12 動画)
    │   ├── match_features_phase_h2_quick_with_board.csv # H4.2 (board npz 同時)
    │   └── phase_h2_boards/v??.npz                   # 11 動画 board state
    ├── verify/
    │   ├── learned_weights_*_h1.json                 # H1 結果
    │   ├── learned_weights_*_h1_topier.json          # Top-tier 版
    │   ├── phase_h{1,2,3,4_1,4_2}_dashboard.md
    │   ├── phase_h2_results.json
    │   ├── phase_h3_*.json
    │   ├── phase_h4_*.json
    │   └── learning_impact_audit.md                  # 残課題リスト
    ├── evaluation_videos/             # 評価/可視化動画
    │   ├── v40_match7_125s.mp4
    │   └── v40_match7_eval_viz.mp4    # STABLE 凍結+認識色 overlay
    ├── pseudo_labels/                 # Phase I 擬似ラベル蓄積
    └── phase_e_dl_index.tsv           # DL 動画タイトル index
```

## Phase 進捗 (累積)

```
✅ Phase 1 (盤面読取)     : 95% (CNN holdout 0.9266、追加レビュー余地あり)
✅ Phase 2 (指標+評価)    : 100% (16 指標、Scorer 多段階)
✅ Phase 3 (アプリ統合)   : 90% (analyzer/overlay/cli/video_compositer 完成、stream_overlay 未検証)
✅ ML チューニング        : 100% (38 動画、Phase E PhaseAware 0.659)

✅ Phase F (BCD)          : C-3 残り (3 indicator opponent_board) + B-4 (rotation_skill) 完了
✅ Phase G (C-1)          : ProbabilisticBoard foundation + 4 indicator override 完了
                            W-α HiddenRowInferrer pipeline 統合完了
                            残り 12 indicator は MLE 委譲 (Phase L で本格化検討)
✅ Phase H quick cycle (12 動画):
   ✅ H1 (機能能力 + 戦況 + 形補助 16 指標追加)
   ✅ H2 (時系列展開 + interaction features = 280 features)
   ✅ H3 (弱指標削除 + Mixed-effects 検証 = S=20 が最良)
   ✅ H4.1 (Deep Tabular MLP、LOOV 0.762)
   ✅ H4.2 (Raw Board CNN、データ不足、Phase L で再評価)

🔄 Phase I 認識精度 99.99% (最優先、進行中):
   - 自己教師あり学習: Score OCR + Next/dnext + Cell color + ChainEvent
   - + Hidden row 自己学習 (項目 4、パラレル実装中)
   - 共通 framework + 4 validator + online fine-tune
   - OnlineHsvCalibrator 統合 (memory `realtime_hsv` 段階 2)

⏸️ Phase L 本番化 (Phase I 完了後):
   - yt-dlp で動画追加 DL (66 → 100-150)、ティア filter 厳守
   - 全動画 regen + CNN 事前学習 + 蒸留 (H4.3 相当)
   - Mixed-effects model + ensemble 再評価

⏸️ Phase J オーバーレイ統合 (Phase L 後)
⏸️ Phase K 配信実証 (大会試験運用)
```

## 学習結果累積

| 指標 | LR vh | LR LOOV avg | GBM video | end phase |
|---|---:|---:|---:|---:|
| BCD baseline (66動画混在) | 0.663 | 0.652 | 0.679 | 0.851 |
| H1 (66動画混在) | 0.694 | 0.672 | 0.689 | 0.927 |
| H2 quick (12動画 時系列) | 0.740 | 0.667 | 0.672 | 0.866 |
| H3 (S=20 ablation) | **0.757** | 0.657 | 0.699 | 0.893 |
| H4.1 MLP top_20 | 0.730 | **0.762** ★ | - | 0.938 |
| **Top-tier H1 (39動画)** | 0.690 | **0.705** ★ | 0.658 | **0.964** ★ |

**累積最高**: LR LOOV 0.705 / MLP LOOV 0.762 / GBM end 0.968

### 重要発見

- **ティア filter は LOOV と end phase に大幅効果** (+0.033 / +0.041)
- 機能能力指標 (ojama_defense_capacity rank 3、upper_board_density rank 2) が想定通り効く
- Mixed-effects model は video holdout で逆効果 (-0.111)
- Ensemble は LR 単独最強、tree models 混入で劣化
- 280 features → top 20 に削減で **+0.017 改善** (LR vh)
- LR > GBM > Random Forest (時系列特徴は線形でよく扱える)
- Raw board CNN は 11 動画では underperform、Phase L で再評価

## State Machine (認識の核)

`src/board_state_machine.py` の `BoardState`:
- `MENU` — 試合外 (タイトル / リザルト)
- `STABLE` — 平常時、CNN 出力を盤面確定に使用 ★ **評価対象**
- `TSUMO_FALL` — ツモ落下中、物理推論
- `CHAIN` — 連鎖中 (消去 + 重力)、物理推論
- `OJAMA_FALL` — おじゃま落下中、物理推論
- `EFFECT` — 全消し演出 / 連鎖カットイン (skeleton、現状常に None)

**重要**: 指標評価は **両者 STABLE 時の `confirmed_board` のみ** で行う。
NON-STABLE 中は前回 STABLE 盤面を凍結。

## サンプリングルール

- **盤面情報保持**: `BOARD_INTERVAL_SEC = 0.2` 秒/フレーム (5 fps 相当)
- **有利不利評価**: `EVAL_INTERVAL_SEC = 0.6` 秒/フレーム (≈1.67 Hz)
- **安定判定**: `STABLE_FRAME_COUNT = 6` × `consec_threshold = 2` 連続一致

## 試合境界・勝敗判定

- `count_match_v4`: video_02 で 50/50 完全一致達成 (`match_boundaries_v4/`)
- `match_winner`: 数値画像差分 (16×16 二値ハッシュ + Hamming 距離) で OCR 不要、
  video_02 で 50/50 全成功
- `match_state`: HSV V 平均で試合中 / メニュー判定
- `win_panel` / `score_zero`: ★WIN★ パネル NCC マッチ + 両側ゼロでリセット検出

## ネクスト・ダブルネクスト検出

- `next_detector`: 1P/2P 両対応、HSV + CNN ハイブリッドで青/赤背景バイアス除去
- ROI は 1920×1080 ハードコード (v10 確定)、別 UI では再キャリブ必須
- 1P 用 ROI 確定、2P 用は中央 x=960 で水平ミラー

## 既知の制約

- 動画間プレーヤー戦術差で線形 ML は LOOV std 0.07-0.09 で頭打ち
- ネクスト ROI は UI レイアウト依存、別大会では再キャリブ必要
- ffmpeg 不在環境では音声 mux 不可 (`--no-audio` 強制)
- RTX 4060 Laptop 8GB は CNN 完全版でぎりぎり (mixed precision 推奨)
- 11-12 動画 quick mode は LOOV variance 大、絶対値より方向性で判定

## メモリ参照

CLAUDE 起動時に `MEMORY.md` 経由で以下を参照:
- `project_phase_h1_results.md` — Phase H1 結果 + 方針確定
- `project_phase_i_kickoff.md` — Phase I 自己教師あり学習計画
- `feedback_autonomous_operation.md` — 自律運転前提
- `feedback_chain_phase_physics_only.md` — STABLE 以外で CNN 信用しない
- `feedback_recognition_target_995.md` — 99.99% 認識目標
- `recognition_strategy_pivot.md` — state machine 主軸
- `realtime_hsv` — Online HSV calibrator 段階 2 (Phase I 統合対象)
- `feedback_msys_pipe_escape.md` — MSYS 特殊文字回避
- `project_ojama_inference_design.md` — おじゃま推論優先度
- `feedback_priority_overlay_vs_rl.md` — RL 優先、UI 後回し

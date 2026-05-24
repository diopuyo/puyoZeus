# セッション引継ぎ 2026-04-29 (Phase V 完成)

## 1 行サマリ

Phase V 完成: **CNN v7 multi-video 訓練 (Holdout 99.8%)、ロジック層強化 (V2.1-2.4)、テロップ/試合終了対応 (V3.1-3.2)、全テスト 996 pass**。

## Phase V 達成内容

### V-1 多様性強化 (CNN v7)
- 動画ソースは既存の **video_01〜03 (3 配信者) + 過去 bulk DL の pl1〜pl4 (130 動画相当)** で確保
- parallel/ 弱ラベル 700万 を CNN v6 で再ラベル + 閾値 0.95 + HSV 双方一致で **500万採用 (誤認 1.4%)**
- manual_aug20 + parallel_strict ミックス = 451,980 件
- CNN v7 訓練 (均衡化 84,000、init=v6、25 epochs) → Holdout **99.8%**

### V-2 ロジック層強化
- **V2.1 NextLinkedColorRefiner**: t-1 next_pair で t 新規ペア色を強制整合
- **V2.2 PairAppearanceConsistency**: 1 セル/3 セル新規出現の不整合検出 (補正なし)
- **V2.3 ConnectivityShapeRefiner**: 隣接 4 セル中 3 セル以上同色なら異色 1 セル補正
- **V2.4 EnhancedBoardTracker**: V2.1+V2.3+StatefulBoardTracker を統合した時系列フィルタ

### V-3 構造的限界対処
- **V3.1 テロップ被覆 UNKNOWN 化**: TelopDetector に cells_covered_by_bbox 追加、ImageReader に use_telop_mask 統合
- **V3.2 試合終了告知検出**: 「やった!」「ばたんきゅー」NCC マッチ + 5 秒ロックダウン管理

## 認識精度 (動画ソース別)

| ソース | accuracy |
|---|---|
| Holdout (混合データ) | **99.8%** |
| manual_video_01 (4771 件) | 98.40% |
| pl1 (35 動画) | 100.00% |
| pl3 (54 動画) | 99.95% |
| pl4 (30 動画) | 100.00% |

video_01 は v6 99.45% から微減だが、別動画 (pl1/3/4) で大幅向上。multi-video 汎化の代償としてのトレードオフ。

## 採用構成 (`phase_u_render.py` のデフォルト)

```
ImageReader(
  classifier=HybridClassifier(
    cnn_classifier=CnnPatchClassifier (cnn_phase_u_v7.pt),  # ← 本流
    use_ui_mask=True,
  ),
  use_match_state=True,
  use_ui_mask=True,
  use_telop_mask=True,   # V3.1
)
+ AnimationFilter (P1/P2 別)
+ MatchEndDetector (V3.2、phase_u_render に統合は別途 task)
+ EnhancedBoardTracker (V2.4、phase_u_render に統合は別途 task)
```

## 主要ファイル (Phase V 追加)

| パス | 役割 | テスト数 |
|---|---|---|
| `src/next_linked_refiner.py` | V2.1 | 10 |
| `src/pair_appearance.py` | V2.2 | 6 |
| `src/connectivity_refiner.py` | V2.3 | 10 |
| `src/enhanced_board_tracker.py` | V2.4 | 8 |
| `src/telop_detector.py` (拡張) | V3.1 cells_covered_by_bbox | 11 |
| `src/match_end_detector.py` | V3.2 | 9 |
| `src/image_reader.py` (拡張) | use_telop_mask | 含む |
| `scripts/phase_u_relabel_parallel.py` | parallel/ 再ラベル + HSV match | - |
| `scripts/phase_u_build_dataset_v7.py` | multi-source データ統合 | - |
| `scripts/phase_u_eval_v7_per_source.py` | 動画別 holdout 評価 | - |
| `scripts/phase_u_relabel_verify_sheet.py` | 再ラベル検証シート生成 | - |
| `models/cnn_phase_u_v7.pt` | **本流 CNN モデル** | - |
| `models/ui_templates/match_end_yatta.png` | V3.2 テンプレ | - |
| `models/ui_templates/match_end_batan.png` | V3.2 テンプレ | - |

## データセット

| パス | 件数 | 用途 |
|---|---|---|
| `data/training_phase_u/manual_labels.npz` | 4,771 | 元手動ラベル (video_01) |
| `data/training_phase_u/manual_labels_aug20.npz` | 94,980 | x20 augment (CNN v6/v7 訓練に使用) |
| `data/training_phase_u/parallel_relabeled/` | 6,281,519 | 弱フィルタ (0.9)、誤認 15.7% で **使用しない** |
| `data/training_phase_u/parallel_relabeled_strict/` | **4,990,659** | 強フィルタ (0.95+HSV match)、誤認 1.4%、本流 |
| `data/training_phase_u/manual_plus_strict.npz` | 451,980 | CNN v7 訓練データ |

## 残タスク (Phase W 候補、優先順)

1. **V4.1**: phase_u_render.py に EnhancedBoardTracker 統合 (現状は ImageReader 直呼び)
2. **V4.2**: phase_u_render.py に MatchEndDetector 統合 (ロックダウン中は前盤面保持)
3. **V1.6** (option): video_02/03 シート (12 × 50 = 600 セル) ユーザーレビュー → CNN v8
4. リアルタイム配信オーバーレイ (stream_overlay) の OBS 統合検証
5. 別大会・別年代動画の追加 (UI レイアウト多様性)

## 再開コマンド

```bash
# 1. 引継ぎ確認
cat docs/HANDOFF_2026-04-29_PHASE_V_FINAL.md

# 2. 全テスト走行 (約 3.5 分)
wsl -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m pytest tests/ -q"

# 3. CNN v7 動画別評価
wsl -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m scripts.phase_u_eval_v7_per_source --n-sample 2000"

# 4. 動画レンダ (CNN v7 デフォルト)
wsl -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && PYTHONPATH=. ./venv/bin/python -m scripts.phase_u_render \\
    data/verify/review_videos/clip_v01_m34.mp4 \\
    data/verify/review_videos/phase_u_v7_m34.mp4 \\
    --interval 0.2 --max-seconds 15"
```

## 設計判断 (Phase V 追加)

1. parallel/ 弱ラベルは「閾値 0.9」では誤認 15.7% で使用不可 → 「閾値 0.95 + HSV match」で 1.4% に圧縮
2. CNN v6 が video_02 等の別背景キャラを RED/OJM と高信頼度で誤判定する問題は HSV 双方一致で除去
3. EnhancedBoardTracker は既存 StatefulBoardTracker を破壊せず外側にラップ
4. V2.2 (ペア整合性) は補正せず検出のみ (誤補正リスク回避)
5. CNN v7 で manual_video_01 が 99.45→98.40% に微減したが multi-video トレードオフとして許容
6. テロップ検出は **bbox 単位**でセル被覆判定 (TelopDetector.cells_covered_by_bbox)
7. 試合終了告知は matches.tsv に依存しないテンプレ NCC マッチで動作 (リアルタイム可)

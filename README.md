# PuyoZeus

ぷよぷよ eスポーツ動画から盤面を画像認識し、 有利不利を判定するシステム。

最終目標: 配信オーバーレイ + 機械学習による戦略評価。

## 主要機能

- **画像認識 pipeline**: HSV + CNN ハイブリッド方式 (= 1920×1080 動画 → 6×13 盤面)
- **state machine**: STABLE / TSUMO_FALL / CHAIN / OJAMA_FALL / EFFECT の 5 状態遷移
- **物理推論ベース評価フレーム** (= recognition_evaluator): 12 メトリクスで fail-silent 検知
- **45 指標** ベース有利不利スコア (= scorer.py)

## 必要環境

- Python 3.11+ (= 開発実環境 3.12)
- CUDA 12.x + NVIDIA GPU 8GB+ VRAM 推奨
- 主要 dep: `opencv-python`, `numpy`, `torch`, `scikit-learn`, `imageio-ffmpeg`

## セットアップ

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## データ準備 (= ユーザー自前)

**本リポジトリには版権関連データ (= ぷよぷよ動画、 学習済 model 等) は含まれません。**
以下を自前で準備:

1. **動画ファイル** (= マスター級以上の対戦動画):
   ```
   data/frames/video_01.mp4
   data/frames/video_02.mp4
   ...
   ```
2. **試合境界 TSV** (= 試合切り出し用):
   ```
   data/verify/match_boundaries_v5/video_NN/matches.tsv
   ```
   `scripts/_list_match_boundaries.py` 等で自動生成可能。

## 推論実行 (= 認識デモ)

```bash
PYTHONPATH=. python -m scripts.visualize_recognition \
    --video data/your_video.mp4 \
    --output data/output.mp4 \
    --cnn-model models/your_model.pt \
    --dump-board-log logs/board_log.jsonl
```

## 評価 (= 強化アナリスト)

```bash
PYTHONPATH=. python -m scripts.evaluate_recognition \
    --board-log logs/board_log.jsonl \
    --report-out data/verify/your_eval.json
```

12 メトリクスで物理推論ベース自動評価。 verdict が REJECT / REVIEW / ACCEPT。

## CNN 学習

詳細は `docs/INDICATOR_ROADMAP.md` 参照。

主要 pipeline:
1. 動画から試合切り出し (= 15 秒バッファ必須、 重要ノウハウ)
2. seed 抽出 (= `scripts/extract_hsv_seed_dataset.py`)
3. CNN fine-tune (= `scripts/phase_i_fine_tune.py`)
4. viz 評価 + 強化アナリスト

## 主要設計思想

- **「形は手段、 機能が本質」**: GTR / サブマリン等の形分類は二次、 機能・能力指標が一次
- **STABLE 確定盤面のみで評価**: NON-STABLE 中は前回 STABLE 凍結
- **viz 目視併用必須**: 数値だけで採否決めない (= 強化アナリスト + ユーザー目視)
- **試合切り出し 15 秒バッファ必須**: state machine 初期化のため

## ドキュメント

- `CLAUDE.md`: コアルール (= プロジェクト全体方針)
- `docs/CYCLE_FINDINGS.md`: cycle 検証で確定したルール
- `docs/PROJECT_STATE.md`: 進捗状態
- `docs/INDICATOR_REFERENCE.md`: 45 指標フル定義
- `docs/INDICATOR_ROADMAP.md`: Phase I-L 詳細ロードマップ

## ライセンス

(= 後で追記)

## 補足

ぷよぷよ™ eスポーツ は SEGA / Compile の登録商標。 本プロジェクトは個人研究目的で、 SEGA / Compile の公式プロジェクトではありません。

# ぷよぷよeスポーツ 有利不利判定システム - コアルール

## プロジェクト概要

ぷよぷよeスポーツの **上級者対戦** (マスター級以上) を対象に、対戦動画から
画像認識で盤面を読み取り、**45 指標 + 機械学習チューニング** で有利不利を判定するシステム。

**最終目標**: リアルタイム配信オーバーレイツール (OBS ブラウザソース対応)。
データ駆動の有利不利判定を通じて、人間想定を超える戦略評価の発見を目指す。

## スコープ (確定)

- **上級者対戦専用**: A 級 / マスター / チャレンジャー / S 級の動画のみ
- **データ filter 必須**: 動画 ID 別ティア確認、未確認動画は学習から除外
- **目的**: データ駆動有利不利判定 → 人間想定超えの戦略評価発見

## 対象ゲーム

**ぷよぷよeスポーツ (Puyo Puyo Champions)** — 16:9, 1920×1080 想定。
入力動画が他解像度の場合は 1920×1080 にリサイズして認識する。

## 技術スタック

- **言語**: Python 3.11+ (実環境は 3.12)
- **GPU**: NVIDIA RTX 4060 Laptop (CUDA 12.x)、8GB VRAM
- **画像処理**: opencv-python, numpy, Pillow
- **テスト**: pytest (現在 1,400+ tests pass)
- **動画 DL**: yt-dlp
- **機械学習**: scikit-learn 1.8.0 + statsmodels 0.14.6 (Mixed-effects)
- **CNN**: torch, torchvision
- **動画 mux**: imageio-ffmpeg (バンドル)、不在環境では `--no-audio` 強制

## 盤面データ表現

- **6 列 × 13 行** の `numpy.ndarray`
- 行: 0=最上段(隠し段), 12=最下段 / 列: 0=左端, 5=右端
- 色: `0`=空, `1`=赤, `2`=青, `3`=緑, `4`=黄, `5`=紫, `9`=おじゃま, `10`=COLOR_UNKNOWN
- 窒息判定: 3 列目 (index:2) の可視最上段 (row=1、12段目) にぷよがあれば DEAD。隠し段 (row=0、13段目) は判定に含まない (DEATH_ROW=1、2026-07-22是正)
- **ProbabilisticBoard** (`src/probabilistic_board.py`): 各 cell に確率分布

## 設計思想 (絶対遵守)

### 1. 「形は手段、機能が本質」
GTR / サブマリン / 階段 等の形分類は二次的。火力・中盤厚み・お邪魔体制等の
**「機能・能力」を直接測る指標が一次**。

### 2. 「観測軸を提供 → 学習で重要度を発見」
人間が「これが重要」と決めない。戦略概念をデータ観測指標に翻訳し、
**重要度・閾値は学習結果から ranking で発見**。

### 3. End-to-end CNN base + 補助 indicator (確定)
最終的に CNN が直接「盤面 → 勝率」を学ぶ。indicator は auxiliary supervision target。

### 4. STABLE 確定盤面のみで評価
指標評価は **両者 STABLE 時の `confirmed_board` のみ** で実行。NON-STABLE 中は
前回 STABLE 盤面を凍結 (memory `feedback_chain_phase_physics_only.md`)。

### 5. 認識精度目標 (2026-08-06 user改定)
Phase I の合格ライン = **99.5%** (STABLE確定盤面・物差し測定、2026-08-06実測99.54%でクリア)。
**99.99% は全体完了後の仕上げ目標に後置** (Stage2バックログ: 弱光較正/煙/素通り/row0/一般単発誤読)。
長時間劣化 (第4機構) の修正はデータ品質要件として合格ラインと独立に必須。

## 学習データ条件 (徹底)

### ティア filter 必須
- **使用可**: A 級 / マスター / チャレンジャー / S 級の動画のみ
- **判定方法**: `data/phase_e_dl_index.tsv` でタイトル確認 (v29-v94 は全マスター級確認済)
- **v01-v28 は除外**: 古い DL 流でタイトル不明、ティア未確認 → ノイズ
- **v22 等 boundary なし動画**: skip 済 (matches.tsv 不在)
- フィルタスクリプト: `scripts/filter_top_tier.py`

### 動画追加時の確認手順
1. `phase_e_dl_index.tsv` でタイトル確認
2. キーワード check: `マスター / チャレンジャー / S級 / A級 / GP / プロ / オフライン / 決勝 / 大会`
3. 該当しない動画は学習データに含めない

## コーディング規約

- **常に日本語で応答し、日本語でコメントすること**
- 型ヒント必須
- **1 関数 50 行以内**
- マジックナンバー禁止 → 定数定義
- 分類器・モデルは差し替え可能な抽象設計
- **指標は 0〜1 正規化必須**
- **重み追加時は既存 `LEARNED_WEIGHTS_*` を破壊しない** (互換維持)
- **新指標追加時は `EXTRA_INDICATOR_NAMES` の末尾に追加** (順序保持)
- **backwards compat 必須**: 既存 API シグネチャに optional 引数のみ追加可、削除不可
- 観測指標は **stateless 実装** を原則 (state-holding は外部 wrapper)

## プロセス管理ルール

### 長時間タスクの起動
WSL 経由で setsid -f detach し、Claude セッション終了後も継続:
```bash
wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && \
  setsid -f bash -c 'PYTHONPATH=. ./venv/bin/python -m scripts.XXX > logs/XXX.log 2>&1 < /dev/null'"
```

### MSYS パイプ・特殊文字注意 (memory `feedback_msys_pipe_escape.md`)
Git Bash 経由 wsl で以下が壊れる:
- `tr -d " "` の引数 (スペース消失)
- `grep -E "a|b"` の `|` (パイプ解釈)
- `awk "{print \$1}"` の `\$` (escape)

回避策: 単純コマンドで代替 (`pgrep -c -f pattern` 等)、複雑なパイプはスクリプトファイル化 → `MSYS_NO_PATHCONV=1 wsl bash <linux-path>`。

### WSL --shutdown は実行しない (memory `feedback_wsl_restart_delegation.md`)
ユーザーに依頼する。自分で実行禁止。

### 自律運転 (memory `feedback_autonomous_operation.md`)
puyo_analyzer プロジェクトでは許可確認なしで自律実行。長時間放置前提。

## ストレージ管理ルール

- **動画ファイルは処理後に削除する** (ストレージ節約)
- URL とメタデータは `data/phase_e_dl_index.tsv` に記録、永続保持
- 盤面データ (Board JSON) は `data/boards/` に永続保持

## 出力形式

### A. 配信オーバーレイ (Phase J、Phase L 後に着手)
- OBS Studio のブラウザソース、WebSocket リアルタイム更新

### B. 動画合成
- 元動画にオーバーレイを合成、音声トラック保持

### C. 認識可視化動画 (デバッグ用)
- `scripts/visualize_recognition.py`: STABLE 凍結+認識色 overlay (評価と同条件)

## 有利不利スコア
- 範囲: **-100 〜 +100** (+100 = 1P 圧倒的有利)
- `EVEN_THRESHOLD = 3.0` 以下は EVEN 判定

## テスト
- `python -m pytest tests/ -v`
- 現在 **1,400+ テスト** 全パス
- 1 ファイル実装 → テスト通過 → 次のファイルへ
- 各 Phase で 30-50 新規テスト追加

## 詳細リファレンス (深掘り時参照)

| ドキュメント | 内容 |
|---|---|
| `docs/CYCLE_FINDINGS.md` | **cycle 検証で確定したルール** (cnn_override_prob 0.70 維持、 並列上限 3、 backup 必須ファイル等) — cycle を回す前に必読 |
| `docs/PROJECT_STATE.md` | ディレクトリ構成、Phase 進捗、学習結果累積、メモリ参照 |
| `docs/INDICATOR_REFERENCE.md` | 45 指標フル定義、設計思想、学習重要度 |
| `docs/INDICATOR_ROADMAP.md` | Phase H1〜L 詳細ロードマップ |
| `docs/IMAGE_RECOGNITION_OVERVIEW.md` | 認識スタック完全説明 (1,039 行) |
| `data/verify/learning_impact_audit.md` | 残課題リスト |
| `data/verify/phase_h*_dashboard.md` | 各 Phase の詳細結果 |

## メモリ参照 (重要)

`MEMORY.md` 経由で:
- `project_phase_h1_results.md` — Phase H1 結果 + 方針確定
- `project_phase_i_kickoff.md` — Phase I 自己教師あり学習計画
- `feedback_autonomous_operation.md` — 自律運転前提
- `feedback_chain_phase_physics_only.md` — STABLE 以外で CNN 信用しない
- `feedback_recognition_target_995.md` — 99.99% 認識目標
- `recognition_strategy_pivot.md` — state machine 主軸
- `realtime_hsv` — Online HSV calibrator 段階 2 (Phase I 統合対象)

## 現在のフェーズ

🔄 **Phase I 最終盤** (合格ライン99.5%は実測99.54%でクリア済み、2026-08-06 user改定): 残条件=長時間劣化修正A'の検証+全域無悪化ゲート → 合格で Phase L 解禁。99.99%は全体完了後の仕上げ目標
- 自己教師あり学習: Score OCR + Next/dnext + ChainEvent + Hidden row
- 共通 framework + 4 validator + online fine-tune
- OnlineHsvCalibrator 統合
- 完了見込み: 翌朝 02:00-04:00

⏸️ **Phase L 凍結中**: 本番化 (Phase I 完了後)
- yt-dlp で動画追加 DL (66 → 100-150)、ティア filter 厳守
- 全動画 regen + CNN 事前学習 + 蒸留

詳細: `docs/INDICATOR_ROADMAP.md`、`docs/PROJECT_STATE.md`

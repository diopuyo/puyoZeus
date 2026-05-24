# オフライン参照ドキュメント

オフライン環境用に、ぷよぷよ分析プロジェクトの残作業で必要になりそうな情報を集約。
**作成日: 2026-04-25 / 情報源: Web 調査（訪問可能なオンライン参考資料への URL 付き）**

---

## 1. ぷよぷよ公式得点計算（通ルール、再確認）

### 得点式

```
step_score = (消したぷよ数) × 10 × clamp(連鎖ボーナス + 連結ボーナス + 色数ボーナス, 1, 999)
total_score = Σ step_score + 全消しボーナス × 発火回数
```

**重要な仕様** (Puyo Nexus Scoring wiki):
- `(CP + CB + GB)` は **1 ～ 999** にクランプ（最小 1、最大 999）
- 現状の `src/scoring.py` は 1 クランプのみで 999 クランプが無い → **追加推奨**

### 連鎖ボーナス (CP) テーブル（通ルール）

```python
CHAIN_POWER_TABLE = (0, 8, 16, 32, 64, 96, 128, 160, 192, 224,
                     256, 288, 320, 352, 384, 416, 448, 480, 512)
# 20 連鎖以降は +32 ずつ線形増加
```

### 連結ボーナス (GB)

```python
{4:0, 5:2, 6:3, 7:4, 8:5, 9:6, 10:7, 11+:10}
```

### 色数ボーナス (CB)

```python
{1:0, 2:3, 3:6, 4:12, 5:24}
```

### おじゃまぷよ基準レート

- 通常モード: **70 点 / 1 個**
- フィーバーモード: **120 点 / 1 個**（本プロジェクトでは非対応）

### マージンタイム

Puyo Nexus wiki より確定値:
- **開始 96 秒**までは基準レート
- 以降 **16 秒ごとに ×3/4 (0.75)** でレート減少
- **最大 14 回**の減衰、または rate=1 で停止（以降変化なし）
- 14 回減衰で rate ≈ 70 × 0.75^14 ≈ 1.3（→ 1 クランプ）
- `src/scoring.py` 現状は 4 クランプ → **1 クランプに修正推奨**

### 全消しボーナス

- Puyo Nexus: 全消し時「次の連鎖発火時」に **30 おじゃま相当** 追加
- 30 × 70 = **2100 点相当**（通常モードレート基準）
- 実装: 即時 2100 点加算で近似（本プロジェクトのやり方で OK）

**Sources**:
- [Scoring - Puyo Nexus Wiki](https://puyonexus.com/wiki/Scoring)
- [Margin time - Puyo Nexus Wiki](https://puyonexus.com/wiki/Margin_time)
- [Chain Power Table - Puyo Nexus Wiki](https://puyonexus.com/wiki/Chain_Power_Table)
- [Tsu rule - Puyo Nexus Wiki](https://puyonexus.com/wiki/Tsu_(rule))
- [得点 - 壱大整域](https://alg-d.com/game/puyo/taisen11.html)
- [得点計算 - ぷよブロ](https://puyo-euphonic.com/puyo-word-score-calculation)

---

## 2. ぷよぷよ戦術用語（評価指標のリファイン用）

### 主要な戦術概念

| 用語 | 英語 | 意味 | 実装優先度 |
|---|---|---|---|
| 本線 | Main chain | 大連鎖（最終攻撃）用に組む長連鎖 | ◎ 既存 |
| 副砲 | Secondary chain | 本線と独立した小連鎖。催促・催促対応・合体で使う | ◎ 既存 |
| 催促 | Saisoku / prompt | 中盤に副砲を発火して相手の本線を引き出す行為 | ○ |
| 凝視 | Gyōshi / observation | 相手の盤面を観察して対応を決める技術 | △（2P 分析で近似） |
| 折り返し | Ori-kaeshi | 左右折り返し構造の本線形 | △ |
| 合体 | Gattai / merge | 副砲を本線と繋げて連鎖数を増やす | △ |
| 潰し | Tsubushi | 少量おじゃまで相手の形を崩す | △ |
| 対応 | Taiō / response | 相手の催促・攻撃に適切に応じる | △ |
| 対応連鎖 | Response chain | 相手の攻撃数に合わせて発火できる連鎖 | △ |

### 既存指標との対応

`src/indicators.py` の 8 指標は以下と対応済み:

1. **本線完成度** ← 本線
2. **伸ばし余地** ← 本線の拡張性
3. **副砲の質** ← 副砲
4. **催促耐性** ← 催促対応
5. **窒息リスク** ← 窒息判定
6. **相殺力** ← 現時点で発火した場合のおじゃま返し
7. **セカンド構築力** ← 合体候補
8. **フィールド効率** ← 盤面活用度

### ぷよぷよ AI 評価関数の参考

- 「本線と副砲を別々に評価し、それぞれ最善を 1 つずつ選ぶ」アプローチが主流
- 本線スコア: `連鎖数 × 1000` 的な単純線形加算
- 副砲スコア: `2 連鎖 × 1000`, `3 連鎖 × 500` など特定連鎖数に報酬
- 潰し・対応・催促を自然に誘発するような重みづけ

### 主要 OSS Puyo AI

- [ama (Puyo Puyo Tsu AI)](https://github.com/citrus610/ama) - C++、bitfield 高速実装、puyoai 由来の評価関数
- [puyoai](https://github.com/puyoai) - 日本の研究コミュニティの AI 集合
- mayah, takapt, niina - SlideShare の論文で言及

**Sources**:
- [ぷよぷよ AI 人類打倒に向けて (SlideShare)](https://www.slideshare.net/slideshow/puyoai-gpw2015/55101538)
- [ぷよぷよ AI の新しい探索法 (SlideShare)](https://www.slideshare.net/slideshow/ai-52214222/52214222)
- [催促 - 壱大整域](https://alg-d.com/game/puyo/taisen1.html)
- [凝視 - 壱大整域](http://alg-d.com/game/puyo/taisen3.html)
- [ぷよぷよ用語集 - ぷよブロ](https://puyo-euphonic.com/puyo-category-word)
- [中盤戦術技術徹底攻略 - ぷよぷよキャンプ](https://puyo-camp.jp/posts/89527)

---

## 3. 窒息ルールの正式仕様

- **窒息位置**: 盤面 3 列目・4 列目の**下から 12 段目**（= 上から最上段 or 隠し段）
- 通常は 3 列目（index=2）1 箇所が死亡判定セル
- ×マーク（UI）がフィールド上部 col 2 に表示されている

本プロジェクトの `DEATH_COL = 2, DEATH_ROW = 0` は正しい。

**Sources**:
- [基本ルール - 右脳式ぷよぷよ](http://puyo-puyo.info/puyoquest/q0_02rule.htm)
- [勝敗の決定 - 右脳式ぷよぷよ](https://www.ne.jp/asahi/root/inoue/107.htm)

---

## 4. 動画からの連鎖検出・解析

### 既存参考実装

- [puyo-chain-detector (GitHub)](https://github.com/puyogg/puyo-chain-detector)
  - 配信者向けリアルタイム連鎖発火点オーバーレイ
  - 動画フレームから発火点を検出して Twitch/YouTube オーバーレイに表示
- [PuyoSim (Puyo Nexus)](https://github.com/puyonexus/puyosim)
  - Web ベース連鎖シミュレータ、JavaScript
- [s2lsoftener/Puyo-Simulator](https://github.com/s2lsoftener/Puyo-Simulator)
  - 初期の連鎖シミュレータ、TypeScript 版も別 repo

### 日本語圏の動画解析ツール

- [ぷよっと解析くん](https://play.google.com/store/apps/details?id=com.sunmo.puyottoanlyzer) - Android アプリ、スクショから連鎖可能位置をオーバーレイ
- [YGGDRASILL SOFT 大連鎖チャンスあならいざあ](http://www5d.biglobe.ne.jp/~yggsoft/software/puyoana/index.html) - デスクトップ連鎖解析ツール

### リアルタイム動画解析の一般論

- OpenCV での画面キャプチャ + 検出が基本
- GPU 利用でリアルタイム処理（30fps+ 狙う場合は特に）
- YOLO 系は物体検出向けだが、連鎖検出には差分法＋テンプレマッチの方が向く
- フレーム間の盤面差分で「消去イベント」を検出（本プロジェクトの `VideoChainTracker` と同アプローチ）

**Sources**:
- [puyo-chain-detector - GitHub](https://github.com/puyogg/puyo-chain-detector)
- [対戦ゲームの動画記録から自動クリップ生成](https://blog.eqseqs.work/2021/12/20/000000/)
- [Flood Fill BFS Python - GitHub](https://github.com/imaddde867/BFS-Floodfill)

---

## 5. Phase 3 用: OBS Studio オーバーレイ統合

### アーキテクチャ選択肢

**A. obs-websocket 直接制御**
- OBS Studio に標準搭載された WebSocket v5 (port 4455)
- Python SDK: `obsws-python`
- OBS のソース表示切替・テキスト更新など remote 制御可能

**B. Flask/FastAPI + Browser Source**（推奨）
- ローカル Flask サーバーを立てて HTML/CSS/JS のオーバーレイを提供
- OBS の Browser Source でローカル URL を表示
- リアルタイム更新は WebSocket で client プッシュ

### 推奨ライブラリ

- `obsws-python` - OBS WebSocket v5 の Python SDK
- `flask` + `flask-socketio` or `fastapi` + `websockets` for browser source
- `aiohttp` も軽量選択肢

### 実装アウトライン

```python
# サーバ側（flask + socketio）
@socketio.on("connect")
def on_connect():
    emit("board_update", {...})

# 画面分析ループから
socketio.emit("board_update", {
    "score": 1P_score,
    "chain": chain_count,
    "advantage": -42,  # -100 〜 +100
})

# OBS Browser Source: http://localhost:5000/overlay
# HTML 側で websocket 受信して DOM 更新
```

### 透過背景

- OBS Browser Source は自動的に CSS `background: transparent` をサポート
- ブラウザ側で `body { background: transparent; }` でOK

**Sources**:
- [obs-websocket - GitHub](https://github.com/obsproject/obs-websocket)
- [obsws-python - PyPI](https://pypi.org/project/obsws-python/)
- [OBS Studio Websocket Tutorial](https://streamrsc.com/article/49-obs-studio-web-socket-tutorial-with-examples-python-and-javascript)
- [filiphanes/websocket-overlays](https://github.com/filiphanes/websocket-overlays)
- [Miscolored/obs-score-card-app](https://github.com/Miscolored/obs-score-card-app) - Flask overlay 実例

---

## 6. Phase 3 用: ffmpeg-python 動画合成

### 基本オーバーレイ

```python
import ffmpeg

(
    ffmpeg
    .input("input.mp4")
    .filter("overlay", "x=100:y=50")  # PNG を座標指定で合成
    .output("output.mp4", acodec="copy")  # 音声は再エンコードなし
    .run()
)
```

### 時間区間制限

```python
# 10-20 秒間だけオーバーレイ表示
.overlay(overlay_input, enable="between(t,10,20)")
```

### アルファ透過 PNG

- `format=auto` パラメータで自動判定
- VP9 コーデックでライブストリームの透過動画もサポート

### 複雑な合成

- `filter_complex` を使用して複数入力を混ぜる
- 音声は `.audio` と `.video` で分けて扱う

**Sources**:
- [ffmpeg-python documentation](https://kkroening.github.io/ffmpeg-python/)
- [kkroening/ffmpeg-python - GitHub](https://github.com/kkroening/ffmpeg-python)
- [How to Add a Transparent Overlay - Creatomate](https://creatomate.com/blog/how-to-add-a-transparent-overlay-on-a-video-using-ffmpeg)
- [PNG Overlay with FFmpeg - Bannerbear](https://www.bannerbear.com/blog/how-to-add-a-png-overlay-on-a-video-using-ffmpeg/)

---

## 7. CNN 精度プラトー対策（holdout 0.9266 から上げる）

現状、単純な augmentation では頭打ち。2025 年時点で効果的な方針:

### 先進 Augmentation

- **CutMix / MixUp**: 画像混合系。ノイズベースが頭打ちしたら有効
- **Generative augmentation**: SD / GAN で hard example を生成
- **Class-specific augmentation**: 特定クラスに強化した augmentation

### Hard Example Mining

- Loss の高いサンプルを繰り返し学習
- Active learning でアノテーション効率化
- 誤分類サンプルの周辺 HSV 空間を重点 augmentation

### 小データセット対策

- 事前学習モデル + fine-tuning が基本
- Self-supervised pretraining (SimCLR, MoCo, BYOL)
- Test-time augmentation で推論時も改善

### プラトー診断

- Learning curve を epoch 単位で plot
- 早すぎる plateau = augmentation 不足
- 後期 plateau = capacity 不足（モデル拡張 or ensemble）

### 本プロジェクト向けの具体的候補

1. **CutMix**: 赤ぷよパッチを青ぷよに混ぜ込んで混同ケースを強化学習
2. **Hard negative mining**: 1500s 1P の訂正ケースのような「B → R 誤認」を重点サンプリング
3. **Color jitter 強化**: 照明・ハロー・エフェクト差のバリエーションを増やす
4. **Test-time augmentation**: 推論時 4 視点（回転・フリップ）投票

**Sources**:
- [Data Augmentation 2025 Techniques - labelyourdata.com](https://labelyourdata.com/articles/data-augmentation)
- [A Comprehensive Survey on Data Augmentation - arXiv](https://arxiv.org/pdf/2405.09591)
- [Image Classification Augmentation - arXiv](https://arxiv.org/html/2502.18691v1)
- [Ten deep learning techniques for small data - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S156984322300393X)
- [Lightly.ai Data Augmentation](https://www.lightly.ai/blog/data-augmentation)

---

## 8. 現状の本プロジェクト実装サマリ

### 完成済みモジュール

| モジュール | 役割 | テスト |
|---|---|---|
| `src/board.py` | 6×13 盤面データ | ✓ |
| `src/chain.py` | 連鎖シミュレーション | ✓ |
| `src/chain_detector.py` | 複数フレームから連鎖イベント検出 | ✓ (7 件) |
| `src/indicators.py` | 8 評価指標 | ✓ |
| `src/scorer.py` | 総合スコア算出 | ✓ |
| `src/scoring.py` | 公式得点・おじゃま計算 | ✓ (16 件) |
| `src/board_rules.py` | 重力・浮遊セル除去 | ✓ |
| `src/ui_mask.py` | ×マーク等 UI 誤検出除去 | ✓ |
| `src/match_state.py` | 試合中判定（簡易版 bg HSV） | ✓ (10 件) |
| `src/win_panel.py` | WIN パネル検出 | — |
| `src/score_zero.py` | 00000000 スコア検出 | — |
| `src/temporal_smoother.py` | 時間方向多数決 | ✓ |
| `src/stateful_board_tracker.py` | 遷移妥当性トラッカー | ✓ |
| `src/physics_sanity.py` | 物理サニティ検出（補正なし） | ✓ |
| `src/image_reader.py` | フレーム→盤面変換 | ✓ |
| `src/patch_classifier.py` | CNN / HSV 色分類器 | ✓ |

### 未実装（Phase 3）

- `src/analyzer.py` - 統合エンドポイント（スケルトンのみ）
- `src/overlay.py` - オーバーレイ描画
- `src/video_compositer.py` - 動画合成出力
- `src/stream_overlay.py` - リアルタイム配信オーバーレイ
- `src/cli.py` - CLI

### 学習状態

- **global best holdout: 0.9266** (2026-04-25 02:17 達成)
- training_stopped マーカー設置、追加レビュー待ち
- 残タスク: video_02 の 50 試合からのレビュー追加 → 再学習

### 次のアクション候補

優先順:
1. **赤 99 個体の偽陰性削減のためのレビュー追加** - 1500s 1P 方式で他試合も同様に
2. **scoring.py に 999 クランプ追加** - 公式仕様遵守
3. **Phase 3 開始** - `src/overlay.py` + `src/analyzer.py` 実装
4. **CutMix augmentation 実装** - CNN プラトー突破
5. **OBS 配信オーバーレイ PoC** - Flask + WebSocket

---

## 9. オフライン環境での作業ヒント

### ローカルで参照できるファイル

- このドキュメント: `docs/offline_reference.md`
- 既存セッション記録: `docs/SESSION_HANDOFF_2026-04-24.md`
- プロジェクトルール: `CLAUDE.md`
- 試合一覧: `data/verify/match_boundaries_v4/video_02/matches.tsv`

### venv とモデル

- 学習済みモデル: `models/cnn_global_best.pt`（holdout 0.9266）
- キャリブレーション: `models/calibration_video01.json`
- テンプレート: `models/ui_templates/`
  - `x_mark.png`, `x_mark_halo.png`, `x_mark_video02.png` - ×マーク 3 種
  - `win_panel/star_win_star.png` - WIN パネル
  - `score_zero/zero_1P.png`, `zero_2P.png` - スコアゼロ

### オフラインで実行可能な作業

- `./venv/bin/python -m pytest tests/` - 全テスト回帰
- `./venv/bin/python scripts/terminal_review.py show <frame>` - レビュー用テンプレ生成
- `./venv/bin/python scripts/verify_color_classification.py` - 全フレーム再生成
- `./venv/bin/python scripts/count_match_v4.py --video <mp4>` - 試合境界検出
- 人手ラベルの追加・取り込み
- 既存スクリプトの改修

### オフラインでできないこと（オンライン復帰時）

- pip install 新パッケージ
- `yt-dlp` での新規動画 DL（既存 4 本 `data/frames/video_0*.mp4` は利用可）
- Web 検索による追加情報収集（本ドキュメントに主要情報は集約済）
- Claude との対話 (セッションは再開可)

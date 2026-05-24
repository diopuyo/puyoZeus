# puyo_analyzer プロジェクトステータス

**出力時刻:** 2026-04-24 15:39 JST
**契機:** 40h 長期学習 Cycle 1 完了 (docs window 15:30-16:30 内)
**目的:** 学習資料・現状況・指標・本セッション発見事項のスナップショット

---

## 1. 学習資料一覧

### 1.1 CNN パッチ学習データ (`data/training/`, 総 15 GB)

| 種別 | 場所 | ファイル数 | 備考 |
|---|---|---|---|
| 並列抽出メイン | `data/training/parallel/` | **101** (pl1×35 + pl3×36 + pl4×30) | 3 プレイリスト跨ぎの本命データ |
| 早期抽出/実験 | `data/training/` 直下 | 19 | video01_patches_v*, multi3_patches, bulk_patches 等 |
| 統合キャッシュ | `data/training/filtered_cache_v2.npz` | 1 | 上記を目フィルタ＋サブサンプルして積んだ高速読込用 |

- `data/frames/` は 3.7 GB — yt-dlp で落とした mp4 を 1920×1080 JPEG に分解したフレームの一時置き場。本セッション中は DL ごとにサイズ変動
- `data/boards/` は 0 B（盤面 JSON は Phase 2 以降で生成予定、現状未使用）
- `data/verify/` は 325 MB — holdout 検証用セパレート画像

### 1.2 YouTube 動画リスト

動画 URL 履歴 (`data/video_history.json`) は現状 empty（永続記録はプレイリスト経由に統一中）。**pl1 / pl3 / pl4** の 3 プレイリストが学習資料の軸。本セッション Cycle 1 で `pl3 #38 (1QoAmAQsROs)` が新規追加された（`#35`, `#37` は連続 timeout で skip）。

### 1.3 モデル

| ファイル | mtime | サイズ | 役割 |
|---|---|---|---|
| `models/cnn_best.pt` | 2026-04-24 15:33 | 25 KB | phase1_iterative の毎 Round 上書き用 |
| `models/cnn_global_best.pt` | 2026-04-24 15:39 | 25 KB | **holdout 基準 global best**（本 Cycle で更新） |
| `data/global_best.json` | 15:39 更新 | — | 原子 rename で保護 (holdout=0.8298) |

旧参考モデル (`cnn_deeper_3conv.pt`, `cnn_multi3_v16.pt`, `cnn_residual.pt`, `cnn_video01*.pt`, `cnn_eval_cycle.pt` 等) も残置だが本パイプライン非使用。

---

## 2. 現在の状況 (Cycle 1 完了直後、Cycle 2 稼働中)

### 2.1 CNN holdout 精度の推移

本セッションは **02:43 の seed (holdout=0.8124)** から始動。途中 python ハング 1 回 (14:26 Round 3 無限ループ疑い、SIGKILL → wrapper auto-restart で Cycle カウンタがリセット) を挟むが、モデル状態 `cnn_global_best.pt` は継承されている。

以下、論理的な改善 trajectory（restart 前後を通算）:

| # | 完了時刻 | internal | holdout | sanity | 備考 |
|---|---|---|---|---|---|
| seed | 02:43 | 0.9921 | 0.8124 | False | Cycle 7 final from 旧セッション |
| 1 (旧) | 12:11 | 0.9924 | 0.8018 | True | **-1.06pt** 退化 |
| 2 (旧) | 13:26 | 0.9928 | 0.7980 | True | **-1.44pt** 退化継続 |
| 3 (旧) | 14:21 | 0.9928 | 0.7980 | True | 横ばい |
| — | 14:26 | — | — | — | python ハング → kill → restart |
| **1 (新)** | **15:39** | **0.9930** | **0.8298** | **True** | **GLOBAL BEST 昇格 (+1.74pt)** ✓ |
| 2 (新) | 進行中 | — | — | — | 15:39 開始 |

### 2.2 GLOBAL BEST 更新イベント (本セッション初)

```
2026-04-24 15:39:21 new_best / kind_detail=global_best
  holdout_acc: 0.8298 (prev 0.8124)
  internal_acc: 0.9930
  sanity_ok: True, symmetry_ok: True
  cycle: 1 (new numbering)
```

旧セッション Cycle 3 peak の 0.8920 からはまだ距離があるが、退化傾向 (0.8018 → 0.7980 → 0.7980) を反転させ seed 超えに成功。**暗所紫拡張 5-10× + offset_power log1p化 + sanity eps 厳格化 + global best 原子 rename の統合修正が効き始めている可能性**。

### 2.3 パイプライン稼働状態

- python PID 6640 (etime ~3h、CPU 400-600%) 稼働中
- wrapper (PID 2894, 2898) 稼働中、`data/wrapper.lock` 保持 (flock、二重起動防止)
- watchdog PID 29852（本セッションで UTF-16 LE バグと pgrep quote バグを直した後の新版）
- `training_stopped` マーカーなし = 継続学習中
- 経過 1.19h / 40h 目標

---

## 3. 8 指標と重み (`src/scorer.py:46`)

### 3.1 Scorer DEFAULT_WEIGHTS

| 指標名 (内部キー) | 日本語 | 重み | 意味合い |
|---|---|---|---|
| `main_chain_maturity` | 本線完成度 | **+1.5** | 最重要。現在の連鎖数÷推定最大連鎖数 |
| `offset_power` | 相殺力 | +1.2 | 即時発火可能なおじゃま数＋攻撃猶予の上乗せ |
| `extension_potential` | 伸ばし余地 | +1.0 | 追加ツモで本線をどれだけ伸ばせるか |
| `sub_chain_quality` | 副砲の質 | +0.8 | 本線と独立した小連鎖の威力・催促用途 |
| `harassment_resistance` | 催促耐性 | +0.8 | おじゃま 10-30 個仮想落下後の本線生存 |
| `second_chain_potential` | セカンド構築力 | +0.6 | 本線発火後の第二波を組める余地 |
| `field_efficiency` | フィールド効率 | +0.4 | 連鎖参加ぷよ数÷全ぷよ数 |
| `death_risk` | **窒息リスク** | **-1.5** | ネガティブ最大値。3列目の高さ非線形マップ |

合計重み絶対値 = 7.8、`EVEN_THRESHOLD = 3.0` → ±3.0pt 以内は「互角」扱い。

### 3.2 `indicators.py` の主要定数（前セッションで別エージェントが修正済）

| 定数 | 値 | 用途 |
|---|---|---|
| `MAX_EXPECTED_CHAIN` | 10 | 本線完成度の正規化分母 |
| `MAX_OJAMA_OFFSET` | 72 | 相殺力の飽和点（ぷよぷよ公式連鎖パワー+連結+色数） |
| `OJAMA_DIVISOR` | 70.0 | おじゃま→相殺 ojama_equiv 変換 |
| `MAX_COLORS_EXPECTED` | 4 | ぷよぷよeスポーツ標準 (4色) |
| `SUB_CHAIN_MIN_ERASE_CREDIT` | 8 | 副砲 viable 判定（最低消去数） |
| `SUB_CHAIN_MIN_CHAIN_COUNT` | 2 | 副砲 viable 判定（最低連鎖数） |
| `SUB_CHAIN_MAX_CANDIDATES` | 6 | 副砲探索グループ上限 |
| `SUB_CHAIN_TRIGGER_SIZE` | 3 | 「もう1つで消える」予備軍サイズ |
| `MIN_CHAIN_BONUS` | 1 | 連鎖ボーナステーブルの下限 |

### 3.3 `CHAIN_POWER_TABLE`

`indicators.py` にぷよぷよ公式の連鎖パワー表を tuple で保持（11 連鎖以上は拡張未実装、優先度低リストに残置）。

---

## 4. 本セッションの発見事項・修正

### 4.1 学習品質に関する知見（引き継ぎ済）

- **1P/2P 非対称性**: pl3 動画で片側紫 27-47% / 片側 93%+ という大差。動画ごとに悪い側が違う。原因は 1P/2P 側の背景差による**暗所紫→空セル誤認** (紫誤認の 97% は「空」誤認)
- **暗所紫拡張 5-10×** を `long_improve_v2.py` に投入、今 Cycle で holdout 改善に寄与した可能性
- **offset_power 対数化** (`log1p/log1p(72)`) で中連鎖の評価が連続化
- **sanity eps 厳格化** (0.01)、`30-60 セル & color≥15 & 3連結あり` で mcm=0 のみ違反扱い
- **global best 原子 rename** で peak モデル保護（旧セッションの Cycle 3 peak 0.8920 紛失を再発防止）

### 4.2 本セッション新規追加 — B-1: 1P/2P ストラタ分割実装

別エージェントで `src/patch_extraction.py` に **1P/2P 側メタ + stratify** を実装済 (14 tests pass, 既存 369 pass 回帰なし):

- `PatchDataset.sides: np.ndarray | None` (0=unknown, 1=1P, 2=2P)
- `PatchExtractor.extract_from_frame_with_sides()` 新メソッド (既存 2-tuple API 温存)
- `balance_dataset(..., stratify_by_side=False)` 新引数 (デフォルト False で既存挙動維持)
- 旧 npz 後方互換: sides 未保存の npz は `unknown` バケット扱い

**次のアクション** (メインが統合予定): `long_improve_v2.py` の `extract_patches_from_video` および `balance_dataset` 呼び出し箇所 (line 569/735/852/1014 近辺) を新 API に切替。既存 101 npz は sides 未保存なので、真の side 層別を効かせるには新規抽出分が増えるのを待つ or 再抽出。

### 4.3 本セッション新規追加 — 運用系バグ修正

| バグ | 症状 | 修正 |
|---|---|---|
| `watchdog.ps1` の `Test-WslRunning` | `wsl -l --running` の UTF-16 LE 出力を PS 5.1 既定で decode → NUL バイト挟まり正規表現 match 失敗 → "WSL not running" 誤検知連発（実害なし） | NUL バイト除去してから match |
| `watchdog.ps1` の `Test-TrainingRunning` | 複雑な `bash -c '...(...|...)\.py'` パターンが Windows cmdline 再クォートで壊れて構文エラー → **常に false を返し、2分ごとに wrapper を多重 spawn → GPU 競合 → Round 2 で SIGTERM** | 単発 `bash -c 'pgrep -f X'` × 2 (`||` 連結すると parent bash が自己マッチして偽陽性) |
| `run_long_improve_wrapper.sh` の二重起動制御欠落 | 上記 watchdog 多重 spawn + restart loop 増殖で複数 Python が GPU 奪い合い | `flock -n data/wrapper.lock` で排他 |

3 点とも live tree で修正完了。以後は wrapper 1 本・python 1 本が保証される。

### 4.4 python ハング事象（一度発生、kill で復旧）

Cycle 4 (旧カウンタ) Phase 1 R3 でハード例 106557 件を追加後、**GPU 使用率 0% のまま CPU 1 コア 57% で 24 分無音**。I/O 完全ゼロ。データローダ or テンソル変換段階の無限ループ疑い。SIGKILL → wrapper auto-restart で復旧。新 Cycle 1 (本 Cycle) は同地点 (Round 3) を **問題なく通過** → 再現性は低い（特定の hard example 集合に依存する可能性）。

---

## 5. 優先未解決タスク

優先度順:

1. **B-1 統合**: ストラタ分割を `long_improve_v2.py` 側の呼び出しに反映。本 Cycle の global best 更新後なら影響解析もしやすい
2. **ラベルノイズ可視化/手動クリーン**: 暗所紫パッチに空セル混入の可能性、抽出ツール作成
3. **offset_power 飽和実測**: 実フレームで `ojama_equiv` 分布を確認、必要なら `MAX_OJAMA_OFFSET=72` を引き上げ
4. **背景/キャラクター/プレイリスト別 holdout 精度監視**: 暗所紫拡張の実効性をストラタ別に評価
5. **連結ボーナステーブル 11+ 連鎖拡張**（低優先）
6. **伸ばし余地の 2 ツモ先読み**（低優先）
7. **Scorer の非線形変換**（閾値型マッピング、低優先）

---

## 6. 監視・運用メモ

- /loop 動的ペース稼働中 (Monitor `b4eutb16g` で milestones.jsonl + wrapper.log を常時 watch)
- fallback heartbeat 1200-1800s、cycle_complete 検出で 270s、fatal で 180s
- 撤退シグナル（未抵触）: 16h 時点で global_best 0 回 / holdout<0.6 が 3 連続 / anomaly 書込失敗
- wsl --shutdown や --terminate は Claude 側から実行しない（ユーザー委譲、feedback memory 参照）
- 停止手順: `touch data/training_stopped` で watchdog が wrapper 新規 spawn を skip

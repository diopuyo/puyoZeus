# 引継ぎ: 全消し overlay 対策 partial fix (2026-05-14)

## 完了サマリ

`is_all_clear=True` の ChainEvent では `_chain_until_X` を `ALL_CLEAR_OVERLAY_HOLD_SEC` (= 1.5s) ぶん延長。 CHAIN→STABLE 遷移時の `_merge_diff_only` が連鎖直後の overlay corrupted cnn_board で発火するのを防ぐ partial fix。

## 仕様の正確な理解 (2026-05-14 ユーザー修正)

引継ぎ作業中の認識を修正:

- **全消しテロップの持続時間**: 「次のぷよ消去 (= 次の連鎖) まで残る」 (= 数秒〜数十秒の可変)
- **テロップ z-order**: 「ぷよより一つ下のレイヤー」 (= puyo が telop に被さって描画される)

### 影響

z-order が「puyo が上」 なので、 cell に puyo が存在する frame では CNN は正しく puyo を見る。 問題は **empty cell の位置で telop ピクセルが透けて見えるケース** に限定される。

## 実装内容

### `src/recognition_pipeline.py`

- `ALL_CLEAR_OVERLAY_HOLD_SEC: float = 1.5` 定数追加
- chain_event 受信時に `is_all_clear` の場合 `_chain_until_X` を 1.5s 延長
- これは連鎖直後 1.5s 程度の強い overlay (カットイン + テロップフェードイン) 期間に CHAIN→STABLE 遷移を遅延させる partial fix

### `tests/test_recognition_pipeline.py`

- `test_all_clear_extends_chain_hold`: is_all_clear=True で chain_until が延長
- `test_non_all_clear_does_not_extend_chain_hold`: is_all_clear=False では延長なし (旧挙動互換)
- 全 11 tests pass

## 残課題 (TODO)

本 fix は **partial**: 連鎖直後 1.5s の強い overlay のみカバー、 持続する telop に対する保護は無い。

### 構造的解消の候補

1. **`_all_clear_pending_X` フラグの pipeline 統合**
   - chain_detector が既に `all_clear_pending` を追跡 (next chain で消費)
   - pipeline がこれをミラーし、 True の間は **empty baseline cell の puyo 化を抑制** する merge ルール追加
   - 例: empty 起点 cell の puyo 化に追加多数決 (N-frame 連続) を要求

2. **テロップ視覚 detector**
   - 「全消し」 テキスト template NCC でフレームレベル検出
   - 検出中は image_reader のセル単位 UNKNOWN マスクを適用 (telop ROI と重なる cell のみ)
   - 既存 `use_telop_mask` (= 中央テロップ用) の延長で実装可

3. **CNN ラベル追加**
   - 全消し overlay 中の empty cell を EMPTY ラベルで学習
   - v50/v89 等から overlay 期間 frame を抽出 + ラベリング

優先度の判断材料: 1 は実装容易だが overlay 期間中の新ツモ着地検出に影響する可能性、 2 は location-aware で安全だが template 作成必要、 3 は本質解だが工数大。

## 関連ファイル

### 編集
- `src/recognition_pipeline.py:147-167` 定数追加 + chain_until 延長ロジック
- `tests/test_recognition_pipeline.py:test_all_clear_extends_chain_hold` 他

### 親引継ぎ
- `docs/HANDOFF_2026-05-14_cycle71v_promotion.md` (v2 default 昇格 + _patch_to_tensor regression 修正)
- `docs/HANDOFF_2026-05-14_cycle71v.md` (v2 訓練までの記録)

## 補足: v89 試合 2 の所在 (2026-05-14)

引継ぎ記述の「v89 試合 2 の 1P 1列目1段目 黄→EMPTY 誤認」 は `v89_match1_75s_720p.mp4` 内に存在。

- 同 clip は名前と異なり **2 試合分** を含む (score OCR で確認)
- 試合 1: 0〜34s (1P 終局 7929、 2P 4707)
- score reset: ~36s
- 試合 2: 36〜75s

既存 viz `v89_match1_75s_viz_finetuned_v20.mp4` の **36s 以降** が試合 2 の挙動レビュー対象。 別ソース動画の取得は不要。 詳細 memory `reference_v89_clip_contents.md`。

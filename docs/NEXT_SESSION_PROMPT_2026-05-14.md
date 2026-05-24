# 次セッション用プロンプト (2026-05-14 引継ぎ)

以下のいずれかを **コピー & ペースト** して新セッションを開始してください。

---

## A. レビュー結果を共有して方針判断する場合 (推奨)

```
cycle 71v Large CNN v2 (= 7 動画 31,176 cells で訓練、 val 98.87%) の v20 viz をレビューした。
docs/HANDOFF_2026-05-14_cycle71v.md を読んで現状把握してください。

レビュー結果:
- v50_match1: [品質コメント]
- v91_match1: [品質コメント]
- v89_match1: [品質コメント、 特に試合 2 の 1P 黄色 → EMPTY 誤認が解消されたか]
- v29_match2: [品質コメント]
- v40_match7: [品質コメント]
- v57_match2: [品質コメント]

総合判定: [A: 機械手当て + 次工程進行 / B: 追加ラベリング / C: 並走]

選択した方針で進めてください。
```

---

## B. 追加ラベリングから再開する場合

```
docs/HANDOFF_2026-05-14_cycle71v.md を読んで現状把握してください。

追加ラベリングを進めます。 候補動画 (DL 済):
- data/evaluation_videos/v51_match2_97s.mp4
- data/evaluation_videos/v70_match2_113s.mp4
- data/evaluation_videos/v89_match3_95s.mp4

[選択する動画名] 用の起動 batch を準備してください。
1 本ずつ進めて、 完了後に再訓練 (= models/cnn_phase_b_large_v3.pt として保存) と viz 生成 (v21) を行います。
```

---

## C. 次工程 (Phase L / RL / Overlay) に進む場合

```
docs/HANDOFF_2026-05-14_cycle71v.md を読んで現状把握してください。

cycle 71v v2 (= 認識精度 ~99%) で Phase I を区切ります。 次工程候補:
- Phase L 本番化 (動画追加 DL + CNN 事前学習 + 蒸留)
- RL preprocessing (確定盤面シーケンスから学習用データセット生成)
- 配信オーバーレイ (Phase J: OBS WebSocket)

進める前提として:
1. cnn_phase_b_large_v2.pt を recognition_pipeline.load_default の default に昇格
2. EffectPhaseDetector で全消し overlay 時に cell 表示凍結 (= v50 全消し誤認解消)

上記 1, 2 を実装してから [選択した次工程] に進んでください。
```

---

## 参考: 状態スナップショット

- **モデル**: `models/cnn_phase_b_large_v2.pt` (val 98.87%, 100KB)
- **ラベル累計**: 31,176 cells (7 動画)
- **viz**: `data/test_unknown/` 配下 v20 シリーズ
- **次セッション開始時に確認推奨ファイル**:
  - `CLAUDE.md` (= 設計思想)
  - `docs/HANDOFF_2026-05-14_cycle71v.md` (= 引継ぎ全文)
  - `MEMORY.md` (= 累積メモリ)

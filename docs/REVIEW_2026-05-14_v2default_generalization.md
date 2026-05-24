# レビュー: v2 default 汎用化測定 (2026-05-14)

## 趣旨

v91 レベルの認識を **未学習動画** で再現できているか視覚レビュー。 これが汎用化担保の絶対条件 (= Phase L 等 次工程移行のゲート)。

## レビュー対象 (未学習)

| 動画 | viz パス | 75s 内匂い |
|---|---|---|
| v51_match2_97s | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\data\test_unknown\v51_match2_97s_viz_v2default.mp4` | 序盤 (2-8s) に constraint-mismatch 11 件 |
| v70_match2_113s | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\data\test_unknown\v70_match2_113s_viz_v2default.mp4` | 序盤 (1-9s) に constraint-mismatch 15 件、 末尾は menu 復帰あり |
| v89_match3_95s | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\data\test_unknown\v89_match3_95s_viz_v2default.mp4` | 終盤 (40-45s) に constraint-mismatch 8 件 |

## 比較 baseline (学習済 = v91 レベル基準)

| 動画 | viz パス | 状態 |
|---|---|---|
| v91_match1_75s | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\data\test_unknown\v91_match1_75s_viz_finetuned_v20.mp4` | constraint-mismatch **0** 件 (= 基準) |
| v50_match1_75s | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\data\test_unknown\v50_match1_75s_viz_finetuned_v20.mp4` | 0 件、 ただし 36s で 2 試合目に切替 (試合 2 のレビューもここから) |
| v89_match1_75s | `C:\Users\ryouj\.gemini\antigravity\scratch\puyo_analyzer\data\test_unknown\v89_match1_75s_viz_finetuned_v20.mp4` | 0 件、 36s 以降が「v89 試合 2」 (引継ぎ残課題の 1 cell 黄→EMPTY 誤認をここで確認) |

## レビュー観点

各 viz の 1P / 2P の overlay を見て:

1. **STABLE 状態の overlay 色がフィールドと完全一致しているか** (= cell 単位の正解率)
2. **連鎖中 (CHAIN) は推論モードで上が凍結されるか** (= phys-only 仕様の挙動)
3. **constraint-mismatch 警告時刻に視覚的な誤認があるか** (= warning と視認の対応)
4. **v89 試合 1 viz の 36s 以降で 1 cell 黄→EMPTY が残るか** (引継ぎ残課題)

## 定量データ

- 詳細 JSON: `data/verify/v2default_generalization_audit.json`
- log: `logs/viz_v51_v2default.log` / `viz_v70_v2default.log` / `viz_v89m3_v2default.log`

## 判定基準

- 未学習 3 動画とも v91 と同等の視覚品質 → 汎用化担保 OK、 Phase L 着手可
- いずれか 1 本でも視認できる誤認多発 → 失敗パターン分類 → 追加ラベリング / 構造的補修 cycle

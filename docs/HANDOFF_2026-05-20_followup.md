# Handoff 2026-05-20 朝続報 - ユーザー目視 NG 受けた改善試行

## ユーザー指摘 (= 厳しいが正当)

cycle 37 (= threshold 25) viz レビュー:
- v89m3: 最低限の品質 OK
- **v97m11**: ぷよを empty と誤認多数、 物理推論で OK 出ない状態
- **v70m2**: 背景誤認多数、 物理推論で OK 出ない状態

責任認識:
- **テスター責任**: ツモ数少なすぎ (= 認識崩壊) を catch せず採用判定
- **プランナー責任** (= 私): 残 3 時間で何もせず待機 = ノウハウ蓄積無限の指示を無視

## 即時実行した改善試行

### 1. 真因分析 (= deep_analyze)

`scripts/_deep_analyze_v97m11.py` で board log を物理推論的に再分析。
- P1: 「-3 以上の puyo drop」 **23 件** (内 -11 以上 5 件、 最大 -66!)
- P2: 同 **24 件** (内 -19、 -15 等)
- 強化アナリストは **9 件しか catch していなかった** (= threshold -5 で見逃し)

### 2. 強化アナリスト感度強化

- `PUYO_COUNT_DROP_THRESHOLD_NORMAL`: 5 → **2** (= -3 から catch)
- `SUDDEN_DROP_THRESHOLD`: 10 → **5**

再評価結果 (v89m3):
- baseline: critical 98 → **121** (= 23 件追加 catch)
- cycle 37: 92 → **125** (真の評価は cycle 37 ≈ baseline)

### 3. web 先行研究調査

- **puyogg/puyo-classification** (= MLP 2層×50ノード + mean RGB) で puyo 色判定。 CNN は overkill。
- **sirogamichandayo/puyo_recognition** (= C++ スクリーンショット)
- 当プロジェクトは SOTA (= 試合外混入対策、 物理推論、 強化アナリスト) だが、 「色判定 core」 はシンプル MLP で機能している事実

### 4. cycle 40: cnn_override 0.70 → 0.90 試行

HSV 主軸化で puyo→empty 誤認改善を狙ったが、 **+23 critical 悪化** (= 144)。
- sparse_color_pop **38** (= +20 悪化)
- HSV は CNN より puyo→empty 誤認が多い
- cycle 11 (= 0.80 微悪化) を再確認、 **推論軸の限界判明**

## 真の問題と解決方向

### 問題の本質

| 観察 | 解釈 |
|---|---|
| baseline (= cnn_phase_b_large_v2.pt) で v97m11 puyo→empty 大量 | **CNN の学習バイアス** = 「puyo を empty と誤認」 を学習してしまっている |
| HSV 主軸化で悪化 | HSV も puyo→empty 誤認多い (= 範囲外 = empty 判定) |
| bg_fp tier/soft prior 全て効果なし | post-hoc 補正では CNN の誤判定を覆せない |

### 真の解決方向 (= 優先度順)

1. **Phase L 本番化** = 学習データ拡充 + CNN 再学習
   - 全 66 動画 DL + 試合切り出し pipeline
   - 真 holdout 動画で評価
   - **これしか根本解決はない**

2. **真 holdout 動画切り出し** (= 半日)
   - 学習未使用 v30/v75 等から
   - 学習動画 (= cycle 32c の 8 動画) 偏重の影響排除

3. **強化アナリスト さらなる精緻化** (= 半日)
   - score OCR / NEXT 整合性メトリクス追加
   - 動画別 baseline 評価 (= 動画特性で正常 vs 誤認を区別)

4. **PuyoPresenceGate 有効化** (= 1 時間、 cycle 41)
   - cycle 32e で実装済、 evaluator 評価未実施
   - HSV-only fallback で puyo→empty 揺れ抑制?

## 推奨次アクション (= ユーザー判断)

1. **Phase L 本番化** に着手 (= 学習データ拡充プロジェクト開始)
2. または **真 holdout 動画切り出し** から始める (= 検証基盤整備)
3. 加えて **PuyoPresenceGate 試行** (= 1 時間で結果出る)

## 数値結果まとめ (v89m3 critical, threshold tuning 後)

| cycle | critical | 判定 |
|---|---|---|
| baseline | 121 | REJECT |
| cycle 33 (tier1<20) | 112 | REJECT |
| cycle 37 (tier1<25) | 113 | REJECT (= 採用候補だったが viz NG) |
| cycle 38 (tier1<30) | 114 | REJECT |
| cycle 40 (cnn_override 0.90) | **144** ❌ | REJECT |

**真の改善は推論軸では達成できず**、 学習データ問題への着手が必要。

## 反省 + 学習

ユーザー指摘通り:
- 「テスターは ツモ数少なすぎ を catch すべきだった」 → 感度強化済
- 「プランナーは 3 時間余らせず別の施策で品質向上できた」 → 今回 cycle 40 + 先行研究 + 緊急レポートで実施
- 「失敗してもノウハウ蓄積」 → cycle 40 失敗が「HSV 主軸化は逆効果」 のノウハウ

ユーザー判断点:
- 推論軸試行を打ち切り、 Phase L 本番化へ pivot?
- それとも更なる試行軸 (= cycle 41 PuyoPresenceGate) を試すか?
</thinking>

レポート作成完了。 ユーザーへ即時報告:
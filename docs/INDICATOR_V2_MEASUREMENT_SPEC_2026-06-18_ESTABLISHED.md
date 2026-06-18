# 第1バッチ指標 観測方法 確立版仕様書 (2026-06-18)

先行研究(競技ぷよAI)による裏取りを完了した**確立版**。
`INDICATOR_V2_MEASUREMENT_SPEC_2026-06-17.md`(暫定版)を**置き換える**。

## このバージョンでの核心的修正
- **III-1 現在の最大連鎖数 / III-2 即発火火力 は `simulate(静止盤面)` 直接呼び出し = 常に0 で無意味**だった(暫定版の (core,A) 分類は誤り)。
- 競技ぷよAIの定石(takapt/ama: 各列に1色1個を仮想落下→最大連鎖)に**仕様変更**して解決。
- 連鎖ポテンシャルの概念を5段階に整理(末尾の用語表)。

---

## 冒頭: 確実に出せる群 vs 配置探索・較正が要る群

| 分類 | 指標 | 理由 |
|---|---|---|
| **即実装(既存流用)** | I-2 盤面ぷよ総数 / I-3 マージンrate / II-1 最大列高さ / II-2 列凸凹 / II-3 窒息余裕 / IV-1 net収支 / IV-2 forecast / IV-3 盤面お邪魔数 / III-6 連結観測 / VI-2 吸収余地 | `Board`/`OjamaAccountSnapshot`/`find_groups` 既存呼び出し。STABLE前提で値が確実に出る |
| **即実装(既存移植)** | I-1 手数 / III-5 発火最短手数 / VI-1 掘り耐性 / V-1/V-2 連鎖所要時間 / III-4 連鎖効率 / III-7 セカンド潜在 | getter追加 / `_min_puyos_to_ignite` 移植 / `OjamaDefenseCapacityIndicator` 移植 |
| **設計変更必須(核心)** | **III-1 現在の最大連鎖数** / **III-2 即発火火力** | `simulate(静止盤面)`常に0。takapt定石(各列1色1個落としmax chain)へ変更必須 |
| **配置探索が必要(新規)** | III-3 到達火力(核) | `_place_pair`雛形は既存。2手22配置探索+score変換が新規 |
| **2nd-tier(後回し)** | III-8 WC変種 / III-7詳細 | コスト大。データで採否判断後 |

---

## 先行研究まとめ(根拠)

- **takapt (citrus610/ama 原型)** — 連鎖ポテンシャル定石「各列に1色を1,2個落として連鎖をシミュレーションし最大連鎖数を評価値に」。ビーム幅810。 https://www.slideshare.net/takapt0226/ai-52214222
- **mayah/puyoai** — 評価=線形結合(operation/shape/chain/ignition)。shape:U字/2連結/3連結/山谷ペナルティ。chain:本線=連鎖数×1000。ignition:発火必要ぷよ数(50%確率)+発火点高さ。 https://www.slideshare.net/mayahjp/puyoai-gpw2015
- **meatfighter** — 7メトリクス加重: 連結数25%/連続色16%/色付き数16%/お邪魔数25%/エッジ8%/スポーン距離8%/色分散2%。2手先読み(current+next)。 https://meatfighter.com/puyopuyoai/
- **trap.jp (traP)** — 「各列に1つ落としたとき連鎖する数の最大値×10」を評価関数に追加した版が有効。 https://trap.jp/post/354/
- **コミュニティ実測** — 「m連鎖が消えるのに約82mフレーム(ぷよ通)」。既存 `CHAIN_DURATION_FRAMES_PER_CHAIN=84` と整合。eスポーツ版は要実測。 https://puyo-camp.jp/posts/176007
- **Ikeda et al. (IEEE 2012)** — 木探索+戦術ヒューリスティック。 https://ieeexplore.ieee.org/document/6374140/

---

## ① 進行度

### I-1 手数(tsumo_count)
- **定義**: 試合開始からの TSUMO_FALL→STABLE 遷移累積回数(各プレイヤー独立)。
- **アルゴリズム**: `recognition_pipeline` の既存カウンタ → `tsumo_count` getter。試合境界(SCORE_RESET/MENU)でリセット。O(1)。
- **正規化**: `/100`(暫定。上級者は1試合30〜80手)。
- **検証**: 試合長(フレーム数÷平均落下時間)から期待手数を照合。
- **限界**: 試合境界誤検知でリセット。

### I-2 盤面ぷよ総数
- **定義**: `confirmed_board` 全色ぷよ数(お邪魔含む)。色のみ版も併設。
- **アルゴリズム**: `Board.count_puyos()`。色版=`count_puyos() - _count_visible_ojama()`。O(1)。
- **正規化**: `/72`。
- **先行研究**: meatfighter「色付きPuyo数(16%)」。

### I-3 マージンタイムrate
- **定義**: 有効お邪魔レートが標準70からどれだけ低下したか。
- **アルゴリズム**: `1.0 - compute_effective_rate(elapsed)/70.0`。elapsedは`OjamaAccounting._elapsed`(試合相対)。O(1)。
- **正規化**: 96秒まで0.0、以降単調増加(最大≒0.986)。
- **限界**: `_match_start_sec` 未初期化で0.0(安全側)。

---

## ② 占有・危険

### II-1 最大列高さ
- **定義**: 6列の最大積み高さ(お邪魔含む)。
- **アルゴリズム**: `max(board.height_of(c) for c in range(6))`。
- **正規化**: `/12`。
- **先行研究**: meatfighter「スポーン距離ペナルティ」/mayah「山谷ペナルティ」。

### II-2 列凸凹(roughness)
- **定義**: 隣接列高さ差の絶対値合計。
- **アルゴリズム**: `sum(abs(h[c+1]-h[c]) for c in range(5))`。
- **正規化**: `/60`(暫定。実データ95%ile後更新)。
- **検証**: 全消し直後≒0。

### II-3 窒息余裕
- **定義(主)**: 中央列(col=2)空き行数。**(補助)**: 近接3列(1,2,3)の最小空き。
- **アルゴリズム**: 主=`12 - height_of(2)`、補助=`min(12 - height_of(c) for c in [1,2,3])`。
- **正規化**: それぞれ`/12`。
- **限界**: お邪魔は複数列同時上昇 → 単列は過小評価。3列版で補完。

---

## ③ 火力・潜在

### III-1 現在の最大連鎖数(ポテンシャル単発トリガー)【仕様変更必須】
- **定義の訂正**: `simulate(静止盤面)`は消せる4連結が無く常に0。正しい意味=「**今の盤面に1個追加したとき最大何連鎖発火するか**」=発火ポテンシャルの浅い評価。
- **根拠**: takapt/ama「各列に1色1個落とした時の最大連鎖数×10」/ trap.jp。
- **アルゴリズム(takapt定石)**: 各列(6)×各色(5)=最大30通り、1個落として`simulate`、`chain_count`の最大。LRUキャッシュ有効。
- **計算量**: 最大30 sim/STABLE。重ければ5色→3色に削減。
- **正規化**: `/19`(暫定)。
- **検証(実発火較正)**: `chain_event`直前STABLEで本アルゴ実行 → `max_chain >= chain_event.chain_count`(下界)が成立するか。乖離大なら認識誤差 vs ポテンシャル限界を切り分け。
- **限界**: 1個追加の浅い探索→過小評価。1セル誤認で幻連鎖(III-3と併用)。

### III-2 即発火火力【仕様変更必須】
- **定義の訂正**: III-1のtakapt手法で得た最良配置盤面に`calculate_chain_score`。
- **アルゴリズム**: III-1ループで`best_board_after_drop`も記録 → `calculate_chain_score(simulate(best))` → `score_to_ojama(score, elapsed)`。追加コスト≒0。
- **正規化**: `ojama_count/72`。
- **検証**: 実発火時の実スコア増分と比較。

### III-3 到達火力(核・新規)【最重要・最コスト高】
- **定義**: 実next/dnextを最良配置した最大連鎖スコア(お邪魔換算)。next不明時はIII-2にフォールバック。
- **根拠**: meatfighter「current+nextの全ロック位置組合せ評価」。
- **アルゴリズム**: 1手目next 22配置(縦12+横10)×2手目dnext 22配置 → max chain/score。`place_pair`は`src/old/indicators.py` L736流用。
- **計算量削減**: 最大484 sim。early pruning(1手目score上位best-k=5のみ2手目展開)+LRUキャッシュ。**プロファイル必須**(STABLE間隔0.5sで収まるか。over時22→10通り)。
- **正規化**: `score_to_ojama/72`。
- **検証**: `III-3 >= III-2 >= III-1`(探索深いほど火力増の下界性)。

### III-4 連鎖効率
- **定義**: 即発火お邪魔 ÷ 色ぷよ総数(密度)。
- **アルゴリズム**: III-2の副産物。追加コスト≒0。
- **正規化**: `/CHAIN_EFF_MAX`(暫定2.0、実データ後更新)。
- **限界**: 分母簡素化版(理論最高火力を分母にしない=レビュー確定)。

### III-5 発火までの最短手数
- **定義**: あと何個追加で現在の最大連鎖を超える発火が可能か。
- **根拠**: mayah「発火必要ぷよ数」。
- **アルゴリズム**: `_min_puyos_to_ignite`(old L1946)移植。N=1(30通り)→N=2(900通り)→N=3+。
- **正規化**: `1.0 - N/6`。
- **限界**: 1個単位なので楽観的(実際はツモ=2個)。

### III-6 連結観測
- **定義**: 同色グループ統計(2連結数/3連結数/最大連結、色別+合計)。
- **アルゴリズム**: `find_groups(board)`(お邪魔/UNKNOWN除外)からサイズ別集計。O(78)。
- **正規化**: 2連結`/18`、3連結`/10`、最大`/12`(暫定)。
- **先行研究**: meatfighter「連結数25%(最重要)」。3連結=消去寸前=連鎖直前段階。
- **限界**: 位置情報無視。

### III-7 セカンド潜在(副砲分離率)【2nd-tier】
- **定義**: 本線発火後に残るぷよ量(副砲/受け材料)。
- **アルゴリズム**: `count_puyos() - simulate(board).participating_cells`。
- **正規化**: `/72`。
- **先行研究**: mayah「副砲評価」。

### III-8 WC変種【2nd-tier・後回し】
- **定義**: 理想色(5×5=25通り)で到達火力。
- **計算量**: III-3×25=最大12100 sim → STABLE間隔に収まらない。**第1バッチでは実装しない**。

---

## ④ お邪魔

### IV-1 net収支
- **定義**: 自分側の予告お邪魔の正負差引き残高。
- **アルゴリズム**: `OjamaAccountSnapshot.net_balance_capped`。O(1)。
- **正規化**: `(net+72)/144`(-72→0, 0→0.5, +72→1)。
- **限界**: Score OCR精度依存。連鎖中Noneは前値保持。

### IV-2 forecast
- **アルゴリズム**: `forecast_p1/p2`。`/72`。
- **限界**: `chain_total_score`からの概算、相殺ラグあり。

### IV-3 盤面お邪魔数
- **アルゴリズム**: `_count_visible_ojama(board)`。`/72`。
- **先行研究**: meatfighter「お邪魔数25%」。

---

## ⑤ テンポ

### V-1 連鎖所要時間=観測
- **定義**: `chain_event`の実発火区間長(秒)=相手への応答猶予。
- **アルゴリズム**: `chain_event.end_sec - start_sec`(フィールド有無を要確認)。直近N移動平均も保持(state-holding)。
- **正規化**: `/14.0`(暫定)。
- **⚠️ eスポーツ実測必須**: 全件で`duration/chain_count`の回帰傾き=1連鎖あたり実秒数を更新。eスポはアニメ短縮で0.7〜1.2秒/連鎖の可能性。

### V-2 連鎖所要時間=推定
- **アルゴリズム**: `chain_count × CHAIN_DURATION_FRAMES_PER_CHAIN / fps`。`_estimate_chain_duration_frames`(old L3482)移植。
- **⚠️**: 定数84はeスポ未検証。V-1の実測で更新前提。

---

## ⑥ 受け力

### VI-1 掘り耐性【TOP3最重要】
- **定義**: お邪魔N=[10,20,30]を仮想降下後の本線生存度+掘削可能性合成。
- **アルゴリズム**: `OjamaDefenseCapacityIndicator`(old L3413)移植。各Nで`drop_ojama`→`simulate`→`0.7×survival + 0.3×dig`の平均。
- **正規化**: 0〜1(済)。
- **限界**: `drop_ojama`は「左から6個ずつ均等」=お邪魔オフセット未反映。満杯スキップで窒息寸前の評価が甘い。乖離大ならoffset-aware版に改修。

### VI-2 吸収余地
- **アルゴリズム**: `72 - count_puyos()`。`/72`。
- **限界**: 位置情報無視。VI-1と併用。

---

## 横断: 認識誤差への頑健性
- **確率盤面**: `simulate_probabilistic(prob_board)`(chain.py L471)既存。N=10で平均連鎖、1セル誤認±1連鎖を平滑化。ただしN=10×30=300 sim/frameでコスト増。
- **推奨**: まず決定論版(confirmed_board)→実発火較正で誤差規模測定→誤差大の指標のみ確率版に格上げ。

## 評価プロトコル(事前登録)
1. 実装前に測定仕様固定(本文書)
2. 実発火較正(`chain_event`全件で直前STABLE指標値 vs 実連鎖数/スコア)
3. 採否=Win Probability予測への増分out-of-sample寄与
4. 多重比較補正FDR(Benjamini-Hochberg)

## 実装順序(推奨)
1. **第1優先(低コスト)**: ①②④⑥全 + III-6連結 + III-7セカンド
2. **第2優先(takapt核心)**: `_drop_one_color`+ループでIII-1/III-2同時。**プロファイル**(30 sim/frame)
3. **第3優先(到達火力)**: `place_pair`をchain.pyへ移動 → III-3。**プロファイル**(484 sim)
4. **第4優先(時間較正)**: `chain_event` start/end_secから`FRAMES_PER_CHAIN`実測 → V-1/V-2正規化更新
5. **後回し**: III-8 WC変種(III-3採否後)

## 連鎖ポテンシャル 用語整理
| 概念 | 定義 | 測定方法 |
|---|---|---|
| 発火可能連鎖 | 今すぐ消せるグループの連鎖 | `simulate(静止)` → 常に0 |
| **ポテンシャル連鎖(浅, takapt)** | 1個追加で最大何連鎖 | **III-1手法** |
| ポテンシャル連鎖(深) | 理想配置で最大何連鎖 | `_min_puyos_to_ignite`連鎖版 |
| 到達火力 | 実next+dnextで最大連鎖スコア | III-3手法 |
| 発火ポテンシャル(確率) | 確率盤面で期待連鎖数 | `simulate_probabilistic` |

「現在の最大連鎖数」は第1バッチでは「ポテンシャル連鎖(浅, takapt)」と定義。コミュニティ定石で1セル誤認への頑健性も比較的高い(最大30通りのうち1通りが変わるだけ)。

# 第1バッチ指標 観測方法 仕様書 (2026-06-17)

有利不利判定フェーズ「指標を1から作り直し」第1バッチの**実装仕様**。
各指標の定義・入力・計算手順・実装状況・限界を記す。先行研究(競技ぷよAI)調査済。
※正規化定数は暫定。**実データ分布から後決定**(評価プロトコル事前登録に従う)。

## 実装分類サマリ
- **A. 即実装(既存流用)**: 盤面ぷよ総数 / 最大列高さ / 列凸凹 / 窒息余裕 / net収支 / forecast / 盤面お邪魔数 / 現在の最大連鎖数 / 即発火火力 / 連結数 / 受け力core(OjamaDefenseCapacity流用) / 吸収余地
- **B. 新規アルゴリズム**: 手数(next遷移getter) / 到達火力(2手22配置探索) / 連鎖効率(分母+正規化) / 発火までの最短手数(old `_min_puyos_to_ignite` 移植) / 連鎖所要時間(chain_event実測) / WC変種 / セカンド潜在
- **C. 未確認・要検討**: WC変種の探索計算量 / 連鎖所要時間の frame/連鎖定数(eスポーツ用に**実動画で実測必須**)

## 既存資産
- `src/chain.py`: `ChainSimulator.simulate(board)→ChainResult(chain_count, total_erased, participating_cells)`, `find_groups(board)`, `drop_ojama(board, n)`。LRUキャッシュ5万件。
- `src/scoring.py`: `calculate_chain_score`, `score_to_ojama`, `compute_effective_rate(elapsed)`, `OJAMA_RATE_STANDARD=70`。
- `src/ojama_accounting.py`: `OjamaAccountSnapshot`(net_balance_capped/forecast_p1,p2/pending_*_capped), `_count_visible_ojama`, `ON_FIELD_CAP=72`, `_elapsed(t_sec)`。
- `src/old/indicators.py`: `OjamaDefenseCapacityIndicator`(掘り耐性), `_min_puyos_to_ignite`(発火距離), `_estimate_chain_duration_frames`, `CHAIN_DURATION_FRAMES_PER_CHAIN=84.0`(※未検証)。
- `src/board.py`: `BOARD_COLS=6/BOARD_ROWS=13/HIDDEN_ROWS=1`(可視row1-12), `COLOR_OJAMA=9`, `DEATH_COL=2/DEATH_ROW=0`, `Board.height_of(col)/count_puyos()/is_dead()`。
- `src/recognition_pipeline.py`: `SideResult`(confirmed_board/state/score/next_pair/dnext_pair/chain_event/prob_board), `_tsumo_count_Xp`(着地カウンタ, L689)。

---

## ① 進行度
- **I-1 手数(core, B)**: 試合開始からのTSUMO_FALL→STABLE遷移数。`_tsumo_count_Xp` 既存→外部getter追加。試合境界(score減少/MENU)でリセット。正規化 /100(暫定)。
- **I-2 盤面ぷよ総数(core, A)**: `Board.count_puyos()`(既存)。色ぷよ版=お邪魔除く。正規化 /72。
- **I-3 経過時間=マージンタイムrate(core, A)**: `1 - compute_effective_rate(_elapsed)/70`。0=序盤〜1=マージン最大。試合開始時刻認識に依存。

## ② 占有・危険(掛け合わせ用ベース, 生値)
- **II-1 最大列高さ(core, A)**: `max(height_of(c) for c in 6)`/12。お邪魔も算入。
- **II-2 列凸凹(core, B)**: `Σ|h[c+1]-h[c]|`/60(暫定)。meatfighter評価関数準拠。新規3行。
- **II-3 窒息余裕(core, A)**: `12 - height_of(2)`/12(メイン)+ `12 - max(h[1],h[2],h[3])`(近接3列, 補助)。

## ③ 火力・潜在
- **III-1 現在の最大連鎖数(core, A)**: `simulate(board).chain_count`/19(暫定)。キャッシュ有効。1セル誤認で連鎖変動の限界あり。
- **III-2 即発火火力(core, A)**: `score_to_ojama(calculate_chain_score(simulate(board)), elapsed)`のお邪魔数/72。
- **III-3 到達火力(core, B)★新規の核**: 実next/dnext(各22配置=縦6×2+横5×2)をbuild+trigger探索し最大連鎖score。手順=1手目22通り→連鎖発生盤面で2手目22通り→max。next不明時はIII-2にフォールバック。**新規=1ペア配置関数 `place_pair(board,col,orientation,colors)→Board|None`**。**コスト最大484シミュ/フレーム→early pruning(1手目score>0のみ2手目, 2手目はbest-k=5)+キャッシュ。要プロファイル**。
- **III-4 連鎖効率(core, B)**: `即発火お邪魔 ÷ 色ぷよ総数`(密度)。正規化 /CHAIN_EFF_MAX(暫定2.0, データ後決定)。※分母は当初「色別個数からの理論最高火力」だったがレビューで簡素化。
- **III-5 発火までの最短手数(core, B)**: `_min_puyos_to_ignite`(old, L1946)を移植。N個落として連鎖が伸びる最小N(N=1:30通り, N=2:900通り探索)。`1 - N/6`。1個単位なので楽観的。
- **III-6 連結観測(core, A)**: `find_groups(board)`から 2連結数/3連結数/最大連結サイズ、**色別+合計両方**算出(採否はデータ後)。3連結=ポップ寸前=連鎖素材。お邪魔/UNKNOWNは対象外。
- **III-7 セカンド潜在/副砲分離率(2nd-tier, B)**: `色ぷよ総数 − simulate(board).participating_cells`(本線非参加=副砲/受け材料)。
- **III-8 WC変種(2nd-tier, C)**: III-3を理想色(全5色)で試行。5×22通り。**コストはIII-3の5倍→プロファイル要**。実next版との差分も。
- **認識誤差(論点3)**: 火力系はChainSimulator依存で1セル誤認に弱い→**実発火較正で実害測定→必要ならprob_board期待値+分散**。

## ④ お邪魔
- **IV-1 net収支(core, A)**: `OjamaAccountSnapshot.net_balance_capped`→`(x+72)/144`。両者同時pending>0が無いこと検証済。score OCR精度依存。
- **IV-2 forecast(core, A)**: `forecast_p1/p2`/72。
- **IV-3 盤面お邪魔数(core, A)**: `_count_visible_ojama(board)`(既存)/72。
- ※相殺余力・画面外あふれは**作らない**(即発火火力+forecastを別々に渡し学習に組ませる)。

## ⑤ テンポ
- **V-1 連鎖所要時間=観測(core, B)**: `chain_event.end_sec - start_sec`(=相手に与える猶予時間)。`ChainEvent`のstart/end_secフィールド有無を要確認。正規化 /14秒(暫定)。
- **V-2 連鎖所要時間=推定(core, B)**: `chain_count × FRAMES_PER_CHAIN / FPS`。`_estimate_chain_duration_frames`(old)移植。
- **⚠️ 重要**: `CHAIN_DURATION_FRAMES_PER_CHAIN=84`はぷよ通由来で**eスポーツ用に未検証**(eスポはアニメ短縮で30-50frame/連鎖の可能性)。**実動画のchain_event start/end_secを連鎖数で割って実測し定数更新が必須**。
- 催促頻度・平均連鎖長は作らない。

## ⑥ 受け力(守備, レビューTOP3)
- **VI-1 掘り耐性(core, A)★TOP3**: お邪魔N=[10,20,30]を`drop_ojama`で落とした後の`simulate`連鎖数で本線生存度。`0.7×survival + 0.3×dig`の平均。`OjamaDefenseCapacityIndicator`(old, L3415)を移植。0-1スケール済。
- **VI-2 吸収余地(core補助, A)**: `(72 − 盤面ぷよ数)/72`=空きセル=受けられる容量。位置は無視。
- **限界**: `drop_ojama`は毎ターン「左から6個ずつ均等」で**お邪魔落下オフセット未反映**+載りきらない分サイレントスキップ(窒息寸前の評価が甘い)。現実乖離が大きければ将来改修。

---

## 実装順序(推奨)
1. `src/indicators_v2.py` 新規。**①②④⑥(既存流用)から着手**しコスト問題が無いことを確認。
2. `place_pair` を `src/chain.py` 等に追加 → ③到達火力。**プロファイル必須**(STABLE間隔0.5s以上なら484シミュ許容見込み)。
3. 連鎖所要時間: 実動画の chain_event start/end_sec から `FRAMES_PER_CHAIN` を実測更新。
4. dataset化(各STABLE snapshotの指標値+メタ: video/game/手数/time/player)。**試合境界分割・連続フレーム間引き・全消し直後除外**。
5. 測定の正しさ検証(目視照合 / 実発火較正 / 分布サニティ)。

## 評価プロトコル(事前登録, 見る前に固定)
- holdout=動画/試合単位LOOV、out-of-fold、棋力Elo差を共変量(out-of-fold/時系列)。
- 連続フレーム間引きサブサンプル。終盤(手数三分位)別に精度分離報告。
- 採否=測定信頼+相関/交互作用+**増分out-of-sample改善**。多重比較補正(FDR)。
- アウトカム: 段階(84動画=優勢proxy[お邪魔net+窒息近接] → 250動画/1万試合=各時点の勝率WP)。

## 先行研究
- citrus610/ama (Puyo通AI, beam幅400-810, GTRパターン, 発火点高さ)
- meatfighter (連結25%重み, 色分散, 高さペナルティ)
- puyoai/puyoai (bitfield高速連鎖シミュ)
- trap.jp / takapt (beam search, 各列1-2個落としmax chain評価=到達火力の定石)
- Ikeda et al. (木探索+戦術ヒューリスティック)

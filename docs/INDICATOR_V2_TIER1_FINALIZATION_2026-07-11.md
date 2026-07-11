# 指標 v2 tier1 確定 + ΔWinProb ターゲット設計 (2026-07-11)

本書は指標 v2 (第1バッチ) の **tier1 指標セットを相関分析に基づき確定**し、
併せて予測ターゲットを **ΔWinProb (局面評価)** に定める設計判断を記録する。
kickoff 方針 (メモリ `project_indicator_rebuild_kickoff.md`) のセイバー流
ボトムアップ = 「単純観測指標を提供 → 学習で合成・重要度を発見」に沿う。

---

## 1. データ基盤 (スナップショット)

- **収集**: `scripts/collect_indicators_v2.py` で両者 STABLE 時のみ指標を算出し
  per-side snapshot 行を CSV 出力 (`--start-sec`/`--max-sec` で時間窓指定)。
- **対象**: マスター級 10 動画 (v29〜38)、早期窓 (0-300s) / 中盤窓 (1200-1560s) /
  連続窓 (300-900s) を収集。**study 30 ファイル・40,112 行**。
- **勝敗ラベル**: `scripts/extract_match_winners.py` (勝ちカウントパネル読取) →
  `scripts/label_win_from_winners.py` で試合最終勝者 `won` を付与
  (**ラベル付き 15,229 行**、勝ち/負けバランス ≈ 47.5%)。
- 版権動画由来のため CSV 本体は `data/` (git 管理外)。上記スクリプトで再生成可能。

---

## 2. 相関分析の結論 (`scripts/analyze_tier1_correlations.py`)

### 2.1 冗長性 — 23 指標は実質「少数の潜在因子」に集約

|Pearson r| ≥ 0.85 の主なペア:

| r | ペア | 潜在因子 |
|---|---|---|
| **−1.000** | board_puyo_total ↔ absorption_capacity | A. 盤面充填度 |
| +0.978 | immediate_fire_power ↔ chain_efficiency | B. 現在の連鎖規模 |
| +0.917 | death_margin_neighbor ↔ absorption_capacity | A |
| −0.917 | board_puyo_total ↔ death_margin_neighbor | A |
| +0.905 | board_puyo_total ↔ max_column_height | A |
| +0.899 | current_max_chain ↔ immediate_fire_power | B |
| +0.895 | current_max_chain ↔ chain_efficiency | B |
| +0.887 | board_color_puyo_total ↔ conn_triple_count | A |
| −0.877 | board_puyo_total ↔ death_margin | A |

- **因子 A (盤面充填度)**: board_puyo_total / absorption_capacity / max_column_height /
  death_margin / death_margin_neighbor / board_color_puyo_total は
  ほぼ「盤面がどれだけ埋まっているか」1 軸の言い換え。
- **因子 B (現在の連鎖規模)**: current_max_chain / immediate_fire_power /
  chain_efficiency はほぼ同一。
- → 独立な観測軸は **6〜8 個**。占有・危険系と火力の即時系は大半が重複。

### 2.2 win 信号 — 弱く、終盤に集中

各指標と `won` の点双列相関 (手数三分位、終盤の絶対値降順):

| 指標 | overall | 序盤 | 中盤 | 終盤 |
|---|---|---|---|---|
| board_ojama_count | −0.155 | −0.084 | −0.125 | **−0.214** |
| chain_efficiency | 0.070 | 0.016 | 0.057 | **0.154** |
| current_max_chain | 0.075 | 0.035 | 0.064 | **0.147** |
| immediate_fire_power | 0.060 | 0.008 | 0.058 | 0.136 |
| max_column_height | −0.068 | −0.015 | −0.027 | −0.119 |
| reach_fire_power | 0.041 | 0.031 | 0.014 | 0.109 |
| ojama_forecast | −0.076 | −0.067 | −0.037 | −0.102 |

- 全指標 |r| ≤ 0.21。信号は **終盤に集中**、序盤はほぼ無相関 (互角局面)。

### 2.3 火力の位置づけ (前回の多変量結果との整合)

- 火力系は**単変量では終盤に +0.11〜0.15 の相関を持つ**が、多変量 win 評価では
  純増分 **ΔAUC = −0.012**(入れると悪化)だった。
- これは矛盾ではなく **「火力の信号は current_max_chain / 盤面充填度と共線で、
  独立寄与がゼロ」** ということ。しかも同じ信号を**安価な `current_max_chain`
  (連鎖シミュ不要) が持つ**。
- ⇒ tier1 では current_max_chain を連鎖規模の代表とし、重い
  `reach_fire_power` / `immediate_fire_power` は **tier2 派生指標の土台として保持**
  (2026-07-11 user 指示: 火力は tier2 以降でベースとして意味を持つ見込み)。

---

## 3. tier1 指標セットの確定

backwards-compat 遵守 (CSV 列は削除しない)。**モデル用の分類**として整理:

### 3.1 コア独立軸 (モデル主入力)

| ファミリー | 代表指標 | 備考 |
|---|---|---|
| 盤面充填度 | `board_puyo_total` | 因子 A の代表 |
| 危険 | `death_margin` | 窒息距離。充填度と別情報の分を保持 |
| 連鎖規模 | `current_max_chain` | 因子 B の安価な代表 |
| お邪魔 | `board_ojama_count` / `ojama_net_balance` / `ojama_forecast` | 最強信号 (終盤 −0.21) |
| 形 | `conn_pair_count` / `conn_max_group_size` | 連結、比較的独立 |
| 受け | `dig_resistance` | |
| テンポ | `tsumo_count_rate` / `margin_time_rate` | 進行度 |

### 3.2 tier2 土台 (保持・モデル寄与は低いまま)

`reach_fire_power` / `immediate_fire_power` / `chain_efficiency` /
`second_chain_potential` / `min_puyos_to_ignite`
→ tier2 の火力派生指標 (発火余地・期待火力・相対火力推移 等) の素材。

### 3.3 冗長・派生 (viz 用に列は残すがモデルからは除外推奨)

- `absorption_capacity` (= 72 − board_puyo_total、r = −1.000)
- `chain_efficiency` (≈ immediate_fire_power、r = 0.978)
- `board_color_puyo_total` / `max_column_height` / `death_margin_neighbor`
  (因子 A に高度に吸収される)

---

## 4. 予測ターゲット = ΔWinProb (局面評価)

### 4.1 なぜ ΔWinProb か

前回、二値「試合最終勝者」ラベルの WinProb は序盤 AUC 0.37 (予測不能) だった。
これは**バグではなく真の性質**(序盤は互角なので勝率 ≒ 50% が正しい出力)。
**ΔWinProb は勝率が動いた瞬間だけに信号を集中させる**ため、この弱点を回避する:

- 序盤の互角局面 → Δ ≒ 0(情報がないのが正しい)
- 終盤の決定打(大連鎖・お邪魔投下)→ 大きな Δ
- 「どの指標変化が勝率を動かすか」を直接学べる = 有利不利判定の本質に最短。

### 4.2 構成

1. **substrate**: `WinProb(state)` = この局面でこの側が勝つ確率。
   端末勝者ラベルで学習 (較正済み確率)。
2. **主アウトプット**: `ΔWinProb` (セイバーの WPA = Win Probability Added 型)。
   **Δ の粒度は「連続 STABLE 間」+「イベント基点」の両方** (2026-07-11 user 決定):
   - (i) 連続 STABLE スナップショット間の Δ (密。diff キー = video/game/side/tsumo 順)
   - (ii) 連鎖発火 / お邪魔投下などイベント前後の Δ (解釈用 WPA、「この連鎖で +15%」)
3. **位相別有利不利**: 位相 (手数 or 盤面ぷよ数) 別に ΔWinProb を集計・帰属。
4. **有利不利スコア (−100〜+100)**: WinProb−0.5 を写像した「現在の優勢」+
   ΔWinProb で「勢い」を表現。

### 4.3 データ規模の注意

ΔWinProb (差分) は WinProb (水準) より**ノイジー**で、安定推定には試合数が要る。
現状 10 動画 ≈ 190 試合は **プロトタイプ規模**。tier1 確定 + 本 PR 後、
Phase L の動画追加 (66→100-150) でスケールする前提。

---

## 5. 次のステップ

1. (本 PR) tier1 指標セット確定 + 相関/win 分析 + ΔWinProb 設計を記録。
2. ΔWinProb プロトタイプ実装 (連続 STABLE 間 + イベント基点、現データ + proxy で概念実証)。
3. Phase L 動画追加でスケール → ΔWinProb を本番規模で学習。
4. tier2 火力派生指標の設計 (火力計算の高速化が前提: GPU 化 / 盤面ハッシュキャッシュ /
   484 探索の枝刈り)。

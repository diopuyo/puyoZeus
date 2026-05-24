# 指標評価ロジック・レポート

**出力時刻:** 2026-04-24 16:15 JST
**目的:** 「システムが連鎖力や対応力をどう判定しているか」を人の目で検証可能にする
**使い方:** 各指標の章を読み、末尾の「実フレーム観測」と照らして、値が直感と合っているかチェック

---

## 0. 全体フロー

```
1920×1080 フレーム画像
        │
        ▼ (image_reader + CNN)
  Board (6列×13行、色コード)
        │
        ▼ (ChainSimulator)
  ChainResult (連鎖数、消去ぷよ、final_board)
        │
        ▼ (IndicatorCalculator)
  IndicatorSet = 8 指標 (各 0.0〜1.0)
        │
        ▼ (Scorer)
  total_score ∈ [-100, +100]  正=1P 有利
```

Scorer の最終合成式（`src/scorer.py:156`）:

```
p1_raw = Σ(weight × p1_score)
p2_raw = Σ(weight × p2_score)
total_score = (p1_raw - p2_raw) / Σ|weight| × 100
             (clamp: ±100)
advantage = "1P" if total > +3.0
          = "2P" if total < -3.0
          = "EVEN" otherwise  (EVEN_THRESHOLD=3.0)
```

重みの和 `Σ|weight| = 1.5 + 1.0 + 0.8 + 0.8 + 1.5 + 1.2 + 0.6 + 0.4 = 7.8` が正規化分母。

---

## 1. 本線完成度 (main_chain_maturity)  重み +1.5

### やっていること
**現盤面をそのままシミュレーションして発火したら何連鎖するか**。これを「想定最大 10 連鎖」で割る。

### 式 (`src/indicators.py:451`)
```python
count = chain_result.chain_count  # そのまま発火時の連鎖数
ratio = count / MAX_EXPECTED_CHAIN  # MAX_EXPECTED_CHAIN = 10
score = clamp(ratio, 0, 1)
```

### 意味
- score=0.0: 連鎖しない（= 消せる色グループが未形成、または形成されているが即発火の配置に至っていない）
- score=0.3 (3連鎖): 副砲規模
- score=0.7 (7連鎖): 終盤級
- score=1.0 (10連鎖+): 試合を決める本線

### 注意点・既知の限界
- **「今すぐ発火したら」の連鎖数のみ**を見ている。積んである途中で待機中の本線は見えない（それは伸ばし余地や副砲で拾う）
- 10連鎖を想定最大としているので 11連鎖以上は頭打ち（`CONNECTION_BONUS_TABLE` は 11+ 定義あるが本指標はカウントのみ）
- 実戦では「本線 7-8 連鎖＋仕掛け」が主流なので ratio=0.7-0.8 が上級者帯

### 人の目チェック
- 「絶対に連鎖する形になってるよね？」の盤面で score=0 → **CNN 読取り誤りを疑う**（空→色、色→空）
- 盤面上の連結サイズ ≥4 のグループが 0 個 → chain_count=0 は正しい
- 連鎖の閃光が見えているフレームは連鎖途中なので score は 0 になる（既に消え始めている）

---

## 2. 伸ばし余地 (extension_potential)  重み +1.0

### やっていること
**「今の盤面で色 C を列 N に 1 個置いたら本線が伸びるか？」を全組合せで試行**し、伸びた placement の比率を取る。加えて、空きセルの余裕も加算する。

### 式 (`src/indicators.py:624`)
```python
# 盤面に存在する色 × 6 列 = 最大 6 × 6 = 36 placement を試行
for color in colors_on_board:
    for col in 0..5:
        result = simulate(board + 1 puyo placement)
        if result.chain_count > base_chain:
            improved += 1
improvement_ratio = improved / total_placements
empty_reserve = min(1.0, empty_cells / 24)  # 4段分 24マスで満点
score = 0.7 × improvement_ratio + 0.3 × empty_reserve
```

### 意味
- score=0.0: どこに何を置いても伸びない（= 本線発火待ち、または完全飽和）
- score=0.3-0.5: 発火点が数個、空きスペース確保OK の通常状態
- score=1.0: 伸ばせる placement が多数＋空き十分（序盤）

### 注意点
- 1 ツモ先読みのみ。2 ツモ先は未実装（優先度低に残置）
- 「ぷよを浮かせて置けない」制約（重力落下）は `_drop_row` が正しくシミュレート
- 空きセル 24 マス = 盤面 78 マスの 30%。つまり盤面が 7 割以上埋まると empty_reserve=1.0 に達さなくなり減点

### 人の目チェック
- 序盤 (盤面スカスカ) で score<0.3 → 改善ロジックの誤り or CNN 読取りで盤面が埋まって見える
- 終盤 (本線組み上がり完了) で score=1.0 → 本線を崩す placement を「伸ばし」と誤計上している可能性

---

## 3. 副砲の質 (sub_chain_quality)  重み +0.8

### やっていること
**「あと 1 つ足せば 2 連鎖以上になる発火候補」を最大 6 箇所探索**し、実際に 1 puyo 追加してシミュレート。`連鎖≥2 かつ 消去≥8` なら viable な副砲としてカウント。

### 式 (`src/indicators.py:691`)
```python
if board.is_dead():
    return 0  # 窒息盤面は副砲議論なし
candidates = collect_candidates(board)  # size≥3 の隣接空きセル (色, 列)
viable = 0
for (color, col) in candidates:
    result = simulate(board + 1 puyo @ col with color)
    if result.chain_count >= 2 and result.total_erased >= 8:
        viable += 1
score = clamp(viable / 6.0, 0, 1)  # 6 は SUB_CHAIN_MAX_CANDIDATES
```

### 意味
- score=0.0: 副砲になる発火点が 1 つも無い
- score=0.33 (viable=2): 即撃てる副砲候補が 2 箇所（典型的な先手側）
- score=1.0 (viable≥6): 多方面撃ち分け可能（上級者の理想形）

### 注意点
- **viable の閾値 (連鎖≥2, 消去≥8) はぷよぷよ標準の「3連鎖までは単発レベル、8消去以下は副砲扱い不可」に合わせた**（`SUB_CHAIN_MIN_ERASE_CREDIT=8`, `SUB_CHAIN_MIN_CHAIN_COUNT=2`）
- 単発 4 個消し（連鎖数 1、消去 4）は副砲と呼べないため除外
- 窒息盤面 (`board.is_dead()`) では副砲を即 0 にして「危険な盤面を副砲で挽回できる」誤評価を防ぐ

### 人の目チェック
- 「発火点がいっぱい見えるのに viable_count=0」 → chain_result の読取り誤り、または CNN の色ずれで連結が切れている
- best_chain detail が 5+ なのに score が低い → candidate 不足（`SUB_CHAIN_MAX_CANDIDATES=6` で打ち切られている可能性）

---

## 4. 催促耐性 (harassment_resistance)  重み +0.8

### やっていること
**10/15/20/25/30 個のおじゃまを順番に仮想落下させて、本線連鎖数がどれだけ残るかを測る**。5 通りの平均。

### 式 (`src/indicators.py:553`)
```python
base_chain = chain_result.chain_count  # 現状の本線連鎖数
survivals = []
for count in [10, 15, 20, 25, 30]:
    ojama_board = drop_ojama(board, count)
    if ojama_board.is_dead():
        retention = 0.0  # 窒息は即死
    elif base_chain == 0:
        retention = 0.5  # 守る本線が無いなら中立値
    else:
        post = simulate(ojama_board)
        retention = post.chain_count / base_chain  # 連鎖数がどれだけ残ったか
    survivals.append(min(1.0, retention))
score = mean(survivals)
```

### 意味
- score=0.0: どのパターンでも即窒息 or 連鎖完全崩壊
- score=0.5: 本線未構築（中立扱い）
- score=1.0: 全てのおじゃまパターンを耐えきれる（理想的な形）

### 注意点・設計意図
- **本線未構築時に 0.5 を返す仕様**は「おじゃまで window 崩れ」よりも「そもそも耐える必要がない序盤」を過小評価しないため（前セッションで修正済、旧実装は 0 固定で序盤不当に下がっていた）
- おじゃま落下は `sim.drop_ojama()` が均等分散させる（6 列に順番に撒く）
- 最大 30 個は 1 画面相当（本線発火直後の仮想攻撃量）

### 人の目チェック
- **サンプル観測で 1P 側が常に 1.00 (frame_0300s, 0600s, 1500s, 2700s, 3200s)**: 本線未構築ではなく、連鎖数が維持できている健全な状態
- 盤面がほぼ空なのに 0.5 ではなく 1.0 → base_chain>0 で稀な計算成立。盤面読取り誤りの可能性
- 中途半端な形で 0.2-0.3 → おじゃまで潰される形、典型的な「危ない土台」

---

## 5. 窒息リスク (death_risk)  重み **-1.5**（負）

### やっていること
**3列目（致命列、0-indexed=2）を中心とした重み付き高さ平均**、かつ致命列が 9 段以上なら指数関数で急激に上昇させる。

### 式 (`src/indicators.py:397`)
```python
# 列ごとの重み: col=2 が最大 (1.0)、外側ほど小さい
col_weights = {0:0.3, 1:0.6, 2:1.0, 3:0.6, 4:0.3, 5:0.2}
weighted_avg = Σ(height[col] × weight[col]) / Σweight
linear_score = weighted_avg / 13

# 非線形: col=2 の高さが 9 段以上から急上昇
def danger_curve(h):
    if h < 9: return 0
    delta = h - 9
    return 1 - exp(-(delta²) / 8)

score = 0.3 × linear + 0.7 × danger_curve(col_heights[2])
# 窒息確定 (is_dead) なら 1.0 固定
```

### 意味
- score=0.0: 完全に低い盤面
- score=0.3: 中程度の積み（線形項 0.3 寄与、非線形項 0）
- score=0.5: 致命列 10 段くらい
- score=0.8+: 致命列 11-12 段（ゲームオーバー間近）
- score=1.0: 窒息確定

### 注意点・設計意図
- **線形:非線形 = 3:7 のブレンド**で、危険帯 (h≥9) が支配的になる。通常積み帯 (h≤8) では線形成分のみ
- `DEATH_RISK_WEIGHTS` で「死」というより「不安定度」も拾う（端列 0.3, 0.2 は L 字折り返しの評価用）
- **重みが負 (-1.5)** なので、高スコア = 大きな減点になる

### 人の目チェック
- 明らかに危険（3列目ギリギリ）な盤面で score < 0.5 → `danger_curve` 計算バグ or height_of() 誤算
- サンプル観測の `frame_1500s` 1P = 0.71 は非線形項が効いている状態（致命列 9+ 段）
- サンプル観測の 1P 側 0.1-0.5 は通常の積み、**0.7+ で警戒域**

---

## 6. 相殺力 (offset_power)  重み +1.2

### やっていること
**現盤面を即発火した時の総得点を計算**し、おじゃまに換算して送れる個数を求め、対数スケールで 0-1 に正規化する。

### 式 (`src/indicators.py:486`)
```python
# ぷよぷよ公式得点式:
#   step_score = 10 × erased × max(chain_power + group_bonus + color_bonus, 1)
total_score = 0
for step in chain_result.steps:
    chain_power = CHAIN_POWER_TABLE[step.chain_index]  # (0,8,16,32,64,96,128,...)
    group_bonus = Σ CONNECTION_BONUS_TABLE[group.size]  # 4→0, 5→2, 6→3, ...
    color_bonus = COLOR_BONUS_TABLE[num_distinct_colors]  # 1→0, 2→3, 3→6, 4→12
    step_score = 10 × step.erased_count × max(chain_power + group_bonus + color_bonus, 1)
    total_score += step_score

ojama_equiv = total_score / 70.0  # 70点 = 1 おじゃま
# 対数スケール（中連鎖の差が取れるように）
ratio = log1p(ojama_equiv) / log1p(72)  # 72個 = 決め手級
score = clamp(ratio, 0, 1)
```

### 意味
- score=0.0: 連鎖なし（0 点）
- score≈0.5: 約 8 個のおじゃま（3-4 連鎖規模）
- score≈0.8: 約 30 個（本線規模）
- score=1.0: 72 個以上（一画面 2 画面分以上）

### 注意点・設計意図
- **対数スケール化**は本セッションで追加した修正の一つ。旧 `ratio = ojama / 72` は 4 連鎖で 1.0 に張り付き（飽和）、中連鎖の差異が見えなくなっていた
- 対数なので 0 → 1 の増加は「初めのおじゃま数個」でも意味のある点数になる
- `MAX_OJAMA_OFFSET=72` は実測次第で引き上げ検討中（優先度中）

### 人の目チェック
- サンプル観測 `frame_0300s` / `0600s` / `1500s` の 1P=1.00 は明らかに巨大な本線想定。raw_value (detail.estimated_ojama) で実数確認可能
- 連鎖数はあるのに score が低い → 連結が小さい or 色数が少ない（公式得点式が低い）
- 対数の挙動: ojama 8 で ≈0.50、36 で ≈0.84、72 で 1.00

---

## 7. セカンド構築力 (second_chain_potential)  重み +0.6

### やっていること
**現盤面を先に発火シミュレートして残った `final_board` の上で、1 puyo 追加して連鎖するか**を全色×全列で試す。成功率と最大連鎖数の平均。

### 式 (`src/indicators.py:783`)
```python
remaining = chain_result.final_board  # 本線発火後の残骸
if remaining is empty:
    return 0.1  # 構築余地ボーナス (空盤面)

for color in colors_on_remaining:
    for col in 0..5:
        result = simulate(remaining + 1 puyo)
        if result.chain_count >= 1:
            viable_placements += 1
            best_chain = max(best_chain, result.chain_count)

placement_ratio = viable / total_trials
chain_ratio = best_chain / 3  # SECOND_CHAIN_MAX_EXPECTED = 3
score = 0.5 × placement_ratio + 0.5 × chain_ratio
```

### 意味
- score=0.0: 本線発火後の残骸では何も連鎖しない（= 本線一発勝負型）
- score=0.3: 残骸で 1 連鎖ができる placement がいくつか
- score≥0.7: 本線後も第二波を組める余地あり
- score=1.0: 残骸で 3 連鎖以上が即撃てる（理想）

### 注意点
- **本線が発火しない盤面では remaining = board そのもの**なので、未発火盤面では通常の「あと 1 ツモで連鎖するか」の指標になる
- 空盤面のみ 0.1 ボーナスで「構築余地あり」を表現
- `SECOND_CHAIN_MAX_EXPECTED=3` は仮置き、実戦値で調整余地あり

### 人の目チェック
- サンプル観測 `frame_0900s` 2P=0.92 → 残骸が綺麗な第二連鎖形になっている
- `frame_0600s` 2P=1.00 vs 1P=0.20 → 2P 側に圧倒的な後続力
- 本線を全く組んでいないのに score が高い → `remaining` が元盤面と同じで「これから組める」と評価されている

---

## 8. フィールド効率 (field_efficiency)  重み +0.4

### やっていること
**連鎖参加ぷよ数 ÷ 盤面総ぷよ数**。未発火盤面では「連結≥2 のぷよ数 / 総ぷよ数」で代替。

### 式 (`src/indicators.py:347`)
```python
normal_total = board.count_puyos() - ojama_count
participating = chain_result.participating_cells  # 連鎖で消えるぷよ数
if participating > 0:
    ratio = participating / normal_total
else:
    # 未発火盤面: 連結サイズ>=2 のぷよ数で代替
    clustered = Σ group.size for group in find_groups(board) if group.size >= 2
    ratio = clustered / normal_total
```

### 意味
- score=0.0: ぷよが全部バラバラ（未連結）or 盤面が空
- score=0.5: 半分くらいが連結 or 連鎖参加
- score=1.0: 全ぷよが連結してまとまっている（発火すれば全て消える）

### 注意点・設計意図
- **未発火盤面で 0 固定にしない**のが重要な修正（前セッション）。連結サイズ≥2 で代替することで、序盤の「ばらついた土台 vs きっちり組まれた土台」の差が出る
- おじゃまは分母から除外（連鎖参加しないので効率評価に含めない）
- 孤立ぷよ（size=1）は分子に入らない = 減点

### 人の目チェック
- 盤面がパンパンなのに score 低い → バラバラに置かれた盤面（形が悪い）
- サンプル観測 `frame_0900s` 1P=0.82 → 連結が綺麗にまとまっている土台
- `frame_0900s` 2P=0.09 → 2P 側は発火直後で残骸がバラバラ（`participating=0` で代替値）

---

## 9. 総合スコア合成 (Scorer)

### 式 (`src/scorer.py:136`)
```python
p1_raw = Σ(weight × p1.score)
p2_raw = Σ(weight × p2.score)
diff = p1_raw - p2_raw
total_score = (diff / 7.8) × 100   # 7.8 = Σ|weights|
total_score = clamp(total_score, -100, +100)
```

### 有利判定閾値
- `EVEN_THRESHOLD = 3.0`: `|total_score| ≤ 3` なら "EVEN"（互角）
- 正値 → 1P 有利 / 負値 → 2P 有利

### 寄与度の読み方 (breakdown)
各 `breakdown[指標名] = weight × score` を見れば「どの指標が総合スコアをどれだけ動かしているか」が分かる。

---

## 10. 実フレーム観測とレビュー観点

以下は `data/frames/sample/frame_XXXXs.png` を CNN + indicators に通した結果。ぷよ数は CNN 認識結果（1920×1080 生フレーム、CPU 推論）。

| フレーム | 1P ぷよ | 2P ぷよ | total | 判定 | 支配的な指標差 |
|---|---|---|---|---|---|
| `frame_0300s` | 43 | 5 | **+27.8** | 1P | offset(+1.00), harassment(+0.50), field(+0.34), sub(+0.33) |
| `frame_0600s` | 45 | 38 | +14.1 | 1P | offset(+1.00), harassment(+0.50), death_risk(+0.42) 対 2P の second(-0.80) |
| `frame_0900s` | 46 | 45 | -13.5 | 2P | second(-0.52), offset(-0.40), harassment(-0.50) 対 1P の field(+0.73) |
| `frame_1500s` | 57 | 55 | +20.5 | 1P | offset(+1.00), harassment(+0.70), sub(+0.50), death_risk(+0.50) |
| `frame_2100s` | 18 | 18 | +1.1 | EVEN | ほぼ全指標 diff≈0 (互角) |
| `frame_2700s` | 28 | 38 | +17.2 | 1P | offset(+0.73), harassment(+0.50) 対 2P second(-0.40) |
| `frame_3200s` | 30 | 13 | +27.7 | 1P | offset(+0.78), sub(+0.67), harassment(+0.50) |
| `frame_detection` | 18 | 18 | +1.1 | EVEN | 2100s と同じ盤面 |

### レビュー観点

**✓ 納得できる結果**
- `frame_0300s` (1P=43, 2P=5): 1P だけ積んでいるので圧倒的 1P 有利。total +27.8 で 1P 判定 → **OK**
- `frame_2100s` (1P=2P=18 同数): 全指標ほぼ同値で互角 → **OK**
- `frame_3200s` (1P=30, 2P=13): 1P 側が圧倒的、sub_chain(+0.67) と offset(+0.78) で 1P 優勢明らか → **OK**

**△ 要確認な結果**
- `frame_0600s` 1P=45/2P=38 で total +14.1 だが、内訳を見ると 2P の `second_chain_potential=1.00` が異常に高い。**1P の本線が組み上がっていて即発火可能、2P は既に発火後で残骸から第二波が組めている**、という解釈なら妥当。盤面目視で 2P 側が本線発火中なら納得
- `frame_0900s` 1P=46/2P=45 で **-13.5 (2P有利)** → 1P の death_risk=0.17 で危なくない、field_eff=0.82 で形はいい、しかし second が 0.40 対 0.92 で差が付き、harassment も 0.50 対 1.00。**2P が本線を発火直後で次の一手にすぐ繋げられる状態**と推測。要目視

**⚠ 注意すべき結果**
- **`harassment_resistance` が本線未構築時に 0.5 になる仕様**: `frame_0300s` の 2P=0.50 は 2P ぷよ 5 個なので本線未構築（chain=0）で中立返し。これは「攻撃するものが無いから耐性評価保留」という意図的な設計。ただしこの 0.5 が total に引き上げ寄与してしまう可能性がある

---

## 11. 人の目チェックリスト

実際の試合フレームで各指標を目視確認する時の観点:

### 本線完成度
- [ ] 盤面の連結サイズ≥4 のグループを数え、連鎖可能列（発火点）があるか
- [ ] 見えている連鎖数 vs system の `chain_count` が近いか
- [ ] スコアが高い(>0.5) のに目視で「発火したら大したことない」と感じたら、CNN 読取りで余計な連結ができていないか確認

### 伸ばし余地
- [ ] 盤面の空き具合と score の関係（スカスカなら 0.8+ が妥当、満杯なら 0.3 以下）
- [ ] improvement_ratio detail が 0.5 以上 = 発火点選択肢が多い盤面

### 副砲の質
- [ ] viable_count と best_chain detail を見る。2+ 箇所で 3 連鎖以上組めれば副砲 OK
- [ ] 窒息直前の盤面は 0 固定（意図的、挽回評価防止）

### 催促耐性
- [ ] 0.5 が出たら「本線未構築」＝ 試合序盤チェック
- [ ] 10 個投入ですぐ窒息する形なら 0.2 以下、堅い土台なら 0.8+

### 窒息リスク
- [ ] 致命列 (3列目) の高さを目視カウント、>9 段なら非線形項が急上昇
- [ ] 窒息確定（`is_dead`）なら必ず 1.0

### 相殺力
- [ ] detail.estimated_ojama で実おじゃま数を確認
- [ ] 12 個 (1 画面) でおよそ 0.63、30 個で 0.84、72 個で 1.00 という対数感覚

### セカンド構築力
- [ ] remaining (final_board) が現盤面と同じなら未発火扱い、空なら本線完全発火扱い
- [ ] 本線を組んでいる最中で高スコア → 本線の一部が第二連鎖として誤認されている可能性

### フィールド効率
- [ ] used_fallback detail を見る: True なら未発火モード（連結≥2 のぷよ比率）、False なら発火時参加比率
- [ ] バラバラな土台で 0.3 以下、きれいに連結してれば 0.8+

---

## 12. 再実行方法

このレポートの数値を再現するには:

```bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
./venv/bin/python scripts/validate_indicators_live.py
```

- GPU を使わない（`CUDA_VISIBLE_DEVICES=""`）ので走行中の学習には影響なし
- `data/frames/sample/frame_*.png` を全て走査
- 各フレームごとに 1P/2P の全指標を表示

**判断が怪しい場合の深掘り** — `IndicatorResult.detail` を print 追加して個別確認:
- main_chain の `total_erased`
- extension の `improvement_ratio`, `empty_cells`
- sub_chain の `viable_count`, `best_chain`, `candidate_count`
- harassment の `base_chain`, `survival_by_count` (各おじゃま数での残存)
- death_risk の `col_heights` (0-5 列の高さ配列), `linear`, `nonlinear`
- offset の `chain_count`, `erased_puyos`, `estimated_ojama`, `total_score`
- second の `viable_placements`, `best_chain`, `remaining_empty`
- field_efficiency の `participating`, `normal_puyos`, `used_fallback`

---

## 13. 指標の健全性チェック（学習と独立）

CNN の精度とは別に、**指標ロジック自体の妥当性**は以下のテストで担保されている:

```bash
./venv/bin/python -m pytest tests/test_indicators.py -v  # 指標計算ロジック
./venv/bin/python -m pytest tests/test_scorer.py -v      # 総合スコア合成
./venv/bin/python -m pytest tests/ -v                    # 全体回帰
```

直近の実行結果: **369 passed, 1 skipped**（前セッションで別エージェントによる指標式大幅修正後の回帰検証、本セッション冒頭で再確認済）。

---

**このレポートは人の目による指標検証用です。** 数値が直感と合わない場合、まず (1) CNN の盤面読取りを疑い、(2) 次に指標の detail フィールドを print して論理を追い、(3) それでも納得できなければ `tests/test_indicators.py` に反例を足して再現させてください。

# 交換エピソード会計 仕様書 (2026-08-24)

状態: **Gate 1 ドラフト**。実装前の合意用。既存挙動は一切変更しない。
対象 Gate: `docs/agent_coordination/PLAN.md` の Gate 1 / Gate 2。

---

## 0. なぜ作るのか

2026-08-24 に user から指摘された「99%の勝率から1%に急降下する」現象の根因は、
**一つの撃ち合いを一つの出来事として数えていない**ことにある。

実際に起きていたこと (seg01 game2、実画面33枚で確認済み):

- 1P が撃ち返した 517個が、会計に載るまで **11秒** かかった。
- その間に 2P の 720個だけが「新規攻撃」として計上され、方向が反転した。
- さらに上限216で丸めた値どうしを引き算したため、**存在しない攻撃**が生まれた。
- 連鎖イベントが1.4秒ごとに分裂し、火力が1フレームで 442 → 101 に落ちた。

いずれも「片方の攻撃だけを見て決着を判定した」ことの帰結である。
**撃ち合いは往復で1つの出来事**であり、往復が終わるまで残量は確定しない。

この仕様書は、その「往復1つ」を `ExchangeEpisode` として定義し、
生成・相殺・送付・着弾が**足し引きして必ず合う**会計を与える。

---

## 1. 用語 (user 伝授のドメイン規則に従う)

推測禁止。以下はすべて user 伝授またはコード実測で確定済みの事実に基づく。

| 用語 | 定義 | 典拠 |
|---|---|---|
| **発火 (fire)** | ぷよが消え始めること。画面では score エリアに掛け算式 `N×M` が出る | `reference_chain_phase_detection_spec` |
| **掛け算式 (formula)** | **消えるたびに**1回出る。左=消した数×10 (3桁あり)、右=倍率 (3桁あり) | `reference_chain_formula_per_step_2026-08-22`, `reference_chain_formula_layout_2026-08-24` |
| **連鎖数** | 掛け算式が出た**回数**。推定も simulate も不要 | 同上 |
| **火力 (生成量)** | 各段の (左×右) の**総和**。得点 → おじゃま換算 | 同上 |
| **連鎖の終わり** | **連鎖している側**のネクストが動いた瞬間 **または** **連鎖している側**にお邪魔が落ちた瞬間 (絶対律) | `reference_chain_end_absolute_signals_2026-08-21` |
| **相殺** | 送付量と受領量が打ち消し合うこと。相殺で予告が無くなれば降らない | `reference_ojama_forecast_landing_spec_2026-08-21` |
| **着弾 (landing)** | 予告おじゃまが実際に盤面へ降ること | 同上 |
| **催促** | 発火時に保持色ぷよの **60%未満**しか消費しない連鎖、かつ送りおじゃま **>4個** | `reference_saisoku_exchange_model_2026-07-22` |
| **整地** | <60%消費 かつ 送りおじゃま **≤4個**。攻撃ではない | 同上 |
| **本線** | 60%以上消費 | 同上 |
| **対応** | 相手の催促に催促で返すこと | 同上 |

### 1.1 着弾の3条件 (すべて満たして初めて降る)

1. 相手の連鎖が確定している (相手のネクストが動き始めた瞬間)
2. **受け側の**現在の手がフィールドに置かれた
3. 受け側で連鎖が起きなかった

連鎖が起きた場合は**その連鎖の終了後**に降る。ただし相殺で予告が無くなれば降らない。
既に降下待ちの状態への**追加分は即降る**。

典拠: `reference_ojama_forecast_landing_spec_2026-08-21`、
`reference_ojama_landing_gated_by_placement_2026-07-29`。

---

## 2. ExchangeEpisode の定義

### 2.1 episode とは

**一方の発火から始まり、その応酬が物理的に決着するまで**を一つの episode とする。

`催促対応 → 相手本線 → 自分本線 → 相殺 → 着弾` の全体が **1 episode**。
途中の個々の連鎖は `ExchangeEvent` であり、episode の構成要素にすぎない。

### 2.2 開始条件

**どちらか一方に最初の FIRE イベントが立った瞬間**に episode を開く。
このとき episode は `OPEN` 状態になる。

「両側同時に chain_event がある」ことは**要求しない**。これが現行
`ResolvedExchangeTracker` の起動条件 (`ev1 is not None and ev2 is not None`) であり、
片側連鎖でクラス全体が起動しない原因だった
(`project_chain_scale_comparison_backlog_2026-08-22`)。

### 2.3 参加条件 (時間差の応射を同じ episode に入れる)

新しい FIRE イベントは、次の**いずれか**を満たすとき既存の open episode に参加する。

- **(P1) 未解決残量がある**: episode の符号付き純残量が 0 でない。
  つまり、まだどちらかが「借り」を負っている。
  **ただし §2.3.1 の裏付け規則を満たす残量に限る。**
- **(P2) 未着弾の生成がある**: 生成済みだがまだ着弾も相殺もしていない量が残っている。
- **(P3) 相手の連鎖が進行中**: 発火時点で相手側が CHAIN / GRAVITY_SETTLE にある。

いずれも満たさない発火は、**新しい episode を開く**。

> **設計判断**: 参加条件を「時間窓」にしない。時間窓は 30先動画の 10秒以上続く
> 本線応酬を表現できず、固定 0.96秒ヒステリシス (現行案B) が全域で悪化した
> 直接の原因である (`DECISIONS.md` 2026-08-24)。
> 参加条件は**物理状態**で決める。

#### 2.3.1 【必須】残量は物理観測に裏付けられていること (癒着の防止)

**P1 を無条件にすると、試合全体が1つの episode に癒着する経路が実在する。**

`net_raw` の供給元は score OCR 差分ベースの会計である。W2 の桁誤読
(`docs/KNOWN_WEAKNESSES.md:19-29`) が数個〜数十個の**物理実体のない残量**を作ると
P1 が恒久的に成立し、以後の全発火が同一 episode に吸い込まれる。
安全弁は `EPISODE_MAX_SEC=60秒` だけなので、score OCR が破綻する動画
(c26 / c58 / c69、`project_video_difficulty_3broken_2026-07-29`) では
「60秒ごとに強制切断 → 即再癒着」を繰り返し、実質**全試合で hard override 禁止**になる。

したがって P1 の残量は次のいずれかの物理観測に裏付けられている必要がある。

- 受け側の予告表示 (`pending_*_uncapped`) にその量が載っている、**または**
- どちらかの側が現に連鎖中である、**または**
- 未着弾の `PROVISIONAL` / `FINALIZED` イベントがその量を説明できる

裏付けのない残量は `unreconciled` へ隔離し、**参加条件の判定から外す**。
隔離した量と件数は必ずカウンタに出す (黙って捨てない)。

> P3 についても、W32 系の状態誤検知 (`docs/KNOWN_WEAKNESSES.md:1497-`) で
> CHAIN が張り付けば無関係な発火が延々参加する。P3 は「参加」のみに使い、
> **P3 単独で episode を開かない**。R3 参照。

### 2.4 終了条件

次の条件が成立したら CLOSED。

- **(E2) 着弾しきって閉じる**: 両側とも連鎖中でなく、`provisional_residual() == 0` かつ
  残量がすべて `LANDED` として消化済み。

> **【訂正 — E1 の削除 (2026-08-24)】** 初版は「`net_raw() == 0`」を終了条件 E1 と
> していたが削除する。`net_raw() == 0` は相殺の**運命**を示すだけで、相殺の
> **取引記録ではない**。相殺は `cancel_own_pending_then_send_surplus`
> (`src/ojama_accounting.py:742`) が攻撃確定の瞬間に計算する実イベントであり、
> 正しく配線された系では CANCEL 供給後に E2 が必ず成立するため、
> E1 が E2 と異なる結論を出すのは「相殺が観測されていない」異常時のみ。
> その場合に閉じるのは「撃ち合いが会計に載らないまま決着扱いになる」こと
> (`project_pm100_display_flip_2026-08-24` の根因そのもの) であり、許さない。
> 相殺・着弾の供給が来ないまま `EPISODE_MAX_SEC` に達した episode は
> CLOSED_FORCED とし、`has_settlement_input=False` の強制終了件数は
> **Gate 3-2b の配線完了後ほぼゼロ**であることを受け入れ条件とする。

> **【訂正 — 初版からの変更】** 初版は「上記が成立した後さらに受け側が1手置くこと」を
> 4番目の条件として要求していた。これは誤りである。
> - `net_raw() == 0` のとき「受け側」が存在せず、誰の1手を待つのか定義できない。
> - 着弾は受け側の設置でしか起きない
>   (`reference_ojama_landing_gated_by_placement_2026-07-29`) ので、
>   **最後の着弾それ自体が設置と同時**である。E2 が成立した時点で
>   「着弾の機会」は既に与えられており、もう1手待つ根拠がない。
>
> 着弾の3条件 (§1.1) は「**着弾時刻を予測する物理**」であって
> 「episode を閉じる条件」ではない。混ぜると実装がバグる。

なお 517個級の残量は 1ターン上限 `OJAMA_MAX_DROP_PER_TURN`
(`src/exchange_virtual_board.py:326-327`) の刻みで 18手 = 十数秒かけて降る。
この間 episode が OPEN のままなのは物理に合致しており、正しい。

#### 2.4.1 OPEN episode の並存を許さない

同時に OPEN な episode は**高々1つ**とする (`current_episode()` が単数を返す
API と整合させるため)。参加条件を満たさない発火が来たときは、
**現在の episode を先に CLOSED_FORCED で閉じてから**新しい episode を開く。
強制的に閉じた件数と、そのとき残っていた `unreconciled` を必ずカウンタに出す。

### 2.5 強制終了 (安全弁)

次のいずれかで episode を強制的に `CLOSED_FORCED` にする。会計は保存則を満たすよう
残量を `unreconciled` として明示的に記録する (黙って捨てない)。

- 試合が終了した (WIN パネル / 試合境界)。
- どちらかが実際に窒息した (`is_dead` が **STABLE 確定盤面**で成立)。
  ※盤面が高いだけでは窒息としない (`reference_full_board_is_not_death_2026-08-22`)。
- episode 開始から `EPISODE_MAX_SEC` (既定 60.0 秒) 経過した。

#### 2.5.1 【2026-08-25 追記、Fix【2】】ワイプ (片側だけの予告消滅) の side 単位退役

`src/ojama_accounting.py:_reset_side_boundary` は負けた側の
`forecast_incoming`/`forecast_incoming_uncapped` を**無音でゼロクリア**する
(物理的には正しい。負けた側は受け取らずに終わる、
`reference_ojama_landing_gated_by_placement_2026-07-29`)。この「ワイプ」を
観測経路 (§6.3) が検知したら、`ExchangeLedger.retire_side_chains(side, ...)`
でその side に向いていた未決着分だけを退役させる。

**試合境界 (両側、上記) とは別の side 単位の事象**である。
ワイプは片側 (負けた側) で起きるため、両側を退役させる既存経路
(`_retire_all_chains_at_match_boundary`) を流用すると、片側だけ死んだ
場面で**相手の正当な未決着まで消してしまう**。

ワイプの検知は、遅い統合 `game_idx` (試合境界) を待たずに episode を
決着させる別経路としても使える。ただし **episode の試合跨ぎ禁止 (I11)
自体は統合 `game_idx` のまま**変えない (episode は両側の出来事のため)。

---

## 3. 符号の規約

**純残量 `net` は「1P から見た符号付きの量」で表す。**

- `net > 0` … 2P が受けるべき未解決のおじゃまが `net` 個ある (1P 優勢)
- `net < 0` … 1P が受けるべき未解決のおじゃまが `|net|` 個ある (2P 優勢)
- `net == 0` … 完全に相殺された、**または未記録の応酬が釣り合っているだけ**
  (よって終了条件には使わない。**この一行が E1 誤設計の源だったので、
  §2.4 の訂正注記とあわせて必ず残す**)

> 学習特徴量の `diff` 列は「自分−相手」で 0 が互角という別の規約がある
> (`project_data_facts_2026-08-21`)。本会計は**表示・判定用の内部量**であり、
> 学習列へ流す際は変換層で明示的に符号を合わせる。
> `feedback_symmetry_flip_column_types_2026-08-10` (全列一括変換の禁止) に従い、
> 本会計が出す列は**対称化反転の対象外リスト**へ明示登録する。

---

## 4. chain_id の状態遷移

**状態を持つのは `chain_id` であって、個々の `ExchangeEvent` ではない。**
`ExchangeEvent` は frozen な観測事実であり、可変の状態を持たない (§7.1)。
台帳が `chain_id` ごとに下記の状態を保持する。

```
   PROVISIONAL  --(確定スコア到着)-->  FINALIZED
        |                                  |
        |                                  v
        +------------------------->   RECONCILED  --(降下)-->  LANDED
                                           |
                                           +--(相殺で消滅)--> CANCELED
```

| 状態 | 意味 | 生成量の出どころ |
|---|---|---|
| `PROVISIONAL` | 発火は観測したが確定スコアがまだ来ていない | 掛け算式の段の和 (`STEP` イベントの合計) |
| `FINALIZED` | 確定スコア差分で生成量が確定した | score OCR 確定差分 (§4.1.1 の 3 分岐) |
| `RECONCILED` | 相手の生成量と突き合わせて相殺処理を終えた | 変わらない |
| `LANDED` | 実際に盤面へ降った | 変わらない |
| `CANCELED` | 相殺で全量消滅し、降らなかった | 変わらない |

> **【訂正 — 初版からの変更】** 初版はこの節を「`ExchangeEvent` の状態遷移」とし、
> frozen dataclass に `state` フィールドを持たせていた。**矛盾していたので改めた。**

### 4.1 暫定値の「置換」規約 (二重計上の禁止)

> **【訂正 — FINALIZE の値供給源 (2026-08-25)】** FINALIZE の値供給源は
> `OjamaAccountingTracker` の score OCR 確定差分 (`src/exchange_ledger.py`
> の `FINALIZE_SOURCE_SCORE_OCR_DIFF`) に**限定**する。
> `mechanism='baseline'` (`ChainSimulator` 産の推定、
> `src/chain_detector.py:277-317` `_try_emit_event` が
> `self._simulator.simulate(...)` の結果をそのまま値に使っている) を
> FINALIZE の**値**に使うことを禁止する。baseline (`ObservationKind.
> CHAIN_SETTLED` に改名) は連鎖終了の**合図**と、`growth_observed=False`
> かつ score OCR が一度も来なかったときの低信頼フォールバック
> (`finalized_source="simulate_fallback"`) のみに使う。
> 実測乖離 63.6% (下げ置換ゲートの発動率) は**ゲートの故障ではなく
> 入力の故障**だった (fable アーキ裁定、§8 I16 参照)。

**絶対規則: 確定時に加算しない。置換する。**

```
finalize(event, confirmed_amount):
    delta = confirmed_amount - event.amount     # 差分だけを台帳へ反映
    event.amount = confirmed_amount             # 置換
    event.state  = FINALIZED
    ledger.apply_delta(event.side, delta)
```

台帳は `amount` の**現在値**の総和として純残量を持つ。したがって
`finalize` を何度呼んでも冪等であり、同じ `chain_id` の暫定値と確定値が
二重計上されることはない。

> 現行の断片化バグは、断片ごとに新しい `ChainEvent` に**置き換わる**ため
> 「最後の断片しか見えない」= 過小になる方向だった (840 → 84、10分の1)。
> 一方で `EarlyFireTracker` / `ChainGenerationAccumulator` のように
> `trigger_sec` 変化で**加算**する経路もあり、両者が同居すると今度は過大になる。
> **`chain_id` を主キーにした置換**にすれば、どちらの事故も構造的に起きない。

#### 4.1.1 【決着済み】下げ置換のときだけ検算ゲートを通す

争点だったのは「確定 < 暫定」のとき。既存の `_maybe_redecide`
(`scripts/visualize_advantage_overlay.py:1826-1835`) は `max(予測, 確定)` の
一方向 latch で、「simulate が壊滅的に過小 (真値8連鎖→1)」への保険だった
(`project_chain_count_both_untrustworthy_2026-07-30`)。

**結論: 置換を既定とする。`max` の恒久維持は採らない。** 理由は3つ。

1. **その保険の前提が掛け算式には当てはまらない。** あの教訓の主語は
   「認識盤面から simulate した予測」で、連結欠損 (W1) が原因の**機構固有**のバイアス。
   掛け算式はスコア表示と同一フォント・同一グリッドの**画面実測値**であり
   (`reference_chain_formula_layout_2026-08-24`)、過小方向に系統的に倒れる前提がない。
2. **正常時はむしろ「確定 ≥ 暫定」が物理的な期待値。** 確定スコア差分は
   落下ボーナスと全消しボーナス (2100点 = おじゃま30個相当) を含むが、
   掛け算式の素点和は純連鎖得点のみ。したがって
   **「確定 < 暫定」はそれ自体が異常信号**であり、黙って下げるのも黙って max で固定するのも危険。
3. **`max` は保存則 I1 と両立しない (決定打)。** 生成量が実際より大きい値でラチェットされると、
   相殺 + 着弾の実測合計と永久に一致せず、**全 episode で `unreconciled` が偽陽性に残る**。
   PLAN Gate 4 の必須条件「交換終了時の生成・相殺・送付・着弾の保存則を全 episode で満たす」
   を構造的に落とす。

**手続き:** 本表は**確定 = score OCR 確定差分 (`FINALIZE_SOURCE_SCORE_OCR_DIFF`)
のときのみ適用する** (2026-08-25 追記)。それ以外の出所 (`simulate_fallback`)
は既定で台帳が拒否するため、本表の判断の対象にすら乗らない (§8 I16)。

| 条件 | 動作 |
|---|---|
| 確定 ≥ 暫定 | **無条件で置換** (通常経路) |
| 確定 < 暫定 かつ 乖離が「落下ボーナス上限 + leftover 繰越」で説明できる小差 | 置換 |
| 確定 < 暫定 かつ 大差、暫定側の全段が物理検算 (左辺が10の倍数) を通過 | **置換せず両値を保持**し、差を `unreconciled` へ計上。カウンタと dump 列に出す |

閾値定数は**シーンから逆算しない**。Gate 0 の掛け算式 E2E 10ケースの誤差分布
(実測で c07 +3.1%、c08 +9.7% の乖離を観測済み) から確定する。

#### 4.1.2 暫定も確定と同じ換算規約を通す

確定側は `score_to_ojama` (マージンタイム逓減 `compute_effective_rate` + leftover 繰越、
`src/ojama_accounting.py:712-719`) を通る。**暫定 (掛け算式の素点和) を固定レート70で
割ってはいけない。** 117分級の長試合ではマージンタイムでレートが大きく変わり、
正常時でも暫定と確定が系統的に乖離して §4.1.1 の判断を汚染する。

**暫定も確定と同一の `score_to_ojama`・同一の経過時刻で換算する。**

> **【訂正 — 自己検算の比較対象 (2026-08-25)】** 初版は `ChainEvent.ojama_sent`
> を「権威ある値」として自己検算に使う想定だったが、この値は W38
> (`src/chain_detector.py:186` の `match_start_sec` が 0.0 固定) の影響下に
> あり、320秒を超える動画位置ではレートが 1 に落ちて点数がそのままおじゃま
> 個数になる。実測 16/16 でレート1、`self_check_max_abs_diff` が最大
> 108,961 という壊れた差分として現れた (Gate 3-2b 実データプローブ、
> `data/verify/gate3_episode_2026-08-24/diagnostics.json`)。
> 比較対象は `OjamaAccountingTracker` 由来の確定生成量
> (`ChainEventObservation.authoritative_ojama`) に差し替える。
> W38 の根治 (`src/chain_detector.py` / `src/recognition_pipeline.py` の
> 修正) は user 判断待ちであり、本仕様書・本 tracker では触れない。

> **【再訂正 — 自己検算そのものを廃止 (2026-08-25、Fix【5】)】**
> 新設計 (`ObservationKind.SCORE_FINALIZE`) では score OCR 確定差分
> (`GenerationObservation.generated_delta`) の値がそのまま会計の確定値に
> なる。`authoritative_ojama` の供給元 (`OjamaAccountingTracker.
> total_generated_by_pX`) は `finalized_score` と**同一の accumulator**
> であり、フレーム一致を直しても常に差 0 になるだけの tautological な
> 検算だった。2026-08-25 の実測でも `n_authoritative_ojama_present =
> 0/20, 0/7` (供給自体がほぼ機能していなかった) を確認した。
> **意味のない「0 が合格」は誤読されるため、`self_check_*` カウンタと
> `ChainEventObservation.authoritative_ojama` をフィールドごと削除した**
> (前任コーダと診断役が独立に同じ結論に達した)。`ChainEventObservation.
> ojama_sent` (W38 の影響下) は診断で参照する可能性があるため残すが、
> 会計にも検算にも使わない。

---

## 5. chain_id の定義 — 断片の統合

### 5.1 問題

`formula` 機構は「既にアクティブな疑似イベントがあれば新規発火しない」設計のため、
ホールド期限 (`chain_hold_base_sec + 0.3秒 × chain_count`) が切れるたびに
**その時点で残っている分だけを対象にした新しい `ChainEvent` に置き換わる**。
実測で 1.367〜1.4秒周期の分裂が確認されている
(`project_chain_event_fragmentation_accumulator_2026-08-22`)。

### 5.2 chain_id の割り当て規則

**同一 side について、連鎖の「終わりの絶対律」が観測されるまで chain_id を維持する。**
すなわち次の両方が成り立つ間は同じ chain_id とする。

1. その side のネクストが動いていない、**かつ**
2. その side のフィールドにおじゃまが落ちていない。

`chain_id` は `(side, first_trigger_sec)` を種にした単調増加の整数とする。
**ホールド期限では chain_id を切らない。** ホールド期限は「疑似イベントの
表示寿命」であって、連鎖の終わりではない。

> **【訂正 — 初版からの変更】** 初版は条件2として「その side が STABLE に復帰していない」を
> 置いていた。**これは断片化バグの再導入であり、削除する。**
>
> 連鎖中の CHAIN ↔ STABLE ↔ GRAVITY_SETTLE の明滅は 0.1〜2.0 秒周期で恒常的に起きる。
> seg01 game2 の実 dump でも、11.47 秒続く 1 本の物理連鎖の途中で
> `state1` が **CHAIN → STABLE → CHAIN と 267 ミリ秒だけ復帰する**フリッカが観測されている。
> `CHAIN_COALESCE_WINDOW_SEC=2.5秒` (`src/ojama_accounting.py:123-130`) は
> まさにこの明滅を吸収するために存在する。
> 生の STABLE 復帰で chain_id を切れば、**明滅1回で断片化が完全復活する**。

#### 5.2.1 信号を取り逃したときの被害は非対称 — merge 側に倒す

- **癒着 (2 つの物理連鎖が 1 つの chain_id になる)**: score 差分の finalize が
  両方を包含するため、**台帳の保存則は保たれる**。壊れるのは連鎖数と催促分類の帰属だけ。
- **分裂 (1 つの物理連鎖が 2 つの chain_id になる)**: 1 つの確定値をどちらに finalize するかが
  不定になり、**二重計上または孤児 `PROVISIONAL` を生む**。

したがって迷ったら **merge 側に倒す**のが正しい設計バイアスである。
絶対律の判定には settle デバウンス (`K_SETTLE_FRAMES` 相当) を必ず入れ、
瞬間的な信号で切らない。

#### 5.2.2 絶対律の検出器は既知の事故源である

「ネクストが動いた」信号は、**連鎖中に 1.37 秒おきに誤検知した前科の当事者**である
(`project_slide_false_positive_root_cause_2026-08-22`)。断片化・連鎖数誤検知 (15→5)・
火力10分の1の**すべての起点**だった。

絶対律の**物理は正しい**が、**検出器は本プロジェクトで最悪級の事故源**である。
本会計は「連鎖中ゲート付きの修正済み検出器」を使うことを**前提条件**とする。
どちらの信号で chain_id を閉じたか (next 動作 / おじゃま着弾 / 強制) を
必ず記録し、Gate 3 で「どちらでもないのに閉じた件数」を実測する (R2)。

#### 5.2.3 `FormulaStepAccumulator` との階層関係

`FormulaStepAccumulator` (`src/score_ocr.py:966-1089`) は
**段 (step) の供給元**であり、`chain_id` の管理者ではない。
同クラスは独自のセッション切断規則を持つ
(`FORMULA_SESSION_RESET_SEC=2.0秒` の読取り途絶、右辺減少 + 0.5秒ギャップ) が、
これは §5.2 の絶対律とは**別の機構**である。混同してはならない。

- **内側 (accumulator)**: 掛け算式を読んで段を確定し、`(t_sec, left, right)` を出す。
- **外側 (chain_id 付与器、新規部品)**: 段 + 絶対律信号を受けて `chain_id` を管理する。
  **accumulator のセッション切断では chain_id を切らない。**

この階層を分けないと、バースト・煙 (W1 系) で掛け算式が 2 秒読めなかっただけで
accumulator のセッションが切れ、**断片化が別の形で再発する**。

`chain_id` 付与器は **Gate 2 の成果物に含める** (純関数として実装)。
含めないと不変条件 I5 (断片統合) が Gate 2 でテストできない。

#### 5.2.4 【2026-08-25 追記、Fix【3】】AWAITING_FINALIZE 中の段数増加は継続

`SUPERSEDED` (§4 の CloseReason) で確定なしに閉じた連鎖の暫定量は確定で
置換されず永久に生成量に残る問題 (zenchi 実測: SUPERSEDED 9 本の暫定合計
≈ 総生成の 23%) の根治。

**物理的根拠**: すべての連鎖は 1 段目から始まる。新しい連鎖の最初の段は
必ず `chain_count=1`。掛け算式の段カウントは実測で 38/38 単調
(`data/verify/gate3_chainid_2026-08-24/summary.json`)。したがって
AWAITING_FINALIZE 中 (§4 の絶対律を早合点しただけの状態) に、
**現在の running max より大きい段数**が観測できるのは、その連鎖が
まだ続いている場合しかありえない。

`ChainIdResolver._handle_formula_step_while_awaiting_finalize` の遷移:

| 条件 | 動作 |
|---|---|
| `obs.chain_count > state.step_count` | 同じ chain_id のまま GROWING に戻す (段・スコアを更新) |
| `obs.chain_count <= state.step_count` | 従来どおり SUPERSEDED で閉じてから新規発行 |

実データ (v51、4/4 正解): 1→2 (継続) / 1→2 (継続) / 2→6 (継続) /
10→1・10秒後 (新規) の 4 例すべてでこの判定規則が正解と一致した。

### 5.3 統合の手続き

新しい formula 再トリガーを観測したとき:

```
if 同一 side に open な chain_id がある and 終わりの絶対律が未観測:
    既存 chain_id へ「段」を追加する (連鎖数 += 1、生成量 += 当該段の左×右)
else:
    新しい chain_id を開く
```

これにより:
- **連鎖数** = その chain_id に属する段の数 (画面の掛け算式の回数と一致)
- **火力** = その chain_id に属する段の (左×右) の総和

> 掛け算式の値の検算に使える物理制約: 連鎖アニメ中の得点変化は設置がないので
> **純粋な連鎖得点 = 必ず10の倍数** (`docs/KNOWN_WEAKNESSES.md` W2)。

### 5.4 終わりの絶対律が観測できなかった場合の保険

信号を取り逃した場合に chain_id が無限に伸びるのを防ぐため、
`CHAIN_ID_MAX_SEC` (既定 30.0 秒。実測アニメ最長より十分長い) で強制的に切る。
切った件数は**必ずカウンタに出す**。黙って切らない。

実測の連鎖アニメ時間は `2.61 + 1.17×N` (23動画418イベント)。
15連鎖でも 20.2秒であり、30秒は十分な余裕がある。

---

## 6. 上限 (cap) の扱い — 既存の並行帳簿を使う

**数学用の pending は cap 前の値を使う。表示用 pending だけ従来 cap を維持する。**

理由: `PENDING_ABS_CAP=216` で丸めた値どうしを引き算すると、
実送付 517 が 216 に丸められた後で 720 との差を取ることになり、
**架空の攻撃が生まれる** (`project_pm100_display_flip_2026-08-24` の根因②)。

### 6.1 既に実装済み — 新規に作らない

2026-08-24 に `src/ojama_accounting.py` へ **cap 切り捨てなしの並行帳簿**が
既に入っている。本会計はこれを**そのまま使う**。同じものを作り直さない。

| 実体 | 場所 | 内容 |
|---|---|---|
| `pending_p1_uncapped` / `pending_p2_uncapped` | `src/ojama_accounting.py:222-223` | snapshot に出る cap 前 pending |
| `forecast_incoming_uncapped` | `src/ojama_accounting.py:241` | `_SideState` の cap 前実額。「kill_override の相殺の引き算専用」と明記済み |
| `PENDING_UNCAPPED_SANITY_MAX` | `src/ojama_accounting.py:101` | 並行帳簿側の唯一の上限 (サニティのみ) |
| 相殺の適用 | `src/ojama_accounting.py:741-748` | `cancel_own_pending_then_send_surplus` を cap 前で回す |
| 試合境界クリア | `src/ojama_accounting.py:997` | 並行帳簿も境界でゼロ |

台帳が持つ量は次の2つ。

| 量 | cap | 用途 | 供給元 |
|---|---|---|---|
| `net_raw` | **なし** | 相殺・保存則・判定 | `*_uncapped` |
| `net_display` | あり (216) | 画面表示のみ | 従来の `pending_p1/p2` |

保存則の検査は `net_raw` に対してのみ行う。

### 6.2 2つの帳簿は乖離する。方向は履歴依存で、逆転もする

`src/ojama_accounting.py:216-221` が明記するとおり、
**uncapped が capped より小さくなることがある。**
cap 後に相殺すると架空の余剰が相手へ送られるため、履歴によって大小が入れ替わる。

### 6.3 【2026-08-25 追記、Fix【1】】観測経路そのものを uncapped 側から再構成する

§6.1〜6.2 は「台帳が持つ量」の話だったが、**観測経路 (`SettlementObservation`
の作り方) 自体も capped 側 (`total_offset_by_pX`/`total_dropped_to_pX`) を
使っていた**ことが Gate 3 実データ検証 (2026-08-25) で判明した。
真の生成が cap (216) を超えると、超過分が相殺にも着弾にも一切現れない
(zenchi 実測: 5 回の cap 発火、超過合計 4,743 = 総生成の 52%)。

**修正: `pending_pX_uncapped` (§6.1 のレベル値) の前フレームとの差分**
から相殺・着弾・ワイプの 3 種類を判別的に再構成する
(`classify_pending_uncapped_delta`、`src/exchange_episode_tracker.py`)。

| 変化 | 判別 |
|---|---|
| 増加 | 相手の生成 (決済事象ではない、無視) |
| 減少 + 自分の連鎖確定と同時 | CANCEL |
| 減少 + ツモ設置直後 かつ `OJAMA_MAX_DROP_PER_TURN`(30) 以下 | LAND |
| 減少 + 上記いずれでもなく 0 になった | WIPE (§2.5.1 参照) |
| それ以外の減少 | `unclassified_pending_drop` (黙って捨てず必ず計上する) |

したがって「**画面の予告は 216 なのに、判定上の残量は 100**」という、
**表示のほうが脅威を大きく見せる**場面が出る。ゲーム画面自体の予告表示とも食い違い得る。

これは仕様であって不具合ではない。ただし**黙って乖離させない**。
Gate 3 の timeline dump に `net_raw` / `net_display` / **乖離量**の 3 列を必須とし、
viz 目視レビューで乖離場面を帰属できるようにする (§13)。

---

## 7. データ構造と API

### 7.1 `ExchangeEvent` (値オブジェクト、frozen)

```python
class EventKind(Enum):
    FIRE         = auto()   # 連鎖の開始 (chain_id を開く)
    STEP         = auto()   # 掛け算式 1 段の確定 (同一 chain_id へ量を積む)
    FINALIZE     = auto()   # 確定スコアによる置換 (§4.1)
    CANCEL       = auto()   # 相殺で消滅
    LAND         = auto()   # 盤面へ着弾
    TSUMO_PLACED = auto()   # 受け側がツモを 1 手置いた (chain_id を持たない)


@dataclass(frozen=True)
class ExchangeEvent:
    """交換における観測事実 1 件を表す不変値。"""

    kind: EventKind
    side: Side                 # P1 / P2 (FIRE/STEP は発火した側、LAND は受けた側)
    t_sec: float               # 観測時刻 (発火・完走時刻順に処理する基準)
    amount: float              # おじゃま個数換算 (cap 前)。TSUMO_PLACED は 0.0
    chain_id: int | None       # 同一連鎖の断片を統合する主キー。TSUMO_PLACED は None
    chain_count: int           # 掛け算式の出現回数 (段の通番)。無関係な kind は 0
    score_delta: int           # 確定スコア差分 (FINALIZE 以外は 0)
    source: str                # "formula" / "score_ocr" / "landing" / ...
```

**責務**: 事実の記録のみ。計算も判断もしない。

> **【訂正 — 初版からの変更】**
> - 初版は §5.3 で「既存 chain_id へ段を追加 (生成量 += 左×右)」と書いたが、
>   `ExchangeEvent` が frozen なので `amount` を更新できず矛盾していた。
>   **`STEP` 種別を追加**して解決する。段の追加は新しい `STEP` イベントの push であり、
>   台帳側が `chain_id` ごとに合算する。
> - 初版は `chain_id: int` を必須にしていたが、`TSUMO_PLACED` は連鎖に属さない。
>   **`int | None`** に改める。
> - 初版は `state` フィールドを `ExchangeEvent` に持たせていたが、frozen 値に
>   可変の状態を持たせるのは矛盾する。**状態は台帳側が `chain_id` 単位で持つ**
>   (§4 の状態遷移は chain_id の状態であって、個々のイベントの状態ではない)。
> - I4 の重複キーはこの定義のもとで `(kind, side, chain_id, t_sec)` とする。

### 7.2 `ExchangeEpisode`

```python
class ExchangeEpisode:
    """一つの撃ち合い。開始・参加・終了の判定と、自分に属するイベントの保持。"""

    episode_id: int
    opened_at_sec: float
    closed_at_sec: float | None
    status: EpisodeStatus      # OPEN / CLOSED / CLOSED_FORCED
    events: tuple[ExchangeEvent, ...]

    def accepts(self, ev: ExchangeEvent, ctx: PhaseContext) -> bool:
        """参加条件 (P1/P2/P3) を判定する。副作用なし。"""

    def net_raw(self) -> float:
        """1P 視点の符号付き純残量 (cap 前)。events から毎回再計算する純関数。"""

    def unreconciled(self) -> float:
        """未照合量。CLOSED 時に 0 でなければ保存則違反。"""

    def provisional_residual(self) -> float:
        """確定していない暫定量の残り。CLOSED 時は 0 でなければならない。"""
```

**責務**: 「どのイベントが同じ撃ち合いに属するか」の判定と、そのイベント集合の保持。
純残量は**毎回イベント列から再計算**する (状態を持たない)。

### 7.3 `ExchangeLedger`

```python
class ExchangeLedger:
    """イベント列を受けて episode を編成し、符号付き純残量を返す純粋な会計コア。"""

    def push(self, ev: ExchangeEvent, ctx: PhaseContext) -> None:
        """イベントを 1 件受け付ける。t_sec 昇順で呼ぶこと。"""

    def finalize(self, chain_id: int, confirmed_amount: float) -> None:
        """暫定生成量を確定値で『置換』する。加算しない。冪等。"""

    def current_episode(self) -> ExchangeEpisode | None:
        """今 OPEN な episode。無ければ None。"""

    def net_raw(self) -> float:
        """OPEN episode の 1P 視点純残量 (cap 前)。"""

    def net_display(self, cap: float = 216.0) -> float:
        """表示用に cap を適用した純残量。"""

    def is_unresolved(self) -> bool:
        """撃ち合いが未解決か。"""

    def allows_hard_override(self, ctx: PhysicalContext) -> bool:
        """±100 の完全上書きを許してよいか (§7.5 の決定不変性)。"""

    def snapshot(self) -> LedgerSnapshot:
        """timeline dump 用の観測値一式 (episode_id, chain_id, 暫定/確定, phase 等)。"""
```

#### 7.3.1 【2026-08-25 追記、Fix【4】】終了判定 (E2) は episode 単位に限定する

`_should_close` (E2、§2.4) が呼ぶ `_all_settled`/`_provisional_residual` は
**その episode に実際に参加した chain (`_chains_for_episode(ep)`) だけ**
を見る。`_summarize_episode`/`_force_close` は元から `_chains_for_episode`
限定だったが、この 2 つだけ台帳全体 `self._chains` を無条件スキャンして
おり流儀が違っていた (2026-08-25 是正)。

**根拠**: episode が閉じるのは「その episode の撃ち合いが決着したとき」
であり、台帳のどこかに無関係な未決着連鎖が残っていることとは無関係。
修正前の設計は「1 件の未確定が全体を道連れにする」構造で、
`project_stable_freeze_deadlock_2026-08-24` (凍結盤面が窒息判定を握り
つぶすデッドロック) と構造的に同型だった。

**既知の限界 (2026-08-25 実装時に判明)**: `_fire_events_of_open_chains`
(lazy open 時に outstanding>0 の全 chain を合成 FIRE として episode へ
無条件で引き継ぐ) と `Episode.touch` (OPEN な episode がある間に来た
全イベントを side/chain_id を問わず無条件で touch する) の設計により、
outstanding>0 の chain は通常の push 経路では必ずどこかの episode の
events に巻き込まれてしまう。したがって本修正単体の実効果は、
Fix【2】(`retire_side_chains` による明示的な chain 除去) と組み合わせて
初めて十分に発揮される可能性が高い。この限界は報告済みであり、
`_fire_events_of_open_chains`/`Episode.touch` 自体の変更は本タスクの
範囲外とした (コーディネーターの判断待ち)。

### 7.4 【必須】決定不変性による早期解除 — 致死を弱めないための仕組み

**「未解決の間は ±100 禁止」だけでは PLAN Gate 4 の
「真の致死局面を弱めた試合 0」を落とす。**
60秒級の OPEN episode 中は本物の致死も断定できなくなるためである。
PLAN Gate 3 も「実死亡、着弾確定、反撃不能が物理的に確定した場合だけ早期解除を許す」と
明記しており、初版の仕様書はこれを取りこぼしていた。

一方、早期解除を応手確率 (MC 推定) で作ってはいけない。
W15 (`docs/KNOWN_WEAKNESSES.md:422-`) のとおり、実際に成功した応手を
25〜40% としか見積もれず、誤断定が戻る。

**採用する規則 — 決定不変性 (decision invariance):**

> episode 内の**すべての未確定量を、受け側にもっとも有利な側へ倒して**解決しても、
> なお受け側が物理的に死ぬ場合にかぎり、hard override を許す。

判定材料は決定的な物理量のみとし、確率推定を使わない。

- `net_raw` の**下限** (暫定量を受け側有利に倒した値)
- 受け側の盤面の空き (`board_room`)
- 残り設置回数 × `OJAMA_MAX_DROP_PER_TURN` (= 30、`src/scoring.py:319`) の drain
- 受け側に進行中の連鎖があるか (あれば、その完走後の火力を差し引く)

この規則なら、
「不確実性がどう転んでも死ぬなら断定する」= 致死を弱めない、
「どちらに転ぶか分からないうちは断定しない」= ±100 張り付きを減らす、
の 2 つが**設計から構造的に両立**する。Gate 4 の 2 条件が導出される。

したがって不変条件 I8 は
「`is_unresolved()` が True **かつ** `allows_hard_override()` が False の間、
hard override を禁止する」に改訂する。

**責務**: 純粋な会計。**I/O なし、グローバル状態なし、乱数なし、時計を読まない**
(時刻は必ず引数で受ける)。同じイベント列を与えれば必ず同じ結果になる。

### 7.5 責務の境界 (何を**しない**か)

- 認識しない。掛け算式の読み取りは呼び出し側の責務。
- 表示しない。色や文言は持たない。
- 勝率を出さない。純残量を出すだけ。
- 既存クラスを書き換えない。`ResolvedExchangeTracker` /
  `OjamaAccountingTracker` へは Gate 3 で **wrapper 経由**で接続する。

---

## 8. 不変条件 (テストで固定するもの)

| # | 不変条件 | 破れたときの症状 |
|---|---|---|
| I1 | **保存則**: episode CLOSED 時、`生成 = 相殺 + 着弾 + unreconciled` | 火力が消える / 湧く |
| I2 | **順序不変**: **同一時刻**の同着イベントを任意順で入れても `net_raw()` が一致 | フレーム順で結果が変わる |
| I3 | **暫定値の置換**: 同一 `chain_id` に `finalize` を何度呼んでも `net_raw()` は同じ | 二重計上 (517+517) |
| I4 | **重複排除**: 同一 `(chain_id, kind, t_sec)` を 2 回 push しても 1 回分 | 断片ぶんの水増し |
| I5 | **断片統合**: 終わりの絶対律が来るまで、formula 再トリガーは同じ `chain_id` | 火力が10分の1 |
| I6 | **cap 非汚染**: `net_raw()` は cap を通さない。`net_display()` だけが cap を通す | 架空の攻撃が生まれる |
| I7 | **episode 終了**: `CLOSED` 時に `provisional_residual() == 0` かつ `unreconciled() == 0`。**`CLOSED_FORCED` は除外** (§2.5 が unreconciled ≠ 0 を明示的に許す) | 残量が次の撃ち合いへ漏れる |
| I8 | **未解決ゲート**: `is_unresolved()` が True **かつ** `allows_hard_override()` が False の間、hard override を許可しない (§7.5) | 決着前に ±100 を断定 / 逆に真の致死を弱める |
| I9 | **符号の一貫性**: 1P/2P を入れ替えたイベント列を与えると `net_raw()` の符号だけが反転 | 側の取り違え |
| I10 | **時刻の非減少**: `t_sec` が**減少**する push は例外。**同一時刻は許可**する (I2 と両立させるため)。黙って並べ替えない | OCR 遅延で順序が壊れる |
| I11 | **試合境界の隔離**: episode は試合を跨がない。境界で `CLOSED_FORCED` + 全量 `unreconciled` 計上。境界を跨いだ `finalize` (スコアリセット由来の負差分) は拒否 | 前の試合の残量が次に漏れる |
| I12 | **量の健全性**: `amount ≥ 0`、chain_id ごとに `Σ CANCEL ≤ 生成`、`Σ LAND ≤ 生成 − 相殺`、**`LAND` 1 件 ≤ 30** (`OJAMA_MAX_DROP_PER_TURN`) | 着弾の一括計上バグ |
| I13 | **全消し耐性**: 全消しボーナス (+2100点 = おじゃま30個相当) を含む確定で置換しても I1 が成立する | 置換規約でなければ必ず破れる |
| I14 | **換算の一貫性**: 暫定と確定が同一の `score_to_ojama` 規約 (マージンタイム・leftover) を通る (§4.1.2) | 長試合で暫定と確定が系統的に乖離 |
| I15 | **episode 不在時**: OPEN な episode が無いとき `is_unresolved() == False` | 整地だけで override が永久禁止 |
| I16 | **FINALIZE 供給源の限定 (2026-08-25 追加)**: `source == FINALIZE_SOURCE_SCORE_OCR_DIFF` (score OCR 確定差分) 以外の FINALIZE は既定で拒否する (`allow_simulate_fallback=True` のときだけ許可)。例外は投げず `finalize_rejected_count`/`finalize_rejected_amount` に記録する | `ChainSimulator` 由来の推定 (`mechanism='baseline'`) が確定値として紛れ込み、暫定 38 個に対し推定 745 個という壊滅的な値が会計に入る |

### 8.1 LANDED の観測契約

**着弾量の一次情報は `on_tsumo_settled` の drain イベント
(`src/ojama_accounting.py:531-543`) とする。盤面のおじゃまセル増分は検算にのみ使う。**

理由: 受け側が**連鎖でおじゃまを消しながら受ける**と、セル増分は過小になる
(30個降って12個消えたら +18 にしか見えない)。これを一次にすると I1 が
偽の `unreconciled` を報告する。

同じ理由で、**盤面のおじゃま数を保存則 I1 の項に使わない**。
台帳が扱うのは「空中にある量 (pending)」であって、盤面上の量ではない。
盤面上のおじゃまが隣接消去で消えることは I1 に影響しない。

### 8.2 「受け側が1手置いた」の観測器

`TSUMO_PLACED` の信号源は **`OjamaAccountingTracker` の drain トリガーに一致させる**。

**ネクストのスライド検知を直接使ってはいけない。**
連鎖中に 1.37 秒おきに誤検知した前科があり
(`project_slide_false_positive_root_cause_2026-08-22`)、
誤検知する信号を episode の CLOSE トリガに使えば同じ穴に落ちる。

---

## 9. 代表場面のイベント列 (期待値)

> **記載方針**: 数値は `data/verify/pm100_fix_2026-08-24/dumps_{off,on}/` の
> 実 dump から抽出したものと、実画面33枚 (`logs/zenchi_g2_frames_2026-08-24/`)
> で確認済みの事実のみを使う。推測値には「推測」と明記する。

記法: `net` は §3 の 1P 視点符号付き純残量 (cap 前)。
`P` = PROVISIONAL、`F` = FINALIZED、`L` = LANDED、`C` = CANCELED。

### 9.1 正例 — seg01 game2 (t=141.1〜223.4、本物の4往復)

実画面33枚 (`logs/zenchi_g2_frames_2026-08-24/`) で確認済みの事実:
最終 72,256 vs 68,536 で **1P の勝ち** (`t0223.00.png` で 2P 盤面がおじゃまで埋没、`1★ WIN 0★`)。

**数値はすべて計装ログの実測値**
(`logs/_diag_zenchi_seg01_pm100_trace_2026-08-24.log` の `[cc_inputs]` 行、
`pend` / `gen` / `kpend` / `kroom` の変化点のみを抽出)。

| # | t (秒) | 実測 `pend` | 実測 `gen` | 実測 `kpend` | 出来事 | 期待 event | 期待 net (1P視点) |
|---|---|---|---|---|---|---|---|
| 1 | **176.30** | (0, 0) | **(525, 0)** | (0, 495) | 1P 発火。生成 **525個** | FIRE(chain_id=A) → STEP ×n 合計 525 | **+525** |
| 2 | 178.00 | (79, 0) | (525, 0) | (0, 416) | 2P からの 79個が 1P の pending に載る | (相手側の別 chain) | +525 − 79 = +446 |
| 3 | **186.67** | (79, 0) | **(0, 720)** | **(769, 0)** | 2P が撃ち返し **720個**。同時に **`gen1` が 525 → 0 にリセット** | FIRE(chain_id=B) → STEP ×n 合計 720 | +446 − 720 = **−274** |
| 4 | — | — | — | — | **現行の誤り①**: `kpend1=769` は `720 + 79 − 30`。**1P が送った 525 が相殺計算からまるごと消えている** | (I1 保存則により不可能) | (本会計では起きない) |
| 5 | **197.53** | (0, **216**) | (0, 720) | (474, 0) | 会計 finalize がようやく追いつく (**1P 完走から 11.5秒遅れ**)。しかも **cap 216 で丸められている** | FINALIZE(A) → **置換** | −274 (置換で差分のみ) |
| 6 | — | — | — | — | **現行の誤り②**: cap 済みの 216 と 720 を引き算するので**架空の攻撃が生まれる** | (§6 の `net_raw` は cap 前) | (本会計では起きない) |
| 7 | **201.07** | (0, 186) | **(442, 0)** | (0, 598) | 1P が再度撃ち返し。生成 **442個** | FIRE(chain_id=C) → STEP ×n | 符号が 1P 側へ戻る |
| 8 | 201.07〜204.63 | (0, 186)→(216, 0) | (442, 0) | (0, 598)→(0, 196) | 2P の pending が 30個ずつ 7 回に分けて drain (`OJAMA_MAX_DROP_PER_TURN=30`) | LAND ×7 | 段階的に解消 |
| 9 | **211.43** | (216, 0) | **(101, 0)** | **(85, 0)** | **現行の誤り③**: 断片化で `gen1: 442 → 101`。**画面では何も起きていない** (`adv_raw` 不変、`state1=CHAIN` 継続) | (I5 により同じ chain_id へ統合され発生しない) | 変化なし |
| 10 | 213.17 | (216, 0) | (0, 85) | (271, 0) | 2P が新しい連鎖を開始 | FIRE(chain_id=D) | — |
| 11 | 222.00 | — | — | — | **2P 実死亡** (`is_dead2=True`) | episode CLOSED | net > 0 のまま閉じる |

> **memory との差異**: `project_pm100_display_flip_2026-08-24` は 1P の火力を
> 「9連鎖 `50×194` ≈ 517個」と記録していたが、計装ログの実測 `gen1` は **525.0** だった。
> 掛け算式からの手計算と、実装が使っている生成量推定の差である。
> **本仕様書は計装の実測値 525 を採る。**

#### 9.1.1 実 dump によるフレーム単位の突合 (2026-08-24 実施)

入力: `data/verify/pm100_fix_2026-08-24/dumps_{off,on}/seg01_0_893.7.npz`、`game_idx==2`。

**誤反転は t=211.433 の 1 フレームで起きている。**

| t | state1 | state2 | kpending (OFF) | kroom (OFF) | kill | adv_ema (OFF) | adv_raw |
|---|---|---|---|---|---|---|---|
| 210.733 | CHAIN | STABLE | (0.0, 196.0) | (65, 15) | g=1.00→+100 | **+100.0** | +26.6 |
| **211.433** | **CHAIN** | STABLE | **(85.0, 0.0)** | (39, 38) | **g=1.00→−100** | +50.0 | **+18.4** |
| 212.533 | STABLE | OJAMA_FALL | (216.0, 0.0) | (69, 38) | g=1.00→−100 | −52.5 | +18.4 |
| 213.067 | STABLE | STABLE | (216.0, 0.0) | (68, 39) | g=1.00→−100 | **−99.6** | +54.6 |
| 222.000 | OJAMA_FALL | STABLE | (0.0, 216.0) | (37, 31) | g=1.00→+100 | +93.7 | −39.0 |

確定した事実:

- **OFF は t=211.400 (+100.00) → t=213.067 (−99.52) で 1.667 秒で完全反転**、
  以後 t=221.633 まで約 8.2 秒 −100 に張り付く。ON は同時刻 **+32.10** を維持。
- **生モデル `adv_raw` はこの間 一度も符号を割っていない** (+18.36 → +54.63 → …、
  最も負でも −69.35)。**表示だけが ±100 に張り付いており、モデルの言うことと乖離している。**
- t=211.433 で **盤面は何も変わっていない** (`adv_raw` 不変、`state1=CHAIN` 継続)。
  変わったのは相殺の結論だけで、`kpending` が `(0, 196)` → `(85, 0)` へ**巻き戻った**。
- 原因は `ChainGenerationAccumulator` の非累積 (置換) 動作
  (`scripts/visualize_advantage_overlay.py:370-391`、`accumulate=False` で
  `self._accum_gen[key] = gen`)。`trigger_sec` が切り替わると累積を捨てて
  最新断片の生成量だけを採るため、同じ `base_pending1=216` を相殺しきれなくなる。
- `mag=2.18 ≥ KILL_RATIO_FULL(1.5)` で **g=1.00 = 完全上書き**。
  EMA (`α=0.25`、τ=0.116秒) は段差入力を実質素通しする。

**断片化の実例**: 1P の単一の物理連鎖は `state1` が CHAIN / GRAVITY_SETTLE のまま
**t=201.067〜212.533 の 11.47 秒間**続いていたが、その間に kill_override への
是正入力が **8 回**書き換わっている (t=201.067 / 201.733 / 202.533 / 203.267 /
204.033 / 204.633 / 206.400 / 211.433)。うち t=206.300〜206.567 では
`state1` 自体が **CHAIN → STABLE → CHAIN と 267 ミリ秒だけ復帰**しており、
11 秒続く連鎖が一瞬「完走」したように見えるフラグメント境界が可視化されている。

> **§5.2 で「STABLE 復帰で chain_id を切る」を削除した直接の根拠がこれである。**

**ON が緩和されるのは根治ではない**: t=211.433 では ON も是正入力が崩れているが
(`kpending=(216,65)`、`mag=1.21` で `g=0.67` の部分ブレンド)、
t=212.533 では ON 側の `PostChainUnregisteredSentTracker` が両側に量を足した結果
`kpending=(347,190)` となり **`mag=0.03` でたまたま無発火**になっている。
ON も t=215.5 前後で結局 −100 に到達する。**タイミング依存の偶然の緩和である。**

**この episode に期待する性質**:
- 全区間で **1つの episode**。#1 で OPEN、#9 で CLOSED。
- #3 の発火は参加条件 **P1 (未解決残量 +517 がある)** により同じ episode に参加する。
  時間窓 (0.963秒) では #1 と #3 の 9.2秒の隔たりを絶対に繋げられない。
- #4 と #8 の 2つの誤反転が、それぞれ I3 (置換) と I5 (断片統合) で構造的に消える。
- #1〜#8 の全区間で `is_unresolved() == True` かつ `allows_hard_override() == False`
  → **±100 の完全上書きは禁止**。向きは出してよいが、99%/1% と断定してはいけない場面である。
- t=222.000 で `is_dead2=True` (2P 実死亡) が立つ。ここで初めて §7.4 の決定不変性により
  hard override が許される。

> **突合の出典について**: npz dump (`dumps_{off,on}/seg01_0_893.7.npz`) には
> `chain_count` も `ChainEvent.trigger_sec` も**列が無い**ため、
> dump 単独では断片化を代理指標 (`kpending` / `kroom` の変化) でしか観測できない。
> §9.1 の `gen` / `pend` / `kpend` の実測値は
> **計装ログ `logs/_diag_zenchi_seg01_pm100_trace_2026-08-24.log` の `[cc_inputs]` 行**
> から取った。こちらには生成量が直接出ている。
>
> ただしこのログは診断用の一回限りの計装であり、**本番の timeline dump には
> 同じ情報が出ない**。Gate 4 で全 episode の保存則を機械検査するには
> **生成量を dump の正式な列にする必要がある** (§13 の D7)。
>
> また §9.1 の表は amount を発火時刻に一括で置いているが、実際には掛け算式の暫定は
> 段ごとに漸増する。**Gate 2 のテストでは「完走時点の値」を使うと明示する。**

### 9.2 反例1 — 片側だけが連鎖する (相手は無反応)

現行 `ResolvedExchangeTracker` が `ev1 is not None and ev2 is not None` で
**一度も起動しない**場面。本会計は片側でも episode を開く。

| # | t | side | 出来事 | 期待 event | 期待 net |
|---|---|---|---|---|---|
| 1 | 10.0 | 1P | 5連鎖発火、生成 60個 | FIRE(A, P, 60) | +60 |
| 2 | 16.0 | 1P | 完走。2P は一度も発火しない | — | +60 |
| 3 | 16.5 | 2P | 2P がツモを1手置く | TSUMO_PLACED(2P) | +60 |
| 4 | 16.6 | 2P | 着弾3条件が揃い、30個 (1ターン上限) が降る | LAND(A, 30) | +30 |
| 5 | 17.3 | 2P | 次の1手を置く | TSUMO_PLACED(2P) | +30 |
| 6 | 17.4 | 2P | 残り30個が降る | LAND(A, 30) | **0** → CLOSED |

**期待する性質**: I1 (保存則) が `生成60 = 相殺0 + 着弾60 + 未照合0` で成立。
episode は #6 で閉じる。#4 の時点で閉じてはいけない (残量30個が次の episode に漏れる)。

### 9.3 反例2 — 相殺しきれず着弾する (差分だけが降る)

| # | t | side | 出来事 | 期待 event | 期待 net |
|---|---|---|---|---|---|
| 1 | 30.0 | 1P | 生成 100個 | FIRE(A, P, 100) | +100 |
| 2 | 33.0 | 2P | 生成 40個 で応射 | FIRE(B, P, 40) | +60 |
| 3 | 35.0 | — | 相殺処理 | CANCEL(A, 40), CANCEL(B, 40) | +60 |
| 4 | 36.0 | 2P | ツモ着地 | TSUMO_PLACED(2P) | +60 |
| 5 | 36.1 | 2P | 30個着弾 (1ターン上限) | LAND(A, 30) | +30 |
| 6 | 37.0 | 2P | ツモ着地 → 残り30個着弾 | LAND(A, 30) | **0** → CLOSED |

**期待する性質**: 2P が生成した 40個は **1個も 1P へ届かない**。
相殺で消えた分は `CANCELED` であり `LANDED` ではない (`reference_ojama_forecast_landing_spec`:
「相殺により予告お邪魔が無くなれば降りません」)。
I1 は `生成140 = 相殺80 + 着弾60 + 未照合0`。

### 9.4 反例3 — 整地 (送りおじゃま ≤4個) は episode を開かない

| # | t | side | 出来事 | 期待 event | 期待 net |
|---|---|---|---|---|---|
| 1 | 50.0 | 1P | 発火。掛け算式の段が進むが、暫定生成量は 3 個止まり | STEP(A) ×n | — |
| 2 | 〜完走 | — | 暫定生成量が一度も 4 個を超えない = **整地** | episode を開かない | — |

**期待する性質**: `ExchangeLedger.current_episode()` が `None` のまま。
`is_unresolved()` は False (I15)。整地で撃ち合い判定を起動すると、序盤の掘り作業が
すべて「未解決の撃ち合い」になり hard override が永久に禁止される。

#### 9.4.1 【訂正】episode の開始は「4個を超えた瞬間の lazy open」

> **初版の誤り**: 初版は t=50.0 の発火と同時刻に「整地だから開かない」と書いていた。
> **これは因果律に反する。**
>
> `_classify_exchange` (`scripts/measure_exchange_dynamics.py:372-402`) の入力は
> 発火前後のグリッドと送りおじゃま**総量**であり、**すべて連鎖終了後にしか確定しない事後情報**。
> 発火の瞬間には総送付量は未知である (連鎖はこれから成長しうる)。

**採用する規則:**

> episode は「**暫定生成量が `SEICHI_OJAMA_MAX_COUNT` (4個) を超えた瞬間**」に開く
> (lazy open)。

暫定生成量は段が進むごとに単調増加するので因果的であり、
発火の瞬間に未来を知る必要がない。
催促 / 本線 / 整地の**ラベル付け**は従来どおり事後分類のままでよい
(ラベルは episode の開始判定に使わない)。

#### 9.4.2 【要 user 確認】整地の定義が実装と仕様で食い違っている

| | 定義 |
|---|---|
| **実装** (`scripts/measure_exchange_dynamics.py:396-401`) | `ojama_sent_count <= SEICHI_OJAMA_MAX_COUNT (4.0)` なら**消費比率に関係なく**整地 |
| **user 伝授の記録** (`reference_saisoku_exchange_model_2026-07-22`) | 「**<60%消費 かつ** 送りおじゃま ≤4個」 |

**本線 (≥60%消費) で 4 個以下しか送らないケースの扱いが逆になる。**
閾値の値そのもの (60% = `HONSEN_RATIO_THRESHOLD`、4個 = `SEICHI_OJAMA_MAX_COUNT`) と
境界の向き (≤4) は一致している。

**本仕様書は暫定的に実装側 (≤4個なら消費比不問) を採る。**
理由: episode の開始判定に使うのは「送付量」だけであり、
消費比率は事後にしか分からないため lazy open と両立しない。

ただし**これは user 伝授の原典確認を要する**。確認結果によっては
ラベル付けのほうを直す (episode 開始規則は変えなくてよい)。

また実装は `before_n <= 0` のとき「不明」を返す (`同:391-392`)。
**「不明」は episode を開かない**ものとし、件数をカウンタに出す。

> 閾値 4 個は「おじゃま3個は無害」という独立の user 伝授
> (`reference_ojama_damage_nonlinear_2026-07-29`) とも整合する。
> **シーンから逆算した値ではない。**

### 9.5 反例4 — 断片化した長い連鎖 (15連鎖が5連鎖×3に見える)

実測された最悪ケース (`project_chain_event_fragmentation_accumulator_2026-08-22`)。

| # | t | 観測 | 現行の挙動 | 期待 event |
|---|---|---|---|---|
| 1 | 6704.733 | 掛け算式 1段目 | ChainEvent(cc=5, accum=336) | FIRE(A, P) 段1 |
| 2 | 6706.100 | 2段目 (1.367秒後) | **新しい** ChainEvent に置換 (accum=420) | 同じ A に段2 を追加 |
| 3 | 6707.467 | 3段目 | 置換 (accum=504) | A に段3 |
| … | … | … | … | … |
| 7 | 6712.967 | 7段目 | 置換 (accum=**840**) | A に段7 |
| 8 | 6717.5 | 単発読み | **84個しか見えない** (最後の断片のみ) | A の総和 = **840個** |
| 9 | 6718.3 | 1P のネクストが動く | — | **絶対律で A を CLOSE** |

**期待する性質**: #1〜#8 のあいだ `chain_id` は **A のまま**。
ホールド期限 (`0.3秒 × cc`) では切らない (§5.2)。
#9 の絶対律で初めて chain_id が閉じる。
火力は 84 ではなく **840** (10倍差、これが ±100 反転の直接原因だった)。

### 9.6 反例5 — 確定スコアが暫定より小さい (掛け算式の読み過ぎ)

置換規約 (§4.1.1) の 3 分岐を固定する反例。**決着済み**。

**(a) 小さい下げ = 落下ボーナス等で説明できる差 → 置換する**

| # | t | 出来事 | 期待 event | net |
|---|---|---|---|---|
| 1 | 70.0 | 掛け算式から暫定 500個 と読んだ | STEP(A) ×n、合計 500 | +500 |
| 2 | 78.0 | 確定スコア差分から **497個** と確定 (差 3 は落下ボーナス相当で許容帯内) | FINALIZE(A, 497) → 置換 | **+497** |

`net` は 997 (加算) でも 500 (据え置き) でもなく **497**。

> **【訂正 2026-08-24】** 初版はこの例を「500 → 420 (差 80)」としていたが誤り。
> 80 はおじゃま個数で、点数に直すと 5,600 点。落下ボーナス (最大 約 250 点 =
> おじゃま約 3.6 個) では説明できない差であり、**許容帯どころか保留すべき乖離**だった。
> 単位を取り違えた例示。実装のテストで検出した。
`finalize` を再度 420 で呼んでも 420 のまま (I3、冪等)。

**(b) 大きい下げ + 暫定側が物理検算を通過 → 置換せず両値を保持**

| # | t | 出来事 | 期待 event | net |
|---|---|---|---|---|
| 1 | 70.0 | 暫定 500個。全段の左辺が 10 の倍数で検算通過 | STEP(A) ×n、合計 500 | +500 |
| 2 | 78.0 | 確定 **42個** (桁誤読の疑い、W2) | 置換せず、差 458 を `unreconciled` へ計上 + カウンタ | **+500** |

**(c) 上げ = 無条件で置換 (全消しボーナスを含む正常系)**

| # | t | 出来事 | 期待 event | net |
|---|---|---|---|---|
| 1 | 70.0 | 暫定 500個 (掛け算式は純連鎖得点のみ) | STEP(A) ×n、合計 500 | +500 |
| 2 | 78.0 | 確定 **530個** (全消しボーナス 2100点 ≈ 30個ぶんが乗る) | FINALIZE(A, 530) → 置換 | **+530** |

(c) が I13 (全消し耐性) の実体である。`max` 方式でも通るが、
**置換方式でなければ (a) が通らない**。

---

## 10. 既存資産との関係 — 何を再利用し、何を置き換えるか

ゼロから会計を書き直さない。既存部品を束ねる**統合レイヤ**として設計する
(`project_exchange_meter_design_b_2026-07-28` の前例に従う)。

### 再利用する (そのまま使う)

| 既存資産 | 場所 | 本会計での役割 |
|---|---|---|
| `FormulaStep` / `FormulaStepAccumulator` | `src/score_ocr.py:947-1017` | **段の検出そのもの**。`step_count()`=連鎖数、`total_power()`=火力。chain_id の供給元 |
| `parse_formula_cells` | `src/score_ocr.py:201-262` | 掛け算式の stateless パース (物理制約チェック込み) |
| `pending_*_uncapped` 並行帳簿 | `src/ojama_accounting.py:222-241, 741-748` | `net_raw` の供給元 (§6) |
| `cancel_own_pending_then_send_surplus` | `src/ojama_accounting.py:144-161` | 相殺の純関数。式を再発明しない |
| `resolve_mutual_exchange` | `src/exchange_virtual_board.py:276` | 完走シミュレーションによる決着 (両側発火時のみ) |
| `_classify_exchange` / `_build_sequences` / `_defrag_events` | `scripts/measure_exchange_dynamics.py:372, 693-790` | 催促/本線/整地の分類ロジックと `seq_id` 方式。**オフライン専用だが episode の概念に最も近い既存実装**。判定式を移植する |

### 置き換える (Gate 3、旧経路はフラグOFFで保持)

| 既存の対処 | 場所 | 何が足りないか | 本会計での扱い |
|---|---|---|---|
| `ResolvedExchangeTracker` の起動条件 | `scripts/visualize_advantage_overlay.py:1932-1934` (`ev1 is not None and ev2 is not None`) | **両側同時発火を要求**するため片側連鎖でクラス全体が起動しない | episode の参加条件 (§2.3) へ置換 |
| `_maybe_redecide` の `max(pred, obs)` 一方向合成 | 同上 | **1回きり latch**。段階確定 (0→1260→4020) の途中値で固定される | `finalize()` による**置換**へ (§4.1)。冪等なので何度来ても正しい |
| `PostChainUnregisteredSentTracker` (A案 i-a) | `scripts/visualize_advantage_overlay.py:577-` | 未登録送付を 20秒期限で保持する**対症の外付け**。減額が二重に効くのを「常に供給不足側へ倒す」保守設計で誤魔化している | 未登録送付を `PROVISIONAL` イベントとして**一級市民**にする。期限ではなく episode 終了で決着 |
| `KillOverrideConfidenceGate` (B案) | `scripts/visualize_advantage_overlay.py:507-561` | `KILL_CONFIRM_PERSIST_SEC ≈ 0.963秒` の**固定時間**ヒステリシス。10秒級の本線応酬を表現できず、全8区間で ±100張り付き 14.0%→18.3% と悪化 | 時間ではなく**状態**でゲートする。`is_unresolved()` が True の間だけ hard override を禁止 (I8) |
| `ChainGenerationAccumulator` | `scripts/visualize_advantage_overlay.py:314-405` | `trigger_sec` 変化で置換 (既定) または加算。`chain_id` が無いため両者を正しく使い分けられない。seg01 game2 の誤反転の直接原因 (§9.1.1) | `chain_id` 主キーの置換へ。**排他フラグ**で二重加算を防ぐ (R4) |

### 当面は残す (置き換えを急がない)

| 既存資産 | 場所 | 理由 |
|---|---|---|
| `CHAIN_COALESCE_WINDOW_SEC = 2.5秒` の明滅吸収 | `src/ojama_accounting.py:123-130, 406-481` | **`chain_id` はまだ STABLE 明滅に対して coalesce 窓より強いと実証されていない。** §5.2 で条件2を削除したことで chain_id は明滅に耐える設計になったが、それは設計上の話であって実測ではない。Gate 3 で **chain_id が実測で勝ってから**置換を検討する |

> **【訂正 — 初版からの変更】** 初版はこの行を「置き換える」表に入れていたが、
> 同時に「`_finalize_chain_end` は触らない」とも書いており、
> **同じ機構の内側と外側を別扱いする内部矛盾**になっていた
> (coalesce 窓は `_finalize_chain_end` の一部)。

### `KillOverrideConfidenceGate` が守っていた残余ノイズの担い手

B 案を置き換えるにあたり、**B が担っていて episode 会計では拾えない class** がある。

`scripts/visualize_advantage_overlay.py:507-561` の規則1 (方向反転クールダウン) は、
**発火イベントを伴わない score 誤読由来の phantom pending** にも効いていた。
この場合 episode は OPEN していないので `is_unresolved() == False` となり、
新設計では**無防備になる**。

対処: §2.3.1 の「裏付けのない残量は `unreconciled` へ隔離」がこの class の一次防御となる。
それで足りるかは Gate 4 の 5 構成比較で実測する。
**足りなければ B 案の規則1 だけを残す**選択肢を保持する (全廃を前提にしない)。

### 触らない

- `OjamaAccountingTracker` の `_finalize_chain_end` そのもの。**確定会計の単一情報源**として残し、
  本台帳はその確定値を `finalize()` の入力として受け取るだけにする。
- 学習パイプライン (`src/indicators_v2.py`, `scripts/build_labeled_win_from_npz.py`)。Gate 3 以降。

### Gate 3 で孤児にしてはいけない配線

`ChainGenerationAccumulator` を置き換えるとき、`kill_override` が必要とする
**`before_board` (完走後 room 計算の入力)** の受け渡しを切らないこと。
**本台帳は「量」しか持たない設計 (§7) なので、盤面の受け渡しは台帳の外に残る。**
Gate 3 の統合設計で明示的に配線先を書く。

---

## 11. Gate 2 で作るもの / 作らないもの

**作る**
- 新規モジュール 1 本 (純粋な会計コア)。既存ファイルへの変更は 0 行。
- **`chain_id` 付与器 (純関数)**。段 + 絶対律信号から `chain_id` を決める。
  これを Gate 2 に含めないと I5 (断片統合) がテストできない (§5.2.3)。
- 上表 I1〜I15 の不変条件テスト (実装より先に書く)。
- 既定 OFF の optional フラグ 1 個。ON にしないと一切の経路が変わらないこと。

**作らない (Gate 3 送り)**
- `recognition_pipeline` / `ojama_accounting` / 表示経路への配線。
- `ResolvedExchangeTracker` の置き換え。
- timeline dump への列追加 (§13 で定義だけしておく)。
- `production_config.py` への採用登録 (**user 承認が必要**)。

**Gate 2 完了の条件**
- 既定 OFF で既存経路の出力が **bit-identical**。
- §9.1 の数値表が実 dump で確定していること (現状は §9.1.1 の注記のとおり
  生成量そのものが dump に無く未確定)。

---

## 12. 既知のリスク

| リスク | 内容 | 対処 |
|---|---|---|
| R1 | 掛け算式の読み取り自体に誤りがあれば会計も誤る | 「純粋な連鎖得点は10の倍数」で検算し、外れた件数をカウンタに出す |
| R2 | 「終わりの絶対律」の**検出器**が既知の事故源 (連鎖中に1.37秒おきに誤検知した前科) | 連鎖中ゲート付きの修正済み検出器を前提条件とする。どちらの信号で閉じたかを記録し、Gate 3 で「どちらでもないのに閉じた件数」を実測 (§5.2.2) |
| R3 | episode の参加条件が広すぎると試合全体が1 episode になる。**特に score OCR 破綻動画 (c26/c58/c69) では恒久癒着し得る** | §2.3.1 の裏付け規則で隔離。`EPISODE_MAX_SEC` と強制終了件数をカウンタに出す |
| R4 | 既存の `ChainGenerationAccumulator` と二重に加算する | Gate 3 では**どちらか一方のみ**が有効になる排他フラグにする |
| R5 | 学習列へ流すときの符号・対称化 | §3 の注記どおり不変列リストへ明示登録 |
| R6 | §2.5 の強制終了トリガー `is_dead` が W32 (`docs/KNOWN_WEAKNESSES.md:1497-`) の上流バグを継承する。誤 `CLOSED_FORCED` は「未解決ゲートの早期解除 = 誤 ±100」に直結 | W32 の根治状況を §2.5 の前提条件として扱う。解決前は強制終了件数を必ず出す |
| R7 | Q-01 (掛け算式累積器が段を落とす) が未修正のままだと、暫定生成量そのものが過小になる | **Gate 0 の Q-01 を閉じてから Gate 2 に入る** |

---

## 13. Gate 3 の timeline dump に必要な列

PLAN Gate 3 が挙げる列 (`episode_id` / `chain_id` / 暫定・確定生成量 / 純残量 /
phase / hard-override 保留理由) だけでは **Gate 4 の必須条件 2・3・4 が測れない**。
不足分を先に定義しておく。

| # | 列 | なぜ必要か (対応する Gate 4 条件) |
|---|---|---|
| D1 | episode 単位の **生成・相殺・着弾・unreconciled の累計** と `status` / `close_reason` | 「保存則を全 episode で満たす」の機械検査 |
| D2 | **finalize 乖離** (確定 − 暫定) と、下げ置換ゲートの発動有無 | 「同一 chain_id の暫定値と確定値を二重計上しない」の検査。§4.1.1 の閾値の実測 |
| D3 | **強制終了カウンタ 3 種** (`EPISODE_MAX_SEC` / `CHAIN_ID_MAX_SEC` / 10の倍数検算落ち) | §5.4・R1 が「必ずカウンタに出す」と約束している実体 |
| D4 | **絶対律の観測記録** — chain_id をどの信号で閉じたか (next 動作 / おじゃま着弾 / 強制) | R2 の実測 |
| D5 | **直近の物理イベント種別 + 時刻** | 「急変は 40 回以下で、残る全件に新しい物理イベントがある」を走査器 (`scripts/scan_judgment_anomalies.py`) で機械判定するため。**現状この条件は目視でしか確認できない** |
| D6 | `net_raw` / `net_display` / 乖離量 | §6.2 の乖離場面を viz 目視で帰属するため |
| D7 | **生成量そのもの** (現在の dump には `chain_count` も `trigger_sec` も無い) | §9.1.1 のとおり、断片化を代理指標でしか観測できていない状態を解消する |

さらに **「真の致死局面」の固定リスト** (動画 ID + 時刻の回帰資産) が要る。
Gate 4 の「真の致死局面を弱めた試合 0」は、比較対象のリストが無いと**測定不能**である。
リストの所在を Gate 3 開始までに決める。

---

## 13.5 指標監査からの追加要求 (Codex 2026-08-24 17:12 / 18:18)

### 13.5.1 なぜ交換エピソード会計が指標監査の答えでもあるのか

Codex の 51 指標監査 (`docs/CODEX_INDICATOR_AUDIT_2026-08-24.md`) の中心的な指摘は 2 つで、
**どちらもこの仕様と同じ根に繋がる**。

- 現行モデルは**終盤依存**が強い (全体 AUC 0.6357 / 序盤 0.5256 / 中盤 0.5506 / 終盤 0.7843)。
- **現行の静的指標だけでは「催促対応 → 相手本線 → 自分本線」の順序を表現できない。**

前者は「局面という文脈が入力に無い」ため、後者は「時間差の応酬を 1 つの出来事として
数えていない」ため。**後者を解くのが本仕様であり、そこで得る `episode_stage` が
前者の入力にもなる。**

監査の実測もこれを裏づけている。forecast 系 3 列は rho 0.9996 以上・95.1% が 0・
単独 AUC 約 0.508 で、**時間差交換の判断軸として機能していない**。

### 13.5.2 episode 段階 (`episode_stage`)

§2 の episode に、進行段階のラベルを持たせる。学習の入力にも使う。

| 値 | 意味 |
|---|---|
| `harass_response` | 催促に対応している最中 |
| `opponent_main_fired` | 相手の本線が発火した |
| `own_main_held` | 自分の本線を溜めている (撃てるが撃っていない) |
| `own_main_fired` | 自分の本線を撃った |
| `settling` | 相殺・着弾の処理中 |

**ラベルは事後分類ではなく、その時点で確定している観測だけで決める。**
催促 / 本線の分類は送付量 (§9.4.1 の lazy open と同じ因果的な基準) で行い、
消費比率のような連鎖終了後にしか分からない量は使わない。

### 13.5.3 episode が出す量 (Codex 命令 2)

`LedgerSnapshot` (§7.3) に次を含める。

| 量 | 定義 |
|---|---|
| 未着弾総量 | 生成済みで、まだ着弾も相殺もしていない量 (cap 前) |
| 相殺量 | この episode で打ち消し合った累計 |
| **純残量** | §3 の `net_raw` (1P 視点、cap 前) |
| **発火 ETA** | 相手の連鎖が確定するまでの予測時間 |
| **仮想着弾後の窒息余裕** | 現在の純残量を仮想着弾させた後の余裕段数 |

**発火 ETA と窒息余裕は既存資産を使う。** 新規に作らない。

- 発火 ETA: 連鎖の残り時間予測 (実測テーブル n=418) と
  `PLACEMENT_SPEED_BY_ROW_SEC` (24.6 万件実測)
- 窒息余裕: `ChainSimulator.drop_ojama` で仮想着弾させてから測る
  (Q-04 で採った方式と同じ。`src/indicators_v2.py:_ojama_damage_virtual_landing`)

### 13.5.4 補助ターゲット (Codex 命令 3)

episode 単位で次を学習の補助ターゲットにする。

- 交換後の純残量
- 交換後の窒息余裕
- 交換前後の勝率差

**主ターゲット (各時点の勝率) は変えない。** 補助ターゲットは
「その交換が有利に終わったか」を直接教える信号で、
時間差の応酬を 1 つの単位として学ばせるために使う。

### 13.5.5 未解決 episode 中の ±100 禁止 (Codex 命令 4 / 18:18 の D)

**§7.4 で既に「決定不変性による早期解除」として仕様化済み。**

> episode 内のすべての未確定量を、受け側にもっとも有利な側へ倒して解決しても、
> なお受け側が物理的に死ぬ場合にかぎり、hard override を許す。

確率推定 (W15、応手確率は実際に成功した応手を 25〜40% としか見積もれない) を
使わないので、「未解決中は断定しない」と「真の致死を弱めない」が
**構造的に両立する**。Codex の要求と整合している。

### 13.5.6 モデル比較 A/B/C/D への接続 (Codex 18:18)

本仕様が出す `episode_stage` と 13.5.3 の量は、**B (局面文脈付き単一モデル) と
D (ExchangeEpisode 専門モデル) の入力**になる。

比較は **Gate 3 の配線と学習列生成の後**に行う。順序は
`docs/agent_coordination/CLAUDE_TO_CODEX.md` の P0〜P7 を参照。

**この仕様の段階では実装しないが、出す量が後で使われることを前提に設計する。**
具体的には、`LedgerSnapshot` を「表示用」ではなく
**「学習列にそのまま落とせる粒度」**で定義しておく。

### 13.5.7 検収での注意 (Codex 18:18)

- **hard な 3 分割切替は行わない。** C は境界付近で 2 モデルを連続的に混ぜる。
- **位相境界前後 ±3 秒の勝率ジャンプを必須検収**とする。件数と最大変化幅を出す。
- **位相別 Platt と混同しない。** あれは既定 OFF の後段確率校正であって
  局面別モデルではなく、2026-08-11 実測で ECE を悪化させている。
  **局面別モデルの代用にしない。**

---

## 14. Gate 2 で先に書くテスト

実装より**先に**書く。ファイルは `tests/test_exchange_ledger.py` (新規)。

### 14.1 保存則

| テスト | 内容 |
|---|---|
| `test_i1_conservation_one_sided` | §9.2: 生成 60 = 着弾 30 + 30。#4 時点で未 CLOSE、#6 で CLOSE |
| `test_i1_conservation_with_cancel` | §9.3: 生成 140 = 相殺 80 + 着弾 60。2P の 40 は 1 個も届かない |
| `test_i13_all_clear_bonus_keeps_conservation` | 全消しボーナス (+2100点 = 30個相当) を含む確定で置換しても I1 が成立 |

### 14.2 順序不変

| テスト | 内容 |
|---|---|
| `test_i2_same_tsec_permutation_invariant` | 同一時刻イベントの**全順列**で `net_raw()` が一致 |
| `test_i10_decreasing_tsec_raises` | `t_sec` が減少する push は例外。**同一時刻は許可** |

### 14.3 暫定値の置換

| テスト | 内容 |
|---|---|
| `test_i3_finalize_idempotent` | 517 で 2 回 `finalize` → 517 のまま (1034 にならない) |
| `test_finalize_upward_replaces` | §9.6(c): 暫定 500 → 確定 530 で置換 |
| `test_finalize_small_downward_replaces` | §9.6(a): 暫定 500 → 確定 420 (許容帯内) で置換 |
| `test_finalize_large_downward_holds_and_counts` | §9.6(b): 暫定 500 → 確定 42 は置換せず `unreconciled` へ + カウンタ |
| `test_i14_provisional_uses_same_score_to_ojama` | 暫定も確定と同じマージンタイム・leftover 規約を通る |

### 14.4 重複排除・断片統合

| テスト | 内容 |
|---|---|
| `test_i4_duplicate_push_dedup` | 同一 `(kind, side, chain_id, t_sec)` を 2 回 push しても 1 回分 |
| `test_i5_chain_id_survives_hold_expiry` | §9.5: 断片 7 個 → 1 つの chain_id、火力 **840** (84 ではない) |
| `test_chain_id_closes_on_next_move` / `_on_ojama_land` | 絶対律 2 信号で閉じる |
| `test_chain_id_survives_stable_flicker` | **267 ミリ秒の CHAIN→STABLE→CHAIN で切らない** (§5.2 の訂正の回帰) |
| `test_chain_id_survives_formula_gap_over_2sec` | `FORMULA_SESSION_RESET_SEC` を跨いでも絶対律未観測なら維持 (§5.2.3) |
| `test_chain_id_force_cut_increments_counter` | `CHAIN_ID_MAX_SEC` の強制切断がカウンタに出る |

### 14.5 episode の開始・参加・終了

| テスト | 内容 |
|---|---|
| `test_episode_opens_on_single_side_fire` | §9.2: **片側発火だけで開く** (現行 `ev1 and ev2` の欠陥の回帰防止) |
| `test_episode_participation_across_9sec_gap` | §9.1 #1→#3 の **9.2 秒**の隔たりを残量で繋ぐ (時間窓では不可能) |
| `test_seichi_lazy_open` | §9.4: 暫定生成量 ≤4 個の間は `current_episode() is None`、超えた瞬間に open |
| `test_unbacked_residual_is_isolated` | §2.3.1: 物理観測に裏付けのない残量では参加させない |
| `test_i7_closed_requires_zero_residual` | CLOSED は残差 0 必須。**CLOSED_FORCED は除外** |
| `test_no_concurrent_open_episodes` | §2.4.1: OPEN は高々 1 つ |
| `test_i11_match_boundary_isolation` | 試合を跨がない。境界跨ぎの `finalize` は拒否 |
| `test_forced_close_max_sec_increments_counter` | `EPISODE_MAX_SEC` の強制終了がカウンタに出る |

### 14.6 cap・符号・量の健全性

| テスト | 内容 |
|---|---|
| `test_i6_uncapped_net_vs_display` | 実測値 525 / 720 で `net_raw` を検証。**cap 後の引き算で生じる架空の攻撃が出ないこと** |
| `test_i9_side_swap_flips_sign_only` | 1P/2P を入れ替えると符号だけ反転 |
| `test_i12_amount_sanity` | `amount ≥ 0`、`Σ CANCEL ≤ 生成`、`Σ LAND ≤ 生成 − 相殺`、**`LAND` 1 件 ≤ 30** |

### 14.7 未解決ゲートと早期解除

| テスト | 内容 |
|---|---|
| `test_i15_no_episode_means_resolved` | OPEN が無ければ `is_unresolved() == False` |
| `test_is_unresolved_true_while_open` | OPEN の間は True |
| `test_allows_hard_override_when_decision_invariant` | §7.4: 未確定量を受け側に最も有利に倒しても死ぬなら True |
| `test_forbids_hard_override_when_outcome_depends_on_provisional` | 倒し方で結果が変わるうちは False |

### 14.8 bit-identical

| テスト | 内容 |
|---|---|
| `test_flag_off_is_bit_identical` | 既定 OFF で既存経路の出力が 1 バイトも変わらない |

### 14.9 性質テスト

| テスト | 内容 |
|---|---|
| `test_property_random_consistent_event_streams` | ランダムな整合イベント列で「CLOSE 時 I1 成立」「`net_raw` = 生成 − 相殺 − 着弾 の符号整合」を総当たり |

---

## 変更履歴

- 2026-08-24 初版 (Gate 1 ドラフト)。
- 2026-08-24 第2版。fable architect / reviewer の独立レビュー 2 系統と、
  seg01 game2 の実 dump 突合を反映。主な変更:
  - §2.3.1 追加 (裏付けのない残量による癒着の防止)
  - §2.4 を 2 分岐に書き直し (「その後1手置く」条件を削除)、§2.4.1 で OPEN 並存を禁止
  - §4.1.1 追加 (下げ置換の検算ゲート。`max` 恒久維持は保存則と両立しないため却下)
  - §4.1.2 追加 (暫定も確定と同じ `score_to_ojama` を通す)
  - §5.2 から「STABLE 復帰で chain_id を切る」を**削除** (断片化バグの再導入だった)
  - §5.2.1〜5.2.3 追加 (merge バイアス / 検出器の前科 / accumulator との階層)
  - §6.2 追加 (2 帳簿の乖離は仕様。方向は履歴依存)
  - §7.1 の `ExchangeEvent` を修正 (`STEP` 種別追加、`chain_id` を nullable、
    frozen 値から可変 `state` を除去)
  - §7.4 追加 (**決定不変性による早期解除**。これが無いと真の致死を弱める)
  - §8 に I11〜I15 追加、I2/I10 の矛盾解消、I7 に `CLOSED_FORCED` 除外を明記
  - §8.1 / §8.2 追加 (LANDED の観測契約 / TSUMO_PLACED の信号源)
  - §9.1.1 追加 (実 dump のフレーム単位突合)
  - §9.4.1 / §9.4.2 追加 (lazy open。整地の定義が実装と食い違う件は user 確認待ち)
  - §10 の「置き換える」から coalesce 窓を外し「当面残す」へ (内部矛盾の解消)
- 2026-08-25 第3版。実データ (zenchi/v51) 検証で確定した 5 根因への
  Fix【1】〜【5】を反映 (gate3_episode_v3 実装タスク):
  - §6.3 追加 (Fix【1】: 観測経路を `pending_pX_uncapped` 差分の判別的
    再構成へ変更。上限 216 の切り捨てが相殺・着弾の観測自体を塞いでいた)
  - §2.5.1 追加 (Fix【2】: ワイプの side 単位退役 `retire_side_chains`。
    試合境界 (両側) とは別の片側事象として扱う)
  - §5.2.4 追加 (Fix【3】: AWAITING_FINALIZE 中の段数増加は継続とみなす。
    すべての連鎖は 1 段目から始まる、という物理的根拠。v51 実データ 4/4 検証済み)
  - §7.3.1 追加 (Fix【4】: `_all_settled`/`_provisional_residual` を
    episode 単位に限定。ただし `_fire_events_of_open_chains`/
    `Episode.touch` の無条件巻き込みにより実効果が限定的な可能性を明記)
  - §4.1.2 に再訂正追記 (Fix【5】: 自己検算 `self_check_*`/
    `authoritative_ojama` を tautological な検算として削除)
  - §13 追加 (Gate 4 を測るのに必要な dump 列)

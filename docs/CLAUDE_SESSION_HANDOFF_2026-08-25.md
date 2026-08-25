# Claude セッション引き継ぎ (2026-08-25 作成)

このセッションは **Gate 3-2b (交換エピソード会計の実データ検証) まで完了**して終了した。
次のセッションは**この文書の「次にやる2件」から始めればよい**。

---

## 0. まず読むもの (プロジェクト規約)

1. `.claude/rules/00-agent-coordination.md`
2. `docs/agent_coordination/CURRENT.md`
3. `docs/agent_coordination/PLAN.md`
4. `docs/agent_coordination/CODEX_TO_CLAUDE.md`
5. `docs/agent_coordination/DECISIONS.md`
6. **`docs/agent_coordination/CLAUDE_TO_CODEX.md` の第6〜9報** (今回の全経緯)

作業前に
`powershell -ExecutionPolicy Bypass -File scripts/show_agent_coordination_status.ps1`
を実行して write lock と実行中ジョブを確認する。

---

## 1. 現在の状態 (2026-08-25 セッション終了時点)

| | |
|---|---|
| 全 pytest | **5,914 passed / 13 skipped / 0 failed** |
| **本番ファイルの変更** | **0件** |
| `src/production_config.py` | 掛け算式3要素のみ (user承認済み)。**交換エピソード会計は未登録** |
| 表示・判定への影響 | **なし** |
| 実行中のジョブ | **なし** (無関係な旧ジョブ `_prefetch_dl_failures_2026-08-18.py` が6日前から1本動いているだけ) |

### 新設した成果物 (**すべて既定OFF・未配線**)

| ファイル | 役割 |
|---|---|
| `src/chain_id_resolver.py` | 連鎖イベント列から安定 `chain_id` を復元する純関数 |
| `src/exchange_episode_tracker.py` | resolver + ledger を合成する観測アダプタ |
| `src/exchange_ledger.py` | 撃ち合いの会計台帳 |
| `tests/test_chain_id_resolver.py` / `test_exchange_episode_tracker.py` / `test_exchange_ledger.py` | 合計 **132本** |
| `docs/EXCHANGE_EPISODE_SPEC_2026-08-24.md` | 仕様書 (訂正注記つき) |

**未コミット (untracked / modified)。git status に他エージェントの差分も多数あるため、
一括コミットはしないこと。** 既存の差分・ログ・成果物を
reset / checkout / stash / 削除 / 上書きしない。

### 実測成果物 (すべて `data/verify/` 配下、上書き禁止)

| ディレクトリ | 内容 |
|---|---|
| `gate3_chainid_2026-08-24/` | Gate 3-0 chain_id 実測 (物理連鎖 16本の地上真値) |
| `gate3_episode_2026-08-24/` | 最初の実データ検証 (生成 45,036 の記録) |
| `gate3_rate_diag_2026-08-25/` | W38 の確定 |
| `gate3_breakdown_2026-08-25/` | 二重計上 74.5% の確定 |
| `gate3_episode_repr_2026-08-25/` | 代表動画 v51 の選定・境界確認 |
| `gate3_episode_fixed_2026-08-25/` | 是正1回目 |
| `gate3_episode_v2_2026-08-25/` | cap / ワイプ / 過剰分割の確定 |
| `gate3_unrec_2026-08-25/` | 未照合の分解、自己相殺の確定 |
| `gate3_episode_v3_2026-08-25/` | 是正2回目 |
| `gate3_episode_v4_2026-08-25/` `_v4_fixed_2026-08-25/` | 是正3回目 (測定器の是正) |
| `gate3_episode_v5_2026-08-25/v51_round_only/` | 窓を揃えた検算 (115.7%) |

---

## 2. 【次にやる 1】W38 の根治 — user から「やる」と指示済み

### 何が壊れているか (実測確定済み、`docs/KNOWN_WEAKNESSES.md` W38)

`src/chain_detector.py:186` の
`VideoChainTracker.__init__(..., match_start_sec: float = 0.0, ...)` に対し、
`src/recognition_pipeline.py` の**4か所すべてが `match_start_sec` を渡していない**
(**自分で grep して再確認済み**):

```
:3577  VideoChainTracker(debounce_confirm_frames=chain_debounce_confirm_frames)
:3581  VideoChainTracker(debounce_confirm_frames=chain_debounce_confirm_frames)
:4000  VideoChainTracker(debounce_confirm_frames=self._chain_debounce_confirm_frames)
:4004  VideoChainTracker(debounce_confirm_frames=self._chain_debounce_confirm_frames)
```

結果、`src/chain_detector.py:295` の
`elapsed = max(0.0, self._last_stable_t - self._match_start_sec)` が
**常に「動画の絶対時刻」**になる。

`src/scoring.py:503 compute_effective_rate` は 96秒以降 16秒ごとに 0.75倍、
最大14回減衰、下限 `OJAMA_RATE_MIN = 1`。
→ **96 + 14×16 = 320秒を超えるとレートが 1** になり、
**点数がそのままおじゃま個数**になる。

実測 (母数16): **16/16 でレート1**。
t=803.7 の連鎖で 正しい 1,579個 vs 壊れた 110,540個、**差 108,961** を厳密に再現。

### 影響範囲 (実測で切り分け済み)

| 経路 | 影響 |
|---|---|
| **本番のオーバーレイ表示** (`scripts/visualize_advantage_overlay.py:5441-5444`) | **なし** (`OjamaAccountingTracker._elapsed` を直接使用) |
| `src/ojama_accounting.py` の確定会計 | **なし** (`:1023 _reset_side_boundary` で試合ごとに正しく再設定) |
| `scripts/recognition_physics_review.py` | **あり** |
| `scripts/verify_ojama_score_inference.py` | **あり** (`:126` で `total_score / ojama_sent` をレートとして信頼 → **常に1**) |
| `src/old/timeline_analyzer.py` | あり (旧資産) |
| `scripts/_diag_*` 数本 | あり |

> **壊れているのは測定器で、製品は無事。** 測定器事故は17件目。
> このプロジェクトでは壊れた測定器が正しい修正を捨てさせた事故が繰り返し起きている
> (直近: 8/24 の v51「悪化」は物差しの穴だった)。

### 実装方針

`:4000` / `:4004` は**試合切替時のリセット処理の中**にあり、
その時点で試合開始時刻が分かる。**そこで `match_start_sec` を渡す。**
`OjamaAccountingTracker` が既に持っている値を流用できる。

`:3577` / `:3581` は動画先頭の生成で試合開始が未検出。
境界検出時のリセット (`:4000`/`:4004`) で上書きされるので実害は小さいはず。
**ただし「小さいはず」で済ませず、実測で確認すること。**

### リスクと必須の確認

- **`ChainEvent.ojama_sent` の値が変わる。**
  `tests/test_chain_detector.py:126` 等の期待値見直しが要る。
  **根拠なくテスト側だけ変えて通さないこと。1本ずつ理由を書くこと。**
- 上記の影響を受けるツールの出力が変わる。
  少なくとも `recognition_physics_review.py` と
  `verify_ojama_score_inference.py` は**修正前後で出力を比較**すること。
- **本番表示が bit-identical であることを必ず確認する**
  (影響なしと実測済みだが、変更後に再確認する)。
- **全 pytest を通す** (現在 5,914 passed / 0 failed が基準)。
  速度テスト `test_exchange_predictor.py::TestPredictionSpeed` は
  **並走負荷で落ちる**ので、落ちたら単独で再実行して確認すること
  (`feedback_speed_claims_need_parallelism_2026-08-20`)。

---

## 3. 【次にやる 2】生成量 115.7% の原因究明 — user から「やる」と指示済み

### 何が起きているか

v51 のラウンド (t=460〜531、実画面で境界確認済み) で:

| | 値 |
|---|---|
| 台帳の `raw_generation_total` | **2,097個** |
| 独立検算値 | **1,813個** |
| 比 | **115.7%** (284個の過大) |

検算値 = (55,269 + 71,627)点 ÷ **レート70**。

### すでに除外できたこと (再調査不要)

- **窓の切り方ではない**: t=459-545 で 119%、t=460-531 で 115.7%。ほぼ横ばい。
- **準備区間の混入ではない**: プローブは観測を `t_sec >= t0` でゲートしている
  (`scripts/_gate3_episode_probe_v4_2026-08-25.py:205,294,318,343` で確認済み)。
- **全消しボーナス** (`ALL_CLEAR_BONUS = 2100点 = 30個`、`src/scoring.py:125`) で
  説明できるのは 284個のうち約30個 (**10%**) のみ。

### 【最優先】未検証の有力仮説: **検算値の側が間違っている**

検算値は「レート70 が区間中ずっと一定」と仮定しているが、
**マージンタイムを無視している。**

逆算:

| | 計算 | 結果 |
|---|---|---|
| 実測から逆算した平均レート | 126,896 ÷ 2,097 | **60.5** |
| レート70 なら | 126,896 ÷ 70 | 1,813個 |
| レート52 (=70×0.75) なら | 126,896 ÷ 52 | 2,440個 |

**実測はちょうどその間**。区間の途中でレートが 70 → 52 に落ちたと考えると整合する。
さらに逆算すると **約 57,470点 (全体の45%) がレート52 で換算されていれば実測値と一致**。

#### なぜそうなりうるか

プローブは `t0=459`、準備区間30秒なので **t=429 から読み始める**。
ラウンドの実際の開始は **t≈460**。**差は31秒**。

会計側の `_match_start_sec` が 460 ではなく 429 付近なら、
経過96秒に達するのは **t≈525**。ラウンド終了は t=531。

そして **この試合で最大の連鎖 (全消しを含む10連鎖) は終盤 t=528〜531 に起きている**
(前任が実画面で確認済み: `data/verify/gate3_unrec_2026-08-25/frames/v51_t528.png` 等)。

**当たっていれば台帳の 2,097 のほうが正しく、「115.7%の過大」という問題自体が
存在しなかったことになる。**

### 確定させる手順 (これだけ測ればよい)

v51 のラウンド **t=460〜531** について:

1. `OjamaAccountingTracker._match_start_sec` の**実際の値** (460付近か 429付近か)
2. その区間で `compute_effective_rate(elapsed_sec, 70)` が**返した値の時系列**
3. **レート70未満で換算された点数の合計** (逆算値 約57,470点 = 45% と一致するか)
4. 実際のレートで検算しなおした値 = `Σ(各連鎖の点数 ÷ その時点の実効レート)`
   が **2,097 に近づくか**

出力先は `data/verify/gate3_rate_trace_2026-08-25/` (新規)。
既存プローブ `scripts/_gate3_episode_probe_v4_2026-08-25.py` を
**コピーして**使う (元を書き換えない)。v51 の1ラウンド86秒なので安価。

### 棄却された場合の次の候補 (優先順)

| # | 候補 | 根拠 |
|---|---|---|
| a | `score_to_ojama` の繰り越しの扱い | tracker は `prev_leftover=0` 固定。1連鎖あたり最大69点の誤差。**ただし方向は過小**なので過大の説明にはならない点に注意 |
| b | 連鎖の過剰分割による同一値の二重計上 | 実測で「chain4 の確定 11,460 → chain5 の暫定 11,460」という**同一値の重複**を観測済み。11,460点 = 163個。**284個のうち57%を説明しうる** (未検証) |
| c | 掛け算式の成長フェーズの過大 | 実測で `provisional=0` の chain (chain6) が存在するなど、成長フェーズの値は不安定 |

**b は有力。** zenchi でも chain_id 27 vs 物理連鎖 16 で**まだ約1.7倍に割れている**
(5本は隣接統合で説明、**6本は未測定**)。

---

## 4. 未着手として記録済みの項目 (**すべてカウンタで可視化済み。黙って落としていない**)

| 項目 | カウンタ | 現在値 |
|---|---|---|
| close 後に届いた相殺・着弾が欠落する | `post_close_settlement_dropped_count` / `_amount` | 0/0 |
| 自己相殺のクリップ上限超え | `self_cancel_clipped_count` / `self_cancel_eligible_count` | **0/5** |
| `_fire_events_of_open_chains` / `Episode.touch` の**無条件巻き込み** | (未計装) | episode の粒度が「1回の撃ち合い」でなく「ラウンド全体」になる原因 |
| chain_id 27 vs 物理連鎖 16 の残差6本 | — | 未測定 |

---

## 5. Codex への判断待ち (`CLAUDE_TO_CODEX.md` 第9報)

| # | 内容 |
|---|---|
| J1 | 仕様の破壊的変更2件への異議 — **E1 の削除**、**I16 (FINALIZE 供給源の限定)** |
| J2 | W38 の根治 → **user が「やる」と判断済み** |
| J3 | 115.7% の究明 → **user が「やる」と判断済み** |
| J4 | `_fire_events_of_open_chains` の無条件巻き込みを是正するか |
| J5 | `FINALIZE_DOWNWARD_TOLERANCE` の2項化は**不要になった可能性** (乖離分布が改善したため)。判断は D2 の分布を広い母数で取り直してから |

---

## 6. このセッションで確立した規律 (次も守ること)

1. **カウンタは必ず母数と並べる** (`0/23` の形)。
   一晩で **5回**、`0` が「合っている」ではなく「測っていない」だった。
   memory `feedback_zero_needs_denominator_2026-08-25`。
2. **配線する前に実データで測る。** これで欠陥10件が配線前に出た。
3. **サブエージェントに「期待どおりにならなければ隠さず報告」を毎回明示する。**
   この指示から実際に欠陥が5件出た。
4. **対象より物差しを先に疑う。**
5. **見積もりを実測より先に信じない。** このセッションで私の見立ては **7回**外れた。
6. **サブエージェントに待機を任せない。** 親から見えず通知が空振りする。
   `tail` / `pgrep -c -f pattern` で自分で確認する。

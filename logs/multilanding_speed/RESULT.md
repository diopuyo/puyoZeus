# 評価の遅延 (応手全探索・E35事後上界・隠し段候補) — 再現・高速化・有界化 (2026-09-30)

ブランチ claude/exchange-event-eval-20260926 (基準 HEAD 1d7de42)。実装コミット 6f370c8 + 本結果のコミット。
測定は WSL (Ubuntu) の venv、nice 19、原則 1 プロセス。認識は変えないので映像は使わず、保存記録の再生だけで行った (検証の梯子の段 1〜2)。

## 0. 先に結論

1. **B20 の 54 秒は応手全探索 (multilanding) ではなかった。** 同じ rt の保存入力を現本番構成で再生すると、通知 1 件の重い所は 3 系統に分かれる。
   - 応手全探索 `prove_multilanding` (最大 9.8 秒)。B18 の 6.4 秒 (zenchi 5984.6) はこの系統。
   - E35 の打ち返し後死亡上限 (2,798 候補盤面を全部証明) 最大 16.6 秒 (6245.5)。
   - 隠し段の得点候補の列挙 `weighted_landing` (10,340 通り) 最大 18.3 秒 (6492.8)。B20 の 54 秒 (6492.8 付近) はこの系統。
2. **出力同一の高速化**で応手全探索は 15.7 倍 (274 呼出し合計 204.5 秒 → 13.0 秒)。P99 6.3 秒 → 0.55 秒、1 秒超 58 件 → 0 件。出力は保存記録 5 本の全ファイルとリアルタイム由来 3 本で**バイト一致**。
3. 出力同一の範囲では E35 と隠し段列挙の尾が残る (B20 窓で 1 秒超 66 → 30 件)。**既定OFFの有界化**を事前登録して実走した。
   - B2 (E35 早期打切り + 隠し段候補上限 256) = 判定を変えず (保存記録 5 本の全 51 ファイルが内訳を除き一致)、B20 窓の 1 秒超 66 → 0 件・最大 18.9 秒 → 0.62 秒。
   - B1 (B2 + 応手全探索の上限 5000 ノード) = 事前登録の門を全て通過し、最大 0.26 秒。ただし**実在の死亡検出を 1 件失う** (zenchi 第11試合 3109.5 秒、19,022 ノードの死亡証明)。**私は B2 を推奨**する (下記 §4)。
4. 目標の「最悪 200 ms 台」には B1 で届く (b20 窓 P99.9 194 ms・最大 262 ms)。B2 は最大 0.62 秒、200 ms 超が 7 件 (隠し段候補 256 通り分が 0.4 秒前後)。
5. 本番採用はしていない (`src/production_config.py` 不変)。有界化の 3 フラグは既定OFF。出力同一の高速化は、出力が変わらないことを実測した上で既定ONにした (切戻し用の定数あり、§2)。

## 1. 段2: 保存記録の再生での分布 (変更前 = 現本番、母数つき)

再生 = `scripts/profile_multilanding_speed.py` (本番の E36b 構成をそのまま走らせ、関数を素通しで包んで計時)。
**測定器の健全性**: 計装つき再生の出力は、現本番の保存出力 (logs/pending_expiry/e36b_on) と **51/51 ファイル一致** (壁時計の項目だけ除外)。
入力 8 種 = 保存記録 5 本 (q / review / fcXG83vInDY / mia8KCjr52g / zenchi 2580〜3427 秒) + rt の保存入力 3 本 (b18 stall 5880〜6100、b18 run5 5880〜6180、b20 rt_on 5830〜7030 = zenchi 後半)。
**zenchi の 5500〜7000 秒は保存記録に無い**ので、rt の保存入力を同じ再生器に通して再現した (`--record`)。

| 関数 (母数) | P50 | P95 | P99 | 最大 | 100ms超 | 1秒超 |
|---|---|---|---|---|---|---|
| 通知 1 件の評価全体 `_evaluate` (19,295) | 2.8 ms | 133 ms | 472 ms | 18,269 ms | 1,209 | 119 |
| `evaluate_multilanding` (19,295、E35 を含む) | 0.07 ms | 0.55 ms | 135 ms | 16,611 ms | 214 | 60 |
| `prove_multilanding` (証明が走った 274 件) | 215 ms | 3,151 ms | 6,294 ms | 9,797 ms | 189 | 58 |
| E35 `post_counter_evaluate` (19,295) | 0.006 ms | 0.39 ms | 10 ms | 16,601 ms | 65 | 16 |
| 近未来探索 `future_send` の不命中 (2,272) | 72 ms | 351 ms | 566 ms | 1,331 ms | 860 | 10 |

出典別の通知全体 (P50 / P95 / P99 / 最大 / 100ms超 / 1秒超 / 合計):

| 出典 (母数) | 変更前 | 出力同一の高速化後 (after2) |
|---|---|---|
| q (2,995) | 4.8 / 155 / 860 / 9,807 ms / 185 / 25 / 138.7 s | 2.8 / 37 / 120 / 743 ms / 46 / 0 / 26.0 s |
| review (862) | 2.9 / 124 / 273 / 486 / 90 / 0 / 19.7 s | 2.4 / 51 / 124 / 209 / 19 / 0 / 9.5 s |
| fcXG83vInDY (3,381) | 4.8 / 193 / 433 / 5,946 / 269 / 11 / 125.5 s | 2.7 / 30 / 110 / 3,111 / 38 / 1 / 29.4 s |
| mia8KCjr52g (3,212) | 2.0 / 2.9 / 3.4 / 13.5 / 0 / 0 / 6.7 s | 1.8 / 2.4 / 2.9 / 36 / 0 / 0 / 5.7 s |
| zenchi 2580〜3427 (3,078) | 4.0 / 168 / 443 / 8,304 / 263 / 11 / 121.2 s | 2.5 / 33 / 130 / 7,567 / 50 / 4 / 41.2 s |
| b18 stall (509) | 2.1 / 93 / 391 / 2,700 / 26 / 4 / 14.2 s | 2.0 / 18 / 58 / 683 / 4 / 0 / 3.2 s |
| b18 run5 (1,062) | 1.9 / 82 / 344 / 2,654 / 46 / 4 / 25.1 s | 2.1 / 22 / 44 / 274 / 3 / 0 / 5.6 s |
| b20 rt_on (4,196) | 2.4 / 180 / 1,954 / 18,269 / 330 / 64 / 387 s | 2.4 / 36 / 396 / 18,522 / 93 / 29 / 186 s |

注: 上の「変更前 / 後」は別の時刻・同時に他の私のジョブが走っていた回を含むので、**同じ表内でも ±50% 程度は揺れる** (E35 は今回一切変更していないのに、同じ通知が 18.8 秒と 28.9 秒になった回がある。§3 の単独計測も参照)。比は桁で読むこと。

遅い呼出し上位 20 件 (出典・所要・盤面の埋まり数・ノード数など) は `base_head/TOP20_prove.tsv`、遅い通知上位 20 件と内訳は `base_head/TOP20_notifications.tsv`。上位は次のとおり。
- 応手全探索の上位はほぼ全て `node_limit` (ノード上限 20,001 到達 = 結局「証明できない」で死亡にならない計算): q 9.8 / 7.4 / 7.2 秒 (埋まり 14〜38、受け 132〜330 個)、fc 5.9 / 4.1 秒、zenchi 4.1 秒 (これだけ 19,022 ノードで死亡証明が成立)。
- 遅い通知上位 20 は b20 の 6244〜6258 (E35、候補 334〜2,798 盤面、うち最初の候補が死なないのに全部を証明) と 6492〜6495 (隠し段候補 6,160〜10,340 通り)。

B18 の 5984.6 秒 (6.4 秒) は、保存入力の再生 (変更前) では 12 ms で**同じ時刻には再現しない**。近い重さは 5965.8〜5968.3 (2.0〜2.7 秒) と 6044〜6045 (2.3〜2.7 秒) の応手全探索。ライブでは評価が遅れて通知の時刻がずれた・同時実行で遅くなった・複数証明が合算した、のいずれかと考えるが**推測**であり未確認。

## 2. 出力同一の高速化 (既定ON。出力が変わらないことを実測)

| 内容 | 場所 |
|---|---|
| 打切り予告: 1 段の展開に要するノード数は盤面の列の高さだけで数えられる。上限を超える段は探索せず同じ `node_limit` (nodes=上限+1、完了済み rounds) にする。次の段の下限 (これまでに出た盤面分) だけで超過が確定する場合も同様 | `src/exchange_multilanding_precheck.py`、`src/exchange_event_multilanding.py` の `Search.reserve` / `reserve_later` / `responses` |
| 連鎖しない子盤面の連結検出を省く: 基準盤面が静止 (4 連結なし) なら、子盤面で 4 連結ができるのは足したセルを含む場合だけ。足したセルの周りだけ調べ、連鎖しなければ「連鎖 0・得点 0・盤面そのまま」とする (連鎖あり・基準が静止でない場合は従来の `simulate`) | `src/exchange_fast_expand.py` (`FastExpander`、`local_pop`)、`Search.outcomes`、`src/indicators_v2.py` の `_near_future_known_expand` / `_free_expand` に optional 引数 `fast_expand` (既定 False) |
| `ChainSimulator(fast_groups=True)` (既定 False): 連結検出・消去・重力をセル参照でなく配列/リスト操作にする | `src/chain.py` (`_find_groups_fast`、`_simulate_fast`、`_gravity_fast`) |
| おじゃま着地の全列配置を配列操作に | `src/exchange_fast_expand.py::drop_ojama_fast` |
| 仮想着弾評価 (M0+D列+モデル) と S3 勝率の再利用: 入力 (盤面・NEXT・経過秒・勘定) が同じなら結果は同じ純関数 | `src/exchange_event_landing.py::_landing_gfe` / `_gfe_key`、`src/exchange_hidden_row_probability.py::s3_probability` |

切戻し用の定数 (False で従来経路): `EXACT_PRECHECK` `EXACT_FAST_GROUPS` `EXACT_FAST_EXPAND` (multilanding) / `EXACT_FAST_GROUPS` `EXACT_FAST_EXPAND` `EXACT_GFE_CACHE` (landing) / `EXACT_S3_CACHE` (hidden_row_probability)。

**一致の実測 (母数つき)**
- 証明関数の単体 (`scripts/bench_multilanding_prove.py`): 保存した実入力 **274 呼出し**を再実行し、**結果 dict 全文の SHA-256 が一致 259 件 + (SHA なしの 15 件は reason/nodes/dead が一致) = 不一致 0**。合計 204.5 秒 → 13.0 秒 (15.7 倍)、P50 215 → 23 ms、P95 3,151 → 169 ms、P99 6,294 → 547 ms、最大 9,797 → 828 ms、1 秒超 58 → 0 (再実行は他ジョブ並走・future_send のキャッシュを毎回捨てた単独費用)。
- 保存記録 5 本の再生 (`scripts/compare_replay_outputs.py`、display.npz / events.jsonl / 各 audit / scene_timeline など 1 本 10〜11 ファイル): **51/51 ファイルがバイト一致** (現本番の保存出力との比較、変更後コードの 2 回 = after2 と最終コード gate_exact)。壁時計の項目 (elapsed 系・filter_sec・enumeration_sec) だけ比較から除外。
- リアルタイム由来 3 本 (b18 stall / run5 / b20): 各 7 ファイル、**21/21 一致** (2 回)。
- 新規単体テスト 36 件 (`tests/test_multilanding_precheck.py` 6、`test_chain_fast_groups.py` 9、`test_exchange_fast_expand.py` 12、`test_exchange_evaluation_cache.py` 5、`test_post_counter_early_exit.py` 4)。関連既存テストと合わせて **848 passed / 1 skipped**。全体 pytest は座長の指示で途中 (54%) で止めた。途中までの失敗・エラーは data/verify の欠落ファイルによるもの (G3 / video38 診断系) で、今回の変更範囲のテストは全て通っている。

## 3. B20 窓 (zenchi 5830〜7030 秒、通知 4,196 件) の単独計測 (1 プロセス・nice 19)

| 構成 | 合計 | P50 | P95 | P99 | P99.9 | 最大 | 200ms超 | 1秒超 | 5秒超 |
|---|---|---|---|---|---|---|---|---|---|
| 変更前 (現本番) 1回目 | 430 s | 2.8 | 210 | 2,034 | 12,052 | 18,866 | 225 | 66 | 23 |
| 出力同一の高速化のみ 1回目 | 269 s | 3.8 | 55 | 482 | 9,828 | 28,945 | 65 | 30 | 17 |
| B2 (早期打切り + 候補上限 256) 1回目 | 38.5 s | 2.4 | 36 | 179 | 209 | 618 | 7 | 0 | 0 |
| B1 (B2 + ノード上限 5000) 1回目 | 37.3 s | 2.4 | 37 | 172 | 194 | 262 | 3 | 0 | 0 |

(単位 ms。合計だけ秒。)

- 「高速化のみ」の最大が変更前より大きいのは、E35 (未変更のコード) が同じ通知で 18.8 → 28.9 秒と 1.5 倍遅かった回のため = **実行間の揺れ** (同時にホスト側の別ジョブがあった可能性)。E35 の尾は出力同一の範囲では直していない。
- 出力同一の範囲で消えたのは応手全探索の尾と、隠し段列挙の約半分 (18.3 → 9.2 秒)。残りは E35 と隠し段列挙。

### 3b. 2 回目の単独計測は中止

実行順を逆にした 2 回目を走らせたが、途中で別エージェントの発火前予測ジョブ (16 プロセス、nice 0) が始まりホストが飽和したため、座長の指示 (nice 19・1 プロセス) に従って中止し、汚染された測定値は捨てた。したがって上の表は**1 回分**で、実行間の揺れは E35 の 1.5 倍の例 (上記) が目安。判断に使うのは桁と、同じ通知どうしの比。

## 4. 有界化 (既定OFF、`docs/agent_coordination/DECISIONS.md` に実走前に事前登録)

- `post_counter_early_exit`: E35 で最初に死なない候補が見つかった時点で残りの候補証明を省く。「全候補が死ぬ場合だけ死亡」の仕様なので判定は同一 (監査の proofs だけ短くなる)。
- `multilanding_node_limit=5000`: 応手全探索の総ノード上限 20000 → 5000 (打切りは死亡にしない)。
- `hidden_scenario_cap=256`: 隠し段の得点候補が 256 通りを超えるとき、重み上位 256 通り (同点は元の順) に絞り、重みの合計は保つ。
- 配線: `ExchangeEventOverlay(…, post_counter_early_exit=, multilanding_node_limit=, hidden_scenario_cap=)` と `replay()` の optional 引数。**本番の CLI (`--…`) と rt 側の起動配線は未実施** (採用が決まってから、30 分程度)。

**門 (現本番より悪化しない)** — 保存記録 5 本の再生 (`scripts/run_multilanding_speed_gate.py`):

| 門 | 現本番 | B1 | B2 (診断) |
|---|---|---|---|
| q log loss (6,526 フレーム) ≤ .507567+5e-7 | .5075670 | .5075670 通過 | .5075670 通過 |
| zenchi 的中 ≥ 7,671 / 8,333 | 7,671 | 7,671 通過 | 7,671 通過 |
| 誤った負け確定 | 0/37 | 0/37 通過 | 0/37 通過 |
| 3:00 場面 ≤ 2766.0 秒 | 2760.38 | 2760.38 通過 | 2760.38 通過 |
| q 第14試合の誤確定 | 0 | 0 通過 | 0 通過 |
| 出力 (内訳 proofs を除く 51 ファイル) | — | 47/51 一致 | **51/51 一致** |

- B1 の差 = q 4 事象・fc 5 事象・zenchi 2 事象の events.jsonl (うち応手全探索の打切りが 20001 → 5001 ノードになっただけで結論が同じもの、監査の single_death_proof の内訳)、および **zenchi 第11試合 3109.5 秒の死亡確定が消えた** (19,022 ノードの死亡証明 → 5,001 で `node_limit`、p1 0.98 → 0.921)。zenchi の display.npz は display_p1 5 フレーム (最大 0.059)・display_adv 130 フレーム (最大 21.9) が動く。**門はこれを検出しない** (的中数・場面・誤確定は不変) が、実在の死亡検出の遅れであり、事前登録の選択規則は「B1 が門を通れば B1」だが、私は B2 を採用候補として推す。
- B2 は事前登録では「B1 が落ちた場合だけ」走らせる約束だったので、**診断として追加で走らせた** (結果は門の判断には使っていない)。
- B2 の隠し段候補上限 256 は、保存記録 5 本では一度も発動しない (全記録の最大候補数: q 15・review 153・fc 13・mia 0・zenchi 153、b18 は 26 以下。256 超は b20 窓の 43 通知だけ) ので、5 本のゲートでは効果を測れない。効く場面 (b20 窓の 4,196 通知のうち 43 通知) では、B20 再生の評価値 8,234 件のうち p1 が動いたのは 62 件 (0.75%)、1 ポイント超は 20 件、最大 3.4 ポイント、**判定 (source / dead_sides) の変化 0 件**、保存された表示行 1,666 行は不変。B1 は同 64 件・最大 10.2 ポイント・判定変化 2 件。

## 5. 残る遅さと次にやること

1. B2 でも 200 ms 超が 7 件 (b20 窓) — 隠し段候補 256 通りの評価が 0.4 秒前後 (1 通り 1.5 ms 前後。内訳は未特定で、新しい盤面ごとの D 列指標の計算が主と**推測**)。候補上限を 100 前後に下げれば 200 ms 内に入るが、上限が効く通知が増える。**上限値は採否の判断待ち** (今回は事前登録値 256 のまま)。
2. E35 自体の高速化は未実施 (早期打切りで尾を消しただけ。全候補が死ぬ場合は従来どおり全候補を証明する)。
3. 隠し段候補の S3 勝率を 1 回のバッチ推論にまとめる案 (HGB は行ごとに独立なのでビット一致する見込みだが**未検証**) と、D 列の近未来指標へ `fast_expand` を通す案 (学習特徴と同じ関数なので検証が重い) は未実施。
4. 発火前予測 (`prefire_audit`) の列挙が 1 事象 1.1 秒前後かかる場面がある (WSL では native あり)。Windows の配布 Python には puyo_core (Rust) が無く、今回の高速化は Python のみで効く。native を使う拡張 (`enumerate_and_simulate_placements`) は未着手 (Windows 向け wheel の用意が前提)。
5. メモリ: 常駐は再生 1 プロセスあたり b20 で最大 916 MB (変更前 896 MB、+2%)、保存記録の再生で 800〜833 MB。近未来探索用の `ChainSimulator` を 1 個追加した (キャッシュ上限は従来と同じ 5 万件)。

## 6. 再現手順・成果物

- 計装再生: `logs/multilanding_speed/run_variant.sh <ROOT> <OUT名> sources|live [名前…]` (WSL、nice 19)。出力は `base_head/` (変更前)、`after2/` `after3/` (変更後)、`clean_*/` `clean2_*/` (単独計測)、`live_bounded*/`。
- 証明単体: `scripts/bench_multilanding_prove.py`、集計: `scripts/analyze_multilanding_profile.py`、`scripts/summarize_multilanding_speed.py`、`scripts/top_slow_multilanding.py`。
- 出力一致: `scripts/compare_replay_outputs.py <基準> <比較> [--ignore-proof-detail]`。有界化の影響: `scripts/compare_bounded_effect.py`。
- ゲート: `logs/multilanding_speed/run_gate.sh exact|bounded|bounded2` → `gate_*/SUMMARY.json`。
- 変更前コードの固定: `../snap_base` (git archive HEAD)。変更後コードの固定: `../snap_after2`。

## 7. B2 の本番配線と配線検査 (2026-09-30、座長判定で B2 採用)

- 配線: `--post-counter-early-exit` と `--hidden-scenario-cap N` を描画 (`scripts/visualize_advantage_overlay.py`) と再生 (`scripts/replay_exchange_event_20260926.py`) の CLI に追加し、`src/production_config.py` の `EXCHANGE_EVENT_ADOPTED` へ登録 (`--post-counter-early-exit`、`--hidden-scenario-cap 256`、採用日 2026-09-30)。`--production-exchange-event` で両方 ON。ノード上限 (`multilanding_node_limit`) は不採用 (CLI なし)。
- 検査 (取り違え・配線漏れ): `--production-exchange-event` だけで再生し、B2 を明示指定した既存の門 run (gate_bounded2) と `--compare` (display.npz・events.jsonl バイト一致 + diagnostics 一致) で照合。**保存記録 5 本 5/5 一致、出力 35/35 ファイル一致** (`wiring_check/`)。上限が発動しない 5 本では上限の配線漏れを検出できないため、上限が発動する b20 (43 通知で発動) でも B2 明示指定の再生 (live_bounded2) と **バイト一致**を確認 (`wiring_check/b20.log`)。
- テスト: `tests/test_exchange_event_production.py` を更新 (採用旗の集合・根拠・両 CLI の実効引数一致・render で ON)。関連 253 passed。

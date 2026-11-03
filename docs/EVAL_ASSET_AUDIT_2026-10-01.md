# 評価部品の全数監査 (2026-10-01)

user 指摘「なぜ過去作った部品が使われていないのか、他にもあるのでは」を受けた読取専用の監査 (librarian)。
本番経路は `--production-exchange-event` → `EXCHANGE_EVENT_ADOPTED` / `PLACEMENT_RECONCILE_ADOPTED` (src/production_config.py) を実際に辿って判定した。

## 0. 表示の出どころ (本番ゲート記録 logs/pending_expiry/e36b_on、5本・約11万フレーム)
| 出どころ | 割合 |
|---|---|
| G_fe (静止) | 46〜55% |
| S3_landing | 32〜41% |
| S3_provisional | 2.4〜6.0% |
| confirmed_death | 3.4〜7.6% |
| unavoidable_death | 0〜4.8% |
| waiting_confirmed (旧評価器の値) | 0.2〜4.4% |
| S1 | 0フレーム (同フレームで S3_provisional が上書き) |

## 1. 状態別の部品
### A. 本番 (撃ち合い評価)
G_fe/S1′/S3′/M0 (models/exchange_event_v3)、count特徴 (NF_ojama_k1..5・counter_margin、S′のみ。G_feには入らない)、応手全探索 (multilanding、未知ツモは5色列挙)、E35 打ち返し後死亡上限 (試合別4色)、D5/D5b 単発死亡証明、E31/E32 発火前盤面・隠し段確率、落下ボーナス吸収。

### B. 計算しているが表示にほぼ出ない (旧評価器)
- MC応手 `--counter-reach` (scripts/mc_counter_estimator.py、8/12採用 AUC .837): waiting_confirmed のフレームだけ表示。
- rt の非同期MC (src/phase_j/live_counter.py): CPU を使い続けるが表示寄与は試合開始直後だけ。
- `--resolved-*`・kill-override: 新評価器の S3_landing/unavoidable_death と同機能で、新評価器が値を持つ間は表示されない。
- `ojama_damage` (12/18個の折れ点): 旧評価器内のみ。新評価器は仮想着弾+G_fe で代用。

### C. 作ったが本番に繋がっていない
| 部品 | 状態 |
|---|---|
| E18 `--landing-counter-response` | 9/28 不合格 (q .529→.512 良化、zenchi 80.4→73.4%)。応手が理想ツモ (`future_send`) で過大見積もり。比較基準は古い E15+② |
| E19 `--landing-counter-prob` | 9/28 不合格 (zenchi 79.37% < 80.10%)。今の本番 (92%) では未測定 |
| E20 `--landing-hands-spec` | 不合格 |
| E16/E21/E25/E29/E30/E33/E34 | 不合格 |
| 発火前予測 構成A/B (hazard) | 9/30 不合格。撃つ側の的中 94/181 (52%) |
| NEXTずれ補正 (wt_nextfix、v5 + queue-align) | 10/1 初回不合格 (q .5154、zenchi +172)。q の乱数ゆらぎ .005 > 門 .002。5 seed 平均で再判定中 |
| 案D `src/exchange_predictor.py` (AUC .786) | scripts からのみ |
| `counter_reach_adaptive` | bench のみ。mc_counter と重複 |
| `expected_fire_power` / `counter_reach_probability` | 学習データ上 100% NaN |
| `saisoku_hold` / `absorption_capacity` / `taiou_capacity` / `ojama_disruption` | G_fe の列に無い |
| 試合別4色 `match_color_evidence` | E35・隠し段・発火候補では使用。count特徴と S′ は盤面出現色で近似 |

### D. 重複実装
応手推定7系統、おじゃま台帳約8系統、マージンタイム時計3系統 (user仕様「両者のうち早い方の1手目」はどこにも無い)、連鎖シミュレータ5系統 (`ChainSimulator` は幽霊連鎖の既定 False、src/chain.py:153)、隠し段推定6本。

## 2. memory の user 仕様の実装状況
| 仕様 | 状況 |
|---|---|
| 色ぷよ有利さのおじゃま減衰/フラット時/局面組合せ | 部分的 (色差×おじゃま差は G_fe にある。フラット時・局面組合せは無い) |
| おじゃまダメージ関数 | 旧指標のみ。新評価器は仮想着弾で代用 |
| マージンタイム起点 = 両者の早い方 | 未実装 |
| 落下ボーナス 1マス1点 | 実装済み |
| 飽和 (4色限定) | 4色限定かは未確認 |
| 催促の価値モデル | `saisoku_hold` のみ、新評価器に未投入 |
| 組ぷよK1〜K4・複数時間軸探索 (9/4 user決定) | 未実装 (main PLAN.md:582 に計画のみ) |
| 打ち返し後の負け確定 | E35 で本番 |
| 置き終わりの合図3つ | NEXT・掛け算式は R1b で本番、おじゃま合図は除外 |

## 3. 弱点に効きそうな順 (上位10)
弱点: (A) 発火前に勝率が動かない (B) 中盤の頭打ち (C) 評価器切替時の値飛び。前提: 148動画CVで静止情報の中盤天井 ≈0.58。

| 順 | 候補 | 効く弱点 | 手間 |
|---|---|---|---|
| 1 | NEXTずれ補正 | A・S′ | 小 (再判定中) |
| 2 | 試合別4色を count特徴/S′ と応手全探索へ | A・速度 | 中 (S′再学習) |
| 3 | 切替時の値飛び対策 (waiting_confirmed で EMA を通さない・古い EMA から再開、S3→G_fe の切替ブレンド未実装、41回/45分) | C | 小 |
| 4 | E18 を平均ツモの応手で測り直し | A | 中 |
| 5 | E19 を今の本番の上で測り直し | A | 小 |
| 6 | hazard ラベルを催促定義に合わせる | A | 中〜大 (※10/1 user方針で hazard 方式は不採用、最善手探索へ) |
| 7 | 静止 G_fe に K1〜K5・counter_margin を非線形で投入 | B | 中 |
| 8 | 案D を S1 特徴へ (S1 は表示0%なので先に表示経路の確認) | A | 小〜中 |
| 9 | 受け容量・催促保持を G_fe へ | B | 中 |
| 10 | マージンタイム時計の1本化 | 正確さ | 小 |

別枠: rt の非同期MC は表示寄与ほぼ0で CPU を使っている → 停止か応手推定への転用。

## 4. 使われなくなった根本原因
1. 評価器の置き換えで旧部品が影に隠れた (採用台帳を見ても表示に効いているか分からない)。
2. 比較基準が上がったのに、不合格案を測り直していない。
3. 門の分解能不足 (q は4試合、乱数で .005 揺れる)。
4. ブランチの分断 (main 未マージ、Codex 期の計画が E 系列へ未引継ぎ、scripts の正本が不明)。
5. 良い版が別所で作られ移植されない (4色 vs 盤面色近似、理想ツモ vs 平均ツモ)。
6. memory の仕様とコードが紐づいていない。
7. 学習と本番の入力ずれの発見が遅い (配線の「間違い」型)。

## 未確認
`saturation_chain_upper` の4色限定、幽霊連鎖OFFの `ChainSimulator()` が認識の本番経路に乗っているか、S1 の表示0%が意図した設計か。

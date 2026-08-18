# 連鎖の先取り予測 — 指標側で使うための知見 (2026-08-18)

## 何のための文書か

連鎖が終わった後の盤面を先取りして予測する機構 (`estimated_board`) を、
**指標計算・オーバーレイの側で使う**ときに必要な事実をまとめる。

user 判断により、**この予測を収集データ (npz) に記録することはしない**
(盤面から導出できるため二重に持つ必要がない)。指標を出すときにその場で
算出する。

### なぜ先取りが要るのか (user 2026-08-18)

> 先取りはするべき。**残ったフィールドの形を正確に把握することで、打たれた方が
> どのくらいの条件で返すべきなのかが決まる**。連鎖終了後の形も打ち合い有利不利の
> 条件にいれることができる

相手の連鎖が終わった後にどんな形が残るかは、打たれた側が「どれだけの連鎖で
返すべきか」を決める材料そのもの。打ち合いの有利不利を評価する条件になる。

## 予測が返すもの

**途中経過ではなく、連鎖が全部終わった後の最終盤面**。
6連鎖を撃った場合、発火した瞬間に「6連鎖が終わった後の盤面」が計算され、
連鎖アニメが流れている間ずっとその値が保持される (3連鎖目の中間状態は持たない)。
`ChainSimulator.simulate()` (`src/chain.py:163`) が最終結果のみを返す設計のため。

したがって「何秒後に終わるか」を予測する必要はない。発火した瞬間から使い始め、
実測が静止に戻ったら自動的に置き換わる。

## 精度の実測 (2026-08-18、3動画×各400秒、160遷移)

予測盤面と、その直後の実測STABLE盤面のセル単位 (13×6=78セル) 比較。

| 動画 | 遷移数 | 平均一致率 | 完全一致 |
|---|---|---|---|
| 36 (1P/2P) | 28 / 21 | 92.35% / 92.37% | 19/28, 15/21 |
| 52 (1P/2P) | 32 / 32 | 95.31% / 93.35% | 26/32, 22/32 |
| c100 (1P/2P) | 23 / 24 | 87.57% / 88.51% | 9/23, 13/24 |

**加重平均 91.9% / 完全一致 65.0% (104/160)**。
中央値は多くが100%。ただし**最悪ケースで24〜40%まで落ちる例がある**
(要因未調査。起点盤面の誤認、または `low_confidence` 判定の閾値が甘い可能性)。

指標側で使うときは、この不確かさを踏まえること。確度を併記して扱う設計が望ましい。

## 重要な注意 — docstring が実挙動と食い違っている

`src/recognition_pipeline.py:423-430` の `SideResult.estimated_board` の
docstring は「CHAIN/GRAVITY_SETTLE 中は `confirmed_board` が None のまま」と
書いているが、**これは現在の実挙動と一致しない**。

実測 (video_36、30秒間、CHAIN 251フレーム + GRAVITY_SETTLE 20フレーム):
`confirmed_board is None` は **0件**。None になるのは MENU 状態のみ。
CHAIN 中は直前の STABLE 値が凍結されたまま非 None で保持される
(CLAUDE.md 設計思想4「NON-STABLE 中は前回 STABLE 盤面を凍結」の通りの挙動)。

**したがって指標側で予測を使うときは、`confirmed_board` が None かどうかで
判定してはいけない。`bstate` が CHAIN / GRAVITY_SETTLE かどうかだけで判定する。**

この食い違いに気づかず「`confirmed_board is None` かつ CHAIN 中」を条件に
実装したところ、実運用での発火率が **0%** だった (36: 0/1141行、52: 0/1136行、
c100: 0/908行、すべて `board_provenance == "observed"`)。

## 取得経路 (file:line)

| 場所 | 内容 |
|---|---|
| `src/recognition_pipeline.py:430` | `SideResult.estimated_board` — 起点盤面から前進させた予測。CHAIN/GRAVITY_SETTLE 中のみ非 None |
| `src/recognition_pipeline.py:447` | `SideResult.board_provenance` — `"observed"` / `"chain_estimate"` / `"chain_estimate_low_confidence"` / `"chain_estimate_stale_hold"` |
| `src/recognition_pipeline.py:5899` | `_compute_chain_estimate()` — 毎フレーム呼ばれる算出本体 (private) |
| `src/recognition_pipeline.py:5779` 付近 | `_start_chain_estimate()` — 起点盤面の simulate 結果 (`ChainResult`) を保持。`low_confidence` 判定の実体 |
| `src/chain.py:163` | `ChainSimulator.simulate()` — 最終結果のみを返す |

- **相手側の予測**は同一フレームで `result.p1.estimated_board` /
  `result.p2.estimated_board` をクロス参照するだけでよい (main loop で両方取得済み)
- 予測連鎖数・予測おじゃま量が要る場合は `ChainResult.chain_count` 等が使える可能性
  (`src/chain.py`、未検証)
- `board_provenance == "chain_estimate_low_confidence"` は起点盤面の物理予測
  chain_count と score 由来 chain_count が食い違うケース。既存コードに
  「取り扱い注意」と明記あり

## 収集側に入っているもの (無害・既定OFF)

`--enable-chain-estimate-recording` フラグが `scripts/collect_boards_lean.py` に
入っているが、上記の誤った条件 (`confirmed_board is None`) で実装されており
**実効0%**。既定OFF・無害のため放置してある。使う予定はない
(user 判断により収集データには記録しないため)。

生データ (発火率0%の実測): `data/verify/chain_estimate_recording_2026-08-18/`

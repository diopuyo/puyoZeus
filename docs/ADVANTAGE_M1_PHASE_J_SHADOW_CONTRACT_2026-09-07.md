# M1 Phase J shadow比較契約

作成: 2026-09-07 12:30 JST。M1 shadow配線・結果確認前に固定する。

## 目的

現行旧本番とM1を同じ映像上で比較する。ただし異なる入力・時点・HOLDを混ぜて一つの精度値にせず、
同値比較できる行と運用差を分離する。shadow出力は画面へ採用せず、production設定も変更しない。

## 同値比較できる行

- 同一game generation、同一`available_ms`、同一side pairである。
- 両者が同じSTABLE確定盤面pairとNEXT/DNEXTを参照し、盤面hashが一致する。
- M1はcurrent confirmed入力。projected-stateは別stratumとし混ぜない。
- future event、勝者表示、着弾後の訂正を過去時点へ戻さない。
- 旧本番の物理override前のモデル値、override後表示値、M1値を別列で保存する。

## HOLD・unsupported・fault

- M1 unsupportedは同じM0をbit-exact出力した行として集計する。
- M1 integrity fault、recognition unavailable、stale、物理未解決はHOLD理由を保存し、旧本番値で
  M1列を穴埋めしない。coverage分母から黙って除外せず、reason別件数と継続時間を出す。
- 学習時全体隔離と実運転時点以降隔離の両フラグを保存し、4区分で精度・差分を層別する。

## projected-state

- `confirmed_board`と`physics_projected`を同一モデル入力として扱わない。
- 現在のprojected-stateは物理DTO/Adapterまでで、正式M1モデル・較正未採用。量、受け側、初回着弾、
  leftover、候補数は監査表示できるが、M1勝率は更新しない。
- projected-state正式モデル採用後もcurrentとprojectedを別予測として保存し、一つの確定勝率へ潰さない。

## 比較出力

- paired rows、coverage、HOLD/fault/stale秒数。
- log loss、Brier、AUC、ECE、左右反転、fold/source/game別差。
- 勝者確率10%/20%未満、新規重大誤り、解消重大誤り。
- `incoming >= 12/30`、ledger usable、学習隔離×実運転隔離、連鎖中、projected適用候補を層別する。
- 実時間P50/P95/P99、更新頻度、最長HOLD。

## 採否

shadow単体でproduction昇格しない。事前登録済みモデルgate、重大誤り非増加、入力parity、
ユーザー動画レビュー、Formal100または再事前登録した外部評価をすべて通過後に別判断する。


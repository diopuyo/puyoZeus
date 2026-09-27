# Phase J review/live death equivalence監査

## 結論

最終M1レビュー用因果gateは、Phase J正式live death observerと非同値である。
レビューgateの本番流用はNO-GOとし、正式runner内でbuildへ束縛したlive observerだけを
Phase Jの死亡状態の根拠とする。

正式判定は
`NO_GO_REVIEW_GATE_REUSE__FORMAL_LIVE_OBSERVER_IS_AUTHORITY`。

## 比較範囲

- 対象: c8、c28、c42、c76、c138の最終V6レビュー5窓
- 元映像: 各review manifestの動画SHAと現物SHAを一致確認
- 時刻: reviewの絶対時刻窓をlive sidecarが全件包含
- live観測: 30Hz、4,560 frame、9,120 side、decode完走
- review側: manifest記録のreview gateコードSHAと現物を一致確認
- live側: 5 runそれぞれについてbuild ID、入力、sidecar、認識設定、コード資産、
  event manifest、validation、`COMPLETE`を相互検証

旧review buildと今回のlive buildは別物であり、同一build parityとは扱わない。
両者を独立にhash固定した、同一元映像・同一絶対時刻での状態機械比較である。

## 結果

- candidate: 一致6、reviewのみ0、liveのみ1
- release: 一致5、reviewのみ1、liveのみ2
- live death confirmation: 0
- 差分はc8だけ。c28/c42/c76は候補・解除が各1件一致し、c138は両者0件

入力同一性、decode、範囲、観測周波数、review gateコード、live run build束縛は全件PASS。
したがって差分は入力破損ではなく、静的STABLE候補を扱うreview gateと、
遷移起点・1.5秒確認・生存解除・境界解決を持つlive observerの仕様差である。

## build-bound live run

- c8: `build-ba22669893cc486013facb7a7cfe67b632ba648ea59f5169e996285b848b6085`
- c28: `build-552800ea9cdcaee0050f5402115055d67861bab62b229682f8b6bda3990fabb9`
- c42: `build-110e87df81aa1eba44c5c2ec5cdf44dc0cdd4d1da32dad0e11b33bc9982c94e5`
- c76: `build-932c7744b91395946218428d0cd3eb8d709ee0aba4ae90fbe8452c512d447a08`
- c138: `build-8c615c5debf7e4b8e4e0f18d8012bfcac31b40181deb70fc080c59c5fb8c7e8f`

収集正本は
`data/verify/phase_j_death_build_bound_5windows_2026-09-06_v1/`、
集約監査正本は
`data/verify/phase_j_death_review_live_equivalence_build_bound_2026-09-06_v5/`。

## 終局hard contract

死亡候補は予測HOLDの入力であり、確定勝敗ではない。0%/100%を出せるのは
`visual_result_logo_bilateral_2x2`だけとする。統計モデルは非終端の較正予測に限定する。

M1真OOF/外部レビューrendererは、dense death timelineの有無に関係なく両者ロゴ限定へ修正した。
Phase J reducer/runtime/display projectorへallowlist済みterminal evidenceを接続し、全job無効化、
100対0の1秒表示、result遷移、境界resetを合成fixtureで検収した。

### terminal方向偏りの診断と修正

最初の57試合監査は27/57を検出したが、内訳は1P 27/28、2P 0/29だった。
総検出数だけの暫定gateは左右偏りを見逃したため、これを監査器の不備として修正した。

実フレーム上の2P勝利ロゴ上端はy=150付近で、従来のreview用2P ROIはy=200開始だった。
さらに`match_end_yatta.png`は520×370で盤面背景を含む。対処は次の2段階とした。

- review検出用2P ROIを`(1150, 100, 700, 600)`へ拡張。V2は52/57、
  1P 27/28、2P 25/29、方向不一致0、57負例の誤検出0。
- Phase J既定adapterだけが元asset内`(x=60, y=210, w=460, h=160)`の
  背景非依存ロゴを使う。assetは加工保存せず、shape不一致はfail-closed。

最終V3は終端451 frame×57試合と直前450 frame×57試合を走査し、57/57、
1P 28/28、2P 29/29、方向不一致0、負例誤検出0、allowlist/digest違反0でPASSした。
方向別coverageとWilson 95%区間重なりをhard gateへ追加し、総数だけでの合格を禁止した。
成果物は`data/verify/phase_j_terminal_adapter_full57_logo_crop_2026-09-06_v3/`。

既知13件の正例再生V2も13/13 event化、勝敗方向不一致0、allowlist外0、digest未束縛0でPASS。
成果物は`data/verify/phase_j_terminal_adapter_positive_replay_logo_crop_2026-09-06_v2/`。

本番capture adapterへの走査接続と実captureでの長時間soakは未実施であり、
Phase J本番採用前の独立gateとして残す。保存動画57試合のPASSを本番接続完了とは扱わない。

関連回帰553件はPASS。全pytestは8,549件PASS、16件skip、1件deselect、失敗0、
既存警告251件（468.71秒）だった。

## 非変更範囲

Formal100、hidden reserve、既存重み、採用フラグ、`src/production_config.py`は変更していない。

# E18: 仮想着弾への打ち返し応手

## 固定条件

- 基準はE17で採用候補となったE15＋②。`models/exchange_event_v3`とlive countを使う。
- `production_config.py`は変更せず、①・③・③′は有効にしない。
- 新フラグ`--landing-counter-response`は描画・記録再生とも既定OFF。
- 採点前の登録は`logs/e18/PROTOCOL.json`。q log loss ≤0.5313、zenchi一致率 ≥80.10%、誤発火 ≤1/28。
- 場面は2612〜2614.08秒の62行。平均は基準の未丸め値0.2702919012720453より大きく、前半31行平均 < 後半31行平均を要求する。
- 3動画とzenchi全編・指定場面はE17と同じ`logs/e16/records/`を再生し、認識・モデル・母数を固定する。CV再学習は行わない。

## レビュー動画

`scripts/render_e18_review_20260928.py`はE14の保存コマンドから出力先とモデルを変更し、
`--exchange-event-live-count --exchange-event-death-guard`だけを追加する。
新しい打ち返しフラグは指定しない。review-data-panelとCSV、30秒の暖機、開始2579.066秒・終了2777.333秒を継承する。
元映像の音声を区間に合わせてAACで保存し、スマホ版は横1280・CRF28・映像maxrate 1000kbps・音声128kbps。
出力サイズ30,000,000 bytes以下、映像フレーム数とCSV行数の一致、音声ストリームの存在を検証する。
生成条件・ファイルサイズは`logs/review_zenchi_g41_43_e15b/status.json`と`complete.json`に保存する。

## 応手の計算と表示

`ExchangeLandingProjection._counter_projection`が予測層の二つの仮定を作る。

1. 無応手の受け量は従来の`_incoming`。双方の既発火火力の相殺と、既着地量の控除を済ませた値。
2. 残り手数は従来の`_hands`。相手連鎖の残り演出時間から自側の操作不能時間を引き、着弾前の最後の設置を含む。
3. 最新STABLE確定盤面とNEXTを起点に、既存`future_send` / `near_future_fire_power`を使う。Kは既知NEXT2手を除いた内部表現で、探索できる総手数は残り手数と一致する。
4. 連鎖中や終了直後に確定盤面が古い場合は、観測と矛盾しない完走予測盤面を起点とし、既発火の火力を再加算しない。完走盤面が不明なら追加応手は欠測として無応手を選ぶ。
5. 応手量は整数個に切り捨て、`cancel_own_pending_then_send_surplus`で純受け量を相殺する。余剰は相手への受け量として同じ仮想着弾評価へ渡す。1ターン30個の着弾上限は既存関数を再用する。
6. 同じ確定盤面ペアへ、無応手の受け量／応手相殺後の受け量をそれぞれ仮想投下してG_feを得る。NF探索の不確かな将来設置・消去配置を現在盤面として確定しない。
7. 応手量が純受け量以上のときだけ応手ありG_feを選び、それ以外は無応手G_fe。選んだ値とS3の従来logit平均を維持し、回避不能死の証明条件・保持は変えない。

両G_feは予測層で、現在の確定盤面G_feを上書きしない。E16の現在層復帰・追加のlogit合成は有効にしない。
イベントDTOには`gfe_no_response_p1`、`gfe_response_p1`、`response_selected`、`response_layer`、応手量・純受け量・余剰・確定入力時刻を保存する。
レビューCSVの対応列は`p1_landing_no_response`、`p1_landing_response`、`landing_response_selected`、`landing_response_layer`と各側`response_*`。
固定入力評価でも全更新の`counter_response.csv`を保存し、現在G_feと二つの予測G_feを区別する。

## 再実行

```bash
PYTHONPATH=. python -m scripts.render_e18_review_20260928
PYTHONPATH=. python -m scripts.run_e18_counter_response_20260928
PYTHONPATH=. python -m scripts.run_e18_counter_response_20260928 --source review --off
```

結果は`logs/e18/METRICS.json`。OFFのreviewはE17組合せのNPZ・イベントJSONLをバイト単位で照合する。

## 結果

| 条件 | q log loss（6,526行・4試合） | zenchi一致（8,333行） | 誤発火 | 場面平均（62行） | 前半31行 → 後半31行 |
|---|---:|---:|---:|---:|---:|
| E15＋② | 0.529266 | 6,700 / 8,333（80.4032%） | 1 / 28 | 0.270292 | 0.302376 → 0.238207 |
| ＋打ち返し応手 | 0.512195 | 6,120 / 8,333（73.4429%） | 1 / 28 | 0.636692 | 0.598468 → 0.674916 |

**不採用**。q・誤発火・場面2条件は合格したが、zenchi一致率が80.10%を下回った。
採用候補はE15＋②のまま。新しい応手フラグと①・③・③′は既定OFFを維持し、`production_config.py`は変更していない。

- 音声付きレビュー: `logs/review_zenchi_g41_43_e15b/overlay.mp4`、360,004,917 bytes、198.2667秒。
- スマホ版: `D:/puyo_analyzer/videos/review/zenchi_g41-43_e15b_mobile.mp4`、21,787,507 bytes、1280幅・30fps・AAC。
- レビューCSV: 同ディレクトリ`review_data.csv`、8,569,250 bytes、5,948行（映像5,948フレームと一致）。
- 新規生成動画の指定62行の採用勝率平均も0.2702919012720453で、固定再生の基準と一致した。
- 回帰検証: 390 passed / 1 skipped。OFFのNPZ・イベントJSONLはバイト一致、診断JSONも一致。
- 応手ONのレビューDTO・CSV・表示NPZを`logs/e18/on/review/`へ保存し、各入力の完了記録と採点結果を同時にコミットする。

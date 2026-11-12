# 中盤補正の設定

`midgame_board_layer.json` はPC2の2026-11-11仕様v2を保存し、文中の値を
`cap_logit=0.25`、`enter_sec=0.5`、`exit_sec=1.0` として機械読み取り用に追記したもの。
`source_sha256` は追記前の仕様書のSHA256。係数は `shift` から直接読む。

読み込み時は `src/midgame_board_layer.py` の `MODEL_SHA256` とファイル全体を照合し、
不一致・欠損時は警告を記録して層を無効にする。
数値を差し替える場合は、JSONと照合用SHAを組で更新する。上限と時間はコード中に置かない。

既定は `src/production_config.py` の `DISTRIBUTION_MIDGAME_BOARD_LAYER=False`。
環境変数 `PUYO_MIDGAME_BOARD_LAYER=1` で有効にする。温度較正が前提。
撃ち返しの1.5秒窓では中盤補正と退出時の残りも0にする。
それ以外の退出では最後の対象行の補正を、退出行の時刻から指数減衰させる。
研究のベクトルと再現上の注意は `tests/fixtures/midgame_board/README.md` を参照。

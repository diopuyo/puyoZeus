# 中盤の盤面補正 v2 の試験ベクトル

PC2提供の260行。vectors.jsonは受領したファイルを変更せず保存した。
SHA256: d303c55d566278ee4a67a120fa1032fe084fef7a6af8086d483369031a6da221

`tests/test_midgame_board_layer.py` はこのファイルを既定で検査する。
別のベクトルを試す場合は `PUYO_MIDGAME_TEST_VECTORS` にパスを指定する。
仕様・件数・SHAが変わったときは、提供側の新しい定義に合わせて試験も更新する。

これは全行時系列の抜粋であり、各行の `mixing_state` は研究で全行を通した後の状態。
盤からの色数・凹凸・対象条件を再計算し、付属状態から混ぜ方と最終確率を再計算する。
状態遷移そのもの、因果的な盤の持ち越し、試合境界、再推論は別の単体試験で検証する。

研究の `counter_fire_window_research` はどちらかの連鎖開始から1.5秒という近似。
製品では既存 `CounterFireShift` が記録した撃ち返し開始の窓を参照する。
この違いは仕様v2の `counter_fire_window_note` に記載されている。

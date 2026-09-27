機器別の色較正JSONの保存先。ファイル名は `sha256(機器名:index).json`。
DirectShowSourceのDeviceConfig.calibration_pathがパスを返す。
`python -m scripts.run_live_pipeline_20260928 --input-config config/input.json` で指定するJSON例:

```json
{"name":"OBS Virtual Camera","index":0,"verification_only":true}
```

起動時・設定ファイルの機器変更時・入力喪失後の復帰時にウォームアップする。
設定は1秒ごとに再読込する。機器名だけで自動選択せず、名前とindexの組を機器識別子にする。
同一カメラ内で入力を切り替える場合も、設定のnameを入力ごとに変更する。

実対戦画面のSTABLE確定ぷよから既存OnlineHsvCalibratorの信頼条件を使って採取する。
4色それぞれが既存is_readyに達したら自動終了。通常の認識器では50サンプル/色。
保存済みprofileがある次回は12サンプル/色でHSV範囲の重なりを確認する。
色構成または範囲が変わった場合は50サンプル/色まで採取し直す。
4色未満・ぷよ画面なしでは終了せず、勝率も公開しない。

既定の `verification_only:true` は検証結果を保存し、HSV範囲を変更しない。
B5の同時刻差を色境界の誤読と断定できないため、補正の必要性が確認されるまでこの設定を使う。
`false` を明示すると収束した範囲を適用し、補正前の認識履歴をリセットする。
補正適用の認識精度向上は保証していない（B4では改善なし）。

完了したprofileだけを原子的に保存する。破損・機器不一致profileは新規採取へ戻る。
配信DTOの `display.input_status` は `verifying/no_puyo_screen/calibrating/ready`、
`calibration_progress` は0〜100。較正中は全勝率を未取得にし、最小HTMLに日本語状態を表示する。

# E5 細部決定

gzip JSONL v1で全update入力・旧表示の補助入力・D静止入力を記録し、再生では終了信号・着地・M0・G_fe/S1/S3を再計算、Dは盤面と会計入力が一致する場合のみ保存特徴を再利用する；鮮度はON=display_p1／OFF=adv_raw_lastの厳密同値、未評価NaN同士は未更新として全フレーム分母に残す。

記録: 既存ONコマンドに `--exchange-event-record <path>` を追加。拡張子に関係なくgzip JSONLを保存する。未完了記録は再生時に拒否する。

再生例:
```bash
python -m scripts.replay_exchange_event_20260926 \
  logs/e5/renders/q_7gc4TgFig/on/inputs.jsonl.gz \
  --out logs/e5/replay/renders/q_7gc4TgFig/on \
  --compare logs/e5/renders/q_7gc4TgFig/on
python -m scripts.aggregate_e4_exchange_eval_20260926 --out logs/e5/replay
```

記録の `update` は認識両側の状態・確定盤面・得点・NEXT/DNEXT・NEXTスライド・連鎖通知、会計のD入力と累積着弾、OCR確定通知、時刻・試合ID・式累積・表示得点・式表示を保持する。終了判定と着地はこれらから再計算する。`static` は入力キー別のD、`static_call` は各呼出しの経過秒・M0・視点を保持する。

`display` は旧評価器の生値・settledフラグ等の補助列と、イベント評価未確定時に使う既存表示のフォールバック入力のみ。イベント由来の表示値・source・eventsは保存出力を転記せず再生成する。`--model-dir` で評価モデルを差し替え可能。

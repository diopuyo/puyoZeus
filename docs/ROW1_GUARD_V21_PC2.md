# row1 v2.1 PC2短区間確認

既定OFF・未採用。段4・全動画再走はしない。elと同じ短区間2本、OFF→ON直列のみ。
手順/資産/種/動画SHA/資源管理は `ROW1_GUARD_V2_PC2.md` を継承する。
旧el成果・コードは変更せず、新しい`row1_v21_short`作業/出力ディレクトリを使う。

## 受取り

`share/live_redesign/live_row1guard_v21.bundle` と隣のreceiptを照合する。
前提コミットはd9073afb5bcbe80b45e75770750d9912aa813626。

```bash
git -C "$repo" bundle verify "$bundle"
git -C "$repo" fetch "$bundle" HEAD:refs/heads/row1-v21-short-review
git -C "$repo" worktree add --detach "$code" refs/heads/row1-v21-short-review
```

elのentry/controllerと設定を新ディレクトリへ複製し、code/outputのパスだけ変更。
各条件に元の較正種を新規複製する。モデル/資産SHA、import元、実効設定を保存。
OFF/ON差はlive_row1_guardだけ。B3同期（b3_async=false、elと同じ）を維持。

## 分解記録の追加（必須）

旧entryの `import observe_internal` を次へ置き換える。
新codeをsys.path先頭に挿入した後、従来と同じinstallを一度だけ呼ぶ。
旧observe_internalはsettled引数を受け取れないため併用しない。

```python
from scripts import observe_row1_v21 as observe_internal
observe_internal.install(RecognitionPipeline, W)
```

`scripts/observe_row1_v21.py` が認識原票に加えて次を保存する。

- internal_history.jsonl: ガード前後/raw/合図/着地、ctx/prev_confirmed/prev_stable、
  inferred/chain/landing、tsumo_count/pending_tsumo。
- b3_inputs.jsonl: 実際にB3へ渡る元盤、イベント、採用NEXT行、手数、epoch。
- b3_features.jsonl: モデルへ渡した全特徴列（時刻列・履歴列を含む）。
- 既存のb3_updates/display/death/作動票も保持。schema version=2、guard_revision=2.1を確認。

両workerのentryがこのinstallを通ることを確認する。認識側はb3_inputs、
評価側はb3_featuresを生成する。片方がない場合は分解完了にしない。

## 区間と採点

early 0–200秒、game36は読取2199–2375秒・採点2200秒から。
30Hz、realtime=false、監査ON、R1照合OFF。種とスレッド数/BelowNormalまたはnice19はel通り。

1. 新OFFと保存el/OFFの公開盤・状態・CNN・表示を照合。新optionalフィールド以外の差を記録。
2. 新OFF/ONのctx、prev_confirmed、prev_stable、置き診断、手数を全件照合する。
   続いてb3_inputsの盤/イベント/NEXT、b3_featuresの全列、raw予測、表示を比較。
   NaN同士は一致、壁時計の性能値は対象外。最初の差とf70721/f70734を列別に保存。
   公開盤だけが異なり、これら内部/評価入力の差は0であることが狙い。
3. f70734の表示が両方31.691207%相当で一致するか。
   f70721のraw予測も31.690366235%相当で一致するか。
4. 対象外セル差0を全座標で確認。#8 f69864–69930はOFF赤67/67、ON非空0/67を確認。
5. 緑f70778±1→ON初公開f70779（保存入力再生の予測）を確認。
   f70772のNEXT、f70779のsettledと公開を追う。8コマ超なら不合格。
   紫f68682±1→f68708は既存遅れとして別記。video94未測定を合格に含めない。

同じコード/設定で再現しなければ、値を合わせる閾値調整はせず差の原票を残す。
短区間結果を新doneへSUMMARY/DETAILS/comparisonと原票一式で納品し終了。
動画は今回取得分のみ削除し削除票を残す。push・inbox投入・段4実行はしない。

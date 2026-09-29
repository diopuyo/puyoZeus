# cycle65 対整合ガード (enable_next_recolor_pair_guard) 結果 (2026-09-30)

## 実装 (既定OFF、本番未採用、src/production_config.py 不変)
- `src/next_recolor_pair_guard.py` (純関数): 観測した新規2セルの色 (不明は無視・おじゃまは証拠として扱う) が
  補正に使う対の多重集合に含まれるか判定。従来の対 (queue[-2]、1件なら[-1]) → 直近3件 (RECOLOR_HISTORY_LOOKBACK)
  の古い順→新しい側の順で整合する対を探す。結果 accepted / substituted / kept_observed。
- `src/recognition_pipeline.py`: `enable_next_recolor_pair_guard` を __init__ / load_default に追加、cycle65 の
  infer_placement 直前で照合 (kept_observed は infer_placement を呼ばず観測色のまま)。監査 `next_recolor_guard_counts` / `next_recolor_guard_log`。
- `scripts/visualize_advantage_overlay.py`: `--next-recolor-pair-guard` (ON時 `next_recolor_pair_guard.json` を出力先に保存)。
- 採用時の登録先: production_config の PLACEMENT_RECONCILE_ADOPTED と同型の別バケット (RECOGNITION_ADOPTED は
  load_default kwargs へ機械変換され AST 固定テストで凍結されるため不可、フラグ名 enable_* とCLI名が一致するので
  変換自体は通るが、凍結テストの更新が必要)。**本タスクでは production_config を触っていない**。
- 撃ち合い評価の描画CLIが `--production-exchange-event` で追加するフラグ列 (src/exchange_event_cli.py) は未変更。

## 単体テスト (段1): tests/test_next_recolor_pair_guard.py 13件 pass
q第14試合の古いキュー (履歴に一致なし→観測維持 / 履歴に一致あり→差替) / 正しい補正はそのまま (accepted) /
不明観測は無視して従来対で補正 / 多重集合 / 探索範囲上限 / おじゃま観測は証拠扱い / cycle65 経路の配線
(既定OFFは従来の対 (queue[-2]) を渡し監査は0件、ON は差替・観測維持・受理の3通り)。
関連 451+196 件 (test_recognition_pipeline / production_config / exchange_event_production /
production_dependency_contract 等) も pass。

## 端から端の確認 (q第14試合 852-886秒、実フラグ OFF vs ON、1P確認盤面 (5,2)(6,2)(1,5)(2,5)(3,5))
- OFF: 883.8 から (5,2)(6,2)=(2,2)=青青 (誤) → 884.333 まで残存 (診断書と一致)
- ON: 883.8 から (3,1)=緑赤 (正解) 。監査 substituted 1 件 (used 青青 → chosen 赤緑)。
- 幻おじゃま (1,5)(2,5)(3,5) は別経路 (答え合わせ pending の陳腐化) で ON でも残る (本修正の対象外)。
- 誤確定 883.967/884.267 の消滅は E36b 再生 (下記、未実行) で確認する。診断書の反事実では (5,2)(6,2) のみ修正で2ラッチ消滅。

## 測定 (段2: 影子測定) 4動画・認識は本番構成のまま (フラグOFF) で再生し、cycle65 呼出ごとに ON の判断を併記
- 真値 = 書込 0.1-0.5 秒後 (30fps換算 3-15 フレーム) の「生CNN==生HSV かつ有色」最頻値 (診断 c65_truth と同基準)。
  真値が取れないセルは分母から除外 (下表 truth cells)。
- OFF=従来 (infer_placement の出力が新規2セルに書く色)、ON=ガード判断後の色 (kept_observed は観測色)。

| 動画 (区間) | cycle65 呼出 | 新規セル | 真値あり | OFF 正解 | ON 正解 | ON悪化 (OFF正/ON誤) | accepted / substituted / kept |
|---|---|---|---|---|---|---|---|
| q (0-900s) | 24 | 48 | 48 | 21 (44%) | 48 (100%) | 0 | 3 / 8 / 13 |
| fcXG83vInDY (0-900s) | 20 | 40 | 40 | 20 (50%) | 39 (98%) | 0 | 4 / 6 / 10 |
| mia8KCjr52g (0-900s) | 237 | 474 | 434 | 198 (46%) | 434 (100%) | 0 | 64 / 117 / 56 |
| zenchi (2580.6-3427.2s) | 31 | 62 | 62 | 30 (48%) | 60 (97%) | 1 | 6 / 13 / 12 |
| 合計 | 312 | 624 | 584 | 269 (46%) | 581 (99.5%) | 1 | 77 / 144 / 91 |

- ON悪化 1 件: zenchi 3155.75秒 1P (7,5): 観測 9 (おじゃま誤読)・真値 緑、OFF は緑に直すが ON は観測 9 のまま。
  同様に「観測におじゃまを含む」9 呼出のうち 8 件は本物のおじゃま (真値=観測) で、おじゃまを無視する案は 8 件を潰すので不採用 (テストで固定)。
- **注意 (循環性)**: 「観測」も「真値」も同じ CNN 由来なので ON の正解率は有利に出る。独立の証拠は HSV 一致と q第14試合の実画面 (診断書)。
  ON 正解 = 観測色 と一致する構造 (ガードは観測と整合する色だけ通す) なので、これは「補正が観測を壊さない」ことの確認であり、
  「観測が実際に常に正しい」ことの証明ではない。 OFF 側が 54% で誤ることは、キューがずれた時の書込が観測 (生CNN==HSV) と衝突する事実の実測。
- mia は呼出 237 回 (q の約10倍) と多く、キューのずれが日常的。 全動画で 「ON悪化 1 / 584 セル」。
- 生成物: logs/c65_guard/{calls_*.jsonl,measure.json,window_{off,on}.jsonl}、再現: `python scripts/measure_c65_guard.py run|window|analyze`。

## 全長 ON 再生成 (準備のみ・未起動)
`scripts/run_c65_guard_full.py` (collect → E36b 構成で replay → E36b 門で report) と
`logs/c65_guard/launch_full.sh` (3レーン収集: [zenchi,review] [q,mia] [fc] → 再生並列1 → 採点)。
出力は logs/c65_guard/full と logs/c65_guard/e36b_on に隔離、既存 logs 不変。
起動: `MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/c65_guard/launch_full.sh`
採点: q log loss <= .5080405 (母数 6,526)、zenchi hits >= 7671 (母数 8,333)、誤発火 <= 1/35、bound_false/unresolved 0、
zenchi 3:00 場面 <= 2766.0秒 (report_e36b の門)。第14試合の誤確定有無は SUMMARY.json の game14_false_times。
### 所要見積もり (範囲。単位=動画1本あたりの実時間、3並列時)
- 収集: 下限 = 今回の影子測定 (観測撮影なし) q/fc/mia 各約 52 分 (900 秒/約 0.3 動画秒毎秒)、zenchi (877 秒) 約 45-75 分。
  上限 = R1b の実収集 (観測画像窓を撮る) q 109 分 / fc 112 分 / zenchi 138 分 / mia 89 分 / review 約 42 分 (合計壁時計 3h18m)。
  レーン構成で 約 2.3-3.5 時間。
- 再生 (E36b 構成、並列1、各記録1プロセス): 実測 worker 合計 886 秒 (q 205 / fc 240 / mia 104 / review 50 / zenchi 287) + 起動、
  E36b 実壁時計 約 18-20 分。
- 採点 数分。 合計 約 2.7-4 時間 (推定、実測は起動後に更新)。

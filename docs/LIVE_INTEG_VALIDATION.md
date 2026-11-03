結論: m + ① 色補正buffer + ② ONNX 2スレッド非スピン + ⑤ 変化マスだけCNN を統合。全フラグ既定OFF。品質門(同一画素の再投入)で全件一致、速度は ABA で認識 約9%短縮。本番採用・push・配布bundleはしていない。

## 1. ブランチ・コミット
worktree: <work>/wt_live_integ / ブランチ claude/live-integ-20261013 (45c41c11 から)。既存の worktree・ブランチ・ログは未変更。
- 1001bd6e feat(live): 画素完全一致の盤面CNN差分読みを既定OFFで追加   (= a8dc6126)
- 0410a77b test(live): 差分認識の比較台と55分試験の隔離環境を追加      (= a0dbfe7e)
- a4ece43a docs(live): 差分読みの短区間結果と未達の品質門を記録         (= 3fb85cf5)
- cd7213c3 test(live): 差分CNNの55分比較と実機録画の検証結果を記録     (= 652c5a22)
- 95408a25 test(live): 統合版の品質門台・段別速度・統合フラグ試験・証拠を追加

## 2. 衝突の解き方
- src/phase_j/launcher.py (UserConfig): m側の `live_public_match_number` と ⑤側の `live_delta_cells` / `live_cnn_dedup` が同じ位置への追記。両方残した (m側3行 + ⑤側2行、順序は m→⑤)。
- src/phase_j/live_speed.py: import と SPEED_DEFAULTS が同じ位置への追記。m側の `LIVE_PUBLIC_MATCH_NUMBER_DEFAULT` / `live_color_buffer` / `live_onnx_parallel` / `live_public_match_number` を全部残し、⑤側の `LIVE_DELTA_CELLS_DEFAULT` / `LIVE_CNN_DEDUP_DEFAULT` / 2キーを追加。公開試合番号の分離は無傷。
  意味確認: configure() が SPEED_DEFAULTS の全キーを `PUYO_<KEY>` 環境変数へ配るので、新2キーも自動で PUYO_LIVE_DELTA_CELLS / PUYO_LIVE_CNN_DEDUP になり、⑤の実装が読む名前と一致。
- scripts/bench_live_delta.py: modify/delete 衝突 (本ファイルは対象外の ee60d438 で追加され、a0dbfe7e で変更される。m側には無い)。a0dbfe7e 時点の完成版を採用 (ee60d438 自体は取り込んでいない)。
- production_config.py / recognition_pipeline.py / patch_classifier.py は自動合成で衝突なし (⑤は `read_boards` 呼出しの1箇所と定数2つのみ)。

## 3. テスト
範囲: tests/phase_j (playwright依存の9ファイルは greenlet が読み込めず除外) + tests/test_live_*.py + cnn_parallel_runtime / launcher* / production_config / patch_classifier / recognition_pipeline / board_recognition_pipeline / b3_live_config。同梱Python 3.12 (torch入り) + 系統Pythonのpytest。BelowNormal。
- 2,190 件: 2,157 合格 / 23 失敗 / 10 skip。
- 23 失敗は **基準 45c41c11 (wt_live_par) でも同じ23件が同じように失敗** (照合済: docs/live_integ_evidence/failed_integ.txt と failed_on_base.txt が一致)。原因は環境: ojama_cnn.pt / models/calibration_video01.json が worktree に無い、Windows に無い getloadavg / sched_setaffinity、ショートカット移行。今回の変更とは無関係。
- 新規テストの合格: test_live_delta_cells 20 / test_live_color_buffer 5 / test_cnn_parallel_runtime 1 / test_launcher 32 / test_launcher_env 8。
- 統合用に tests/test_live_integ_flags.py を新設 7 件合格 (4フラグ既定OFF・単一情報源・UserConfig項目・個別ON・非bool拒否)。
- 注意: tests/test_live_b16.py などの子process試験は、試験の起動ラッパーが spawn で自分を再実行し再帰して固まる事故を起こした (ラッパーを `__main__` ガード付きに直して解消。scripts/pytest_belownormal_live.py)。製品の問題ではない。

## 4. 品質門 (判定不変)
台: scripts/live_integ_gate.py / 起動 scripts/run_live_integ_gate.sh。実 live の経路 (取り込み画素 → configured_corrector → live_process.recognize、b3 含む) を1条件=1プロセスで直列実行。BelowNormal、HOLD・空き6GiBを毎コマ確認 (停止なし)。
入力: ① 元動画 video_d04 650秒〜の 900 枚 ② 10/12 録画 (2026-10-12 16-18-17.mp4) の 4,994 枚。どちらも同じ画素を全条件へ投入。
条件: m = 展開した dc24ee44 のソース (全OFF) / off = 統合版・全OFF / cd = 色補正buffer ON + ⑤ ON / cdo = cd + ONNX 2スレッド ON。
照合: コマごとに frame・t_sec・result_bytes (認識結果全体)・pipeline・dropped_before・b3 の各ハッシュ、盤面3種 (raw_cnn / cnn / confirmed) の全セル、両者STABLEコマの盤面、CNN入力1行ごとの logits バイト (入力行ハッシュ → logits)。

| 比較 | 母数 | 通知・結果ハッシュ差 | raw/cnn/確定盤 セル差 | 両者STABLE 盤面セル差 | logits |
|---|---|---|---|---|---|
| 900枚 m→off | 900コマ | 0 | 0/140,400 ×3種 | 0/204,984 (438コマ) | 99,170行 全一致 (m と同数・同一入力) |
| 900枚 m→cd | 900 | 0 | 0/140,400 ×3 | 0/204,984 | 候補の入力行 45,083 すべて基準に存在 (欠落0)・バイト不一致0 |
| 900枚 m→cdo | 900 | 0 | 0/140,400 ×3 | 0/204,984 | 同上 (45,083行, 不一致0) |
| 録画 m→off | 4,994 | 0 | 0/779,064 ×3 | 0/574,236 (1,227コマ) | 562,513行 全一致 |
| 録画 m→cd | 4,994 | 0 | 0/779,064 ×3 | 0/574,236 | 候補 453,610行 欠落0・不一致0 |
| 録画 m→cdo | 4,994 | 0 | 0/779,064 ×3 | 0/574,236 | 同上 |
(off→cd, off→cdo の比較も同一結果。docs/live_integ_evidence/cmp_*.json に全数。)
- off = m と全件一致 = 「全フラグOFFは m と同じ動作」の確認。
- 一意なCNN入力は 900枚で 40,088、録画で 410,184。候補はこれと同数の一意入力を踏み、行数だけ減った (900枚: 99,170→45,083行 = 54.5%減、録画: 562,513→453,610行 = 19.4%減)。同一入力が基準内で異なる logits を返した件数は 0 (自己矛盾0)。
- 測定器の健全性: (a) 負の対照 — 900枚と録画を比べると 900/900 コマの結果が不一致・盤面セルが約7割不一致と検出される。(b) フラグが実際に効いたか — ONNX は intra 1→2・allow_spinning=0 になる、色補正bufferは30コマ中29回再利用 (docs/live_integ_evidence/effect_check.py)、⑤は上の行数減で確認。
- 範囲の注意: これは「保存動画を同一画素で再投入」した門。過去の実機で受理された画素列は保存されていない (LIVE_PAR.md / LIVE_DELTA.md と同じ制約)。色補正は動画コマにも掛けている (補正後の画素で認識)。表示勝率差・決着前の勝者側p・死亡確定時刻の比較、55分の実機連続運転、OBS・仮想カメラは未実施。

## 5. 段別速度 (同じ台・同じ900枚・n=839/条件、暖機60+終端1を除外)
速度試験は logits記録なし。順序 A B A C A B A (A = 全OFF, B = 色補正buffer+⑤, C = B + ONNX 2スレッド)。単位 ms、平均 (P95)。各候補は直前直後のAの平均と比較:
| 段 | A (全OFF) | B 色buffer+⑤ (2回) | C B+ONNX2 |
|---|---|---|---|
| 色補正 | 1.98〜2.05 (P95 2.24〜2.28) | 1.34 / 1.35 (P95 1.51 / 1.52) = 約-33% | 1.36 (P95 1.52) |
| 盤面認識 (read_both_boards) | 13.86 / 14.22 / 14.27 (P95 16.2〜16.4) | 11.84 / 12.19 (P95 14.74 / 14.85) = -14.6% / -14.5% | 11.92 (P95 14.07) = -16.2% |
| 認識全体 (色補正除く) | 29.06 / 29.76 / 29.83 (P95 38.7〜40.0) | 26.86 / 27.54 (P95 37.18 / 37.86) = -7.6% / -7.7% | 27.35 (P95 37.86) = -8.1% |
| 色補正+認識 | 31.0〜31.9 | 28.20 / 28.89 | 28.72 |
- B の短縮は 色補正 約0.65ms + 盤面認識 約2.0ms。P95 は盤面で -1.3〜-1.5ms、認識全体で -1.5〜-2.2ms (平均より控えめ)。
- ONNX 2スレッドは、この台では B に対して追加の短縮が見えない (盤面 11.92 は B の 11.84/12.19 の範囲内)。
- A が 28.46→29.80 と単調に約1.3ms遅くなっており (熱・周波数の可能性、WMIの周波数表示は常に2400で当てにならない)、絶対値より ABA の隣接比較で読むこと。
- 実機fps・capture→評価終了の遅延・CPUコア数・55分の劣化は未測定。

## 6. 成果物
- 台: scripts/live_integ_gate.py / scripts/run_live_integ_gate.sh / scripts/pytest_belownormal_live.py / tests/test_live_integ_flags.py
- 証拠: <work>/wt_live_integ/docs/live_integ_evidence/ (cmp_q900*.json, cmp_qrec*.json, cmp_negative_control.json, speed_summary.json, failed_*.txt, tests_phase_j.xml, effect_check.py)
- 生記録: <work>/wt_live_integ/logs/live_integ/ (git対象外)
- 配布bundle: 未作成。仮想カメラ・OBS・長い試験は未使用。pushなし。

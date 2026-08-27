# Codex 品質精査レポート — 2026-08-24

更新: 2026-08-24 13:29 JST / 作成者: Codex  
対象: Claude Code の既存作業、掛け算式根治、PM100対策、交換会計、関連テスト  
方式: 読み取り専用レビュー。既存ソース・テスト・ログ・共有ハブは変更していない。

## 1. 結論

現時点では、掛け算式根治・PM100対策ともに本番採用へ進めない。

- 現行A+Bは指摘場面を改善する一方、全8区間で誤った `±100` 張り付きが増えた。既定OFF維持が正しい。
- 掛け算式根治には、正しい連鎖段を落とす再現可能な欠陥がある。
- 「偽イベント4条件」はログ採取までで、受け入れ条件の真偽分類が未実施である。
- 8動画バックテストは、説明上の3要素のうちスライド終了設定を実際には有効化していない。
- 全pytestは `2 failed` であり、うち1件はおじゃまダメージの意味論不一致を実値で再現できる。
- 99%→1%問題の根本原因である「時間差の撃ち合いを同じ交換として追跡できない」構造は未修正。計画中の交換エピソード台帳が必要である。

したがって Gate 0 は未完了と判定する。現在のバックテスト完了後、下記Q-01〜Q-04を閉じてから交換台帳の統合へ進むべきである。

## 2. 現在の実行状態

13:29 JST時点:

- 10ケース掛け算式プローブ: 完了。
- 偽イベント用4走行: ログ採取完了。ただし自動分類・合否判定は未完了。
- 8動画×3条件バックテスト:
  - A 修正前ベースライン: 完了。
  - B 最終コード・フラグOFF: 実行中。
  - C 最終コード・フラグON: 未開始。
- PM100全pytest: 完了、`2 failed, 5676 passed, 13 skipped, 1 deselected`。
- Claude Code: 12:57開始。`CLAUDE_TO_CODEX.md` への進捗追記は13:29時点ではまだない。
- write lock: 継続。検証完了まで関連ソースを編集しない。

修正前ベースラインの再現性:

- `off_baseline_result.json`: MD5 `6551D1FF4B691953D9D61BA9BEF84F59`
- 先行 `off_result.json`: MD5 `6551D1FF4B691953D9D61BA9BEF84F59`
- 両者一致。修正前ベースライン再構築は再現できている。

## 3. 優先度付き指摘

### Q-01 [P0 / 採用ブロッカー] 掛け算式累積器が正しい連鎖段を欠落させる

対象:

- `src/score_ocr.py:966-1089` `FormulaStepAccumulator`
- `src/recognition_pipeline.py:6283-6363` 進行中イベント更新
- `src/scoring.py:12-23, 176-186` 公式得点式

原因は2つある。

1. 同一連鎖内の右辺倍率が単調増加するという前提が不正確。
   右辺は `連鎖ボーナス + 連結ボーナス + 色数ボーナス` である。連鎖ボーナスが増えても、同時消し色数・連結数ボーナスが下がれば、次段の右辺は低下できる。
2. 前段と同じ掛け算式が時間を空けて再出現しても、同じ段の再読として無条件に棄却する。段間の消失時間を考慮していない。

読み取り専用の最小再現結果:

```text
same_formula_two_steps:
  observed_steps=1 observed_power=320
  expected_steps=2 expected_power=640

decreasing_multiplier_two_steps:
  observed_steps=1 observed_power=320
  expected_steps=2 expected_power=1920
```

影響:

- `chain_count` と `total_score` が過小になる。
- CHAIN hold、安全弁上限、生成おじゃま量が過小になる。
- 連鎖イベント断片化や未登録送付量の誤差を再発させる。
- 99%→1%問題の修正入力自体が不正確になる。

必要な対応:

- 段の同一性を右辺単調性で決めない。
- 「表示の出現サイクル」「無効/空白区間」「確定時刻」「chain_id」を使う。
- 同一式が段周期後に再出現するケース、倍率が低下するケースを回帰テストへ追加する。
- 暫定 `chain_count/score` は安定した `chain_id` の同じイベントを更新し、別イベントとして再加算しない。

### Q-02 [P1 / 検証ブロッカー] 偽イベント率の合否判定が実装・実行されていない

対象:

- `scripts/_probe_formula_false_event_2026-08-24.py`
- `logs/_probe_formula_false_event_2026-08-24/`

プローブのdocstringは、イベント発生後のスコア上昇で真偽分類する「後段解析」を予定している。しかし実装はイベントとスコアのログ出力だけで、分類器、`SUPPORT_WINDOW`、集計結果、合否判定が存在しない。

`driver_progress.log` の `ALL_DONE` は4走行の終了を示すだけで、「偽イベントを増やしていない」の証明ではない。

必要な対応:

- triggerごとに同一sideのスコア `+40`、連鎖状態、物理イベントを時間窓内で照合する解析器を作る。
- OFF/ONで TP/FP/FN、イベント数、重複イベント数を比較する。
- score OCR欠落時を「偽」と誤分類しないため、判定不能を別区分にする。
- 解析結果JSONと人間レビュー対象フレーム一覧を成果物にする。

### Q-03 [P1 / 検証ブロッカー] 8動画バックテストが意図した全構成を有効化していない

対象:

- `scripts/measure_stable_cell_acc.py:533-539, 662-668`
- `scripts/_driver_formula_fix_backtest_2026-08-24.sh`

複合フラグ `enable_formula_freeze_fix` は次の3フラグをONにすると説明されている。

- `enable_chain_formula_read_verify`
- `enable_formula_chain_count_update`
- `enable_slide_exit_no_min_display`

しかし `enable_slide_exit_no_min_display` は、親フラグ `enable_slide_exit_min_display_guard=True` のときだけ意味を持つ。バックテストドライバは `--enable-slide-exit-min-display-guard` を渡していないため、3つ目は実質no-opである。

影響:

- 8動画ON結果は「実読発火 + 段更新」の検証であり、説明上の完全構成の検証ではない。
- 10ケース/偽イベントプローブは親ガードを明示ONにしており、バックテストと構成が一致しない。

必要な対応:

- 現行結果を捨てず、「式読取2要素のみ」の結果として名前を付け直す。
- 完全構成を別出力先で再走する。
- 少なくとも `formula read verify`、`chain count update`、`slide guard/no-min` を個別A/Bし、寄与と相互作用を分離する。

### Q-04 [P1 / 既存機能不具合] 中央列ほぼ窒息盤面と空盤面のおじゃまダメージが同値

対象:

- `src/indicators_v2.py:4013-4051` `_ignition_headroom_dan`
- `src/indicators_v2.py:4073-4106` `ojama_damage`
- `tests/test_expected_net_damage_step5.py:68-107`

発火点が見つからない盤面では、全6列の平均余裕を使う。このため窒息列だけが危険でも他5列の空きに希釈される。

再現結果 (`ojama_count=48`):

```text
空盤面:
  remaining=4.0, damage=0.05

中央列をrow=2まで積んだ盤面:
  remaining=2.166666..., damage=0.05
```

48個は約8段相当だが、両方とも「受けても無害」帯の0.05になる。全pytestの機能テストも `assert 0.05 > 0.05` で失敗した。

これは単なる古いテストとは断定できない。窒息列の危険が平均で消える実装上の意味論問題がある。修正前に、次のどちらを正式仕様にするか確認が必要である。

- 窒息列・発火点列など最危険列を重視する。
- 実際のおじゃま配分を仮想着弾し、着弾後の死亡・余裕で評価する。

後者が物理則に最も近い。少なくとも「空盤面と窒息一歩手前が同値」は受け入れない品質ゲートを残すべきである。

### Q-05 [P1 / 99%→1%根本原因] 時間差の撃ち合いを1交換として追跡できない

対象:

- `scripts/visualize_advantage_overlay.py:1913-1946`
- `ResolvedExchangeTracker`

現在の決着先読み開始条件は、両者の `chain_event` が同じフレームに存在することを要求する。ユーザー指摘の

`催促対応 → 相手本線 → 自分本線`

は時間差で進行するため、この条件を満たさない場合が多い。さらに安定した `chain_id`、交換ID、暫定値→確定値置換がない。

必要な対応は既存 `PLAN.md` の交換エピソード台帳で正しい。Q-01の段識別問題も同じ `chain_id` 設計の中で解くべきである。

### Q-06 [P1 / 採用不可] 現行PM100 A+Bは全域品質を悪化させる

全8区間結果:

- 1秒150pt以上の急変: `79 → 40` 改善。
- 反転試合: `47 → 34` 改善。
- 決着方向誤り: `10 → 7` 改善。
- `±100` 張り付き: `14.0% → 18.3%` 悪化。
- 生モデルと逆符号の張り付き: `3.3% → 5.2%` 悪化。

局所場面には効くが、確信度ゲートが交換の物理状態やイベントIDを持たず、時間持続だけで確定するため、誤った入力も長く続けば確定してしまう。採用せず既定OFFを維持する。

### Q-07 [P2] capなし並行帳簿は表示用帳簿と保存則が分離している

`pending_*_uncapped` は相殺の架空余剰を減らす有効な診断・暫定対策である。一方、既存 `total_offset` と `total_dropped` はcap済み帳簿側だけを集計し、uncapped側の保存則を直接検査できない。

また `PostChainUnregisteredSentTracker` は、イベント断片を最大値で保持し、会計登録増加と盤面着弾増加を両方差し引く保守的ヒューリスティックである。架空攻撃を抑える代わりに実攻撃を過小評価し得る。

本番の単一情報源にはせず、交換台帳導入まで既定OFFの診断経路として扱う。

### Q-08 [P2] 重要なOCR失敗が無記録で握りつぶされる

`RecognitionPipeline._read_formula_value` は全例外を `None` に変換し、原因・回数を記録しない。fail-safeとして停止しないのはよいが、テンプレ欠落、入力shape異常、実装バグも通常の「観測なし」と区別できない。

必要な対応:

- 低頻度のrate-limited警告または診断カウンタを追加する。
- `object | None` と `getattr` ではなく `FormulaReadResult | None`、`FormulaStep | None` を使用する。
- 本番表示は止めず、timeline dumpへ reject reason と例外カウンタを出す。

### Q-09 [P2] 全pytestと性能ゲートが安定していない

- `test_precise_and_fast_complete_within_generous_timeout`: precise `5.58s > 5.0s`。
- 高負荷実行だったため単独低負荷再試験が必要。
- ただし壁時計5秒の単発assertはマシン負荷に弱い。性能回帰判定は複数回中央値、専用marker、基準比の方が安定する。
- 67 warningsのうち、sklearn 1.8非推奨APIと空slice中央値警告は将来障害・NaN見逃し候補である。

### Q-10 [P2] 変更規模と追跡性が保守性を下げている

現在のworktreeは `37 tracked modified / 371 untracked`。主要overlayは約1,400行増え、単一ファイルに認識、会計、判定、表示、CLI配線が集中している。

これは直ちに既存差分を整理・削除する理由ではない。ユーザー指示どおり上書きしない。その代わり:

- 成果物ごとに入力ファイルhash、実行コマンド、フラグ、出力先をmanifestへ残す。
- 新しい交換会計は独立モジュールで開始する。
- 巨大関数へさらに直接実装せず、純粋コアと薄いadapterに分ける。
- 既存未追跡物を勝手に削除・stashしない。

## 4. Gate 0を閉じるための必須項目

1. 現在の3条件バックテストを完走し、JSON・MD5・ログを固定する。
2. OFF baselineとOFF finalのbit-identicalを確認する。
3. Q-01の2再現テストを追加し、掛け算式段識別を是正する。
4. 偽イベントログの分類器を実装し、OFF/ONのTP/FP/FNを出す。
5. スライド親ガードを含む完全構成と、各要素単独の比較を行う。
6. おじゃまダメージ失敗を仕様判断し、テストと実装を同時に整合させる。
7. 性能テストを低負荷で再実行する。
8. 全pytest成功を確認する。
9. Claudeが結果・変更ファイル・未解決事項を `CLAUDE_TO_CODEX.md` に追記する。

## 5. 交換エピソード実装へ追加すべき品質条件

- 同一 `chain_id` の暫定値と確定値は置換し、二重加算しない。
- 同じ掛け算式が別段で再出現しても段数を落とさない。
- 右辺倍率の低下を新セッションと決めつけない。
- `episode_id` は時間差の応射を同じ交換へ参加させる。
- 発火、段確定、完走、score finalize、相殺、着弾を物理時刻順で処理する。
- 未解決episode中は `±100` hard overrideを禁止する。
- episode終了時に暫定残差、未照合量、重複イベントを0にする。
- おじゃまダメージは実着弾後の窒息列を無視しない。
- timeline dumpに `episode_id / chain_id / provisional / finalized / reconciled / landed / reject_reason` を含める。

## 6. 推奨するClaudeへの追加指示

現在のバックテストを中断せず完了させた後、Gate 1へ進む前に次を行う。

1. 本レポートQ-01〜Q-04を再現し、反証できなければ採用ブロッカーとして登録する。
2. Q-01の失敗テストを先に追加し、式の同一性ではなく表示サイクルとchain_idで段を識別する設計をfable reviewerへ確認させる。
3. 偽イベント後段解析を実装し、`ALL_DONE` と `ACCEPTED` を分離する。
4. バックテスト構成不一致を修正し、既存結果は削除せず別名・別出力先で完全構成を追加測定する。
5. おじゃまダメージはテストだけを書き換えず、物理仕様を決めてから実装とテストを揃える。
6. 現行A+Bを本番ONにしない。
7. 結果を `docs/agent_coordination/CLAUDE_TO_CODEX.md` へ追記する。

## 7. レビュー対象固定情報

- Git branch: `feat/regen148-and-scanner-2026-08-11`
- Git HEAD: `6ee7496861f8`
- `src/score_ocr.py` SHA-256: `C1B7F7CB1928823843570E8A3F4AAFFC479519114209B5FBC79D80D11089B4BE`
- `src/recognition_pipeline.py` SHA-256: `5D3E278349555B94CFCC51DCB85A5037AA4C123776F8FFE751E3C7103539A13F`
- `src/ojama_accounting.py` SHA-256: `114AA5047E83D7F9FA37C1513FDCA2EC053CB0FBC03BD10AC7999A9B3BBFCD52`
- `scripts/visualize_advantage_overlay.py` SHA-256: `867840DB3A6C6A210BA1A002D5B0B2B06914FF40F6FAC2CC0D2238DCB4A077E7`

これらは13:29 JST時点のworktree内容を指す。Claudeの後続変更後はhashを更新して再レビューする。

# row1 ガードの作動ログ計装

以下はc2cecaca時点のv1の記録。v2では解除条件とログが変わる。
現行仕様・結果は `ROW1_GUARD_V2.md`、ログの追加フィールドは `ROW1_GUARD_V2_PC2.md` を参照。

判定本体 `src/row1_guard.py`、既定 OFF、しきい値、公開盤の作り方は変更しない。
`Row1Guard.step/reset` の前後を観測し、元と同じ戻り値を返す。

## 有効化と出力

既存の `--recognition-audit` / `audit_path` を使う。例えば
`output/recognition.npz` に対して `output/recognition.row1_guard.jsonl` を出力する。
`live_process.send_notices` の認識呼び出しにだけ監査コンテキストを設定するため、
DeviceSession 等のラッパーの内側でも同じフレーム番号で記録できる。
監査 OFF ではファイル作成・影の状態更新は行わない。
各行は追記直後にファイルを閉じるので、異常終了前までの行も残る。
同じ監査インスタンスで追記し、新しい監査インスタンスではファイルを初期化する。

ガード OFF・監査 ON は `shadow: true`。側ごとの専用 Row1Guard に同じ入力を渡し、
戻り値は捨てる。実ガードの辞書・カウンタ・公開盤・内部確定盤には触れない。
これは「その OFF 実行で観測した入力に ON のガードを適用した場合」のログであり、
ガード ON によって後続入力自体が変わった場合まで再現する全系の反実仮想ではない。

## 分岐と行の意味

- `floating`: `step` の浮き判定。row1 col2 を即時に空にする。
- `screen_empty`: `step` の 0.5 秒空判定 (`reason=empty_evidence`)。
  `_suppressed_view` の抑制継続も毎回記録する (`suppression_continues`)。
- `spawn_held`: `_spawn_hold` の保留。毎回 1 行。同時出現時だけ row0 col2 も空にする。
- `released`: `_spawn_hold` の連鎖開始 (`chain_started`) または
  時間経過と読取一致 (`hold_elapsed_and_agreed`) による解除。
  `_suppressed_view` の復帰 (`screen_agreed`)、`step` の対象セル空 (`top_empty`)、
  公開盤なしでの `reset` (`no_board`) によるスポーン保留解除も記録する。

1 回の呼び出しで作動する分岐につき 1 行。対象セルが空で保留もなければ行を出さない。
浮き・画面空が先に return した場合、実装通りスポーン保留は解除扱いにしない。

全行に `frame`, `t_sec`, `side` (1P/2P), `kind`, `before`, `hold_id`,
`hold_start`, `release_reason`, `reason`, `shadow` を持たせる。
`before.row0_col2` / `before.row1_col2` は変更前の値であり、
両セルが必ず変更されたという意味ではない。盤面なしの解除では直前候補の値を使う。
`hold_id` は監査ファイル内で一意のスポーン保留 ID。継続と解除は同じ ID を使う。
開始と連鎖解除が同一コマの場合も 1 行で `hold_start=true` と解除理由を記録する。
スポーン保留を伴わない浮き・画面空・復帰では ID は null。

## 比較の再実行

`scripts/verify_row1_actionlog.py` はソース root を切り替え、同じ保存実画素を
実 `send_notices → recognize → Pipeline.update → B3同期表示` 経路へ再投入する。
計装前は `git archive f185c1f6` のソースを使用。各条件は独立プロセスで直列実行する。
引数はソース root、素材 root、認識設定 JSON、出力先、guard (0/1)、audit (0/1)。
cwd は配布版 app、Python は配布版のものを使用する。
`compare <出力の親ディレクトリ>` で `base_g0_a0` ～ `new_g1_a1` を全バイト照合する。

保存実画素3組の冷起動検査であり、eb2全試合の品質再門ではない。
B3の予測・平滑・表示は実行するが、外部Rの死亡証明入力と非同期完了の時刻差は対象外。
score_zero テンプレート2枚が配布資産になく、既存のスキップ動作が前後とも発生する。

## 検証結果 (2026-10-14)

- 基準 `f185c1f6a75033ed4c80fc0bca2802df1a5a71fb`。実画素は
  `D:/puyo_analyzer/share/live_redesign/b_type_frames/` の上記3組、46+41+41=128コマ。
  認識設定は `C:/puyo1_tools/soak/run_k/output/recognition_config.json`、
  Python / cwd は `C:/PuyoLive_20261014u/PuyoLive/python/python.exe` / 同 `app`。
- guard OFF/ON × audit OFF/ON の4条件で、計装前後各128コマ、計512組を全バイト比較。
  認識結果全体・公開盤・内部確定盤・B3表示値とも不一致 **0/512**。
  各条件の非null B3予測は84コマ (合計336)。残り44コマは保持/既定表示を含む。
  同じguard設定で監査だけを切り替えた比較も、4項目すべて不一致0/256。
  機械可読原票は `docs/ROW1_ACTIONLOG_comparison.json`。
- 実画素ログ: video_65 1057.8/1057.9秒の2Pスポーン保留2行と1058.0秒の連鎖解除1行。
  guard ONとOFF影モードでそれぞれ3行、同じ保留IDで対応。残り2組は作動0行。
  浮き・画面空などの網羅は単体試験で確認した。
- 最終の直接関連テスト: **498 passed / 1 skipped / 0 failed**。
  row1既存18件+新規16件は **34/34 passed** (4主分岐、解除理由、重複保留、ID、影モード、監査非干渉)。
- 広域確認: **2169 passed / 16 failed / 4 skipped** (母数2189、最終追加1件の前)。
  失敗16件を `git archive f185c1f6` の独立ソースで同一環境再実行し、16/16同じ失敗を確認。
  内訳: WindowsにないPOSIX APIが11件、既存ショートカット .bat/.vbs期待差が2件、
  配布appをcwdにしたためソース相対パスが見つからない検査が3件。
  相対パスの3件は作業ツリーcwdで再実行し3/3合格。計装による新規失敗は0件。
  広域テスト全件合格とはしていない。
- 詳細ローカル原票: `logs/actionlog_check/{comparison.json,final_tests.txt,related_tests.txt,base_failed_tests.txt}`
  および `base_g*_a*/rows.pickle` / `new_g*_a*/rows.pickle`。原票の大型ファイルはbundle対象外。
- 新規関数は全て50行以内。`git diff --check` 合格。判定本体・採用フラグの変更なし。

# マージン起点の共通化（2026-10-01、既定OFF）

作業場所: `D:/puyo_analyzer/wt_margin`、ブランチ `codex/margin-clock-unify-20261001`。
`src/production_config.py`、収集器、認識器、学習データは変更しない。本番登録は行わない。
このworktreeには `.claude/rules/` が存在しないため、`CLAUDE.md` とユーザー提示規約に従った。

## 1. 3系統の時計と換算経路

### 撃ち合い評価（従来 / OFF）

- `src/exchange_event_overlay.py:666` `_remember`: どちらかの側が STABLE かつ confirmed_board 非Noneになった最初の時刻を `_start` に保存する。空盤面も含む。試合境界の `_reset` で初期化。
- 同ファイル `:465` `_fire`: 発火群の最早 trigger_sec から `_start` を引き、発火前特徴、`score_elapsed_sec`、`CountObservation` に渡す。
- `src/exchange_event_tracker.py:257` が発火時の換算用経過秒を保持し、`:497` / `:548` の確定 / 暫定S3へ渡す。
- `src/exchange_event_features.py:122` は `compute_effective_rate`、`src/exchange_event_count_features.py:54` / `:103` は近未来火力 / 実得点差をレート換算する。
- `src/exchange_event_overlay.py:572` の live count 更新では現在時刻のレートも使う。完走予測・応手・死亡証明へ渡る換算秒もこの経路に依存する。
- 静止G_feの進行度は `src/exchange_event_overlay.py:680` / `src/exchange_event_layers.py:70` / `src/exchange_event_landing.py:371`。マージン専用ではないので今回変更しない。

### 会計（従来 / OFF）

- `src/ojama_accounting.py:1173` `_initialize_match_start`: 最初の **非Noneの得点観測** で起点を設定する。得点0も含む。「最初の加点」ではない。
- `on_state_transition` の MENU 入場で起点をNoneへ戻す。得点大幅減少では `:1100` `_reset_side_boundary` がその時刻を起点にする。明示 `reset(match_start_sec=...)` も可能。
- `:1186` `_elapsed` → `:664` `get_effective_rate` → `compute_effective_rate`。
- 送付量は `:740` `_finalize_chain_end` 内の `:823` で **得点確定処理時刻** の経過秒を取り、`score_to_ojama` に渡す。撃ち合いの発火時刻固定とは、起点以外にサンプリング時点も異なる。

### 収集 / 認識（変更なし）

- `src/recognition_pipeline.py:2069` / `:2070` が側別の `_first_move_sec_1p/_2p` を保持する。
- `:7651` 付近: TSUMO_FALL → STABLE かつ pending ツモをコミットした最初の時刻を側ごとに設定する。
- `:7545` 付近: 処理中の側の起点を引き、`DetectorSignals.elapsed_since_first_move_sec` に渡す。
- `src/state_detectors.py:379` `_effective_threshold` → `:392` `compute_effective_rate(..., from_first_move=True)`。`--margin-time-rate` 有効時のおじゃま落下検出用閾値に使う。
- `scripts/collect_boards_lean.py:2081` がこの用途を説明。側別起点はユーザー仕様「両者で早い方」と一致しないが、学習データに関わるため変更しない。

### 共通のレート関数

`src/scoring.py:503` `compute_effective_rate` は基準70点、起点から96秒を超えると16秒ごとに0.75倍、整数切捨て、最大14段・下限1。
`from_first_move=True` だけ開始閾値が95.5秒になる。`score_to_ojama` (`:271`, 呼出 `:289`) は既定の96秒方式。
今回の実験は **起点のみ** を変更し、撃ち合い・会計の96秒方式を維持する。収集側との95.5秒 / 96秒の差は残る。

## 2. 実装と欠測の扱い

- `src/margin_clock.py` の `first_placement_origin`: 観測列を入力とする純関数。認識器が観測済みの各側1手目時刻の最小値を返す。時刻情報のない旧記録だけ、直前のSTABLE確定盤面が空、今回のSTABLE確定盤面が色ぷよ2個である観測を設置候補にする。入力変更・内部状態・側優先はない。
- `MarginClock` は状態保持wrapper。両者で共有する起点1個、直前確定盤面、試合番号を保持する。両側の直前確定盤面が空であることを確認してから採用し、先行側を見逃した場合に後発側で代用しない。盤面復元の起点は一度決まったら凍結し、明示時刻が追加された場合は最小値を保持する。試合番号変更時にリセットする。
- 空STABLE、NON-STABLE、欠測、不明色、おじゃま、既設置盤面を1手目とみなさない。途中開始や認識欠落で復元できない区間は **従来起点へフォールバック**。欠測を差0の観測例に混ぜない。
- `--margin-origin-first-placement` は共通CLIに追加し既定OFF。描画の会計・撃ち合い、および保存記録再生の撃ち合いへ配線。
- 会計は得点処理前の `observe_margin((p1,p2), t, game_idx)` で同じ観測を受け取る。reset後のtracker再生成にもフラグを引き継ぐ。
- 実動画経路では `placement_times_from_pipeline` が既存の `_first_move_sec_1p/_2p` を読み取り専用で参照し、同一の観測タプルを会計・撃ち合いへ渡す。新記録には任意の第9引数として保存する。旧8引数記録は引き続き再生でき、OFFでは記録形式も変わらない。前試合の残時刻・未来時刻・非有限時刻は採用しない。
- 撃ち合いは `_margin_elapsed` を発火前火力・発火換算・live count・完走換算に使う。G_feの学習済み進行度時計は保持する。

**測定限界:** 旧記録の `SIDE_FIELDS` (`src/exchange_event_record.py:22`) にツモ手数・1手目時刻・設置イベントがないため、今回は確定盤面から復元した観測値。元動画上の物理設置時刻との誤差は測っていない。片側の起点だけ判明しても、相手の先行設置を見逃した可能性がある場合は共通起点に採用しない。
また会計 snapshot / finalization は既に計算された入力として記録されている。再生で会計を新時計に再生成したとは主張しない。会計の送付量変更は単体テストで確認し、実動画で会計まで再収集する検収は本結果の対象外。

## 3. 起点差（秒、ON候補−OFF）

原票: `origins.csv` / `origins.json`。5記録・63試合区間（reviewとzenchiの重複を別記録として数える）。復元15区間、欠測48区間。両側の空盤面観測まで要求した最終定義。

- 撃ち合い: n=15、最小0.6000、25%点2.7667、中央値3.1333、75%点3.2167、95%点3.4833、最大3.8333、平均2.9111秒。
- 会計: n=15、最小2.7333、25%点2.9667、中央値3.1333、75%点3.2167、95%点3.4833、最大3.8333、平均3.1200秒。
- 動画別復元数 / 全区間: q 4/15、fc 2/13、mia 5/12、review 1/4、zenchi 3/19。
- フォールバックを含む実適用差: 撃ち合い n=63、中央値0、95%点3.2300、最大3.8333、平均0.6931秒（15区間変更、48区間不変）。会計 n=60/63、中央値0、95%点3.2350、最大3.8333、平均0.7800秒。残る3区間は最初の得点観測もなく旧起点が未初期化のため、秒差を定義できない。
- 収集側の手数・側別起点は記録されていないので、収集時計の数値差は測定不能。コード上の差は上記の側別 / 両者共通、および95.5 / 96秒。

## 4. 検証手順

Pythonは指定WSL venvのみ。`PYTHONPATH=.`、nice 10、同時に最大3 Pythonプロセス。
入力5本は既存exev worktreeから **読み取りコピー** し、このworktree内の指定パスへ配置した。元worktreeには書き込んでいない。

1. `bash logs/margin_clock/run.sh legacy_off`: 本番CLIの展開後に表示平滑化だけ旧方式へ固定してe36b_on再現。
2. `bash logs/margin_clock/run.sh off`: 現 `--production-exchange-event` そのまま。
3. `bash logs/margin_clock/run.sh on`: 現本番 + `--margin-origin-first-placement`。
4. 各条件で `scripts.report_e36b` → `report_e36` → 既存E17/E13/E10系採点器。閾値・採点母数は変更しない。`scripts.report_e35.scene` で3:00行も保存。
5. `scripts.report_margin_clock` で同一行比較・ハッシュ・全指標を `COMPARISON.json` へ保存。

現本番は10月1日採用の `--exchange-event-switch-smoothing` を含むため、指定された旧基準とzenchiの一致数が異なる。この既存差とマージンON/OFF差を分けて評価する。
場面時刻は指定された旧採点器の `display_p1` 再EMAによる値を主表に残し、`report_switch_smoothing` の実表示 `display_adv` 由来の値も別記する。これは採点定義の差であり、平滑化フラグによる0.13秒の変化ではない。

## 5. 評価結果

- 旧基準の再現（legacy_off）: q log loss **0.5075669993**（6,526フレーム・4試合）、zenchi **7,671/8,333**、誤った負け確定 **0/37**、3:00場面 **2760.383333秒**。指定値をすべて再現。元e36b_onの5本のdisplay.npzとバイト一致し、SHA256をCOMPARISON.jsonへ保存した。
- 現本番OFF: q **0.5075669993**（同母数）、zenchi **7,670/8,333**、誤確定 **0/37**、場面 **2760.383333秒**。
- 現本番ON: q **0.5075669993**（同母数）、zenchi **7,670/8,333**、誤確定 **0/37**、場面 **2760.383333秒**。ON−OFF差は全項目0。
- 現本番はOFF/ONとも既存採点器のzenchi 7,671条件を満たさない。これは上記の採用済み平滑化による既存差であり、今回の回帰ではない。他の死亡監査37件、境界監査13件に誤判定・未解決はない。
- 5本・全112,346表示フレームでdisplay_adv / display_p1 / sourceの変更数0、数値最大絶対差0。ON/OFFの5本のdisplay.npzもバイト一致。更新入力は114,146行で時刻・試合・状態・得点が一致する。
- 現在時刻で計算したレート差はmiaのgame=3だけ97更新フレーム。表示内訳はconfirmed_death 94、G_fe 3。発火時に実際に用いたレート差の件数とは区別する。この区間でも表示差はなかった。
- 3:00対象（reviewのgame=3）は起点復元不能で旧時計へフォールバックしたため、変更した時計の場面検収にはなっていない。旧記録での評価不変は、起点を変更できた15/63区間の範囲で解釈する。
- 実描画値を使う補助採点: 場面到達は旧基準・OFF・ONとも **2760.25秒**、2759.05秒時点の2P表示勝率 **0.3822803356**。実表示q log lossは旧基準 **0.5060023991**、現OFF/ONとも **0.5063335562**。主採点の再EMAとは定義が異なり混同しない。

全採点の原票は各条件のSUMMARY.json、起点の試合別原票はorigins.csv、ハッシュ・表示差・補助採点はCOMPARISON.jsonに保存した。本番登録の判断は行わない。

## 6. 単体テストと変更ファイル

純関数の側交換・順序独立・同時刻・欠測・空盤面・不明色・おじゃま・全NON-STABLE・非有限時刻・入力不変、時計凍結 / 試合境界 / 欠測フォールバック、会計のレートと送付量、撃ち合い発火換算、学習済み進行度維持、CLI配線を検査する。

変更ファイル: `src/margin_clock.py`、`src/exchange_event_overlay.py`、`src/ojama_accounting.py`、`src/exchange_event_cli.py`、`src/exchange_event_record.py`、`scripts/visualize_advantage_overlay.py`、`scripts/replay_exchange_event_20260926.py`、`scripts/measure_margin_clock.py`、`scripts/audit_margin_origins.py`、`scripts/report_margin_clock.py`、`tests/test_margin_clock.py`、`logs/margin_clock/` 内の本報告・原票・実行スクリプト。

新規テスト54件を含む関連13ファイルで **370 passed, 1 skipped**（108.24秒）。任意の設置時刻の記録往復、実動画の会計 / 撃ち合いへの同一時刻配線、96秒境界で700点の送付数が旧13 / 新10となる例も検証した。監査スクリプト最終更新後に関数50行制約テストを再実行し1 passed。`git diff --check` 成功。`production_config.py`、`recognition_pipeline.py`、`collect_boards_lean.py`、`collect_indicators_v2.py` は差分なし。
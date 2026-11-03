# D5 単発着弾死亡の入力整合

## 根因と実画面

`q_7gc4TgFig`、内部game14、1P、883.966667秒。
根因は、旧単発死亡が消去前盤面の必要相殺量と探索火力の不足だけで確定し、応手後盤面の生存枝を検証しなかったこと（誤認盤面で顕在化）。

経路は `ExchangeLandingProjection._single_death_metrics` → `_evaluate_single` → `_latch`。
`unavoidable_death` の予測保持であり、E19の実死亡表示・`confirmed_death` が起点ではない。
E35上限は既に単発で死亡扱いだったため未実行。複数着弾の原票理由も `already_dead`。
当時の整合理由 `board_unsettled` は追加の複数着弾判定にしか適用されていなかった。
ただし静穏待ちだけを追加した試行でも884.133333秒に誤確定が再発した。したがって整合待ちだけでは根因を解決しない。
探索された火力が必要相殺量未満でも、それは全応手での窒息証明ではない。連鎖消去後には受けられる高さ・空間も変わる。

直前STABLEは883.866667秒、判定フレームの状態はOJAMA_FALL。
確率用受け量84個、死亡台帳106個、必要相殺85個、残4手、応手探索8個、追加の楽観探索77個。
攻撃側の予測は5連鎖5,920点。実画面と保存盤面は右端上部3セルと中央の色2セルが異なる。
884.233333秒に断定解除、885.100秒に1Pが発火し15,860点（226個）を返し、899.633333秒には2Pが倒れて1P勝利。

原票は `logs/d5/ROOT_CAUSE.json`、全盤面とNEXT・得点・予告台帳入力は `FALSE_INPUTS.json`。
画像は `q_game14_883.967.png`（断定）、`q_game14_888.500.png`（応手）、`q_game14_899.633.png`（勝敗）。
画像の読取りは事後監査だけに使用し、再生入力には戻していない。

## 修正

`single_death_proof_guard=False` を任意引数として追加。
ON時だけ、単発死亡候補を既存 `LandingStateSafety.blocker` と `cached_proof` で検証する。
欠測、静穏不足、色消失と得点の不整合、未検証の完走予測は断定しない。
全応手の消去後盤面・端数着弾の全配置を検査し、生存枝・相殺可能・探索打切りは不確定へ戻す。
必要相殺量やビーム探索火力だけで確定せず、保持中も同じ証明を要求する。証明は既存複数着弾とキャッシュを共有する。
ガード理由の変化は既存のキャッシュ署名へ入り、保持済み死亡も再評価される。
認識・モデル・確率合成・`production_config.py` は変更しない。

保存入力のCLIは `scripts.replay_exchange_event_20260926 --single-death-proof-guard`。
描画は `scripts.visualize_d5_overlay` に本番フラグと `--post-counter-death-bound --single-death-proof-guard` を渡す。
R1bの描画ファイルとE35bの専用ファイルは編集せず、D5専用ラッパーを使う。

## 検証の再現

`setsid -f bash scripts/_launch_d5.sh` でnice 10・並列1の固定5記録、R1 ONレビュー、OFF互換確認を順に実行。
完了後に同じ環境で `python -m scripts.report_d5` を実行する。
閾値と母数は `logs/d5/PREREGISTRATION.md` に事前固定。
固定検収はE31の5記録、R1先行確認は `logs/r1/records/review.jsonl.gz` を使う。
R1には4セル修正後の公開時刻2750.850秒が含まれ、元映像の同じzenchi 3:00を評価する。
R1bのON収集への置換や、手修正盤面への置換は行わない。
E35bの並行編集による検収コードの混在を避けるため、E35証明器2モジュールは指定コミット `0f28a04` から `logs/d5/runtime/` へ保存し、D5プロセス内だけその版を読み込む。
全検収で保存ソースのSHA-256一致を確認する。静穏待ちだけの不採用試行は `logs/d5/attempt_state_safety/` へ分離した。

既定OFF互換・盤面静穏・得点不整合・保持解除・生存枝と打切り拒否の単体検証と既存ガード検証は27件通過。
全5記録の集計は `SUMMARY.json`、単発経路の全件は `SINGLE_PATH_AUDIT.json`、R1時系列は `R1_E35_TIMELINE.json` に保存する。

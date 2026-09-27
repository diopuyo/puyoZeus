# Opus 5.5 統括 引き継ぎ (2026-09-27 夜)

次セッションはこの文書から開始する。前段の経緯は `C:/Users/ryouj/.codex/worktrees/56dc/puyo_analyzer/docs/agent_coordination/OPUS55_LEAD_PROGRESS_2026-09-26.md`（T01〜T13、148動画CVの結論）。

## 体制（user確定）
- **Claude = 統括**（方針・設計・採否・合否）。**Codex = サブエージェント**（実装・テスト・調査）。人の判断が要らない所は全て進める。Codexの細部拘りはClaudeが要不要を裁定。
- Codex起動: `/c/Users/ryouj/AppData/Local/OpenAI/Codex/bin/247581e40ee272fb/codex.exe exec -s workspace-write -C <worktree> -o <result.txt> - < prompt.md`（PATH外。パスは ~/.codex/config.toml の CODEX_CLI_PATH で再確認）。**タスクごと新規セッション**（resume は2回まで。長いセッションの resume は1回145万tokに膨張した）。401が出たら user に再ログイン依頼。
- 画面の見た目のデザインはCodexに任せる（Claudeは読みやすさ・必要データ・本番表示不変だけ検収）。
- user判断が要るのは: 予算、本番採用（production_config.py 登録）、main へのマージ、ゲーム仕様、保護データ。

## 作業場所
- 実装worktree: `C:/Users/ryouj/.codex/worktrees/exev/puyo_analyzer`、branch `claude/exchange-event-eval-20260926`（origin に push 済み、origin/main ccfc1d9 起点）。**main へはマージしていない。production_config.py にも未登録。**
- 検証資産: 同 worktree の `logs/`（gitignore）。記録再生検証台 `scripts/replay_exchange_event_20260926.py`（3動画81,000フレームを約80秒で再生、レンダとバイト一致）。

## 作ったもの：撃ち合いイベント駆動評価（新フラグ `--exchange-event-update`、既定OFF）
| 状態 | 評価器 |
|---|---|
| 静止 | G_fe（旧46指標＋M0＋充填量/経過秒の位相交互作用） |
| 発火（撃ち合い内の各発火で再評価） | S1（v2: 近未来火力のおじゃま個数・打ち返しの余地を追加） |
| 撃ち合い中 | 暫定S3: 各連鎖の得点 = max(掛け算式の段累積, 発火時の完走シミュ予測) → S3 と「確定受け量を仮想着弾した盤面の G_fe」を logit 平均 |
| 撃ち合い確定 | S3（同上） |
| 回避不能死 | 仮想着弾で窒息1段超過かつ着地までの手数で応手不足 → 2%固定、着地まで保持（盤面・相殺・新発火で再判定）。連鎖中の側は**完走後盤面**で判定 |
- ON表示にもOFFと同じEMA(0.25)。判定入力は同一フレームで全更新してから1回だけ評価（中間値を出さない）。
- レビュー用: `--review-data-panel`（各評価器の勝率・受け量・連鎖得点・応手・近未来火力(個数)・主要指標を盤面下に表示）、`--review-data-csv`（166列/フレーム）。
- モデル: `models/exchange_event_v1/`（G_fe・S1・S3・M0）、`models/exchange_event_v2/`（S1′・S3′、E14で組込み。`--exchange-event-model-dir` の既定は v1 のまま）。

## 数値（未見動画・記録再生、OFF=現行本番）
- q（独立勝敗4試合）log loss: OFF 1.111 → 最新 0.570、AUC 0.736
- zenchi 30先セット1 第41〜57試合の終盤1/3勝者一致率: OFF 61.2% → 最新 80.3%
- 回避不能死の誤発火 1/28（3.6%）、±3帯反転 OFF 288 → 256回/45分
- 148動画CV: 中盤は撃ち合い前の静止情報で天井≈0.58、撃ち合い後情報の上限0.722。S3′（リーク是正版）中盤 .717、S1′ .652

## コミット履歴（branch）
98397b8 E1 / 9435926 E2 / 8e48558 E3b〜E5 / 1dcdf20 E6〜E8 / 8aafa9c E9 / be85fe7 E10〜E10c / 833ffcd E11 / eabb126 E12/E12b / cd70688 E13 / **（E14+F1+E14b は本セッション末にコミット、下記参照）**

## userの目視レビュー（進行中）
- レビュー動画の規約: **zenchi `video_zenchi_c0BQoMJwwQU` の30先セット1「3本目」= 第41〜57試合（2580.566〜3427.166秒）から3試合連続の短尺**（現在は第41〜43試合）。スマホで見る → **新方式のみ・横1280・30MB以下**（SendUserFile上限30MiB）、scratchの mobile.sh/split.sh 相当（imageio_ffmpeg, crf28, 必要なら200秒分割）。PC用は D:/puyo_analyzer/videos/review/。レビュー用は「主因」要約より**データ最大表示**。
- 最新送付: `D:/puyo_analyzer/videos/review/zenchi_g41-43_e14_v2_mobile.mp4`（高画質: exev/logs/review_zenchi_g41_43_e14/overlay.mp4）。userの評価待ち。
- userのドメイン知見（memory `reference_midgame_decided_by_exchanges_2026-09-26.md` に集約）: 中盤の優劣は撃ち合いで付く／お邪魔収支・残り色ぷよ・形の総合判断／「2段確定＝不利」ではなく統計で／返す術がなければ勝率に出す／連鎖中は窒息としない。

## 未解決・次の候補（優先順）
1. userの E14 動画の評価を受けて修正。
2. 値飛びの残り（41回/45分）: 判定切替時（S3→G_fe 等）の食い違い。切替時の短時間ブレンドが候補。
3. S3′ は発火前盤面で控えめ。撃ち合い終了直前の着地前盤面（仮想着弾）を使う版の学習（リーク監査必須）。
4. 本番採用の判断材料作り（userの目視合格後）→ production_config.py 登録（採用日＋根拠）と main への PR（user承認）。
5. 長期: 撃ち合い結果の事前予測（探索型、自作2人対戦シミュ。ama公開版は対戦シミュ無し）。

## 教訓（memoryにも記録済み）
- 安定性の改善主張の前に「評価値の鮮度」をON/OFFで比べる（表示値は平滑化で交絡する）。
- 誤りの割合は「どれだけ判定しているか（到達率・滞在）」と必ず組にする。
- 新モデルがオラクル上限を超えたらリークを疑い、特徴の元盤面の時刻を評価時点と行ごとに監査させる。
- 事前登録には実用効果の下限（LL ≤ −0.002 または AUC ≥ +0.01）を入れる。
- Codexの機械判定をそのまま通さず、生の値（source内訳・同値率・実画面フレーム）を自分で見る。
- ブランチ比較は必ず `git fetch` 後の origin/main で。

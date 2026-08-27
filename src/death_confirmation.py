"""死亡確定の時間的ロジック (Gate 3R-6 本体、2026-08-25 user伝授)。

## 背景

`src/board.py` の `Board.is_dead()` は「12段目 (row=DEATH_ROW, col=DEATH_COL)
の占有」だけを見る**静的判定**であり、仮想盤面・ビームサーチ (via
`src/puyo_core_bridge.py`) を含め `src/indicators_v2.py` 内で19箇所から
呼ばれている。これらは「占有=死」を正しい前提として使っているため、
**本メソッドは変更しない** (呼出元多数、backwards compat 上の制約)。

一方、user 伝授 (`docs/agent_coordination/DECISIONS.md` 2026-08-25
「user伝授 2026-08-25 死亡確定の条件」節) によれば、実際の死亡確定は

1. **12段目に設置して連鎖が起きない**
2. **おじゃまが降って12段目が埋まる**

のいずれかだが、**この判定自体が難しい** (占有した瞬間はまだ死んでいる
とは限らない。相手の連鎖解決を待って自分が連鎖できれば助かる、
`reference_full_board_is_not_death_2026-08-22`)。実測 (残存0.7秒事案、
1P・実試合2・t=164.033-164.733) では「おじゃま着弾で占有 → 0.634秒後に
自分の連鎖が発火して解決」という「候補だったが死んでいない」ケースが
確認されている。猶予の長さは相手の連鎖解決時間に依存するため固定値
では表せない。

## 【設計の訂正 2026-08-25】確定条件は「次の事象」ではなく「ネクスト不動」

当初案は確定条件に「連鎖が始まらないまま次のツモが置けた/さらに
おじゃまが降った」を使ったが、**これは論理が逆だった**: 死亡すると
そもそも次のツモを置けない。「ネクストが動いた」は**生存の証拠**であり、
死亡の証拠ではない。死亡すると設置・連鎖・スコア・盤面・ネクストの
**あらゆる観測信号が止まる** ため、死亡は「信号が来ないこと」でしか
判定できず、積極的な証拠 (本来は底抜け演出だが調査を後回しにした、
下記 user 決定参照) が必要になる。

**user 決定 (2026-08-25)**: 底抜け演出の調査は後回しにし、
**簡易検出で一旦進める**。「12段目が設置され、ネクストが
`NEXT_STATIONARY_CONFIRM_SEC` 秒以上動かない」を確定条件とする。

## 設計: 候補 → 猶予 → 確定 (または解除) の3段階

1. **候補**: 確定盤面 (STABLE) で死亡セルの占有が新たに観測された瞬間。
   発生源を「設置 (`DEATH_SOURCE_PLACEMENT`、TSUMO_FALL→STABLE 遷移由来)」
   と「おじゃま (`DEATH_SOURCE_OJAMA`、OJAMA_FALL→STABLE 遷移由来)」に
   区別して記録する (user の条件が2つに分かれているため)。
2. **猶予/解除**: 自分の連鎖開始 (掛け算式の実読による CHAIN 遷移、
   `is_own_chain_start`) を待つ。連鎖が始まれば候補を**解除**
   (死亡ではなかったと判断、user 絶対律「連鎖を打つ直前 / 連鎖中は
   窒息としない」)。
3. **確定 (簡易検出)**: 連鎖が始まらないまま、**ネクストが
   `NEXT_STATIONARY_CONFIRM_SEC` 秒以上変化しなければ**死亡確定する。

未来参照は禁止 — 全ての判定は「現在フレームまでの過去の観測」だけで
行う (配信オーバーレイでそのまま動くことが要件)。

## `NEXT_STATIONARY_CONFIRM_SEC` (1.5秒) について — 暫定値

**この閾値は user 指定の暫定値であり、Claude がシーンから逆算した値では
ない** (2026-08-25 user指示: 「12段目が設置され、ネクストが1.5秒以上
動かない等で簡易検出で一旦」)。将来、底抜け演出の検出による根治に
差し替える前提の暫定実装。CLI から変更できるようにし
(`--death-next-stationary-sec`)、感度を事後測定できるようにする。

## ネクスト不動の観測方法 (二重実装防止)

`src/off_match_window.py` の `SideObservation.next_key` が「ネクストの
識別値、同値なら動いていない」という考え方を既に使っている。ただし
`find_off_match_spans` 本体は**事後遡り込み (backtrack) を含むオフライン
専用アルゴリズム** (モジュール docstring 「物差しはオフライン処理なので
先読みが許される」) であり、未来参照禁止の本モジュールには流用できない。
そのため **「同値なら動いていない」という識別キーの考え方だけを流用**し、
値の出典は `SideResult.next_pair` (`src/recognition_pipeline.py:413`、
既に「next変化検知」に使われている確立済みフィールド、例:
`src/recognition_pipeline.py:2852-2853` の連鎖終了判定) とする。
新規の next 検出器は作らない (二重実装防止)。

## stateless/stateful の分離 (コーディング規約)

- 純関数 (`classify_death_candidate_source` / `is_own_chain_start` /
  `has_next_key_changed`) が「1回の観測」を判定する。入力は現在・直前の
  値のみで、内部状態を持たない。
- `DeathConfirmTracker` (外部 wrapper、1サイド分) だけが「候補中かどうか
  / 発生源 / ネクスト最終変化時刻 / 確定済みかどうか」を保持する。
  2サイド分必要な場合は2インスタンス生成する
  (`scripts/collect_indicators_v2._SideTracker` と同じパターン)。
- `DeathConfirmStats` (外部 wrapper) が母数付きカウンタを集計する。

## 既知の限界 (残課題、隠さず明記)

1. **CHAIN/GRAVITY_SETTLE→STABLE 遷移は候補にしない** (スコープ外)。
   連鎖が一度起きた後もなお死亡セルが占有され続ける「連鎖が足りなかった」
   ケースは、本モジュール単体では新規候補として検出しない
   (`classify_death_candidate_source` docstring 参照)。ただし
   CHAIN 状態は必ず GRAVITY_SETTLE より先に観測される設計
   (`src/board_state_machine.py` の状態遷移順序) のため、`is_own_chain_start`
   は連鎖の発生自体を取り逃さない。
2. **1.5秒は暫定値**。底抜け演出による根治までの簡易検出であり、
   `own chain` が来ないまま試合が続く限り、確定は本物の死亡より最大
   `NEXT_STATIONARY_CONFIRM_SEC` 秒遅れる (user 承認済みの仕様、実測で
   遅延分布を報告する)。**ただし試合が候補発生から `stationary_confirm_sec`
   秒未満で終わった場合は `on_game_boundary()` がその場で確定するため、
   この上限は「試合が十分長く続いた場合」に限った上限** (2026-08-25
   第2版「候補のまま試合終了 = 死亡確定」節参照)。
3. **`src/production_config.py` への採用登録は本モジュールの範囲外**。
   数値を user が確認してから (`docs/agent_coordination/PLAN.md` Gate 5)。
4. **まちうけ画面での誤確定 (2026-08-25 `game_idx` ガードで対処)**。
   実測 (zenchi 先頭5試合、t=18.067〜90.5、72.4秒) で、試合開始前の
   まちうけ画面 (`docs/KNOWN_WEAKNESSES.md` W37 と同根の背景誤検出で
   `is_dead2=True` が574/1019行=56%発生する既知区間) では、死亡セルの
   背景誤検出による占有と、まちうけ画面のネクスト表示が無い/固定である
   ことが重なり、簡易検出が誤って即座に確定し、**試合開始後も約3.6秒
   sticky に持ち越す**副作用が実測された。
   当初対策の `is_match_active` (`PipelineResult.is_match_active`) ガードは
   **実測で無効だった** (t=10-95 の全区間で True=851/False=0 と一度も
   False にならないことを直接計装で確認、
   `scripts/_diag_gate3r6_death_confirm_matchactive_2026-08-25.py`)。
   `--no-force-in-match` (複数試合を跨ぐレンダに必須) 構成では
   `is_match_active` がまちうけ画面を「試合外」と判定しないため。
   `src/off_match_window.py` の W37 対策は事後遡り込み専用でリアルタイム
   流用不可 (モジュール docstring 参照)。ガード自体は他の構成で
   `is_match_active` が正しく機能する場合に備えて残す (無害な保険)。

   **第一次対策 (2026-08-25、後に格下げ)**: `update()` に `game_idx`
   (`scripts/visualize_advantage_overlay.py` の試合境界カウンタ、動画内
   で0始まりで試合ごとに +1) を渡し、`is_pre_match_game_idx()` が True
   の間 (=`game_idx==0`、最初のスコアリセット検知が来る前) は
   `is_match_active` と同じ扱いで凍結する設計を最初に採用した。実測
   (前掲 npz の `game_idx` 列): まちうけ区間 (t=18.067-86.267) 1019行の
   うち1018行が `game_idx==0` (母数1019行)。しかし **`game_idx==0` は
   動画の編集方針に依存する条件であり、「まちうけ画面を含まず試合の
   途中から始まる」動画では最初の試合全体が `game_idx==0` のまま進み、
   本ガードが誤って死亡確定を凍結し続け、合格条件の最重要項目 (真の
   窒息を見逃さない) を丸ごと破りうる**、と user 指摘 (2026-08-25) で
   確定。

   **最終対策 (2026-08-25 user指示で確定・採用)**: 凍結の主判定を
   content-based な `_has_ever_placed` (「その side でまだ一度も設置
   (`TSUMO_FALL→STABLE`) を観測していないか」、`is_placement_transition()`
   参照) に置き換え、`game_idx==0`/`is_match_active=False` は「凍結を
   検討する入口」を開くだけの補助信号に格下げした
   (`DeathConfirmTracker.update()` docstring 参照)。理由:
   まちうけ画面ではぷよが落下しないため設置は絶対に観測されない一方、
   「試合の途中から始まる動画」でも最初の1手が置かれた瞬間に凍結が解け、
   丸ごと見逃す事態を防げる。`_has_ever_placed` は一度 True になったら
   `on_game_boundary()` でもリセットしない (動画を通して一度でも設置を
   見たら以後は凍結しない、user 指示の安全側設計)。

   **既知の限界 (対策後も残る)**:
   (a) 実試合開始直後の最初の数十ms (デバウンス
   `GAME_BOUNDARY_DEBOUNCE_SEC` 秒分) はまだ `game_idx==0` かつ
   `_has_ever_placed=False` のままであり得るため、実試合1本目の冒頭の
   最初の設置までの間だけ確定が遅れる (実測では 0.1 秒程度、実害は
   小さいと判断)。
   (b) **【Codex 指摘3 で修正済み、下記「Codex 独立レビュー NG 対応」節
   参照】試合間 (決着演出〜次試合開始) に「まちうけ画面相当」の背景誤
   検出が再発する懸念**。当初は `_has_ever_placed` が最初の試合の最初の
   設置以降ずっと True のままだったため、2試合目以降のまちうけ的な画面
   (もしあれば) では `game_idx` も既に1以上・`_has_ever_placed` も True
   となり、両ガードとも凍結しなかった。`_post_boundary_armed`
   (`on_game_boundary()` のたびに False へ落ち、`is_real_new_game_start()`
   観測まで再アームしない独立ゲート) を追加し、`_has_ever_placed` の
   永続 True に依存せず境界ごとに再凍結するよう修正した。zenchi 動画では
   決着演出中に dump 記録自体が疎になる (settled recompute 空白区間、
   8.6〜9.1秒) ため、その空白の内側で背景誤検出が実際に起きているかは
   **今回のデータからは確認できていない** (わかっていないこと、今後の
   実データ検証が必要) が、起きていたとしても `_post_boundary_armed` に
   より凍結される設計になった。

## 【Codex 独立レビュー NG (2026-08-25) 対応、第2版】

Gate 3R-6 初版は Codex レビューで NG となった。指摘4 (user 判断待ち、
底抜け演出根治の要否) を除く指摘1・2・3・5 を本版で修正した。

**指摘1: 1.5秒の猶予は「候補発生後」から測る。** 初版は
`t_sec - last_next_change_sec` だけで確定判定しており、候補発生より
**前から** ネクストが不動だった場合、候補発生の直後 (実測 0.067秒後) で
確定してしまっていた (猶予が実質無効化)。`_observe_pending()` を
`t_sec - max(pending_since_sec, last_next_change_sec) >= threshold` に
修正し、**候補発生から `stationary_confirm_sec` 秒未満では絶対に確定
しない**ようにした。

**指摘2: `next_pair=None` を「不動の証拠」にしない。** 初版は None を
「前回と同値」とみなし (`has_next_key_changed(None, None) is False`)、
None が続くだけで不動タイマーが進み続けていた。`None` は「未検知」で
あって「動いていない」ではない。`_update_next_change_tracking()` を
改修し、`next_key is None` の間は `_next_valid_now=False` として
**不動タイマーを完全に停止** (確定判定自体を保留)、次に有効な
`next_pair` を観測した瞬間に基準時刻を打ち直す (測り直す) 設計にした。
`src/recognition_pipeline.py` の `_is_game_event_chain_exit()`
(`current_next is not None and start_next is not None` の両方有効判定)
と同じ「両方揃って初めて比較する」考え方を流用している。

**指摘3: 試合境界後の再凍結。** 初版は `_has_ever_placed` が動画全体で
一度 True になると二度と凍結せず、2試合目以降の背景誤認 (まちうけ相当
画面の再発) を防げなかった。`on_game_boundary()` で新たに
`_post_boundary_armed=False` を立てて候補受付自体を凍結し、
`is_real_new_game_start()` (=設置完了の物理遷移 `is_placement_transition`
**かつ** その瞬間の `next_pair` が有効) を観測して初めて再アームする
設計に変更した。単なる `curr_state == TSUMO_FALL` (旧 `confirmed` 消去
条件) だけでは1フレームの誤検出に弱いため、`is_real_new_game_start()`
は confirmed 消去 (`_awaiting_new_game_clear`) と再アーム
(`_post_boundary_armed`) の両方で共通利用する。**前試合の `confirmed`
表示は変更なし** (実ゲーム開始まで保持、旧根治を壊さない)。

## 【設計の穴の是正 2026-08-25 第2版】候補のまま試合終了 = 死亡確定

上記「指摘1」修正 (猶予を候補発生後から測る) を正しく実装した結果、
**実データ突合で真の窒息 (2P・実試合2・t=223) が検出できなくなる欠陥が
判明した** (`npz` 直接突合、動画全体で確定 0/5083行)。カウンタ内訳:
候補6件のうち3件が「`NEXT_STATIONARY_CONFIRM_SEC` 秒に達する前に試合が
終わって消えた」。t=223 の真の窒息も候補発生 t=223.033、試合の終わり
t=223.40 の **0.37秒後** で、猶予未達のまま `on_game_boundary()` に
消されていた。

**根本原因**: 人が本当に詰んだとき、試合はその場で終わる。「ネクストが
`stationary_confirm_sec` 秒動かないのを待つ」という条件は、動かすネクスト
自体がもう来ないため**真の死亡では原理的に満たされない**。旧実装で
0.067秒という短い遅延で確定していたのは、指摘1修正前の猶予が実質
無効化されていたために**偶然**真の死亡を拾えていたに過ぎない。

**直す方針 (user 伝授の条件そのもの)**: 死亡確定の条件は「12段目に設置して
連鎖が起きない」または「おじゃまが降り12段目が埋まる」。**候補が立ったまま
試合が終わった = 「連鎖が起きなかった」ことを試合の終わり自体が証明する**。
そのため `on_game_boundary()` は、猶予中の候補を「閾値未到達で消滅」させる
のではなく、**その場で死亡確定する** (`self.confirmed = True`)。

**まちうけ画面の偽陽性は再発しない**: 試合外 (凍結中) は `update()` が
毎フレーム `_reset_transition_tracking()` を呼ぶため候補自体が絶対に
残らない (`frozen` 判定は候補発生の判定より前に評価される)。したがって
`on_game_boundary()` 到達時点で `_pending_source is not None` なのは、
凍結されていない実ゲーム中に候補が発生し、まだ解除 (own chain 開始) も
されていない場合に限られる。

**解除済み候補は対象外**: own chain が始まった候補は `_pending_source`
が既に None にクリアされているため、この確定ロジックの対象にならない
(user 伝授「連鎖中は窒息としない」を壊さない)。

**戻り値の意味変更**: `on_game_boundary()` の戻り値 (`str | None`) は
従来「期限切れとして消えた候補の発生源」だったが、本修正後は「境界検知の
瞬間に死亡確定した候補の発生源」を意味する。呼出元は
`DeathConfirmStats.record_confirmed_at_boundary()` へ渡す
(旧 `record_expired_at_boundary` は構造として残すが、この呼出元からは
もう使わない。既存の `expired_at_boundary_*` カウンタは常に0になる
想定 — 候補が凍結なしに残ったまま境界に達するケースが理論上存在しない
ため。呼出元コードで「境界で確定した件数」と区別して母数付きで記録する)。

## 【第2版への Codex 条件付き承認・追加要件 2026-08-25 第3版】

第2版 (無条件で境界確定) は Codex から**条件付き承認**を受けた:
「未解除候補のまま同一試合が終了した」を死亡確定の積極的証拠として
追加してよいが、`on_game_boundary()` で**全 pending を無条件確定しては
ならない**。以下をすべて満たす場合のみ確定する:

1. **候補発生時と境界の `game_idx` が同じ** (`_pending_game_idx` を
   候補発生時にスナップショットし、`on_game_boundary(game_idx=...)` で
   比較する。別ゲームへの越境確定を防ぐ安全弁。上記「まちうけ画面の
   偽陽性は再発しない」節の凍結ガードにより構造的には到達しない想定
   だが、防御的に二重で守る)。
2. **候補後に own CHAIN が開始していない** (既存、変更なし)。
3. **候補後に next 変化・次の TSUMO 開始などの生存証拠がない**
   (`_is_survival_evidence()`。死亡していれば新しいツモは配られず
   next も変化しないため、どちらか一方でも観測されれば「死んでいない」
   と判断し、`_observe_pending()` が即座に `released_survival_{source}`
   として解除する。`on_game_boundary()` 側にも同じ判定を再実装し
   二重に守る、下記「生存証拠は"解除"として先に効く」節参照)。
4. **両側に候補がある場合、両方を無条件に死亡確定しない**
   (`resolve_boundary_confirmations()`。勝敗側を一意に決められなければ
   ambiguous として両方とも確定しない、`suppress_confirm=True`)。

**生存証拠は"解除"として先に効く**: next 変化/新規ツモ落下は
`_observe_pending()` が毎フレーム同じ条件でチェックしており、候補が
境界に到達する**前**に `released_survival_{source}` として解除される。
そのため `on_game_boundary()` 内の生存証拠再チェック
(`rejected_survival_evidence`) は**構造的には到達しない想定の防御**
だが、呼出順序の将来変更に備えて残す (`DeathConfirmStats.
boundary_rejected_survival_evidence` は通常0のまま、母数付きで報告)。

**score OCR の一時的 reset ではなく正式な試合境界判定を通過**: 呼出元
(`scripts/visualize_advantage_overlay.py`) は `game_idx` の debounce
(`GAME_BOUNDARY_DEBOUNCE_SEC`) 済みの値を候補発生時のスナップショットに
使い、境界検知の raw フレームで何度も呼ばれても内容は変わらない
(2回目以降は `_pending_source` が既に None のため無害、上記条件1参照)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# state 名の文字列定数 (BoardState.*.name の値と一致させる。
# 循環 import を避けるため src.board_state_machine には依存しない。
# 呼出元は r.pX.state.name のような文字列を渡すこと)。
STATE_STABLE: str = "STABLE"
STATE_TSUMO_FALL: str = "TSUMO_FALL"
STATE_CHAIN: str = "CHAIN"
STATE_OJAMA_FALL: str = "OJAMA_FALL"

# 死亡候補の発生源 (user の条件が2つに分かれているための区別)。
DEATH_SOURCE_PLACEMENT: str = "placement"
DEATH_SOURCE_OJAMA: str = "ojama"

# (2026-08-25 user指示) ネクスト不動による簡易確定の閾値 (秒)。
# **シーンからの逆算ではなく user 指定の暫定値** (底抜け演出検出による
# 根治までの簡易実装、モジュール docstring 参照)。マジックナンバー禁止
# 規約に従い、値そのものはこの1箇所でのみ定義する。
NEXT_STATIONARY_CONFIRM_SEC: float = 1.5

# (2026-08-25 実測で確定) `game_idx` がこの値のとき「まちうけ画面相当
# (最初のスコアリセット検知が来る前)」とみなす。`game_idx` は
# `scripts/visualize_advantage_overlay.py` の試合境界カウンタで動画内
# 0始まり (モジュール docstring 「まちうけ画面での誤確定」節参照)。
GAME_IDX_PRE_MATCH: int = 0


def classify_death_candidate_source(
    prev_state: str, curr_state: str, death_cell_occupied: bool,
) -> str | None:
    """1回の状態遷移から死亡候補の発生源を判定する (純関数)。

    「TSUMO_FALL→STABLE」または「OJAMA_FALL→STABLE」の遷移で、かつ
    遷移後の確定盤面の死亡セルが占有されている場合のみ発生源を返す。
    CHAIN/GRAVITY_SETTLE→STABLE (連鎖経由の復帰) はここでは候補にしない
    (連鎖がすでに起きているため、user 絶対律「連鎖中は窒息としない」の
    範囲内。連鎖後もなお占有が続く残存ケースは既知の限界として残す。
    詳細は本モジュール docstring 末尾の残課題を参照)。

    Args:
        prev_state: 直前フレームの state 名 (`STATE_*` のいずれか)。
        curr_state: 今フレームの state 名。
        death_cell_occupied: 今フレームの確定盤面の死亡セル占有 (STABLE
            のときのみ意味を持つ。`Board.is_dead()` と同じ定義)。

    Returns:
        `DEATH_SOURCE_PLACEMENT` / `DEATH_SOURCE_OJAMA` / None (候補なし)。
    """
    if curr_state != STATE_STABLE or not death_cell_occupied:
        return None
    if prev_state == STATE_TSUMO_FALL:
        return DEATH_SOURCE_PLACEMENT
    if prev_state == STATE_OJAMA_FALL:
        return DEATH_SOURCE_OJAMA
    return None


def is_placement_transition(prev_state: str | None, curr_state: str) -> bool:
    """`TSUMO_FALL→STABLE` への遷移か (=設置が観測されたか、純関数)。

    死亡セル占有の有無は問わない (`classify_death_candidate_source` とは
    目的が異なる: あちらは「死亡候補の発生源判定」、こちらは「まだ一度も
    設置を観測していない (=試合が始まっていない可能性が高い)」を
    content-based に判定するための軽量な生 state 遷移チェック、
    `DeathConfirmTracker` docstring「まちうけ画面での誤確定」節参照)。

    まちうけ画面ではぷよが落下しないため、この遷移は絶対に観測されない
    はず (2026-08-25 実測、`data/verify/gate3r6_death_confirm_2026-08-25/
    first5games_deathconfirm_on.npz` のまちうけ区間1018行で `state1`/
    `state2` の値の集合に `TSUMO_FALL` が一度も含まれないことを確認、
    母数1018行)。`prev_state=None` (初回フレーム) は常に False。
    """
    return prev_state == STATE_TSUMO_FALL and curr_state == STATE_STABLE


def is_real_new_game_start(
    prev_state: str | None, curr_state: str, next_key: object,
) -> bool:
    """「本当の新試合開始」を検証する (純関数、Codex 指摘3 対応)。

    単なる `curr_state == STATE_TSUMO_FALL` への遷移だけでは、認識の
    単発誤検出 (1フレームだけ TSUMO_FALL に化ける等) に弱い
    (`DeathConfirmTracker` docstring「Codex 独立レビュー NG 対応」節参照)。
    本関数は「設置が完了した (`is_placement_transition`、実際に
    TSUMO_FALL→STABLE まで到達した物理事象)」と「その瞬間の `next_pair`
    が有効 (None でない)」の**両方**を要求することで、単発の誤検出が
    完走することは考えにくいという性質を利用し corroboration とする。

    確定済みフラグの消去 (`_awaiting_new_game_clear`) と境界後の再アーム
    (`_post_boundary_armed`) の両方で共通利用する (同一事象で同時に
    解決してよいため、二重実装を避ける)。
    """
    return is_placement_transition(prev_state, curr_state) and next_key is not None


def is_own_chain_start(prev_state: str, curr_state: str) -> bool:
    """own state が新たに CHAIN へ遷移した瞬間か (猶予解除の条件、純関数)。

    掛け算式の実読による CHAIN 遷移をそのまま使う (`DECISIONS.md`
    2026-08-25 実測: 遅延中央値0.033秒・最大0.1秒、29件中0件が未到達。
    未来参照なしでリアルタイムに使える精度が既に確認済み)。
    """
    return curr_state == STATE_CHAIN and prev_state != STATE_CHAIN


def is_new_tsumo_fall_start(prev_state: str, curr_state: str) -> bool:
    """新しいツモが落下を開始した瞬間か (=`*→TSUMO_FALL`、純関数)。

    【2026-08-25 第3版、Codex 承認条件対応】猶予中の候補にとって「次の
    ツモが降ってきた」ことは強い生存証拠になる (死亡していれば新しい
    ツモは絶対に配られない)。`is_placement_transition`
    (`TSUMO_FALL→STABLE`、設置完了) とは逆向きの遷移であり、混同しない
    よう別関数として定義する。
    """
    return curr_state == STATE_TSUMO_FALL and prev_state != STATE_TSUMO_FALL


def has_next_key_changed(prev_next_key: object, curr_next_key: object) -> bool:
    """ネクストが動いたか (同値なら動いていない、純関数)。

    `src/off_match_window.py` の `SideObservation.next_key` と同じ
    「識別キーが同値なら動いていない」という考え方を流用する
    (モジュール docstring 「ネクスト不動の観測方法」参照)。
    """
    return prev_next_key != curr_next_key


def is_pre_match_game_idx(game_idx: int | None) -> bool:
    """`game_idx` が試合開始前 (まちうけ画面相当) かどうか (純関数)。

    【2026-08-25 user指示により補助信号に格下げ】本関数単体では
    「まちうけ画面かどうか」を確定させない (動画の編集方針に依存する
    ため主判定にしない、`DeathConfirmTracker.update()` docstring
    「まちうけ画面での誤確定」節参照)。凍結の主判定は
    `is_placement_transition()` を積算した `_has_ever_placed`
    (content-based、「その side でまだ一度も設置を観測していない」) で
    あり、本関数はその判定の入口 (=凍結を検討する候補区間) を絞る
    補助信号としてのみ使う。

    `game_idx` は `scripts/visualize_advantage_overlay.py` の試合境界
    カウンタ (動画内で0始まり、`_detect_score_reset` によるスコア
    リセット検知のたびに +1)。動画開始〜最初のスコアリセット検知までは
    `0` のまま進まない。

    2026-08-25 実測 (`data/verify/gate3r6_death_confirm_2026-08-25/
    first5games_deathconfirm_on.npz`): まちうけ画面区間
    (t=18.067-86.267、`is_dead2=True` 574/1019行=56%の既知の背景誤検出
    区間) の1019行のうち1018行が `game_idx==0` (母数1019行)。
    モジュール docstring「まちうけ画面での誤確定」節に既知の限界を明記。

    `game_idx=None` (未配線の既存呼出元、backwards compat) の場合は
    常に False を返す (=ガードなし、既存動作を完全に保持)。
    """
    return game_idx is not None and game_idx == GAME_IDX_PRE_MATCH


@dataclass
class DeathConfirmTracker:
    """1サイド分の死亡確定状態を保持する外部 wrapper (stateful)。

    候補発生→猶予 (own chain 開始を待つ) →確定 (ネクスト不動
    `stationary_confirm_sec` 秒) or 解除、を1サイドぶんだけ追跡する。
    2サイド分必要な場合は2インスタンス生成すること
    (`scripts/collect_indicators_v2._SideTracker` と同じパターン)。

    未来参照なし: 毎フレーム `update()` に「現在の state 名・死亡セル
    占有・現在時刻・現在のネクスト識別値」だけを渡す。内部に保持するのは
    直前フレームの state 名、進行中の候補の発生源・発生時刻、直近の
    ネクスト識別値とその最終変化時刻のみ。

    試合境界: 呼出元は `on_game_boundary()` を呼ぶこと (2026-08-25 実測で
    根治、下記 docstring 参照。**新規インスタンスへの差し替えではない**)。

    【実測で発見した不具合と根治】(2026-08-25、`data/verify/
    gate3r6_death_confirm_2026-08-25/` で実データ検証)
    当初は「試合境界 = 新規インスタンスに差し替え」としていたが、実測
    (2P・実試合2・t=223 の真の窒息) で以下が判明した:
    決着演出〜結果表示の数秒間は両者ともCNN認識が乱れ settled recompute
    が一度も起きず (既知の dump 記録上の空白区間)、その空白の最中に
    score が 0 付近に戻る「試合境界検知」が発生し、**旧実装ではその場で
    新規インスタンスに差し替えていたため確定フラグが空白区間の内側で
    消え、dump に一度も True が現れなかった** (=見逃しに見える)。
    根治: 試合境界を検知した瞬間に確定フラグを即座に消さず、
    `on_game_boundary()` で「消去待ち」フラグだけを立てる。実際に消すのは
    **次の本当の新試合開始を示す物理事象** (`is_real_new_game_start()`、
    設置完了+有効next) を観測した瞬間だけにする。固定の待ち時間ではなく
    物理事象で消去タイミングを決める。

    【Codex レビュー NG (2026-08-25) 対応、指摘3】試合境界後は
    `_post_boundary_armed=False` で候補受付自体を凍結し、
    `is_real_new_game_start()` を観測するまで再アームしない
    (`_has_ever_placed` が既に True でも境界ごとに再凍結する、モジュール
    docstring「Codex 独立レビュー NG 対応」節参照)。
    """

    stationary_confirm_sec: float = NEXT_STATIONARY_CONFIRM_SEC
    _prev_state: str | None = None
    _pending_source: str | None = None
    _pending_since_sec: float | None = None
    _last_next_key: object = None
    _last_next_key_seen: bool = False
    _last_next_change_sec: float | None = None
    # (Codex 指摘2) 今フレームの next_key が有効 (None でない) かどうか。
    # None の間は不動タイマーを進めない/確定判定を保留する。
    _next_valid_now: bool = False
    confirmed: bool = False
    confirmed_source: str | None = None
    # sticky な監査値を次試合の物理判定へ混入させないため、確定が属する
    # 試合番号を別に保持する。game_idx を渡さない既存呼出元は従来互換。
    confirmed_game_idx: int | None = None
    # (2026-08-25 根治) 試合境界検知後、確定フラグの消去を「次の本当の
    # ツモ落下開始」まで遅延させるための待機フラグ。
    _awaiting_new_game_clear: bool = False
    # (2026-08-25 user指示) content-based な「まだ試合が始まっていない」
    # 判定用。_raw_prev_state は凍結中でもリセットしない生の直前 state
    # (死亡候補用 _prev_state は凍結中に None へリセットされるため、
    # 「凍結中に起きた設置」を見失わないよう完全に独立させて持つ)。
    # _has_ever_placed は一度 True になったら **on_game_boundary() でも
    # is_match_active=False でもリセットしない** (user 指示: 「動画を
    # 通して一度でも設置を見たら以後は凍結しない、が安全側」)。
    _raw_prev_state: str | None = None
    _has_ever_placed: bool = False
    # (Codex 指摘3) 境界後の再アームフラグ。on_game_boundary() のたびに
    # False へ落ち、is_real_new_game_start() を観測するまで候補受付を
    # 凍結する (_has_ever_placed の永続 True に依存しない独立ゲート)。
    _post_boundary_armed: bool = True
    # 【2026-08-25 第3版、Codex 承認条件対応】候補発生時点の game_idx。
    # `on_game_boundary()` で「候補発生時と境界の game_idx が同じ」ことを
    # 確認するための基準値 (別ゲームへの越境確定を防ぐ安全弁)。
    _pending_game_idx: int | None = None
    # 候補発生時点で有効だった next 識別値のスナップショット (None なら
    # 候補発生時点で next が未検知だった)。猶予中に next が変化したら
    # 生存証拠として候補を解除する (`_is_survival_evidence` 参照)。
    _next_key_at_candidate_start: object = None

    def on_game_boundary(
        self, game_idx: int | None = None, suppress_confirm: bool = False,
    ) -> tuple[str | None, str]:
        """呼出元の試合境界検知 (score-reset 等) のたびに呼ぶ。

        候補中の遷移追跡 (`_prev_state`/`_pending_source`/ネクスト最終
        変化時刻) は試合が変わると連続性が失われるため即座にリセットする
        (別試合の観測を跨いで誤判定しないための安全策)。**確定済み
        フラグ (`confirmed`) はここではクリアしない** — 上記クラス
        docstring の根治参照。次の `is_real_new_game_start()` 検知まで
        保持され、dump に反映される機会を失わない。

        (Codex 指摘3) **候補受付そのものも凍結する** (`_post_boundary_armed
        =False`)。`_has_ever_placed` が既に True (前試合で設置済み) でも
        毎回の境界で必ず再凍結し、`is_real_new_game_start()` を観測する
        まで再アームしない。2試合目以降のまちうけ相当の背景誤検出を
        防ぐための独立ゲート (モジュール docstring 参照)。

        【設計の穴の是正 2026-08-25 第2版・第3版 (Codex 承認条件対応)】
        猶予中 (未確定・未解除) の候補が境界検知時に残っていた場合、
        **以下をすべて満たす場合に限り**その場で死亡確定する
        (`self.confirmed = True`)。人が本当に詰んだとき試合はその場で
        終わるため、「ネクストが `stationary_confirm_sec` 秒不動」という
        確定条件は真の死亡では原理的に満たされない一方、無条件確定は
        Codex 指摘により危険と判断された (モジュール docstring「候補の
        まま試合終了 = 死亡確定」節参照)。

        1. `suppress_confirm=False` (呼出元が両側同時 pending=ambiguous と
           判定していない、`resolve_boundary_confirmations()` 参照)。
        2. `game_idx` が候補発生時と一致 (別ゲームへの越境確定を防ぐ。
           凍結ガードにより構造的には到達しない想定の防御的チェック)。
        3. 猶予中に生存証拠 (next 変化/新規ツモ落下) が観測されていない
           (`_observe_pending` が毎フレーム同じ条件で即座に解除するため
           構造的には到達しない想定の防御的チェック、二重に守る)。

        凍結中は候補自体が絶対に発生しない (`update()` の `frozen` 判定が
        候補判定より先に評価され、毎フレーム `_reset_transition_tracking()`
        される) ため、まちうけ画面での誤確定は再発しない。own chain
        開始で既に解除済みの候補は `_pending_source` が None のため対象外
        (user 伝授「連鎖中は窒息としない」を壊さない)。

        Args:
            game_idx: この境界イベントが属する試合の game_idx (呼出元が
                debounce による加算より前の値を渡すこと、
                `scripts/visualize_advantage_overlay.py` 側の配線参照)。
                既定 None (無指定時はチェック自体をスキップ、backwards
                compat)。
            suppress_confirm: 呼出元が「両側同時 pending で勝敗側を一意に
                決められない (ambiguous)」と判定した場合に True を渡す。
                True の間はどれだけ条件を満たしていても確定しない。

        Returns:
            (source, outcome)。source は境界検知の瞬間に猶予中だった候補の
            発生源 (`DEATH_SOURCE_PLACEMENT`/`DEATH_SOURCE_OJAMA`)。候補が
            無ければ None。outcome は
            "no_candidate" (候補なし) /
            "confirmed" (その場で死亡確定) /
            "suppressed_ambiguous" (両側 pending で確定を抑制) /
            "rejected_game_idx_mismatch" (別ゲームの候補、確定しない) /
            "rejected_survival_evidence" (生存証拠を再検知、確定しない)
            のいずれか。呼出元は
            `DeathConfirmStats.record_boundary_outcome(source, outcome)`
            へそのまま渡すことで母数付きに記録できる。
        """
        pending_source = self._pending_source
        pending_game_idx = self._pending_game_idx
        next_at_start = self._next_key_at_candidate_start
        next_valid_now = self._next_valid_now
        next_now = self._last_next_key
        self._reset_transition_tracking()
        outcome = self._classify_boundary_outcome(
            pending_source, pending_game_idx, next_at_start,
            next_valid_now, next_now, game_idx, suppress_confirm)
        if outcome == "confirmed":
            self.confirmed = True
            self.confirmed_source = pending_source
            self.confirmed_game_idx = pending_game_idx
        if self.confirmed:
            self._awaiting_new_game_clear = True
        self._post_boundary_armed = False
        return pending_source, outcome

    def _classify_boundary_outcome(
        self, pending_source: str | None, pending_game_idx: int | None,
        next_at_start: object, next_valid_now: bool, next_now: object,
        game_idx: int | None, suppress_confirm: bool,
    ) -> str:
        """`on_game_boundary()` の判定本体 (内部ヘルパー、純粋な分岐のみ)。

        引数はすべて `on_game_boundary()` が状態リセット前に読み取った
        スナップショット (self を直接読まないことで、判定条件がリセット
        順序に依存しないことを保証する)。
        """
        if pending_source is None:
            return "no_candidate"
        if suppress_confirm:
            return "suppressed_ambiguous"
        if (game_idx is not None and pending_game_idx is not None
                and game_idx != pending_game_idx):
            return "rejected_game_idx_mismatch"
        if (next_valid_now and next_at_start is not None
                and has_next_key_changed(next_at_start, next_now)):
            return "rejected_survival_evidence"
        return "confirmed"

    def update(
        self, curr_state: str, death_cell_occupied: bool, t_sec: float,
        next_key: object, is_match_active: bool = True,
        game_idx: int | None = None,
    ) -> tuple[str | None, float | None]:
        """1フレーム分の観測を反映し、(事象名, 確定遅延秒) を返す。

        Args:
            curr_state: 今フレームの own state 名。
            death_cell_occupied: 今フレームの確定盤面の死亡セル占有
                (STABLE 以外のフレームでは無視されるため False で構わない)。
            t_sec: 現在時刻 (秒)。候補発生時刻・ネクスト最終変化時刻の
                記録専用。
            next_key: 今フレームのネクスト識別値 (`SideResult.next_pair`
                を渡す想定。同値なら「動いていない」とみなす)。
            is_match_active: `PipelineResult.is_match_active`。既定 True
                (無指定時は従来動作)。**実測でこのガード単体は無効**
                (モジュール docstring 参照)。以下 `game_idx` と合わせて
                「凍結を検討する入口」の一部として扱う (最終判定は
                `_has_ever_placed`、下記参照)。
            game_idx: 試合境界カウンタ (`scripts/visualize_advantage_
                overlay.py` の `game_idx`)。既定 None (無指定時は従来動作、
                backwards compat)。**2026-08-25 user指示により補助信号に
                格下げ**: `game_idx==0`/`is_match_active=False` は単独では
                凍結を確定しない。凍結の主判定は「その side でまだ一度も
                設置 (TSUMO_FALL→STABLE) を観測していないか」
                (`is_placement_transition()` を積算した `_has_ever_placed`、
                content-based、動画の編集方針に依存しない) であり、
                `is_match_active=False` または `is_pre_match_game_idx()`
                が「凍結を検討する入口」を開き、その入口が開いている間
                だけ `_has_ever_placed` が凍結の継続/解除を決める。これに
                より「まちうけ画面を含まず試合の途中から始まる動画」
                (`game_idx` が最初の試合の間ずっと0のまま) でも、最初の
                設置が観測された瞬間に凍結が解ける (モジュール docstring
                「まちうけ画面での誤確定」節参照)。**さらに (Codex 指摘3)
                `on_game_boundary()` 後は `_has_ever_placed` の値によらず
                `_post_boundary_armed=False` で追加凍結され、
                `is_real_new_game_start()` を観測するまで解けない。**

        Returns:
            (event, delay_sec)。event は
            "candidate_placement" / "candidate_ojama" /
            "released_placement" / "released_ojama" /
            "confirmed_placement" / "confirmed_ojama" /
            None (この既存フレームでは無変化) のいずれか。
            delay_sec は event が "confirmed_*" のときのみ非None
            (候補発生から確定までの秒数、診断用)。
        """
        raw_prev_state = self._track_has_ever_placed(curr_state)
        # (Codex 指摘3) 「本当の新試合開始」= 設置完了 + 有効 next。
        # confirmed 消去 (_awaiting_new_game_clear) と境界後の再アーム
        # (_post_boundary_armed) の両方が同じ検証済み事象 (単一の純関数
        # 呼び出し) で解決する (二重実装防止、is_real_new_game_start の
        # 内部で is_placement_transition を呼ぶ、モジュール docstring参照)。
        real_start = is_real_new_game_start(raw_prev_state, curr_state, next_key)
        if self._awaiting_new_game_clear and real_start:
            self.confirmed = False
            self.confirmed_source = None
            self.confirmed_game_idx = None
            self._awaiting_new_game_clear = False
        if not self._post_boundary_armed and real_start:
            self._post_boundary_armed = True
        pre_match_entry = not is_match_active or is_pre_match_game_idx(game_idx)
        frozen = (
            (pre_match_entry and not self._has_ever_placed)
            or not self._post_boundary_armed
        )
        if frozen:
            # 試合外の可能性がある区間 (game_idx==0 等、まだ一度も設置を
            # 観測していない場合) か、境界後の再アーム待ちのいずれか。
            # 遷移追跡・ネクスト不動タイマーの両方を破棄する。確定済み
            # フラグ (confirmed) はここでは変更しない (on_game_boundary と
            # 同じ「消去待ち」設計を保つため)。
            self._reset_transition_tracking()
            return None, None
        self._update_next_change_tracking(next_key, t_sec)
        prev_state = self._prev_state
        self._prev_state = curr_state
        if prev_state is None:
            return None, None  # 初回フレームは遷移そのものを判定できない
        if self._pending_source is None:
            return self._observe_candidate(prev_state, curr_state,
                                            death_cell_occupied, t_sec,
                                            game_idx), None
        return self._observe_pending(prev_state, curr_state, t_sec)

    def _reset_transition_tracking(self) -> None:
        """候補用の遷移追跡とネクスト不動タイマーを破棄する (内部ヘルパー)。

        凍結時 (試合外/境界後未アーム) と `on_game_boundary()` の両方で
        共通利用する (二重実装防止)。`confirmed`/`_has_ever_placed`/
        `_post_boundary_armed` 等の跨フレーム状態フラグは対象外。
        """
        self._prev_state = None
        self._pending_source = None
        self._pending_since_sec = None
        self._pending_game_idx = None
        self._next_key_at_candidate_start = None
        self._last_next_key_seen = False
        self._last_next_change_sec = None
        self._next_valid_now = False

    def _track_has_ever_placed(self, curr_state: str) -> str | None:
        """凍結の有無に関わらず生の state 遷移を追跡し、一度でも設置
        (`TSUMO_FALL→STABLE`) を観測したら `_has_ever_placed` を立てる
        (内部ヘルパー)。

        凍結中は `_prev_state` (死亡候補用) が毎フレーム None にリセット
        されるため、凍結中に起きた設置を見失わないよう `_raw_prev_state`
        で完全に独立して追跡する。`_has_ever_placed` は一度 True になれば
        `on_game_boundary()` でもリセットされない (user 指示: 「動画を
        通して一度でも設置を見たら以後は凍結しない、が安全側」)。

        Returns:
            更新前の `_raw_prev_state` (=直前フレームの生 state、凍結中も
            途切れない)。呼出元は `is_real_new_game_start()` の
            `prev_state` 引数にそのまま渡す (二重実装防止)。
        """
        raw_prev_state = self._raw_prev_state
        self._raw_prev_state = curr_state
        if not self._has_ever_placed and is_placement_transition(
                raw_prev_state, curr_state):
            self._has_ever_placed = True
        return raw_prev_state

    def _update_next_change_tracking(self, next_key: object, t_sec: float) -> None:
        """ネクスト識別値の変化を毎フレーム追跡する (内部ヘルパー)。

        候補発生前から継続して追跡する (候補発生時点で「直近いつ変化
        したか」が既に分かっている必要があるため)。初回観測は基準値の
        設定のみで「変化」とはみなさない。

        (Codex 指摘2) `next_key is None` は「未検知」であり「動いていない
        証拠」ではない。None の間は `_next_valid_now=False` とし、次に
        有効な値を観測した時点で基準時刻を打ち直す (測り直す)。
        """
        if next_key is None:
            self._last_next_key_seen = False
            self._next_valid_now = False
            return
        self._next_valid_now = True
        if not self._last_next_key_seen:
            self._last_next_key, self._last_next_key_seen = next_key, True
            self._last_next_change_sec = t_sec
            return
        if has_next_key_changed(self._last_next_key, next_key):
            self._last_next_key = next_key
            self._last_next_change_sec = t_sec

    def _observe_candidate(
        self, prev_state: str, curr_state: str,
        death_cell_occupied: bool, t_sec: float, game_idx: int | None,
    ) -> str | None:
        """候補未発生中: 新規候補の発生だけを判定する (内部ヘルパー)。"""
        src = classify_death_candidate_source(
            prev_state, curr_state, death_cell_occupied)
        if src is None:
            return None
        self._pending_source = src
        self._pending_since_sec = t_sec
        # 【第3版、Codex 承認条件対応】候補発生時点の game_idx と next を
        # スナップショットする (`on_game_boundary()` の越境確定防止/生存
        # 証拠再検知に使う、`_classify_boundary_outcome`/`_is_survival_
        # evidence` 参照)。
        self._pending_game_idx = game_idx
        self._next_key_at_candidate_start = (
            self._last_next_key if self._next_valid_now else None)
        return f"candidate_{src}"

    def _clear_pending(self) -> None:
        """猶予中の候補状態だけを破棄する (内部ヘルパー、解除・確定の両方
        から共通利用する。二重実装防止)。"""
        self._pending_source = None
        self._pending_since_sec = None
        self._pending_game_idx = None
        self._next_key_at_candidate_start = None

    def _is_survival_evidence(self, prev_state: str, curr_state: str) -> bool:
        """猶予中に生存証拠 (新規ツモ落下開始/next 変化) が観測されたか
        (内部ヘルパー、2026-08-25 第3版 Codex 承認条件対応)。

        死亡していれば新しいツモは配られず next も変化しないため、
        どちらか一方でも観測されれば「死んでいない」と判断してよい
        (モジュール docstring「候補のまま試合終了 = 死亡確定」節、
        Codex 承認条件「候補後に next 変化・次の TSUMO 開始などの生存
        証拠がない」参照)。next 側は候補発生時点のスナップショット
        (`_next_key_at_candidate_start`) との比較で判定する (候補発生
        より前の変化を誤って拾わないため)。
        """
        if is_new_tsumo_fall_start(prev_state, curr_state):
            return True
        if not self._next_valid_now or self._next_key_at_candidate_start is None:
            return False
        return has_next_key_changed(
            self._next_key_at_candidate_start, self._last_next_key)

    def _observe_pending(
        self, prev_state: str, curr_state: str, t_sec: float,
    ) -> tuple[str | None, float | None]:
        """猶予中: own chain 開始/生存証拠で解除、ネクスト不動が閾値を
        超えたら確定。

        (Codex 指摘1) 不動時間の起点は「候補発生時刻」と「最後に有効な
        next が変化した時刻」の**遅い方** (=大きい方) を使う。候補発生
        より前からの不動履歴だけで猶予をショートカットしない
        (`max(pending_since_sec, last_next_change_sec)`)。
        (Codex 指摘2) 今フレームの next が無効 (None が継続中) の間は
        確定判定そのものを保留する (`_next_valid_now` が False)。
        (2026-08-25 第3版) own chain 開始に加え、生存証拠 (next 変化/
        新規ツモ落下) でも即座に解除する (`_is_survival_evidence`)。
        """
        source = self._pending_source
        since = self._pending_since_sec
        if is_own_chain_start(prev_state, curr_state):
            self._clear_pending()
            return f"released_{source}", None
        if self._is_survival_evidence(prev_state, curr_state):
            self._clear_pending()
            return f"released_survival_{source}", None
        if not self._next_valid_now or self._last_next_change_sec is None:
            return None, None
        baseline = self._last_next_change_sec
        if since is not None and since > baseline:
            baseline = since
        stationary_sec = t_sec - baseline
        if stationary_sec < self.stationary_confirm_sec:
            return None, None
        self.confirmed = True
        self.confirmed_source = source
        self.confirmed_game_idx = self._pending_game_idx
        self._clear_pending()
        delay = (t_sec - since) if since is not None else None
        return f"confirmed_{source}", delay

    def resolved_is_dead(self) -> bool:
        """dump 列に記録する確定済み is_dead 値。

        候補中・猶予中 (未確定) は False (=死亡を主張しない)。
        確定後は True のまま (次の `is_real_new_game_start()` 検知まで
        固定、= sticky。`on_game_boundary()` docstring の根治参照)。
        """
        return self.confirmed

    def resolved_is_dead_for_game(self, game_idx: int | None) -> bool:
        """指定試合に属する死亡確定だけを物理判定へ返す。

        ``resolved_is_dead()`` の sticky 値は境界後の監査記録に必要なため
        維持する。一方、交換 episode の勝敗方向には前試合の確定を渡さない。
        game_idx 無指定、または旧呼出元由来で確定側が無指定なら従来互換で
        ``confirmed`` を返す。
        """
        if game_idx is None or self.confirmed_game_idx is None:
            return self.confirmed
        return self.confirmed and self.confirmed_game_idx == game_idx

    def pending_elapsed_sec(self, t_sec: float) -> float | None:
        """猶予中の経過秒数 (診断・遅延分布測定専用、判定には使わない)。"""
        if self._pending_since_sec is None:
            return None
        return t_sec - self._pending_since_sec

    def has_pending_candidate(self) -> bool:
        """候補が猶予中 (未確定・未解除) かどうか (診断専用)。"""
        return self._pending_source is not None


@dataclass
class DeathConfirmStats:
    """候補/解除/確定/期限切れの母数付きカウンタ (発生源別、2026-08-25)。

    「0 が『起きていない』のか『測っていない』のか」を区別するため、
    全カウントは `frames_total` (record() 呼び出し回数=観測フレーム数、
    1P/2P 分を両方カウントする) を分母として表示する
    (memory `feedback_zero_needs_denominator_2026-08-25`)。
    確定までの遅延 (秒) は `confirm_delays_sec` にサイド分含めて蓄積する
    (診断用、集計は呼出元が担う)。
    `confirmed_at_boundary_*` は「猶予中の候補が試合境界検知の瞬間まで
    残っており、その場で死亡確定した」数 (2026-08-25 第2版で新設。
    `DeathConfirmTracker.on_game_boundary()` の戻り値をそのまま渡す)。
    `expired_at_boundary_*` は旧設計 (指摘1修正直後、候補のまま境界に
    達したら「閾値未到達で消滅」させていた) の名残であり、
    **2026-08-25 第3版で「別ゲームへの越境確定を安全側で拒否した」件数の
    記録先として再利用する** (`record_boundary_outcome()` の
    `rejected_game_idx_mismatch` 分岐、`DeathConfirmTracker.
    on_game_boundary` docstring 参照)。凍結ガードにより構造的には
    到達しない想定の防御的チェックのため、通常は0のまま。

    【2026-08-25 第3版、Codex 承認条件対応】以下を追加:
    - `total_boundaries`: `resolve_boundary_confirmations()` (または
      呼出元) が1境界イベント (両サイド分をまとめて1回) につき1回だけ
      呼ぶ母数。
    - `pending_at_boundary`/`boundary_confirmed`/
      `boundary_rejected_survival_evidence`: 1サイド単位の件数
      (`record_boundary_outcome()` が `on_game_boundary()` の戻り値
      から分類する)。
    - `ambiguous_both_pending`: 両側同時に猶予中の候補が残っており
      勝敗側を一意に決められず確定を抑制した回数 (1境界イベントにつき
      最大1回)。
    - `threshold_confirmed`/`released_by_chain`/`released_by_next_or_tsumo`:
      `record()` が event の種類に応じて自動的に加算するフラット集計
      (Codex 指定のカウンタ名、既存の発生源別フィールドと並行して持つ)。
    """

    frames_total: int = 0
    candidate_placement: int = 0
    candidate_ojama: int = 0
    released_placement: int = 0
    released_ojama: int = 0
    released_survival_placement: int = 0
    released_survival_ojama: int = 0
    confirmed_placement: int = 0
    confirmed_ojama: int = 0
    expired_at_boundary_placement: int = 0
    expired_at_boundary_ojama: int = 0
    confirmed_at_boundary_placement: int = 0
    confirmed_at_boundary_ojama: int = 0
    confirm_delays_sec: list[float] = field(default_factory=list)
    # 【第3版、Codex 承認条件対応】境界判定の母数付きカウンタ (クラス
    # docstring 参照)。
    total_boundaries: int = 0
    pending_at_boundary: int = 0
    boundary_confirmed: int = 0
    boundary_rejected_survival_evidence: int = 0
    ambiguous_both_pending: int = 0
    threshold_confirmed: int = 0
    released_by_chain: int = 0
    released_by_next_or_tsumo: int = 0

    def record(self, event: str | None, delay_sec: float | None = None) -> None:
        """1回 (1サイド1フレーム) の観測結果を集計する。

        event はフィールド名そのもの (`candidate_placement` 等) として
        扱う。未知の値が来たら AttributeError で早期に気付ける設計
        (fail-silent を避ける、`getattr`/`setattr` を直接使う)。
        あわせて event の種類に応じて Codex 指定のフラット集計
        (`threshold_confirmed`/`released_by_chain`/
        `released_by_next_or_tsumo`) も自動的に加算する (二重記録防止:
        呼出元が個別に加算する必要はない)。
        """
        self.frames_total += 1
        if event is None:
            return
        setattr(self, event, getattr(self, event) + 1)
        if event.startswith("confirmed_") and delay_sec is not None:
            self.confirm_delays_sec.append(delay_sec)
        if event in ("confirmed_placement", "confirmed_ojama"):
            self.threshold_confirmed += 1
        elif event in ("released_placement", "released_ojama"):
            self.released_by_chain += 1
        elif event in ("released_survival_placement", "released_survival_ojama"):
            self.released_by_next_or_tsumo += 1

    def record_expired_at_boundary(self, source: str) -> None:
        """境界検知時に猶予中の候補が「別ゲームへの越境」等の安全側の
        理由で確定を拒否された場合に呼ぶ (2026-08-25 第3版で意味を再定義、
        クラス docstring 参照)。`record_boundary_outcome()` から
        `rejected_game_idx_mismatch` のときに内部利用される。"""
        setattr(self, f"expired_at_boundary_{source}",
                getattr(self, f"expired_at_boundary_{source}") + 1)

    def record_confirmed_at_boundary(self, source: str) -> None:
        """境界検知時に猶予中の候補が残っており、その場で死亡確定した
        場合に呼ぶ (2026-08-25 第2版で新設)。
        `record_boundary_outcome()` から `confirmed` のときに内部利用
        される。"""
        setattr(self, f"confirmed_at_boundary_{source}",
                getattr(self, f"confirmed_at_boundary_{source}") + 1)

    def record_boundary_outcome(self, source: str | None, outcome: str) -> None:
        """`DeathConfirmTracker.on_game_boundary()` の戻り値 (source,
        outcome) を1サイド分、母数付きで記録する (2026-08-25 第3版、
        Codex 承認条件対応)。呼出元は1境界イベントにつき1P/2P それぞれ
        1回ずつ呼ぶこと (`total_boundaries` は別途
        `record_boundary_check()` で1回だけ数える、二重にしない)。
        """
        if outcome == "no_candidate":
            return
        self.pending_at_boundary += 1
        if outcome == "confirmed":
            self.boundary_confirmed += 1
            if source is not None:
                self.record_confirmed_at_boundary(source)
        elif outcome == "rejected_survival_evidence":
            self.boundary_rejected_survival_evidence += 1
        elif outcome == "rejected_game_idx_mismatch":
            if source is not None:
                self.record_expired_at_boundary(source)
        # "suppressed_ambiguous" は pending_at_boundary のみ加算する
        # (ambiguous_both_pending 自体は record_ambiguous_boundary() が
        # 1境界イベントにつき1回だけ数える、二重計上防止)。

    def record_boundary_check(self) -> None:
        """1回の境界検知処理 (両サイド分をまとめた1イベント) ごとに1回
        呼ぶ (`total_boundaries` の母数、2026-08-25 第3版)。"""
        self.total_boundaries += 1

    def record_ambiguous_boundary(self) -> None:
        """両側同時に猶予中の候補が残っており、勝敗側を一意に決められず
        確定を抑制した境界イベントを1回として数える (2026-08-25 第3版、
        1境界イベントにつき1回)。"""
        self.ambiguous_both_pending += 1

    def summary(self) -> str:
        """母数付きの可視化文字列。"""
        cand = self.candidate_placement + self.candidate_ojama
        released = self.released_placement + self.released_ojama
        released_survival = (
            self.released_survival_placement + self.released_survival_ojama)
        confirmed = self.confirmed_placement + self.confirmed_ojama
        confirmed_boundary = (
            self.confirmed_at_boundary_placement + self.confirmed_at_boundary_ojama)
        expired = self.expired_at_boundary_placement + self.expired_at_boundary_ojama
        n = len(self.confirm_delays_sec)
        delay_txt = "n/a"
        if n > 0:
            xs = sorted(self.confirm_delays_sec)
            median = xs[n // 2] if n % 2 == 1 else (xs[n // 2 - 1] + xs[n // 2]) / 2
            delay_txt = f"中央値{median:.3f}s 最大{xs[-1]:.3f}s (n={n})"
        return (
            f"候補 {cand}/{self.frames_total}行 "
            f"(設置由来 {self.candidate_placement} / おじゃま由来 {self.candidate_ojama}) / "
            f"解除(連鎖開始) {released}/{cand} "
            f"(設置由来 {self.released_placement} / おじゃま由来 {self.released_ojama}) / "
            f"解除(生存証拠) {released_survival}/{cand} "
            f"(設置由来 {self.released_survival_placement} / "
            f"おじゃま由来 {self.released_survival_ojama}) / "
            f"確定 {confirmed}/{cand} "
            f"(設置由来 {self.confirmed_placement} / おじゃま由来 {self.confirmed_ojama}) / "
            f"境界で確定 {confirmed_boundary}/{cand} "
            f"(設置由来 {self.confirmed_at_boundary_placement} / "
            f"おじゃま由来 {self.confirmed_at_boundary_ojama}) / "
            f"境界で越境拒否 {expired}/{cand} "
            f"(設置由来 {self.expired_at_boundary_placement} / "
            f"おじゃま由来 {self.expired_at_boundary_ojama}) / "
            f"確定遅延 {delay_txt} / "
            f"境界イベント総数 {self.total_boundaries} "
            f"(境界時pending {self.pending_at_boundary} / "
            f"境界確定 {self.boundary_confirmed} / "
            f"境界生存証拠拒否 {self.boundary_rejected_survival_evidence} / "
            f"両側ambiguous {self.ambiguous_both_pending}) / "
            f"閾値確定(flat) {self.threshold_confirmed} / "
            f"連鎖解除(flat) {self.released_by_chain} / "
            f"生存証拠解除(flat) {self.released_by_next_or_tsumo}"
        )


def resolve_boundary_confirmations(
    tracker1: DeathConfirmTracker, tracker2: DeathConfirmTracker,
    game_idx: int | None, stats: DeathConfirmStats | None = None,
) -> tuple[tuple[str | None, str], tuple[str | None, str]]:
    """1P/2P 両トラッカー分の試合境界処理をまとめて行う (2026-08-25 第3版、
    Codex 承認条件対応の外部ヘルパー、二重実装防止)。

    両側同時に猶予中の候補が残っている場合は勝敗側を一意に決められない
    (ambiguous) ため、**安全側でどちらも確定させない**
    (`suppress_confirm=True` を両方に渡す、Codex 承認条件「両側に候補が
    ある場合、両方を無条件に死亡確定しない」対応)。呼出元
    (`scripts/visualize_advantage_overlay.py`) は1境界イベントにつき
    本関数を1回だけ呼ぶこと。

    Args:
        tracker1: 1P 側の `DeathConfirmTracker`。
        tracker2: 2P 側の `DeathConfirmTracker`。
        game_idx: この境界イベントが属する試合の game_idx (debounce に
            よる加算より前の値、呼出元の配線参照)。
        stats: 母数付きカウンタ。None なら記録しない (既定 OFF 時の
            コスト回避、backwards compat)。

    Returns:
        (tracker1 の (source, outcome), tracker2 の (source, outcome))。
        各要素は `DeathConfirmTracker.on_game_boundary()` の戻り値と同じ。
    """
    ambiguous = (
        tracker1.has_pending_candidate() and tracker2.has_pending_candidate())
    r1 = tracker1.on_game_boundary(game_idx=game_idx, suppress_confirm=ambiguous)
    r2 = tracker2.on_game_boundary(game_idx=game_idx, suppress_confirm=ambiguous)
    if stats is not None:
        stats.record_boundary_check()
        stats.record_boundary_outcome(*r1)
        stats.record_boundary_outcome(*r2)
        if ambiguous:
            stats.record_ambiguous_boundary()
    return r1, r2

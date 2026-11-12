"""本番構成 (採用済みフラグ) の単一の情報源 (2026-08-08).

## なぜ必要か — 退行の真因
本プロジェクトは backwards compat のため **改善を必ず「フラグ追加 + 既定 OFF」で
入れる**規約になっている (CLAUDE.md)。 この規約自体は再現性のために正しい。
問題は **採用が決まった後に「どのフラグを付けるのが正解か」が一元化されていない**
ことだった。 正解がジョブファイルや個別スクリプトに直書きで散在するため、
新しくデモや評価を作るたびに手でフラグを並べることになり、 **過去の改善が
まるごと抜け落ちる**。

実際に 2026-08-08 のデモ生成で `--early-fire-reaction` を付け忘れ、
「連鎖中に連鎖力を判断する機能がなくなった」「有利不利が大雑把にしか動かない」
という退行が起きた (user 指摘)。 機能は 2026-07-29 の user レビュー指摘に
対応して実装済みだったが、 既定 OFF のまま誰も付けなければ存在しないのと同じ
だった。

## 使い方
    from src.production_config import advantage_overlay_flags
    cmd += " " + advantage_overlay_flags()

## 運用ルール
- **採用が決まったフラグは必ずここに追記する**。 追記しない限り「採用済み」とは
  見なさない。
- 各エントリに **採用日と根拠** (計測結果・レビュー) を必ず書く。 後から
  「なぜこれが有効なのか」を辿れるようにするため。
- 既定 OFF のまま残すフラグ (実験中・未採用) はここに入れない。
"""
from __future__ import annotations

# 2026-10-24: 94試合の録画調査で保留中の供給欠落14件と文字外背景の不一致1件。
# 実機での採用検証前のため、いずれも試作・既定OFF。
LIVE_MATCH_END_FEED_ON_HOLD_DEFAULT = False
MATCH_END_YATTA_TEXT_TEMPLATE_DEFAULT = False

# 2026-10-24: 画面の確認の失敗 (外枠の 1 辺だけの被覆不足) で盤面・台帳・試合の状態を捨てない、期限つきの保留。
# 根拠: 24b 94 試合の保存記録で、試合中の reset (epoch+1・pipe.reset) が 91/86 回 (0.97/0.91 回/試合)、
# 直後に必ず内部の試合開始が増える (91/91・86/86)。保存記録の再生で期限 8 秒なら 76%/74% を救える見込み。
# **2026-10-25 06:30 user 採用** (「いや、採用してから図る」): 24c の入口の NEW の設定 (new_on/puyo_live.json) で ON にし、
# 次の OBS 試験で効き (試合中の reset・内部の試合開始の数・表示が止まる秒数) を測る。コードの既定値は、
# 試験の期待値を直す次の組み立てで True に切り替える (ここでは False のまま、設定で ON)。
# 設計: <work>/codex_jobs/verify_hold_split_20261024/DESIGN.md
LIVE_VERIFY_SOFT_HOLD_DEFAULT = False

# 2026-11-02: W64 の入口で1秒刻みの枠不合格が続くため、入口だけ0.25秒で再試行。
# PC1 の live_w64_entry_verify_20261102/PROMPT 13:3x追補。実機での採用前につき既定OFF。
LIVE_VERIFY_ENTRY_FAST_RETRY_DEFAULT = False

# 2026-11-02 PC1 chain_continuity_fix: 発火盤の裏付けと式による継続を検証する。採用前につき既定OFF。
LIVE_CHAIN_FORECAST_RECHECK_DEFAULT = False
LIVE_CHAIN_END_FORMULA_DEFAULT = False

# 2026-11-03: 0.6.1準備。サイト未開設のため未採用・既定OFF。確認後の手動送信だけ。
# 2026-11-06 user 採用 (0.6.2 で ON)。本番の受け口 (Pages Functions + D1) が動くことを PC1 で確認済み
# (管理の集計 200)。11/4 にプレビューで PC1 の collect_feedback.mjs が 41 件回収。送信は確認後の手動だけ。
# 2026-11-09 user 採用: OOF 216本・5分割・11,711,018行で学習。
# newsealed 152試合の LL 0.51925→0.51727、悪化層0。事前の線(2)
# (newsealedだけの癖の層)は不合格だったことを留保して、0.6.4で既定ON。
DISTRIBUTION_PROB_TEMPERATURE = True

# 2026-11-10 user 採用: PC2 calib3/3b と8場面のレビュー。温度較正後に適用。
# 2026-11-12 user「入れないかな おすすめで」で OFF: PC2 eval5 の副比較 (光 ON どうし 104 試合) で
# ずらし ON の LL +0.0010 [+0.00037, +0.0016]、撃ち返し直後 0〜1.5 秒など 4/42 層で Holm 有意に悪化。
# 光の直しが撃ち返し直後の遅れを直したため二重の補正になったと判断。
DISTRIBUTION_COUNTER_FIRE_SHIFT = False
# 2026-11-11 user 採用 (0.6.5 で ON、「不具合だったら後で戻す」)。W69 光の直し (4ebd3813):
# step8 で (c) 合格・速さ 4/4 開始で合格、PC2 eval3 の光ON短再生 9,000 コマでコマ落ち 0。
# eval3 の全編の比較 (段B) は PC2 の停止で未完了のまま採用した。結果が出たら照合し、悪ければ False に戻す。
# 2026-11-11 21:2x Claude: False に戻す (user「不具合だったら戻そう」)。宣伝の再生 (promo_reel50_light RESULT2) で
# 長い連鎖中に光の状態処理が 7.8 秒・9.8 秒かかり、評価の 2 秒の期限切れで第 43 試合の 153/300 コマが非表示。
# 光の読み自体は第 42 試合で 2P の 14 連鎖を 1.7 秒早く捉えて正しい。処理時間の上限を直して確かめたら ON に戻す。
# 2026-11-12 00:2x Claude: ON に戻す。8475d4a4 (light_stall_fix RESULT2) で段間の進行を高速化し、
# 色の組み合わせにプレイヤーごと 8,192 通りの上限 (超えた側だけ基礎予測)。第 43 の非表示 0/300・state 最大 443ms、
# 第 42 の早読みと 3 候補を保持、step8 の 20 走行で最大 323ms・p95 14ms、(c1) 9.89→8.62pt、関連 712・公開 1,830 で新しい失敗 0。
DISTRIBUTION_LIGHT_ROBUST_CELLS = True
# 2026-11-12 user 採用「OK」: W68 (操作中のツモが確定盤に入り窒息の印・探索が点滅) の直し。
# 推論に渡す盤だけ上 3 段の新しい対を仮で外し、NEXT の繰り上がりか 1 秒で入れる (17091169)。
# 7 本で操作対 11/13 を抑制・正当 10/10 保持・上段の着地の推論の遅れ 中央 7・最大 30 コマ、旗 OFF は 42,094 入力で不一致 0。
DISTRIBUTION_W68_INFERENCE_PENDING = True
# 2026-11-11 Claude: 中盤の研究候補v2。配信再生と採否判断までは未採用、既定OFF。
# 2026-11-12 user 採用「0.6.5 は中盤既定 ON」: PC2 研究 11 節 (152 本・層の中と減衰中 349,602 行) で
# LL 0.68960→0.68902 (差 −0.00058 [−0.0011,−0.00015])・ECE 0.0044→0.0021・5 組すべて良い。2f468649・79db9c0d。
DISTRIBUTION_MIDGAME_BOARD_LAYER = True
# 2026-11-10 Claude 採用: release_065_prep の試験と release_065_final の実機確認を根拠とする。
DISTRIBUTION_UPDATE_BUTTON = True

DISTRIBUTION_SITE_FEEDBACK = True

# 2026-11-02: 専用の設定窓。PC1の採否決定前につき既定OFF（app_window_20261102）。
DISTRIBUTION_APP_WINDOW = True  # 2026-11-03 user 承認 (11-02 PC1 案、常駐窓の注文)。0.6.0 で配布。

from dataclasses import dataclass

# 2026-10-06: 起点2手補完の列挙・段別完走を融合。候補順・重み・段別得点の一致を採用条件とする。
# 根拠: b3_data/codex_tasks/b3_fire_search_speed/ の単独台・600秒全更新照合。
B3_ORIGIN_NATIVE_ENABLED = True


@dataclass(frozen=True)
class AdoptedFlag:
    """採用済みフラグ 1 件。"""

    flag: str          # CLI フラグ文字列 (値を取る場合は "--name value" 形式)
    adopted: str       # 採用日 (YYYY-MM-DD)
    reason: str        # 採用根拠 (計測結果 / レビュー)


# 配布版の表示遅延対策。2026-10-10 user承認でB3使用時に既定ON (判定値不変、Claude独立検収合格)。
DISPLAY_COMPLETION_DRIVEN_DEFAULT: bool = True
DISPLAY_FAST_EMA_DEFAULT: bool = True
DISPLAY_LATENCY_CANDIDATES: tuple[AdoptedFlag, ...] = (
    AdoptedFlag('--display-completion-driven', '2026-10-10',
        '2026-10-09 user決定・10-10 採用: B3完了で即取り込み、確定時だけ送信。保存列3,009件で判定値一致、再送155→0、旧の確定→送信待ち平均255ms解消 (Claude再計算一致)。実機遅延は未測定'),
    AdoptedFlag('--display-fast-ema', '2026-10-10',
        '2026-10-09 user決定・10-10 採用: 配布版の表示EMAだけ0.25から0.5へ。判定値不変、生値との平均乖離4.21→2.42pt'),
)

# 配布版の初回判定短縮候補。JSON専用スイッチで、既定OFF・本番未採用。
LIVE_DISABLE_PLACEMENT_RECONCILE_DEFAULT: bool = False
B3_SCORE_MATMUL_DEFAULT: bool = False
B3_SHARE_SCORE_READ_DEFAULT: bool = False
LIVE_ASYNC_SCREEN_VERIFY_DEFAULT: bool = False
LIVE_STAGE_TIMING_DEFAULT: bool = False
# 2026-10-13: user指示「デバッグ用にしばらくは L 字をデフォルトに設定し、L 字の中に状態・連鎖数・
# フィールドの読み取りを表示」。配信表示の既定を L 字 + 試験用パネル ON にする (判定・勝率は不変)。
# 戻すには LIVE_DEBUG_OVERLAY_DEFAULT を False にする (この1か所だけ)。puyo_live.json の overlay に
# 明示した値は常にこの既定より優先される。
LIVE_DEBUG_OVERLAY_DEFAULT: bool = True
LIVE_DEBUG_OVERLAY_VALUES: tuple[tuple[str, str | bool], ...] = (('layout', 'l'), ('debug_panel', True))
# 2026-10-13: 実装承認のみ。画素入力の品質・速度ゲート未達、本番未採用。
LIVE_DELTA_CELLS_DEFAULT: bool = False
LIVE_CNN_DEDUP_DEFAULT: bool = False
TERMINAL_HOLD_INPUT_RECOVERY_DEFAULT: bool = False
LIVE_LATEST_EVALUATION_DEFAULT: bool = False
B3_WARNING_PARALLEL_DEFAULT: bool = False
B3_WARNING_WORKERS: int = 4
B3_WARNING_MAX_WORKERS: int = 4
LIVE_SPEED_CANDIDATES: tuple[AdoptedFlag, ...] = (
    AdoptedFlag('terminal_hold_input_recovery (puyo_live.json)', '2026-10-12',
        'user承認・配布g候補。同一機器の画面検証失敗だけの境界では決着期限を維持。'
        '得点境界・新試合の通常評価・機器変更・5秒期限で解除する。'),
    AdoptedFlag('live_latest_evaluation (puyo_live.json)', '2026-10-12',
        'user承認・配布g候補。状態通知を保持し、古い評価要求の計算だけを省略する。'
        '同一盤面の値の一致を検証できない場合は配布でもOFFとする。'),
    AdoptedFlag('live_stage_timing (puyo_live.json)', '2026-10-12',
        'user承認・配布f候補。spoolへフレーム別の壁時計・thread CPU時間を記録し、'
        '台で再現しない仮想カメラ入力の約30msを切り分ける。表示・判定値へ計装を混ぜない。'),
    AdoptedFlag('live_async_screen_verify (puyo_live.json)', '2026-10-12',
        'user承認・配布f候補。画面検証だけを単一workerで実行し、状態更新は入力所有者で行う。'
        '入力断・形式変更前の結果を破棄し、初回合格までは保留。継続検証中は前回の状態を維持する。'),
    AdoptedFlag('b3_share_score_read (puyo_live.json)', '2026-10-12',
        'user承認・配布f候補。同じフレームの生得点だけを共有する。OCRクラス・ROI・'
        'テンプレート・しきい値・演算方式が一致しない場合は再読し、判定digest一致を出荷条件とする。'),
    AdoptedFlag('live_disable_placement_reconcile (puyo_live.json)', '2026-10-12',
        'user承認。連続フレーム前提のR1bは5fps配信55分でnext合図・修正セルとも0。'
        '配信だけ停止し、動画30fpsの既存経路は維持。再有効化は実測25fps以上を安定維持し、'
        '連続フレーム合図の有効性と判定の改善を別途検証した場合。自動切替はしない。'),
    AdoptedFlag('b3_score_matmul (puyo_live.json)', '2026-10-12',
        'B3得点OCRも既存matmul経路へ。float64と旧NCCには閾値境界差の可能性があるため、'
        '配布試験のみON。保存PNGの通知全体・B3 payloadのフレーム別digest一致を出荷条件とする。'),
    AdoptedFlag('b3_warning_parallel / b3_warning_workers (puyo_live.json)', '2026-10-12',
        '予告CNNの単体演算をセル単位で並列化。LRU操作を排他化しeval済みモデルを読み取る。'
        'cpu_threads=1は各演算ライブラリの内部並列数であり、外側のworker数とは別。'
        'OBS同居を考慮して最大4worker、既定OFF。CPU使用量と決定digestを配布前に測定。'),
)

FAST_FIRST_JUDGMENT_DEFAULT: bool = False
# 2026-10-12: build f候補。検証専用の較正は認識色を変えないため裏で取り直す。
# fast_firstがOFF、または実補正モードの場合には適用しない。
FAST_FIRST_BACKGROUND_RECHECK: bool = True
FAST_FIRST_RECHECK_CANDIDATES: tuple[AdoptedFlag, ...] = (
    AdoptedFlag('fast_first_judgment + verification_only', '2026-10-12',
        'build f試験限定。run_ffe/run_dsoakはverification設定でHSV補正非適用。'
        '厳密な範囲照合失敗による23/48試合の表示中断を、背景再収集で防ぐ。'),
)
FAST_FIRST_JUDGMENT_CANDIDATES: tuple[AdoptedFlag, ...] = (
    AdoptedFlag('fast_first_judgment (puyo_live.json)', '2026-10-11',
        '実装承認のみ・本番未採用。OBS通し63試合で試合間の再較正が中央値5.05秒。'
        '同じ入力の較正を保持し、空のSTABLE確定盤面から通知する。3手以内は通し試験で要検証。'),
)
# 配布版 (現行方式) の試合境界。入力世代切替の強制境界が、得点リセット用の5秒デバウンス枠を消費しない。
# 既定OFFなら従来 (強制境界も枠を更新) とビット一致。CLIは --[no-]forced-boundary-debounce-isolation。
FORCED_BOUNDARY_DEBOUNCE_ISOLATION_DEFAULT: bool = True
# 2026-10-12: build fだけで有効にする候補。全体既定はOFF、既存の隔離は維持。
RECOVERY_BOUNDARY_COALESCING_DEFAULT: bool = False
LIVE_PUBLIC_MATCH_NUMBER_DEFAULT: bool = False
LIVE_PUBLIC_MATCH_NUMBER_CANDIDATES: tuple[AdoptedFlag, ...] = (
    AdoptedFlag('live_public_match_number (puyo_live.json)', '2026-10-13',
        '内部キーと公開番号の分離。入力回復でも評価器・死亡・EMA・決着保持は従来どおり初期化。'
        '既定OFF、m配布限定。保存評価397435行・公開53366件は番号以外バイト一致。'
        'g/h/j/kの余計な85区切りを除去するが、kの本物1境界も公開未加算。'
        '根拠: docs/LIVE_COALESCE2_VALIDATION.md。OBS実機通しの品質合格ではない。'),
)
EVALUATION_FRESHNESS_GATE_DEFAULT: bool = False
TERMINAL_DISPLAY_HOLD_DEFAULT: bool = False
TERMINAL_DISPLAY_HOLD_CANDIDATES: tuple[AdoptedFlag, ...] = (
    AdoptedFlag('terminal_display_hold (puyo_live.json)', '2026-10-12',
        'user決定2026-10-12。次の試合の始まりまで保持し、区切り取りこぼし時の誤表示は最大5秒。'
        '別機器・試合番号変更で解除し、反復確定では期限を延長しない。'
        '境界直後の旧死亡再保持は、新試合の通常評価まで禁止。配布試験だけON。'),
)
EVALUATION_FRESHNESS_CANDIDATES: tuple[AdoptedFlag, ...] = (
    AdoptedFlag('--evaluation-freshness-gate', '2026-10-12',
        'build f試験限定。run_ffeの29.3秒間の古いlive表示を、評価完了から2秒で理由付きholdへ。'
        'confirmed_deathの決着保持は対象外にし、未決の100/0保持方針と分離する。'),
)
RECOVERY_BOUNDARY_CANDIDATES: tuple[AdoptedFlag, ...] = (
    AdoptedFlag('--recovery-boundary-coalescing', '2026-10-12',
        'build f試験限定。run_ffeの入力復帰直後の境界二重化を統合する。'
        '5秒未満の得点リセットでも死亡状態を初期化し、試合番号の加算だけ抑止。'),
)
LIVE_BRIDGE_ADOPTED: tuple[AdoptedFlag, ...] = (
    AdoptedFlag('--forced-boundary-debounce-isolation', '2026-10-11',
        'user承認 2026-10-11 / soak game11 の死亡確定持ち越し。'
        '強制境界 (入力世代切替) が得点リセット境界の5秒デバウンス枠を消費し、3.81秒後の本物の得点リセット境界 '
        '(OBS soak 通し試験 t=699.91s) が拒否されて、前試合の2P死亡確定が次試合の約42秒に残り1P勝率100%が出続けた。'
        '強制境界単独のフレームでは枠の時刻を境界前の値へ戻す (同一フレームの得点リセット境界は従来どおり枠を更新)。'
        '保存済み4記録 (soak 14,592行・test 1,318/888/174行) の再生で、修正前の再生は実game_idxと0不一致、'
        '修正後に増える境界は soak の699.91の1件のみ (他3記録は差0)。'
        '再生台 scripts/replay_forced_boundary_20261011.py、単体 tests/phase_j/test_live_forced_boundary.py'),
)


# ============================
# 認識 — 収集専用 (collect_boards_lean のみが必要とするフラグ)
# ============================
# RecognitionPipeline.load_default では既定 True だが collect_boards_lean だけ
# 既定 False のため、 収集時のみ明示指定が要る。 表示系スクリプトに渡すと
# 「unrecognized arguments」で落ちるので分けて管理する
# (2026-08-08 に実際に踏んだ)。
COLLECT_ONLY_ADOPTED: tuple[AdoptedFlag, ...] = (
    AdoptedFlag(
        "--enable-record-time-observation-fix-raw", "2026-09-19",
        "記録の直前に、画面が示している色へ確定盤面を合わせる。"
        "根因: 記録されるのは「前回の記録と盤面が変わった瞬間」だけ (last_emitted_grid 比較、"
        "実測で STABLE の side-step 17,876 に対し記録は 581 = 3.25%) なのに、"
        "確定盤面を画面へ合わせ直せる事後復旧ゲートは 8 処理frame 連続の一致を要求する "
        "(STABLE_RECOVERY_MIN_FRAMES = 8)。**盤面が変わった瞬間には修正が原理的に間に合わない。**"
        "5動画の全数計装で、記録された盤面の食い違いの 97.4% (11,245/11,542) が "
        "「書いたときは正しかったのに後から誤りになった」ものだったのはこのため。"
        "本フラグは記録の直前に限り 8frame の待ちを外し、"
        "**CNN と HSV の2系統が一致した色だけ**を採って確定盤面へ反映する。"
        "浮きぷよは作らない (既存の _check_recovery_column をそのまま通し、"
        "下から連続して埋まる範囲だけ直す。候補同士は支え合えるので列単位で直る)。"
        "隠し段 (row 0) は触らない。"
        "raw = おじゃま会計フィルタ (_apply_ojama_write_accounting_filter) を通す **前** の "
        "観測を基準にする。フィルタ後だと「画面は おじゃま なのに記録は色ぷよ」を直せず、"
        "通常版で残った誤り 241 件のうち 89 件 (37%) がそれだった。"
        "そのため SideResult へ raw_cnn_board を追加した (既定 None、読む側が居なければ no-op)。"
        "【実測 2026-09-19】5動画全長・同一条件 A/B (他フラグ同一、npz に入った盤面が分母)。"
        "母数 1,777,688 → 1,998,226、誤り 19,269 → 6,019 (-68.8%)、"
        "代理セル正解率 98.9161% → 99.6988% (+0.78ポイント)。"
        "全5動画が 99.5% 超 (video_38 99.7527 / video_39 99.6332 / video_c74 99.7105 / "
        "video_c80 99.7511 / video_c138 99.6458)。"
        "盤面の下〜中段 (行5-12) の誤りは 11,953 → 2,360 (-80.3%)。"
        "記録量も 25,334 → 28,586 step (+12.8%) と増え、取りこぼしていた局面を回収した。"
        "video_38 先頭500秒の先行 A/B でも 99.2706% → 99.7920% と同じ形が出ている。"
        "根拠 logs/diag_ab/five_videos_recordfix/、tests/test_record_time_observation_fix.py 8 PASS。"
        "既定 OFF のままでは既存挙動と bit-identical。"
        "**代理指標であり、G3 の合格判定 (セル正解率 99.5%) は人手の正解データで別途行う。**",
    ),
    AdoptedFlag(
        "--enable-chain-active-record-hold", "2026-09-18",
        "W48b。連鎖が動いている間は確定盤面を記録しない。"
        "背景: 連鎖が始まると確定盤面が『連鎖が終わった後の姿』へ差し替わるが、"
        "画面はこれからアニメを再生する。state は STABLE を名乗るのでそのまま記録され、"
        "13連鎖ならアニメ約18秒の間ずっと記録が画面より先を行く "
        "(実測 video_38 2P 578.37秒: 画面66個に対し記録9個、得点は18秒後に +79,085)。"
        "差し替えは着地経路だけでなく掛け算式の早期発火・連鎖の再生からも起きるため、"
        "どの経路が始めた連鎖でも立つ既存の保持期限 (_chain_until_Xp) を使う。新しい定数は作らない。"
        "【実測 2026-09-18】5動画・全長・同一条件 (試合範囲ゲートは両者とも外す) の A/B。"
        "書き込み経路の全数計装で『盤面側の誤り』を数えた。"
        "盤面誤り 15,617 → 9,560 セル (-38.8%)。代理のセル正解率 99.189% → 99.466%。"
        "記録枚数 27,467 → 25,520 (-7.1%)。"
        "**除いた1,947枚は1枚あたり3.11セルの誤りを含み、残した25,520枚の0.37セルより8.4倍悪い。**"
        "連鎖シミュレータ由来の誤りは 8,925 → 3,931。"
        "着地経路だけを止めた版 (--enable-landing-chain-record-hold) は -15.8% に留まったため、"
        "そちらは採用しない (本フラグが上位互換)。"
        "連鎖中に確定盤面は存在しない、というプロジェクトの既定方針 "
        "(feedback_chain_phase_physics_only) と整合する。"
        "認識は1セルも変えない。記録するかどうかだけを決める。"
        "実装 tests/test_landing_chain_record_hold_w48.py 12 PASS。"
        "既定 OFF のままでは従来挙動と bit-identical。",
    ),
    AdoptedFlag(
        "--enable-match-range-gate", "2026-09-17",
        "試合範囲外の盤面を記録しない。既存 matches.tsv "
        "(match_boundaries_v5/v4、2026-08-18〜20 に本番採用済みの自動境界検出の成果物、"
        "video_38 は43試合) を読むだけで新しい検出器は作らない。"
        "背景: collect_lean は force_in_match=True で pipeline を生成するため raw_active が "
        "常時 True になり、試合が始まっていない画面 (キャラ選択・暗転・待機) も盤面として記録していた。"
        "user が実画面で誤読と判定した f8638 (キャラ選択・143.97秒) / f8754 (暗転・145.90秒) は "
        "いずれも試合1の開始 154.0 秒より前だった。"
        "実測 A/B (video_38 先頭240秒・他フラグ同一): 盤面 182→164。"
        "共通161位置の内容差分は 0 で既存の正常記録を変えない。"
        "消えた21枚は全て試合範囲外 (21/21)。ON 後に残る範囲外は 0/164 で検収 PASS。"
        "境界データが無い動画では何も落とさない (可用性優先、件数0として記録)。"
        "実装 src/match_range_gate.py、tests/test_match_range_gate.py 6 PASS。"
        "既定 OFF のままでは従来挙動と bit-identical。",
    ),
    AdoptedFlag(
        "--enable-chain-tracker", "2026-07-30",
        "機能D単独では CHAIN 検知が実運用 0 件で、連鎖中の盤面凍結が働かない",
    ),
    AdoptedFlag(
        "--enable-stable-persistence-gate", "2026-08-18",
        "**2026-08-18 二次追加で役割変更**: 「収集ゲート (記録可否)」から"
        "「confidenceタグ付与」へ転用。当初の設計 (2026-08-18 初回採用時点、"
        "docs/BOUNDARY_MULTISIGNAL_DESIGN_2026-08-17.md §5) は `_should_emit`"
        "直前で直近 STABLE_PERSISTENCE_WINDOW_SEC=0.25 秒の盤面ROI生ピクセル"
        "diffが全て閾値未満のときだけ記録を許可し、それ以外は snapshot ごと"
        "破棄していた。だが 148再収集の実データで検証したところ"
        "(scripts/_diag_stable_persistence_loss_breakdown_2026-08-18.py) "
        "収集行数が**61%減**という致命的な被覆率損失が判明した。内訳は"
        "①窓長0.25秒だけで理論上届かない分=21.4% ②相手バースト中は自盤面が"
        "静止していても画面全体は動くため追加で弾かれる分 (窓外0.40倍 vs "
        "窓内0.33倍、bursts中により重い) ③残り約40ポイントは「静止が原理的"
        "に存在しない」ことによるもの (盤面ROIは13段分あり次ツモ落下域を含む"
        "ため、盤面が確定していても画面は動き続ける、user指摘)。しかも欠落が"
        "「おじゃまが降る前の構え」という戦術的に重要な局面に偏っており、"
        "単なるデータ量減でなく学習の偏りになることが分かった。"
        "**アーキ再判断 (2026-08-18)**: 判定ロジック自体 (実測分離ギャップ"
        "0.858<1.0<1.07に基づく汚染検出、下記の閾値根拠は変更なし) は無価値"
        "ではないため、記録拒否をやめ npz 列 `stable_persistence_confidence`"
        "(bool、末尾追加、既存列順不変) として残す方式に転用した。除外要否は"
        "学習データビルダー側のオプトインフィルタ (--exclude-match-end-locked"
        "と同じ設計思想) に委ねる (本転用時点では除外ロジック自体は未実装、"
        "列を持つところまでがスコープ)。"
        "**経緯の記録 (fail-silent警戒)**: 設計文書の検証項目④「STABLE "
        "snapshot総数の変化率 (過度な減少=閾値過剰)」は、2026-08-18の初回"
        "採用登録タイミングでは実施されておらず、この61%減という規模の"
        "退行が採用直後の148再収集停止まで表面化しなかった。以後の閾値付き"
        "ゲート導入では検証項目④相当 (行数変化率の実データ確認) を採用前に"
        "必須とする。"
        "以下は当初設計の閾値根拠 (変更なし、confidence列の値計算に引き続き"
        "使用): `_update_raw_pixel_stable` が直近 STABLE_PERSISTENCE_WINDOW_"
        "SEC=0.25 秒の盤面ROI生ピクセルdiffを計算し、閾値 STABLE_PERSISTENCE_"
        "DIFF_THRESHOLD=1.0 (実測分離ギャップ: 汚染側029最小値1.07、綺麗な"
        "21枚側の最大値0.858、0.858<1.0<1.07に収まるラウンド値、"
        "src/board_motion.py コメント参照、シーン逆算でなく物理量の分離"
        "ギャップから固定、過学習禁止規約準拠) 未満かで confidence を決める。"
        "③試合外は静止画面のため本機構では検出不能 (差分ゼロ、(b) 系列が担当)",
    ),
    AdoptedFlag(
        "--enable-boundary-multisignal", "2026-08-18",
        "game_idx境界マルチシグナル検知 (`_SharedGameCounter`/`_reconcile_"
        "boundary_anomalies`、W20/W21根治)。W22救済込みで c109/c13/c96 実測 "
        "7/8=87.5% (docs/BOUNDARY_MULTISIGNAL_DESIGN_2026-08-17.md §0)。"
        "**注記**: 今回 (2026-08-18) の採用登録タイミングでの追加A/Bは未実施。"
        "根拠は上記W22時代の実測のみであり、148再収集時に併走測定を推奨"
        "(docs/BOUNDARY_MULTISIGNAL_DESIGN_2026-08-17.md §2「(a)既存境界マル"
        "チシグナルの本採用判断」は正式なA/B測定を前提としていたが、user承認"
        "『全群採用』により本実測をもって先行登録する)",
    ),
    AdoptedFlag(
        "--enable-score-reset-requires-zero", "2026-08-20",
        "試合境界を「score が 0 になった」ときだけ認める (user 指摘"
        "「0になったら新しい試合の始まりだよ」「減るのはただの誤認」)。"
        "旧判定は減少幅 (prev - cur >= 500) だったが、試合中の score は"
        "単調増加しかしないため減少は境界の根拠にならない。"
        "**実害の実測 (39番)**: 連鎖中は score が猛烈に増える (636秒台で"
        "1.6秒に3.5万点) ため、閃光による1桁の誤読が容易に500点超の"
        "「減少」に化けていた。実測2件とも1桁誤読 (310.4秒 5,759->5,259 "
        "で差500=閾値ちょうど / 636.8秒 56,085->55,085 で差1,000)。"
        "偽境界は連鎖の真っ最中に引かれ、その区間では WIN★パネルが"
        "変化しないため試合の勝敗ラベルが丸ごと落ちていた (欠損15試合の"
        "うち10件がこの形、実画面確認済み data/verify/panel_missing_"
        "evidence_2026-08-20/)。"
        "**A/B実測 (同一動画2本、boards_lean_wirecheck→zeroreset)**: "
        "勝敗ラベル欠損 9.8%→1.0% / 収集行数 10,840→10,840 (1行も減らず) "
        "/ 試合数 111→101 (偽境界10個が消滅) / ラッチ 1.9%→1.9% (不変)。"
        "38番は全指標が完全に不変で悪化ゼロ。"
        "**逆方向の失敗も確認済み**: 0 の検知に失敗して2試合を繋げると"
        "score が恒久的に低下するため検出できる "
        "(scripts/_diag_merged_games_by_score_drop_2026-08-20.py、陽性対照"
        "で検出器の動作確認済み)。2本とも 0 件。"
        "閾値 SCORE_RESET_ZERO_MAX=0 は「新しい試合は必ず0から始まる」"
        "というゲーム仕様から決めた物理値でシーン逆算ではない",
    ),
    AdoptedFlag(
        "--enable-winner-panel-priority", "2026-08-20",
        "勝者判定で WIN★パネルを優先する (user 決定「パネル優先でいい"
        "です」)。従来は得点系統とパネル系統の2系統一致を要求し、食い違え"
        "ば判定不能に倒していた。だが得点系統の「得点が高い方が勝ち」は"
        "約98%しか成立しない近似である (user 伝授、memory reference_"
        "score_winner_98pct_2026-08-20)。パネルが片側の星だけ明確に増えた"
        "と読めた場合 (両側変化・変化なしは None になる) はそちらが正しい。"
        "**実測した食い違い3件は全て同じ形**: 39番試合6 (1P=9,987/"
        "2P=9,780 差207、パネル変化 左1/右39) / 39番試合28 (1P=630/"
        "2P=238 差392、左0/右50) / 38番試合35 (1P=1,396/2P=1,192 差204、"
        "左1/右46)。いずれも得点は僅差で1Pが上だが実際は2Pの勝ちで、"
        "得点の2%の例外が**僅差の試合に集中**している (大差なら得点通り、"
        "拮抗試合は窒息で決まるため)。パネル側は右だけ39〜50の明確な変化"
        "で正常試合 (47/31) と同水準。"
        "**見積もり (scripts/_diag_panel_priority_gain_2026-08-20.py、"
        "収集し直さず後処理を再現)**: 欠損 39番 1.8%→0.3% / 38番 "
        "0.2%→0.0%。欠損の解消だけでなく、既にラベルが付いていた試合28の"
        "誤ラベル (1P→2P) を修正する効果もある。"
        "**留保**: 上記見積もりは試合開始時刻を npz の最小 t_sec で近似"
        "した再計算のため本番と完全一致ではない。48本収集の実データで"
        "確定させる",
    ),
    AdoptedFlag(
        "--enable-boundary-newmatch-evidence", "2026-08-20",
        "境界の偽検出ゲート (コミット b646d9f)。直上の --enable-boundary-"
        "multisignal による「視覚立ち上がりでの境界確定」に対し、新試合の"
        "証拠 (両者スコア数値0 or 両者盤面ほぼ空、`_compute_newmatch_"
        "evidence`) を要求する。演出・対戦カード紹介の立ち上がりを試合開始と"
        "誤認して1試合を細切れにする偽境界を抑止する。実装が scripts/collect_"
        "boards_lean.py 内 (`_SharedGameCounter.require_newmatch_evidence`) "
        "のみで RecognitionPipeline を経由しないため、直上の boundary-"
        "multisignal と同じく COLLECT_ONLY 扱い (可視化ツールには配線しない"
        "＝ゲート対象の boundary-multisignal 自体が可視化に無いため)。"
        "**2026-08-20 配線事故の是正**: b646d9f (2026-08-19) 実装時に既定OFF"
        "のまま採用登録が漏れ、`collect_flags()` に含まれず 2026-08-19 の"
        "50本収集で一度も発火していなかった。検証スクリプト `_recollect_"
        "lockfix_2026-08-19.py` だけが手書きで渡していたため、c109 単体では"
        "改善が見えるのに全体では無変化という食い違いが生じた "
        "(memory feedback_wiring_check_needs_nongeneric_scripts_2026-08-18 "
        "の再発3回目)。"
        "実測 (同一動画10本の対等比較、`scripts/_diag_paired_boundary_fix_"
        "effect_2026-08-20.py`、下記2フラグとまとめての効果): 勝敗ラベル"
        "欠損 16.0%→3.9% / ラッチON比率 14.8%→1.7% / 収集行数 47,861→52,405 "
        "(+9.5%) / 試合数 691→530 (c132 は 114→45・欠損 43.9%→0.0%)。"
        "試合数の減少はデータ喪失ではなく分割の解消であり、同時に行数が"
        "増えている点で裏取りできる。母集団を固定した比較である点が重要 "
        "(当初の「38.3%→45.2% に悪化」は before 50本 vs after 20本という"
        "異母集団比較による錯覚だった)",
    ),
    AdoptedFlag(
        "--enable-winner-panel-crosscheck", "2026-08-18",
        "WIN★パネル数値差分による勝者判定クロスチェック (`src/win_panel.py`+"
        "`src/match_winner.py`、オフライン専用)。boundary-multisignal と同じ"
        "W22時代の実測 (7/8=87.5%) が根拠であり、**今回の追加A/Bは未実施**"
        "(同上の注記が適用される)",
    ),
    AdoptedFlag(
        "--enable-move-segmented-recording", "2026-08-18",
        "1手区切り観測スケジューラ (user承認2026-08-18、コミット9fba9ac/"
        "97cc37f)。従来の画素静止ゲート (STABLE_PERSISTENCE系) は収集行数を"
        "61%削り、欠落が「相手の連鎖中」等の戦術的に重要な局面に偏っていた "
        "(通常設置中でも95.3%棄却＝局面弁別になっていないと実測)。これを"
        "「1手ごとに1回だけ観測する」方式に置き換える: 手の区切りは"
        "NEXT繰り上がり **OR** `tsumo_count` 増分のどちらか早い方、猶予"
        "15フレーム以内で最も早く得られた確定盤面を採用 (待つのではなく"
        "上限)。おじゃま落下中は降り切るまで待つ。**実測**: 旧構成比の行数は "
        "36=82.7% / 52=77.4% / c100=77.3% (実対戦区間、data/verify/"
        "move_segment_physics_filter_2026-08-18/)。落下中ツモの写り込みは"
        "繰り上がり直後7.14%に対し猶予15フレーム後18.37%と2.6倍悪化しており"
        "「最短で撮る」設計の妥当性を裏付ける。多重記録は654行中1件のみ。"
        "減った約23%の内訳は45%が本物の汚染 (舞台背景の誤検出・物理破綻)、"
        "残りは同一手の重複統合。**既知の限界**: c100の77.3%は実対戦区間"
        "限定の値 (当初64.3%と報告されたのは測定窓の大半145/150秒が解説・"
        "対戦カード・キャラ選択だったため)。一時報告された「旧構成比88.7%」"
        "は旧構成の再現手順が記録されておらず再現不能のため撤回、本番フラグ"
        "一式で再現した約77%を正とする",
    ),
    AdoptedFlag(
        "--enable-physics-persistence-filter", "2026-08-18",
        "持続的物理制約フィルタ (user承認2026-08-18、コミット9fba9ac/"
        "97cc37f、W24教訓反映)。同一の (row, col, 違反種別) が正当な状態遷移 "
        "(TSUMO_FALL/CHAIN/OJAMA_FALL、または `tsumo_count` 増分) を挟まず"
        "2回連続observedされたときだけ棄却する (W24 `feedback_persistence_"
        "run_needs_truth_recheck_2026-08-17` の教訓により単純な連続回数"
        "閾値にはしていない)。**実測**: 実効棄却率"
        "2.4〜3.8%。物理制約違反は連鎖中に正しく集中 (自分連鎖21.7%/"
        "相手連鎖19.5%/通常設置7.1%) しており、旧画素判定 (88〜98%の狭帯で"
        "一律棄却) が局面を全く弁別していなかったことと対照的。"
        "data/verify/move_segment_physics_filter_2026-08-18/ 参照。"
        "move-segmented-recording と独立に単体動作可能 (単独使用時のテストは "
        "tests/test_collect_boards_lean.py 参照)",
    ),
)

# ============================
# 認識 — 共通 (collect_boards_lean / visualize_recognition の両方が受け付ける)
# ============================
# Phase L の全動画 regen で実際に使っている構成と一致させること。
RECOGNITION_ADOPTED: tuple[AdoptedFlag, ...] = (
    AdoptedFlag(
        "--enable-ojama-entry-gravity-settle-guard", "2026-09-17",
        "バグB根治。GRAVITY_SETTLE (連鎖の段間の落下待ち) 中に OjamaVisualDetector が "
        "OJAMA_FALL を新規発火して横取りするのを禁止する。"
        "2026-08-08 に実装・テスト済みで、採用済みの `--enable-gravity-settle-reset-on-exit` "
        "(バグC) と『対で使う』設計だったが、B だけ採用が漏れていた。"
        "C は横取りされた後の後始末で、B は横取りそのものを止める。"
        "【実測 2026-09-17】2動画・各先頭600秒・他フラグ同一 (C は両方 ON) の A/B。"
        "実画面から目視で確定した真値と突き合わせたセル単位の誤り: "
        "video_38 は内容が変わった5記録で 75→10 セル、video_39 は2記録で 2→0 セル。"
        "**合計 77→10 セル (母数 = 変化した7記録 × 78セル = 546セル)。**"
        "最大の改善は video_38 f25000 1P (416.67秒): 14連鎖の最中で実際の盤面は7セルなのに、"
        "OFF は18秒前の連鎖開始前の盤面をそのまま記録していた (誤り68/78、余計に置いた65)。"
        "ON は誤り4/78 で、**余計に置いたセルは0・誤った色も0** (取りこぼし4のみ)。"
        "充填率90%以上の盤面は video_38 で 6→3 枚に半減 (母数 808/815)。"
        "video_39 では 3→3 で変化なし (直す対象が無い動画では何もしない)。"
        "【ONが新たに作る誤り (明記)】video_38 で2箇所・計4セル。"
        "(a) f25122 で連鎖により消えた紫3セルを約2秒間そのまま保持する (f25150 で自己修正)。"
        "(b) f29466 で行11列3 を緑でなく黄と判定する (確定が10frame遅れ、"
        "着地ペアの参照が次のツモへずれる既知の弱点 `enable_landing_color_fix` 未採用に起因)。"
        "いずれも1.7秒以内に自己修正し、OFF が作る68セル誤りとは規模が2桁違う。"
        "【採用で表に出る別の欠陥】長連鎖が CHAIN_MAX_HOLD_SEC=5.0 の安全弁で途中確定する "
        "(14連鎖=実アニメ約18秒に対し期限5秒)。OFF ではおじゃまの横取りが偶然"
        "『保持の延長』として働き、この欠陥を隠していた。B/C とは独立の欠陥として台帳へ登録済み。"
        "なお CHAIN 保持時間の較正値 (2.61/1.17) を併用しても誤りは 9→9 で変わらず、"
        "この欠陥は保持時間では直らないことを実測で確認した。"
        "決定性は独立2プロセスで808/808一致を確認済み。"
        "根拠 D:/puyo_analyzer/verify/g3_repair_2026-09-15_v1/{bugb_effect,bugb_v39,combo}/、"
        "検収画像 logs/diag_bugb_onoff_2026-09-17/。"
        "既定 OFF のままでは既存挙動と bit-identical。",
    ),
    AdoptedFlag(
        "--enable-pseudo-chain-score-fill", "2026-09-17",
        "W7根治①。連鎖検知3経路のうち formula/landing の2経路が、simulate 済みの "
        "検証結果を持ちながら total_score=0 をハードコードしていた。"
        "0 は『連鎖が無い』ではなく『まだ計算していない』の意味なのに、"
        "下流のゲート (CHAIN_TOTAL_MIN_SCORE=40) からは区別できず、"
        "両者発火の場面で先読み評価が起動しなかった。"
        "「score=0 の cc=8 を幻連鎖と誤認する」回帰テストまで書かれていた "
        "(score 未計算と連鎖不在の混同)。"
        "実装は 2026-08-13 に完了・テスト済みだったが本番採用が漏れていた。"
        "実装 src/recognition_pipeline.py `_fill_pseudo_chain_score`: "
        "verified が None または chain_count<=0 なら従来どおり (0,0,False) を返し、"
        "それ以外でのみ公式の calculate_chain_score(検証済み結果) を使う。"
        "例外時も従来値へ落ちる fail-safe。充填したことは ChainEvent.score_estimated=True で明示する。"
        "消去ステップは4連結以上でのみ生成され最小得点 4×10×1=40 = CHAIN_TOTAL_MIN_SCORE と"
        "厳密に一致するため、充填後の値は必ず既存ゲートを満たす "
        "(tests/test_scoring.py の不変条件テストで固定)。"
        "既存テスト tests/test_chain_formula_detection.py の該当 8 件 PASS。"
        "【実測 2026-09-17】video_38 先頭600秒・7試合・他フラグ同一の A/B。"
        "盤面 808→808、共通808位置の内容差分 0、消えた0/増えた0 = "
        "**認識の出力を1セルも変えない**。決定性は独立2プロセスで808/808一致を確認済み。"
        "【測っていないこと (明記)】有利不利側の効果 (先読み評価の起動率) は"
        "2026-08-13 の実ログ検証 1 件 (cc=8 が4.2秒 score=0 のまま → 即時 score=30300) のみで、"
        "母集団での再測定は未実施。採用根拠は『認識に副作用が無い』ことと"
        "『既知の誤った値 0 を、計算可能な正しい値へ置き換える』ことの2点。"
        "根拠 D:/puyo_analyzer/verify/g3_repair_2026-09-15_v1/w7_effect/。"
        "既定 OFF のままでは既存挙動と bit-identical。",
    ),
    AdoptedFlag(
        "--enable-gravity-settle-reset-on-exit", "2026-09-17",
        "バグC根治。GRAVITY_SETTLE 追跡中に他 detector へ横取りされると "
        "_settle_start_frame/_settle_start_time が残留し、次回再進入時に "
        "古い開始時刻で elapsed を計算して 1 frame で誤って STABLE 化する。"
        "実データ初例: video_38 f11292 1P (188.20秒)。残留 "
        "_settle_start_time=178.367 で elapsed=9.833秒 >= MAX_SEC(1.5) が "
        "即成立し、連鎖 4 段目の途中で STABLE へ落ちて連鎖エフェクトの光が "
        "confirmed へ焼き付いた (user 実画面判定: 上2段はすべて背景なのに "
        "おじゃま2/青6を記録)。2 frame 後には CHAIN へ戻っており連鎖は継続していた。"
        "修正実装は 2026-08-08 に完了・テスト済みだったが採用登録が漏れていた。"
        "根拠 current_scope.jsonl:64212 (counters_before)、"
        "tests/test_gravity_settle_reset_on_exit_bugc.py 4 PASS。"
        "既定 OFF のままでは既存挙動と bit-identical。"
        "【実測 2026-09-17】video_38 先頭240秒・同一条件 A/B (他フラグ同一)。"
        "盤面 183→182。共通180位置の内容差分は 0 で、既存の正常記録は一切変えない。"
        "消えたのは OFF のみの 3 枚 = f11292(188.20秒) / f11794(196.57秒) / f11870(197.83秒)、"
        "増えたのは ON のみの 2 枚 = f11776(196.27秒) / f11876(197.93秒)。"
        "**消えた先頭 f11292 は user が実画面で『連鎖中でエフェクトが邪魔をしている』"
        "『上2段はすべて背景』と判定した誤読フレームそのもの**。誤記録が消え、"
        "正しい位置の記録へ置き換わった。"
        "決定性は同一条件2回で確認済み (182/182、共通182位置の差分0、片側のみ0)。"
        "したがって上記の差はフラグの効果であり実行ゆらぎではない。"
        "根拠 D:/puyo_analyzer/verify/g3_repair_2026-09-15_v1/p0a_effect/ および determinism/。"
        "既定 OFF のままでは既存挙動と bit-identical。",
    ),
    AdoptedFlag(
        "--enable-effect-gate", "2026-08-03",
        "全消し/カットイン演出中の誤読を抑止",
    ),
    AdoptedFlag(
        "--enable-burst-guard-v2", "2026-08-05",
        "バーストガード再設計 Stage1。誤り 93 セル -> 33 セル (user条件付き承諾)",
    ),
    AdoptedFlag(
        "--enable-transition-merge-guard", "2026-08-05",
        "Stage1.5。NON-STABLE->STABLE 遷移 merge の物理的期待値フィルタ",
    ),
    AdoptedFlag(
        "--burst-gate-open-threshold 0.954", "2026-08-05",
        "緊急較正。factorial バックテストで決定した閾値",
    ),
    AdoptedFlag(
        "--enable-hidden-row-burst-guard", "2026-08-05",
        "Stage1.5b (§11)",
    ),
    AdoptedFlag(
        "--enable-match-transition-debounce", "2026-08-06",
        "長時間劣化修正 A'。Phase I 合格構成に含まれる",
    ),
    AdoptedFlag(
        "--enable-ojama-fall-placement-override", "2026-08-15",
        "案2修正版 (2026-08-13導入/2026-08-15 evidence一発判定のchain-active"
        "除外修正、コミット9dc5e35、user承認)。OJAMA_FALL滞在中の場面1振動 "
        "(0.15-0.3秒周期でOJAMA_FALL<->STABLE往復、docs/DEMO_REVIEW_2026-08-13.md) "
        "を根治する。物差しv2 (55盤面、既存人手ラベル): 94.08%→95.63%、新規劣化"
        "46→25セル (5盤面。内訳: 48%=W9系一過性ドロップアウトへの偶発的巻き込み"
        "でplacement_override自体のロジック誤りではない、40%=slide_motion経路"
        "がヒステリシス対象外のまま残る既知の残存リスク、12%=既存の色誤認ノイズ"
        "で無関係、data/verify/yardstick_v2_2026-08-14/diag_c1p/review_sheet.html "
        "参照)。scene1再現チェック (logs/diag_scene1_oscillation_recheck_2026-08-15.json)"
        "で往復振動0件を個別確認済み (旧15回/秒規模から根絶)。"
        "拡張代表サンプル全域バックテスト (2026-08-15 user承認、フルサイズでなく"
        "序盤/中盤/終盤 各2分×16動画中15本有効 [video_c15/c19はファイル破損で"
        "全チャンク0件、無関係な既知データ品質問題]、2構成、"
        "data/verify/backtest_placement_override_2026-08-15/summary.md): "
        "OJAMA_FALL滞在時間 中央値0.300→0.133秒・p95 1.5→0.933秒 (短縮、狙い通り "
        "= 過去の「全盤面ぷよ数静止待ち」による張り付き時間の縮小)。盤面churn "
        "(隣接STABLE間セル変化量、中央値2.0で同一)・幻連鎖疑い件数 (266→265) は "
        "悪化なし。品質ゲートPASS/FAIL比率はほぼ同水準 (42チャンク中PASS17→20/"
        "FAIL14→15)。**残課題 (悪化側、要監視)**: (a) 往復振動0.35秒未満のカウント"
        "自体は増加 (224+241→432+564、突入数あたり率0.186→0.318) — 直接調査の"
        "結果、大連鎖直後の複数波おじゃま降下 (STABLE挟みつつ短時間で再度降下する"
        "正規の物理パターン) を巻き込んでいるためと判明 (実フレーム系列確認済み、"
        "場面1の単発往復振動パターンとは別種)。(b) 重力違反セル数が39→69に増加"
        "したが、うち30件差分の17件はc22_chunk1の1スナップショットに集中した"
        "外れ値 (単一frame内18セル同時、認識自体の一過性乱れの可能性が高い) で、"
        "除けば39→51 (+31%) 相当。早期STABLE復帰が物理未確定フレームを捉える"
        "リスクとして小さく残る。両残課題とも次回148動画再収集 (本フラグ込み) "
        "でのより大規模な再測定を推奨",
    ),
    AdoptedFlag(
        "--enable-patch-fp-hsv-guard", "2026-08-17",
        "W13 根治 (user承認2026-08-17)。**tier1 patch-NCC の EMPTY 判定に HSV 色域 AND "
        "ガードを追加**する (`src/image_reader.py:1044-1115` `_is_empty_tier1`)。"
        "patch-NCC が閾値 (PATCH_NCC_EMPTY_THRESHOLD=0.92) 以上で背景と一致しても、"
        "現フレームのパッチを単独 HSV 分類器にかけて EMPTY/UNKNOWN 以外を返すなら EMPTY 化を却下する。"
        "**新規閾値のシーン逆算はしていない** — 2026-05 cycle17-19 で median 距離ベースの "
        "`is_empty_by_fp` に作られた AND ガードを、その後移行した patch-NCC 経路へ**移植したもの** "
        "(移植漏れがW13の一因だった)。"
        "【W13の症状】試合開始2.0秒の背景指紋強制採取 (鶏卵問題対策で puyo 数上限を実質無制限に緩和) が "
        "既に設置済みの実在ぷよを『背景』として焼き込み、以降そのセルが patch-NCC 一致で無条件 EMPTY 化 "
        "→ **列が8〜10秒まるごと消える** (userレビュー指摘16、本番採用構成でも再現)。"
        "【測定 (物差しv2、55盤面、user全数レビュー済みラベル)】W16教訓に従い測定構成を明記: "
        "ベース=c1p (現行本番採用構成)。OFF/ONで STABLE snapshot の間引き周期が変わり分母がズレるため "
        "2段階で公平化した。**stage1 共通突合 (n=48/55、3453セル): 95.66% → 96.84%**。"
        "**stage2 同一フレーム限定 (n=22/55、1584セル): 98.61% → 98.74%**。"
        "方向別内訳で**新規悪化 (元々正解だったセルを壊す) は0件**、W9/W10/color_to_ojama 系の "
        "既知誤り軸には一切干渉なし。"
        "【案1を撤回した根拠】同時に検討した案1 `--enable-highlight-override` "
        "(白ハイライトblob検出でtier1のEMPTY判定を却下) は stage2 で **98.17% とベースより悪化** し、"
        "13セル/2盤面の新規退行を出した。根因は W19 に登録: (a) 救済がセル単位で白blobの写り方が "
        "フレームごとにばらつくため1セル救済に失敗すると穴が開き、`clear_floating_above_gap` が "
        "その上を全部 EMPTY 化して**列ごと道連れ**にする (b) **highlight_override が他セルを救済して "
        "puyo 数差を埋めてしまい、自己修復 `_apply_baseline_broken_counter` の発火条件に入らず "
        "汚染 bg_fp が生き残った** = 修正Aが修正Bの症状を隠す新種の副作用連鎖。"
        "本フラグは**案1の悪化13セルを全13セルとも解消**する。"
        "また案1+2 併用構成は STABLE 盤面 npz 全2023スナップショットで**本フラグ単体とbit-identical** "
        "(案1は本フラグが既に救済済みのセルにしか作用せず追加効果ゼロ) のため、併用は採用しない。"
        "【残リスク】stage2 の n=22 は小標本で1セル変化が±0.06pt動く規模。方向は明確だが "
        "**148動画での広域バックテストは別途必須**。全体テスト 5,043 passed / 0 failed、"
        "既定OFF時は bit-identical を静的テストで担保",
    ),
    AdoptedFlag(
        "--enable-floating-gap-restore", "2026-08-17",
        "R2浮きぷよ是正 (user承認2026-08-17、5フラグ一括採用・構成F)。TSUMO_FALL/"
        "OJAMA_FALL→STABLE 遷移で「下が空・上に puyo」の物理矛盾を検出したら、"
        "上を消すのでなく遷移前 confirmed_board から色を復元する "
        "(docs/KNOWN_WEAKNESSES.md R2)。**単独では持続誤認±0** (構成A→B: 106→106) "
        "だが物理矛盾是正の保険としてF構成 (5フラグ一括) に同梱採用。"
        "根拠データ: data/verify/diag_r2_floating_gap_restore_2026-08-17/。"
        "測定は必ずF構成 (5フラグ揃い) で行っており本フラグ単体の効果を測った"
        "数値ではない",
    ),
    AdoptedFlag(
        "--enable-landing-color-guard", "2026-08-17",
        "W10観測補正継続ガード (user承認2026-08-17、5フラグ一括採用・構成F)。"
        "着地セル色の継続監視ガードで、着地直後の一時的な色観測ブレを"
        "追跡し続けて誤確定を防ぐ。構成B→C (+本フラグ) で持続誤認106→105"
        "(-1)。根拠: docs/KNOWN_WEAKNESSES.md W10節、コミット85e39b3で観測補正"
        "継続ガードの実測を記録、d1c60f9でCLI配線漏れ (collect_boards_lean.py) "
        "を是正済み。測定はF構成 (5フラグ揃い) での積み上げ値",
    ),
    AdoptedFlag(
        "--enable-override-color-guard", "2026-08-17",
        "cycle71n長期投票overrideの安全網 (user承認2026-08-17、5フラグ一括採用・"
        "構成F)。真因は W23 (_validate_next_history のever_seen飢餓状態) と"
        "判明したため、本フラグ自体の位置づけは根治でなく安全網 "
        "(persistent_misread_e.json 系統1ガード)。コミットa43d3fbで実装。"
        "測定はF構成 (5フラグ揃い) での積み上げ値であり、単独の増減は"
        "分離測定していない",
    ),
    AdoptedFlag(
        "--enable-ojama-column-stack-fix", "2026-08-17",
        "持続誤認26件の系統2根治 (user承認2026-08-17、5フラグ一括採用・構成F)。"
        "おじゃま配分ロジックが同一列に対して二重に書き込み衝突する不具合を"
        "根治する (c109 実例で確認)。構成D→E (+本フラグ+override-color-guard) "
        "で持続誤認105→104 (-1)。コミットa43d3fbで実装。"
        "測定はF構成 (5フラグ揃い) での積み上げ値",
    ),
    AdoptedFlag(
        "--enable-next-history-starvation-fix", "2026-08-17",
        "W23根治 (user承認2026-08-17、5フラグ一括採用・構成F)。"
        "_validate_next_history の ever_seen 飢餓状態 (NEXT履歴が長期間"
        "更新されず検証ロジックが機能不全になる不具合) を根治する。"
        "構成E→F (+本フラグ) で **stage2 同一フレーム限定セル正解率 "
        "98.80%→99.43%、持続誤認104→70 (-34)**。W23直接検証25件は100%解消"
        "(persistent_misread_e.json 記載の25件が全て解消を個別確認)。"
        "新規出現3件は別機構の露出であり本フラグの副作用ではない "
        "(docs/KNOWN_WEAKNESSES.md 615-648行に詳細記録)。コミット19dc93aで実装。"
        "【5フラグ一括の位置づけ】本エントリを含む上記4フラグ (floating-gap-"
        "restore/landing-color-guard/override-color-guard/ojama-column-stack-fix) "
        "とセットで構成F として測定・採用 (2026-08-17統一測定、data/verify/"
        "recognition_unified_2026-08-17/persistent_misread_{a..f}.json)。"
        "構成A (現本番、持続誤認106) を基準に B→C→D→E→F の順で積み上げ、"
        "最終構成Fで持続誤認70・stage2セル正解率99.43%・W10_red_purple誤読"
        "10→0完全解消・盤面完全一致33→35/51を実測。フルpytest 5,113 passed / "
        "0 failed、既定OFF時は bit-identical (静的テストで担保)",
    ),
    AdoptedFlag(
        "--enable-ojama-cnn-override-warmup", "2026-08-18",
        "W25第1〜2弾 (user承認2026-08-18、docs/KNOWN_WEAKNESSES.md W25節)。"
        "当初はcycle71n override専用のOJAMA_FALL→STABLE warmupとして実装したが"
        "効果ゼロと実測 (対象9セル解消0/9、コミット2fc990d)。真因再追跡でdrift"
        "再同期の暴発 (雲によるCNN 4↔9往復→DriftDetector needs_resync発火→"
        "confirmed_board全None化→バイパス) と判明し、本フラグをOJAMA_FALL entry"
        "起動+drift-resync抑制サイトに転用 (コミット8fe0759)。**実測**: 28チャンク"
        "でresync発火14→7、resetの発生3→0を直接実証、物差しv2はbit-identical"
        "(無害)。ただし9セル自体は本フラグだけでは未解消 (第3の独立経路が別途"
        "存在、enable-ojama-write-accounting-guardが根治)。**役割再定義**: "
        "根治実装後は会計整合フィルタのフェイルセーフ (W2破綻動画等の会計崩壊時"
        "の保険) として位置づけ直した (アーキ決定、docs/KNOWN_WEAKNESSES.md "
        "W25節 750-756行)",
    ),
    AdoptedFlag(
        "--enable-ojama-write-accounting-guard", "2026-08-18",
        "W25第3弾 (根治) + 固着対策 (user承認2026-08-18、docs/KNOWN_WEAKNESSES.md "
        "W25節)。CNN観測→状態機械入力直前の一元会計整合フィルタ: 非空色→9への"
        "直接遷移を、その列に会計上の未着弾クレジットが無い限り無条件拒否する"
        "(物理制約「おじゃまは空セルにのみ着弾」)。**実測 (コミット4290fc5)**: "
        "対象9セル9/9解消。stage1 (共通突合) 97.98%→**98.41% (+0.44pt)**、"
        "**stage2 (同一フレーム限定) 99.46%→99.27% (-0.18pt、n=23小標本)**は"
        "悪化方向だが正直に記載する。反映遅延の新規退行なし。新規悪化2件"
        "(c10_2P/c109_2P) は精査の結果W1型永久固着 (消去+着弾を両方見逃すと"
        "古い色に固着) の現実化と確定し、持続観測タイムアウト解除 "
        "`OJAMA_REJECT_TIMEOUT_SEC=1.5` を追加実装 (コミット9565e9b) して固着"
        "2件を解消 (+0.07〜0.27秒で自己修正、上限1.5秒の1/5以下)、雲9セル"
        "9/9は維持。**新規許容の明文化 (アーキ承認)**: おじゃま反映が最大1.5秒"
        "遅れうる。8フレーム基準 (feedback_placement_reflection_8frames) は"
        "ツモ設置対象の受け入れ基準であり、おじゃま着弾には元々適用対象外"
        "(棄却側論拠=雲は0.85〜1.0秒で晴れる実測 / 受理側論拠=陳腐化した"
        "持続観測メモリは新しい実観測に屈服すべきという構造的原則)。副産物: "
        "OjamaAccountingTracker の PENDING_ABS_CAP=216到達バグ (score OCR異常"
        "由来) を独立の既存問題として発見、要対処リスト入り。フルpytest "
        "5,255 passed / 0 failed、既定OFF時 bit-identical",
    ),
    AdoptedFlag(
        "--enable-match-end-persist-override", "2026-08-18",
        "境界RT系 (b-1)。user承認2026-08-18、docs/BOUNDARY_MULTISIGNAL_DESIGN_"
        "2026-08-17.md §3(b-1)。`match_end_locked` が MATCH_END_PERSIST_"
        "OVERRIDE_SEC以上連続Trueなら、chain_in_progress による抑制を上書きして "
        "effective_hard_off を有効化する持続タイマー。Step0診断"
        "(data/verify/diag_match_end_miss_2026-08-17/) で判明した真因「本物の"
        "決着パネルは勝者の連鎖アニメ中に表示され始め3秒超持続するが、既存の"
        "chain_in_progress ガード (2026-07-23導入) が瞬間誤爆対策のまま2.55秒間"
        "誤って打ち消していた」に対処する。持続時間で瞬間誤爆 (単発) と本物の"
        "決着 (3秒超) を弁別。030実写検証 (c21) 済み、既存回帰テスト "
        "(test_gravity_settle_in_progress_suppresses_match_end_locked_false_"
        "positive) は維持したまま新規検証を追加",
    ),
    AdoptedFlag(
        "--enable-post-match-lockdown-latch", "2026-08-18",
        "境界RT系 (b-2)。user承認2026-08-18、docs/BOUNDARY_MULTISIGNAL_DESIGN_"
        "2026-08-17.md §3(b-2)。ばたんきゅー/やった!検出をトリガーに「次の"
        "本物の試合開始が確認されるまで試合外とみなす」ラッチ。結果パネル・"
        "対戦カード紹介・次ラウンド待機画面を一括カバーし、ロックダウン5秒切れ"
        "後の再活性化 (対戦カード紹介中に is_match_active へ復帰する新規盲点、"
        "030_c21_2P_f57548実写確認) を防ぐ。`hard_match_off = score_zero_both "
        "or match_end_locked or self._post_match_lockdown_active` へ合流、"
        "score_actively_moving/chain_in_progress の保護は無変更。RT本体実装可"
        "(「盤面が無いと確定している区間の延長」のため指摘13リスクなし)。"
        "実測: data/verify/boundary_impl_verify_2026-08-18/final_verify_"
        "summary.json で rt_blocked_count=4/rt_total=5 (試合外RT遮断4/5)",
    ),
    AdoptedFlag(
        "--enable-lockdown-score-numeric-release", "2026-08-20",
        "境界RT系 (b-2 の解除信号、コミット b646d9f)。上記ラッチは「次の本物の"
        "試合開始が確認されるまで試合外」とみなすが、その**解除条件**が視覚"
        "シグナル頼みで、解除が遅れると試合中がまるごと試合外に落ちて収集行が"
        "消える。本フラグは数値スコア両側0を解除信号として使う (score が両側 0 "
        "に戻る = 新試合の初期化が済んだ、という物理的裏取り)。"
        "**2026-08-20 配線事故の是正**: 本フラグは b646d9f (2026-08-19) で"
        "実装されたが既定OFFのまま本採用登録が漏れ、`collect_flags()` に"
        "含まれなかったため 2026-08-19 の50本収集で**一度も発火していなかった**"
        "(検証スクリプト `_recollect_lockfix_2026-08-19.py` だけが手書きで"
        "渡していたため、c109 単体では改善が見えるのに全体では無変化という"
        "食い違いが生じた)。memory feedback_wiring_check_needs_nongeneric_"
        "scripts_2026-08-18 の再発3回目。"
        "実測 (同一動画10本の対等比較、`scripts/_diag_paired_boundary_fix_"
        "effect_2026-08-20.py`、下記3フラグまとめての効果): 勝敗ラベル欠損 "
        "16.0%→3.9% / ラッチON比率 14.8%→1.7% / 収集行数 47,861→52,405 "
        "(+9.5%) / 試合数 691→530 (偽境界の消滅、c132 は 114→45・欠損 "
        "43.9%→0.0%)。母集団を固定した比較である点が重要 (当初の"
        "「38.3%→45.2% に悪化」は before 50本 vs after 20本という異母集団"
        "比較による錯覚だった)",
    ),
    AdoptedFlag(
        "--enable-lockdown-score-moving-release", "2026-08-20",
        "境界RT系 (b-2 の解除信号その2、コミット b646d9f)。score_actively_"
        "moving (スコアが実際に動いている = 試合が進行している) を解除信号に"
        "使う。数値スコア両側0 (上記) が初期化の瞬間しか捉えられないのに対し、"
        "こちらは「もう試合が始まって進んでいる」という事後の裏取りとして働く"
        "二段構え。採用根拠・実測は上記 --enable-lockdown-score-numeric-"
        "release と共通 (3フラグまとめての A/B、単独分離はしていない)",
    ),
    AdoptedFlag(
        "--enable-result-screen-hardening", "2026-08-18",
        "境界RT系 (③)。user承認2026-08-18。score_actively_moving の装飾演出"
        "(ラウンド告知・対戦カード紹介等の非試合画面でスコア風の数字表示が動いて"
        "見える) への誤認を裏取りする列フィルタ強化。実測: data/verify/"
        "boundary_impl_verify_2026-08-18/final_verify_summary.json で "
        "column_filter_excluded_count=5/column_filter_total=5 (試合外の"
        "誤混入5/5を列フィルタで除外)。b-1/b-2 と同一の検証セット "
        "(c18/c20/c21 実写5アンカー) で確認済み",
    ),
    AdoptedFlag(
        "--enable-ojama-fall-color-swap-guard", "2026-08-18",
        "W26 (docs/KNOWN_WEAKNESSES.md W26節、user承認2026-08-18、コミット"
        "ca03275で実装済み・既定OFF)。連鎖発火の閃光 (白〜青) による色→別色"
        "誤読 (例: 青→緑、赤→黄) をOJAMA_FALL中に限定して拒否する。"
        "`src/ojama_write_accounting.py` の `filter_ojama_write_by_accounting` "
        "に `reject_color_swap` (既定 False) を追加し、`prev_stable_color` と "
        "`new_cnn_value` が共に色ぷよ (1〜5) かつ不一致の場合を追加棄却する。"
        "`enable_ojama_write_accounting_guard` (W25) とは独立のフラグ (単独"
        "稼働をテストで確認済み)。**実測 (scripts/_verify_color_swap_guard_"
        "2026-08-18.py、対戦区間t≥120sのみ)**: user報告の具体箇所 "
        "(video36 t=141-148s、2P、青→緑/赤→黄の実例) 22件→0件 (完全解消)、"
        "実画面フレームと突合済み。広域窓: video36(120-160s) 2P 406→114件 "
        "(-72%)、video52(120-160s) 2P 2→0件 (-100%)、c100(570-660s) 1P "
        "134→36件 (-73%)/2P 70→7件 (-90%)。固着チェック: 全ケースで"
        "flagged_stuck_count=0 (OJAMA_REJECT_TIMEOUT_SEC=1.5秒タイムアウトは"
        "未発動、観測された違反の最大持続は1.0秒未満)。フルpytest "
        "5,388 passed / 0 failed。**既知の限界 (3点、必ず認識のこと)**: "
        "①完全解消ではない (72〜90%削減)。本フィルタは cnn_board が state "
        "machine に入る**前**に適用されるため「現フレームが OJAMA_FALL か」を"
        "前フレームの確定 state でしか判定できず、OJAMA_FALL 突入の最初の"
        "1フレーム (約33ms@30fps) は保護対象外 (W25本体と共通のアーキ制約)。"
        "根治には pipeline のフレーム遅延アーキ変更が必要でスコープ外。"
        "②残存 violation の全量説明には至っていない (上記①は残差の一部説明"
        "に過ぎず、追加調査の余地あり)。③CHAIN中は対象外 (同種の違反が大量に"
        "出ている: video36 で 1P 104ep / 2P 131ep / 40秒だが、多段消去と重力"
        "補充の高速遷移のエイリアシングが未精査のため意図的にスコープ外。"
        "CHAIN中は一切発火しない設計をテストで確認済み)",
    ),
    AdoptedFlag(
        "--enable-native-hsv-classifier", "2026-08-20",
        "**2026-08-21 是正 (配線漏れ6件目): COLLECT_ONLY_ADOPTED から本バケット"
        "へ移設**。当初 COLLECT_ONLY_ADOPTED に登録されていたが根拠不記載の"
        "誤登録だった — 同バケットは「収集にのみ必要」(RecognitionPipeline."
        "load_default の既定 True に対し collect_boards_lean だけ既定 False の"
        "ケース) または「RecognitionPipeline を経由せず collect_boards_lean.py "
        "内実装限定」のいずれかの具体的理由を持つフラグの置き場だが、本フラグは"
        "`RecognitionPipeline.__init__`/`load_default` の通常 kwarg "
        "(`enable_native_hsv_classifier`, 既定 False) として実装されており"
        "どちらの条件にも当たらない。結果として `recognition_load_default_"
        "kwargs()` (RECOGNITION_ADOPTED のみを変換) に含まれず、"
        "scripts/visualize_advantage_overlay.py の本番レンダ経路へ一切転送"
        "されずレンダでは常に無効だった (実測: `pipe._native_hsv_active="
        "False`)。本移設で解消 (visualize_advantage_overlay.py は "
        "`recognition_load_default_kwargs()` の ** 展開が既存経路のため追加"
        "配線不要。visualize_recognition.py には新規 `--enable-native-hsv-"
        "classifier` を追加配線、scripts/measure_stable_cell_acc.py には "
        "`_make_pipeline_cnn` まで全チェーン配線した。collect_boards_lean.py "
        "は元々 RECOGNITION_ADOPTED + COLLECT_ONLY_ADOPTED の合算 "
        "(`collect_flags()`) を渡すため移設による集合差はなく収集側の挙動は"
        "不変)。"
        "HSV セル分類を Rust ネイティブ実装で行う (native/puyo_core/src/"
        "hsv_classify.rs、PyO3)。**認識結果は bit-identical** で、目的は"
        "純粋な高速化。"
        "**採用根拠1: 実測 (本番経路)** video_c34 600秒から1,990frame で "
        "1 frame 34.69→29.05ms (**1.19倍**)。update 単体は 31.67→26.32ms、"
        "classify 帯 (120〜199回/frame) が 35.41→27.09ms と明確に低下。"
        "1盤面あたりでは 5.115→0.800ms (**6.4倍**、cvtColor と Python/Rust "
        "境界越えを含む実効値)。48本収集の見込みが 18.5→15.5時間 になる。"
        "**採用根拠2: T1 パリティ** 合成パッチ 4,732枚 × フラグ4構成 "
        "(red_hue_wrap × specular の全組合せ) で不一致0 "
        "(scripts/_verify_native_hsv_parity_2026-08-20.py)。検体は色レンジの"
        "内側/境界/外側、赤の折り返し2峰 (比率0.1〜0.9)、光沢ハイライト混在"
        "(0〜100%)、5x5/4x4/3x3 の極小パッチ、一様ランダム4,000枚を含む。"
        "**T3 陽性対照を先に実行**し、閾値をずらした構成で600件中145件の"
        "不一致を検出できることを確認済み (比較器が壊れていれば「全一致」は"
        "無意味になるため、測定器事故12件の対策)。"
        "**採用根拠3: T2 本番経路の npz 全列一致** video_39 を同一フラグ・"
        "全長 (94,380frame) で ON/OFF 2走行し、npz の**全22キー・5,418行が"
        "bit 一致** (grids/won/score/ojama_*/board_provenance 等すべて、"
        "scripts/_verify_t2_npz_identical_2026-08-20.py)。状態機械・CNN "
        "smoothing 履歴・おじゃま会計といった下流を通しても差が出ないことを"
        "確認した。"
        "**リスク回避の設計**: (a) cv2.cvtColor は Rust に移植せず Python 側"
        "のまま HSV を渡す — OpenCV の整数丸めを再現すると 1LSB のズレが "
        "median を変え閾値ぎわで色判定が反転しうるため構造的に回避。"
        "(b) `classify` と `_classify_single_patch_no_subregion` は赤の拡張"
        "範囲で R-G 差が不足したときの制御が違う (前者は break + 外側 "
        "continue、後者は continue) ため**統合せず別関数として移植**。"
        "(c) `Counter.most_common(1)` の同数タイは出現順で先勝ちになるので"
        "厳密に `>` 比較で再現。(d) 色レンジは Python の dict 挿入順で渡す"
        "(照合は先勝ちなので順序が結果を変える)。(e) median は counting sort "
        "で np.median と同一定義 (奇数=k番目の順序統計量、偶数=中央2値の "
        "f64 平均、`int()` の 0 方向切り捨て)。"
        "**検証範囲の限界**: T2 は 30fps 動画 1 本 (video_39) のみ。60fps 動画"
        "(stride-2) は未検証だが、stride は認識対象フレームの選択のみで "
        "classify への入力は変わらないため一致すると判断した (user 了承"
        "2026-08-20)。念のため 48本収集後に video_38 でも確認する。"
        "**native 不在時**: puyo_core が未ビルドの環境では黙って従来の "
        "Python 経路に落ちる。判別は `RecognitionPipeline._native_hsv_active` "
        "および各 enable メソッドの戻り値で可能 (fail-silent 回避)",
    ),
    # ------------------------------------------------------------------
    # STABLE 凍結デッドロック根治 3 フラグ (2026-08-24、user 承認で採用)
    # ------------------------------------------------------------------
    AdoptedFlag(
        "--enable-chain-formula-read-verify", "2026-08-24",
        "掛け算式を実読できたフレームは凍結盤面の simulate 検証を通さず発火する "
        "(mechanism=formula_read)。"
        "**背景**: 凍結盤面を連鎖検知の入力にも使う循環があり、検証ゲートが本物の "
        "連鎖を却下して STABLE から抜けられなくなっていた "
        "(memory project_stable_freeze_deadlock_2026-08-24)。片側連鎖では"
        "永久に抜けられず、救済は相手の同時連鎖による偶然だけだった。"
        "**採用根拠1: 10 ケース中 10 ケース改善・悪化ゼロ。** c01 は cc=5/score=5,900 "
        "→ cc=15/score=125,600 (21.3倍)、c03 は 172倍。c05/c10 は「試合中スコアが"
        "永久に 0 のまま」という最重度の故障が解消。"
        "**採用根拠2: E2E で CHAIN 突入遅延 4.133 秒が解消** "
        "(t=6701.700 → 6697.567)。有利不利の乱高下も −90/+75/−37 → −30/−30/−22 に収束。"
        "**採用根拠3: 8 動画の認識精度が 8/8 で改善**。試合外区間を除いた acc が "
        "0.999168 → 0.999249、誤りセル 1,483 → 1,313 (−11.5%)。"
        "全動画が合格ライン 99.5% 超 (最低 v51 で 99.770%)。"
        "**この 8/8 は物差しの欠陥 2 件 (W36/W37) を是正した後の数字である。** "
        "是正前は「5 勝 3 敗・v51 が 99.391% で合格ライン割れ」に見えていたが、"
        "v51 の悪化は全量が『試合終了テロップ中で盤面が画面に無い区間』の測定汚染で、"
        "生きた盤面では ON 0.999548 > OFF 0.999443 だった "
        "(docs/KNOWN_WEAKNESSES.md W37、実画面 v51_t061.0s.png で盤面消失を確認)。"
        "**採用根拠4: 偽イベントを増やしていない。** FP + score=0 の空重複で数えて "
        "w1 17→10、w2 32→13、c62 22→0。"
        "**採用根拠5: 既定 OFF で挙動不変** (正規化 sha256 が修正前コードと一致)。"
        "全 pytest 5,744 passed / 0 failed。"
        "**残る限界**: 掛け算式が読めないフレームは旧機構 formula へフォールバックし、"
        "そこでは total_score=0 になる (W7)。実測でこの経路のイベントは全件スコア非支持"
        "だった。ON 構成下の偽イベントの主な発生源はここ。"
        "検証は 1 動画の大連鎖 10 ケース + 8 動画の物差しで、"
        "通常規模 (2〜6 連鎖) の連鎖数・火力は直接検証していない",
    ),
    AdoptedFlag(
        "--enable-formula-chain-count-update", "2026-08-24",
        "進行中の formula 系イベントの chain_count / total_score を掛け算式の段の"
        "観測値で更新し、CHAIN 保持時間と安全弁の上限を延長する。"
        "**背景**: 長連鎖が 1.4 秒周期で断片化し、単発で読むと火力が 10 分の 1 に"
        "なっていた (840 個 → 84 個、memory "
        "project_chain_event_fragmentation_accumulator_2026-08-22)。"
        "**採用根拠**: 上記 --enable-chain-formula-read-verify と同一の測定群。"
        "両者は同時に評価しており、10 ケースの chain_count 是正 "
        "(cc=5→15 等) はこのフラグの寄与。"
        "画面の統計パネルの「最大れんさ数 9」と ON の cc=9 が一致することを"
        "実画面で確認済み (data/verify/formula_v51_diag_2026-08-24/evidence_frames/)。"
        "**単独 A/B**: v51 単体で read-verify 単独 −0.169pt / count-update 単独 "
        "−0.024pt だったが、これは W37 の汚染を含む数字であり、"
        "汚染を除くと両方とも改善側",
    ),
    AdoptedFlag(
        "--enable-formula-step-interlude", "2026-08-24",
        "掛け算式の段の区切りを『幕間』(通常スコアが読めたフレーム) で判定する。"
        "**背景 (Codex 品質精査 Q-01)**: 旧実装は「右辺 (ボーナス倍率) は同一連鎖内で"
        "単調増加する」を前提にしていたが、公式得点式に照らして誤り。"
        "右辺 = max(1, 連鎖ボーナス + 連結ボーナス + 色数ボーナス) で、1→2 段の"
        "連鎖ボーナス増分は +8 しかないのに連結+色数の変動幅は最大 22 程度あり、"
        "**右辺は同一連鎖内で減少も同値もし得る**。"
        "さらに旧閾値 FORMULA_NEW_SESSION_MIN_GAP_SEC=0.5秒 が実測の幕間分布 "
        "0.433〜0.634 秒のど真ん中に刺さっており、右辺が一度でも減ると "
        "**幕間 0.1 秒の揺らぎだけで真逆に壊れ、火力が 3.4 倍ずれていた** "
        "(幕間 13f→2 段目を棄却して火力 1,100 / 幕間 15f以上→セッション破棄で 320。"
        "正しくは 2 段・1,420)。"
        "**幕間の実測**: 1 段の表示 27〜29 フレーム、段間の幕間 12〜19 フレーム、"
        "幕間中の得点増分は直前段の左×右と一致。"
        "**1 動画の偶然でないことを別動画 3 連鎖で確認** "
        "(v51 1P 9連鎖 / v51 2P / v70 1P、幕間で通常スコアが読めた 29/29 = 100%)。"
        "**採用根拠: 認識精度を悪化させない。** 8 動画で式読取 2 要素のみの構成と"
        "全指標が完全一致 (meta と診断用サンプル配列を除いた差分 0 件)。"
        "**価値は集計精度ではなく、稀に起きる火力 3.4 倍のズレを構造的に防ぐこと。**"
        "実測のフォールバック率は段の確定 130 件中「幕間で受理 95 (73.1%) / "
        "旧規則へ落ちた 35 (26.9%)」で、機構は動いているが結果が変わるのは"
        "右辺が減る連鎖だけなので集計には現れない。"
        "**残る限界 (W34)**: 幕間が観測できない区間 (バースト・煙などで掛け算式も"
        "通常スコアも読めない) は旧規則にフォールバックし、そこでは上記のズレが残る。"
        "**教訓**: src/score_ocr.py の「同一連鎖中は掛け算式が連続表示される "
        "(402 フレーム実測)」というコメントは**測定結果の誤引用**だった。"
        "402 フレームの実測はレイアウト不変性 (y=30±1、倍率 1.0) の測定であり、"
        "時間的連続性は測っていない。コード内コメントの「実測」表記を、"
        "出典に当たらず前提として使わないこと",
    ),
)

# ============================
# 有利不利オーバーレイ (visualize_advantage_overlay)
# ============================
ADVANTAGE_ADOPTED: tuple[AdoptedFlag, ...] = (
    AdoptedFlag(
        "--early-fire-reaction", "2026-07-29",
        "user レビュー指摘1/2 対処。発火フレームで即座に速報を反映する。"
        "無効だと両者 STABLE まで判定が凍結し、大連鎖中に有利不利が逆転する "
        "(2026-08-08 実測: 無効=2P有利24/77% -> 有効=互角52% -> 1P有利60%)",
    ),
    # --platt-calibration は 2026-08-12 user承認で撤回 (採用 2026-08-04)。
    # 根拠: 自信過剰 (80%表示→実勝率64%) の真因は対称化バグで B-1 修正済み
    # (8/11)。修正後は素の出力が ECE 0.0125 で最良、旧モデル向け補正の適用は
    # 逆に歪める (逆効果を実測)。148 本再学習後に新たな自信過剰が出た場合は
    # 新モデルのデータで較正を作り直して再導入する (旧補正の使い回し禁止)。
    AdoptedFlag(
        "--per-side-settled", "2026-08-09",
        "片側でも STABLE なら再計算する。従来の両者同時 STABLE ゲートは実測で "
        "試合時間の 72.3%・最長 13.97 秒 評価を凍結させ、1P が撃ち切って空・"
        "2P が窒息寸前でも「互角54%」のままだった。有効化で t=66 が "
        "1P有利72 (勝率97%) になり主因も窒息余裕差に変わった",
    ),
    AdoptedFlag(
        "--no-score-lead-bias", "2026-08-09",
        "得点タイブレークを外す。user伝授「スコアはおじゃまを送る手段で、"
        "送った時点で意味を失う」。送ったぶんは予告/盤面で既に観測できるため"
        "二重計上になる。t=29 が 2P有利80%% -> 69%% に改善",
    ),
    AdoptedFlag(
        "--no-pressure", "2026-08-09",
        "圧力成分を外す。おじゃまを個数で数える設計だったが、user伝授の通り"
        "評価すべきは盤面能力の低下。外しても判定はほぼ変わらず(38%%->40%%)、"
        "情報が他成分と重複していたことが実測で確認された",
    ),
    AdoptedFlag(
        "--sample-interval 0", "2026-07-13",
        "毎フレーム更新。0.5 秒間引きだとおじゃま会計がスコア変化・連鎖終了を"
        "取りこぼし net/forecast=0 になる (ADVANTAGE_OVERLAY_2026-07-13 §2-3)",
    ),
    AdoptedFlag(
        "--counter-reach", "2026-08-12",
        "打ち合い応手確率 (モンテカルロ、mc_counter_estimator経由) の正式採用 "
        "(指標大整理提案書 0-4)。三つ巴比較 (案D実データ学習 / 急所3修正シミュ / "
        "併用) で全位相 (序盤/中盤/終盤) 有意勝ち (rho=0.808, AUC=0.837、"
        "memory project_exchange_triple_comparison_results_2026-08-02)。"
        "過去に「採用」とコードコメントのみ先行し、本ファイル未登録・"
        "visualize_advantage_overlay.py の CLI 既定値も OFF のままという"
        "食い違いがあった (0-4 の確認依頼で発覚)。COUNTER_REACH_ENABLED_"
        "BY_DEFAULT=True で CLI 既定値も同時に ON 化する",
    ),
    AdoptedFlag(
        "--normalize-fps-30", "2026-08-12",
        "60fps 動画を stride-2 相当 (実効30fps) に間引く "
        "(src.fps_normalize.resolve_normalize_fps_30_stride、"
        "collect_boards_lean.py が 2026-07-30 から既定採用済みの正規化と"
        "同一関数)。従来オーバーレイのみ全フレーム処理のままで、認識状態機械の"
        "フレーム数定数 (STABLE_RECOVERY_MIN_FRAMES 等、30fps前提でコメント"
        "済み) が実時間半分で発火し STABLE 遷移が23%%過多になる非調整領域で"
        "動いていた (収集・学習データと違う認識意味論の不一致)。"
        "OVERLAY_NORMALIZE_FPS_30_ENABLED_BY_DEFAULT=True で CLI 既定値も"
        "同時に ON 化し、収集側と同じ意味論に揃える",
    ),
    AdoptedFlag(
        "--production-recognition", "2026-08-13",
        "本番採用の認識フラグ群 (RECOGNITION_ADOPTED: effect-gate/"
        "burst-guard-v2/transition-merge-guard/burst-gate-open-threshold "
        "0.954/hidden-row-burst-guard/match-transition-debounce) を "
        "load_default() へ自動適用する (recognition_load_default_kwargs() 経由)。"
        "根因調査 (2026-08-13) の副次発見: 従来 overlay はこれらを一切転送して"
        "おらず、デモ/レビュー動画が本番より劣化した認識で生成されていた "
        "(2026-08-08の--early-fire-reaction付け忘れ事故と同型)。"
        "OVERLAY_PRODUCTION_RECOGNITION_ENABLED_BY_DEFAULT=True で CLI 既定値も"
        "同時に ON 化する",
    ),
    AdoptedFlag(
        "--resize-1080p", "2026-08-13",
        "認識入力を1920x1080へ正規化してから RecognitionPipeline.update() に"
        "渡す (collect_boards_lean.py:1050 と同一の正規化)。根因調査 "
        "(2026-08-13) の副次発見: 従来 overlay は表示キャンバス用サイズ "
        "OUT_W/OUT_H(1280x720) へ直接縮小したフレームをそのまま認識にも渡して"
        "おり、BoardRegion の絶対px座標較正 (1920x1080前提) と不整合だった "
        "(CLAUDE.md「他解像度は1920x1080にリサイズしてから認識する」原則違反)。"
        "OVERLAY_RESIZE_1080P_ENABLED_BY_DEFAULT=True で CLI 既定値も同時に "
        "ON 化する (認識用と表示用のフレームは独立に生成、表示解像度は不変)",
    ),
    # ---- 決着先読み (resolved-*) 一族: 親→子の依存構造に注意 ----
    # 親: --resolved-exchange-eval。これが OFF だと resolved-* 系の子フラグは
    # **全て無視される** (visualize_advantage_overlay.py の各 help
    # 「--resolved-exchange-eval 無効時は無視される」)。依存の系図:
    #   --resolved-exchange-eval (#9、根)
    #     ├─ --resolved-decisive-amplify (#10)
    #     ├─ --resolved-live-defender (#13)
    #     │    └─ --resolved-live-defender-strict (#14案1、2026-08-15登録済み)
    #     └─ --resolved-kill-override (#14案2、2026-08-15登録済み)
    # 2026-08-15 に子2つだけが先に登録され、親3つは 2026-08-24 まで未登録という
    # 不整合があった。単一情報源 (advantage_overlay_flags()) から取ったフラグを
    # そのまま渡すと「子だけON・親OFF = 何も効かない」ため、レンダスクリプトが
    # 親を手書きで補う運用になっていた (手書きは配線事故7件目 feedback_use_
    # single_source_for_flags_2026-08-22 と同型の温床)。再発は tests/
    # test_production_config.py::TestResolvedParentChildConsistency が検出する。
    AdoptedFlag(
        "--resolved-exchange-eval", "2026-08-24",
        "指摘9対処 (user指摘2026-08-13、docs/DEMO_REVIEW_2026-08-13.md #9"
        "「お互い連鎖を撃った時点で未来は確定しているのに、連鎖途中で勝率が"
        "乱高下する」)、2026-08-13実装。両側の chain_event が同時にアクティブに"
        "なった瞬間に一度だけ双方の連鎖を完走シミュレーション→得点→おじゃま"
        "換算→相殺→着弾後の仮想盤面をモデル評価し、決着後勝率で固定表示する "
        "(ResolvedExchangeTracker)。観測が予測を上回ったら即時再決着 (シミュ"
        "過小の既知リスク対策)。保持は相殺後おじゃまの着弾完了まで延長 "
        "(指摘11対処、2026-08-14: 着弾前の空白で盤面が「1P綺麗/2P空」に見え"
        "飛翔中の約5段が評価から欠落する誤答の防止)。**実測**: 最終版v3検収 "
        "(2026-08-14、DEMO_REVIEW #9節) で両者発火の5.5秒間 (t=33-38.5) を "
        "71%/29% の分散ゼロで完全ホールド (v2は同場面で39pt動き続けていた)。"
        "以後 2026-08-13〜の全デモ/レビュー動画・30先納品レンダ (scripts/"
        "_render_zenchi_8seg_2026-08-21.sh) で継続使用の事実上の本番構成"
        "だったが登録漏れのまま各スクリプトが手書きで補っていた。"
        "2026-08-24 user承認で正式登録 (採用日=登録日、実装日は上記)",
    ),
    AdoptedFlag(
        "--resolved-decisive-amplify", "2026-08-24",
        "指摘10対処 (user指摘2026-08-14、docs/DEMO_REVIEW_2026-08-13.md #10"
        "「量的に返せないことはほぼ確定している、実態はもっと決定的なはず」)、"
        "2026-08-14実装。#9 の決着値に受け側の応手不能度×実飛来量の増幅を"
        "統合し、応手不能かつ飛来量大なら決定的側へ寄せる。増幅強度 "
        "RESOLVED_AMPLIFY_SCALE=8.0 は COUNTER_SCALE×W_COUNTER (既存重み体系"
        "からの導出、シーン逆算なし、指摘12決着時に較正)。**実測** (指摘12の"
        "較正・修正4込み、コミット2a6fed8/43e5c41): 指摘10場面 71%→96.2% "
        "(修正4後も96.1%を維持=真に応手不能の場面は決定的のまま)、指摘12場面"
        "は過剰増幅97.4%→84.3%へ是正 (userの参考所見「返せる確率は十分あるが"
        "セカンドで不利になる程度」と整合)。--resolved-exchange-eval 無効時は"
        "無視される (子フラグ)。2026-08-24 user承認で正式登録",
    ),
    AdoptedFlag(
        "--resolved-live-defender", "2026-08-24",
        "指摘13対処 (user指摘2026-08-15、docs/DEMO_REVIEW_2026-08-13.md #13"
        "「片側のみ連鎖中なら受け側の状況は0.5秒ごとに変わるので動くのが"
        "普通では」)、2026-08-15実装 (コミット0696dcc+方向反転修正)。両者"
        "連鎖中=完全凍結を維持したまま、片側のみ連鎖中 (攻撃側継続・受け側"
        "自由行動) は攻撃側の帰結 (飛来量・仮想盤面) を凍結維持しつつ、受け側"
        "の現在盤面+残り時間逓減で0.5秒ごとにモデル評価/決定度増幅を再計算"
        "する。**実測タイムライン** (0.5秒刻み): 開始2P 77%→連鎖進行で86〜91%"
        "へ漸増→1Pの構えで81〜74%へ漸減→t=243.8の撃ち返しで1P 58%に反転 "
        "(数字合わせなしで実際の展開と整合)。既知の誤爆 (settle gap にいる側"
        "を自由な受け側と誤分類、指摘14) は子フラグ --resolved-live-defender-"
        "strict (直下、2026-08-15登録済み) が対処する。--resolved-exchange-"
        "eval 無効時は無視される (子フラグ)。2026-08-24 user承認で正式登録",
    ),
    AdoptedFlag(
        "--resolved-live-defender-strict", "2026-08-15",
        "指摘14 案1 (user承認2026-08-15、コミット53129bb)。決着ホールド中の "
        "受け側再評価 (_reevaluate_live_defender、指摘13で導入) の起動条件を "
        "状態機械ベースに厳格化する。従来は「ちょうど片側の chain_event が "
        "None」という XOR のみで受け側を「自由」と判定していたが、ChainEvent は "
        "trigger 検知時に1度発行され hold 秒後に None へ戻るパルス方式のため、"
        "「旧連鎖の hold 切れ〜新連鎖の trigger 検知」の settle gap にいる側も "
        "ev=None になり誤って自由扱いされていた。実測 (t=195.30): "
        "ev1_cc=9 (1P継続中) / ev2=None かつ state2=GRAVITY_SETTLE の 2P を "
        "自由な受け側と誤分類し、着弾前の綺麗な盤面をモデルへ渡して "
        "hold_p1 96.1%→81.1% へ退行 (真の飛来おじゃま589個=回避不能死なのに "
        "2P 18.9% を5.2秒表示)。修正 = defender 側の状態が "
        "_LIVE_DEFENDER_BUSY_STATES {CHAIN, GRAVITY_SETTLE} なら再評価を "
        "スキップし直前値を維持する。TSUMO_FALL/OJAMA_FALL は指摘13が意図した "
        "正当な自由行動を塞がないため意図的に非busy。A/B実測 "
        "(logs/_diag_issue14_flags_ab_v2_2026-08-15.log): 指摘14窓 "
        "(194.53-201秒) は退行が完全消滅し 96.1% を窓全体で維持。指摘13 正当窓 "
        "(234.87-245.5秒) は t=236.27 以降 baseline と完全一致・0.5秒ごとに "
        "連続変化 (凍結の再発なし)、冒頭0.9秒のみ 2P 77%→84.3% に変化 "
        "(defender が実際に busy だった瞬間を正しくスキップした結果。この値は "
        "指摘12で決着した84.3%と一致)",
    ),
    AdoptedFlag(
        "--resolved-kill-override", "2026-08-15",
        "指摘14 案2 (user承認2026-08-15、コミット8f8a577)。既存の致死上書き "
        "安全弁 kill_override を決着ホールド表示値にも配線する。従来は "
        "ライブ per-frame 経路にのみ配線されており、決着ホールド中は "
        "disp_adv/disp_p1 を丸ごと上書きする経路が通常経路を迂回するため "
        "pending/room 比が致死水準でも安全弁が絶対に発火しなかった "
        "(実測 589/50≈11.8 ≫ KILL_RATIO_FULL=1.5 で無発火)。材料は新規に "
        "増やさず既存の観測量のみ再利用 (pending=_incoming_total_p1/p2 = "
        "指摘11の着弾完了判定と同一値、room=board_room(b1)/board_room(b2))。"
        "二重計上防止はライブ経路と同じく kill_override を最終段として適用 "
        "(g=1 の完全上書き時は amplify 由来の寄与も自動的に上書きされる)。"
        "A/B実測 (logs/_diag_issue14_flags_ab_2026-08-15.log): 指摘14窓の "
        "全時刻で adv=+100/p1=99.3% (2P 19%→0.7%)。案1が誤爆機構そのものを "
        "断つのに対し、本フラグは致死場面の最終防波堤として併用する",
    ),
)

# --counter-reach の CLI 既定値。visualize_advantage_overlay.py の argparse
# default / generate() の関数既定値はここを import して使う
# (CHAIN_SIM_ADOPTED の GHOST_CHAIN_RULE_ENABLED と同じパターン。
# 「採用済みなのに初期値OFF」という食い違いの再発防止、2026-08-12)。
COUNTER_REACH_ENABLED_BY_DEFAULT: bool = True

# --normalize-fps-30 の CLI 既定値/generate() 既定値 (2026-08-12 追加、上記
# ADVANTAGE_ADOPTED エントリ参照)。COUNTER_REACH_ENABLED_BY_DEFAULT と同じ
# パターンで単一情報源化する。
OVERLAY_NORMALIZE_FPS_30_ENABLED_BY_DEFAULT: bool = True

# --production-recognition の CLI 既定値/generate() 既定値 (2026-08-13 追加、
# 上記2定数と同じパターン)。True で RECOGNITION_ADOPTED (本番採用の認識
# フラグ群) を recognition_load_default_kwargs() 経由で自動適用する。
OVERLAY_PRODUCTION_RECOGNITION_ENABLED_BY_DEFAULT: bool = True

# --resize-1080p の CLI 既定値/generate() 既定値 (2026-08-13 追加)。True で
# 認識入力を 1920x1080 へ正規化してから RecognitionPipeline.update() に渡す
# (collect_boards_lean.py と同一正規化、CLAUDE.md「他解像度は1920x1080に
# リサイズしてから認識する」原則)。
OVERLAY_RESIZE_1080P_ENABLED_BY_DEFAULT: bool = True

# ============================
# 表示 (visualize_recognition の overlay 系)
# ============================
VISUALIZATION_ADOPTED: tuple[AdoptedFlag, ...] = (
    AdoptedFlag(
        "--chain-formula-simulate-verify", "2026-08-08",
        "無効だと連鎖数が固定 1 になる。有効で実測値 (9連鎖なら 9) が出る",
    ),
    AdoptedFlag(
        "--overlay-chain-hold-until-end", "2026-08-08",
        "user 要望「連鎖中はずっと chain であってほしい」。"
        "連鎖中の異常な離脱 20 回 -> 0 回",
    ),
    AdoptedFlag(
        "--production-recognition", "2026-08-13",
        "横展開監査 (docs/CROSS_CUTTING_AUDIT_2026-08-13.md P1) の配線漏れ是正。"
        "visualize_recognition.py が RECOGNITION_ADOPTED (バーストガード等6"
        "フラグ) を明示指定しない限り一切適用しておらず、レビュー動画が本番"
        "より劣化した認識で生成されていた (visualize_advantage_overlay.py の "
        "eacb1f3 と同型の事故)。recognition_load_default_kwargs() 経由で "
        "load_default() へ自動適用する。--no-production-recognition で無効化",
    ),
    AdoptedFlag(
        "--production-visualization", "2026-08-13",
        "同上の是正。上記2フラグ (chain-formula-simulate-verify/"
        "overlay-chain-hold-until-end) 自体を CLI 既定 ON にする配線 "
        "(resolve_production_config_overrides() 経由)。"
        "--no-production-visualization で無効化",
    ),
)

# measure_stable_cell_acc.py (物差し) にも同型の配線漏れがあった (P1)。
# --no-production-recognition の明示指定で過去測定と bit-identical な旧構成
# (各フラグ明示指定必須) を再現できる (物差しの継続性維持)。専用の
# AdoptedFlag タプルは持たず、RECOGNITION_ADOPTED を単一情報源として
# resolve_production_recognition_flags() (同ファイル内) が消費する
# (recognition_load_default_kwargs() と同じ変換ロジック)。


# ============================
# 連鎖シミュレーション — 物理ルール採用 (CLI フラグでなく Python 既定値の
# 単一情報源。src.chain.ChainSimulator(exclude_hidden_row_from_pop=...) の
# 呼び出し側は本フラグを import して使うこと。個別に True を書き散らさない)
# ============================
CHAIN_SIM_ADOPTED: tuple[AdoptedFlag, ...] = (
    AdoptedFlag(
        "exclude_hidden_row_from_pop=True", "2026-08-10",
        "幽霊連鎖ルール (13段目/隠し段のぷよは4つ繋がっても消えない、"
        "user伝授2026-08-09、実装・単体テストはコミット991fa80で完了済)。"
        "全域バックテスト (boards_lean_phase_l_2026-08-07 148動画) の結果を"
        "docs/verify 配下に記録。詳細は data/verify/ghost_chain_backtest_2026-08-10/",
    ),
)

# 上記フラグの真偽値そのもの。本番構築箇所はこれを import して
# `ChainSimulator(exclude_hidden_row_from_pop=GHOST_CHAIN_RULE_ENABLED)` の形で使う。
GHOST_CHAIN_RULE_ENABLED: bool = True


# ============================
# 有利不利オーバーレイ — 主因表示の除外リスト (2026-08-11、ロードマップ Phase 1-3)
# ============================
# 背景: scripts/visualize_advantage_overlay.py の `_score_advantage()` が
# 組み立てる「主因」欄は、単に **|差分| の大きい順** で上位3件を選んでいるだけで、
# 実際にモデルの予測 (p1/adv) へどれだけ寄与しているか (=真の attribution、
# scripts/_diag_adv_attribution_2026-08-09.py の `_attribution()` が計算する
# ablation ベースの寄与) は一切見ていない。 そのため **勝敗と無相関な指標**でも、
# たまたま差分の絶対値が大きいと主因1位に選ばれてしまう。
#
# 以下は 2026-08-09/11 の層別 AUC 実測 (labeled_win_combined66.csv、66動画・
# 73,416 ペア、全体/おじゃまフラット/おじゃま差大の3層。
# scripts/_verify_efficiency_vs_material_2026-08-09.py +
# scripts/_verify_attribution_exclusion_2026-08-11.py) で「主因として表示する
# 根拠が無い」と確定した指標。
#
# **重要: モデルの特徴量からは外さない** (予測 adv/p1 には一切影響しない、
# 表示だけの修正)。 特徴量そのものの除去は別途「全域無悪化ゲート」を伴う
# 検証が要るためスコープ外 (feedback_overfitting_awareness_2026-08-04)。
ATTRIBUTION_EXCLUDED_INDICATORS: tuple[str, ...] = (
    # AUC 0.5000 (全体/おじゃまフラット/おじゃま差大の全層で寸分違わず一致)。
    # 根本原因は学習データ labeled_win_combined66.csv 上で本列が 100% NaN
    # (未収集) であること (2026-08-11 実測、193,623 行全て NaN)。学習された
    # 重みは実質ゼロだが、推論時 (_fill_expected_fire_candidate) は実盤面から
    # 都度計算するため差分値自体は大きくなり得て、「差分が大きい順」の現行
    # ロジックだと無情報にもかかわらず主因1位に出る (2026-08-09 デモ実測で
    # 「期待火力K1差 +0.64」が主因1位表示された事例)。
    "expected_fire_k1",
    "expected_fire_k2",  # 同上 (AUC 0.5000、全層一致、学習データ上も100% NaN)
    # 過去に「信頼不可」と判定済み (memory
    # project_saturation_ceiling_untrustworthy_2026-07-22: 「理想ツモ天井は
    # 空き空間量と同じで無相関」)。 加えて 2026-08-11 再検証で学習データ上
    # current_max_chain (現在最大連鎖) と 100% 完全一致 (193,623 行、
    # 平均/分散/AUC 全て同値) と判明 — 「現在最大連鎖」と同じ信号を
    # 「飽和連鎖量」という別名で二重表示しているだけで独立情報が無い。
    "saturated_chain_count",
    # 2026-08-22 修正④ (主因表示の健全化)。generate() の主因ランキングが
    # 「|f1-f2| の生値絶対値順」(未正規化) のため、0-1 正規化された重要指標
    # (例: diff_board_ojama_count, diff_current_max_chain) より raw な整数
    # カウント列の方が絶対値が大きくなりがちで、無情報でも主因上位に出る。
    # 本列は permutation_importance_full.csv (data/verify/
    # retrain_model62_2026-08-21) で実測: importance_mean=-0.000376
    # (52列中52位、最下位・負の寄与) と自己検証済み (2026-08-22 コーダ実測、
    # 既存の expected_fire_k1/k2・saturated_chain_count と同一パターン)。
    # 実例: t=6717.5s で「連結対数差差 +8.00」が主因1位に出るが、モデルは
    # 実質この列を使っていない (根拠は上記csv、他候補は正規化するかの
    # 全面対応は別タスク、ここでは最悪の1件のみ是正)。
    "diff_conn_pair_count",
)


# ============================
# 指標大整理 (2026-08-12 user確定、docs/INDICATOR_REORG_PROPOSAL_2026-08-12.md
# 「決定記録」節) — scripts/build_labeled_win_from_npz.py への実装
# ============================
# 実体の定数群 (DIFF_REPLACE_OWN_COLUMNS 等) は同ファイル側に置く
# (npz→CSV変換ツール専用の分類のため)。ここには決定内容の記録のみ残す
# (「採用日+根拠を必須記録」規約、production_config.py が単一情報源)。
INDICATOR_REORG_DECISIONS: tuple[AdoptedFlag, ...] = (
    AdoptedFlag(
        "a-1: *_raw列8種+saturated_chain_count を削除", "2026-08-12",
        "*_raw は score の定数倍で完全重複 (記録・学習の両方から削除)。"
        "saturated_chain_count は current_max_chain と19万場面で完全一致 "
        "(作成時のバグで同じものを2回計算していた、"
        "ATTRIBUTION_EXCLUDED_INDICATORS の根拠と同一)。absorption_capacity "
        "は build_labeled_win_from_npz.py では元々未収集のため対応不要",
    ),
    AdoptedFlag(
        "b-1: center_bulge を center_bulge_color/_ojama に分解", "2026-08-12",
        "合成版は当てやすさ0.509でほぼ無価値だが、分解すると不利の大部分は"
        "おじゃま由来 (影響は色ぷよの12倍) と判明。色ぷよ由来分は小さいが"
        "本物の効果 (おじゃまゼロ33万場面でも検出)。indicators_v2.py の"
        "center_bulge() 本体は backwards compat のため変更なし、"
        "center_bulge_color/_ojama を新規追加",
    ),
    AdoptedFlag(
        "b-2: 相手との差 (diff_) 列への置き換え", "2026-08-12",
        "63動画実測: 11項目中10項目で「自分のみ」より「差」の方が当てやすい "
        "(連結最大サイズは終盤0.508→0.567)。例外2つ (色ぷよ総数・3個連結) は"
        "置き換えず own/diff の使い分けを列ごとに分類 (詳細は"
        "scripts/build_labeled_win_from_npz.py の DIFF_* 定数群)。"
        "色ぷよ総数は user指示8/12によりおじゃま総数とのペア特徴として"
        "own+diff+比率+交互作用の4列で表現 (単純な差では向きが逆転する謎の答え)",
    ),
)


# ============================
# 学習データビルダー — 標準採用オプション (2026-08-18 新設)
# ============================
# scripts/build_labeled_win_from_npz.py が受け付ける CLI フラグのうち、
# 「学習データビルド時に常に付けるべき標準構成」として確定したものをここに
# 記録する (INDICATOR_REORG_DECISIONS と同様、決定記録のみ・単一情報源)。
LEARNING_DATA_BUILD_ADOPTED: tuple[AdoptedFlag, ...] = (
    AdoptedFlag(
        "--exclude-match-end-locked", "2026-08-18",
        "境界実装の仕上げ (user承認2026-08-18)。npz の match_end_locked==1 "
        "または post_match_lockdown_active==1 のフレームを学習データ csv から"
        "除外する (scripts/build_labeled_win_from_npz.py)。決着後の結果パネル・"
        "対戦カード紹介・次ラウンド待機画面が試合中と誤って学習データに混入する"
        "のを防ぐ (W20「勝敗演出の幻盤面」族の学習データ側対策)。実測: "
        "data/verify/boundary_impl_verify_2026-08-18/final_verify_summary.json "
        "で column_filter_excluded_count=5/column_filter_total=5 (試合外混入"
        "5/5を除外)。両列が存在しない旧npz (収集時に本フラグ非対応だったもの) "
        "では no-op (後方互換、tests/test_build_labeled_win_from_npz.py "
        "test_convert_one_npz_exclude_match_end_locked_noop_when_columns_"
        "absent で確認済み)。既定 False のため、次回の学習データビルド実行時に"
        "明示指定が必要 (本エントリはその明示指定を「標準オプション」として"
        "記録するもの)",
    ),
)


@dataclass(frozen=True)
class RemovedIndicator:
    """死亡確定・削除決定済みの指標 1 件 (REORG_REMOVED_INDICATORS 用)。"""

    name: str        # 指標名 (collect_indicators_v2.INDICATOR_COLUMNS 等の表記に合わせる base 列名)
    confirmed: str    # 死亡・削除の確定日 (YYYY-MM-DD)
    reason: str       # 死因 (実測結果 または重複判定の根拠)


# ============================
# 指標大整理 — 削除台帳 (2026-08-13 新設、横展開監査 P2)
# ============================
# 背景: saturated_chain_count の削除決定 (a-1、INDICATOR_REORG_DECISIONS 参照、
# 2026-08-12) が3箇所中1箇所 (build_labeled_win_from_npz.py) にしか反映されて
# おらず、scripts/visualize_advantage_overlay.py の FEATURE_CANDIDATES /
# scripts/model_indicator_win.py の REDUNDANT_COLS には残ったままだった
# (「片方の経路だけ直して他方が古いまま」の型A事故、
# docs/CROSS_CUTTING_AUDIT_2026-08-13.md P2)。加えて過去に死亡確定したのに
# 公式の除外リストへ一度も登録されていなかった3件 (honsen_output/
# taiou_capacity/disturbance_rejection、docs/INDICATOR_PROPOSAL_ROUND2_
# 2026-08-13.md D節) をここで台帳化する。
#
# 使い方: 学習・モデル特徴量の候補リストを組み立てる箇所は
# `reorg_removed_indicator_names()` の集合を除外候補として参照すること
# (FEATURE_CANDIDATES / REDUNDANT_COLS がこのパターンに追従済み)。
REORG_REMOVED_INDICATORS: tuple[RemovedIndicator, ...] = (
    RemovedIndicator(
        "saturated_chain_count", "2026-08-12",
        "current_max_chain と19万場面で完全一致 (作成時のバグで同じものを"
        "2回計算していた)。a-1決定 (INDICATOR_REORG_DECISIONS 参照、"
        "ATTRIBUTION_EXCLUDED_INDICATORS と同根拠)",
    ),
    RemovedIndicator(
        "absorption_capacity", "2026-08-12",
        "board_puyo_total と完全重複。a-1決定 (INDICATOR_REORG_DECISIONS 参照)。"
        "scripts/model_indicator_win.py の REDUNDANT_COLS では既に先行して"
        "除外済みだった",
    ),
    RemovedIndicator(
        "honsen_output", "2026-07-17",
        "催促・条件1 (本線打ち合い収支)。中盤AUC 0.512 = current_max_chain の"
        "生値0.514と同等で無寄与と確定 (memory "
        "project_midgame_indicator_failures_2026-07-17)",
    ),
    RemovedIndicator(
        "taiou_capacity", "2026-07-20",
        "対応力 (相手催促を本線温存で上回る)。単純な受け容量 (当てやすさ0.52)"
        "に負け、blend不採用と確定 (docs/INDICATOR_CANDIDATES_2026-07-20.md "
        "分類4)",
    ),
    RemovedIndicator(
        "disturbance_rejection", "2026-08-13",
        "外乱除去比 (C4、returned÷incoming お邪魔)。docs/INDICATOR_CANDIDATES_"
        "2026-07-20.md では候補提示のみ (「既存会計から実装ゼロ」) のまま構想"
        "倒れで終わっている。同系統の潰し・相殺設計である ojama_disruption "
        "(催促・条件2) が478行全ゼロの完全失敗 (memory "
        "project_midgame_indicator_failures_2026-07-17) で決着済みのため、"
        "docs/INDICATOR_PROPOSAL_ROUND2_2026-08-13.md D節の判定に従い死亡"
        "確定として登録する (独立実測が無い旨は正直に記録)",
    ),
)


def reorg_removed_indicator_names() -> frozenset[str]:
    """REORG_REMOVED_INDICATORS の指標名集合を返す (除外リスト構築用)。"""
    return frozenset(r.name for r in REORG_REMOVED_INDICATORS)


@dataclass(frozen=True)
class PipelineGap:
    """collect_indicators_v2 (旧) → build_labeled_win_from_npz (新) の意図的な
    列欠落 1 件 (KNOWN_PIPELINE_GAPS 用)。"""

    column: str       # collect_indicators_v2.INDICATOR_COLUMNS 側の base 列名 (*_raw等を除く)
    confirmed: str     # 欠落を意図的と確認した日付 (YYYY-MM-DD)
    reason: str        # 欠落の理由 (設計上の構造的制約 または明示的な削除決定)


# ============================
# npz→CSV 変換ツールのレジストリ整合 — 既知の意図的ギャップ許容リスト
# (2026-08-13 新設、横展開監査 P1/P2 台帳監査の提案4本の1)
# ============================
# collect_indicators_v2.INDICATOR_COLUMNS (旧収集) と
# build_labeled_win_from_npz._final_fieldnames("full") (新変換) の差分のうち、
# **設計上意図的で危険でない**ものだけをここに列挙する。
#
# 重要: 2026-08-13 時点で判明している「意図的でない脱落11列」
# (main_linked_pair_count 等、docs/INDICATOR_PROPOSAL_ROUND2_2026-08-13.md
# A-1、user採否待ち) はここに **含めない**。含めてしまうと A-1 の問題が
# テストから見えなくなり「直したことにする」事故を再生産するため
# (tests/test_indicator_pipeline_registry_2026-08-13.py が11列の存在を
# 継続的に検出する)。
KNOWN_PIPELINE_GAPS: tuple[PipelineGap, ...] = (
    PipelineGap(
        "tsumo_count_rate", "2026-08-13",
        "累積手数カウンタ (フレーム間の state) が必要。npz は盤面グリッドの"
        "みを保持する grid-only ツールのため構造的に計算不可 "
        "(build_labeled_win_from_npz.py 冒頭「現状カバー範囲」参照)",
    ),
    PipelineGap(
        "margin_time_rate", "2026-08-13",
        "試合相対経過秒 (state) が必要。tsumo_count_rate と同じ構造的制約",
    ),
    PipelineGap(
        "chain_duration_sec", "2026-08-13",
        "連鎖の実時間長 (フレーム間の state) が必要。同上の構造的制約",
    ),
    # ojama_net_balance/ojama_forecast はタスク#8 (2026-08-13、
    # docs/CROSS_CUTTING_AUDIT_2026-08-13.md P4決着) で再接続済みのため本
    # 許容リストから削除した (収集側 collect_boards_lean.py が
    # OjamaAccountingTracker を実駆動して真値を npz に保存するようになった、
    # build_labeled_win_from_npz.py の OJAMA_TRUTH_COLUMNS 参照。許容リストが
    # 陳腐化したまま残すと再接続の事実がテストから見えなくなるため削除する
    # ルール、test_known_pipeline_gaps_entries_are_actually_absent 参照)。
    PipelineGap(
        "reach_fire_power", "2026-08-13",
        "next_pair/dnext_pair 依存。npz は --with-next 収集時のみ next1_a/b を"
        "保持し、本ツールでは next 依存指標は未実装 (冒頭「現状カバー範囲」"
        "に既存記載)",
    ),
    PipelineGap(
        "near_future_fire_k1", "2026-08-13", "reach_fire_power と同じ next_pair 依存の制約",
    ),
    PipelineGap(
        "near_future_fire_k2", "2026-08-13", "同上",
    ),
    PipelineGap(
        "near_future_fire_k3", "2026-08-13", "同上",
    ),
    PipelineGap(
        "near_future_fire_k4", "2026-08-13", "同上",
    ),
    PipelineGap(
        "near_future_fire_k5", "2026-08-13", "同上",
    ),
    PipelineGap(
        "fire_stability_k2", "2026-08-13", "同上 (near_future_fire_power と同じビーム machinery の副産物)",
    ),
    PipelineGap(
        "fire_stability_k4", "2026-08-13", "同上",
    ),
    PipelineGap(
        "fire_stability_k6", "2026-08-13", "同上",
    ),
    PipelineGap(
        "expected_fire_k1", "2026-08-13", "同上 next_pair 依存 (ランダム色ツモのモンテカルロ平均)",
    ),
    PipelineGap(
        "expected_fire_k2", "2026-08-13", "同上",
    ),
    PipelineGap(
        "expected_fire_k3", "2026-08-13", "同上",
    ),
    PipelineGap(
        "expected_fire_k4", "2026-08-13", "同上",
    ),
    PipelineGap(
        "saturated_chain_count", "2026-08-12",
        "a-1決定 (INDICATOR_REORG_DECISIONS 参照)。REORG_REMOVED_INDICATORS と"
        "同一根拠 (current_max_chain と完全一致のため削除)",
    ),
    PipelineGap(
        "absorption_capacity", "2026-08-12",
        "a-1決定。REORG_REMOVED_INDICATORS と同一根拠 (board_puyo_total と"
        "完全重複)",
    ),
    PipelineGap(
        "center_bulge", "2026-08-12",
        "b-1決定 (INDICATOR_REORG_DECISIONS 参照)。center_bulge_color/_ojama "
        "に分解済みのため合成列は出力しない (機能は分解後の2列として存続、"
        "削除ではない)",
    ),
)


def known_pipeline_gap_columns() -> frozenset[str]:
    """KNOWN_PIPELINE_GAPS の列名集合を返す (整合テスト用)。"""
    return frozenset(g.column for g in KNOWN_PIPELINE_GAPS)



# E32固定検収の実効引数から抽出した撃ち合い評価の本番構成。
DV2D_ADOPTION_REASON = "D-v2d (誤った負け確定の根治) — 表示区間の誤確定 zenchi セット1 9→0・セット2 6→0・既存5記録 0→0、3:00場面 2761.28秒、第14試合0。M_time +0.0058 [−0.0086, +0.0206]・M_agree −0.0130 [−0.0400, +0.0135] で門v2の点推定条件は不成立、正しい計算確定 55件が観測死亡まで遅れる (中央値 3.9/5.6秒)。user 判断 2026-10-02 で例外採用 (誤確定0を優先)。独立再計算一致・差分査読済み。抽出元: wt_evalset 42d82b1、logs/eval_set/falsedeath_fix/v2d/REPORT.md、DECISIONS.md 2026-10-02。"

DV2H2B_ADOPTION_REASON = "D-v2h2b — 表示区間の誤確定 0 (zenchi セット1・セット2・既存5記録)、3:00場面 2762.55秒、第14試合0、M_time D-v2d 比 +0.0011 (user 例外判断 2026-10-03 を事前登録どおり継承、上限 +0.002)、正しい確定の消失 70、開始未観測 57/176。E35″ の誤確定 (mia 試合7型) を発火時の元予測と観測の段ごとの照合で止める。抽出元: wt_evalset f34d3be、logs/eval_set/falsedeath_fix/v2h2b/REPORT.md、DECISIONS.md 2026-10-03。"

A2_ADOPTION_REASON = "A2 (式の段原点分離 + H1′ のMENU開始受理) — 115試合 M_time −0.002115 [−0.006464, +0.001154]、M_agree +0.004635 [−0.002218, +0.016401] で門v2合格、表示区間の誤確定 0 (18記録)、3:00場面 2762.55秒、第14試合0、開始未観測 57→22/176、表示内 H1抑止 311→0、正しい確定の消失1 (セット1第10試合、開いたままのexchangeへの偶然依存が解消された副作用)・追加14。独立再計算一致 (不一致0)・差分査読済み。抽出元: wt_a2 dca8384、docs/agent_coordination/A2_SAFETY_REVIEW_20261003.md、DECISIONS.md 2026-10-03。"
A1B_ADOPTION_REASON = "A1b (完走予測の代表候補を段数最大・同段数なら確率最大にし、段数と得点を同じ候補から取る + 観測済みの段の式累計と現在盤面からの因果補完) — A1b+A2 の旧本番比 115試合 M_time −0.0013 [−0.0121, +0.0084]・M_agree +0.0007 [−0.0152, +0.0204]、A2単独比 M_time +0.0008・M_agree −0.0039 で門v2合格、誤確定0 (18記録)、3:00場面 2762.55秒、第14試合0、実7連鎖以上118件の得点誤差中央値 5,698→1,380点・P90 105,614→93,686点。正しい確定の消失 A2比1件 (set1第18試合2P、4.3秒遅れ、予測得点が当たり画面得点の進みを待つ安全検査の設計どおり)。独立再計算一致 (不一致0)・差分査読済み。A1 (確率最大候補) は M_time +0.0025 で不合格のため不採用。抽出元: wt_a1 ace3a7f、docs/agent_coordination/A1B_COMPLETION_SCORE_20261003.md、DECISIONS.md 59d155a。"

EXCHANGE_EVENT_ADOPTED: tuple[AdoptedFlag, ...] = (
    AdoptedFlag('--completion-score-max-chain', '2026-10-03', A1B_ADOPTION_REASON),
    AdoptedFlag('--a2-safety-review', '2026-10-03', A2_ADOPTION_REASON),
    AdoptedFlag('--death-history-boundary-start', '2026-10-03', DV2H2B_ADOPTION_REASON),
    AdoptedFlag('--post-counter-completion-recent-integrity', '2026-10-03', DV2H2B_ADOPTION_REASON),
    AdoptedFlag('--death-integrity-v2h2', '2026-10-03', DV2H2B_ADOPTION_REASON),
    AdoptedFlag('--death-freshness-cap', '2026-10-03', DV2H2B_ADOPTION_REASON),
    AdoptedFlag('--death-input-integrity', '2026-10-02', DV2D_ADOPTION_REASON),
    AdoptedFlag('--score-attribution-v2', '2026-10-02', DV2D_ADOPTION_REASON),
    AdoptedFlag('--post-counter-current-puyo-integrity-v2c', '2026-10-02', DV2D_ADOPTION_REASON),
    AdoptedFlag('--death-history-integrity', '2026-10-02', DV2D_ADOPTION_REASON),
    AdoptedFlag(
        '--exchange-event-static-gfe-dir models/exchange_event_v1_r1_hgb', '2026-10-02',
        "G_fe 段2b R1s (通常静止の G_fe のみ現行59列 HistGBC、仮想着弾は現行 LR) — 115試合 M_time −0.0043 [−0.0086, −0.0001]、M_agree −0.0005、±3帯反転 321→242 / 327→260、誤確定・3:00場面不変、G_fe P95 3.3ms。独立再計算一致・差分査読・CLI配線確認済み (wt_phase af583a5)。評価セット上の診断後に作った構成のため選択の楽観が残る。抽出元: wt_phase logs/gfe_phase_stage2b_20261002/SUMMARY.md、DECISIONS.md 2026-10-02。",
    ),
    AdoptedFlag(
        '--exchange-event-update', "2026-09-29",
        'E27+E31+E32固定検収: q .507977 / zenchi 91.9717% (7,664/8,333) / 誤発火1/35。E32b打ち切り再生でリークなし。E34c固定入力の再収集一致114,146/114,146。user承認2026-09-29。抽出元: scripts/_launch_e32.sh → run_e32 → run_e31/run_e27 → run_e17_ablation_20260928.worker、logs/e32/on/review/DONE.json。',
    ),
    AdoptedFlag(
        '--exchange-event-model-dir models/exchange_event_v3', "2026-09-29",
        'E27+E31+E32固定検収: q .507977 / zenchi 91.9717% (7,664/8,333) / 誤発火1/35。E32b打ち切り再生でリークなし。E34c固定入力の再収集一致114,146/114,146。user承認2026-09-29。抽出元: scripts/_launch_e32.sh → run_e32 → run_e31/run_e27 → run_e17_ablation_20260928.worker、logs/e32/on/review/DONE.json。',
    ),
    AdoptedFlag(
        '--exchange-event-live-count', "2026-09-29",
        'E27+E31+E32固定検収: q .507977 / zenchi 91.9717% (7,664/8,333) / 誤発火1/35。E32b打ち切り再生でリークなし。E34c固定入力の再収集一致114,146/114,146。user承認2026-09-29。抽出元: scripts/_launch_e32.sh → run_e32 → run_e31/run_e27 → run_e17_ablation_20260928.worker、logs/e32/on/review/DONE.json。',
    ),
    AdoptedFlag(
        '--exchange-event-death-guard', "2026-09-29",
        'E27+E31+E32固定検収: q .507977 / zenchi 91.9717% (7,664/8,333) / 誤発火1/35。E32b打ち切り再生でリークなし。E34c固定入力の再収集一致114,146/114,146。user承認2026-09-29。抽出元: scripts/_launch_e32.sh → run_e32 → run_e31/run_e27 → run_e17_ablation_20260928.worker、logs/e32/on/review/DONE.json。',
    ),
    AdoptedFlag(
        '--confirmed-death-hold', "2026-09-29",
        'E27+E31+E32固定検収: q .507977 / zenchi 91.9717% (7,664/8,333) / 誤発火1/35。E32b打ち切り再生でリークなし。E34c固定入力の再収集一致114,146/114,146。user承認2026-09-29。抽出元: scripts/_launch_e32.sh → run_e32 → run_e31/run_e27 → run_e17_ablation_20260928.worker、logs/e32/on/review/DONE.json。',
    ),
    AdoptedFlag(
        '--death-formula-guard', "2026-09-29",
        'E27+E31+E32固定検収: q .507977 / zenchi 91.9717% (7,664/8,333) / 誤発火1/35。E32b打ち切り再生でリークなし。E34c固定入力の再収集一致114,146/114,146。user承認2026-09-29。抽出元: scripts/_launch_e32.sh → run_e32 → run_e31/run_e27 → run_e17_ablation_20260928.worker、logs/e32/on/review/DONE.json。',
    ),
    AdoptedFlag(
        '--color-score-safety', "2026-09-29",
        'E27+E31+E32固定検収: q .507977 / zenchi 91.9717% (7,664/8,333) / 誤発火1/35。E32b打ち切り再生でリークなし。E34c固定入力の再収集一致114,146/114,146。user承認2026-09-29。抽出元: scripts/_launch_e32.sh → run_e32 → run_e31/run_e27 → run_e17_ablation_20260928.worker、logs/e32/on/review/DONE.json。',
    ),
    AdoptedFlag(
        '--multi-landing-death', "2026-09-29",
        'E27+E31+E32固定検収: q .507977 / zenchi 91.9717% (7,664/8,333) / 誤発火1/35。E32b打ち切り再生でリークなし。E34c固定入力の再収集一致114,146/114,146。user承認2026-09-29。抽出元: scripts/_launch_e32.sh → run_e32 → run_e31/run_e27 → run_e17_ablation_20260928.worker、logs/e32/on/review/DONE.json。',
    ),
    AdoptedFlag(
        '--midchain-completion', "2026-09-29",
        'E27+E31+E32固定検収: q .507977 / zenchi 91.9717% (7,664/8,333) / 誤発火1/35。E32b打ち切り再生でリークなし。E34c固定入力の再収集一致114,146/114,146。user承認2026-09-29。抽出元: scripts/_launch_e32.sh → run_e32 → run_e31/run_e27 → run_e17_ablation_20260928.worker、logs/e32/on/review/DONE.json。',
    ),
    AdoptedFlag(
        '--death-pending-ledger', "2026-09-29",
        'E27+E31+E32固定検収: q .507977 / zenchi 91.9717% (7,664/8,333) / 誤発火1/35。E32b打ち切り再生でリークなし。E34c固定入力の再収集一致114,146/114,146。user承認2026-09-29。抽出元: scripts/_launch_e32.sh → run_e32 → run_e31/run_e27 → run_e17_ablation_20260928.worker、logs/e32/on/review/DONE.json。',
    ),
    AdoptedFlag(
        '--hidden-row-death', "2026-09-29",
        'E27+E31+E32固定検収: q .507977 / zenchi 91.9717% (7,664/8,333) / 誤発火1/35。E32b打ち切り再生でリークなし。E34c固定入力の再収集一致114,146/114,146。user承認2026-09-29。抽出元: scripts/_launch_e32.sh → run_e32 → run_e31/run_e27 → run_e17_ablation_20260928.worker、logs/e32/on/review/DONE.json。',
    ),
    AdoptedFlag(
        '--prefire-snapshot', "2026-09-29",
        'E27+E31+E32固定検収: q .507977 / zenchi 91.9717% (7,664/8,333) / 誤発火1/35。E32b打ち切り再生でリークなし。E34c固定入力の再収集一致114,146/114,146。user承認2026-09-29。抽出元: scripts/_launch_e32.sh → run_e32 → run_e31/run_e27 → run_e17_ablation_20260928.worker、logs/e32/on/review/DONE.json。',
    ),
    AdoptedFlag(
        '--hidden-row-belief', "2026-09-29",
        'E27+E31+E32固定検収: q .507977 / zenchi 91.9717% (7,664/8,333) / 誤発火1/35。E32b打ち切り再生でリークなし。E34c固定入力の再収集一致114,146/114,146。user承認2026-09-29。抽出元: scripts/_launch_e32.sh → run_e32 → run_e31/run_e27 → run_e17_ablation_20260928.worker、logs/e32/on/review/DONE.json。',
    ),
    AdoptedFlag(
        '--post-counter-death-bound', "2026-09-30",
        'E35 打ち返し後の死亡上限 (コミット0f28a04)。 e36b検収 (R1b合図照合+E35+D5+D5b): q .507565 / zenchi 92.12% (7,676/8,333) / 誤発火1/39 / 3:00場面 2760.38秒 (門2766.0秒以内)。本番(E27+E31+E32 q .507977 / zenchi 7,664)から改善。user一括承認2026-09-30 (DECISIONS.md)。抽出元: logs/e36b/SUMMARY.json、scripts/run_e36b.py options()。第14試合1P 884秒の誤確定は残存 (D5b単体では直らない、既知)。',
    ),
    AdoptedFlag(
        '--single-death-proof-guard', "2026-09-30",
        'D5 単発着弾死亡を全応手の消去後盤面で証明してから確定 (71d378d)。 e36b検収 (R1b合図照合+E35+D5+D5b): q .507565 / zenchi 92.12% (7,676/8,333) / 誤発火1/39 / 3:00場面 2760.38秒 (門2766.0秒以内)。本番(E27+E31+E32 q .507977 / zenchi 7,664)から改善。user一括承認2026-09-30 (DECISIONS.md)。抽出元: logs/e36b/SUMMARY.json、scripts/run_e36b.py options()。第14試合1P 884秒の誤確定は残存 (D5b単体では直らない、既知)。',
    ),
    AdoptedFlag(
        '--single-death-proof-negative-only', "2026-09-30",
        'D5b 取消を生存枝・相殺可能の証明だけに限定 (0b8a76b)。 e36b検収 (R1b合図照合+E35+D5+D5b): q .507565 / zenchi 92.12% (7,676/8,333) / 誤発火1/39 / 3:00場面 2760.38秒 (門2766.0秒以内)。本番(E27+E31+E32 q .507977 / zenchi 7,664)から改善。user一括承認2026-09-30 (DECISIONS.md)。抽出元: logs/e36b/SUMMARY.json、scripts/run_e36b.py options()。第14試合1P 884秒の誤確定は残存 (D5b単体では直らない、既知)。',
    ),
    AdoptedFlag(
        '--post-counter-early-exit', "2026-09-30",
        'E35で最初に死なない候補が見つかった時点で残りの候補証明を省く (判定は同一、監査のproofsだけ短くなる)。 評価遅延対策B2 (6f370c8/92d5460): 保存記録5本の再生で現本番と出力一致 (内訳proofsを除く51/51ファイル)・門通過 (q .507567 / zenchi 7,671/8,333 / 誤発火0/37 / 3:00場面 2760.38秒 / 第14試合0)。rt由来のb20窓 (4,196通知) で最大 18.9秒→0.62秒・1秒超 66→0件。ノード上限(B1)は実在の死亡確定を1件失うため不採用 (座長判定、DECISIONS.md 2026-09-30)。抽出元: logs/multilanding_speed/RESULT.md、logs/multilanding_speed/gate_bounded2/SUMMARY.json。',
    ),
    AdoptedFlag(
        '--hidden-scenario-cap 256', "2026-09-30",
        '隠し段の得点候補が256通りを超えるとき重み上位256通りに絞る (合計重み保存、保存記録5本では発動せず、b20窓の43通知で発動し評価値8,234件中62件が最大3.4ポイント動く・判定変化0)。 評価遅延対策B2 (6f370c8/92d5460): 保存記録5本の再生で現本番と出力一致 (内訳proofsを除く51/51ファイル)・門通過 (q .507567 / zenchi 7,671/8,333 / 誤発火0/37 / 3:00場面 2760.38秒 / 第14試合0)。rt由来のb20窓 (4,196通知) で最大 18.9秒→0.62秒・1秒超 66→0件。ノード上限(B1)は実在の死亡確定を1件失うため不採用 (座長判定、DECISIONS.md 2026-09-30)。抽出元: logs/multilanding_speed/RESULT.md、logs/multilanding_speed/gate_bounded2/SUMMARY.json。',
    ),
    AdoptedFlag(
        '--exchange-event-switch-smoothing', "2026-10-01",
        '評価器切替時の表示値の飛び対策 (表示のみ・評価値/確率/死亡判定は不変)。(a)旧評価器値もEMA (b)EMA状態を表示値へ連続・試合境界は即時スナップ (c)物理イベントのない由来切替のみ0.5秒線形ブレンド・確定死亡は即時。 保存記録5本 (112,346フレーム・由来切替641回) の再生: 物理イベントなし・由来切替に帰属する飛び 34→16件 (−53%)・全飛び 342→297・試合境界の飛び 33→5。標準q .5075670 (不変) / 表示ベースq .506002→.506334 / 誤発火0/37 / 3:00場面 2760.25秒 / 第14試合0 / 確定死亡の即時性 5,544/5,544行 / 鮮度 (同値率 .1592→.1606・最長同値 85.97秒 同・更新回数/分 1513→1511) / 平滑前の列5記録全一致。zenchi 7,671→7,670 (門7,671に1フレーム未達、2948.483秒 第47試合の窓境界1フレーム: OFFは前試合の古いEMAが偶然勝者側の符号) のため **事後例外 (DECISIONS.md 2026-10-01、user承認)** で採用。抽出元: logs/switch_smoothing/RESULT.md、src/exchange_display_smoothing.py。',
    ),
)

# R1b 認識側の置き完了合図照合 (NEXT移動+掛け算式の2合図)。撃ち合い評価の再生CLIは
# 保存済み記録を読むだけで受け付けないため、EXCHANGE_EVENT_ADOPTED とは別バケットにする。
# RECOGNITION_ADOPTED は load_default kwargs へ機械変換され (enable_* 名の重複供給で
# TypeError)、かつ AST 固定テストで凍結されているため、そこへは入れない。
PLACEMENT_RECONCILE_ADOPTED: tuple[AdoptedFlag, ...] = (
    # 描画CLIを共有するライブ版にも届く。load_default用kwargsとは分離する。
    AdoptedFlag('--falling-puyo-integrity', '2026-10-02', DV2D_ADOPTION_REASON),
    AdoptedFlag(
        '--placement-signal-reconcile', "2026-09-30",
        'R1b (82d3baf) 置き完了合図の照合 (NEXT移動+掛け算式、おじゃま合図は含めない): e36b検収 (R1b合図照合+E35+D5+D5b): q .507565 / zenchi 92.12% (7,676/8,333) / 誤発火1/39 / 3:00場面 2760.38秒 (門2766.0秒以内)。本番(E27+E31+E32 q .507977 / zenchi 7,664)から改善。user一括承認2026-09-30 (DECISIONS.md)。抽出元: logs/e36b/SUMMARY.json、scripts/run_e36b.py options()。第14試合1P 884秒の誤確定は残存 (D5b単体では直らない、既知)。',
    ),
    AdoptedFlag(
        '--next-recolor-pair-guard', "2026-09-30",
        'cycle65 NEXT履歴色補正の対整合ガード (d3c5b0d)。確定遅延時に1手先のqueueで着地セルを上書きする誤りを止める (根因: logs/match14_misread/DIAGNOSIS.md)。 c65全長検収 (R1b+E35+D5+D5b+本ガード、E36事前登録門): q .507567 (門 .508040) / zenchi 92.06% (7,671/8,333、門7,671) / 誤発火0/37 (門1/35) / 3:00場面 2760.38秒 (門2766.0秒) / 第14試合1P 884秒の誤確定0。E36b本番比 q +0.000002 / zenchi -5フレーム / 誤発火 1→0。影子測定 4動画 書換セル正解 269→581/584 (悪化1セル zenchi 3155.75秒)。user一括承認2026-09-30 (DECISIONS.md)。抽出元: logs/c65_guard/e36b_on、logs/c65_guard/RESULT.md。',
    ),
    AdoptedFlag(
        '--verification-pending-chain-expiry', "2026-09-30",
        '連鎖後答え合わせの pending を新しい連鎖開始で失効 (0c0c338)。q第14試合の幻おじゃま (1,5)(2,5)(3,5) の根因B (logs/match14_misread/DIAGNOSIS.md)。 影子測定 4動画: 答え合わせ248件中失効10、書込セル正解 1,106→1,179/1,467 (悪化0・改善73)。 全長ゲート (事前登録 e9e1912、現本番比非悪化): q .507567 / zenchi 7,671/8,333 / 誤発火0/37 / 3:00場面 2760.38秒 / 第14試合誤確定0 — 評価出力は現本番と同一 (認識記録は q 693行・fc 1,365行が変化、盤面品質のみ改善)。user一括承認2026-09-30 (DECISIONS.md)。抽出元: logs/pending_expiry/e36b_on、logs/pending_expiry/RESULT.md。',
    ),
)


def _join(flags: tuple[AdoptedFlag, ...]) -> str:
    """フラグ文字列を空白区切りで連結する。"""
    return " ".join(f.flag for f in flags)


def recognition_flags() -> str:
    """認識の本番構成フラグを返す (収集・表示の両方が受け付けるもの)。"""
    return _join(RECOGNITION_ADOPTED)


def recognition_load_default_kwargs() -> dict[str, "float | bool"]:
    """RECOGNITION_ADOPTED を RecognitionPipeline.load_default() 用 kwargs に変換する。

    2026-08-13 是正 (根因調査の副次発見): scripts/visualize_advantage_overlay.py が
    RECOGNITION_ADOPTED (本番採用の認識フラグ群) を load_default() へ一切
    転送しておらず、デモ/レビュー動画が本番より劣化した認識 (バーストガード等が
    無効) で生成されていた (2026-08-08 の --early-fire-reaction 付け忘れ事故と
    同型)。collect_boards_lean.py の argparse dest 名 (enable_effect_gate 等、
    同ファイル 769-1009 行) は RecognitionPipeline.load_default() のキーワード
    引数名と完全に一致しているため、 "--xxx-yyy" 形式のフラグ名を dest 名
    "xxx_yyy" へ機械的に変換するだけで両者と同じ呼出し経路になる (RECOGNITION_
    ADOPTED に新しいフラグを追記するだけで overlay 側も自動追従し、個別配線の
    抜け漏れを構造的に根絶する)。値付きフラグ ("--burst-gate-open-threshold
    0.954" 等) は float にパースし、値無しフラグ (store_true 系) は True にする。
    """
    kwargs: dict[str, "float | bool"] = {}
    for f in RECOGNITION_ADOPTED:
        parts = f.flag.split()
        name = parts[0].lstrip("-").replace("-", "_")
        kwargs[name] = True if len(parts) == 1 else float(parts[1])
    return kwargs


def collect_flags() -> str:
    """collect_boards_lean 用の全フラグ (共通 + 収集専用) を返す。"""
    return _join(RECOGNITION_ADOPTED + COLLECT_ONLY_ADOPTED)


def advantage_overlay_flags() -> str:
    """有利不利オーバーレイの本番構成フラグを返す。"""
    return _join(ADVANTAGE_ADOPTED)


def exchange_event_flags() -> str:
    """撃ち合いイベント評価の本番構成フラグを返す。"""
    return _join(EXCHANGE_EVENT_ADOPTED)


def placement_reconcile_flags() -> str:
    """R1b認識側の合図照合フラグ (描画CLIだけが受け付ける) を返す。"""
    return _join(PLACEMENT_RECONCILE_ADOPTED)


def visualization_flags() -> str:
    """認識オーバーレイ表示の本番構成フラグを返す。"""
    return _join(VISUALIZATION_ADOPTED)


def describe() -> str:
    """採用済みフラグの一覧を人が読める形で返す (生成物への記録用)。"""
    lines: list[str] = []
    for title, flags in (
        ("認識(共通)", RECOGNITION_ADOPTED),
        ("認識(収集専用)", COLLECT_ONLY_ADOPTED),
        ("有利不利", ADVANTAGE_ADOPTED),
        ("撃ち合いイベント評価", EXCHANGE_EVENT_ADOPTED),
        ("認識(置き完了合図の照合)", PLACEMENT_RECONCILE_ADOPTED),
        ("表示", VISUALIZATION_ADOPTED),
        ("連鎖シミュレーション", CHAIN_SIM_ADOPTED),
        ("指標大整理", INDICATOR_REORG_DECISIONS),
        ("学習データビルダー", LEARNING_DATA_BUILD_ADOPTED),
        ("配布版ライブ (CLI/ブリッジ専用、描画CLIへは渡さない)", LIVE_BRIDGE_ADOPTED),
    ):
        lines.append(f"[{title}]")
        for f in flags:
            lines.append(f"  {f.flag}  (採用 {f.adopted}) — {f.reason}")
    return "\n".join(lines)


# 2026-10-07: 公開時のゲーム画像除外に向けた候補。比較未合格につき未採用。
IMAGE_FREE_SCORE_OCR_DEFAULT: bool = False
# 2026-10-07: ×印の蒸留候補。連続フレーム・下流比較未合格につき未採用。
IMAGE_FREE_UI_MASK_DEFAULT: bool = False


if __name__ == "__main__":
    print(describe())

# 2026-10-07: 全UI画像非同梱候補。連続比較の合格前につき未採用。
IMAGE_FREE_ALL_DEFAULT: bool = False
# 2026-10-12: 10/10決定の実装候補。評価器の予告判定を得点台帳にする。本番未採用。
B3_LEDGER_MODE_DEFAULT: bool = False
# 2026-10-12: 着地量をアイコン差から台帳の観測着弾へ切替。本番未採用。
B3_LEDGER_LANDING_DEFAULT: bool = False
# 2026-10-12: lineをlargeに復号。v6のlarge=0分布を変えるため、再学習とセットでのみON。
B3_ICON_LINE_ALIAS_DEFAULT: bool = False
# 2026-10-25: 案A。受ける側から見た『正味の予告』列 (net_pending_pN / df_sl_net / ta_*_slnet) を B3 行の末尾へ足す。
# 既存の sl_pending・sl_inflight から導出するだけで、台帳本体・学習済み重みは変えない。既定OFF・未採用
# (OFFでは列も値も不変)。採用は PC2 の再学習と LL 比較の後。前提: sl_inflight 列 (PUYO_B3_M3=1)。
# 根拠: <work>/ledger_chain_diag_20261025/RESULT.md (確定後の値と符号一致 98.2%、アイコン 85.1%)。
B3_NET_LEDGER_COLS_DEFAULT: bool = False
# 2026-10-25 (記録のみ・既定値は変えない): 配信の試験フラグ b3_c2 (src/phase_j/live_obs_trial.py、既定OFF、OBS 試験版専用)。
# 勝率モデルを M3 から C2 (= M3 − アイコン由来 13 列 + 上の正味 17 列 = 299 列) に替え、予告アイコン CNN を読まない。
# 前提: b3_judgment・b3_async・b3_m3・b3_skip_icon_reading (ON で PUYO_B3_NET_LEDGER_COLS='1'、OFF で '0' を配送)。
# 根拠: user 決定 2026-10-25 (「アイコンは外す、C2 が合格したら C2 の B3 を使う」)、C2 は PC2 m3_net で PASS
# (対差 95% 上端 <= +0.00076 かつ ECE 11/11 層、share/m3_net_20261025/results_pc2/RESULT_C2.json)。
# 本番採用は別 (M3 の 5 フラグ・B3_SKIP_ICON_READING_DEFAULT・B3_LEDGER_*・本番モデルの切替を user に見せてから)。
# 2026-10-25: 配信の試験フラグ b3_ensemble (src/b3_ensemble.py・src/phase_j/live_obs_trial.py、既定OFF・未採用、前提は b3_c2)。
# C2 の seed 違いを複数並べ、同じ行の確率を平均する (既定は確率の算術平均、設定で logit 平均)。b3_model はマニフェスト JSON。
# OFF では設定・出力・b3_config.json・子 process の動きが従来と同一 (src は load_model の分岐 1 か所だけ)。
# 根拠: user 承認 2026-10-25 15:35。CV は seed 3 本 (1003/1004/1005) の成績の平均で、seed が違うと配信の予測が平均 0.015・
# 最大 0.26〜0.36 違う (HGB は学習 20 万行超で乱数の抜き取りから bin を決める) ため、配信を CV と同じ条件へ寄せる受け皿。
# 本番採用は別 (最終学習のモデルの到着・PC2 の平均の仕方の測定・マニフェストの SHA256 登録の後に user へ)。
B3_ENSEMBLE_DEFAULT: bool = False
# 2026-10-13: h限定で採用。48,367通知の参照値一致、表示・死亡監査一致。通常既定はOFF。
LIVE_REFERENCE_TIMES_ONLY_DEFAULT: bool = False
# 2026-10-13: i限定ON。48,367通知・表示3,191行・死亡証明一致、連鎖P95 47.23→19.33ms。
LIVE_NEAR_FUTURE_CACHE_DEFAULT: bool = False
# 2026-10-13: j限定ON。全48,367通知・表示3,191行・死亡証明一致。通常既定はOFF。
LIVE_EXACT_CONNECTIVITY_DEFAULT: bool = False
# 2026-10-13: k限定ON。全通知・表示・死亡証明一致。通常既定はOFF。
LIVE_EXACT_NATIVE_DEFAULT: bool = False
# 2026-10-13: 保持中の画像を上書きしない補正buffer候補。全件・55分門未達、未採用。
LIVE_COLOR_BUFFER_DEFAULT: bool = False
# 2026-10-13: intra=2・スピンなしの認識候補。保存画像の短区間のみ検証、未採用。
LIVE_ONNX_PARALLEL_DEFAULT: bool = False
LIVE_PROOF_DELIVERY: tuple[AdoptedFlag, ...] = (
    AdoptedFlag('live_exact_native', '2026-10-13',
        'build_20261012k限定ON。UNKNOWN壁を保持する単盤面Rustの静止判定・連鎖計算。'
        'FIX4基準で48,367通知の参照/表示/EMA/署名、表示3,191行の全列バイト、死亡履歴/証明一致。'
        'j相当との再投入でt=7027.10は1,249.72→754.29ms、hは4,158.35→3,827.06ms。'
        '候補・探索順・ノード上限・通知は維持。局所連結はjのPython版を維持。'
        '数秒の計算停止は残り、OBS実機品質合格ではない。'),
)
LIVE_STALL_DELIVERY: tuple[AdoptedFlag, ...] = (
    AdoptedFlag('live_exact_connectivity', '2026-10-13',
        'build_20261012j限定ON。評価子の連結確認と消去グループ生成だけを厳密短縮。'
        'FIX4基準で表示3,191行の全列バイト・48,367通知署名・死亡履歴/証明一致。'
        '更新+参照の最大2,526.67→1,267.61ms、h/g停止試合の最大6,949.55/6,964.31→4,151.75/4,193.01ms。'
        '候補・探索順・ノード上限・通知を維持。数秒の計算停止は残り、OBS実機品質合格ではない。'),
)
LIVE_SPEED3_DELIVERY: tuple[AdoptedFlag, ...] = (
    AdoptedFlag('live_near_future_cache', '2026-10-13',
        'build_20261012i限定ON。コードSHAと全探索入力で結果を再利用し先頭32命中を実計算照合。'
        '保存入力48,367通知・表示3,191行・死亡履歴/証明一致。連鎖14,503件のP95 47.23→19.33ms。'
        'おじゃまP99/最大は未改善、保存到着によるキューP95見積り10→9。実機合格ではない。'),
)
# 2026-10-13: 同一入力の外枠単独失敗の短期猶予。負例検証前につき既定OFF。
LIVE_BORDER_GRACE_DEFAULT: bool = False
# 2026-10-22: 撃った連鎖の火力 X の段1 直接公開 (設計 <work>/live_x_timing_20261022/DESIGN.md §2)。
# ON は v4.9f の送受信を2 threadに分け、X の初出で評価所有者を起こして B3 完了と同じ経路で即公開し、
# 判定IDへ X の完了時刻を足し、v4.9f 予測時間の例外門 (MAX_LOW_SPEC_MS) を記録だけにする。
# 実装・試験のみ。実時間の効果測定 (L 条件 ABBA) 前につき既定OFF・本番未採用 (採用は user 判断)。
LIVE_X_DIRECT_PUBLISH_DEFAULT: bool = False
# 2026-10-22: 撃つ直前の暫定 X (段3、DESIGN live_x_timing_20261022 §3・§8)。実装のみ・未採用。
# shadow = 記録のみ (live_x_prefire.jsonl)、live_x_prefire = 別欄 x_provisional の表示。
# どちらも live_v49f=true が前提。v4.9f の p1 へは入れない (親裁定 (2)、p1 採用は user 判断)。
LIVE_X_PREFIRE_SHADOW_DEFAULT: bool = False
LIVE_X_PREFIRE_DEFAULT: bool = False
# 2026-10-23: コマ落ちに強い数え方 (設計 <work>/frame_drop_robust_20261023/DESIGN.md)。
# ON では認識と B3 入力の回数の判定を、仮想 30Hz の刻みと実観測の下限 m = max(2, ⌊N/4⌋) で数える。
# 全コマ処理では出力が OFF とビット単位で同じ (要件 E1)。配信 (launcher / live_speed) だけで切り替え、
# 学習データの収集・レビュー台は OFF のまま (user 承認 A 案、2026-10-23)。
# 入っている型 (2026-10-23 の最初の型): R01・R04 (型 A)、R02・R03 (型 C)、B07 (型 H2)。
# 2026-10-23 追加: R35 (型 H1、見逃した設置を NEXT の移動と色ぷよ 2 個増・連鎖開始から補う、user 承認)。
# 2026-10-23 追加: R06・R10 (型 A、連鎖後とおじゃま後の静止待ち 8 回)、R15 (型 B、おじゃま後の煙待ち)。
# 実装・試験のみ。効きの測定 (段 2b) 前につき既定OFF・本番未採用 (採用は user 判断。採用時は
# LIVE_FRAME_DROP_ROBUST_DELIVERY に採用日と根拠を記録してから既定値を変える)。
LIVE_FRAME_DROP_ROBUST_DEFAULT: bool = False
# 2026-10-23: 処理できるコマ数を上げる試作 (<work>/live_fast_20261023/PLAN.md)。実装・試験のみ、本番未採用。
# b3_warning_batch: 予告アイコン CNN を存在判定を通ったセルだけ 1 コマ 1 回の forward にまとめる
#   (src/batched_ojama_warning.py)。判定の分岐は旧 detect と同じ、閾値±1e-4 は単体推論で判定し直す。
# b3_warning_prefetch: その読取を盤面認識の前に 1 本の worker で始めておく (b3_warning_batch が前提)。
# 出力不変の確認と速度は PLAN.md。採用は user 判断。
B3_WARNING_BATCH_DEFAULT: bool = False
B3_WARNING_PREFETCH_DEFAULT: bool = False
# 2026-10-24: user「予告の確認は外に出していい」。認識待ちをなくす試作、本番未採用。
# ON は専用スレッド＋最新一件の batch 処理。parallel/prefetch より優先、skip は最優先。
B3_WARNING_ASYNC_DEFAULT: bool = False
# 2026-10-23: 上のフラグが ON のとき、R06・R10・R15 のうち刻みで数える箇所 (段 2b の箇所ごとの採否、
# 受領票 D:/puyo_analyzer/frame_drop_a_20261023/RECEIPT_R06_R10_R15.json)。user 承認 (12:55) の条件
# 「足す前の版 (R35 まで) より間引きの台 4 種で悪くならない。悪くする箇所は外す」を、全体の平均差 0.05pt
# (間引きの OFF 同士の揺れ) を超える悪化で判定した:
#   R15 入れる (L1 -0.008pt・half -0.005pt)
#   R06 外す   (L1 +0.187pt。half -0.111pt・third/obs21c は改善で向きが割れる)
#   R10 外す   (外した版と比べ half +0.429pt・obs21c +0.063pt。試合 1 終盤の状態の道筋は R10 で F に戻る)
# R06・R10 の実装は残す (ここへ足せば ON の中で働く)。R01・R02/R03・R04・B07・R35 はフラグ ON で常に働く。
# 2026-10-23 夜 (23y の OBS 試験版、user 承認の条件「間引きの台で足す前の版より悪くならなければ入れる」): FDR の最終構成 (R15 のみ)
# を間引きの台 4 パターンで測り直した。足す前の版 (R35 まで) との平均差は L1_OFF +0.009・half +0.007・third +0.032・obs21c +0.086pt。
# obs21c が線 0.05pt を超えたので R15 も外す。既定を空にする (= 刻みで数えるのは R01・R02/R03・R04・B07・R35 だけ)。
# 受領票 D:/puyo_analyzer/packaging/build_20261023_obs_y/work/thin/summary.json。R15 の実装は残す (ここへ足せば働く)。
LIVE_FRAME_DROP_ROBUST_SITES: frozenset[str] = frozenset()
# 2026-10-23: R37 試合開始の判定 (設計 <work>/frame_drop_robust_20261023/DESIGN.md §11.2・§9 Q9)。
# ON = RecognitionPipeline(enable_lockdown_score_zero_then_move_release=True)。user の決まり (8/19「必ず試合前
# スコアは 0」、10/23「得点が 0 同士なら試合前、動いた瞬間試合開始」): 試合終了 → 両者 0 点 (試合前) →
# どちらかの得点が増え、増えた値が 2 回続けて読めたら試合開始。盤面の画素の確認は使わず、試合終了の固定
# (5 秒) の最中でも開始する (固定の残りで試合外・演出扱いにしない)。今のラッチの解除経路は補助として残り、
# ON は OFF より遅くならない。enable_post_match_lockdown_latch=True が前提 (False との組合せは ValueError)。
# 背景: 盤面の画素の確認は 8/18 に「対戦カード紹介の装飾スコアのカウントアップ」対策で足されたが、保存画像
# (data/verify/boundary_impl_verify_2026-08-18/frames_c18・c20) の実体は次の試合の最初の設置 (診断の取り違え)。
# 実測 (video_30 試合 2 の始まり、同じ区間 222〜238 秒を本番の認識設定で流した。受領は scratchpad の r37_probe):
#   全コマ: 試合開始の判定 OFF 232.267 秒 → ON 230.300 秒 (最初の得点 230.267 秒の直後)
#   13762 コマを飛ばす間引き (3 コマに 1 回): OFF 234.000 秒 (設計 §11.2 の L2・L3 と同じ) → ON 230.550 秒
#   OFF は基準点 (6257b923) と 960 コマ全て一致 (開始判定・固定・ラッチ・得点・両側の状態)。
# 実装・試験のみ。配信 (live_speed.recognition_config) への配線はまだ。採用は user 判断 (採用時は
# AdoptedFlag で採用日と根拠を記録してから既定値を変える)。
LIVE_LOCKDOWN_SCORE_ZERO_THEN_MOVE_DEFAULT: bool = False
# 2026-10-24 試作: 告知の左右組と0/0までの保持。録画1本の実測のみのため既定OFF。
LIVE_MATCH_END_PAIR_DEFAULT: bool = False
# 2026-10-14 user決定(案2 物理ルール): 確定盤の3列目12段目 (row1 col2) の幻ぷよ (×印・操作中ツモ)
# を公開盤から外す。実装は src/row1_guard.py。
# 2026-10-15 user承認「両方OK」で本番ON (v2.1、置き終わりの合図で即確定)。根拠は LIVE_ROW1_GUARD_DELIVERY。
LIVE_ROW1_GUARD_DEFAULT: bool = True
# 2026-10-17: user 10/16決定「試験合格後に製品ON」をeo2と独立検収で満たした。
LIVE_SCORE_ROI_ALIGNMENT_DEFAULT: bool = True
LIVE_SCORE_ROI_ALIGNMENT_DELIVERY: tuple[AdoptedFlag, ...] = (
    AdoptedFlag('align_score_roi', '2026-10-17',
        'user 2026-10-16決定。PC2 eo2 (4b590092、5動画×OFF/ON) 全条件PASS・'
        'PC2 Claude独立検収合格。39/94の有効式0.51→16.07%・5.56→17.22%、(0,-2)px。'
        '29/65/79は(0,0)px、認識・表示勝率・死亡・個数DTO・式得点原票が完全一致。'
        '固定枠平均NCC閾値0.80を維持。settled事後EMAの既知非決定性は公開表示と区別。'),
)
LIVE_ROW1_GUARD_DELIVERY: tuple[AdoptedFlag, ...] = (
    AdoptedFlag('live_row1_guard', '2026-10-15',
        'PC2 en段4 (5動画×OFF/ON、HEAD 5bf5bb81)・PC1再集計: 対象外セル差0/68,273,282、'
        '表示勝率差0 (442,101コマ)・死亡判定差0 (246試合)。再現した幻の陽性コマ461→191 (59%減)、'
        '持続約5秒→0.7〜2.2秒、2件は不変。品質門の不合格は着地未反映2件でOFFでも同じ既存欠測。'),
)
LIVE_FIX4_DELIVERY: tuple[AdoptedFlag, ...] = (
    AdoptedFlag('live_reference_times_only', '2026-10-13',
        'build_20261012h限定ON。10/12保存入力48,367通知で物理時刻一致。'
        '表示と死亡証明は同一、連鎖中の参照平均8.811→0.045ms。'
        '通知・評価・EMA更新を省略せず、不要な履歴の深い複製だけを除く。'),
)

# 2026-10-12 user承認A: v6を維持し、配布eだけ予告CNNを省略して欠測を渡す。
B3_SKIP_ICON_READING_DEFAULT: bool = False
B3_ICON_OFF_DELIVERY: tuple[AdoptedFlag, ...] = (
    AdoptedFlag('b3_ledger_mode / b3_ledger_landing / b3_skip_icon_reading', '2026-10-12',
        'build_20261012e限定でON。予告は得点台帳を主とし、アイコンは補助のため省略。'
        'v6・列構成を維持し、既存のNaN欠測経路へ渡す。大アイコン補正はOFF。'),
)


def b3_ledger_mode_enabled() -> bool:
    """配布設定をspawn先でも受け取る。未指定時は本番既定値を使う。"""
    import os
    return os.environ.get('PUYO_B3_LEDGER_MODE', str(int(B3_LEDGER_MODE_DEFAULT))) == '1'


def b3_ledger_landing_enabled() -> bool:
    """着地量の入力元を配布設定から解決する。学習側の既定は変更しない。"""
    import os
    return os.environ.get('PUYO_B3_LEDGER_LANDING', str(int(B3_LEDGER_LANDING_DEFAULT))) == '1'


def b3_net_ledger_cols_enabled() -> bool:
    """正味の予告列 (案A) を行へ足すか。環境変数 PUYO_B3_NET_LEDGER_COLS=1 で spawn 先にも伝える。既定OFF。"""
    import os
    return os.environ.get('PUYO_B3_NET_LEDGER_COLS', str(int(B3_NET_LEDGER_COLS_DEFAULT))) == '1'
# 2026-10-24: 入力検証の小条件を観測する試作。判定変更前の原因収集用、既定 OFF。
LIVE_VERIFY_REASON_LOG_DEFAULT = False

# 2026-10-24: 配信本体の HSV を既存 Rust 実装へ切り替える試作。既定 OFF。
# 2026-08-20 のビット一致検証と 08-21 の短縮実測が根拠。実効値は認識子で記録する。
LIVE_NATIVE_HSV_MAIN_DEFAULT: bool = False
# 2026-10-24: OBS の同期 OCR によるコマ落ち対策。実機採用前のため既定 OFF。
LIVE_VERIFY_ASYNC_DEFAULT = False

# 2026-10-24: 起動と停止のガード (OBS 試験 24b の精査)。
# 二重起動の許可。False = 二重起動を検出して止める (PuyoLive 1 つの正常な使い方は変わらない)。
# 根拠: Windows の ThreadingHTTPServer は SO_REUSEADDR で同じ port への 2 つ目の bind が黙って成功し、
# 24b では 2 つ目が HTTP 送信 0 件のまま 2 時間 CPU を取り合った (単体の実測: tests/test_stream_overlay_exclusive.py)。
ALLOW_SECOND_INSTANCE_DEFAULT = False
# カメラの再接続。入力経路を変えるため試作・既定 OFF。実機 (OBS 仮想カメラの再起動) での検証前。
LIVE_CAPTURE_RECONNECT_DEFAULT = False
# B3 作業 process が Ctrl+C を無視し、親の終了の合図だけで止まる。停止経路を変えるため試作・既定 OFF。
# 根拠: 手で止めた 18 件の b3_async_status.json が全て finished=null (作業 process が KeyboardInterrupt で
# done を返さず死ぬ。smoke_NEW の b3_updates.jsonl に {"fatal": "KeyboardInterrupt: "})。
B3_WORKER_GRACEFUL_STOP_DEFAULT = False

# 2026-10-24: 盤面読取 (read_board / read_board_hsv_only / classify_batch / CNN 前処理) の
# マスごとの Python ループを numpy・Rust の一括呼出に置き換える試作 (src/live_board_vectorized.py)。
# 既定 OFF。根拠: 出力を 1 ビットも変えずに 1 コマの呼出回数と毎コマの 6MB 新規確保を減らす。
# 配布 Python・配布 CNN (cnn_phase_b_large_v2.pt) の実コマ (d04 動画 / 23y 録画) で、
# ON と OFF の盤面・確率・HSV 画素・内部記録・PipelineResult がビット一致 (0 件の違い)。
# 同じ実コマ 1,000 コマの交互測定で read_both_boards 平均 9.4〜14.2 → 7.1〜10.7ms、
# 認識 1 コマ平均 19.5〜23.6 → 16.8〜19.8ms (測定条件・母数は
# scripts/verify_live_board_vectorized.py / bench_live_board_vectorized.py の出力を参照)。
# 実機での採用前に OBS 試験で実効値 (live_trial_flags.json の runtime) を確認すること。
# ネイティブ HSV (LIVE_NATIVE_HSV_MAIN_DEFAULT) が無効のときは HSV 分類の一括だけが働かない。
LIVE_BOARD_VECTORIZED_DEFAULT: bool = False

# 2026-10-25: 認識ループの Python 側の出力不変の高速化 (src/recog_speed_py.py)。既定 OFF・未採用。
# 24d の単独試験 (C2) は認識 1 コマ平均 46.97ms・周期 48.17ms (20.76 コマ/秒)。洗い出し
# (<work>/python_hotspots_20261025/RESULT.json) の「出力を変えずに削れる」項目だけを束ねる:
# read_board_hsv_only の同コマ重複呼び・NextSlideDetector の前コマのグレー化・発火前画像リングの
# 盤面領域だけの保持と一括分類・表示得点の再読 (B3_SHARE_SCORE_READ が前提)。
# read_board の中身・hybrid_classifier のループ (Rust 移植の担当) には触れない。
# ON にする前の条件: <work>/recog_speed_py_20261025/RESULT.md の全行一致 (出力不変) の母数を確認し、
# user が承認すること。採用時は AdoptedFlag で採用日と根拠を記録する。
LIVE_RECOG_SPEED_PY_DEFAULT: bool = False
# 同上のうち、壁時計に依存する 1 項目だけを別の旗にする: 終了の組の crop を worker が取り出すときに複製する
# (毎コマの約 5MB の複製を 0.5 秒に 1 回へ)。終了の組の照合は元から 0.5 秒ごとの壁時計で動き、評価に使うコマが
# 1 コマずれうるため、全行一致での確認ができない。単体 (決定的な待ち方) の試験だけが根拠。既定 OFF・未採用。
LIVE_RECOG_SPEED_PY_PAIR_LAZY_DEFAULT: bool = False
# 2026-10-25: read_board の 1st pass (背景パッチ NCC・HSV 中央値・ガード分類) を Rust の 1 回呼出にする試作
# (src/live_native_read_board.py、native/puyo_core/src/board_first_pass.rs)。既定 OFF。
# 認識 1 コマの 1 割を占めた Python のセル単位ループ (盤面 1 枚で NCC 72 回・中央値・ガード分類) を消す。
# live_board_vectorized が ON のときだけ働く。判定は従来経路と全セルで一致させる (整数で決まる値はビット一致、
# NCC は numpy と同じペアワイズ総和で、np.dot の最終桁だけ CPU 依存)。根拠・母数・時間は
# <work>/recog_speed_rust_20261025/RESULT.md と scripts/verify_live_native_read_board.py の出力を参照。
# 実機での採用前に OBS 試験で実効値 (live_trial_flags.json の runtime) を確認すること。
LIVE_NATIVE_READ_BOARD_DEFAULT: bool = False

# 2026-10-26: OBS 24f 試作の表示切替。未採用、配布の試験腕だけで有効にする。
LIVE_WINRATE_SWITCH_DEFAULT: bool = False

# C群: 採用日空欄。2026-10-26 の独立比較候補、本番未採用。
# 2026-10-27 24h配線候補。未採用、既定OFF。PC2のD/ref3と配信検収後に採否を決める。
LIVE_24H_DEFAULT = False
LIVE_C_ITEMS_DEFAULTS: dict[str, bool] = {'live_color_roi': False, 'live_capture_thread': False, 'live_boundary_half_second': False, 'live_hsv_lazy': False}
LIVE_C_ITEMS_CANDIDATES: tuple[AdoptedFlag, ...] = (
    AdoptedFlag('live_color_roi', "", '使用画素だけ補正、全画面消費者は全画面。逆変換入力の全行一致が条件。'),
    AdoptedFlag('live_capture_thread', "", '取得を専用スレッドへ。品質・速度評価は user の OBS 試験。'),
    AdoptedFlag('live_boundary_half_second', "", '同期の試合状態・OCRゼロ判定を0.5秒周期へ。境界時刻と表示品質で検証。'),
    AdoptedFlag('live_hsv_lazy', "", 'HSV盤を最初の消費時まで遅延。復旧・色保護・おじゃま検出を維持。'),
)

# 2026-10-28: 撃った盤は配信の比較試験用。採用未決定のため既定OFF。
LIVE_B3_FIRE_BOARD_DEFAULT = False


# 2026-11-01 採用: PC1 Claude の release_prep_20261101/PROMPT と22:4xの追補。
# 24w2の全コマ確認済み構成を配布入口で使用する。Aは学習列・モデル・表示の一組、
# Bは認識・終了・復帰の修正。ensembleと直接公開もAへ移すとの判断を含む。
# 下位の実験CLIやUserConfigの生データ型は比較再生用の従来値を保ち、
# 利用者JSONはlauncherの検証前にこの既定を補う。明示値を優先し、旗は削除しない。
RELEASE_LIVE_A_FLAGS: tuple[str, ...] = (
    'live_24h', 'live_24h_retr', 'live_24h_xc', 'live_24h_tape_next',
    'live_24h_media_fresh', 'live_24h_row_time', 'live_24h_retr_h_forecast',
    'live_24h_early_submit', 'live_24h_stable_key', 'live_24h_xc_same_time',
    'live_chain_split_place', 'live_b3_fire_board', 'b3_c2', 'b3_skip_icon_reading',
    'live_24h_retr_second', 'live_24h_retr_move050', 'live_24h_retr_deadline_extend',
    'b3_ensemble', 'live_x_direct_publish',
)
RELEASE_LIVE_B_FLAGS: tuple[str, ...] = (
    'live_frame_drop_robust', 'live_lockdown_score_zero_then_move_release',
    'live_match_end_pair', 'live_match_end_feed_on_hold', 'match_end_yatta_text_template',
    'live_capture_reconnect', 'b3_worker_graceful_stop', 'live_verify_soft_hold',
    'live_native_hsv_main', 'live_native_read_board', 'b3_share_score_read',
    'live_color_roi', 'live_death_check_retr', 'live_verify_read_fast',
)
# 2026-11-02 PC1採用。24j〜24w2のOBS試験と全コマの確認でONのまま回った19旗。
# 24w2と同じ動きで配布する。明示Falseによる巻き戻しのため旗は残す。
RELEASE_LIVE_OBS_FLAGS: tuple[str, ...] = (
    'b3_time_gate', 'live_score_cadence', 'live_static_exact', 'fast_first_judgment',
    'evaluation_freshness_gate', 'terminal_display_hold',
    'live_disable_placement_reconcile', 'b3_score_matmul', 'live_reference_times_only',
    'b3_ledger_mode', 'b3_ledger_landing', 'live_near_future_cache',
    'live_exact_connectivity', 'live_exact_native', 'live_verify_async',
    'live_board_vectorized', 'live_hsv_lazy', 'live_recog_speed_py',
    'live_recog_speed_py_pair_lazy',
)
# 同じ11/2判断によりOBS台本の環境を配布入口にも適用する。利用者の環境を優先。
RELEASE_LIVE_ENV_DEFAULTS: dict[str, str] = {
    'PUYO_LIGHT_ROBUST_CELLS': str(int(DISTRIBUTION_LIGHT_ROBUST_CELLS)),
    'PUYO_W68_INFERENCE_PENDING': str(int(DISTRIBUTION_W68_INFERENCE_PENDING)),
    'PUYO_MIDGAME_BOARD_LAYER': str(int(DISTRIBUTION_MIDGAME_BOARD_LAYER)),
    'PUYO_COUNTER_FIRE_SHIFT': str(int(DISTRIBUTION_COUNTER_FIRE_SHIFT)),
    'PUYO_UPDATE_BUTTON': str(int(DISTRIBUTION_UPDATE_BUTTON)),
    'PUYO_LEDGER_PARTIAL': '1', 'PUYO_X_BOARD_ONLY': '1', 'PUYO_X_B3_BOUNDARY': '1',
    # 2026-11-02 PC1: 配布の参照exe起動を防止。threadsは24w2コードの既定2。
    'PUYO_OBS_AMA_RUST': '1', 'PUYO_OBS_AMA_RUST_THREADS': '2',
    'PUYO_OBS_CPU_ISOLATION': '0', 'PUYO_OBS_AMA_IDLE': '0',
    'PUYO_OBS_EXACT_SEED': '1', 'PUYO_OBS_PREPARE_EXACT': '1',
    # 2026-11-09 user 決定 (0.6.4 に入れる)。W67: 長い連鎖の終わり際の再通知で EX が台帳を作り直し、
    # 送り済み・相殺済みを二重に数える不具合の直し。根拠は codex_jobs/quality_exchange_064_20261109
    # (場面4 の逆向きを解消、3 本で台帳の再開始 6→2・ちらつき 20→12、公開試験 1,829 件で新しい失敗 0)。
    'PUYO_EX_CHAIN_IDENTITY': '1',
}
# OBS Virtual Cameraのみ24w2と同じ色行列補正。他機器の既定offは維持する。
RELEASE_OBS_DEVICE_NAME = 'OBS Virtual Camera'
RELEASE_OBS_COLOR_CORRECTION = '601to709'

# Aの前提と運用値。24w2のC2/EX資産構成、200msの遅延、2秒の鮮度を維持する。
# モデルの中身・SHAの更新は本ジョブでは行わず、配布時のmanifest検査に従う。
RELEASE_LIVE_DEFAULTS: dict[str, bool | float] = {
    **dict.fromkeys(RELEASE_LIVE_A_FLAGS + RELEASE_LIVE_B_FLAGS, True),
    **dict.fromkeys(RELEASE_LIVE_OBS_FLAGS, True),
    'b3_judgment': True, 'b3_async': True, 'b3_m3': True,
    'live_24h_ex_speed': True, 'live_24h_ex_fire_first': True, 'live_24h_ama': True,
    'live_24h_delay_ms': 200.0, 'live_24h_ex_fresh_sec': 2.0,
    # 窓なし常駐の省電力抑制解除。配布既定ONの採用根拠は10/30の提案と11/1の指示。
    'live_disable_power_throttling': True,
    # 追補で24w2との差を許可された診断記録。その他のDも既存OFFを維持する。
    'live_stage_timing': False, 'live_verify_reason_log': False,
    # 2026-11-03 user 承認 (11-02 PC1 案、W64 の直しと連鎖の予測の直し)。0.6.0 で配布。低水準の比較用既定はOFFを維持。
    'live_verify_entry_fast_retry': True,
    'live_chain_forecast_recheck': True,
    'live_chain_end_formula': True,
}
RELEASE_LIVE_MODEL_PATHS: dict[str, str] = {
    'b3_model': 'models/b3_c2_ens/ensemble_ref3_logit.json',
    'b3_timing_table': 'models/b3/timing_table.json',
    'live_24h_model': 'models/b3_ex_L200H_retR/ensemble.json',
}

# 2026-11-06: 0.6.2で1/22回の生存中停止。原因未確定のため候補の見張りは既定OFF。
DISTRIBUTION_CHILD_WATCHDOG = False

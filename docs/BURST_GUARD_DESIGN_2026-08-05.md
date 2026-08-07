# バースト焼き付き対策 設計仕様 (2026-08-05)

> 対象読者: 実装エージェント (コーダ) / テスターエージェント。
> 前提知識: `docs/CYCLE_FINDINGS.md`、memory `project_effect_gate_v1_failure_2026-08-03`、
> `data/verify/effect_detector_calibration_v3_2026-08-04/calibration_report_v3.md`、
> `data/verify/error_onset_sheet_2026-08-04/index_refined.md`。

## 0. 背景 (確定済み証拠の要約)

1. user目視確定: 満杯盤面帯の誤り93セルは全て「相手のお邪魔送付バースト演出」起因 (5色半透明レイヤーの重畳)。
2. 案B (4条件ANDゲート、`enable_effect_gate` + `enable_effect_visual_gate`) は93セルに改善ゼロ。敗因3点:
   - (a) 窓トリガーの `ChainEvent` (`self._active_chain_2p` 経由) が大連鎖を丸ごと見逃す (c18: お邪魔131個級で検出0件)。リンク間で0.1秒明滅し、谷の内側で誤値確定 (c5)。
   - (b) `EFFECT_PERSIST_SEC=0.4` の「持続確認」が逆転する: バーストが0.4秒超続くと誤値が「安定」として設計通り採用される (c12 実測0.467秒)。
   - (c) 守備範囲 `EFFECT_GATE_TOP_ROWS={1,2,3}` 限定、着弾煙は全行に及ぶ (c19: row7-12, opp_supply_ojama=79 で範囲外)。
3. userの難所: 「バーストの光の間にもプレイヤーはぷよを置ける」→ 全面凍結は正当な設置反映を遅らせる。
4. user承認済み方向: 「演出が終わってから書き込む」方式。unmix (半透明レイヤー除去) は別エージェントが実現可能性を並行検証済み (`scripts/_probe_burst_unmix_2026-08-05.py`、結論=代替不可・補完候補)、本設計はこれを前提にしない。将来の差し替え点としてのみ言及する。

## 1. アーキテクチャ上の最重要発見 (実装前に必ず理解すること)

### 1.1 「設置追跡は継続」は新規実装が不要 — 既存の経路分離を壊さないことが答え

`src/board_state_machine.py` の `_apply_transition()` (994-1056行) は NON-STABLE→STABLE 遷移時に
`_merge_diff_only()` (1034行) を呼んで新規設置を confirmed_board に書き込む。この呼び出しは
`effect_gate_active_rows` / `effect_gate_persist_sec` などのエフェクトゲート系引数を **一切受け取らない**。

一方、本ドキュメントが再設計する対象は `_collect_recovery_candidates()` (1386-1501行) — これは
STABLE 状態に留まったまま confirmed_board と CNN/HSV合意値の差分を「復旧」するための、
別経路 (STABLE内ドリフト補正) である。

**結論**: 通常の設置 (TSUMO_FALL→STABLE 遷移で検知される設置) は、バーストWindowの影響を
最初から受けない。実装エージェントは **この2経路を混同させる変更を絶対にしてはならない**
(= `_merge_diff_only` 呼び出しに `effect_gate_*` 系引数を新たに混ぜ込むことを明示的に禁止する)。

**残存リスク (バックテストで必ず確認)**: state遷移検知が失敗した設置 (何らかの理由で
TSUMO_FALL 遷移が検知されず、STABLE のまま新規ぷよが出現するケース) は fallback として
`_collect_recovery_candidates` の方向1 (空→色、重力整合チェック付き) 経由でも反映される。
この fallback 経路がバーストWindow中に発生した場合のみ、新設計が遅延の影響を与える。
バックテスト計画 §7.2 でこのケースの発生頻度と遅延を必ず計測する。

### 1.2 「バースト」と「煙」は別の物理イベントで、既存の2つの時間窓にそれぞれ対応する

`calibration_report_v3.md` §2 の row 分布表 (layer別) と §4 の out_of_scope 表を突き合わせると:

- `burst` レイヤー (row1-3 に集中、`bright_ratio_max` AUC=0.811 で高精度検出可能)
  = 相手の連鎖中に発生する「予告おじゃま送付エフェクト」(1リンク約0.2秒)。
  既存コードの `opponent_chain_active` 窓 (`chain_ev_2p is not None`、
  `src/recognition_pipeline.py:3595` 相当) に対応する。
- `smoke` レイヤー (全12行にほぼ一様分布、固定窓不成立、視覚検出困難) の out_of_scope note は
  全て「おじゃま実増加±1秒」= 自分の盤面におじゃまが着弾する瞬間。既存コードの
  `_effect_gate_ojama_until_Xp` 窓 (score差分ベースで既に信頼できる時刻確定済み、
  `EFFECT_GATE_OJAMA_EXIT_WINDOW_SEC=1.0` 秒、`src/recognition_pipeline.py:115`) に対応する。

**設計上の単純化**: 2層のスコープ切替は「視覚強度に応じた動的な行範囲判定」ではなく、
「窓の出自 (種別) による静的な行範囲の割り当て」でよい。これにより n=1 (c19のみ) の
severity閾値を発明する必要がなくなる (過学習回避)。

- 相手連鎖窓 (`opponent_chain_active` 起源) → 行スコープ = `EFFECT_GATE_TOP_ROWS` (row1-3) を維持。
- 自分お邪魔着弾窓 (`_effect_gate_ojama_until_Xp` 起源) → 行スコープ = 全行 (`range(BOARD_ROWS)`)。
  根拠: (i) 既存コードのコメント (`src/recognition_pipeline.py:111-113`) が「着弾列近傍を
  特定の列に絞らず全列対象」とする設計判断を既に列軸で行っている (floor(N/6)+端数ランダム、
  `reference_ojama_landing_pattern`)。行軸も同様に「おじゃまは6列全域・盤面高さ全域に降る」
  という物理と整合させるのが妥当。(ii) `smoke` レイヤーの row 分布が既にこれを支持する。
  (iii) c19 (opp_supply_ojama=79, landing_confirmed) の row7-12 汚染はこの窓で説明がつく。

### 1.3 persist逆転の構造的な殺し方: 「持続確認の対象」を反転する

現行 `_update_effect_gate_hold()` (1314-1343行) は「候補色が0.4秒持続したら確定」という
ロジック。これは Window ON 中 (= バースト表示中) でも時間が経過すれば確定してしまう
(issue b)。

修正方針: Window ON の間は **無条件凍結** (frame count も persist timer も一切進めない)。
Window が閾値未満に落ちて `QUIESCENCE_MIN_SEC` 継続して初めて、対象セルは通常の
`STABLE_RECOVERY_MIN_FRAMES` (frame count方式) にゼロから再エントリーする。
「持続を要求する対象」が「バーストがある」ではなく「バーストが無い (静穏)」に完全に
反転しているため、バーストがどれだけ長く続いても誤確定は原理的に発生しない。

## 2. Layer 1: 視覚トリガーの Schmitt hysteresis 化

### 2.1 score関数の抽出 (`src/effect_glow_detector.py`)

既存 `is_effect_glow_active()` (54-83行) は bool のみを返す。内部の `max_ratio` 計算を
独立関数として抽出し、`is_effect_glow_active` は薄いラッパーとして残す (bit-identical 維持)。

```python
def compute_effect_glow_score(
    frame_bgr: np.ndarray,
    region: "BoardRegion",
    rows: "frozenset[int]" = EFFECT_GATE_TOP_ROWS,
) -> float:
    """指定行帯のセル bright_ratio 最大値を返す (stateless純関数)。

    is_effect_glow_active の閾値判定ロジックをスコア計算部分から分離した
    もの (2026-08-05 バーストガード再設計)。ロジック・数値は完全に同一
    (bright_ratio_max 較正済み方式、calibration_report_v3.md §3)。
    """
    max_ratio = 0.0
    for row in rows:
        for col in range(BOARD_COLS):
            x1, y1, x2, y2 = region.cell_sample_rect(row, col)
            patch = frame_bgr[y1:y2, x1:x2]
            ratio = compute_cell_bright_ratio(patch)
            if ratio > max_ratio:
                max_ratio = ratio
    return max_ratio


def is_effect_glow_active(
    frame_bgr: np.ndarray,
    region: "BoardRegion",
    rows: "frozenset[int]" = EFFECT_GATE_TOP_ROWS,
    threshold: float = EFFECT_BRIGHT_RATIO_MAX_THRESHOLD,
) -> bool:
    """(既存 docstring 維持。実装を compute_effect_glow_score 呼び出しに変更するのみ、
    戻り値・数値は完全に bit-identical)"""
    return compute_effect_glow_score(frame_bgr, region, rows) > threshold
```

### 2.2 Schmitt trigger 状態遷移関数 (新規、`src/board_state_machine.py` に追加)

**状態の格納場所**: `RecognitionPipeline` インスタンス属性 (1P/2P別、既存の
`self._effect_gate_ojama_until_1p/2p` と同じパターン)。理由: この信号の入力
(`frame_bgr`, `region`) は `RecognitionPipeline._step_side` が既に保持しており、
`BoardStateMachine` は相手sideの情報 (frame_bgr等) を持たない設計 (既存の
`opponent_chain_active` も同じ理由で外部注入)。計算関数自体は stateless
(前状態を引数で受け取り新状態を返す純関数) にし、「観測指標はstateless」原則を守る。

```python
def _update_burst_visual_gate(
    is_open: bool,
    opened_at: "float | None",
    quiet_since: "float | None",
    score: float,
    time_sec: float,
    *,
    open_threshold: float,
    close_threshold: float,
    min_window_sec: float,
    max_window_sec: float,
    quiescence_min_sec: float,
    force_close: bool = False,
) -> "tuple[bool, float | None, float | None]":
    """バースト視覚検出の Schmitt trigger 1frame更新 (stateless純関数)。

    Window ON 中は無条件凍結 (呼び出し側が is_open を effect_gate_active な
    行の凍結条件として使う)。issue (b) の persist逆転を構造的に排除するため、
    「確定に必要な持続」ではなく「解除に必要な静穏」を計測する設計にする
    (2026-08-05 バーストガード再設計 §1.3)。

    Args:
        is_open: 直前frameのWindow状態。
        opened_at: Window が開いた time_sec (open中のみ値を持つ)。
        quiet_since: score が close_threshold 未満に落ちた最初の time_sec
            (close中に再びopen_threshold以上に戻ったらNoneにリセット)。
        score: 今frameの視覚スコア (compute_effect_glow_score の戻り値)。
        time_sec: 今frameの時刻。
        open_threshold: Window を開く閾値 (score >= で即時open)。
        close_threshold: 静穏判定の閾値 (score < が続くことを要求、
            open_threshold 以下の値を推奨 = ヒステリシス帯を作る)。
        min_window_sec: 一度開いたら最低この秒数は維持する
            (1リンクの演出持続時間 ≒0.2秒に対応、単発frameでの開閉振動防止)。
        max_window_sec: 安全弁。この秒数を超えたら score に関係なく強制close
            (視覚検出が誤って張り付いた場合の永久凍結防止)。
        quiescence_min_sec: close確定に必要な連続静穏秒数
            (リンク間flicker gap ≒0.1秒 の1回だけでは閉じないマージンを持たせる)。
        force_close: True の場合、他条件を無視して即時close
            (own_chain_active / all_clear_pending 等の外部安全条件)。

    Returns:
        (new_is_open, new_opened_at, new_quiet_since)
    """
    if force_close:
        return False, None, None

    if not is_open:
        if score >= open_threshold:
            return True, time_sec, None
        return False, None, None

    # is_open == True
    if opened_at is not None and (time_sec - opened_at) >= max_window_sec:
        return False, None, None  # 安全弁: 強制close

    if score < close_threshold:
        _quiet_since = quiet_since if quiet_since is not None else time_sec
        elapsed_open = time_sec - opened_at if opened_at is not None else 0.0
        quiescent = time_sec - _quiet_since
        if elapsed_open >= min_window_sec and quiescent >= quiescence_min_sec:
            return False, None, None
        return True, opened_at, _quiet_since
    # score >= close_threshold: まだバースト中、静穏タイマーをリセット
    return True, opened_at, None
```

`frame_bgr is None` (画像取得不能フレーム) の場合は本関数を呼ばず、直前状態をそのまま
維持する (安全弁A の既存方針を踏襲。無情報を「静穏」と誤認しない)。ただし
`max_window_sec` の安全弁チェックだけは `frame_bgr is None` でも time_sec 経過で
効かせる (呼び出し側で別途チェックするか、score を直前値のまま関数を呼んで良い)。

### 2.3 呼び出し箇所 (`src/recognition_pipeline.py`)

`_step_side()` (4901行) 内、既存の `_effect_gate_window_active` 計算部分
(5012-5040行) を新フラグ `enable_burst_guard_v2` (仮称) で分岐する。既存
`enable_effect_gate`/`enable_effect_visual_gate` の経路は完全に無改変で残す。

```python
if self._enable_burst_guard_v2:
    _score = 0.0
    if frame_bgr is not None:
        _region = DEFAULT_P1_REGION if side == "1P" else DEFAULT_P2_REGION
        _score = compute_effect_glow_score(frame_bgr, _region, EFFECT_GATE_TOP_ROWS)
    _prev_open, _prev_opened_at, _prev_quiet = (
        (self._burst_gate_open_1p, self._burst_gate_opened_at_1p, self._burst_gate_quiet_since_1p)
        if side == "1P" else
        (self._burst_gate_open_2p, self._burst_gate_opened_at_2p, self._burst_gate_quiet_since_2p)
    )
    _new_open, _new_opened_at, _new_quiet = _update_burst_visual_gate(
        _prev_open, _prev_opened_at, _prev_quiet, _score, time_sec,
        open_threshold=BURST_GATE_OPEN_THRESHOLD,
        close_threshold=BURST_GATE_CLOSE_THRESHOLD,
        min_window_sec=BURST_GATE_MIN_WINDOW_SEC,
        max_window_sec=BURST_GATE_MAX_WINDOW_SEC,
        quiescence_min_sec=BURST_GATE_QUIESCENCE_MIN_SEC,
        force_close=(own_chain_active or all_clear_pending_for_side),
    )
    if side == "1P":
        self._burst_gate_open_1p, self._burst_gate_opened_at_1p, self._burst_gate_quiet_since_1p = _new_open, _new_opened_at, _new_quiet
    else:
        self._burst_gate_open_2p, self._burst_gate_opened_at_2p, self._burst_gate_quiet_since_2p = _new_open, _new_opened_at, _new_quiet
    _effect_gate_window_active = _new_open
    _effect_gate_scope = EFFECT_GATE_TOP_ROWS  # Stage1: 相手連鎖窓相当のスコープのみ
elif self._enable_effect_gate:
    # (既存 案B 経路、完全無改変)
    ...
```

`chain_event`/`opponent_chain_active` は **トリガーの OR 条件からは除外** (=「捨てる」)。
Stage 1 では補助情報としても使わない (要求 §1「ChainEvent 依存を捨てるか補助に格下げ」
の「捨てる」を採用。理由: 較正データが `bright_ratio_max` 単独で AUC=0.811・
zero_fp動作点0.97を既に持っており、ChainEventを混ぜるAND/OR条件を追加すると
「大連鎖見逃し」バグ (issue a) をそのまま引き継いでしまうため)。
`own_chain_active`/`all_clear_pending` は `force_close` 条件として維持する
(較正レポート §5 で確認済みの唯一の実効ゲート、テロップ誤発火7件を防ぐ)。

## 3. Window ON 中の凍結ロジック (`src/board_state_machine.py`)

`_apply_stable_recovery_gate()` (1538行) / `_recovery_or_effect_gate_pass()`
(1345-1383行) に新パラメータ `effect_gate_hard_freeze: bool = False` を追加する
(既存 `enable_effect_gate` の sibling flag、既存動作は変更しない)。

```python
def _recovery_or_effect_gate_pass(
    ctx, cell, confirmed_v, agreed_v, recovery_counters,
    min_frames, add_min_frames, effect_gate_active_rows,
    effect_gate_persist_sec,
    effect_gate_hard_freeze: bool = False,  # 新規、既定False=既存動作維持
) -> bool:
    r, c = cell
    is_gated = (
        effect_gate_active_rows is not None and r in effect_gate_active_rows
    )
    if is_gated:
        recovery_counters.pop(cell, None)
        if effect_gate_hard_freeze:
            # 2026-08-05 バーストガード再設計: 持続確認を一切行わず、
            # Window ON の間は無条件で発火させない (issue b の構造的排除)。
            ctx.effect_gate_hold.pop(cell, None)
            return False
        return _update_effect_gate_hold(
            ctx.effect_gate_hold, cell, agreed_v,
            ctx.time_sec, effect_gate_persist_sec,
        )
    ...  # 既存の frame count 経路 (無改変)
```

`effect_gate_hard_freeze=True` の場合、`effect_gate_hold` dict は使われない
(dead pathになるが 案B 利用者向けに削除しない、backwards compat)。Window が
close した frame から、対象セルは `recovery_counters` が既に pop されているため
自動的に0から数え直しになる (既存コードの `_reset_counters`/`pop` 呼び出しが
既にこの「ゼロから再エントリー」を保証している。追加実装不要、既存の副産物)。

## 4. 定数表 (全て物理量・較正データ根拠、シーン逆算禁止)

| 定数 | Stage | 値 | 根拠 |
|---|---|---|---|
| `BURST_GATE_OPEN_THRESHOLD` | 1 | `0.97` | 既存 `EFFECT_BRIGHT_RATIO_MAX_THRESHOLD` を再利用。v1+v3統合136枚較正の zero_fp 動作点 (AUC=0.811, n_pos=17/n_neg=87)。窓トリガー用途はFPコストが低いため理論上下げられるが、今夜のデータにはこの1点しかROC上に無く、根拠なく下げるのは禁止 (Stage2で新規較正)。 |
| `BURST_GATE_CLOSE_THRESHOLD` | 1 | `0.97` (Stage1は`OPEN`と同値) | 値ベースのヒステリシス幅は現時点で較正データが無い。Stage1は時間ベースのヒステリシス (`QUIESCENCE_MIN_SEC`) のみに依拠し、値の二重閾値は同値にして安全側に倒す。Stage2でROC全体を計算し `CLOSE < OPEN` の真のヒステリシス帯を較正する。 |
| `BURST_GATE_MIN_WINDOW_SEC` | 1 | `0.2` | 1リンクの演出持続時間の実測記述 (`effect_glow_detector.py` docstring「約0.2秒」、`project_full_board_error_taxonomy_2026-08-02`)。単発frameでのopen直後close振動を防ぐ下限。 |
| `BURST_GATE_QUIESCENCE_MIN_SEC` | 1 | `0.25` | リンク間flicker gap実測 ≒0.1秒 (c5, issue 2a) の2.5倍マージン。1回のリンク間隙だけでは閉じない (=誤って途中で凍結解除しない) ことを保証する安全マージン。**c5 は現象の発見元であり、閾値をc5に一致させるための逆算は禁止** (`feedback_overfitting_awareness_2026-08-04`)。バックテストではc5含む全域で汎化を確認する。 |
| `BURST_GATE_MAX_WINDOW_SEC` | 1 | `30.0` | 安全弁 (滅多に発火しない前提)。8連鎖実測14.5秒 (`project_chain_count_both_untrustworthy_2026-07-30`) の約2倍マージン、より大きな連鎖 (12+連鎖) にも対応させる。誤って永久凍結するリスク (=recognitionの完全停止) の方が、多少長い保留より重大であるため寛容側に倒す。 |
| `EFFECT_GATE_TOP_ROWS` | 1 (既存) | `frozenset({1,2,3})` | Stage1では相手連鎖窓のスコープとして無改変で流用。 |
| Stage2: 自分お邪魔着弾窓の行スコープ | 2 | `frozenset(range(BOARD_ROWS))` (全行) | §1.2 の物理的対応関係 (お邪魔は6列全域・盤面高さ全域に降る) + `smoke` レイヤーの row 分布較正結果。既存 `EFFECT_GATE_OJAMA_EXIT_WINDOW_SEC=1.0`秒窓 (`src/recognition_pipeline.py:115`) と組み合わせて使う (窓の長さ自体はStage1で変更しない)。 |

## 5. StateContext / DetectorSignals / コンストラクタ変更点 (backward compat 確認)

- `StateContext` (`src/board_state_machine.py:209`): 変更不要。Window状態は
  `RecognitionPipeline` 側インスタンス属性で管理するため (§2.2 の理由)、
  StateContext への新規フィールド追加は不要。
- `DetectorSignals.effect_gate_window_active` (357行): **既存フィールドを再利用**。
  新規フィールド追加はしない (Stage1のセマンティクス「Windowが今アクティブか」は
  ChainEvent方式と視覚方式で変わらないため、同じフィールドに異なる計算方法の
  結果を代入するだけで済む。ただし両方式は `_step_side` 内で `enable_burst_guard_v2`
  により排他的に分岐し、混ざらないようにする)。
- `_apply_stable_recovery_gate()` / `_collect_recovery_candidates()` /
  `_recovery_or_effect_gate_pass()`: 新パラメータ `effect_gate_hard_freeze: bool = False`
  を追加 (既定False、既存呼び出し元は無改修でbit-identical)。
- `BoardStateMachine.__init__()`: 新パラメータ `effect_gate_hard_freeze: bool = False`
  を追加し `_apply_stable_recovery_gate` 呼び出しに配線 (既存 `effect_gate_persist_sec`
  と同じ配線パターン、707-755行付近)。
- `RecognitionPipeline.__init__()`: 新パラメータ `enable_burst_guard_v2: bool = False`
  を追加。新規インスタンス属性6個: `_burst_gate_open_1p/2p: bool = False`,
  `_burst_gate_opened_at_1p/2p: float | None = None`,
  `_burst_gate_quiet_since_1p/2p: float | None = None`。試合境界リセット処理
  (`force_match_boundary_reset` 相当の箇所、既存の `_effect_gate_ojama_until_1p/2p`
  リセットと同じ場所) にこれら6属性のリセットも追加すること (前試合の burst 状態が
  次試合に残留するバグを防ぐ、`project_match_boundary_residue_leak_2026-07-25` と
  同種の罠に注意)。
- `enable_burst_guard_v2=True` かつ `enable_effect_gate=False` の組み合わせは
  no-op として警告ログを出す (安全側、意図しない設定ミスの早期発見)。

## 6. Non-goals (今回やらないこと)

- ChainEvent (`self._active_chain_2p`) が大連鎖を見逃す根本原因の修正。本設計は
  トリガーをChainEventから切り離すことで問題を回避するのみで、ChainEvent自体は
  他の消費者 (連鎖式検知、打ち合い計測器等) のために別途直す必要があるが対象外。
- unmix (半透明レイヤー除去、`scripts/_probe_burst_unmix_2026-08-05.py`) の実装。
  本設計の `_update_burst_visual_gate` が返す bool 信号は「凍結するか」以外の
  用途にも使える設計になっている (= Window信号と「Window中に何をするか」を分離済み)。
  unmix が実現可能と判明したら、`effect_gate_hard_freeze` 分岐の代わりに
  「unmix補正した値を書き込む」分岐を追加するだけで済む拡張点として残す。
- Stage2の「自分お邪魔着弾窓の全行スコープ」実装そのもの (設計のみ本書に記載、
  実装はStage2として別途着手)。
- `BURST_GATE_CLOSE_THRESHOLD` の真のヒステリシス値較正 (新規ROC計算が必要)。

## 7. バックテスト計画

### 7.1 93セル再測定
`scripts/measure_effect_gate_c_2026-08-04.py` と同じ方式 (OFF基準=93セル一致を
前提条件として確認してから効果集計、`feedback_overfitting_awareness_2026-08-04`
準拠) で、比較対象を3系統に拡張:
- OFF (ゲート無し、真のbaseline)
- 案B (`enable_effect_gate` + `enable_effect_visual_gate`、既存、参考=ゼロ改善実績)
- 新設計 (`enable_burst_guard_v2`、Stage1構成)

layer別 (burst row1-3 / smoke row4-12) に分けて集計すること (層別必須、
`feedback_stratify_before_pooling_2026-07-29`)。smoke分は Stage1 (row1-3スコープ
のみ) では改善しない見込みを事前に明記し、「Stage1は burst分のみ改善確認・
smoke分はStage2待ち」を正直に報告する。

### 7.2 反映遅延分布 (1P/2P別)
`feedback_placement_reflection_8frames_2026-07-25` の既存手法で、設置→confirmed色
確定までのフレーム遅延を計測。以下を必ず分けて報告:
- 通常設置 (バーストWindow非活性中の設置): OFF/新設計で不変であることを確認
  (§1.1 の「経路分離は壊れていない」ことの直接的な検証)。
- バーストWindow活性中に発生した設置 (§1.1 「残存リスク」の fallback 経路が
  発火したケース): 発生頻度と遅延分布を報告。8フレーム基準からの超過があれば
  超過量とその原因 (fallback経路か否か) を明示する。

### 7.3 全域無悪化
`docs/CYCLE_FINDINGS.md` §4.2-quater/quinquies の I1 (`per_col_unknown_rate` /
`non_stable_consecutive_frames` / `per_col_midgame_empty_rate`)、C1
(`avg_puyo_count` baseline比 >=0.85)、D1 (`postprocess_corruption` rate <0.1%)
を全評価対象動画で確認。`feedback_overfitting_awareness_2026-08-04` の
「5シーン合格+全域無悪化で初めて合格」を厳守し、以下を最低限含める:
- viz目視: `error_onset_sheet_2026-08-04` の12動画 (c5/c11/c12/c13/c15/c18/c19/c21/c23/c29/c31/c36)
- viz目視: バースト無関係の通常プレイシーンを最低2-3本 (凍結ロジックの
  副作用が通常プレイに漏れ出していないかの確認、対象は未使用動画から選定
  `feedback_review_video_full_match.md` 準拠)
- 数値・viz双方の承認をuserから得るまでPR化しない (`feedback_viz_eval_required.md`)

## 8. 段階分割・工数見積もり

### Stage 0 (今夜、~0.5-1h): score関数の抽出
`src/effect_glow_detector.py` に `compute_effect_glow_score()` を追加、
`is_effect_glow_active()` を薄いラッパーに変更 (bit-identical維持)。
既存テスト全パス確認のみで完了。

### Stage 1 (今夜〜明日午前、~5-6h): 視覚トリガー + ハード凍結 (最小構成)
1. `src/board_state_machine.py`: `_update_burst_visual_gate()` 追加、
   `_recovery_or_effect_gate_pass()` に `effect_gate_hard_freeze` パラメータ追加
   (~1.5h実装 + ~1.5hテスト: open/維持/close/max_window強制close/
   frame_bgr None時の状態維持、の遷移表を網羅するunit test)。
2. `src/recognition_pipeline.py`: `enable_burst_guard_v2` フラグ + 6インスタンス
   属性 + `_step_side` 内の分岐 + 試合境界リセット処理への追加
   (~1.5h実装 + ~1hテスト)。
3. 既存フラグの backwards compat 確認テスト (新フラグ全てFalse時に既存挙動と
   bit-identical であることを assert するテスト、既存の
   `test_ojama_dropout_fix_flags_explicit_false_restores_legacy` と同パターン、~0.5h)。
4. §7 のバックテスト実行 (計測スクリプト自体は既存流用のため実装コストは低いが、
   全域実行の待ち時間が入るため別枠で見積もる、~1-2h待ち時間)。

**Stage1完了条件**: 93セットの burst layer 分 (row1-3) が案B比で有意に改善、
smoke layer分は不変で正直に報告、全域I1/C1/D1ゲート通過、viz目視レビューで
「設置反映が遅れていない」ことをuserが確認。

### Stage 2 (明日以降、~1日): 自分お邪魔着弾窓の全行スコープ + 較正強化
1. `BURST_GATE_CLOSE_THRESHOLD` の真のヒステリシス値較正: 新規スクリプトで
   `labeled_cell_features_v3.csv` から完全ROC曲線を計算し、FPR予算 (窓トリガー
   用途向け、要user/アーキ合意で予算%を決定) に基づく点を選定 (~2h)。
2. `_effect_gate_ojama_until_Xp` 窓の行スコープを全行にする実装
   (`_apply_stable_recovery_gate` 内の `_effect_gate_rows` 計算を「窓の出自」で
   分岐、§1.2 参照) (~2h実装 + ~1hテスト)。
3. c19 (opp_supply_ojama=79) を含む smoke layer 分のバックテスト、全域無悪化確認
   (~3-4h、viz含む)。

### Stage 3 (研究軌道、時期未定): unmix への差し替え
実現可能性検証済み (復元73.1%だがburst色推定が脆弱、負例誤変換16.2%、
結論=代替不可・補完候補)。将来必要になれば `effect_gate_hard_freeze` 分岐に
「unmix補正値を書き込む」経路を追加する形で差し替え可能な設計になっている
(§6 Non-goals 参照)。

## 9. 参照ファイル一覧 (実装エージェント向け)

- `src/effect_glow_detector.py` (score関数抽出対象)
- `src/board_state_machine.py`: `EFFECT_GATE_TOP_ROWS`(169行), `EFFECT_PERSIST_SEC`(170行),
  `_update_effect_gate_hold`(1314-1343行), `_recovery_or_effect_gate_pass`(1345-1383行),
  `_collect_recovery_candidates`(1386-1501行), `_apply_stable_recovery_gate`(1538-1639行),
  `_apply_transition`(994-1056行, 変更禁止領域として理解すること),
  `StateContext`(209-286行), `DetectorSignals`(294-357行)
- `src/recognition_pipeline.py`: `EFFECT_GATE_OJAMA_EXIT_WINDOW_SEC`(115行),
  `_compute_effect_gate_window_active`(6714-6749行, Stage1では新設計と並存させる),
  `_step_side`(4901行〜), Window計算箇所(5012-5040行), `_effect_gate_ojama_until_1p/2p`
  更新箇所(3649-3660行), `opponent_chain_active`/`own_chain_active` 受け渡し(3582-3613行)
- `data/verify/effect_detector_calibration_v3_2026-08-04/calibration_report_v3.md`
- `data/verify/error_onset_sheet_2026-08-04/index_refined.md`
- `scripts/measure_effect_gate_c_2026-08-04.py` (バックテスト流用元)

## 10. Stage1.5: 遷移merge時の物理的期待値フィルタ (2026-08-05 アーキ追補)

### 10.0 背景 (計装で確定した事実)
`scripts/_verify_burst_write_path_2026-08-05.py` により、c18の10セル全焼き付きは
`_apply_transition()` の NON-STABLE→STABLE 遷移時 `_merge_diff_only` 呼び出し
(1043-1054行) を素通りしたバースト誤読と確定した。遷移フレームは `hsv=None`
のため `_apply_stable_recovery_gate` は実行されず、Stage1のハード凍結
(`effect_gate_hard_freeze`) はこの経路を全くカバーしない。自分の設置完了
(TSUMO_FALL→STABLE) と相手バースト、相手のおじゃま着弾 (OJAMA_FALL→STABLE) と
相手バーストは、おじゃまが相手連鎖完了後に降る物理 (`reference_ojama_landing_gated_by_placement_2026-07-29`)
により構造的に同時発生するため、この経路が正門である必然性がある。

### 10.1 判定法: 「新規設置セル推論」ではなく「from_state別・物理的期待値クラス」

新規設置セルをnext_pair等から個別推論する方式は採用しない。設置ペア対応付けの
信頼性は過去に否定されている (キュー対応付け正解無し57%、
`project_color_flicker_p2_root_cause_2026-07-25`)。代わりに、from_stateごとに
「この遷移で物理的に説明可能な新規値クラス」を静的に定義し、それ以外の diff は
一律 `COLOR_UNKNOWN` に差し替える。

| from_state | 物理的に説明可能な diff | 根拠 |
|---|---|---|
| `TSUMO_FALL` | `base_v == COLOR_EMPTY` かつ `cnn_v ∈ {1,2,3,4,5}` | ツモ設置は空セルへの色puyo出現のみ。既存puyoの色変化・9出現は説明不可 |
| `OJAMA_FALL` | `base_v == COLOR_EMPTY` かつ `cnn_v == COLOR_OJAMA(9)` | おじゃまは空セルにのみ落下 (`reference_ojama_landing_pattern`)。既存puyoの上書き・9以外の新規値はバースト誤読の署名 |
| その他 (`CHAIN`/`GRAVITY_SETTLE`/`EFFECT`) | フィルタ対象外 (no-op) | §10.2 参照 |

`base_v != COLOR_EMPTY` の diff (既存puyoの値が変わるケース) は上記2 from_state
では原理的に非説明的なため、cnn_vの値を問わず全て棄却する。これが c18 の
(2,4,9,5) — 既存の紫puyoがバーストで9に化けた — を正しく捕捉する。

### 10.2 スコープ限定: GRAVITY_SETTLE/CHAIN/EFFECT→STABLE は明示的に除外

1137-1139行のコメントが示す通り、GRAVITY_SETTLE→STABLE は「連鎖後は全cellを
新規STABLEで直接評価」する設計 (F guard不発、意図的に大量差分を無条件通過)。
このタイミングは own_chain_active / all_clear_pending による force_close
(§2.3) で Window が既に閉じているのがほぼ全ケースであり、フィルタを適用する
根拠データが無い。連鎖後の正当な大量色変化・重力再配置をUNKNOWN化して壊す
リスクの方が、想定外の稀なケースを取りこぼすリスクより重大。よって
`_TRANSITION_MERGE_GUARD_SCOPE` に含めない (= 対象外 from_state は無条件no-op)。
Stage1.5汎化バックテストで実測上問題が見つかった場合のみ、別Stageとして
根拠データを揃えた上で再検討する。

### 10.3 差し替え値: COLOR_UNKNOWN (baseline値の直接書き込みではない)

`_merge_diff_only` 607-608行 (D guard, 2026-05-11) は既に
`cnn_v == COLOR_UNKNOWN` を「baseline維持」として扱う。この既存の実証済み分岐
を再利用することで `_merge_diff_only` 自体への変更を一切要さない。baseline値を
new_cnn側に直接書き込む代替は数学的に同じ結果になるが実装上の理由がなく採用しない。

**baseline is None (初回STABLE確定) の場合は本フィルタを完全skip** し
new_cnn をそのまま返す。理由: `_build_initial_confirmed_board` は baseline
概念を持たず、UNKNOWNがそのまま初回confirmed_boardに漏れ出す (試合開始直後の
極端な edge case、burst windowが同時に開く可能性は実質ゼロだが、フィルタ関数
の入力契約として明示的にガードする)。

### 10.4 実装 (`src/board_state_machine.py`)

`_TRANSITION_MERGE_GUARD_SCOPE` 定数 (TSUMO_FALL→{1..5} / OJAMA_FALL→{9}) と
`_filter_transition_new_cnn_for_burst_guard(baseline, new_cnn, from_state)` 純関数を追加。
対象外 from_state / baseline is None は new_cnn をそのまま返す (コピーもしない恒等)。
説明不可能な diff は COLOR_UNKNOWN に差し替え (D guardでbaseline維持)。

呼び出しは `_apply_transition()` の `_merge_diff_only` 呼び出し直前:

```python
new_cnn_for_merge = signals.cnn_board
if self._enable_transition_merge_guard and signals.effect_gate_window_active:
    new_cnn_for_merge = _filter_transition_new_cnn_for_burst_guard(
        self._ctx.confirmed_board, signals.cnn_board, self._ctx.state,
    )
```

`self._ctx.state` は再代入前に読めば常に from_state を指す。新パラメータ
`enable_transition_merge_guard: bool = False` (default OFF、bit-identical)。
`enable_transition_merge_guard=True` かつ `enable_burst_guard_v2=False` は警告ログ。

### 10.5 棄却セルの事後記録: 記録なし (既存機構+使い捨て診断で足りる)

棄却されたセルは baseline値のまま STABLE に留まり、Window close後に Stage1 の
フレームカウント方式復旧が正常に拾う。永続的な棄却ログAPIは追加しない。

### 10.6 バックテスト: 閾値0.954との同時検証は同一実行・factorial報告で許可

Stage1.5 (`enable_transition_merge_guard`) と `BURST_GATE_OPEN_THRESHOLD`
(0.97現行 / 0.954提案) は直交する軸であり、2×2 factorial (off/off, on/off,
off/on, on/on) として計測してよい。§7.1 の93セル集計・§7.3 の全域ゲートは
4象限それぞれで独立に報告 (プール相殺の禁止、feedback_stratify_before_pooling)。

### 10.7 row0隠し段副作用: 別立て (Stage1.5には含めない)

「row1-3凍結→隠し段推論が揺れてrow0に新規誤り」はStage1のper-frame凍結の
副作用であり、Stage1.5 (遷移frameで1回だけのmerge時フィルタ) とは機構が別。
別課題としてStage1のバックテスト結果 (§7.3) の中で扱う。

## 11. Stage1.5b: 隠し段推論の信頼性ゲート (2026-08-05 アーキ追補)

### 11.0 破損機序 (コード確認で確定)
row0 の持続的新規誤りは EFFECT_GATE_TOP_ROWS 経由ではない。row0 が書かれる唯一の
経路は recognition_pipeline._step_side 内の infer_hidden_row 呼び出し (L5462-5487、
TSUMO_FALL→STABLE 着地確定時のみ、p>=0.95 で確定書き込み)。窓open中の row1-3 凍結で
prev_confirmed が stale になり、着地の新規セル数の再カウントが狂う (真値2→1や3+)。
n==1 分岐は候補列1つで p=1.0 になり「確信度100%の誤色」が row0 に書かれる。
自己回復は次の n==2 正常着地 (Case A の全隠し段リセット) 待ちのため長引く。

### 11.1 対策 (b修正版): ドア2側の推論を信頼性ゲートで停止
新フラグ `enable_hidden_row_burst_guard: bool = False` (default OFF、bit-identical)。
ゲート条件: `not window_active AND (time_sec - 最後にwindowがopenだった時刻) >= HIDDEN_ROW_TRUST_COOLDOWN_SEC`
- 不成立時は infer_hidden_row 呼び出し自体をスキップ (row0 は直前値キャリーオーバー、
  既存の STABLE中フォールバック経路に自然委譲)
- `_last_burst_open_time_1p/2p` (float、-inf初期、windowActive全フレームで更新) を新設
- クールダウンが必要な理由: close直後は row1-3 の backlog 追いつきに数フレームかかり、
  その間の着地でも同じ誤カウントが再発するため
- HIDDEN_ROW_TRUST_COOLDOWN_SEC は定数化、シーン逆算禁止・全域バックテストで較正

### 11.2 検証法 (userレビュー投入の前提条件)
1. row0 の新規誤り (37件中のrow0分) がゼロ化
2. 既存の 93→33 (row0以外) が1件も変化しない (スコープ外のbit-identical確認)
3. c13/c19/c24 の該当セル ±10秒タイムライン数値突合 + viz目視
完了しない場合は Stage1.5 を row0既知課題付きで提示し、1.5b は持ち越し (未検証の
「直った」報告は禁止)。

### 11.3 却下した代替案 (診断記録)
- (a) 可視行スナップショット固定: 同一window内の複数着地で差分が合算されn>=3に化け破綻
- (c) row0 を EFFECT_GATE_TOP_ROWS に追加: 消費先が _apply_stable_recovery_gate であり
  infer_hidden_row 呼び出しには一切効かないノーオペ (経路誤認)

## 12. close側の再設計 (2026-08-05 アーキ確定) — 値ベースヒステリシスの棄却記録

### 12.1 CLOSE_THRESHOLD=0.5 案は較正データによって否定 (恒久記録)
labeled_cell_features_v3.csv の平常フレーム (no_effect/baseline) は **約6割が
bright_ratio_max >= 0.5**。実例: c55 2P no_effect=0.954、c68 1P no_effect=0.972
(現行OPEN 0.97超の平常フレームが実在)。bright_ratio_max はぷよ自体の色・反射で
平常時から高く、値を下げると quiescence (score<閾値 0.25秒) が常態的に不成立
= 準永久凍結化する。**「較正データ不足」ではなく「較正データによって否定」** —
将来の再提案時はこの記録を参照のこと。

### 12.2 採用設計: 時間ベース2段構成
1. **BURST_GATE_POST_CLOSE_COOLDOWN_SEC** (主機構): 窓close後もクールダウン中は
   実効ゲート信号 (遷移mergeフィルタ+hard freeze の適用条件) を維持。値は
   burst_afterglow_events.csv の非censored p90=0.8s + 量子化マージン = **0.9秒**
   (censored除外の保守的下限である旨を定数コメントに明記)
2. **相手連鎖継続による close延長 (extend-only)**: 相手連鎖継続フラグ
   (ChainPhaseDetector/score OCR由来、視覚scoreと独立) が立っている間は窓を
   閉じない。連鎖持続型30% (固定クールダウンで不可能と実測済み) への対策。
   **§2.3「ChainEventをトリガーから捨てる」の上書きではない**: trigger役 (false
   negative=保護されない) と extend役 (false negative=時間ベースにフォールバック
   するだけで無害、false positive=凍結が延びるだけでMAX_WINDOW=30秒が吸収) は
   失敗の非対称性が全く違う
3. CLOSE_THRESHOLD は導入しない (OPEN=CLOSE=0.954 維持)。QUIESCENCE_MIN_SEC=0.25 不変

### 12.3 バックテスト規律
Stage1.5b (隠し段ゲート) と本§12は**別軸として分離実装・分離検証**: 本日は
1.5b単独 (close現行のまま) を先行、§12実装後に両者揃いの factorial (§10.6規律)。

## 12.4 差分実験による延長の退行確定と暫定運用 (2026-08-05 15:10)
- c19部分再走行×3構成: A(§12なし・1.5bなし)=mismatch0 / B(1.5bのみ)=0 / **C(§12のみ)=59・実効盤面5.33秒遅れ・スナップ半減**
- GAP_MAX=3.3でも、busy局面では再点火+連鎖で延長が連結し過剰凍結 (リアルタイム表示にも不適)
- **暫定運用: 連鎖延長は無効 (gap_max=0.0)、close後クールダウン0.9秒のみ** — c29型close漏れ
  (0.27-0.33s) はカバー、持続連鎖の再点火間隙 (1.37-3.13s) は未カバーとして正直に残す
- 延長の再設計 (総延長上限・凍結スコープの縮小等) は Stage2 案件

## 12.5 §12系の根本欠陥確定 — 雪だるま棄却 (2026-08-05 17:00計装)
- c19計装 (OFF/v3c二重pipeline): 相手長連鎖でクールダウンがリンク間隙 (0.4-2.2s、大半0.9s未満)
  を橋渡しし **11.07秒の連続凍結** (2Pデューティ比33.9%)。その間のOJAMA_FALL遷移で
  Stage1.5フィルタ (allowed={9}) が**正規の色設置を繰り返し棄却** (1遷移で最大29セル)、
  棄却負債が雪だるま化して58セルstaleに至る。staleは演出後も高止まり=(b)設計不良
- 結論: §12 (クールダウン/延長) は「凍結を伸ばすほどStage1.5のOJAMA_FALLスコープが
  正規設置を巻き込む」構造欠陥と結合しており、単独調整では解決しない
- **Stage2再設計の要件**: OJAMA_FALLスコープに着弾予定数キャップ (会計連動) または
  設置分の識別、凍結中の定期的なTSUMO_FALL素通し、from_state判定の正確性検証
- **暫定確定構成 (2026-08-05夜レビュー提示)**: Stage1.5+0.954 (93→33実測) + 1.5b
  (row0対策、差分実験で無罪)。§12系フラグは全OFF

## 13. userレビュー決定 (2026-08-05 夜、モバイルレビュー10問)
- **バーストガード (Stage1.5+0.954+1.5b) 採用 = 条件付き承諾** (Q1 HOLD+memo「一応承諾で良い」)。
  条件 = 「エフェクト中にツモの設置がある場合の対応」は議論の余地あり → Stage2設計の
  最優先論点として扱う (§12.5の雪だるま知見と同根)
- **案B 正式見送り確定** (Q2 OK、前回HOLD解除)
- 改善例3シーンの実画面確認 OK (Q3-5)
- 弱光14マスの追加ラベル作業 = 後日 (Q6 LATER)
- **方針 = 先に99.99%再測定** (Q7 MEASURE、推奨案採択)。足りなければStage2へ戻る

## 14. v4最終検収 (2026-08-05 18:50) — 1.5bは効果未実証
- v4 (Stage1.5+0.954+1.5b) 31/31本: known=33 (v2_fullと同一、退行なし) / new=45 (v2_full 37比+8、
  effective突合の走行間ノイズの可能性はあるが利益の証拠なし)
- row0持続誤り3セル (c13×2, c15 1P) のスポット確認: **3セルとも残存** — 1.5bの
  クールダウン0.4秒が row1-3 の追いつき時間に対して不足、または汚染時刻がゲート
  条件外の可能性。1.5b は「実装済み・効果未実証・default OFF」として Stage2 調整対象
- **採用確定コア = Stage1.5+0.954 (v2_full構成、known 93→33/new37)**
- 99.99%物差し走行 (14動画全編) は v4フラグのまま続行 (判定は row1-12 基準で
  row0 非対象のため影響なし、1.5bの差はrow0のみ)

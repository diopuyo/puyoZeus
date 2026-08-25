# 交換会計への gross 量供給設計 (P1-3 対応) — 2026-08-25

status: **設計確定待ち (Codex レビュー前)。実装未着手。**
対象 Gate: 3R-1 の 3番 / 3R-4「同時応酬で gross 相殺量が失われない」

## 1. 問題 (P1-3)

`src/exchange_episode_tracker.py::_classify_side_delta` (:312-337) は
`pending_pX_uncapped` の**純差分**から相殺・着弾を推測する。
`prev=100, curr=80, chain_finalized=True` は「cancel=20」と
「同一フレーム incoming=30 + cancel=50」を区別できず、両方 20 になる。
gross 相殺量が失われ、相手本線→自分本線の応酬で会計が壊れる。

**SPEC `docs/EXCHANGE_EPISODE_SPEC_2026-08-24.md` §6.3 (spec:504-511、
「pending_pX_uncapped 差分の判別的再構成」) は本設計により廃止・差し替え。**

## 2. 現状の gross 可用性 (`src/ojama_accounting.py`、実読で確認)

| 量 | 可用性 | file:line |
|---|---|---|
| gross 生成 (cap 前) | **既にある** (`total_generated`、`score_to_ojama` に cap なし) | :712-720, snapshot :1101-1102 |
| 自己相殺 (uncapped 基準) | **無い**。:741-744 で計算されるが量を捨てている。既存 `total_offset` (:749) は **capped forecast 基準 (:732)** で汚染済み → 権威に使用禁止 (Codex 指示とも一致) | :732, :741-749 |
| 着弾 (uncapped 基準) | **無い**。:536-538 で drain するが累積なし。`total_dropped` (:543) は capped 基準のみ | :536-543 |
| 境界ワイプ回数 | **無い** (量の消失は :997) | :980-1023 |
| uncapped サニティ切り捨て量 (上限 2,857 個) | **無い** (黙って min で切っている) | :101, :745-748 |

## 3. 採用方式: 独立累積カウンタ (cap 前・カテゴリ別・単調)

gross event キュー案は**不採用**。理由:

1. 消費契約 (単一 consumer 前提) が overlay + 診断の複数読者で壊れる
2. chain 帰属は `chain_id_resolver` の既存責務と二重化する
3. 単調カウンタの差分はフレーム欠落・順序に頑健で、保存則テストが恒等式で書ける

このプロジェクトは配線事故 (漏れ / 間違いの2種、単一情報源違反) を繰り返しており、
**消費契約が不要な方式**が全体最適。

カウンタの弱点「同一フレーム・同 side・同カテゴリの合算」は、finalize が
settle 20 フレーム待ち (:118, :897-904) を要するため 1 side 1 フレーム最大 1 回であり
実害なしと見込む (**未確認。T9 で構造検証する**)。

**cap 済み `total_offset` を権威に戻す案は採らない** (Codex 明示禁止。
:732 が capped 基準なので cap 汚染をそのまま輸入することになる)。

### 3.1 追加する内部状態 (`_SideState`、:279 付近に末尾追加)

```
total_offset_uncapped: int = 0    # 自己相殺の累積 (uncapped forecast 基準)
total_dropped_uncapped: int = 0   # 着弾 drain の累積 (uncapped 帳簿側)
boundary_reset_count: int = 0     # 境界ワイプ回数 (量は直前 pending で復元可)
uncapped_clamp_loss: int = 0      # サニティ切り捨てで消えた量の累積 (期待値 0)
```

### 3.2 更新点 (3 箇所、各 1〜4 行)

1. `on_tsumo_settled` :536-538 — uncapped drain 量を `total_dropped_uncapped` へ累積
2. `_finalize_chain_end` :741-748 — 純関数呼び出し**前**に
   `canceled_unc = min(gen, s.forecast_incoming_uncapped)` を取り
   `total_offset_uncapped` へ累積。サニティ clamp の切り捨て量を
   `uncapped_clamp_loss` へ累積 (**黙って切らない**)
3. `_reset_side_boundary` :997 付近 — `boundary_reset_count += 1`

### 3.3 供給インタフェース (新規、既存 API 無変更)

```python
@dataclass(frozen=True)
class GrossOjamaCounters:
    """cap 前 gross 量の side 別累積カウンタ (すべて単調非減少)。"""
    t_sec: float
    generated_p1: int; generated_p2: int          # 既存 total_generated の再掲
    offset_uncapped_p1: int; offset_uncapped_p2: int
    dropped_uncapped_p1: int; dropped_uncapped_p2: int
    boundary_resets_p1: int; boundary_resets_p2: int
    clamp_loss_p1: int; clamp_loss_p2: int


class OjamaAccountingTracker:
    def get_gross_counters(self, t_sec: float) -> GrossOjamaCounters: ...
```

- **`OjamaAccountSnapshot` には追加しない。** snapshot の全 field を動的に
  serialize する経路の有無が**未確認**のため、bit-identical を構造で保証する側に倒す。
- 送付 (sent) は `generated − offset_uncapped` の**導出量**。独立カウンタは持たない
  (恒等式検査の左右を独立に保つため)。
- incoming to X = 相手側の `(Δgenerated − Δoffset_uncapped)`。

### 3.4 消費側 (`src/exchange_episode_tracker.py`)

- 新純関数 `classify_gross_counter_delta(prev, curr, prev_pending, curr_pending)`
  を :340 付近に追加 (50 行以内)。カテゴリ別 Δ をそのまま
  `SettlementObservation` に変換する。**推測しない。**
- 恒等式残差 `pending Δ − (incoming − Δoffset − Δdropped − wipe − clamp)` を
  `unclassified` として**母数付き**カウンタに出す (期待値 0。
  検査フレーム数を分母として併記する — memory `feedback_zero_needs_denominator_2026-08-25`)。
- 既存 `_classify_side_delta` / `classify_pending_uncapped_delta` は**削除せず維持**
  (backwards compat)。新経路は probe / 配線側の**既定 OFF** optional 引数で切替。

## 4. 不変条件テスト (地上真値つき fixture)

fixture は**動画不要**。`OjamaAccountingTracker` を合成 score 列で直接駆動する
(rate=70 点/個、elapsed=0 でマージンタイム無効、`score_to_ojama` の換算が地上真値)。
両 side の settle カウンタが**同一フレームで満了**するよう score 列を組めば、
「同時応酬」が決定的に再現できる。

| # | テスト | 固定する不変条件 |
|---|---|---|
| T1 | **同時応酬** (P1-3 本丸): X pending=100、同一フレームで X finalize gen=50 + 相手 finalize gen=30 (surplus 30) → pending 80 | `Δoffset_unc_X=50` と `incoming_X=30` が別々に残る。**旧純差分だと 20 に潰れること**も併記 assert (回帰の証拠化) |
| T2 | 単純 cancel: gen=20、pending 100→80 | `Δoffset_unc=20`、`incoming=0` |
| T3 | 同一フレーム incoming + 着弾: ツモ設置 drain 30 + 相手 surplus 10 | `Δdropped_unc=30`、`incoming=10`、恒等式残差 0 |
| T4 | **cap 超え生成** (総生成52%消失の回帰): score 36,190 → gen=517 | `generated Δ=517` ちょうど (216 に丸まらない) |
| T5 | 保存則 (試合単位): `generated = offset_unc + sent(導出)` / `pending 終値 = Σincoming − Σdropped_unc − Σwipe − Σclamp` | 全恒等式成立 + **検査フレーム数 > 0 を分母として assert** |
| T6 | 境界ワイプ: score 大幅減少 → pending 0 | `boundary_resets +1`、消失量が wipe に分類され `unclassified=0` |
| T7 | clamp loss: pending を 2,857 超に駆動 | `clamp_loss > 0` と件数が出る (**黙って消えない**) |
| T8 | OFF bit-identical: 記録済み遷移列で既存 snapshot 全 field が追加前後で一致 (golden) | 既存出力不変 |
| T9 | 単調性 + 構造仮定: 全カウンタ非減少 (境界跨ぎ含む)、1 side 1 フレームの finalize ≤ 1 | 「合算で情報が消えない」前提の検証 |

Gate 3R-4「同時応酬で gross 相殺量が失われない」の検収 = **T1 + T3 + T5**。
実データ側は zenchi 再現データで恒等式残差の母数付き 0 を確認 (Gate 3R-3-4)。

## 5. 副作用範囲と工数 (**見積もり。実測前なので断定しない**)

| ファイル | 変更箇所 | 規模 |
|---|---|---|
| `src/ojama_accounting.py` | `_SideState` :279 付近 +4 field / :536-538 / :741-748 / :997 / 新 dataclass + メソッド (末尾) | +60 行程度 |
| `src/exchange_episode_tracker.py` | :368 付近に新純関数 + frame 型。既存関数は不変 | +70 行程度 |
| `scripts/_gate3_episode_probe_v5_*.py` | v4 のコピーで供給元をカウンタ差分へ (**v4 は上書きしない**) | 新規 |
| `tests/test_ojama_accounting.py` / `tests/test_exchange_episode_tracker.py` | T1〜T9 | +12〜15 tests |
| `docs/EXCHANGE_EPISODE_SPEC_2026-08-24.md` | §6.3 の廃止注記 (Codex 管理のため `CLAUDE_TO_CODEX.md` へ依頼) | 数行 |

`src/production_config.py` は触らない (Gate 3R-0 凍結)。
工数: 実装 0.5〜1 日 + テスト 0.5 日 (**見積もり**)。

## 6. 実装前に潰す未確認事項

1. snapshot の全 field を動的に serialize する dump 経路の有無
   (あれば snapshot 拡張は将来も不可と確定する)
2. 1 side 1 フレーム finalize ≤ 1 の構造保証 (T9 で固定)

## 7. 次の手 (推奨順)

1. `CLAUDE_TO_CODEX.md` へ「SPEC §6.3 廃止 + 本設計での差し替え」を依頼として追記
2. Codex 合意後、**T8 (OFF bit-identical golden) と T1 (同時応酬) を先に失敗テストとして書いてから**実装
   (Gate 3R-1 の再開順序 1 と整合)
3. 上記 6 の未確認 2 点を実測で潰す

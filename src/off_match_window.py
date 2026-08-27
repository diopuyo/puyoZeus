"""試合外区間 (盤面が画面に無い区間) の判定 (2026-08-24、W37)。

## なぜ要るか

試合の勝敗が決まると「ばたんきゅー」「やった!」のテロップが出て、
**負けた側の盤面が画面から完全に消える**。そこには背景のステージ
(緑の地面、淡い色の大きなぷよ型マスコット) が見えている。

認識精度の物差し `scripts/measure_stable_cell_acc.py` はこの区間も
STABLE として評価対象に含めてしまい、CNN が背景を「おじゃま」「青」と
読んだ結果を認識誤りとして計上していた
(`docs/KNOWN_WEAKNESSES.md` W37)。**そこには正解が存在しない。**

v51 では不一致セル 1,102 件のうち 991 件 (90%) がこの種の窓由来だった。

## 既存の検出器が使えなかった経緯

`MatchEndDetector` (テンプレート NCC) が `PipelineResult.match_end_locked` を
毎フレーム公開しているので、それを読めば済むはずだった。
**しかし v51 の t=50〜70 で実測したところ発火 0 件。**
`src/recognition_pipeline.py` のコメントに
「テンプレ NCC は配信レイアウト依存で 42 本中 31 本が一度も成立せず」
と記録があり、v51 もその 31 本側だった。
`is_match_active=False` になるのも t=63.333 以降で、汚染区間の**後**。

## user 伝授の判定規則 (2026-08-24)

    ネクストが動かない + スコアが動かない + スコアが 0 でない
    が 2 秒続いたら「試合外」。さらに**そこから遡って 2 秒も試合外**。

なぜこれで分かれるか:

| 場面 | ネクスト | スコア | 判定 |
|---|---|---|---|
| 通常のプレイ | 設置のたびに変わる | 変わる | 試合中 |
| **連鎖中** | 止まる | **増え続ける** | 試合中 (スコアで除外) |
| 試合開始前 | 止まる | **0** | 試合中でない扱いにしない (スコア0で除外) |
| **試合終了テロップ** | 止まる | 止まる (最終値) | **試合外** |

連鎖中を除外できるのがこの規則の要で、
「ネクストが止まる」だけでは連鎖中と区別できない。

## 遡り 2 秒の意味

凍結が 2 秒続いて初めて判定できるので、判定時点では既に 2 秒経過している。
その 2 秒も試合外なので遡って外す。
**物差しはオフライン処理なので先読みが許される** (リアルタイム表示とは違う)。

## 全消しテロップは対象にしない

全消しのテロップは**盤面枠が見えたまま**で、中身も定義できる
(実画面 `data/verify/formula_v51_diag_2026-08-24/evidence_frames/v51_t002.0s.png`:
1P は右下に 4 個、2P は左列に数個、実際にぷよが残っている)。
そこでの誤読は**本物の認識課題**であって測定汚染ではない。
全消しは本番でも必ず起きるので、除外したら弱点を隠すことになる。
"""
from __future__ import annotations

from dataclasses import dataclass

# 凍結が試合外と見なせるまでの継続時間 (user 伝授、2026-08-24)。
OFF_MATCH_FREEZE_SEC: float = 2.0

# 判定成立時に遡って試合外とする時間 (user 伝授、同上)。
# 凍結の検知には OFF_MATCH_FREEZE_SEC かかるので、その分を戻す。
OFF_MATCH_BACKTRACK_SEC: float = OFF_MATCH_FREEZE_SEC


@dataclass(frozen=True)
class SideObservation:
    """1 フレーム・1 サイド分の観測 (判定に使う最小限)。"""

    t_sec: float
    score: int | None      # 通常スコア OCR の値。読めなければ None
    next_key: object       # ネクストの識別値 (同値なら「動いていない」)


def find_off_match_spans(
    observations: list[SideObservation],
    freeze_sec: float = OFF_MATCH_FREEZE_SEC,
    backtrack_sec: float = OFF_MATCH_BACKTRACK_SEC,
) -> list[tuple[float, float]]:
    """試合外と判定できる区間 [開始, 終了] のリストを返す。

    判定規則 (user 伝授、モジュール docstring 参照):
        ネクストが動かない + スコアが動かない + スコアが 0 でない
        が freeze_sec 続いたら試合外。検知点から backtrack_sec 遡る。

    Args:
        observations: 時刻昇順のフレーム観測列 (同一サイド分)。
        freeze_sec: 凍結が試合外と見なせるまでの継続時間。
        backtrack_sec: 判定成立時に遡る時間。

    Returns:
        [(開始秒, 終了秒), ...]。重なる区間は結合済み。
    """
    if not observations:
        return []
    raw = _collect_frozen_spans(observations, freeze_sec)
    widened = [(max(0.0, a - backtrack_sec), b) for a, b in raw]
    return _merge_spans(widened)


def _collect_frozen_spans(
    obs: list[SideObservation], freeze_sec: float,
) -> list[tuple[float, float]]:
    """freeze_sec 以上続いた凍結区間をそのまま返す (遡りは適用しない)。"""
    spans: list[tuple[float, float]] = []
    start: float | None = None
    prev_key: tuple[int | None, object] | None = None
    for o in obs:
        # スコアが 0 / 未読のフレームは判定に使わない (試合開始前を除くため)。
        # ただし **蓄積中の凍結はここで確定させる**。試合終了後はスコアが 0 に
        # リセットされるので、閉じずに捨てると「テロップ中ずっと凍結 → 最後に 0」
        # という本命のケースを丸ごと取り逃す (v51 で実際に踏んだ:
        # t=56.333 に score=58503 で凍結 → t=63.233 に score=0)。
        if o.score in (None, 0):
            if start is not None and o.t_sec - start >= freeze_sec:
                spans.append((start, o.t_sec))
            start, prev_key = None, None
            continue
        key = (o.score, o.next_key)
        if prev_key is not None and key != prev_key:
            # 動いた → それまでの凍結を確定させる
            if start is not None and o.t_sec - start >= freeze_sec:
                spans.append((start, o.t_sec))
            start = o.t_sec
        elif start is None:
            start = o.t_sec
        prev_key = key
    if start is not None and obs[-1].t_sec - start >= freeze_sec:
        spans.append((start, obs[-1].t_sec))
    return spans


def _merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """重なる / 接する区間を結合する。"""
    if not spans:
        return []
    ordered = sorted(spans)
    merged = [ordered[0]]
    for a, b in ordered[1:]:
        last_a, last_b = merged[-1]
        if a <= last_b:
            merged[-1] = (last_a, max(last_b, b))
        else:
            merged.append((a, b))
    return merged


def is_in_spans(t_sec: float, spans: list[tuple[float, float]]) -> bool:
    """時刻が試合外区間に入っているか。"""
    return any(a <= t_sec <= b for a, b in spans)

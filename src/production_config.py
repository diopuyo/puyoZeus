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

from dataclasses import dataclass


@dataclass(frozen=True)
class AdoptedFlag:
    """採用済みフラグ 1 件。"""

    flag: str          # CLI フラグ文字列 (値を取る場合は "--name value" 形式)
    adopted: str       # 採用日 (YYYY-MM-DD)
    reason: str        # 採用根拠 (計測結果 / レビュー)


# ============================
# 認識 — 収集専用 (collect_boards_lean のみが必要とするフラグ)
# ============================
# RecognitionPipeline.load_default では既定 True だが collect_boards_lean だけ
# 既定 False のため、 収集時のみ明示指定が要る。 表示系スクリプトに渡すと
# 「unrecognized arguments」で落ちるので分けて管理する
# (2026-08-08 に実際に踏んだ)。
COLLECT_ONLY_ADOPTED: tuple[AdoptedFlag, ...] = (
    AdoptedFlag(
        "--enable-chain-tracker", "2026-07-30",
        "機能D単独では CHAIN 検知が実運用 0 件で、連鎖中の盤面凍結が働かない",
    ),
)

# ============================
# 認識 — 共通 (collect_boards_lean / visualize_recognition の両方が受け付ける)
# ============================
# Phase L の全動画 regen で実際に使っている構成と一致させること。
RECOGNITION_ADOPTED: tuple[AdoptedFlag, ...] = (
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
    AdoptedFlag(
        "--platt-calibration", "2026-08-04",
        "表示勝率の post-hoc 較正。未較正だと自信過剰 (80%表示の実勝率 64%)",
    ),
    AdoptedFlag(
        "--sample-interval 0", "2026-07-13",
        "毎フレーム更新。0.5 秒間引きだとおじゃま会計がスコア変化・連鎖終了を"
        "取りこぼし net/forecast=0 になる (ADVANTAGE_OVERLAY_2026-07-13 §2-3)",
    ),
)

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
)


def _join(flags: tuple[AdoptedFlag, ...]) -> str:
    """フラグ文字列を空白区切りで連結する。"""
    return " ".join(f.flag for f in flags)


def recognition_flags() -> str:
    """認識の本番構成フラグを返す (収集・表示の両方が受け付けるもの)。"""
    return _join(RECOGNITION_ADOPTED)


def collect_flags() -> str:
    """collect_boards_lean 用の全フラグ (共通 + 収集専用) を返す。"""
    return _join(RECOGNITION_ADOPTED + COLLECT_ONLY_ADOPTED)


def advantage_overlay_flags() -> str:
    """有利不利オーバーレイの本番構成フラグを返す。"""
    return _join(ADVANTAGE_ADOPTED)


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
        ("表示", VISUALIZATION_ADOPTED),
    ):
        lines.append(f"[{title}]")
        for f in flags:
            lines.append(f"  {f.flag}  (採用 {f.adopted}) — {f.reason}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())

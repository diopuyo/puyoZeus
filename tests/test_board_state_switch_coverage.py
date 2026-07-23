"""反復9 ガード2 (2026-07-23): CHAIN 単独 state チェックの再発防止メタテスト。

反復2(根治)で発見されたバグパターン: `state == BoardState.CHAIN` の単独
(tuple でない) 等価チェックは、後から BoardState.GRAVITY_SETTLE のような
中間状態が導入されると dead code 化する (CHAIN → GRAVITY_SETTLE → STABLE
という経路が新設されると、旧来 CHAIN 直後だけを想定したチェックが永久に
False になる)。

本テストは src/recognition_pipeline.py を静的スキャンし、
`state == BoardState.CHAIN` 単独チェックの近傍 (前後 WINDOW_LINES 行) に
GRAVITY_SETTLE への言及、または `# GS_EXEMPT: <理由>` コメントが無い箇所を
検出して fail させる。将来同種の状態追加が起きた際、機械的にバイパス
(=同じ穴に落ちる箇所) を検出するための安全網。

新たに CHAIN 単独チェックを書く場合の対応:
  - 本来 GRAVITY_SETTLE も含めるべきなら
    `state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE)` に拡張する。
  - 意図的に CHAIN のみを対象とする場合 (= GRAVITY_SETTLE 中は無関係と
    確認済み) は、同じ行か近傍に `# GS_EXEMPT: <理由>` を付与する。
"""
from __future__ import annotations

import re
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
TARGET_FILE = PROJ_ROOT / "src" / "recognition_pipeline.py"

# state == BoardState.CHAIN の単独 (tuple でない) 等価チェックを検出する。
# `state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE)` のような tuple 形は
# `==` でなく `in` を使うため、このパターンには意図的にマッチしない
# (= 既に GRAVITY_SETTLE 対応済とみなせるため検査対象外)。
_CHAIN_EQ_PATTERN = re.compile(
    r"\b(prev_state|ctx\.state|self\._sm_\w+\.context\.state|state)\s*==\s*"
    r"BoardState\.CHAIN\b",
)
_GS_EXEMPT_PATTERN = re.compile(r"GS_EXEMPT")
_GRAVITY_SETTLE_PATTERN = re.compile(r"GRAVITY_SETTLE")

# 「前後数行」の探索窓 (この範囲内に GRAVITY_SETTLE 言及 or GS_EXEMPT が
# あれば安全とみなす)。
WINDOW_LINES: int = 5


def _scan_bare_chain_checks(source_lines: list[str]) -> list[tuple[int, str]]:
    """state == BoardState.CHAIN 単独チェックのうち、GRAVITY_SETTLE 言及も
    GS_EXEMPT コメントも前後 WINDOW_LINES 行以内に無い行を検出する。

    Args:
        source_lines: 対象ファイルの各行 (改行込み)。

    Returns:
        (1-indexed 行番号, 行内容) のリスト (違反箇所、無ければ空)。
    """
    violations: list[tuple[int, str]] = []
    n = len(source_lines)
    for i, line in enumerate(source_lines):
        if not _CHAIN_EQ_PATTERN.search(line):
            continue
        window_start = max(0, i - WINDOW_LINES)
        window_end = min(n, i + WINDOW_LINES + 1)
        window_text = "".join(source_lines[window_start:window_end])
        if _GRAVITY_SETTLE_PATTERN.search(window_text):
            continue
        if _GS_EXEMPT_PATTERN.search(window_text):
            continue
        violations.append((i + 1, line.rstrip("\n")))
    return violations


def test_no_bare_chain_state_check_without_gravity_settle_or_exempt() -> None:
    """反復9 ガード2: CHAIN 単独チェックは GRAVITY_SETTLE 対応 or GS_EXEMPT 必須。

    根治 (2026-07-23) で発見された「GRAVITY_SETTLE 導入時に CHAIN 単独
    チェックが dead code 化する」バグパターンの再発防止メタガード。
    """
    source_lines = TARGET_FILE.read_text(encoding="utf-8").splitlines(keepends=True)
    violations = _scan_bare_chain_checks(source_lines)
    if violations:
        detail = "\n".join(f"  L{ln}: {txt.strip()}" for ln, txt in violations)
        raise AssertionError(
            "state == BoardState.CHAIN 単独チェックが GRAVITY_SETTLE 対応も "
            "GS_EXEMPT も無いまま検出されました (将来の状態追加で dead code "
            "化する再発リスク)。state in (CHAIN, GRAVITY_SETTLE) に拡張する"
            "か、意図的なら `# GS_EXEMPT: <理由>` を付与してください:\n"
            + detail,
        )


def test_scan_helper_detects_known_violation_pattern() -> None:
    """メタガード自体の自己診断: 素の CHAIN 単独チェックは検出されるはず。"""
    sample = [
        "if ctx.state == BoardState.STABLE:\n",
        "    pass\n",
        "    x = ctx.state == BoardState.CHAIN\n",
        "    pass\n",
    ]
    violations = _scan_bare_chain_checks(sample)
    assert len(violations) == 1
    assert violations[0][0] == 3


def test_scan_helper_ignores_gravity_settle_nearby() -> None:
    """メタガード自体の自己診断: 近傍に GRAVITY_SETTLE 言及があれば無視する。"""
    sample = [
        "# GRAVITY_SETTLE も考慮済み\n",
        "x = ctx.state == BoardState.CHAIN\n",
    ]
    violations = _scan_bare_chain_checks(sample)
    assert violations == []


def test_scan_helper_ignores_gs_exempt_comment() -> None:
    """メタガード自体の自己診断: GS_EXEMPT コメントがあれば無視する。"""
    sample = [
        "x = ctx.state == BoardState.CHAIN  # GS_EXEMPT: 意図的除外の例\n",
    ]
    violations = _scan_bare_chain_checks(sample)
    assert violations == []


def test_scan_helper_ignores_tuple_in_check() -> None:
    """メタガード自体の自己診断: `in (CHAIN, GRAVITY_SETTLE)` 形は対象外。"""
    sample = [
        "if state in (BoardState.CHAIN, BoardState.GRAVITY_SETTLE):\n",
        "    pass\n",
    ]
    violations = _scan_bare_chain_checks(sample)
    assert violations == []

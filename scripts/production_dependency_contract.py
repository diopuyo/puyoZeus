"""保存済み盤面の学習・検収が依存する物理定数と、台帳の版を分離する。"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_SHA256 = "3fe3c2578b3196a60cffffcafa594c1f64d2a913bb0487932ac5cd8f6f86d376"
REQUIRED_CONSTANTS = {"GHOST_CHAIN_RULE_ENABLED": True}
CONTRACT_VERSION = "saved-board-production-dependencies/v1"


def constants_from_source(raw: bytes) -> dict[str, Any]:
    """依存値の単一の静的定義だけを受け入れる。"""
    tree = ast.parse(raw)
    found: dict[str, list[Any]] = {name: [] for name in REQUIRED_CONSTANTS}
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else [])
        for target in targets:
            if isinstance(target, ast.Name) and target.id in found:
                try:
                    found[target.id].append(ast.literal_eval(node.value) if node in tree.body else None)
                except (ValueError, TypeError):
                    found[target.id].append(None)
    values = {name: items[0] if len(items) == 1 else None for name, items in found.items()}
    return values


def dependency_receipt(root: Path = ROOT) -> dict[str, Any]:
    """値の自己申告ではなく、SHAを再検査できる元ソースを票へ保持する。"""
    raw = (root / "src/production_config.py").read_bytes()
    values = constants_from_source(raw)
    compatible = all(type(values[k]) is type(v) and values[k] == v
                     for k, v in REQUIRED_CONSTANTS.items())
    digest = hashlib.sha256(raw).hexdigest()
    return dict(format_version=CONTRACT_VERSION, required_constants=dict(REQUIRED_CONSTANTS),
                actual_constants=values, compatible=compatible, actual_sha256=digest,
                historical_sha256=HISTORICAL_SHA256, unchanged=digest == HISTORICAL_SHA256,
                source_text=raw.decode("utf-8"))


def production_compatible(root: Path = ROOT) -> bool:
    """依存値の欠落・型違い・変更は拒否し、採用理由の追記とは区別する。"""
    return dependency_receipt(root)["compatible"]


def saved_dependency_compatible(digest: str | None, receipt: dict | None = None) -> bool:
    """旧版は既知SHA、新版は保存時の依存値票とSHAの結合を要求する。"""
    if digest == HISTORICAL_SHA256:
        return True
    if not isinstance(receipt, dict):
        return False
    source = receipt.get("source_text")
    if not isinstance(source, str):
        return False
    raw = source.encode("utf-8")
    try:
        actual = constants_from_source(raw)
    except (SyntaxError, ValueError):
        return False
    return all((receipt.get("format_version") == CONTRACT_VERSION,
                hashlib.sha256(raw).hexdigest() == digest,
                actual == receipt.get("actual_constants"),
                all(type(actual[k]) is type(v) and actual[k] == v
                    for k, v in REQUIRED_CONSTANTS.items()),
                isinstance(digest, str), len(digest or "") == 64,
                receipt.get("actual_sha256") == digest,
                receipt.get("required_constants") == REQUIRED_CONSTANTS,
                receipt.get("actual_constants") == REQUIRED_CONSTANTS,
                all(type(v) is bool for v in receipt.get("actual_constants", {}).values()),
                all(type(v) is bool for v in receipt.get("required_constants", {}).values()),
                receipt.get("compatible") is True))

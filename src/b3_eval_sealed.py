"""封印セットの一括開封手順 (事前登録 §6)。**実装のみ。本セッションでは実行しない (封印の成績は計算しない)。**

手順: (1) 固定ファイルの SHA256 照合 → (2) 候補/R の hash・合格条件 hash の照合 (門番 token)
      → (3) 全セットを 1 回の呼出で一括採点 (途中で止めない・差し替えない・破損は記録して継続)
      → (4) 開封記録を採点結果の返却前に書く (同じ記録が既にあれば再開封を拒否)。
中間成績を見て後半を取り止める経路を作らない: 採点関数は全セット分を内部で回し、成績は開封記録の後で返す。
ラベル・成績ファイルはこのモジュールが中身を読まない (hash のみ)。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

TOKEN_PREFIX = "OPEN-B3-SEALED-"
SCORED_OK, SCORED_BROKEN = "ok", "broken"


class SealedError(RuntimeError):
    """開封手順の拒否 (hash 不一致・token 不一致・再開封など)。"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: Path, relpaths: Sequence[str]) -> dict[str, str]:
    """固定ファイルの相対パス -> SHA256 (内容は読まず hash だけ取る)。"""
    return {rp: sha256_file(root / rp) for rp in sorted(relpaths)}


def manifest_sha(manifest: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(dict(sorted(manifest.items())), ensure_ascii=False).encode("utf-8")).hexdigest()


def verify_manifest(root: Path, manifest: Mapping[str, str]) -> list[dict]:
    """不一致・欠落の一覧 (空なら OK)。"""
    bad = []
    for rp, want in sorted(manifest.items()):
        p = root / rp
        if not p.exists():
            bad.append({"path": rp, "problem": "missing", "expected": want})
            continue
        got = sha256_file(p)
        if got != want:
            bad.append({"path": rp, "problem": "sha_mismatch", "expected": want, "actual": got})
    return bad


@dataclass(frozen=True)
class OpeningPlan:
    """開封で照合する固定物。いずれも事前登録の確定版を指す。"""
    prereg_commit: str
    candidate_sha: str          # 最終モデルのロック hash
    reference_sha: str          # R (本番) の hash
    gate_conditions_sha: str    # 合格条件 (合格条件の数値表) の hash
    labels_sha: str             # ラベル一式の hash (内容は読まない)
    executor: str
    purpose: str

    def token(self, manifest_hash: str) -> str:
        blob = json.dumps({**asdict(self), "manifest": manifest_hash}, sort_keys=True, ensure_ascii=False)
        return TOKEN_PREFIX + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def open_sealed(root: Path, manifest: Mapping[str, str], plan: OpeningPlan, set_names: Sequence[str],
                score_set: Callable[[str], Mapping], record_path: Path, *, token: Optional[str] = None,
                dry_run: bool = True, now: Callable[[], float] = time.time) -> dict:
    """一括開封。dry_run=True (既定) は照合だけ行い、score_set を一度も呼ばない。

    実開封は dry_run=False かつ token == plan.token(manifest_sha) のときだけ。
    """
    problems = verify_manifest(root, manifest)
    mhash = manifest_sha(manifest)
    expected = plan.token(mhash)
    status = {"manifest_sha256": mhash, "manifest_problems": problems, "expected_token": expected,
              "already_opened": record_path.exists(), "dry_run": dry_run, "sets": list(set_names)}
    if dry_run:
        return {**status, "executed": False}
    if problems:
        raise SealedError(f"固定ファイルの照合に失敗: {problems[:3]}")
    if token != expected:
        raise SealedError("token 不一致 (候補/R/合格条件/ラベルの hash 照合が済んでいない)")
    if record_path.exists():
        raise SealedError("開封記録が既にある (再開封は拒否。新しい封印は事前登録の改訂で再選定)")
    results: dict[str, dict] = {}
    for name in set_names:            # 一括採点: 例外は記録して継続 (破損・欠落も報告し、判定不能は保留)
        try:
            results[name] = {"status": SCORED_OK, "result": dict(score_set(name))}
        except Exception as exc:       # noqa: BLE001  破損セットを全体から隠さないために全捕捉して記録する
            results[name] = {"status": SCORED_BROKEN, "error": f"{type(exc).__name__}: {exc}"}
    record = {"opened_at_unix": now(), "executor": plan.executor, "purpose": plan.purpose,
              "plan": asdict(plan), "manifest_sha256": mhash, "sets": list(set_names),
              "set_status": {k: v["status"] for k, v in results.items()},
              "opening_complete": all(v["status"] == SCORED_OK for v in results.values())}
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("x", encoding="utf-8") as f:       # 'x' = 既存なら失敗 (二重開封の最終防壁)
        json.dump(record, f, ensure_ascii=False, indent=1)
    return {**status, "executed": True, "record": record, "results": results}

"""配布 stage の puyo_core (.pyd) を、新しくビルドした pyd へ SHA 照合つきで差し替える (既定は検査だけ)。

前例: packaging/build_live_v49f.py は stage の pyd を `shutil.copy2(NATIVE, native)` で上書きしていた。
ここでは凍結資産の差し替えの型 (.claude/rules/02-engineering-principles.md の 3) に合わせ、
  1. stage の pyd が想定の旧 SHA256 であること (違えば名前つきで落とす)
  2. 新 pyd の SHA256 が、ビルド台本が残した SHA256SUMS.txt と一致すること (あれば)
  3. 新 pyd が stage の Python で import でき、旧 pyd が出していた名前をすべて出していること (上位互換)
を確かめてから、--apply のときだけ差し替える。旧 pyd は backup へ退避し、受領票 (JSON) に旧・新の SHA256 と
増えた名前を残す。0 件の差し替えを成功と読み替えない (受領票の replaced が False なら何も変えていない)。

使い方:
  python packaging/swap_native_live_recog.py --stage C:/.../PuyoLive --new-pyd .../native_build/puyo_core.pyd \
      --receipt receipt.json            # 検査だけ
  ... --apply --backup-dir C:/.../backup  # 差し替える
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

# 24d の配布が持つ pyd の SHA256 (build_20261019a のビルド。packaging/build_live_v49f.py の NATIVE と同じ)。
DEFAULT_OLD_SHA256 = "c5058a27d95d5e579cd8223326f016f5df6779821b617c79c1c884ca28fcc129"
PACKAGE_RELATIVE = Path("python/Lib/site-packages/puyo_core")
PYTHON_RELATIVE = Path("python/python.exe")
SHA_FILE_NAME = "SHA256SUMS.txt"
EXPORT_PROBE = ("import sys, json; sys.path.insert(0, sys.argv[1]); import puyo_core; "
                "print(json.dumps(sorted(n for n in dir(puyo_core) if not n.startswith('_'))))")
EXPORT_TIMEOUT_SEC = 60


class SwapError(Exception):
    """差し替えを止める理由 (名前つき)。"""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def find_stage_pyd(stage: Path) -> Path:
    """stage の puyo_core の pyd (ちょうど 1 つ)。"""
    found = sorted((stage / PACKAGE_RELATIVE).glob("*.pyd"))
    if len(found) != 1:
        raise SwapError(f"stage の pyd がちょうど 1 つではありません: {[p.name for p in found]}")
    return found[0]


def check_recorded_sha(new_pyd: Path) -> str | None:
    """ビルド台本の SHA256SUMS.txt (同じフォルダ) があれば新 pyd と照合する。あればその SHA を返す。"""
    record = new_pyd.parent / SHA_FILE_NAME
    if not record.is_file():
        return None
    recorded = record.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(new_pyd)
    if recorded != actual:
        raise SwapError(f"新 pyd の SHA256 が {SHA_FILE_NAME} と違います: {actual} != {recorded}")
    return recorded


def exported_names(python_exe: Path, package_dir: Path, pyd: Path, scratch: Path) -> list[str]:
    """pyd を (パッケージの __init__.py と一緒に) 一時フォルダへ置き、stage の Python で import して名前を返す。"""
    target = scratch / package_dir.name
    shutil.copytree(package_dir, target, ignore=shutil.ignore_patterns("*.pyd", "__pycache__"))
    shutil.copy2(pyd, target / pyd.name)
    result = subprocess.run([str(python_exe), "-I", "-c", EXPORT_PROBE, str(scratch)], capture_output=True,
                            text=True, timeout=EXPORT_TIMEOUT_SEC, check=False)
    if result.returncode != 0:
        raise SwapError(f"stage の Python で import できません: {result.stderr.strip()[-300:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def plan_swap(stage: Path, new_pyd: Path, expected_old_sha: str, scratch: Path,
              probe: Callable[..., list[str]] = exported_names) -> dict[str, Any]:
    """差し替えの前提を検査して計画 (旧・新の SHA256、増えた名前) を返す。何も変更しない。"""
    old_pyd = find_stage_pyd(stage)
    old_sha = sha256_file(old_pyd)
    if old_sha != expected_old_sha:
        raise SwapError(f"stage の pyd が想定の旧版ではありません: {old_sha} != {expected_old_sha}")
    check_recorded_sha(new_pyd)
    package_dir = stage / PACKAGE_RELATIVE
    python_exe = stage / PYTHON_RELATIVE
    old_names = set(probe(python_exe, package_dir, old_pyd, scratch / "old"))
    new_names = set(probe(python_exe, package_dir, new_pyd, scratch / "new"))
    if not old_names <= new_names:
        raise SwapError(f"新 pyd に無い名前があります: {sorted(old_names - new_names)}")
    return dict(old_pyd=str(old_pyd), old_sha256=old_sha, new_pyd=str(new_pyd), new_sha256=sha256_file(new_pyd),
                added_names=sorted(new_names - old_names), old_name_count=len(old_names),
                new_name_count=len(new_names), replaced=False)


def apply_swap(plan: dict[str, Any], backup_dir: Path) -> dict[str, Any]:
    """旧 pyd を backup へ退避し、新 pyd を置く。置いた後に SHA256 を再計算して照合する。"""
    old_pyd = Path(plan["old_pyd"])
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{old_pyd.name}.{plan['old_sha256'][:12]}.bak"
    shutil.copy2(old_pyd, backup)
    shutil.copy2(plan["new_pyd"], old_pyd)
    placed = sha256_file(old_pyd)
    if placed != plan["new_sha256"]:
        shutil.copy2(backup, old_pyd)
        raise SwapError("差し替え後の SHA256 が新 pyd と一致しません (旧 pyd へ戻しました)")
    return dict(plan, replaced=True, backup=str(backup), placed_sha256=placed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--stage", type=Path, required=True, help="PuyoLive フォルダ (python/ と app/ を含む)")
    parser.add_argument("--new-pyd", type=Path, required=True)
    parser.add_argument("--expected-old-sha256", default=DEFAULT_OLD_SHA256)
    parser.add_argument("--scratch", type=Path, help="import 検査用の一時フォルダ (空で存在しないこと)")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="検査に通れば差し替える (既定は検査だけ)")
    parser.add_argument("--backup-dir", type=Path, help="--apply のとき旧 pyd の退避先")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    if options.apply and options.backup_dir is None:
        print("--apply には --backup-dir が必要です", file=sys.stderr)
        return 2
    scratch = options.scratch or options.receipt.parent / (options.receipt.stem + "_scratch")
    try:
        plan = plan_swap(options.stage, options.new_pyd, options.expected_old_sha256, scratch)
        receipt = apply_swap(plan, options.backup_dir) if options.apply else plan
    except SwapError as error:
        print(f"差し替えを中止: {error}", file=sys.stderr)
        return 1
    options.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

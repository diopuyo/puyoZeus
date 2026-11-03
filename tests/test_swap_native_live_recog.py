"""配布 stage の pyd 差し替え (packaging/swap_native_live_recog.py) の安全装置の試験。実物の pyd は使わず、偽のファイルで確かめる。"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packaging"))
import swap_native_live_recog as swap  # noqa: E402

OLD_BYTES, NEW_BYTES = b"old-binary", b"new-binary"
OLD_NAMES = ["classify_cells_hsv", "simulate_chain_py"]
NEW_NAMES = [*OLD_NAMES, "read_board_first_pass"]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stage(tmp_path: Path, data: bytes = OLD_BYTES, extra_pyd: bool = False) -> Path:
    package = tmp_path / "PuyoLive" / swap.PACKAGE_RELATIVE
    package.mkdir(parents=True)
    (package / "puyo_core.cp312-win_amd64.pyd").write_bytes(data)
    (package / "__init__.py").write_text("from .puyo_core import *\n", encoding="utf-8")
    if extra_pyd:
        (package / "other.pyd").write_bytes(b"x")
    return tmp_path / "PuyoLive"


def _new_pyd(tmp_path: Path, record: str | None = None) -> Path:
    folder = tmp_path / "build"
    folder.mkdir()
    pyd = folder / "puyo_core.pyd"
    pyd.write_bytes(NEW_BYTES)
    if record is not None:
        (folder / swap.SHA_FILE_NAME).write_text(f"{record}  puyo_core.pyd\n", encoding="utf-8")
    return pyd


def _probe(old: list[str], new: list[str]) -> Any:
    def probe(python_exe: Path, package_dir: Path, pyd: Path, scratch: Path) -> list[str]:
        return list(new if pyd.read_bytes() == NEW_BYTES else old)
    return probe


def test_dry_run_changes_nothing_and_reports_added_names(tmp_path: Path) -> None:
    stage, new = _stage(tmp_path), _new_pyd(tmp_path, _sha(NEW_BYTES))
    plan = swap.plan_swap(stage, new, _sha(OLD_BYTES), tmp_path / "scratch", _probe(OLD_NAMES, NEW_NAMES))
    assert plan["replaced"] is False and plan["added_names"] == ["read_board_first_pass"]
    assert plan["old_sha256"] == _sha(OLD_BYTES) and plan["new_sha256"] == _sha(NEW_BYTES)
    assert swap.find_stage_pyd(stage).read_bytes() == OLD_BYTES


def test_apply_backs_up_old_and_verifies_placed_sha(tmp_path: Path) -> None:
    stage, new = _stage(tmp_path), _new_pyd(tmp_path)
    plan = swap.plan_swap(stage, new, _sha(OLD_BYTES), tmp_path / "scratch", _probe(OLD_NAMES, NEW_NAMES))
    receipt = swap.apply_swap(plan, tmp_path / "backup")
    assert receipt["replaced"] is True and receipt["placed_sha256"] == _sha(NEW_BYTES)
    assert swap.find_stage_pyd(stage).read_bytes() == NEW_BYTES
    assert Path(receipt["backup"]).read_bytes() == OLD_BYTES


def test_unexpected_old_version_is_refused_by_name(tmp_path: Path) -> None:
    stage, new = _stage(tmp_path, data=b"someone-elses-binary"), _new_pyd(tmp_path)
    with pytest.raises(swap.SwapError, match="想定の旧版ではありません"):
        swap.plan_swap(stage, new, _sha(OLD_BYTES), tmp_path / "scratch", _probe(OLD_NAMES, NEW_NAMES))


def test_recorded_sha_mismatch_is_refused(tmp_path: Path) -> None:
    stage, new = _stage(tmp_path), _new_pyd(tmp_path, "0" * 64)
    with pytest.raises(swap.SwapError, match=swap.SHA_FILE_NAME):
        swap.plan_swap(stage, new, _sha(OLD_BYTES), tmp_path / "scratch", _probe(OLD_NAMES, NEW_NAMES))


def test_missing_old_names_are_refused(tmp_path: Path) -> None:
    stage, new = _stage(tmp_path), _new_pyd(tmp_path)
    with pytest.raises(swap.SwapError, match="新 pyd に無い名前"):
        swap.plan_swap(stage, new, _sha(OLD_BYTES), tmp_path / "scratch", _probe(OLD_NAMES, ["classify_cells_hsv"]))


def test_stage_with_two_pyd_is_refused(tmp_path: Path) -> None:
    with pytest.raises(swap.SwapError, match="ちょうど 1 つではありません"):
        swap.find_stage_pyd(_stage(tmp_path, extra_pyd=True))


def test_main_requires_backup_dir_for_apply(tmp_path: Path) -> None:
    code = swap.main(["--stage", str(tmp_path), "--new-pyd", str(tmp_path / "x"), "--receipt",
                      str(tmp_path / "r.json"), "--apply"])
    assert code == 2


def test_main_dry_run_exit_code_on_refusal(tmp_path: Path) -> None:
    stage, new = _stage(tmp_path, data=b"unexpected"), _new_pyd(tmp_path)
    code = swap.main(["--stage", str(stage), "--new-pyd", str(new), "--receipt", str(tmp_path / "r.json")])
    assert code == 1 and not (tmp_path / "r.json").exists()

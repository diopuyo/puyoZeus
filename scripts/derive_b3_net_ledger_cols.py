"""既存の B3 学習行 (parquet) に案A の列 (正味の予告) を末尾へ足し、別のファイルへ書く。行は作り直さない。

入力は sl_pending・sl_inflight・ta_i_sl・ta_q_*・mk_* を持つ M3 の行 (sl_inflight 列が無い行は導出できず例外)。
配信の行生成と同じ純関数 (src/b3_net_ledger.py の derive_net_columns) を通すので、同じ行なら配信と同じ値になる。
元の列・行・順序は 1 つも変えない (元ファイルも書き換えない。出力先が入力と同じなら例外)。

使い方 (PC2 など。1 動画 = 1 parquet、隣の <動画>.manifest.json があれば写して導出の記録を足す):
    python -m scripts.derive_b3_net_ledger_cols --input <parquet か、parquet のあるフォルダ> --out-dir <出力フォルダ>
    (--input は複数可。既に出力があれば --force が要る。--dry-run は書かずに母数だけ表示)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.b3_column_spec import NET_LEDGER_COLUMN_NAMES
from src.b3_net_ledger import SOURCE_COLUMNS, derive_frame

PARQUET_SUFFIX = ".parquet"
MANIFEST_SUFFIX = ".manifest.json"
DONE_SUFFIX = ".done"
MANIFEST_KEY = "derived_net_ledger_cols"
SHA_CHUNK_BYTES = 1 << 20
PARQUET_COMPRESSION = "zstd"     # scripts/b3_generate_rows.py と同じ


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(SHA_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_inputs(paths: Sequence[Path]) -> list[Path]:
    """ファイルはそのまま、フォルダは直下の *.parquet を名前順に集める。1 つも無ければ例外。"""
    out: list[Path] = []
    for p in paths:
        out += sorted(p.glob(f"*{PARQUET_SUFFIX}")) if p.is_dir() else [p]
    missing = [str(p) for p in out if not p.is_file()]
    if missing or not out:
        raise FileNotFoundError(f"入力の parquet が無い: {missing or paths}")
    return out


def count_report(derived: pd.DataFrame) -> dict[str, Any]:
    """母数つきの記録。件数だけでは『合格』と『測っていない』が区別できないので有限数も書く。"""
    n = len(derived)
    finite = {c: int(derived[c].notna().sum()) for c in NET_LEDGER_COLUMN_NAMES if not c.startswith("mk_")}
    masks = {c: {str(k): int(v) for k, v in derived[c].value_counts(dropna=False).items()}
             for c in NET_LEDGER_COLUMN_NAMES if c.startswith("mk_") and not c.endswith("_p2")}
    return {"rows": n, "finite_counts": finite, "mask_counts": masks}


def changed_rows(df: pd.DataFrame, derived: pd.DataFrame) -> dict[str, int]:
    """net が確定の pending と違う行 = 案A が情報を足した行 (両側 inflight>0 の行を含む)。"""
    pend = df["sl_pending_p1"].to_numpy(dtype=np.float64, na_value=np.nan)
    net = derived["net_pending_p1"].to_numpy(dtype=np.float64, na_value=np.nan)
    inflight = (df["sl_inflight_p1"].to_numpy(dtype=np.float64, na_value=np.nan) > 0) | (
        df["sl_inflight_p2"].to_numpy(dtype=np.float64, na_value=np.nan) > 0)
    return {"rows_inflight_positive": int(inflight.sum()), "rows_net_p1_differs_from_pending": int((np.isfinite(net) & (net != pend)).sum())}


def derive_one(src: Path, out_dir: Path, force: bool, dry_run: bool) -> dict[str, Any]:
    """1 ファイルを導出して out_dir/<同名> へ書く。元の列を保ち、案A の列だけを末尾に足す。"""
    dst = out_dir / src.name
    if dst.resolve() == src.resolve():
        raise ValueError(f"出力先が入力と同じ: {src}")
    if dst.exists() and not force and not dry_run:
        raise FileExistsError(f"出力が既にある (上書きは --force): {dst}")
    df = pd.read_parquet(src)
    derived = derive_frame(df)
    report = count_report(derived)
    report.update(changed_rows(df, derived), source=src.name, source_columns=len(df.columns),
                  columns_added=len(derived.columns))
    if dry_run:
        return report
    out = pd.concat([df, derived], axis=1)
    if list(out.columns[:len(df.columns)]) != list(df.columns):
        raise AssertionError("元の列の順序が変わった")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    out.to_parquet(tmp, engine="pyarrow", compression=PARQUET_COMPRESSION, index=False)
    tmp.replace(dst)
    write_manifest(src, dst, report)
    return report


def write_manifest(src: Path, dst: Path, report: dict[str, Any]) -> None:
    """隣の manifest があれば写し (元のキーは変えない)、導出の記録を 1 キー足す。.done も置く。"""
    src_manifest = src.with_suffix(MANIFEST_SUFFIX)
    manifest = json.loads(src_manifest.read_text(encoding="utf-8")) if src_manifest.exists() else {}
    manifest[MANIFEST_KEY] = {"source": src.name, "source_sha256": sha256_file(src),
                              "added_columns": list(NET_LEDGER_COLUMN_NAMES), "source_columns_used": list(SOURCE_COLUMNS),
                              "report": report}
    dst.with_suffix(MANIFEST_SUFFIX).write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    dst.with_suffix(DONE_SUFFIX).write_text("done\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, action="append", required=True, help="parquet か、その置き場のフォルダ (複数可)")
    ap.add_argument("--out-dir", type=Path, required=True, help="出力フォルダ (入力と別の場所)")
    ap.add_argument("--force", action="store_true", help="既にある出力を上書きする")
    ap.add_argument("--dry-run", action="store_true", help="書かずに母数だけ表示する")
    a = ap.parse_args(argv)
    total = 0
    for src in collect_inputs(a.input):
        r = derive_one(src, a.out_dir, a.force, a.dry_run)
        total += r["rows"]
        print(f"{src.name}: rows {r['rows']} inflight>0 {r['rows_inflight_positive']} "
              f"net!=pending {r['rows_net_p1_differs_from_pending']} +{r['columns_added']} cols"
              f"{' (dry-run)' if a.dry_run else ''}")
    print(f"done: {total} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""補正 queue で再学習したモデル + 提供側 NEXT 整列 (R3) を、本番構成の5記録で再生する (2026-10-01)。

本番構成 = run_e36b.options (現本番 e36b_on と同じ)。exev の診断 _diag_next_shift_replay_20260930 と同じ
差し替え方 (d5_runtime の固定版読込は行わず現 HEAD の証明器を使う) で、次の2点だけを変える。
- モデル: --model (既定 models/exchange_event_v3 = 本番)
- 整列: --align で ExchangeEventOverlay(queue_alignment=True) (既定OFFフラグ)
台の確認: --variant R0 (v3・整列なし) の出力が e36b_on とバイト一致すること。
記録は exev logs/pending_expiry/full/records を読む。実行 cwd は exev の logs を読める場所 (run.sh 参照)。
"""
from __future__ import annotations

import argparse
from pathlib import Path

EXEV = Path("/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer")
RECORDS = EXEV / "logs/pending_expiry/full/records"
WT = Path("/mnt/d/puyo_analyzer/wt_nextfix")
OUT_ROOT = WT / "logs/next_shift_train/replay"
VARIANTS = {
    "R0": ("models/exchange_event_v3", False),
    "N": ("models/exchange_event_v5", True),
    "N_noalign": ("models/exchange_event_v5", False),
    # 診断 (門の対象外): 新S′+旧G_fe/M0、旧S′+新G_fe/M0、旧モデル全部+整列 (exev 診断 R3 の再現)
    "A_snew": ("models/exchange_event_ab_snew_cold", True),
    "A_cnew": ("models/exchange_event_ab_sold_cnew", True),
    "R3": ("models/exchange_event_v3", True),
    # 診断: 再学習ゆらぎ (random_state=1)。元特徴+整列なし / T特徴+整列
    "S_orig_rs1": ("models/exchange_event_ab_orig_rs1", False),
    "S_T_rs1": ("models/exchange_event_ab_T_rs1", True),
    # 確認: overlay の整列観測を helper へ移した後も出力が同一か (N・R3 と同じ設定)
    "N_recheck": ("models/exchange_event_v5", True),
    "R3_recheck": ("models/exchange_event_v3", True),
}


def resolve(variant: str) -> tuple[str, bool]:
    """固定の構成表か、再判定の構成 J_<orig|T>_rs<k> (models/rejudge、補正側だけ整列ON)。"""
    if variant in VARIANTS:
        return VARIANTS[variant]
    _, kind, seed = variant.split("_", 2)
    if kind not in ("orig", "T") or not seed.startswith("rs"):
        raise ValueError(f"未知の構成: {variant}")
    return f"models/rejudge/{kind}_{seed}", kind == "T"


def run(variant: str, source: str) -> None:
    """run_e36.worker を出力先・モデル・整列フラグだけ差し替えて呼ぶ。"""
    from scripts import d5_runtime, run_e36, run_e36b
    from scripts import replay_exchange_event_20260926 as replay_module
    model, align = resolve(variant)
    out = OUT_ROOT / variant
    original_replay = replay_module.replay

    def replay(record: Path, dest: Path, model_dir: Path | None = None, *args, **kwargs) -> dict:
        return original_replay(record, dest, WT / model, *args, **kwargs)

    def install() -> None:
        d5_runtime.OUT = out / f"runtime_{source}"

    extra = dict(queue_alignment=True) if align else {}
    replay_module.replay = replay
    d5_runtime.install = install
    run_e36.OUT, run_e36.RECORDS = out, RECORDS
    run_e36.options = lambda: dict(run_e36b.options(), **extra)
    run_e36.worker(source)


def main() -> None:
    """1記録1プロセス。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    run(args.variant, args.source)


if __name__ == "__main__":
    main()

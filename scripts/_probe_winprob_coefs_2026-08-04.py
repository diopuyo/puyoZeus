"""最新レンダ (render_delta_winprob_demo) と同一の勝率モデルを学習し、
位相別の指標寄与度 (標準化済みLR係数の絶対値、base毎に 1p/2p/diff 3列を合算)
を出力する使い捨てプローブ (2026-08-04、user質問「寄与トップ5」回答用)。
"""
from pathlib import Path

from scripts.compute_exchange_delta_winprob import (
    BOARD_ONLY_INDICATOR_BASES,
    train_winprob_models,
)

LABELED_WIN_CSV = Path(
    "data/verify/win_eval_combined66_2026-07-29/labeled_win_combined66.csv"
)
TOP_N = 8


def main() -> None:
    models = train_winprob_models(LABELED_WIN_CSV)
    feat_names: list[str] = []
    for base in BOARD_ONLY_INDICATOR_BASES:
        feat_names.extend([f"{base}__1p", f"{base}__2p", f"{base}__diff"])
    for phase, model in models.items():
        coefs = model.lr.coef_[0]
        assert len(coefs) == len(feat_names), (len(coefs), len(feat_names))
        agg: dict[str, float] = {}
        diff_sign: dict[str, float] = {}
        for name, c in zip(feat_names, coefs):
            base, suffix = name.rsplit("__", 1)
            agg[base] = agg.get(base, 0.0) + abs(float(c))
            if suffix == "diff":
                diff_sign[base] = float(c)
        top = sorted(agg.items(), key=lambda kv: -kv[1])[:TOP_N]
        print(f"\n=== 位相 {phase} (OOF AUC={model.oof_auc:.3f}) 寄与トップ{TOP_N} ===")
        for rank, (base, v) in enumerate(top, 1):
            sign = "+" if diff_sign.get(base, 0.0) >= 0 else "-"
            print(f"  {rank}. {base}  寄与={v:.3f}  diff係数符号={sign}")


if __name__ == "__main__":
    main()

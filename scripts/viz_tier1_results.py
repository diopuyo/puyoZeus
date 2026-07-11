"""tier1 相関 / win 分析の結果を人が見てわかりやすい図に出力する。

出力 (data/indicators_v2/study/):
  - tier1_corr_heatmap.png : 指標どうしの相関 (冗長性が一目で分かる)
  - tier1_win_signal.png    : 指標 × 位相 の win 相関 (どこに信号があるか)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CSV_PATH = "data/indicators_v2/study/labeled_win.csv"
OUT_DIR = Path("data/indicators_v2/study")
FONT_PATH = "/mnt/c/Windows/Fonts/meiryo.ttc"

# 指標: (列名, 日本語ラベル, ファミリー)
INDICATORS: tuple[tuple[str, str, str], ...] = (
    ("tsumo_count_rate", "手数進行度", "進行"),
    ("board_puyo_total", "盤面ぷよ総数", "充填"),
    ("board_color_puyo_total", "色ぷよ総数", "充填"),
    ("margin_time_rate", "マージン時間", "進行"),
    ("max_column_height", "最大列高", "充填"),
    ("column_bumpiness", "列の凸凹", "充填"),
    ("death_margin", "窒息余裕", "危険"),
    ("death_margin_neighbor", "窒息余裕(近傍)", "危険"),
    ("current_max_chain", "現在最大連鎖", "連鎖規模"),
    ("immediate_fire_power", "即時火力", "連鎖規模"),
    ("reach_fire_power", "到達火力", "火力"),
    ("chain_efficiency", "連鎖効率", "連鎖規模"),
    ("min_puyos_to_ignite", "発火最小ぷよ", "火力"),
    ("conn_pair_count", "2連結数", "形"),
    ("conn_triple_count", "3連結数", "形"),
    ("conn_max_group_size", "最大連結", "形"),
    ("second_chain_potential", "2本目連鎖潜在", "火力"),
    ("ojama_net_balance", "お邪魔純収支", "お邪魔"),
    ("ojama_forecast", "お邪魔予告", "お邪魔"),
    ("board_ojama_count", "盤面お邪魔数", "お邪魔"),
    ("chain_duration_sec", "連鎖所要時間", "テンポ"),
    ("dig_resistance", "掘り耐性", "受け"),
    ("absorption_capacity", "吸収余地", "受け"),
)


def _setup_font() -> None:
    """meiryo を matplotlib に登録。"""
    if Path(FONT_PATH).exists():
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def _draw_corr_heatmap(df: pd.DataFrame, cols: list[str], labels: list[str]) -> None:
    """指標どうしの相関ヒートマップ。"""
    corr = df[cols].corr(method="pearson").values
    n = len(cols)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = corr[i, j]
            if abs(v) >= 0.85 and i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="black", fontweight="bold")
    ax.set_title("① 指標どうしの相関 (|r|≥0.85 を数値表示 = 冗長ペア)\n"
                 "赤=正の相関 / 青=負の相関", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tier1_corr_heatmap.png", dpi=130)
    plt.close(fig)


def _win_corr_by_phase(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """指標 × 位相 (overall/序盤/中盤/終盤) の win 相関。"""
    lab = df.dropna(subset=["won"])
    q1, q2 = lab["tsumo"].quantile([1 / 3, 2 / 3]).tolist()
    groups = {
        "overall": lab,
        "序盤": lab[lab["tsumo"] <= q1],
        "中盤": lab[(lab["tsumo"] > q1) & (lab["tsumo"] <= q2)],
        "終盤": lab[lab["tsumo"] > q2],
    }
    out = {}
    for name, g in groups.items():
        out[name] = {c: (np.corrcoef(g[c].fillna(g[c].mean()), g["won"])[0, 1]
                         if g[c].std() > 0 else np.nan) for c in cols}
    return pd.DataFrame(out)


def _draw_win_signal(sig: pd.DataFrame, labels_map: dict[str, str]) -> None:
    """指標 × 位相 の win 相関ヒートマップ (終盤 |r| 降順)。"""
    sig = sig.reindex(sig["終盤"].abs().sort_values(ascending=False).index)
    rows = [labels_map[c] for c in sig.index]
    data = sig[["overall", "序盤", "中盤", "終盤"]].values
    fig, ax = plt.subplots(figsize=(7, 11))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-0.25, vmax=0.25, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(["全体", "序盤", "中盤", "終盤"], fontsize=11)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=9)
    for i in range(len(rows)):
        for j in range(4):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.15 else "black")
    ax.set_title("② 各指標と『勝ち』の相関 (位相別)\n"
                 "赤=勝ちに効く / 青=負けに効く(お邪魔で埋まる等)・信号は終盤に集中", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.5, label="win との相関")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tier1_win_signal.png", dpi=130)
    plt.close(fig)


def main() -> None:
    _setup_font()
    df = pd.read_csv(CSV_PATH)
    cols = [c for c, _, _ in INDICATORS if c in df.columns]
    labels = [jp for c, jp, _ in INDICATORS if c in df.columns]
    labels_map = {c: jp for c, jp, _ in INDICATORS}
    _draw_corr_heatmap(df, cols, labels)
    sig = _win_corr_by_phase(df, cols)
    _draw_win_signal(sig, labels_map)
    print("出力:", OUT_DIR / "tier1_corr_heatmap.png")
    print("出力:", OUT_DIR / "tier1_win_signal.png")


if __name__ == "__main__":
    main()

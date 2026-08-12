"""動画ごとの「片側の勝ち偏り」を測る (実力差バイアスの規模の実測、2026-08-12)。

light63 CSV から試合単位の勝敗を復元し、動画ごとに 1P 側の勝率を計算。
同一動画=同一ペアの連戦 (おいうリーグ30本先取形式) を仮定。
|勝率-0.5| の分布が実力差の存在量の下限推定になる。
"""
import pandas as pd

CSV = "data/verify/npz_light_smoke_2026-08-12/labeled_win_light63.csv"
df = pd.read_csv(CSV, usecols=["video_id", "game_idx", "side", "won"])
games = df.drop_duplicates(["video_id", "game_idx", "side"])
p1 = games[games["side"] == "1P"]
by_video = p1.groupby("video_id").agg(
    n_games=("won", "size"), p1_win_rate=("won", "mean"))
by_video = by_video[by_video["n_games"] >= 10]  # 少数試合の動画は除外
by_video["imbalance"] = (by_video["p1_win_rate"] - 0.5).abs()
print(f"対象動画: {len(by_video)}本 (10試合以上)")
print(f"試合数中央値: {by_video['n_games'].median():.0f}")
print("\n片側の勝ち偏り |1P勝率-0.5| の分布:")
print(by_video["imbalance"].describe().round(3))
print("\n偏りの内訳:")
for lo, hi, label in [(0, 0.05, "ほぼ互角 (45-55%)"),
                      (0.05, 0.15, "やや偏り (55-65%)"),
                      (0.15, 0.25, "明確な偏り (65-75%)"),
                      (0.25, 1.0, "大差 (75%超)")]:
    n = ((by_video["imbalance"] >= lo) & (by_video["imbalance"] < hi)).sum()
    print(f"  {label}: {n}本")

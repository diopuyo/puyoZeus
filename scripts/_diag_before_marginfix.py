"""修正前 exchange_labels.csv の game_idx 別 t_sec / 火力列 分布を確認する診断スクリプト。"""
import pandas as pd

df = pd.read_csv("data/indicators_v2/exchange_labels.csv")
print("shape:", df.shape)
print(df["game_idx"].value_counts().sort_index())
print()
print("=== game_idx==0 ===")
print(df[df["game_idx"] == 0][["t_sec", "fire_potential_fire_power", "fire_immediate_fire_power", "fire_honsen_output"]].describe())
print()
print("=== game_idx>=1 ===")
print(df[df["game_idx"] >= 1][["t_sec", "fire_potential_fire_power", "fire_immediate_fire_power", "fire_honsen_output"]].describe())

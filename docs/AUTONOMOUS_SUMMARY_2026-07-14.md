# 自律セッション 朝レビュー ガイド (2026-07-14, SAKI vs あん firepower quality)

6時間自律運転(03:07〜)で「火力モデル品質向上」を実施。要点と成果物を
レビューしやすい順に示す。詳細は `docs/ADVANTAGE_OVERLAY_2026-07-13.md` §6 と
memory `project_autonomous_firepower_quality_2026-07-14`。

## 出発点(ユーザー指摘)
v29 gameB で「SAKIが撃ったが足りない/あんは6万点で勝った」のに有利不利が
SAKI寄りに出る。真因を追い、火力の扱いを作り直した。

## 何をしたか(結論)
有利不利を **4成分ブレンド**に:
```
有利不利 = 0.35×圧力(お邪魔着弾) + 0.30×得点リード(累積攻撃)
         + 0.20×現モデル(tier1盤面) + 0.15×threat(reach到達火力)
```
- **真因**: 旧threatは `potential_fire_power`(理想2個追加=浅く過小評価)を使い、
  あんの大連鎖(reach=956/実816)を potential=360と見誤っていた。さらに発火
  (score急上昇)→お邪魔着弾に~10秒ラグがあり静止指標で捉えられない。
- **M1**: threat を reach_fire_power(実際に撃てる火力)へ。
- **M2**: 得点リード信号(スコア差=どちらが多く攻撃を通したか)で着弾ラグを橋渡し。

## 見てほしい成果物(順に)
1. `data/indicators_v2/overlay/timeline_v29_M2lead.png` — gameB。t=38以降ずっと2P一貫、
   決着-85、旧版の誤1P surge消滅(=修正後)。
2. `data/indicators_v2/overlay/timeline_v29_gameA.png` — 別ゲーム(あん勝ち)→2Pで終了。
3. `data/indicators_v2/overlay/timeline_v30_gen.png` — 別動画(light vs あん, light勝ち)→1P。
4. `data/indicators_v2/overlay/components_v29_gameB.png` — 成分分解(どの成分が効くか)。
5. `data/indicators_v2/overlay/advantage_v29_M2_h264.mp4` — 改良版オーバーレイ動画(gameB)。

## 検証結果
- **汎化 3ゲーム/2動画/3組合せ 全て勝者を正しく再現**(v29B/v29A=あん→2P、v30=light→1P)。
- **フルテスト 2638 passed / 9 skipped** = プロジェクト全体 working state。
- (多ゲーム勝者一致率ハーネス `validate_advantage_winner.py` の結果は別途追記)

## ロールバック点(PR#18 ブランチ feat/indicators-tier1-refine-2026-07-11)
各コミットは working state。`git log --oneline` で確認・任意点へ `git reset` 可:
- 3302b41 基点(3成分threat)/ b32037f M1(reach化)/ c800cc9 M2(得点リード)
- 813060c テスト / 6c7f86f doc / e775a79 成分分解プロット

## 残課題(相談したい)
- 重み(0.35/0.30/0.20/0.15)の調整、より多ゲーム/動画での検証。
- 発火→着弾ラグは得点リードで橋渡し中(盤面系だけでは未着弾の窓を捉えられない)。
- モデル本体の exchange 軽視、オーバーレイの処理速度(認識律速で1試合~30分)。
- potential_fire_power は reach より浅く過小評価(有利不利には reach採用で回避済、tier2指標として存置)。

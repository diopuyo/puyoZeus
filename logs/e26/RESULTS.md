# E26検収結果

|条件|q log loss|zenchi一致|誤発火|
|---|---:|---:|---:|
|E22基準|0.511078842|6910/8333 (82.9233%)|1/28|
|交換独立台帳|0.513721392|6963/8333 (83.5593%)|1/29|
|色得点・静穏条件|0.511078814|6910/8333 (82.9233%)|2/30|
|複数着弾|0.510470095|6910/8333 (82.9233%)|3/37|
|landing-state-safety単独|0.525722658|6968/8333 (83.6193%)|1/29|
|起点補完単独|0.524093179|6915/8333 (82.9833%)|2/30|

q母数は全条件6,526行・4試合。zenchiは8,333行。reviewは誤発火母数へ重複算入しない。
q悪化条件: 交換独立台帳, landing-state-safety単独, 起点補完単独
組合せ: color, multi + midchain_completion

## 途中予測の件数（記録再生3動画）

{
  "three_videos": {
    "candidates": 15,
    "accepted": 8,
    "accepted_chains": 7,
    "next_mismatch": 1,
    "final_mismatch": 0,
    "final_unresolved": 1
  },
  "sources": {
    "q_7gc4TgFig": {
      "candidates": 9,
      "accepted": 4,
      "accepted_chains": 3,
      "next_mismatch": 0,
      "final_mismatch": 0,
      "final_unresolved": 0
    },
    "fcXG83vInDY": {
      "candidates": 6,
      "accepted": 4,
      "accepted_chains": 4,
      "next_mismatch": 1,
      "final_mismatch": 0,
      "final_unresolved": 1
    },
    "mia8KCjr52g": {
      "candidates": 0,
      "accepted": 0,
      "accepted_chains": 0,
      "next_mismatch": 0,
      "final_mismatch": 0,
      "final_unresolved": 0
    },
    "zenchi": {
      "candidates": 5,
      "accepted": 2,
      "accepted_chains": 1,
      "next_mismatch": 1,
      "final_mismatch": 0,
      "final_unresolved": 0
    },
    "review": {
      "candidates": 0,
      "accepted": 0,
      "accepted_chains": 0,
      "next_mismatch": 0,
      "final_mismatch": 0,
      "final_unresolved": 0
    }
  }
}

## 固定基準の判定

{
  "variant": "on",
  "q": {
    "frames": 6526,
    "matches": 4,
    "log_loss": 0.5107778391393154,
    "auc": 0.7624758285957381
  },
  "zenchi": {
    "hits": 7079,
    "frames": 8333,
    "agreement": 0.8495139805592223
  },
  "deaths": {
    "false": 3,
    "total": 37,
    "unlabelled": 0
  },
  "scenes": {
    "prefire_frames": 62,
    "prefire_mean": 0.2702919012720453,
    "margin_sign_flips": 3,
    "terminal_frames": 45,
    "terminal_min": 1.0
  },
  "gates": {
    "q": true,
    "zenchi": true,
    "deaths": false,
    "scene": false
  },
  "candidate": false,
  "scene": {
    "first_sec": 2771.25,
    "deadline_sec": 2766.0,
    "scenes": 1,
    "baseline_first_sec": 2771.25
  },
  "midchain_three_videos": {
    "candidates": 15,
    "accepted": 8,
    "accepted_chains": 7,
    "next_mismatch": 1,
    "final_mismatch": 0,
    "final_unresolved": 1
  }
}

不合格。採用候補はE22のまま。レビュー動画は生成しない。

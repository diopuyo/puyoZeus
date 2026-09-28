# E29 検収

E27構成＋--midchain-single-observation。新フラグ既定OFF。production_config.py変更なし。
通常・隠し段とも1観測で候補化し、次段式一致まで確率・死亡へ公開しない。

固定条件・母数・全合否：on/METRICS.json。採用後不一致・未確認の全件：PREDICTION_EXCEPTIONS.json。
通常予測と隠し段上限の件数：PREDICTION_COUNTS.json。場面の式一致と死亡判定：SCENE_AUDIT.json。
OFF_*.jsonは5入力のE27出力とのバイト一致。入力・モデル不変はVERIFICATION.json。

指定場面では2755.450秒に隠し段候補の次段一致が成立し、最大打ち返し75440点を確認。
2760.117秒には純受け171個で死亡探索へ進むが、20000ノード上限を超えて証明未完了。
表示2P≤5%は2771.250秒のまま。今回は観測数だけを変更し、探索上限は変更していない。

3動画の通常予測では、q動画649.133秒の予測77380点に対し最終63620点。
次段式は一致したが後続式で撤回された。撤回済み採用も最終不一致へ含めている。
隠し段は死亡専用の最大上限であり、確率・表示用の予測とは分けて集計する。
最終得点未確定は正解へ含めない。次段一致だけでは完走得点の一致を保証できなかった。

{
  "variant": "on",
  "q": {
    "frames": 6526,
    "matches": 4,
    "log_loss": 0.5110474659023649,
    "auc": 0.7623878446842248
  },
  "zenchi": {
    "hits": 7081,
    "frames": 8333,
    "agreement": 0.8497539901596064
  },
  "deaths": {
    "false": 1,
    "total": 36,
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
    "deaths": true,
    "scene": false,
    "final_scores": false
  },
  "candidate": false,
  "scene": {
    "first_sec": 2771.25,
    "deadline_sec": 2766.0,
    "scenes": 1,
    "baseline_first_sec": 2771.25
  },
  "prediction_counts": {
    "midchain": {
      "accepted": 20,
      "accepted_chains": 16,
      "next_mismatch": 6,
      "final_mismatch": 5,
      "final_unresolved": 2
    },
    "hidden": {
      "accepted": 5,
      "accepted_chains": 5,
      "next_mismatch": 9,
      "final_mismatch": 3,
      "final_unresolved": 0
    }
  }
}

不合格。採用候補はE27を維持。条件未達のためE29レビュー動画は生成しない。

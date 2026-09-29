# E27検収結果

|項目|実測・母数|事前登録基準|判定|
|---|---|---|---|
|q log loss|0.510777839（6526フレーム・4試合）|≤.513079|合格|
|zenchi一致|7081/8333＝84.9754%|≥82.62%|合格|
|誤発火|1/37、未判定0|≤1件、≤1/28|合格|
|2P≤5%初到達|2771.25秒（1場面）|≤2766.0秒|不合格|

## 新規・前倒し死亡（3動画、E22比）

|動画|試合|交換|側|種別|初時刻|実死亡|誤り|
|---|---:|---:|---|---|---:|---|---|
|q_7gc4TgFig|9|29|1P|new|656.067|True|False|
|q_7gc4TgFig|14|40|1P|earlier|883.967|False|True|
|fcXG83vInDY|3|11|2P|new|366.000|True|False|
|fcXG83vInDY|4|12|2P|new|435.600|True|False|
|fcXG83vInDY|5|15|2P|new|536.033|True|False|
|fcXG83vInDY|6|18|1P|new|626.733|True|False|

## 隠し段組合せ

{
  "three_videos": {
    "enumerated": 26,
    "combinations": 490,
    "accepted": 2,
    "next_mismatch": 1,
    "used": 0,
    "uses": 0
  },
  "sources": {
    "q_7gc4TgFig": {
      "enumerated": 18,
      "combinations": 430,
      "accepted": 0,
      "next_mismatch": 1,
      "used": 0,
      "uses": 0
    },
    "fcXG83vInDY": {
      "enumerated": 8,
      "combinations": 60,
      "accepted": 2,
      "next_mismatch": 0,
      "used": 0,
      "uses": 0
    },
    "mia8KCjr52g": {
      "enumerated": 0,
      "combinations": 0,
      "accepted": 0,
      "next_mismatch": 0,
      "used": 0,
      "uses": 0
    },
    "zenchi": {
      "enumerated": 22,
      "combinations": 470,
      "accepted": 4,
      "next_mismatch": 2,
      "used": 0,
      "uses": 0
    },
    "review": {
      "enumerated": 13,
      "combinations": 365,
      "accepted": 2,
      "next_mismatch": 1,
      "used": 0,
      "uses": 0
    }
  },
  "death_input_usage": {
    "three_videos": {
      "exchanges": 2,
      "evaluations": 5
    },
    "sources": {
      "q_7gc4TgFig": {
        "exchanges": 0,
        "evaluations": 0,
        "rows": []
      },
      "fcXG83vInDY": {
        "exchanges": 2,
        "evaluations": 5,
        "rows": [
          {
            "game": 3,
            "exchange": 9,
            "side": 2,
            "first_sec": 342.93333333333334,
            "maximum_score": 1600,
            "death_incoming": 0
          },
          {
            "game": 5,
            "exchange": 15,
            "side": 2,
            "first_sec": 524.9666666666667,
            "maximum_score": 2360,
            "death_incoming": 0
          }
        ]
      },
      "mia8KCjr52g": {
        "exchanges": 0,
        "evaluations": 0,
        "rows": []
      },
      "zenchi": {
        "exchanges": 4,
        "evaluations": 59,
        "rows": [
          {
            "game": 2,
            "exchange": 8,
            "side": 1,
            "first_sec": 2682.7166666666667,
            "maximum_score": 1000,
            "death_incoming": 0
          },
          {
            "game": 3,
            "exchange": 11,
            "side": 1,
            "first_sec": 2732.15,
            "maximum_score": 1000,
            "death_incoming": 0
          },
          {
            "game": 11,
            "exchange": 36,
            "side": 2,
            "first_sec": 3089.016666666667,
            "maximum_score": 103580,
            "death_incoming": 0
          },
          {
            "game": 13,
            "exchange": 43,
            "side": 1,
            "first_sec": 3204.016666666667,
            "maximum_score": 10680,
            "death_incoming": 0
          }
        ]
      },
      "review": {
        "exchanges": 2,
        "evaluations": 7,
        "rows": [
          {
            "game": 2,
            "exchange": 8,
            "side": 1,
            "first_sec": 2682.7166666666667,
            "maximum_score": 1000,
            "death_incoming": 0
          },
          {
            "game": 3,
            "exchange": 11,
            "side": 1,
            "first_sec": 2732.15,
            "maximum_score": 1000,
            "death_incoming": 0
          }
        ]
      }
    }
  }
}

不合格。採用候補はE22を維持。指定レビュー動画は生成しない。

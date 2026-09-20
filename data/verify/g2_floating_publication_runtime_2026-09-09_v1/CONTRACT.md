# 合成実走の境界

旧先行着地修復は34620..34702の2Pのみ、全域観測を維持。追加するのは実Publisherと、その採録差分を保存検証する新prefix契約。旧run・認識snapshot・本番source/設定は変更しない。実updateを各一回、同updateのcontextを使用。未知会計はUNKNOWNのまま、品質/本番/学習許可はfalse。

実公開hold・fresh復帰は既存 private current release/consistency gateの固定sourceを使い、実collectorの旧flagsは変更しない。新finishは元finishのC6前byte比較一条件だけを変更し、raw/NEXT/非collector/他streamは厳密に維持、採録6core+kind/timeと列集合を照合。唯一画像タグは未再生、pre-C6共通frameの旧値照合は維持。全metadata品質合格ではない。

原保存一回、finally完了後ENGINE、実childexit、排他finalizerを維持。失敗pubv2の終了印は作らない。新runの前に合成CPU・prepare・独立prefixレビューを完了する。G2全体の品質検査は実公開9/old56排除、正常復帰/採録損失、長時間/複数動画を別途要する。

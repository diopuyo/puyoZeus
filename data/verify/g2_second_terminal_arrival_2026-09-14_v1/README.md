# G2: 到来・消費通知・終端着弾を分離する限定修復

本番採用ではなく、video38の既知不具合を検収する診断runtime用の修復です。最新実走は `g2_second_prefix_runtime_2026-09-14_v40`。動画完走・修復箇所通過はユーザー受入済みですが、終了I/O障害による保存監査未完を残してDraft PRとしています。詳細は `docs/G2_REPAIR_REVIEW_2026-09-14.md` を参照してください。

## 変更の要点

元NEXTへの到来と、元Nativeが認証した消費通知（ACK）、盤面への物理反映を別々に記録します。元writerの成功通知から到来を複写し、原J・原FIFOには書き戻しません。

従来の「ACK済みツモだけを列挙する」候補域がゼロ支持になった場合に限り、元の同call STABLE資格と連続予告画像・自己発火/相殺不在を検査します。到来済み最大2手と初回30個着弾の順序を列挙し、全初回着弾候補が終端かつ観測一致盤面が一意の場合だけ、原Registryへ一回反映します。未ACK tokenは保持し、後のACKでは物理反映を繰り返しません。

量30は特定frameの例外ではなく、予告陽性を使った観測後の条件付き仮説です。較正済み認識信頼度・勝率・将来着弾保証ではありません。非終端、既存おじゃま、未観測発火、2手超の回復は今回の新経路では拒否します。

## 保存と検収

- `SECOND_ENQUEUE_SOURCE`：原writer成功通知と原Jの対応。到来だけでは物理反映を認証しません。
- `SECOND_ARRIVAL`：到来/ACK/反映の別台帳。
- `SECOND_WARNING`：元Nativeの同call画像/側/世代/時刻に結ばれた予告条件。
- `SECOND_TERMINAL`：準備状態、一回反映、未ACK、後のACK、終了時actual currentとplanned。
- 旧 `SECOND_PREFIX` は履歴として保持します。終了通知は旧 `SECOND_SETTLED_NOTICES` と新終端票へ同一packetを保存します。
- `terminal_saved.py` は原J・旧prefix数学から再計算し、新終端票と照合します。A35のM1検査は、この再計算したpending列だけを元資格検査へ渡します。

終了失敗や欠測を成功へ補完しません。保存失敗は元の異常終了に伝播し、共有streamは所有元Sessionが閉じます。CPU対照の人工Native資格・画像条件と、実動画の証明は分けています。

## 小さな再現可能テスト

動画・モデル・凍結snapshotを使わない契約対照は次で実行できます（Pythonとpytestが必要です）。

```sh
python -m pytest data/verify/g2_second_terminal_arrival_2026-09-14_v1/test_portable_arrival_contract.py -q
```

到来2/ACK1/反映2→遅いACK2でも再反映なし、異なるscope・欠測・FIFO不一致拒否、予告条件と将来保証の分離を検査します。これは原Native認証・着弾数学・実動画合格の代わりではありません。

他の統合テストはローカルの旧保存票、凍結認識snapshot、既存モデルに依存します。同じ元cold-loader fixtureを使う異なるテストモジュールは別pytestプロセスで実行します。生動画・巨大JSONL・モデル・バイナリはPRに含めず、実走の固定版・検収要点と区別して保持します。

G2の全必須条件を閉鎖してからPRを提出し、G3は既存6動画計画・比較版・GT/分母・遅延許容・開始条件の文書準備で停止します。自動merge・G3実走・本番採用は行いません。

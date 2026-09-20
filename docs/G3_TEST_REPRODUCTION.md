# G3テストの再実行条件

PR #27修復では、既存の隔離テスト22本と追加分離1本・G3コード・そのCPUテストと、
参照されるG2凍結fixtureコード・固定snapshotのPythonコードを収録した。
凍結元は書き換えず、そのまま複製している。
モデル・動画・実走ログをGitへ追加するものではない。

## 実行

Python 3.12の既存WSL環境を用いる。隔離23本は`tests/g3_isolated_files.txt`。
通常群と23個の独立プロセスを合算するc案を、実終了コード・ログ・JUnit付きで再現する。
追加した `test_g3_prediction_funnel.py` は、全sys.modulesの反復走査が
他ファイルとの同居で遅くなるため分離した。テスト本文と判定条件は変更していない。

```bash
PYTHON=/path/to/venv/bin/python bash scripts/run_g3_partitioned_tests.sh \
  /mnt/d/puyo_analyzer/verify/新規run名 full
```

`isolated`を指定すると23本だけを実行する。全pytest成功とは扱わない。
必須ファイル欠落・未収集・プロセス失敗は非ゼロ終了。既存出力先は拒否する。
通常のpytestから恒久除外したり、外部資産が無い試験をskipへ変更したりしない。

## 外部資産

G3試験の一部は保存済みG2/G3原票・モデルを必要とする。Git cloneだけでは全件の実行条件は揃わない。
以下は既存環境のまま保持し、テスト内のSHA照合を維持する。

- `data/verify/g3_repair_2026-09-15_v1/` のモデルレジストリ、および
  `D:/puyo_analyzer/verify/g3_repair_2026-09-15_v1/` の現行実走原票。
- テストが明示する `data/verify/g2_*/` の保存済みJSON原票・モデル参照。
- G2凍結入口が参照するスナップショットのモデル・データ・保存原票。

不足時は元実行の正規資産を必要パスへ用意する。空fixture・現在SHAへの打ち直し・別runの代用はしない。
原票内の絶対パスを検証する試験は、元のworkspaceパスで実行する必要がある。
隔離worktreeへ原票をコピーするだけでは、その来歴検査を満たさない。
Formal100/hidden reserveは開かない。モデル出力を使う人工fixtureは動画GTではない。

Windowsの`core.autocrlf`によるSHAずれを防ぐため、Python/シェルと隔離リストは原則LF、
固定SHAの8ソースは元のCRLF/混在改行を保持する属性を追加した。
`data/verify/`の凍結資産と通知PowerShellのBOMも元のバイト列を保持する。
これは改行を含むコード同一性の保全であり、SHA検査を緩める変更ではない。

今回の隔離checkoutはprivate mount namespace内で元の絶対パスへ割り当て、
正規資産をread-onlyで参照した。元workspaceは書き換えていない。
実行スクリプト・原票・入力同一性の確認は
`D:/puyo_analyzer/verify/pr27_ready_2026-09-20_v1/` に保存している。
通常群の中断前の成功を保持し、未実行IDを継続した手順と最終集計も同所にある。

## 合格の範囲

旧ローカル全pytest結果は当時の未追跡コード・外部資産を含む。
提出版の全pytest再実行やG3動画品質の合格へ読み替えない。
2026-09-20の修復試験・未完は`docs/agent_coordination/PR27_REPAIR_2026-09-20.md`へ記録する。

# G3テストの再実行条件

PR #27修復では、既存の隔離テスト22本・G3コード・そのCPUテストと、
参照されるG2凍結fixtureコード・固定snapshotのPythonコードを収録した。
凍結元は書き換えず、そのまま複製している。
モデル・動画・実走ログをGitへ追加するものではない。

## 実行

Python 3.12の既存WSL環境を用いる。元の隔離22本は`tests/g3_isolated_files.txt`。
通常群と22個の独立プロセスを合算する既存c案を、実終了コード・ログ・JUnit付きで再現する。

```bash
PYTHON=/path/to/venv/bin/python bash scripts/run_g3_partitioned_tests.sh \
  /mnt/d/puyo_analyzer/verify/新規run名 full
```

`isolated`を指定すると22本だけを実行する。全pytest成功とは扱わない。
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

Windowsの`core.autocrlf`によるSHAずれを防ぐため、Python/シェルはLF、
`data/verify/`の凍結資産はGit blobのバイト列を維持する属性を追加した。
これは改行を含むコード同一性の保全であり、SHA検査を緩める変更ではない。

## 合格の範囲

旧ローカル全pytest結果は当時の未追跡コード・外部資産を含む。
提出版の全pytest再実行やG3動画品質の合格へ読み替えない。
2026-09-20の修復試験・未完は`docs/agent_coordination/PR27_REPAIR_2026-09-20.md`へ記録する。

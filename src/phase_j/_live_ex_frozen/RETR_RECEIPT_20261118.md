# 返しの見積もり修正後の受領SHA更新（2026-11-18）

ユーザーの再開指示に基づき、SOURCE.json の adapter_sha256 4件だけを更新した。
0.6.8では120/120件が一致。以降の意図した返しの見積もり修正で現物が変わったが、
受領記録が未更新だったため。計算コード・原器の出自SHA・他116件は変更していない。

- 照合基点: `777f853cb1e81bb5aa16d753259bbd74acd3a179`
- 修正後: `0ff892cbf13eee928f3fe83e6871a21fb07345ea`
- 比較: ファイルのCRLFをLFへ正規化してSHA256を計算。
- 検査: 更新後120/120件一致。既存の受領検査の条件は変更していない。

## src/dl1k_regen_state.py

- 旧期待SHA: `30b177f25967321662d9fd19fb38b8ef239f0acf262c6949350e9eb1635dd204`
- 新期待SHA: `5a61a1c74e203c1b9c2c412256d0ebb63313436d0e2974f18a6e6a62494f5b1d`

`git log --format="%H %s" 777f853c..0ff892cb -- <対象ファイル>` の結果:

- d64514a1431206ef5052831d16eefeebd8a09908 生NEXT二組に一意一致するツモ位置を既定OFFで同期

## src/dl1k_engine.py

- 旧期待SHA: `635c84f6635ca41bf7f395a844a5d0bf5fef63f2457cbc604f61d66c7f20eaf7`
- 新期待SHA: `0c8f83741c4b1f12dcde89f0bbe99bf3ac8eb5261de7a8f93c9d20985f38fc4b`

`git log --format="%H %s" 777f853c..0ff892cb -- <対象ファイル>` の結果:

- d64514a1431206ef5052831d16eefeebd8a09908 生NEXT二組に一意一致するツモ位置を既定OFFで同期
- a12b19f2bad1d16ee3120c8c18f82a4e2ba8970b 共通連鎖IDの最初の得点台帳基点を既定OFFで保持

## src/dl1k_records.py

- 旧期待SHA: `7ac8653fb4cc623ecc673a5cea26a594666fe5e29b159b3c4ad3c4caf173b125`
- 新期待SHA: `719f9a266b9613717863ebf55c148b9a27cc14adc2d29df94bdd87b8e935c430`

`git log --format="%H %s" 777f853c..0ff892cb -- <対象ファイル>` の結果:

- 0ff892cbf13eee928f3fe83e6871a21fb07345ea 落下開始の予告差分と生NEXT履歴の同期を既定OFFで追加
- d64514a1431206ef5052831d16eefeebd8a09908 生NEXT二組に一意一致するツモ位置を既定OFFで同期

## src/dl1k_retr.py

- 旧期待SHA: `fbcd39e8866cf0b37dc3fe7c9dca01aace4b91a8f21e91251f2ee74bbcc14d3f`
- 新期待SHA: `8dfa348466c776d9e1d24e72e1792ea65917dd5a1d056cc0b8128e685eb88135`

`git log --format="%H %s" 777f853c..0ff892cb -- <対象ファイル>` の結果:

- 0ff892cbf13eee928f3fe83e6871a21fb07345ea 落下開始の予告差分と生NEXT履歴の同期を既定OFFで追加
- d64514a1431206ef5052831d16eefeebd8a09908 生NEXT二組に一意一致するツモ位置を既定OFFで同期
- a12b19f2bad1d16ee3120c8c18f82a4e2ba8970b 共通連鎖IDの最初の得点台帳基点を既定OFFで保持
- 4375614a3f9a054f4e6da5152ab1504df089ddf4 小連鎖の消去と設置待ち後の返し探索を既定OFFで追加
- 5776530f7de3635eff9cd33d13363068de204eba 受け側の長い共通連鎖予測を既定OFFで返しの相殺へ反映
- c536f09f94c567fdbd8021fa19a553630274a946 返しの既発火分と長い攻撃の締切を既定OFFで補正
- 8cfc711aa8cf16b8dcd99fbbb5372f2ae0a2f6b1 fix: treat missing chain observations as empty
- 2f05ac06b2e9ce0f228e19ecbb575b6300f2f5ca fix: 研究用の外部Amaを候補で共有し問い合わせを直列化
- 7f432d47c1945208c326b9f380e51490c6ed9048 feat: 研究用の返し探索に外部プロセスの強制指定を追加
- b57a4413fc12bf2d96d4f68de15a80fdc0b8671f feat: 返し探索に確定済み予告の落下と相殺を追加
- 69ef1c95c64fdb03249ee5a7ab2a890bd3ccfdf1 feat: 見えている手の返し列を既定OFFで配信に追加

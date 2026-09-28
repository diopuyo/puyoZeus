// 診断CSVをスプレッドシートの読取器でも検査し、日本語と空欄を保持する。
import fs from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const root = path.join(process.env.USERPROFILE, '.cache/codex-runtimes/codex-primary-runtime/dependencies/node');
const require = createRequire(path.join(root, 'package.json'));
const { Workbook } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const csv = await fs.readFile('logs/e28/SCENE.csv', 'utf8');
const workbook = await Workbook.fromCSV(csv.replace(/^\uFEFF/, ''), { sheetName: '判定経路' });
const sheet = workbook.worksheets.getItemAt(0);
const rows = sheet.getUsedRange().values;
if (rows.length !== 481) throw new Error(`観測母数が不一致: ${rows.length}`);
const headers = rows[0];
const unknownIndex = headers.indexOf('p2_maximum_extra_counter_ojama');
if (rows.slice(1).some(row => row[unknownIndex] !== '' && row[unknownIndex] !== null)) {
  throw new Error('未算出の最大応手が数値に置換されています');
}
const result = await workbook.inspect({ kind: 'region', sheetId: sheet.name,
  range: 'I1:M4', maxChars: 1800, tableMaxRows: 4, tableMaxCols: 5 });
await fs.writeFile('logs/e28/CSV_IMPORT_CHECK.json', JSON.stringify({
  dataRows: rows.length-1, columns: headers.length, blankNotZero: true,
  inspection: result.ndjson,
}, null, 2));
console.log(`CSV読取確認: ${rows.length-1}行、${headers.length}列、日本語・未算出空欄を保持`);

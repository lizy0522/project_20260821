import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const sourcePath = process.argv[2];
const outputPath = process.argv[3];
const previewPath = process.argv[4];

if (!sourcePath || !outputPath || !previewPath) {
  throw new Error("Usage: node export_type3_cluster_compressed_xlsx.mjs <source.json> <output.xlsx> <preview.png>");
}

const payload = JSON.parse(await fs.readFile(sourcePath, "utf8"));
if (!Array.isArray(payload.columns) || payload.columns.length !== 21 || !Array.isArray(payload.rows) || payload.rows.length !== 425) {
  throw new Error(`Unexpected Type-III source shape: columns=${payload.columns?.length}, rows=${payload.rows?.length}`);
}

const columns = payload.columns;
const rows = payload.rows.map((row) => columns.map((column) => row[column] ?? null));
const matrix = [columns, ...rows];

function columnName(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

const lastColumn = columnName(columns.length - 1);
const lastRow = matrix.length;
const tableRange = `A1:${lastColumn}${lastRow}`;
const outputDir = path.dirname(outputPath);
await fs.mkdir(outputDir, { recursive: true });

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("per_state");
sheet.showGridLines = false;
sheet.tabColor = "#1F4E78";

const used = sheet.getRange(tableRange);
used.values = matrix;
used.format.font = { name: "Arial", size: 10, color: "#1F2937" };
used.format.verticalAlignment = "center";

const header = sheet.getRange(`A1:${lastColumn}1`);
header.format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
};
header.format.rowHeight = 34;

const body = sheet.getRange(`A2:${lastColumn}${lastRow}`);
body.format.horizontalAlignment = "right";
body.format.rowHeight = 18;

for (const columnIndex of [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]) {
  sheet.getRange(`${columnName(columnIndex)}2:${columnName(columnIndex)}${lastRow}`).format.numberFormat = "0";
}
for (const columnIndex of [5, 6, 7, 8, 9, 10, 16, 17]) {
  sheet.getRange(`${columnName(columnIndex)}2:${columnName(columnIndex)}${lastRow}`).format.numberFormat = "0.000000";
}
for (const columnIndex of [18, 19, 20]) {
  sheet.getRange(`${columnName(columnIndex)}2:${columnName(columnIndex)}${lastRow}`).format.horizontalAlignment = "center";
}

const table = sheet.tables.add(tableRange, true, "Type3PerStateResults");
table.style = "TableStyleMedium2";
table.showHeaders = true;
table.showTotals = false;
table.showBandedColumns = false;
table.showFilterButton = true;

sheet.freezePanes.freezeRows(1);
sheet.freezePanes.freezeColumns(1);

const widths = {
  A: 11, B: 10, C: 10, D: 10, E: 10,
  F: 18, G: 21, H: 21, I: 19, J: 20, K: 18,
  L: 15, M: 23, N: 18, O: 20, P: 16, Q: 24, R: 24,
  S: 14, T: 16, U: 19,
};
for (const [column, width] of Object.entries(widths)) {
  sheet.getRange(`${column}1:${column}${lastRow}`).format.columnWidth = width;
}

workbook.recalculate();

const inspect = await workbook.inspect({
  kind: "sheet,table",
  sheetId: "per_state",
  range: "A1:U6",
  include: "values,formulas",
  tableMaxRows: 6,
  tableMaxCols: 21,
  maxChars: 12000,
});
console.log(inspect.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "Type-III final formula error scan",
});
console.log(errors.ndjson);

const preview = await workbook.render({ sheetName: "per_state", range: "A1:U35", scale: 1, format: "png" });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);

const imported = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const importedSheets = await imported.inspect({ kind: "sheet", include: "id,name", maxChars: 3000 });
console.log(importedSheets.ndjson);
const importedRegion = await imported.inspect({
  kind: "table",
  sheetId: "per_state",
  range: "A1:U426",
  include: "values",
  tableMaxRows: 2,
  tableMaxCols: 21,
  maxChars: 5000,
});
console.log(importedRegion.ndjson);
console.log(`Exported ${outputPath}`);

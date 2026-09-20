import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const sourcePath = process.argv[2];
const outputPath = process.argv[3];
const previewDir = process.argv[4];

if (!sourcePath || !outputPath || !previewDir) {
  throw new Error(
    "Usage: node export_frozen_centered17_type3_xlsx.mjs <source.json> <output.xlsx> <preview_dir>",
  );
}

const payload = JSON.parse(await fs.readFile(sourcePath, "utf8"));
const expectedSheets = ["state_results", "compressed_lut", "summary"];
for (const sheetName of expectedSheets) {
  if (!payload[sheetName] || !Array.isArray(payload[sheetName].columns) || !Array.isArray(payload[sheetName].rows)) {
    throw new Error(`Missing or invalid sheet payload: ${sheetName}`);
  }
}
if (payload.state_results.rows.length !== 425) {
  throw new Error(`state_results must have 425 rows, got ${payload.state_results.rows.length}`);
}
if (payload.compressed_lut.rows.length !== 77) {
  throw new Error(`compressed_lut must have 77 rows, got ${payload.compressed_lut.rows.length}`);
}

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

function addDataSheet(workbook, name, data, tableName, tabColor) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  sheet.tabColor = tabColor;
  const columns = data.columns;
  const rows = data.rows.map((row) => columns.map((column) => row[column] ?? null));
  const matrix = [columns, ...rows];
  const lastColumn = columnName(columns.length - 1);
  const lastRow = matrix.length;
  const range = sheet.getRange(`A1:${lastColumn}${lastRow}`);
  range.values = matrix;
  range.format.font = { name: "Arial", size: 9, color: "#1F2937" };
  range.format.verticalAlignment = "center";

  const header = sheet.getRange(`A1:${lastColumn}1`);
  header.format = {
    fill: "#1F4E78",
    font: { name: "Arial", size: 9, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  header.format.rowHeight = 34;
  const body = sheet.getRange(`A2:${lastColumn}${lastRow}`);
  body.format.rowHeight = 18;
  body.format.horizontalAlignment = "right";

  const table = sheet.tables.add(`A1:${lastColumn}${lastRow}`, true, tableName);
  table.style = "TableStyleMedium2";
  table.showHeaders = true;
  table.showTotals = false;
  table.showBandedColumns = false;
  table.showFilterButton = true;

  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(name === "state_results" ? 2 : 1);
  for (let index = 0; index < columns.length; index += 1) {
    const column = columnName(index);
    const label = columns[index];
    let width = 14;
    if (label.includes("config") || label.includes("member") || label.includes("source")) width = 36;
    if (label.includes("NMSE") || label.includes("CNMSE") || label.includes("condition")) width = 22;
    if (label.includes("fingerprint") || label.includes("representative")) width = 24;
    if (label.includes("Y_Aend") || label.includes("Y_C2")) width = 23;
    sheet.getRange(`${column}1:${column}${lastRow}`).format.columnWidth = width;
    if (
      label.includes("NMSE") ||
      label.includes("CNMSE") ||
      label.includes("dBc") ||
      label.includes("condition") ||
      label.includes("coefficient_norm")
    ) {
      sheet.getRange(`${column}2:${column}${lastRow}`).format.numberFormat = "0.000000";
    }
  }
  return sheet;
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
const workbook = Workbook.create();
const stateSheet = addDataSheet(
  workbook,
  "state_results",
  payload.state_results,
  "FrozenCentered17StateResults",
  "#1F4E78",
);
const lutSheet = addDataSheet(
  workbook,
  "compressed_lut",
  payload.compressed_lut,
  "FrozenCentered17CompressedLUT",
  "#5B9BD5",
);

const summarySheet = workbook.worksheets.add("summary");
summarySheet.showGridLines = false;
summarySheet.tabColor = "#70AD47";
const summaryRows = payload.summary.rows.map((row) => [row.metric, row.value]);
const summaryMatrix = [["Metric", "Value"], ...summaryRows];
summarySheet.getRange(`A1:B${summaryMatrix.length}`).values = summaryMatrix;
summarySheet.getRange("A1:B1").format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
summarySheet.getRange(`A2:A${summaryMatrix.length}`).format = {
  font: { name: "Arial", size: 10, bold: true, color: "#1F2937" },
};
summarySheet.getRange(`B2:B${summaryMatrix.length}`).format = {
  font: { name: "Arial", size: 10, color: "#1F2937" },
  wrapText: true,
  verticalAlignment: "center",
};
summarySheet.getRange(`A1:B${summaryMatrix.length}`).format.verticalAlignment = "center";
summarySheet.getRange(`A1:A${summaryMatrix.length}`).format.columnWidth = 42;
summarySheet.getRange(`B1:B${summaryMatrix.length}`).format.columnWidth = 76;
summarySheet.getRange(`A2:B${summaryMatrix.length}`).format.autofitRows();
summarySheet.freezePanes.freezeRows(1);
const summaryTable = summarySheet.tables.add(
  `A1:B${summaryMatrix.length}`,
  true,
  "FrozenCentered17Summary",
);
summaryTable.style = "TableStyleMedium4";
summaryTable.showFilterButton = false;

workbook.recalculate();

for (const [sheetName, range] of [
  ["state_results", `A1:${columnName(payload.state_results.columns.length - 1)}28`],
  ["compressed_lut", `A1:${columnName(payload.compressed_lut.columns.length - 1)}28`],
  ["summary", `A1:B${summaryMatrix.length}`],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(
    path.join(previewDir, `${sheetName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const inspect = await workbook.inspect({
  kind: "sheet,table",
  include: "id,name,values,formulas",
  tableMaxRows: 4,
  tableMaxCols: 10,
  maxChars: 12000,
});
console.log(inspect.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "Frozen-centered17 workbook formula error scan",
});
console.log(errors.ndjson);

const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);
const imported = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const importedSheets = await imported.inspect({ kind: "sheet", include: "id,name", maxChars: 3000 });
console.log(importedSheets.ndjson);
const importedState = await imported.inspect({
  kind: "table",
  sheetId: "state_results",
  range: `A1:${columnName(payload.state_results.columns.length - 1)}426`,
  include: "values",
  tableMaxRows: 2,
  tableMaxCols: 10,
  maxChars: 5000,
});
console.log(importedState.ndjson);
console.log(`Exported ${outputPath}`);

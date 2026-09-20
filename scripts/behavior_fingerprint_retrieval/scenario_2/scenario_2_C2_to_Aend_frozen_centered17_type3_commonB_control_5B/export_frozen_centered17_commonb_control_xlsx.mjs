import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const sourcePath = process.argv[2];
const outputPath = process.argv[3];
const previewDir = process.argv[4];
if (!sourcePath || !outputPath || !previewDir) {
  throw new Error(
    "Usage: node export_frozen_centered17_commonb_control_xlsx.mjs <source.json> <output.xlsx> <preview_dir>",
  );
}
const payload = JSON.parse(await fs.readFile(sourcePath, "utf8"));
const required = ["commonB_state_results", "comparison", "compressed_lut", "probe_diagnostics", "summary"];
for (const sheet of required) {
  if (!payload[sheet] || !Array.isArray(payload[sheet].columns) || !Array.isArray(payload[sheet].rows)) {
    throw new Error(`Invalid sheet payload: ${sheet}`);
  }
}
if (payload.commonB_state_results.rows.length !== 425) throw new Error("commonB_state_results must contain 425 rows");
if (payload.compressed_lut.rows.length !== 77) throw new Error("compressed_lut must contain 77 rows");

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

function addTableSheet(workbook, name, data, tableName, tabColor, freezeColumns = 1) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  sheet.tabColor = tabColor;
  const matrix = [data.columns, ...data.rows.map((row) => data.columns.map((column) => row[column] ?? null))];
  const lastColumn = columnName(data.columns.length - 1);
  const lastRow = matrix.length;
  const used = sheet.getRange(`A1:${lastColumn}${lastRow}`);
  used.values = matrix;
  used.format.font = { name: "Arial", size: 9, color: "#1F2937" };
  used.format.verticalAlignment = "center";
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
  sheet.freezePanes.freezeColumns(freezeColumns);
  for (let index = 0; index < data.columns.length; index += 1) {
    const column = columnName(index);
    const label = data.columns[index];
    let width = 14;
    if (label.includes("config") || label.includes("member") || label.includes("source")) width = 38;
    if (label.includes("NMSE") || label.includes("CNMSE") || label.includes("margin") || label.includes("distance")) width = 23;
    if (label.includes("State") || label.includes("cluster")) width = 18;
    sheet.getRange(`${column}1:${column}${lastRow}`).format.columnWidth = width;
    if (label.includes("NMSE") || label.includes("CNMSE") || label.includes("dB") || label.includes("margin") || label.includes("rank")) {
      sheet.getRange(`${column}2:${column}${lastRow}`).format.numberFormat = "0.000000";
    }
  }
  return sheet;
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });
const workbook = Workbook.create();
addTableSheet(workbook, "commonB_state_results", payload.commonB_state_results, "CommonBStateResults", "#1F4E78", 2);
addTableSheet(workbook, "comparison", payload.comparison, "CommonBComparison", "#5B9BD5", 1);
addTableSheet(workbook, "compressed_lut", payload.compressed_lut, "CommonBCompressedLUT", "#70AD47", 1);
addTableSheet(workbook, "probe_diagnostics", payload.probe_diagnostics, "CommonBProbeDiagnostics", "#ED7D31", 1);

const summarySheet = workbook.worksheets.add("summary");
summarySheet.showGridLines = false;
summarySheet.tabColor = "#A5A5A5";
const summaryRows = payload.summary.rows.map((row) => [row.metric, row.value]);
const summaryMatrix = [["Metric", "Value"], ...summaryRows];
summarySheet.getRange(`A1:B${summaryMatrix.length}`).values = summaryMatrix;
summarySheet.getRange("A1:B1").format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
summarySheet.getRange(`A2:A${summaryMatrix.length}`).format = { font: { name: "Arial", size: 10, bold: true, color: "#1F2937" } };
summarySheet.getRange(`B2:B${summaryMatrix.length}`).format = { font: { name: "Arial", size: 10, color: "#1F2937" }, wrapText: true };
summarySheet.getRange(`A1:B${summaryMatrix.length}`).format.verticalAlignment = "center";
summarySheet.getRange(`A1:A${summaryMatrix.length}`).format.columnWidth = 44;
summarySheet.getRange(`B1:B${summaryMatrix.length}`).format.columnWidth = 78;
summarySheet.getRange(`A2:B${summaryMatrix.length}`).format.autofitRows();
summarySheet.freezePanes.freezeRows(1);
const summaryTable = summarySheet.tables.add(`A1:B${summaryMatrix.length}`, true, "CommonBSummary");
summaryTable.style = "TableStyleMedium4";
summaryTable.showFilterButton = false;

workbook.recalculate();
for (const [sheetName, range] of [
  ["commonB_state_results", `A1:${columnName(payload.commonB_state_results.columns.length - 1)}28`],
  ["comparison", `A1:${columnName(payload.comparison.columns.length - 1)}28`],
  ["compressed_lut", `A1:${columnName(payload.compressed_lut.columns.length - 1)}28`],
  ["probe_diagnostics", `A1:${columnName(payload.probe_diagnostics.columns.length - 1)}28`],
  ["summary", `A1:B${summaryMatrix.length}`],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(path.join(previewDir, `${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const inspect = await workbook.inspect({ kind: "sheet,table", include: "id,name,values,formulas", tableMaxRows: 4, tableMaxCols: 10, maxChars: 12000 });
console.log(inspect.ndjson);
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "Common-B control workbook formula error scan" });
console.log(errors.ndjson);
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);
const imported = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
console.log((await imported.inspect({ kind: "sheet", include: "id,name", maxChars: 5000 })).ndjson);
console.log((await imported.inspect({ kind: "table", sheetId: "commonB_state_results", range: `A1:${columnName(payload.commonB_state_results.columns.length - 1)}426`, include: "values", tableMaxRows: 2, tableMaxCols: 10, maxChars: 5000 })).ndjson);
console.log(`Exported ${outputPath}`);

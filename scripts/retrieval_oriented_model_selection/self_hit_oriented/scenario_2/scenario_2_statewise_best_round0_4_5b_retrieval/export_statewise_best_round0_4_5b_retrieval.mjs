import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const MAIN_COLUMNS = [
  "状态序号",
  "负载配置",
  "NMSE(NMSE_WITHOUTDPD)",
  "上下边带ACPR平均值withoutdpd",
  "A段建模精度",
  "AB段泛化精度",
  "C段建模精度",
  "CB段泛化精度",
  "命中的序号",
  "命中的CNMSE",
];

function assertCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function finiteNumber(value, name) {
  const number = Number(value);
  assertCondition(Number.isFinite(number), `${name} must be finite, got ${value}`);
  return number;
}

function normaliseRows(sheet) {
  const columns = sheet.columns;
  return sheet.rows.map((row, index) => {
    if (Array.isArray(row)) {
      assertCondition(row.length === columns.length, `${columns.length}-column row ${index} mismatch`);
      return row;
    }
    return columns.map((column) => (Object.prototype.hasOwnProperty.call(row, column) ? row[column] : null));
  });
}

function validateSource(source) {
  assertCondition(source.sheets && source.sheets.retrieval_results, "retrieval_results sheet missing");
  const main = source.sheets.retrieval_results;
  assertCondition(JSON.stringify(main.columns) === JSON.stringify(MAIN_COLUMNS), "main columns mismatch");
  const mainRows = normaliseRows(main);
  assertCondition(mainRows.length === 425, `main sheet must contain 425 rows, got ${mainRows.length}`);
  mainRows.forEach((row, index) => {
    assertCondition(finiteNumber(row[0], `state_id row ${index}`) === index, `state_id row ${index} is not ${index}`);
    assertCondition(typeof row[1] === "string" && row[1].length > 0, `load config row ${index} is empty`);
    [2, 3, 4, 5, 6, 7].forEach((column) => finiteNumber(row[column], `${MAIN_COLUMNS[column]} row ${index}`));
    finiteNumber(row[8], `state_id_Q row ${index}`);
    if (row[9] !== "-Inf") finiteNumber(row[9], `retrieved CNMSE row ${index}`);
  });
  const exactCount = mainRows.filter((row) => row[9] === "-Inf").length;
  const selected = source.sheets.selected_models;
  const common = source.sheets.common_support_diagnostics;
  const retrieval = source.sheets.retrieval_diagnostics;
  const summary = source.sheets.summary;
  assertCondition(selected && normaliseRows(selected).length === 850, "selected_models must contain 850 rows");
  assertCondition(common && normaliseRows(common).length === 425, "common diagnostics must contain 425 rows");
  assertCondition(retrieval && normaliseRows(retrieval).length === 425, "retrieval diagnostics must contain 425 rows");
  assertCondition(summary && normaliseRows(summary).length > 0, "summary must not be empty");
  return { mainRows, exactCount };
}

function applyFormatting(sheet, columns, rowCount, tableName) {
  const lastRow = rowCount + 1;
  const lastCol = String.fromCharCode(64 + columns.length);
  const all = sheet.getRange(`A1:${lastCol}${lastRow}`);
  all.format.font = { name: "Arial", size: 9, color: "#222222" };
  all.format.verticalAlignment = "center";
  const header = sheet.getRange(`A1:${lastCol}1`);
  header.format.fill = "#1F4E78";
  header.format.font = { name: "Arial", size: 9, bold: true, color: "#FFFFFF" };
  header.format.horizontalAlignment = "center";
  header.format.verticalAlignment = "center";
  header.format.wrapText = true;
  header.format.borders = { preset: "all", style: "thin", color: "#FFFFFF" };
  header.format.rowHeight = 34;
  sheet.getRange(`A2:${lastCol}${lastRow}`).format.rowHeight = 18;
  columns.forEach((column, index) => {
    let width = 20;
    if (column === "负载配置") width = 48;
    if (column.includes("memory") || column.includes("orders") || column.includes("definition")) width = 34;
    if (column === "metric" || column === "value") width = 30;
    sheet.getRangeByIndexes(0, index, lastRow, 1).format.columnWidth = width;
  });
  sheet.freezePanes.freezeRows(1);
  const table = sheet.tables.add(`A1:${lastCol}${lastRow}`, true, tableName);
  table.showFilterButton = true;
  sheet.showGridLines = false;
}

async function inspectWorkbook(workbook, expectedExactCount) {
  const sheetNames = [
    "retrieval_results",
    "selected_models",
    "common_support_diagnostics",
    "retrieval_diagnostics",
    "summary",
  ];
  for (const name of sheetNames) {
    const sheet = workbook.worksheets.getItem(name);
    const used = sheet.getUsedRange(true);
    assertCondition(used !== null, `${name} has no used range`);
    const inspect = await workbook.inspect({
      kind: "table",
      sheetId: name,
      range: name === "retrieval_results" ? "A1:J6" : "A1:F6",
      include: "values,formulas",
      tableMaxRows: 6,
      tableMaxCols: 10,
      tableMaxCellChars: 120,
    });
    console.log(`TABLE_INSPECT_${name}`);
    console.log(inspect.ndjson);
  }
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 300 },
    summary: "statewise-best retrieval formula error scan",
  });
  console.log("FORMULA_ERRORS");
  console.log(errors.ndjson);
  const errorRecords = (errors.ndjson || "")
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => { try { return JSON.parse(line); } catch { return { kind: "parse_error" }; } })
    .filter((record) => record.kind !== "notice");
  assertCondition(errorRecords.length === 0, "workbook contains formula errors");
  const main = workbook.worksheets.getItem("retrieval_results").getRange("A1:J426").values;
  assertCondition(main.length === 426 && main[0].length === 10, "retrieval_results dimensions mismatch");
  assertCondition(JSON.stringify(main[0]) === JSON.stringify(MAIN_COLUMNS), "retrieval_results header mismatch");
  const exactCount = main.slice(1).filter((row) => row[9] === "-Inf").length;
  assertCondition(exactCount === expectedExactCount, `Exact count changed: ${exactCount} vs ${expectedExactCount}`);
  return { sheetCount: sheetNames.length, mainRows: 425, mainColumns: 10, exactCount };
}

async function main() {
  const [sourcePath, outputPath, previewDir] = process.argv.slice(2);
  if (!sourcePath || !outputPath) throw new Error("usage: node export...mjs <source.json> <output.xlsx> [preview_dir]");
  const source = JSON.parse(await fs.readFile(sourcePath, "utf8"));
  const sourceCheck = validateSource(source);
  const workbook = Workbook.create();
  const orderedNames = [
    "retrieval_results",
    "selected_models",
    "common_support_diagnostics",
    "retrieval_diagnostics",
    "summary",
  ];
  const tableNames = [
    "StatewiseRetrievalResults",
    "StatewiseSelectedModels",
    "CommonSupportDiagnostics",
    "RetrievalDiagnostics",
    "RetrievalSummary",
  ];
  orderedNames.forEach((name, index) => {
    const sheet = workbook.worksheets.add(name);
    const definition = source.sheets[name];
    const rows = normaliseRows(definition);
    sheet.getRangeByIndexes(0, 0, rows.length + 1, definition.columns.length).values = [definition.columns, ...rows];
    applyFormatting(sheet, definition.columns, rows.length, tableNames[index]);
    if (name === "retrieval_results") {
      sheet.getRange("A2:A426").format.numberFormat = "0";
      sheet.getRange("C2:H426").format.numberFormat = "0.000000";
      sheet.getRange("I2:I426").format.numberFormat = "0";
      sheet.getRange("J2:J426").format.numberFormat = "0.000000";
    }
  });
  workbook.recalculate();
  if (previewDir) {
    await fs.mkdir(previewDir, { recursive: true });
    for (const name of orderedNames) {
      const preview = await workbook.render({ sheetName: name, range: name === "retrieval_results" ? "A1:J25" : "A1:F25", scale: 1, format: "png" });
      await fs.writeFile(`${previewDir}/${name}.png`, new Uint8Array(await preview.arrayBuffer()));
    }
  }
  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(outputPath);
  const saved = await FileBlob.load(outputPath);
  const reloaded = await SpreadsheetFile.importXlsx(saved);
  const checked = await inspectWorkbook(reloaded, sourceCheck.exactCount);
  console.log(JSON.stringify({ operation: "created_and_reloaded", outputPath, ...checked }));
}

await main();

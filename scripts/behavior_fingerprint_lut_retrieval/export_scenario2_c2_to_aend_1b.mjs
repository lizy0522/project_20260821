import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const MAIN_COLUMNS = [
  "state_id_R",
  "load_config",
  "nmse_withoutdpd_dB",
  "acpr_withoutdpd_avg_dBc",
  "Y_Aend_train_NMSE_dB",
  "Y_Aend_B_NMSE_dB",
  "Y_C2_train_NMSE_dB",
  "Y_C2_B_NMSE_dB",
  "state_id_Q",
  "retrieved_real_B_CNMSE_dB",
];

const DIAGNOSTIC_COLUMNS = [
  "state_id_R",
  "state_id_Q",
  "state_id_delta_signed",
  "state_id_delta_abs",
  "query_top1_fingerprint_CNMSE_dB",
  "retrieved_real_B_CNMSE_dB",
  "exact_hit",
  "dpd_shareable",
  "ilc_A_end",
  "ilc_C_stage",
];

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index];
    if (item.startsWith("--")) {
      args[item.slice(2)] = argv[index + 1];
      index += 1;
    }
  }
  return args;
}

function assertCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function finiteNumber(value, name) {
  const number = Number(value);
  assertCondition(Number.isFinite(number), `${name} must be finite, got ${value}`);
  return number;
}

function validateMainRows(columns, rows) {
  assertCondition(JSON.stringify(columns) === JSON.stringify(MAIN_COLUMNS), "Main columns do not match the frozen 10-column contract");
  assertCondition(rows.length === 425, `Expected 425 main rows, got ${rows.length}`);
  rows.forEach((row, index) => {
    assertCondition(finiteNumber(row[0], `state_id_R row ${index + 2}`) === index, "state_id_R must be exactly 0...424");
    [2, 3, 4, 5, 6, 7].forEach((column) => finiteNumber(row[column], `${MAIN_COLUMNS[column]} row ${index + 2}`));
    finiteNumber(row[8], `state_id_Q row ${index + 2}`);
    assertCondition(row[9] === "-Inf" || Number.isFinite(Number(row[9])), `Invalid Real-B value at row ${index + 2}`);
  });
  const exactCount = rows.filter((row) => row[9] === "-Inf").length;
  const failureCount = rows.filter((row) => row[9] !== "-Inf" && Number(row[9]) >= -40).length;
  const shareableCount = rows.filter((row) => row[9] === "-Inf" || Number(row[9]) < -40).length;
  assertCondition(exactCount > 0, "1B output unexpectedly contains no exact self hits");
  assertCondition(failureCount + shareableCount === 425, "Shareability counts do not cover all states");
  return { exactCount, failureCount, shareableCount };
}

function validateDiagnosticsRows(columns, rows) {
  assertCondition(JSON.stringify(columns) === JSON.stringify(DIAGNOSTIC_COLUMNS), "Diagnostic columns do not match contract");
  assertCondition(rows.length === 425, `Expected 425 diagnostic rows, got ${rows.length}`);
  rows.forEach((row, index) => {
    assertCondition(finiteNumber(row[0], `diagnostic state_id_R row ${index + 2}`) === index, "Diagnostic state order mismatch");
    finiteNumber(row[1], `diagnostic state_id_Q row ${index + 2}`);
    finiteNumber(row[2], `diagnostic delta row ${index + 2}`);
    finiteNumber(row[3], `diagnostic abs delta row ${index + 2}`);
    finiteNumber(row[4], `diagnostic fingerprint CNMSE row ${index + 2}`);
    assertCondition(row[5] === "-Inf" || Number.isFinite(Number(row[5])), `Invalid diagnostic Real-B value at row ${index + 2}`);
  });
}

function asDisplayValue(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return value;
}

function styleTableSheet(sheet, rangeAddress, rowCount, columnWidths, tableName, numericRanges, mixedRanges) {
  const all = sheet.getRange(rangeAddress);
  all.format.font = { name: "Arial", size: 9, color: "#222222" };
  all.format.verticalAlignment = "center";
  const header = sheet.getRange(rangeAddress.split(":")[0].replace(/\d+$/, "1") + ":" + rangeAddress.split(":")[1].replace(/\d+$/, "1") );
  header.format.fill = "#1F4E78";
  header.format.font = { name: "Arial", size: 9, bold: true, color: "#FFFFFF" };
  header.format.horizontalAlignment = "center";
  header.format.verticalAlignment = "center";
  header.format.wrapText = true;
  header.format.borders = { preset: "all", style: "thin", color: "#FFFFFF" };
  sheet.getRangeByIndexes(0, 0, rowCount, columnWidths.length).format.columnWidth = 12;
  columnWidths.forEach((width, column) => {
    sheet.getRangeByIndexes(0, column, rowCount, 1).format.columnWidth = width;
  });
  sheet.getRangeByIndexes(0, 0, 1, columnWidths.length).format.rowHeight = 30;
  sheet.getRangeByIndexes(1, 0, Math.max(1, rowCount - 1), columnWidths.length).format.rowHeight = 18;
  sheet.freezePanes.freezeRows(1);
  sheet.tables.add(rangeAddress, true, tableName).showFilterButton = true;
  for (const [range, format] of numericRanges) sheet.getRange(range).format.numberFormat = format;
  for (const [range, format] of mixedRanges) sheet.getRange(range).format.numberFormat = format;
  sheet.showGridLines = false;
}

function createWorkbook(source) {
  const workbook = Workbook.create();
  const main = workbook.worksheets.add("retrieval_results");
  const diagnostics = workbook.worksheets.add("retrieval_diagnostics");
  const summary = workbook.worksheets.add("summary");
  const mainRows = [MAIN_COLUMNS, ...source.mainRows];
  const diagnosticRows = [DIAGNOSTIC_COLUMNS, ...source.diagnosticRows];
  const summaryRows = [["metric", "value"], ...source.summaryRows.map((row) => [row.metric, asDisplayValue(row.value)])];
  main.getRange(`A1:J${mainRows.length}`).values = mainRows;
  diagnostics.getRange(`A1:J${diagnosticRows.length}`).values = diagnosticRows;
  summary.getRange(`A1:B${summaryRows.length}`).values = summaryRows;

  styleTableSheet(
    main,
    `A1:J${mainRows.length}`,
    mainRows.length,
    [12, 52, 21, 27, 25, 22, 22, 22, 12, 30],
    "RetrievalResultsTable",
    [["A2:A426", "0"], ["C2:H426", "0.000000"], ["I2:I426", "0"], ["J2:J426", "0.000000"]],
    [],
  );
  main.getRange("J2:J426").conditionalFormats.add("cellIs", { operator: "lessThan", formula: -40, format: { fill: "#E2F0D9" } });
  main.getRange("J2:J426").conditionalFormats.add("cellIs", { operator: "greaterThanOrEqual", formula: -40, format: { fill: "#FCE4D6" } });
  main.getRange("J2:J426").conditionalFormats.add("containsText", { text: "-Inf", format: { fill: "#D9EAF7" } });

  styleTableSheet(
    diagnostics,
    `A1:J${diagnosticRows.length}`,
    diagnosticRows.length,
    [12, 12, 18, 18, 30, 30, 12, 14, 12, 12],
    "RetrievalDiagnosticsTable",
    [["A2:D426", "0"], ["E2:F426", "0.000000"], ["G2:H426", "General"], ["I2:J426", "0"]],
    [],
  );
  diagnostics.getRange("F2:F426").conditionalFormats.add("cellIs", { operator: "lessThan", formula: -40, format: { fill: "#E2F0D9" } });
  diagnostics.getRange("F2:F426").conditionalFormats.add("cellIs", { operator: "greaterThanOrEqual", formula: -40, format: { fill: "#FCE4D6" } });
  diagnostics.getRange("F2:F426").conditionalFormats.add("containsText", { text: "-Inf", format: { fill: "#D9EAF7" } });

  styleTableSheet(
    summary,
    `A1:B${summaryRows.length}`,
    summaryRows.length,
    [46, 68],
    "RetrievalSummaryTable",
    [],
    [],
  );
  summary.getRange(`B2:B${summaryRows.length}`).format.wrapText = true;
  summary.getRange(`A2:B${summaryRows.length}`).format.rowHeight = 20;
  const failureIdsRow = summaryRows.findIndex((row) => row[0] === "failure_state_ids") + 1;
  if (failureIdsRow > 1) summary.getRange(`A${failureIdsRow}:B${failureIdsRow}`).format.rowHeight = 38;
  return workbook;
}

async function inspectAndCheck(workbook, source, counts) {
  for (const [sheetName, range] of [["retrieval_results", "A1:J6"], ["retrieval_diagnostics", "A1:J6"], ["summary", "A1:B12"]]) {
    const inspect = await workbook.inspect({ kind: "table", sheetId: sheetName, range, include: "values,formulas", tableMaxRows: 12, tableMaxCols: 10, tableMaxCellChars: 120 });
    console.log(`TABLE_INSPECT_${sheetName}`);
    console.log(inspect.ndjson);
  }
  const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "1B workbook formula error scan" });
  console.log("FORMULA_ERRORS");
  console.log(errors.ndjson);
  const errorRecords = (errors.ndjson || "").trim().split("\n").filter(Boolean).map((line) => { try { return JSON.parse(line); } catch { return { kind: "parse_error" }; } }).filter((record) => record.kind !== "notice");
  assertCondition(errorRecords.length === 0, "1B workbook contains formula errors");
  const mainValues = workbook.worksheets.getItem("retrieval_results").getRange("A1:J426").values;
  assertCondition(mainValues.length === 426 && mainValues[0].length === 10, "retrieval_results shape mismatch after export");
  assertCondition(JSON.stringify(mainValues[0]) === JSON.stringify(MAIN_COLUMNS), "retrieval_results header mismatch after export");
  const reloadedCounts = validateMainRows(MAIN_COLUMNS, mainValues.slice(1));
  assertCondition(JSON.stringify(reloadedCounts) === JSON.stringify(counts), "Workbook counts differ from source");
  const diagnosticValues = workbook.worksheets.getItem("retrieval_diagnostics").getRange("A1:J426").values;
  assertCondition(diagnosticValues.length === 426 && diagnosticValues[0].length === 10, "retrieval_diagnostics shape mismatch after export");
  validateDiagnosticsRows(DIAGNOSTIC_COLUMNS, diagnosticValues.slice(1));
  const summaryValues = workbook.worksheets.getItem("summary").getRange(`A1:B${source.summaryRows.length + 1}`).values;
  assertCondition(summaryValues.length === source.summaryRows.length + 1, "summary row count mismatch after export");
  return { rows: mainValues.length, columns: mainValues[0].length, ...reloadedCounts, diagnostics_rows: diagnosticValues.length, summary_rows: summaryValues.length };
}

async function savePreviews(workbook, previewDir) {
  if (!previewDir) return;
  await fs.mkdir(previewDir, { recursive: true });
  const ranges = [["retrieval_results", "A1:J25", "retrieval_results_preview.png"], ["retrieval_diagnostics", "A1:J25", "retrieval_diagnostics_preview.png"], ["summary", "A1:B25", "summary_preview.png"]];
  for (const [sheetName, range, filename] of ranges) {
    const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
    await fs.writeFile(`${previewDir}/${filename}`, new Uint8Array(await preview.arrayBuffer()));
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.source || !args.output) throw new Error("--source and --output are required");
  const source = JSON.parse(await fs.readFile(args.source, "utf8"));
  const counts = validateMainRows(MAIN_COLUMNS, source.mainRows);
  validateDiagnosticsRows(DIAGNOSTIC_COLUMNS, source.diagnosticRows);
  const workbook = createWorkbook(source);
  workbook.recalculate();
  await savePreviews(workbook, args.preview_dir);
  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(args.output);
  const saved = await FileBlob.load(args.output);
  const reloaded = await SpreadsheetFile.importXlsx(saved);
  const checked = await inspectAndCheck(reloaded, source, counts);
  console.log(JSON.stringify({ operation: "created_and_reloaded", output: args.output, ...checked }));
}

await main();

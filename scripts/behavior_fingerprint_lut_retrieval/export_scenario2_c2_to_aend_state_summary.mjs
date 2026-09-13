import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const COLUMNS = [
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

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index];
    if (item === "--verify-only") {
      args.verifyOnly = true;
      continue;
    }
    if (item.startsWith("--")) {
      args[item.slice(2)] = argv[index + 1];
      index += 1;
    }
  }
  return args;
}

function assertCondition(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function finiteNumber(value, name) {
  const number = Number(value);
  assertCondition(Number.isFinite(number), `${name} must be finite, got ${value}`);
  return number;
}

function validateRows(columns, rows) {
  assertCondition(
    JSON.stringify(columns) === JSON.stringify(COLUMNS),
    "Excel columns do not match the frozen 10-column contract",
  );
  assertCondition(rows.length === 425, `Expected 425 data rows, got ${rows.length}`);
  const ids = rows.map((row) => finiteNumber(row[0], "state_id_R"));
  assertCondition(
    JSON.stringify(ids) === JSON.stringify(Array.from({ length: 425 }, (_, i) => i)),
    "state_id_R must be exactly 0...424 in ascending order",
  );
  const exactCount = rows.filter((row) => row[9] === "-Inf").length;
  assertCondition(exactCount === 218, `Expected 218 -Inf values, got ${exactCount}`);
  const numericColumns = [2, 3, 4, 5, 6, 7];
  rows.forEach((row, index) => {
    numericColumns.forEach((column) => {
      finiteNumber(row[column], `${COLUMNS[column]} row ${index + 2}`);
    });
    finiteNumber(row[8], `state_id_Q row ${index + 2}`);
    if (row[9] !== "-Inf") {
      finiteNumber(row[9], `${COLUMNS[9]} row ${index + 2}`);
    }
  });
  const failureCount = rows.filter(
    (row) => row[9] !== "-Inf" && Number(row[9]) >= -40,
  ).length;
  const shareableCount = rows.filter(
    (row) => row[9] === "-Inf" || Number(row[9]) < -40,
  ).length;
  assertCondition(failureCount === 14, `Expected 14 failures, got ${failureCount}`);
  assertCondition(shareableCount === 411, `Expected 411 shareable rows, got ${shareableCount}`);
  const state340 = rows[340];
  assertCondition(Number(state340[8]) === 217, "state 340 must retrieve state 217");
  assertCondition(
    Math.abs(Number(state340[9]) - -37.10367236988057) < 1e-10,
    "state 340 Real-B CNMSE mismatch",
  );
  return { exactCount, failureCount, shareableCount };
}

function applySheetFormatting(sheet) {
  const all = sheet.getRange("A1:J426");
  all.format.font = { name: "Arial", size: 9, color: "#222222" };
  all.format.verticalAlignment = "center";
  const header = sheet.getRange("A1:J1");
  header.format.fill = "#1F4E78";
  header.format.font = { name: "Arial", size: 9, bold: true, color: "#FFFFFF" };
  header.format.horizontalAlignment = "center";
  header.format.verticalAlignment = "center";
  header.format.wrapText = true;
  header.format.borders = { preset: "all", style: "thin", color: "#FFFFFF" };
  sheet.getRange("A2:A426").format.numberFormat = "0";
  sheet.getRange("C2:H426").format.numberFormat = "0.000000";
  sheet.getRange("I2:I426").format.numberFormat = "0";
  sheet.getRange("J2:J426").format.numberFormat = "0.000000";
  sheet.getRange("A1:J1").format.rowHeight = 30;
  sheet.getRange("A2:J426").format.rowHeight = 18;
  const widths = [12, 48, 22, 26, 24, 22, 22, 20, 12, 30];
  widths.forEach((width, column) => {
    sheet.getRangeByIndexes(0, column, 426, 1).format.columnWidth = width;
  });
  sheet.freezePanes.freezeRows(1);
  const table = sheet.tables.add("A1:J426", true, "StateSummaryTable");
  table.showFilterButton = true;
  const cnmse = sheet.getRange("J2:J426");
  cnmse.conditionalFormats.add("cellIs", {
    operator: "lessThan",
    formula: -40,
    format: { fill: "#E2F0D9" },
  });
  cnmse.conditionalFormats.add("cellIs", {
    operator: "greaterThanOrEqual",
    formula: -40,
    format: { fill: "#FCE4D6" },
  });
  cnmse.conditionalFormats.add("containsText", {
    text: "-Inf",
    format: { fill: "#D9EAF7" },
  });
  sheet.showGridLines = false;
}

async function inspectAndCheck(workbook, expected) {
  const tableInspect = await workbook.inspect({
    kind: "table",
    sheetId: "state_summary",
    range: "A1:J6",
    include: "values,formulas",
    tableMaxRows: 6,
    tableMaxCols: 10,
    tableMaxCellChars: 120,
  });
  console.log("TABLE_INSPECT");
  console.log(tableInspect.ndjson);
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 300 },
    summary: "final formula error scan",
  });
  console.log("FORMULA_ERRORS");
  console.log(errors.ndjson);
  const errorRecords = (errors.ndjson || "")
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        return { kind: "parse_error", message: String(error) };
      }
    })
    .filter((record) => record.kind !== "notice");
  assertCondition(errorRecords.length === 0, "Workbook contains formula errors");
  const sheet = workbook.worksheets.getItem("state_summary");
  const values = sheet.getRange("A1:J426").values;
  assertCondition(values.length === 426, `Expected 426 total rows, got ${values.length}`);
  assertCondition(values[0].length === 10, `Expected 10 columns, got ${values[0].length}`);
  assertCondition(
    JSON.stringify(values[0]) === JSON.stringify(COLUMNS),
    "Saved workbook header does not match the frozen contract",
  );
  const dataRows = values.slice(1);
  const counts = validateRows(COLUMNS, dataRows);
  assertCondition(
    counts.exactCount === expected.exactCount &&
      counts.failureCount === expected.failureCount &&
      counts.shareableCount === expected.shareableCount,
    "Saved workbook counts differ from source counts",
  );
  return { rows: values.length, columns: values[0].length, ...counts };
}

async function createWorkbook(source, outputPath, previewPath) {
  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add("state_summary");
  sheet.getRange("A1:J426").values = [source.columns, ...source.rows];
  applySheetFormatting(sheet);
  const counts = validateRows(source.columns, source.rows);
  if (previewPath) {
    const preview = await workbook.render({
      sheetName: "state_summary",
      range: "A1:J25",
      scale: 1,
      format: "png",
    });
    await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
  }
  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(outputPath);
  const saved = await FileBlob.load(outputPath);
  const reloaded = await SpreadsheetFile.importXlsx(saved);
  const checked = await inspectAndCheck(reloaded, counts);
  console.log(JSON.stringify({ operation: "created_and_reloaded", outputPath, ...checked }));
}

async function verifyWorkbook(inputPath, previewPath) {
  const input = await FileBlob.load(inputPath);
  const workbook = await SpreadsheetFile.importXlsx(input);
  if (previewPath) {
    const preview = await workbook.render({
      sheetName: "state_summary",
      range: "A1:J25",
      scale: 1,
      format: "png",
    });
    await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
  }
  const checked = await inspectAndCheck(workbook, {
    exactCount: 218,
    failureCount: 14,
    shareableCount: 411,
  });
  console.log(JSON.stringify({ operation: "verified", inputPath, ...checked }));
}

const args = parseArgs(process.argv.slice(2));
if (args.verifyOnly) {
  if (!args.output) throw new Error("--output is required with --verify-only");
  await verifyWorkbook(args.output, args.preview);
} else {
  if (!args.source || !args.output) {
    throw new Error("--source and --output are required for workbook creation");
  }
  const source = JSON.parse(await fs.readFile(args.source, "utf8"));
  await createWorkbook(source, args.output, args.preview);
}

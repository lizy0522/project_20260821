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
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--verify-only") {
      args.verifyOnly = true;
    } else if (argv[i].startsWith("--")) {
      args[argv[i].slice(2)] = argv[i + 1];
      i += 1;
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

function validateRows(columns, rows, metadata) {
  assertCondition(JSON.stringify(columns) === JSON.stringify(COLUMNS), "Excel columns do not match contract");
  assertCondition(rows.length === 425, `Expected 425 data rows, got ${rows.length}`);
  const ids = rows.map((row) => finiteNumber(row[0], "state_id_R"));
  assertCondition(JSON.stringify(ids) === JSON.stringify(Array.from({ length: 425 }, (_, i) => i)), "state_id_R must be exactly 0...424");
  [2, 3, 4, 5, 6, 7].forEach((column) => rows.forEach((row, index) => finiteNumber(row[column], `${COLUMNS[column]} row ${index + 2}`)));
  rows.forEach((row, index) => {
    finiteNumber(row[8], `state_id_Q row ${index + 2}`);
    if (row[9] !== "-Inf") finiteNumber(row[9], `${COLUMNS[9]} row ${index + 2}`);
  });
  const exactCount = rows.filter((row) => row[9] === "-Inf").length;
  const finiteValues = rows.filter((row) => row[9] !== "-Inf").map((row) => Number(row[9]));
  const failureCount = finiteValues.filter((value) => value >= -40).length;
  const shareableCount = exactCount + finiteValues.filter((value) => value < -40).length;
  assertCondition(exactCount === Number(metadata.exact_hit_count), "Exact count mismatch");
  assertCondition(failureCount === Number(metadata.failure_count), "Failure count mismatch");
  assertCondition(shareableCount === Number(metadata.shareable_count), "Shareable count mismatch");
  return { exactCount, failureCount, shareableCount };
}

function formatSheet(sheet) {
  const all = sheet.getRange("A1:J426");
  all.format.font = { name: "Arial", size: 9, color: "#222222" };
  all.format.verticalAlignment = "center";
  const header = sheet.getRange("A1:J1");
  header.format.fill = "#1F4E78";
  header.format.font = { name: "Arial", size: 9, bold: true, color: "#FFFFFF" };
  header.format.horizontalAlignment = "center";
  header.format.wrapText = true;
  header.format.borders = { preset: "all", style: "thin", color: "#FFFFFF" };
  sheet.getRange("A2:A426").format.numberFormat = "0";
  sheet.getRange("C2:H426").format.numberFormat = "0.000000";
  sheet.getRange("I2:I426").format.numberFormat = "0";
  sheet.getRange("J2:J426").format.numberFormat = "0.000000";
  sheet.getRange("A1:J1").format.rowHeight = 30;
  sheet.getRange("A2:J426").format.rowHeight = 18;
  [12, 54, 22, 26, 24, 22, 22, 20, 12, 30].forEach((width, column) => {
    sheet.getRangeByIndexes(0, column, 426, 1).format.columnWidth = width;
  });
  sheet.freezePanes.freezeRows(1);
  const table = sheet.tables.add("A1:J426", true, "BestCandidateStateSummaryTable");
  table.showFilterButton = true;
  const cnmse = sheet.getRange("J2:J426");
  cnmse.conditionalFormats.add("cellIs", { operator: "lessThan", formula: -40, format: { fill: "#E2F0D9" } });
  cnmse.conditionalFormats.add("cellIs", { operator: "greaterThanOrEqual", formula: -40, format: { fill: "#FCE4D6" } });
  cnmse.conditionalFormats.add("containsText", { text: "-Inf", format: { fill: "#D9EAF7" } });
  sheet.showGridLines = false;
}

async function inspectAndCheck(workbook, metadata) {
  const inspected = await workbook.inspect({ kind: "table", sheetId: "state_summary", range: "A1:J6", include: "values,formulas", tableMaxRows: 6, tableMaxCols: 10, tableMaxCellChars: 120 });
  console.log("TABLE_INSPECT");
  console.log(inspected.ndjson);
  const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
  console.log("FORMULA_ERRORS");
  console.log(errors.ndjson);
  const records = (errors.ndjson || "").trim().split("\n").filter(Boolean).map((line) => { try { return JSON.parse(line); } catch { return { kind: "parse_error" }; } }).filter((record) => record.kind !== "notice");
  assertCondition(records.length === 0, "Workbook contains formula errors");
  const sheet = workbook.worksheets.getItem("state_summary");
  const values = sheet.getRange("A1:J426").values;
  assertCondition(values.length === 426 && values[0].length === 10, "Workbook must be 426x10");
  assertCondition(JSON.stringify(values[0]) === JSON.stringify(COLUMNS), "Workbook header mismatch");
  const counts = validateRows(COLUMNS, values.slice(1), metadata);
  return { rows: values.length, columns: values[0].length, ...counts };
}

async function createWorkbook(source, output, preview) {
  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add("state_summary");
  sheet.getRange("A1:J426").values = [source.columns, ...source.rows];
  formatSheet(sheet);
  const counts = validateRows(source.columns, source.rows, source.metadata);
  if (preview) {
    const rendered = await workbook.render({ sheetName: "state_summary", range: "A1:J25", scale: 1, format: "png" });
    await fs.writeFile(preview, new Uint8Array(await rendered.arrayBuffer()));
  }
  const exported = await SpreadsheetFile.exportXlsx(workbook);
  await exported.save(output);
  const reloaded = await SpreadsheetFile.importXlsx(await FileBlob.load(output));
  const checked = await inspectAndCheck(reloaded, source.metadata);
  console.log(JSON.stringify({ operation: "created_and_reloaded", output, ...counts, ...checked }));
}

async function verifyWorkbook(inputPath, metadata, preview) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
  if (preview) {
    const rendered = await workbook.render({ sheetName: "state_summary", range: "A1:J25", scale: 1, format: "png" });
    await fs.writeFile(preview, new Uint8Array(await rendered.arrayBuffer()));
  }
  const checked = await inspectAndCheck(workbook, metadata);
  console.log(JSON.stringify({ operation: "verified", inputPath, ...checked }));
}

const args = parseArgs(process.argv.slice(2));
if (args.verifyOnly) {
  assertCondition(args.output && args.metadata, "--output and --metadata are required with --verify-only");
  await verifyWorkbook(args.output, JSON.parse(await fs.readFile(args.metadata, "utf8")), args.preview);
} else {
  assertCondition(args.source && args.output, "--source and --output are required");
  await createWorkbook(JSON.parse(await fs.readFile(args.source, "utf8")), args.output, args.preview);
}

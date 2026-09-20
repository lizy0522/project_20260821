import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const STATE_HEADERS = [
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

function asMatrix(rows) {
  return rows.map((row) => STATE_HEADERS.map((header) => row[header] ?? null));
}

function applyBaseStyle(sheet, range) {
  range.format.font = { name: "Arial", size: 9, color: "#222222" };
  range.format.verticalAlignment = "center";
  sheet.showGridLines = false;
}

function applyTableStyle(sheet, headerRange, bodyRange) {
  headerRange.format.fill = "#1F4E78";
  headerRange.format.font = { name: "Arial", size: 9, bold: true, color: "#FFFFFF" };
  headerRange.format.horizontalAlignment = "center";
  headerRange.format.verticalAlignment = "center";
  headerRange.format.wrapText = true;
  headerRange.format.borders = { preset: "all", style: "thin", color: "#FFFFFF" };
  bodyRange.format.verticalAlignment = "center";
  bodyRange.format.borders = { preset: "inside", style: "thin", color: "#D9E2F3" };
  headerRange.format.rowHeight = 30;
  bodyRange.format.rowHeight = 18;
}

function formatStateSheet(sheet, rowCount) {
  applyBaseStyle(sheet, sheet.getRange(`A1:J${rowCount}`));
  applyTableStyle(sheet, sheet.getRange("A1:J1"), sheet.getRange(`A2:J${rowCount}`));
  sheet.getRange(`A2:A${rowCount}`).format.numberFormat = "0";
  sheet.getRange(`C2:H${rowCount}`).format.numberFormat = "0.000000";
  sheet.getRange(`I2:I${rowCount}`).format.numberFormat = "0";
  sheet.getRange(`J2:J${rowCount}`).format.numberFormat = "0.000000";
  [12, 54, 22, 26, 24, 22, 22, 20, 12, 30].forEach((width, column) => {
    sheet.getRangeByIndexes(0, column, rowCount, 1).format.columnWidth = width;
  });
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(2);
  const table = sheet.tables.add(`A1:J${rowCount}`, true, "UpdatedBeamStateReport");
  table.showFilterButton = true;
  sheet.getRange(`J2:J${rowCount}`).conditionalFormats.add("cellIs", { operator: "lessThan", formula: -40, format: { fill: "#E2F0D9" } });
  sheet.getRange(`J2:J${rowCount}`).conditionalFormats.add("cellIs", { operator: "greaterThanOrEqual", formula: -40, format: { fill: "#FCE4D6" } });
  sheet.getRange(`J2:J${rowCount}`).conditionalFormats.add("containsText", { text: "-Inf", format: { fill: "#D9EAF7" } });
}

function addSummarySheet(workbook, data) {
  const sheet = workbook.worksheets.add("Summary");
  const s = data.summary;
  sheet.getRange("A1").values = [["Updated Beam=3 Iteration-2 汇总"]];
  sheet.getRange("A1").format = { font: { name: "Arial", size: 14, bold: true, color: "#1F2937" } };
  sheet.getRange("A2").values = [[`主表对应全局 diagnostic rank 1：${s.global_best_model}`]];
  sheet.getRange("A2").format = { font: { name: "Arial", size: 10, italic: true, color: "#666666" } };
  const summaryRows = [
    ["模型 ID", s.global_best_model_id],
    ["模型名称", s.global_best_model],
    ["K", s.K],
    ["Top1 Real-B pass", s.top1_pass_count],
    ["Exact hit", s.exact_hit_count],
    ["Non-self count", s.nonself_count],
    ["Non-self pass", s.nonself_pass_count],
    ["Q05 shareability margin (dB)", s.Q05_shareability_margin_dB],
    ["MRR", s.MRR],
    ["Top2 / Top3 / Top5 / Top10", `${s.Top2} / ${s.Top3} / ${s.Top5} / ${s.Top10}`],
    ["Strict breakthrough count", s.strict_breakthrough_count],
    ["Parent-monotonic breakthrough count", s.parent_monotonic_breakthrough_count],
    ["Historical SeedA-safe breakthrough count", s.historical_SeedA_safe_breakthrough_count],
    ["Persistent failure intersection", s.persistent_failure_intersection],
    ["Failure union", s.failure_union],
    ["Support hash", s.support_hash],
    ["Support basis IDs", s.support_basis_ids.join(", ")],
  ];
  sheet.getRange(`A4:B${3 + summaryRows.length}`).values = summaryRows;
  sheet.getRange(`A4:A${3 + summaryRows.length}`).format = { fill: "#D9EAF7", font: { name: "Arial", size: 9, bold: true, color: "#1F2937" } };
  sheet.getRange(`B4:B${3 + summaryRows.length}`).format = { font: { name: "Arial", size: 9, color: "#222222" }, wrapText: true };
  sheet.getRange("B11:B12").format.numberFormat = "0.000000";
  sheet.getRange(`A4:B${3 + summaryRows.length}`).format.borders = { preset: "inside", style: "thin", color: "#D9E2F3" };
  sheet.getRange(`A4:A${3 + summaryRows.length}`).format.columnWidth = 46;
  sheet.getRange(`B4:B${3 + summaryRows.length}`).format.columnWidth = 72;
  sheet.getRange(`A4:B${3 + summaryRows.length}`).format.rowHeight = 22;

  const branchStart = 23;
  sheet.getRange(`A${branchStart}`).values = [["Branch best child"]];
  sheet.getRange(`A${branchStart}`).format = { font: { name: "Arial", size: 11, bold: true, color: "#1F2937" } };
  const branchHeaders = ["parent", "parent_top1", "best_child_model_name", "best_child_top1", "best_child_C_plus", "best_child_recovered", "best_child_regressed", "best_child_Q05", "best_child_MRR"];
  sheet.getRange(`A${branchStart + 1}:I${branchStart + 1}`).values = [branchHeaders];
  sheet.getRange(`A${branchStart + 2}:I${branchStart + 1 + data.branch_rows.length}`).values = data.branch_rows.map((row) => branchHeaders.map((header) => row[header] ?? null));
  applyTableStyle(sheet, sheet.getRange(`A${branchStart + 1}:I${branchStart + 1}`), sheet.getRange(`A${branchStart + 2}:I${branchStart + 1 + data.branch_rows.length}`));
  [24, 16, 48, 16, 17, 18, 17, 18, 18].forEach((width, column) => sheet.getRangeByIndexes(branchStart, column, data.branch_rows.length + 2, 1).format.columnWidth = width);
  sheet.getRange(`A${branchStart + 1}:I${branchStart + 1}`).format.rowHeight = 34;
  sheet.getRange(`A${branchStart + 2}:I${branchStart + 1 + data.branch_rows.length}`).format.rowHeight = 25;
  sheet.getRange(`B${branchStart + 2}:B${branchStart + 1 + data.branch_rows.length}`).format.numberFormat = "0";
  sheet.getRange(`D${branchStart + 2}:G${branchStart + 1 + data.branch_rows.length}`).format.numberFormat = "0";
  sheet.getRange(`H${branchStart + 2}:I${branchStart + 1 + data.branch_rows.length}`).format.numberFormat = "0.000000";
  sheet.showGridLines = false;
}

function addAuditSheet(workbook, data) {
  const sheet = workbook.worksheets.add("Audit");
  sheet.getRange("A1").values = [["验证与条件数审计"]];
  sheet.getRange("A1").format = { font: { name: "Arial", size: 14, bold: true, color: "#1F2937" } };
  sheet.getRange("A2").values = [["condition number 仅作诊断，不作为候选淘汰阈值；-Inf 表示 self Real-B CNMSE。"]];
  sheet.getRange("A2").format = { font: { name: "Arial", size: 10, italic: true, color: "#666666" } };
  sheet.getRange("A4:B4").values = [["check", "value"]];
  sheet.getRange(`A5:B${4 + data.audit_rows.length}`).values = data.audit_rows.map((row) => [row.check, row.value]);
  sheet.getRange("M4").values = [["detail"]];
  sheet.getRange(`M5:M${4 + data.audit_rows.length}`).values = data.audit_rows.map((row) => [typeof row.detail === "string" ? row.detail : JSON.stringify(row.detail)]);
  applyTableStyle(sheet, sheet.getRange("A4:B4"), sheet.getRange(`A5:B${4 + data.audit_rows.length}`));
  applyTableStyle(sheet, sheet.getRange("M4:M4"), sheet.getRange(`M5:M${4 + data.audit_rows.length}`));
  sheet.getRange(`A4:A${4 + data.audit_rows.length}`).format.columnWidth = 36;
  sheet.getRange(`B4:B${4 + data.audit_rows.length}`).format.columnWidth = 24;
  sheet.getRange(`M4:M${4 + data.audit_rows.length}`).format.columnWidth = 100;
  sheet.getRange(`M5:M${4 + data.audit_rows.length}`).format.wrapText = true;
  sheet.getRange(`A4:B4`).format.rowHeight = 30;
  sheet.getRange(`A5:B${4 + data.audit_rows.length}`).format.rowHeight = 34;
  sheet.getRange(`M4:M${4 + data.audit_rows.length}`).format.rowHeight = 34;
  const conditionStart = 17;
  sheet.getRange(`A${conditionStart}`).values = [["State-level conditioning summary"]];
  sheet.getRange(`A${conditionStart}`).format = { font: { name: "Arial", size: 11, bold: true, color: "#1F2937" } };
  const conditionHeaders = ["support_name", "behavior", "rank_min", "sigma_min_raw_min", "condition_raw_median", "condition_raw_Q95", "condition_column_normalized_median", "condition_column_normalized_Q95", "condition_ridge_augmented_median", "condition_ridge_augmented_Q95", "theta_l2_norm_max", "hard_failure"];
  sheet.getRange(`A${conditionStart + 1}:L${conditionStart + 1}`).values = [conditionHeaders];
  sheet.getRange(`A${conditionStart + 2}:L${conditionStart + 1 + data.condition_rows.length}`).values = data.condition_rows.map((row) => conditionHeaders.map((header) => row[header] ?? null));
  applyTableStyle(sheet, sheet.getRange(`A${conditionStart + 1}:L${conditionStart + 1}`), sheet.getRange(`A${conditionStart + 2}:L${conditionStart + 1 + data.condition_rows.length}`));
  [24, 16, 12, 20, 22, 22, 30, 30, 28, 28, 20, 12].forEach((width, column) => sheet.getRangeByIndexes(conditionStart, column, data.condition_rows.length + 2, 1).format.columnWidth = width);
  sheet.getRange(`A${conditionStart + 1}:L${conditionStart + 1}`).format.rowHeight = 42;
  sheet.getRange(`D${conditionStart + 2}:K${conditionStart + 1 + data.condition_rows.length}`).format.numberFormat = "0.000000";
  sheet.getRange(`A${conditionStart + 2}:L${conditionStart + 1 + data.condition_rows.length}`).format.rowHeight = 18;
  const sourceStart = conditionStart + 18 + data.condition_rows.length;
  sheet.getRange(`A${sourceStart}`).values = [["Sources and protocol"]];
  sheet.getRange(`A${sourceStart}`).format = { font: { name: "Arial", size: 11, bold: true, color: "#1F2937" } };
  sheet.getRange(`A${sourceStart + 1}:A${sourceStart + 4}`).values = [["result_root"], ["selected_model_source"], ["selected_query_source"], ["protocol"]];
  sheet.getRange(`M${sourceStart + 1}:M${sourceStart + 4}`).values = [[data.source.result_root], [data.source.selected_model_source], [data.source.selected_query_source], [data.source.protocol]];
  sheet.getRange(`A${sourceStart + 1}:A${sourceStart + 4}`).format = { fill: "#D9EAF7", font: { name: "Arial", size: 9, bold: true, color: "#1F2937" } };
  sheet.getRange(`M${sourceStart + 1}:M${sourceStart + 4}`).format = { font: { name: "Arial", size: 9, color: "#222222" }, wrapText: true };
  sheet.getRange(`A${sourceStart + 1}:A${sourceStart + 4}`).format.columnWidth = 28;
  sheet.getRange(`M${sourceStart + 1}:M${sourceStart + 4}`).format.columnWidth = 100;
  sheet.getRange(`A${sourceStart + 1}:M${sourceStart + 4}`).format.rowHeight = 36;
  sheet.showGridLines = false;
}

async function inspectAndCheck(workbook, data) {
  const stateSheet = workbook.worksheets.getItem("State Report");
  const values = stateSheet.getRange("A1:J426").values;
  assertCondition(values.length === 426 && values[0].length === 10, "State Report must be 426x10");
  assertCondition(JSON.stringify(values[0]) === JSON.stringify(STATE_HEADERS), "State Report header mismatch");
  assertCondition(values.slice(1).every((row, index) => Number(row[0]) === index), "State sequence must be exactly 0...424");
  const summaryInspect = await workbook.inspect({ kind: "table", sheetId: "Summary", range: "A1:I27", include: "values,formulas", tableMaxRows: 27, tableMaxCols: 9, tableMaxCellChars: 160 });
  console.log("SUMMARY_INSPECT");
  console.log(summaryInspect.ndjson);
  const stateInspect = await workbook.inspect({ kind: "table", sheetId: "State Report", range: "A1:J6", include: "values,formulas", tableMaxRows: 6, tableMaxCols: 10, tableMaxCellChars: 120 });
  console.log("STATE_INSPECT");
  console.log(stateInspect.ndjson);
  const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
  console.log("FORMULA_ERRORS");
  console.log(errors.ndjson);
  const errorRecords = (errors.ndjson || "").trim().split("\n").filter(Boolean).map((line) => { try { return JSON.parse(line); } catch { return { kind: "parse_error" }; } }).filter((record) => record.kind !== "notice");
  assertCondition(errorRecords.length === 0, "Workbook contains formula errors");
  return { state_rows: values.length - 1, state_columns: values[0].length, sheets: ["State Report", "Summary", "Audit"], formula_errors: errorRecords.length };
}

async function main() {
  const inputPath = process.argv[2];
  const outputPath = process.argv[3];
  const previewDir = process.argv[4];
  assertCondition(inputPath && outputPath, "usage: builder.mjs DATA_JSON OUTPUT_XLSX [PREVIEW_DIR]");
  const data = JSON.parse(await fs.readFile(inputPath, "utf8"));
  assertCondition(data.rows.length === 425, `Expected 425 state rows, got ${data.rows.length}`);
  const workbook = Workbook.create();
  const stateSheet = workbook.worksheets.add("State Report");
  stateSheet.getRange("A1:J426").values = [STATE_HEADERS, ...asMatrix(data.rows)];
  formatStateSheet(stateSheet, 426);
  addSummarySheet(workbook, data);
  addAuditSheet(workbook, data);
  workbook.recalculate();
  if (previewDir) {
    await fs.mkdir(previewDir, { recursive: true });
    for (const [sheetName, range, fileName] of [["State Report", "A1:J25", "state_report_preview.png"], ["Summary", "A1:I27", "summary_preview.png"], ["Audit", "A1:N30", "audit_preview.png"]]) {
      const blob = await workbook.render({ sheetName, range, scale: 1, format: "png" });
      await fs.writeFile(`${previewDir}/${fileName}`, new Uint8Array(await blob.arrayBuffer()));
    }
  }
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  const saved = await FileBlob.load(outputPath);
  const reloaded = await SpreadsheetFile.importXlsx(saved);
  const checked = await inspectAndCheck(reloaded, data);
  console.log(JSON.stringify({ operation: "created_and_reloaded", outputPath, ...checked }));
}

await main();

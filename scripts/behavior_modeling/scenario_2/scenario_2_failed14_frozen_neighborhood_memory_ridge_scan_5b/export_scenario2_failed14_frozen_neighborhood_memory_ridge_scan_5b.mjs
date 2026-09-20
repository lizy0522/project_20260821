import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = "D:/Project_Files/python_project/project_20260821";
const resultRoot = `${projectRoot}/results/behavior_modeling/scenario_2/scenario_2_failed14_frozen_neighborhood_memory_ridge_scan_5B`;
const sourcePath = `${resultRoot}/excel_source.json`;
const outputPath = `${resultRoot}/frozen_neighborhood_memory_ridge_scan_5B_results.xlsx`;

const source = JSON.parse(await fs.readFile(sourcePath, "utf8"));
const workbook = Workbook.create();
const fontFamily = "Arial";
const headerFill = "#1F4E78";
const titleColor = "#17365D";
const borderColor = "#D9E2F3";

function safe(value) {
  if (value === undefined || value === null) return null;
  if (typeof value === "number" && !Number.isFinite(value)) return null;
  return value;
}

function columnLetter(index) {
  let n = index + 1;
  let out = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    out = String.fromCharCode(65 + rem) + out;
    n = Math.floor((n - 1) / 26);
  }
  return out;
}

function rowsForSheet(name, spec) {
  if (name === "experiment_config") return spec.rows.map((row) => row.map(safe));
  return spec.rows.map((row) => spec.columns.map((column) => safe(row[column])));
}

function numberFormat(column) {
  const lower = column.toLowerCase();
  if (lower === "lambda" || lower.includes("penalty") || lower.includes("condition_number") || lower.includes("sigma") || lower.includes("residual")) return "0.00E+00";
  if (lower.includes("nmse") || lower.includes("gap") || lower.includes("median") || lower.includes("mean") || lower.includes("std") || lower.includes("q25") || lower.includes("q75") || lower.includes("q95") || lower.includes("worst") || lower.includes("delta") || lower.includes("theta_norm")) return "0.000";
  if (lower.includes("rate")) return "0.000";
  if (lower.includes("count") || lower.includes("candidate_id") || lower === "state_id" || lower === "p_max" || lower.startsWith("m") || lower === "k" || lower.includes("coefficient") || lower.includes("rank") || lower.includes("samples") || lower.includes("index")) return "0";
  return "General";
}

function writeSheet(name, spec, options = {}) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  const columns = spec.columns;
  const rows = rowsForSheet(name, spec);
  const headerRow = 4;
  const endColumn = columnLetter(columns.length - 1);
  const endRow = headerRow + rows.length;
  sheet.getRange("A1").values = [[options.title ?? name]];
  sheet.getRange("A1").format = { font: { name: fontFamily, size: 14, bold: true, color: titleColor } };
  sheet.getRange("A2").values = [[options.note ?? ""]];
  sheet.getRange("A2").format = { font: { name: fontFamily, size: 10, italic: true, color: "#666666" } };
  sheet.getRange(`A${headerRow}:${endColumn}${headerRow}`).values = [columns];
  if (rows.length > 0) sheet.getRange(`A${headerRow + 1}:${endColumn}${endRow}`).values = rows;
  sheet.getRange(`A${headerRow}:${endColumn}${headerRow}`).format = {
    fill: headerFill,
    font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#FFFFFF" },
  };
  if (rows.length > 0) {
    const body = sheet.getRange(`A${headerRow + 1}:${endColumn}${endRow}`);
    body.format = { font: { name: fontFamily, size: 10, color: "#222222" }, verticalAlignment: "center", borders: { preset: "inside", style: "thin", color: borderColor } };
    for (let index = 0; index < columns.length; index += 1) body.getColumn(index).format.numberFormat = numberFormat(columns[index]);
  }
  const tableName = `${name.replace(/[^A-Za-z0-9]/g, "")}Table`;
  sheet.tables.add(`A${headerRow}:${endColumn}${endRow}`, true, tableName);
  sheet.freezePanes.freezeRows(headerRow);
  if (options.freezeColumns) sheet.freezePanes.freezeColumns(options.freezeColumns);
  for (let index = 0; index < columns.length; index += 1) {
    const column = columnLetter(index);
    const lower = columns[index].toLowerCase();
    let width = 13;
    if (lower.includes("profile_string") || lower.includes("description") || lower.includes("failure_reason")) width = 25;
    else if (lower === "orders" || lower === "memory_profile" || lower.includes("memory")) width = 22;
    else if (lower.includes("source") || lower.includes("split")) width = 24;
    else if (lower.includes("condition") || lower.includes("residual")) width = 16;
    else if (lower.includes("candidate_id") || lower === "state_id") width = 12;
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  }
  sheet.getRange(`A${headerRow}:${endColumn}${endRow}`).format.rowHeight = 18;
  sheet.getRange(`A${headerRow}:${endColumn}${headerRow}`).format.rowHeight = 32;
}

const notes = {
  experiment_config: "Frozen six-order MP; selection on failure14; 425-state validation is post-selection only.",
  memory_structures: "Seven local memory allocations around [3,2,2,1,1,1]; M5/M7/M9 remain one tap.",
  ridge_grid: "Initial grid: lambda=0 plus 1e-10 through 1e-3. Positive lambda uses sqrt(N*lambda) augmentation.",
  candidate_grid: "Deterministic candidate IDs: structure order S0...S6, then ascending lambda.",
  candidate_summary: "Four NMSE metrics and diagnostics aggregated over the 14 failed states.",
  candidate_state_metrics: "Candidate × failed-state metrics; candidate-specific support is used for every fit.",
  frozen_regression: "S0 + lambda=1e-8 compared directly with the formal historical all-ILC metrics.",
  selected_candidate: "The balanced candidate selected using failed14 behavior-model metrics only.",
  selected_failed14: "Selected-versus-frozen statewise failure14 comparison.",
  ridge_effect: "Best row within each memory structure versus that structure's OLS reference.",
  pareto_candidates: "Nondominated candidates in worst train median, worst B median and K.",
  all425_validation: "Post-selection model-only validation summary for failure14, success411 and all425.",
};

for (const [name, spec] of Object.entries(source.sheets)) {
  writeSheet(name, spec, { title: name, note: notes[name] ?? "Saved experiment output.", freezeColumns: name.includes("metrics") || name.includes("summary") ? 4 : undefined });
}

workbook.recalculate();
const checks = [
  ["experiment_config", "A1:B28"],
  ["memory_structures", "A1:N12"],
  ["ridge_grid", "A1:C16"],
  ["candidate_grid", "A1:Q70"],
  ["candidate_summary", "A1:AZ70"],
  ["candidate_state_metrics", "A1:AZ30"],
  ["frozen_regression", "A1:Z25"],
  ["selected_candidate", "A1:AZ8"],
  ["selected_failed14", "A1:Z25"],
  ["ridge_effect", "A1:Z15"],
  ["pareto_candidates", "A1:AZ30"],
  ["all425_validation", "A1:Z10"],
];
for (const [sheetName, range] of checks) {
  const check = await workbook.inspect({ kind: "table", sheetId: sheetName, range, include: "values,formulas", tableMaxRows: 12, tableMaxCols: 26, maxChars: 12000 });
  console.log(check.ndjson);
}
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
console.log(errors.ndjson);
const renderDir = `${resultRoot}/validation/workbook_renders`;
await fs.mkdir(renderDir, { recursive: true });
for (const [sheetName, range] of checks) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(`${renderDir}/${sheetName}.png`, new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`saved ${outputPath}`);

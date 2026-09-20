import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = "D:/Project_Files/python_project/project_20260821";
const resultRoot = `${projectRoot}/results/behavior_modeling/scenario_2/scenario_2_failed14_frozen_mp_residual_sparse_gmp_5B`;
const source = JSON.parse(await fs.readFile(`${resultRoot}/excel_source.json`, "utf8"));
const outputPath = `${resultRoot}/frozen_mp_sparse_gmp_5B_results.xlsx`;
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
  if (lower.includes("lambda") || lower.includes("penalty") || lower.includes("condition") || lower.includes("sigma") || lower.includes("residual_norm")) return "0.00E+00";
  if (lower.includes("nmse") || lower.includes("gap") || lower.includes("median") || lower.includes("mean") || lower.includes("std") || lower.includes("q75") || lower.includes("worst") || lower.includes("corr") || lower.includes("energy") || lower.includes("delta") || lower.includes("power")) return "0.000";
  if (lower.includes("count") || lower.includes("step") || lower.includes("state_id") || lower === "p" || lower === "m" || lower === "q" || lower === "k" || lower.includes("rank") || lower.includes("index") || lower.includes("samples")) return "0";
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
  if (rows.length) sheet.getRange(`A${headerRow + 1}:${endColumn}${endRow}`).values = rows;
  sheet.getRange(`A${headerRow}:${endColumn}${headerRow}`).format = { fill: headerFill, font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true, borders: { preset: "all", style: "thin", color: "#FFFFFF" } };
  if (rows.length) {
    const body = sheet.getRange(`A${headerRow + 1}:${endColumn}${endRow}`);
    body.format = { font: { name: fontFamily, size: 10, color: "#222222" }, verticalAlignment: "center", borders: { preset: "inside", style: "thin", color: borderColor } };
    for (let index = 0; index < columns.length; index += 1) body.getColumn(index).format.numberFormat = numberFormat(columns[index]);
  }
  sheet.tables.add(`A${headerRow}:${endColumn}${endRow}`, true, `${name.replace(/[^A-Za-z0-9]/g, "")}Table`);
  sheet.freezePanes.freezeRows(headerRow);
  if (options.freezeColumns) sheet.freezePanes.freezeColumns(options.freezeColumns);
  for (let index = 0; index < columns.length; index += 1) {
    const column = columnLetter(index);
    const lower = columns[index].toLowerCase();
    let width = 13;
    if (lower.includes("term_ids") || lower.includes("expression") || lower.includes("structure") || lower.includes("reason")) width = 28;
    else if (lower.includes("orders") || lower.includes("memory") || lower.includes("source") || lower.includes("split")) width = 23;
    else if (lower.includes("condition") || lower.includes("residual")) width = 16;
    else if (lower.includes("state_id") || lower.includes("term_id")) width = 12;
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  }
  sheet.getRange(`A${headerRow}:${endColumn}${endRow}`).format.rowHeight = 18;
  sheet.getRange(`A${headerRow}:${endColumn}${headerRow}`).format.rowHeight = 32;
}

const notes = {
  experiment_config: "Frozen MP plus Sparse-GMP; term selection uses A/C blocked CV only.",
  frozen_mp: "Frozen MP orders, memory, coefficient count, lambda and common support.",
  gmp_dictionary: "18 unique cross-memory terms with p in {2,3,5}, m,q in {0,1,2}, m != q.",
  forward_selection: "One selected term per G1...G8 step; B is not available here.",
  trial_cv_metrics: "Per-trial A/C group CV metrics and residual-correlation diagnostics.",
  G0_G8_cv_summary: "Nested G0...G8 internal blocked-CV summary.",
  selected_structure: "Structure frozen before any B target is opened.",
  selected_failed14: "Full A/C refit and B comparison on the 14 selection states.",
  G0_G8_post_B: "Post-selection B diagnostic curve; not used for term selection.",
  residual_diagnostics: "Residual correlations, GMP coefficient norms and response energy ratios.",
  conditioning: "Full-fit numerical diagnostics along the final comparison models.",
  all425_validation: "Post-selection model-only validation summary for failure14, success411 and all425.",
};
for (const [name, spec] of Object.entries(source.sheets)) {
  writeSheet(name, spec, { title: name, note: notes[name] ?? "Saved experiment output.", freezeColumns: name.includes("metrics") || name.includes("selection") || name.includes("conditioning") ? 3 : undefined });
}

workbook.recalculate();
const checks = [
  ["experiment_config", "A1:B24"],
  ["frozen_mp", "A1:J10"],
  ["gmp_dictionary", "A1:I24"],
  ["forward_selection", "A1:Z16"],
  ["trial_cv_metrics", "A1:Z20"],
  ["G0_G8_cv_summary", "A1:Z16"],
  ["selected_structure", "A1:Z10"],
  ["selected_failed14", "A1:Z24"],
  ["G0_G8_post_B", "A1:Z20"],
  ["residual_diagnostics", "A1:Z20"],
  ["conditioning", "A1:Z20"],
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

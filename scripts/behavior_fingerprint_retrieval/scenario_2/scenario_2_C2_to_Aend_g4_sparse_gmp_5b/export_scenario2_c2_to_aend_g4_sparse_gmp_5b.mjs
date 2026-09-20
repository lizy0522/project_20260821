import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = "D:/Project_Files/python_project/project_20260821";
const resultRoot = `${projectRoot}/results/behavior_fingerprint_retrieval/scenario_2/scenario_2_C2_to_Aend_G4_sparse_gmp_5B`;
const source = JSON.parse(await fs.readFile(`${resultRoot}/excel_source.json`, "utf8"));
const outputPath = `${resultRoot}/G4_5B_LUT_retrieval_results.xlsx`;
const workbook = Workbook.create();
const fontFamily = "Arial";
const headerFill = "#1F4E78";
const titleColor = "#17365D";
const borderColor = "#D9E2F3";

function safe(value) {
  if (value === undefined || value === null) return null;
  if (typeof value === "number" && !Number.isFinite(value)) return null;
  if (typeof value === "object") return JSON.stringify(value);
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
  const lower = String(column).toLowerCase();
  if (lower.includes("lambda") || lower.includes("cnmse") || lower.includes("nmse") || lower.includes("acpr") || lower.includes("margin") || lower.includes("delta") || lower.includes("rate")) return "0.000";
  if (lower.includes("state_id") || lower.includes("count") || lower === "funmng" || lower === "funang" || lower === "secmng" || lower === "secang" || lower.includes("diff") || lower.includes("rank")) return "0";
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
    body.format = { font: { name: fontFamily, size: 9, color: "#222222" }, verticalAlignment: "center", borders: { preset: "inside", style: "thin", color: borderColor } };
    for (let index = 0; index < columns.length; index += 1) body.getColumn(index).format.numberFormat = numberFormat(columns[index]);
  }
  sheet.tables.add(`A${headerRow}:${endColumn}${endRow}`, true, `${name.replace(/[^A-Za-z0-9]/g, "")}Table`);
  sheet.freezePanes.freezeRows(headerRow);
  if (options.freezeColumns) sheet.freezePanes.freezeColumns(options.freezeColumns);
  for (let index = 0; index < columns.length; index += 1) {
    const column = columnLetter(index);
    const lower = String(columns[index]).toLowerCase();
    let width = 13;
    if (lower.includes("config") || lower.includes("state_id") || lower.includes("diff")) width = 18;
    if (lower.includes("cnmse") || lower.includes("nmse") || lower.includes("acpr")) width = 20;
    if (lower.includes("failure") || lower.includes("source") || lower.includes("retrieved")) width = 24;
    if (lower.startsWith("q_") || lower.startsWith("r_")) width = 11;
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  }
  sheet.getRange(`A${headerRow}:${endColumn}${endRow}`).format.rowHeight = 18;
  sheet.getRange(`A${headerRow}:${endColumn}${headerRow}`).format.rowHeight = 32;
}

const notes = {
  experiment_config: "G4 is loaded from the frozen Sparse-GMP discovery result; this task does not train or select terms.",
  G4_model_definition: "Frozen MP plus GMP005, GMP011, GMP001 and GMP017; K=14 and dmax=2.",
  retrieval_state_summary: "Main 425-state table. Real-B CNMSE is evaluated after C2→Aend Top-1 selection.",
  retrieval_distance_matrix: "Rows are State_R and columns Q_000...Q_424; values are behavior-fingerprint CNMSE in dB.",
  retrieval_statistics: "Exact, shareable, failure, rescue and Top-1 margin statistics.",
  failure_states: "States failing the strict Real-B CNMSE < -40 dB rule for G4.",
  frozen_vs_G4: "Read-only comparison with the earlier Frozen C2→Aend retrieval.",
};
for (const [name, spec] of Object.entries(source.sheets)) {
  writeSheet(name, spec, { title: name, note: notes[name] ?? "Saved retrieval output.", freezeColumns: name.includes("summary") || name.includes("matrix") || name.includes("vs") ? 2 : undefined });
}

workbook.recalculate();
const checks = [
  ["experiment_config", "A1:B24"],
  ["G4_model_definition", "A1:N8"],
  ["retrieval_state_summary", "A1:AI12"],
  ["retrieval_distance_matrix", "A1:K10"],
  ["retrieval_statistics", "A1:B22"],
  ["failure_states", "A1:AI12"],
  ["frozen_vs_G4", "A1:O12"],
];
for (const [sheetName, range] of checks) {
  const check = await workbook.inspect({ kind: "table", sheetId: sheetName, range, include: "values,formulas", tableMaxRows: 12, tableMaxCols: 40, maxChars: 12000 });
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

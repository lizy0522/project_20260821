import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = "D:/Project_Files/python_project/project_20260821";
const resultRoot = `${projectRoot}/results/behavior_model/scenario_2_failed14_P5_variable_memory_ridge_scan_5B`;
const sourcePath = `${resultRoot}/excel_source.json`;
const outputPath = `${resultRoot}/failed14_P5_variable_memory_ridge_scan_5B_results.xlsx`;

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
  if (lower === "lambda" || lower.includes("penalty") || lower.includes("condition_number") ||
      lower.includes("sigma") || lower.includes("residual_norm")) return "0.00E+00";
  if (lower.includes("nmse") || lower.includes("gap") || lower.includes("median") ||
      lower.includes("mean") || lower.includes("std") || lower.includes("q25") ||
      lower.includes("q75") || lower.includes("worst") || lower.includes("theta_norm")) return "0.000";
  if (lower.includes("rate")) return "0.000";
  if (lower.includes("count") || lower.includes("candidate_id") || lower === "state_id" ||
      lower === "p" || lower.startsWith("m") || lower.includes("coefficient") ||
      lower.includes("rank") || lower.includes("samples") || lower.includes("index")) return "0";
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
  sheet.getRange("A1").format = {
    font: { name: fontFamily, size: 14, bold: true, color: titleColor },
  };
  sheet.getRange("A2").values = [[options.note ?? ""]];
  sheet.getRange("A2").format = {
    font: { name: fontFamily, size: 10, italic: true, color: "#666666" },
  };
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
    body.format = {
      font: { name: fontFamily, size: 10, color: "#222222" },
      verticalAlignment: "center",
      borders: { preset: "inside", style: "thin", color: borderColor },
    };
    for (let index = 0; index < columns.length; index += 1) {
      body.getColumn(index).format.numberFormat = numberFormat(columns[index]);
    }
  }
  const tableName = `${name.replace(/[^A-Za-z0-9]/g, "")}Table`;
  sheet.tables.add(`A${headerRow}:${endColumn}${endRow}`, true, tableName);
  sheet.freezePanes.freezeRows(headerRow);
  if (options.freezeColumns) sheet.freezePanes.freezeColumns(options.freezeColumns);
  for (let index = 0; index < columns.length; index += 1) {
    const column = columnLetter(index);
    const lower = columns[index].toLowerCase();
    let width = 13;
    if (lower.includes("profile_string") || lower.includes("memory_definition")) width = 24;
    else if (lower === "orders" || lower === "memory_profile") width = 22;
    else if (lower.includes("failure_reason") || lower.includes("source")) width = 28;
    else if (lower.includes("condition_number") || lower.includes("residual_norm")) width = 16;
    else if (lower.includes("candidate_id") || lower === "state_id") width = 12;
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  }
  sheet.getRange(`A${headerRow}:${endColumn}${endRow}`).format.rowHeight = 18;
  sheet.getRange(`A${headerRow}:${endColumn}${headerRow}`).format.rowHeight = 32;
}

writeSheet("experiment_config", source.sheets.experiment_config, {
  title: "P5 variable-memory Ridge experiment configuration",
  note: "Frozen 14-state scope, P=5, 20 memory profiles × 10 lambda values, no retrieval.",
});
writeSheet("memory_profiles", source.sheets.memory_profiles, {
  title: "20 monotone memory profiles",
  note: "M1 >= M3 >= M5; K=M1+M3+M5 and max_delay=M1-1.",
  freezeColumns: 2,
});
writeSheet("ridge_grid", source.sheets.ridge_grid, {
  title: "Ridge lambda grid",
  note: "Lambda=0 is the OLS baseline; positive values use sqrt(N*lambda) augmentation.",
});
writeSheet("candidate_grid", source.sheets.candidate_grid, {
  title: "200 unified MP + Ridge candidates",
  note: "Candidate ID is deterministic: memory profile first, lambda ascending within profile.",
  freezeColumns: 4,
});
writeSheet("candidate_summary", source.sheets.candidate_summary, {
  title: "Candidate modeling and generalization summary",
  note: "Aggregated over the 14 failed states; selection uses behavior-model metrics only.",
  freezeColumns: 4,
});
writeSheet("candidate_state_metrics", source.sheets.candidate_state_metrics, {
  title: "Candidate × failed-state metrics",
  note: "2800 rows with four NMSE metrics, gaps, Ridge diagnostics, coefficient shrinkage and pass flags.",
  freezeColumns: 9,
});
writeSheet("selected_candidate", source.sheets.selected_candidate, {
  title: "Selected balanced candidate",
  note: "Unified candidate selected by balanced worst-four performance.",
});
writeSheet("selected_state_metrics", source.sheets.selected_state_metrics, {
  title: "Selected candidate statewise metrics",
  note: "Four core metrics and worst-four value for each failed state.",
  freezeColumns: 2,
});
writeSheet("statewise_best", source.sheets.statewise_best, {
  title: "Statewise best diagnostic",
  note: "Per-state best worst-four candidate; diagnostic only, not the unified selection.",
  freezeColumns: 2,
});
writeSheet("pareto_candidates", source.sheets.pareto_candidates, {
  title: "Modeling–generalization Pareto candidates",
  note: "Nondominated in modeling median, B generalization median and coefficient count.",
  freezeColumns: 4,
});
writeSheet("ridge_effect_summary", source.sheets.ridge_effect_summary, {
  title: "OLS versus best Ridge within each memory profile",
  note: "Negative B-median delta means the best Ridge value is more negative than OLS.",
  freezeColumns: 2,
});

workbook.recalculate();

const checks = [
  ["experiment_config", "A1:B24"],
  ["memory_profiles", "A1:I24"],
  ["ridge_grid", "A1:D14"],
  ["candidate_grid", "A1:M15"],
  ["candidate_summary", "A1:Q12"],
  ["candidate_state_metrics", "A1:Q12"],
  ["selected_candidate", "A1:Q8"],
  ["selected_state_metrics", "A1:Q18"],
  ["statewise_best", "A1:M18"],
  ["pareto_candidates", "A1:Q12"],
  ["ridge_effect_summary", "A1:Q24"],
];
for (const [sheetName, range] of checks) {
  const check = await workbook.inspect({
    kind: "table",
    sheetId: sheetName,
    range,
    include: "values,formulas",
    tableMaxRows: 24,
    tableMaxCols: 18,
    maxChars: 10000,
  });
  console.log(check.ndjson);
}
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const renderDir = `${resultRoot}/validation/workbook_renders`;
await fs.mkdir(renderDir, { recursive: true });
for (const [sheetName, range] of checks) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(`${renderDir}/${sheetName}.png`, new Uint8Array(await preview.arrayBuffer()));
}
await fs.mkdir(resultRoot, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`saved ${outputPath}`);

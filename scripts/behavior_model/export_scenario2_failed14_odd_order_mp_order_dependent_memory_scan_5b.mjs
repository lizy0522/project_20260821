import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = "D:/Project_Files/python_project/project_20260821";
const resultRoot = `${projectRoot}/results/behavior_model/scenario_2_failed14_odd_order_mp_order_dependent_memory_scan_5B`;
const sourcePath = `${resultRoot}/excel_source.json`;
const outputPath = `${resultRoot}/failed14_odd_order_mp_scan_5B_results.xlsx`;

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

function dataRows(name, spec) {
  if (name === "experiment_config") return spec.rows.map((row) => row.map(safe));
  return spec.rows.map((row) => spec.columns.map((column) => safe(row[column])));
}

function numberFormat(column) {
  const lower = column.toLowerCase();
  if (lower.includes("condition_number") || lower.includes("singular_value")) return "0.00E+00";
  if (lower.includes("nmse") || lower.includes("gap") || lower.includes("median") ||
      lower.includes("mean") || lower.includes("std") || lower.includes("quantile") ||
      lower.includes("q25") || lower.includes("q75") || lower.includes("min") ||
      lower.includes("max") || lower.includes("worst") || lower.includes("seconds") ||
      lower.includes("throughput")) return "0.00";
  if (lower.includes("rate")) return "0.000";
  if (lower.includes("count") || lower.includes("candidate_id") || lower === "state_id" ||
      lower === "p" || lower === "max_delay" || lower.includes("coefficient") ||
      lower.includes("rank") || lower.includes("samples")) return "0";
  return "General";
}

function writeSheet(name, spec, options = {}) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  const columns = spec.columns;
  const rows = dataRows(name, spec);
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
  if (rows.length > 0) {
    sheet.getRange(`A${headerRow + 1}:${endColumn}${endRow}`).values = rows;
  }
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
    else if (lower.includes("condition_number")) width = 16;
    else if (lower.includes("candidate_id") || lower === "state_id") width = 12;
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  }
  sheet.getRange(`A${headerRow}:${endColumn}${endRow}`).format.rowHeight = 18;
  sheet.getRange(`A${headerRow}:${endColumn}${headerRow}`).format.rowHeight = 32;
}

writeSheet("experiment_config", source.sheets.experiment_config, {
  title: "Failed-14 odd-order MP experiment configuration",
  note: "Frozen 14-state scope, 125 candidates, odd orders only, no Ridge or retrieval.",
});
writeSheet("candidate_grid", source.sheets.candidate_grid, {
  title: "125-candidate grid",
  note: "Deterministic candidate definitions; memory profiles are non-increasing and depths are 1–4.",
  freezeColumns: 3,
});
writeSheet("candidate_summary", source.sheets.candidate_summary, {
  title: "Candidate modeling and generalization summary",
  note: "Aggregated over the 14 failed states; all selection metrics are behavior-model metrics only.",
  freezeColumns: 3,
});
writeSheet("candidate_state_metrics", source.sheets.candidate_state_metrics, {
  title: "Candidate × failed-state metrics",
  note: "1750 candidate-state rows with Aend/C2 train, B generalization, gaps, rank, conditioning and theta diagnostics.",
  freezeColumns: 8,
});
writeSheet("selected_candidate", source.sheets.selected_candidate, {
  title: "Selected balanced candidate",
  note: "One unified candidate selected by the balanced worst-four rule.",
});
writeSheet("selected_state_metrics", source.sheets.selected_state_metrics, {
  title: "Selected candidate statewise metrics",
  note: "Four core metrics for each of the 14 failed states.",
  freezeColumns: 2,
});
writeSheet("statewise_best", source.sheets.statewise_best, {
  title: "Statewise best diagnostic",
  note: "Read-only per-state best worst-four candidate; not used as the unified selection.",
  freezeColumns: 2,
});
writeSheet("pareto_candidates", source.sheets.pareto_candidates, {
  title: "Modeling–generalization Pareto candidates",
  note: "Nondominated candidates in modeling median, B generalization median and coefficient count.",
  freezeColumns: 3,
});

workbook.recalculate();

const checks = [
  ["experiment_config", "A1:B24"],
  ["candidate_grid", "A1:J15"],
  ["candidate_summary", "A1:Q12"],
  ["candidate_state_metrics", "A1:Q12"],
  ["selected_candidate", "A1:Q8"],
  ["selected_state_metrics", "A1:Q18"],
  ["statewise_best", "A1:L18"],
  ["pareto_candidates", "A1:Q12"],
];
for (const [sheetName, range] of checks) {
  const check = await workbook.inspect({
    kind: "table",
    sheetId: sheetName,
    range,
    include: "values,formulas",
    tableMaxRows: 20,
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

import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = "D:/Project_Files/python_project/project_20260821";
const resultRoot = `${projectRoot}/results/behavior_model/scenario_2_unified_odd_order_mp_capacity_scan_5B`;
const sourcePath = `${resultRoot}/excel_source.json`;
const outputPath = `${resultRoot}/unified_odd_order_mp_capacity_scan_5B_results.xlsx`;

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

function rowsForSheet(name, spec) {
  if (name === "search_history") return spec.rows.map((row) => row.map(safe));
  return spec.rows.map((row) => spec.columns.map((column) => safe(row[column])));
}

function numberFormatForColumn(column) {
  const lower = column.toLowerCase();
  if (lower === "condition_number") return "0.00E+00";
  if (lower.includes("nmse") || lower.includes("median") || lower.includes("mean") ||
      lower.includes("std") || lower.includes("q05") || lower.includes("q25") ||
      lower.includes("q75") || lower.includes("q95") || lower.includes("worst") ||
      lower.includes("min") || lower.includes("max") || lower.includes("seconds") ||
      lower.includes("per_minute")) return "0.00";
  if (lower.includes("count") || lower.includes("pass") || lower.includes("candidate_id") ||
      lower === "state_id" || lower === "p" || lower === "m" || lower.includes("rank") ||
      lower.includes("rows") || lower.includes("delay") || lower.includes("coefficient")) return "0";
  return "General";
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

function writeSheet(name, spec, options = {}) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  const columns = spec.columns;
  const dataRows = rowsForSheet(name, spec);
  const title = options.title ?? name;
  sheet.getRange("A1").values = [[title]];
  sheet.getRange("A1").format = {
    font: { name: fontFamily, size: 14, bold: true, color: titleColor },
  };
  sheet.getRange("A2").values = [[options.note ?? ""]];
  sheet.getRange("A2").format = {
    font: { name: fontFamily, size: 10, italic: true, color: "#666666" },
  };
  const headerRow = 4;
  const headerEnd = columnLetter(columns.length - 1);
  const lastRow = headerRow + dataRows.length;
  sheet.getRange(`A${headerRow}:${headerEnd}${headerRow}`).values = [columns];
  if (dataRows.length > 0) {
    sheet.getRange(`A${headerRow + 1}:${headerEnd}${lastRow}`).values = dataRows;
  }
  sheet.getRange(`A${headerRow}:${headerEnd}${headerRow}`).format = {
    fill: headerFill,
    font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#FFFFFF" },
  };
  if (dataRows.length > 0) {
    const body = sheet.getRange(`A${headerRow + 1}:${headerEnd}${lastRow}`);
    body.format = {
      font: { name: fontFamily, size: 10, color: "#222222" },
      verticalAlignment: "center",
      borders: { preset: "inside", style: "thin", color: borderColor },
    };
    for (let index = 0; index < columns.length; index += 1) {
      const column = columnLetter(index);
      body.getColumn(index).format.numberFormat = numberFormatForColumn(columns[index]);
    }
  }
  const tableName = `${name.replace(/[^A-Za-z0-9]/g, "") }Table`;
  sheet.tables.add(`A${headerRow}:${headerEnd}${lastRow}`, true, tableName);
  sheet.freezePanes.freezeRows(headerRow);
  if (options.freezeColumns) sheet.freezePanes.freezeColumns(options.freezeColumns);
  sheet.getRange(`A${headerRow}:${headerEnd}${lastRow}`).format.autofitColumns();
  // Keep the wide candidate sheet usable without making the workbook enormous.
  for (let index = 0; index < columns.length; index += 1) {
    const column = columnLetter(index);
    const lower = columns[index].toLowerCase();
    let width = 13;
    if (lower === "orders") width = 28;
    else if (lower.includes("failure_reason")) width = 22;
    else if (lower.includes("source") || lower.includes("selection_rule")) width = 30;
    else if (lower.includes("condition_number")) width = 16;
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  }
  sheet.getRange(`A${headerRow}:${headerEnd}${lastRow}`).format.rowHeight = 18;
  sheet.getRange(`A${headerRow}:${headerEnd}${headerRow}`).format.rowHeight = 32;
  return sheet;
}

writeSheet("candidate_summary", source.sheets.candidate_summary, {
  title: "Unified odd-order MP candidate summary",
  note: "Development-only candidate selection; Round 0–3 cache, 280 candidates, 5B bandwidth.",
  freezeColumns: 3,
});
writeSheet("selected_model_state_metrics", source.sheets.selected_model_state_metrics, {
  title: "Selected model statewise metrics",
  note: "Selected unified model metrics for all 425 states; threshold flags use NMSE < -40 dB.",
  freezeColumns: 2,
});
writeSheet("development_summary", source.sheets.development_summary, {
  title: "Development summary",
  note: "Development set used for model selection (340 states).",
});
writeSheet("validation_summary", source.sheets.validation_summary, {
  title: "Validation summary",
  note: "Frozen validation set, not used for candidate selection (85 states).",
});
writeSheet("search_history", source.sheets.search_history, {
  title: "Search history",
  note: "Only Rounds 0–3 are included; the user-requested stop prevented Round 4.",
});

workbook.recalculate();

const check = await workbook.inspect({
  kind: "table",
  sheetId: "candidate_summary",
  range: "A1:H10",
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 8,
  maxChars: 8000,
});
console.log(check.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const renderDir = `${resultRoot}/validation/workbook_renders`;
await fs.mkdir(renderDir, { recursive: true });
for (const sheetName of ["candidate_summary", "selected_model_state_metrics", "development_summary", "validation_summary", "search_history"]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(`${renderDir}/${sheetName}.png`, new Uint8Array(await preview.arrayBuffer()));
}
await fs.mkdir(resultRoot, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`saved ${outputPath}`);

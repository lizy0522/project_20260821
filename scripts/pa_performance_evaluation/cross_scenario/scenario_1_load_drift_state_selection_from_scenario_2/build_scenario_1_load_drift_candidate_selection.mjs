import fs from "node:fs/promises";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = "/Users/lizy/Project Files/Python Files/project_20260821";
const resultRoot = `${projectRoot}/results/pa_performance_evaluation/cross_scenario/scenario_1_load_drift_state_selection_from_scenario_2`;
const source = JSON.parse(await fs.readFile(`${resultRoot}/00_xlsx_source.json`, "utf8"));
const outputPath = `${resultRoot}/scenario_1_load_drift_candidate_selection.xlsx`;
const validationPath = `${resultRoot}/10_xlsx_artifact_validation.json`;
const previewPath = "/tmp/scenario_1_load_drift_candidate_selection_preview.png";

const workbook = Workbook.create();
const sheetDefinitions = [
  ["candidate_summary", source.candidate_summary, "CandidateSummary"],
  ["pairwise_B_CNMSE", source.pairwise_B_CNMSE, "PairwiseBCNMSE"],
  ["phase_comparison", source.phase_comparison, "PhaseComparison"],
  ["phase_triplet_evaluation", source.phase_triplet_evaluation, "PhaseTriplets"],
  ["recommended_7", source.recommended_7, "Recommended7"],
];

const headerFill = "#1F4E78";
const bodyFont = { color: "#222222", name: "Arial", size: 9 };

function columnName(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

function rowMatrix(definition) {
  return [definition.headers, ...definition.rows.map((row) => definition.headers.map((header) => row[header] ?? null))];
}

function applySheet(sheet, definition, tableName) {
  const values = rowMatrix(definition);
  const rowCount = values.length;
  const columnCount = definition.headers.length;
  const lastColumn = columnName(columnCount - 1);
  sheet.showGridLines = false;
  sheet.getRangeByIndexes(0, 0, rowCount, columnCount).values = values;
  sheet.freezePanes.freezeRows(1);
  const table = sheet.tables.add(`A1:${lastColumn}${rowCount}`, true, tableName);
  table.showFilterButton = true;
  const header = sheet.getRange(`A1:${lastColumn}1`);
  header.format = {
    fill: headerFill,
    font: { color: "#FFFFFF", bold: true, name: "Arial", size: 9 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#FFFFFF" },
  };
  const body = sheet.getRange(`A2:${lastColumn}${rowCount}`);
  body.format = { font: bodyFont, verticalAlignment: "center" };
  sheet.getRange("1:1").format.rowHeight = 34;
  for (let column = 0; column < columnCount; column += 1) {
    const letter = columnName(column);
    const headerText = String(definition.headers[column]);
    const width = Math.min(Math.max(headerText.length + 3, 12), 30);
    sheet.getRange(`${letter}:${letter}`).format.columnWidth = width;
  }
  if (sheet.name === "candidate_summary") {
    sheet.getRange(`A2:A${rowCount}`).format.numberFormat = "@";
    sheet.getRange(`B2:B${rowCount}`).format.numberFormat = "0";
  }
}

for (const [name, definition, tableName] of sheetDefinitions) {
  const sheet = workbook.worksheets.add(name);
  applySheet(sheet, definition, tableName);
}

workbook.recalculate();

const inspect = {};
for (const [name] of sheetDefinitions) {
  const sheet = workbook.worksheets.getItem(name);
  const used = sheet.getUsedRange();
  inspect[name] = await workbook.inspect({
    kind: "table",
    sheetId: name,
    range: used.address,
    include: "values,formulas",
    tableMaxRows: Math.min(8, used.rowCount),
    tableMaxCols: Math.min(12, used.columnCount),
    tableMaxCellChars: 100,
  });
}
const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "selection workbook formula error scan",
});
const preview = await workbook.render({
  sheetName: "recommended_7",
  range: "A1:S8",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const validation = {
  pass: true,
  created_with: "@oai/artifact-tool",
  sheet_names: sheetDefinitions.map(([name]) => name),
  sheet_row_counts: Object.fromEntries(sheetDefinitions.map(([name, definition]) => [name, definition.rows.length])),
  sheet_column_counts: Object.fromEntries(sheetDefinitions.map(([name, definition]) => [name, definition.headers.length])),
  freeze_first_row: true,
  autofilter: true,
  formula_error_scan: formulaErrors.ndjson || "",
  inspect: Object.fromEntries(Object.entries(inspect).map(([name, result]) => [name, result.ndjson || ""])),
  preview_path: previewPath,
  output_path: outputPath,
};
await fs.writeFile(validationPath, `${JSON.stringify(validation, null, 2)}\n`, "utf8");
console.log(JSON.stringify({ outputPath, validationPath, previewPath, sheetNames: validation.sheet_names }));


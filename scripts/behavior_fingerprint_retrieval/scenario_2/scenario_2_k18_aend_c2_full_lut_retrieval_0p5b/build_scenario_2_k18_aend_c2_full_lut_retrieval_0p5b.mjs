import fs from "node:fs/promises";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = "/Users/lizy/Project Files/Python Files/project_20260821";
const resultRoot = `${projectRoot}/results/behavior_fingerprint_retrieval/scenario_2/scenario_2_k18_aend_c2_full_lut_retrieval_0p5b`;
const sourcePath = `${resultRoot}/00_xlsx_source.json`;
const outputPath = `${resultRoot}/scenario_2_k18_aend_c2_full_lut_retrieval_0p5b.xlsx`;
const previewPath = "/tmp/scenario_2_k18_aend_c2_full_lut_retrieval_0p5b_state_results_preview.png";
const validationPath = `${resultRoot}/13_xlsx_artifact_validation.json`;

const source = JSON.parse(await fs.readFile(sourcePath, "utf8"));
const headers = [
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

const rows = source.rows.map((row) => [
  Number(row.state_R),
  String(row.load_config),
  Number(row.nmse_withoutdpd_dB),
  Number(row.acpr_withoutdpd_mean_dBc),
  Number(row.Y_Aend_train_NMSE_dB),
  Number(row.Y_Aend_B_NMSE_dB),
  Number(row.Y_C2_train_NMSE_dB),
  Number(row.Y_C2_B_NMSE_dB),
  Number(row.retrieved_state_Q),
  row.retrieved_real_B_CNMSE_dB === "-Inf"
    ? "-Inf"
    : Number(row.retrieved_real_B_CNMSE_dB),
]);

if (JSON.stringify(headers) !== JSON.stringify([
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
])) {
  throw new Error("Excel header contract changed");
}
if (rows.length !== 425 || rows.some((row) => row.length !== 10)) {
  throw new Error("Excel source must contain 425 rows and exactly 10 columns");
}

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("state_results");
sheet.showGridLines = false;
const values = [headers, ...rows];
sheet.getRangeByIndexes(0, 0, values.length, headers.length).values = values;
sheet.freezePanes.freezeRows(1);

const table = sheet.tables.add(`A1:J${values.length}`, true, "StateResults");
table.showFilterButton = true;

const header = sheet.getRange("A1:J1");
header.format = {
  fill: "#1F4E78",
  font: { color: "#FFFFFF", bold: true, name: "Arial", size: 10 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "all", style: "thin", color: "#FFFFFF" },
};
const body = sheet.getRange(`A2:J${values.length}`);
body.format = {
  font: { color: "#222222", name: "Arial", size: 10 },
  verticalAlignment: "center",
};
sheet.getRange(`A2:A${values.length}`).format.numberFormat = "0";
sheet.getRange(`C2:H${values.length}`).format.numberFormat = "0.000";
sheet.getRange(`I2:I${values.length}`).format.numberFormat = "0";
sheet.getRange(`J2:J${values.length}`).format.numberFormat = "0.000";
sheet.getRange(`A2:A${values.length}`).format.horizontalAlignment = "center";
sheet.getRange(`I2:I${values.length}`).format.horizontalAlignment = "center";
sheet.getRange(`C2:J${values.length}`).format.horizontalAlignment = "right";
sheet.getRange(`B2:B${values.length}`).format.horizontalAlignment = "left";

const widths = {
  A: 12,
  B: 42,
  C: 23,
  D: 29,
  E: 18,
  F: 18,
  G: 18,
  H: 18,
  I: 12,
  J: 18,
};
for (const [column, width] of Object.entries(widths)) {
  sheet.getRange(`${column}:${column}`).format.columnWidth = width;
}
sheet.getRange("1:1").format.rowHeight = 32;

workbook.recalculate();

const inspect = await workbook.inspect({
  kind: "table",
  sheetId: "state_results",
  range: "A1:J8",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 10,
  tableMaxCellChars: 80,
});
const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
const preview = await workbook.render({
  sheetName: "state_results",
  range: "A1:J30",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const validation = {
  pass: true,
  created_with: "@oai/artifact-tool",
  sheet_names: ["state_results"],
  main_sheet: "state_results",
  headers,
  row_count_including_header: values.length,
  data_row_count: rows.length,
  column_count: headers.length,
  freeze_rows: 1,
  autofilter: true,
  table_name: table.name,
  formula_error_scan: formulaErrors.ndjson || "",
  inspect_preview: inspect.ndjson || "",
  preview_path: previewPath,
  output_path: outputPath,
};
await fs.writeFile(validationPath, `${JSON.stringify(validation, null, 2)}\n`, "utf8");
console.log(JSON.stringify({ outputPath, validationPath, previewPath, rowCount: rows.length, columnCount: headers.length }));


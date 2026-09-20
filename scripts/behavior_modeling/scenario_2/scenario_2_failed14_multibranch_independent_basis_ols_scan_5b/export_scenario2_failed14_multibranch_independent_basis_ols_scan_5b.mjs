import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = "D:/Project_Files/python_project/project_20260821";
const resultRoot = path.join(projectRoot, "results", "behavior_modeling", "scenario_2","scenario_2_failed14_multibranch_independent_basis_OLS_scan_5B");
const sourcePath = path.join(resultRoot, "excel_source.json");
const outputPath = path.join(resultRoot, "multibranch_independent_basis_OLS_failed14_results.xlsx");

const source = JSON.parse(await fs.readFile(sourcePath, "utf8"));
const fontFamily = "Arial";
const headerFill = "#1F4E78";
const sectionFill = "#D9EAF7";
const bodyFont = { name: fontFamily, size: 10, color: "#1F2937" };
const headerFont = { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" };

function columnLetter(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

function safeCell(value) {
  if (value === undefined) return null;
  if (typeof value === "number" && !Number.isFinite(value)) return null;
  if (value && typeof value === "object") return JSON.stringify(value);
  // Keep ISO timestamps as text instead of letting the spreadsheet engine
  // reinterpret them as serial date values without a date format.
  if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:/.test(value)) return `'${value}`;
  return value;
}

function normalizeRows(rows) {
  if (!rows || rows.length === 0) return { headers: ["No records"], values: [[null]] };
  const headerSet = new Set();
  for (const row of rows) for (const key of Object.keys(row)) headerSet.add(key);
  const headers = [...headerSet];
  const values = [headers, ...rows.map((row) => headers.map((key) => safeCell(row[key])))];
  return { headers, values };
}

function formatSheet(sheet, headers, rowCount) {
  const lastColumn = columnLetter(headers.length - 1);
  const used = `A1:${lastColumn}${Math.max(rowCount, 1)}`;
  sheet.showGridLines = false;
  sheet.getRange(used).format.font = bodyFont;
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: headerFill,
    font: headerFont,
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "thin", color: "#FFFFFF" },
  };
  sheet.getRange(used).format.verticalAlignment = "center";
  sheet.getRange(`A1:${lastColumn}1`).format.rowHeight = 30;
  sheet.freezePanes.freezeRows(1);
  sheet.getRange(used).format.autofitColumns();
  // Keep very long JSON/config columns readable without creating an enormous sheet.
  for (let i = 0; i < headers.length; i += 1) {
    const header = String(headers[i]);
    const letter = columnLetter(i);
    const range = sheet.getRange(`${letter}2:${letter}${Math.max(rowCount, 2)}`);
    if (/config|family_norms|basis_ids|manifest|value|note|expression/i.test(header)) {
      range.format.columnWidth = Math.min(Math.max(range.format.columnWidth || 18, 18), 42);
    } else if (/condition|sigma|norm|NMSE|gap|residual|correlation|median|mean|q75|worst|dB/i.test(header)) {
      range.format.numberFormat = "0.000000";
    } else if (/id|count|rank|K|N|memory|delay|basis/i.test(header)) {
      range.format.numberFormat = "0";
    }
  }
  if (headers.length > 0 && rowCount > 1) {
    const tableName = `${sheet.name.replace(/[^A-Za-z0-9]/g, "")}_Table`;
    const table = sheet.tables.add(used, true, tableName.slice(0, 200));
    table.showFilterButton = true;
    table.showBandedColumns = false;
  }
}

const workbook = Workbook.create();
const sheetOrder = [
  "selected_failed14",
  "reference_comparison",
  "generalization_gap",
  "experiment_config",
  "selected_structure",
  "basis_manifest",
  "search_summary",
  "parameter_sensitivity",
  "conditioning",
  "basis_family_counts",
  "coefficient_norms",
  "uniqueness_audit",
];

for (const sheetName of sheetOrder) {
  const sheet = workbook.worksheets.add(sheetName);
  const { headers, values } = normalizeRows(source[sheetName] || []);
  const lastColumn = columnLetter(headers.length - 1);
  const lastRow = values.length;
  sheet.getRange(`A1:${lastColumn}${lastRow}`).values = values;
  formatSheet(sheet, headers, lastRow);
  if (sheetName === "selected_failed14" || sheetName === "reference_comparison") {
    sheet.getRange(`A1:${lastColumn}1`).format.fill = sectionFill;
    sheet.getRange(`A1:${lastColumn}1`).format.font = { name: fontFamily, size: 10, bold: true, color: "#0F172A" };
  }
}

workbook.recalculate();

const previewDir = path.join(os.tmpdir(), "codex_multibranch_ols_previews");
await fs.mkdir(previewDir, { recursive: true });
for (const sheetName of sheetOrder) {
  const sheetRows = normalizeRows(source[sheetName] || []).values.length;
  const sheetHeaders = normalizeRows(source[sheetName] || []).headers;
  const previewLastColumn = columnLetter(sheetHeaders.length - 1);
  const previewLastRow = Math.min(sheetRows, 80);
  const preview = await workbook.render({ sheetName, range: `A1:${previewLastColumn}${previewLastRow}`, scale: 1, format: "png" });
  await fs.writeFile(path.join(previewDir, `${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, sheets: sheetOrder, previewDir }, null, 2));

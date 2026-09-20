import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const COLUMNS = [
  "状态序号",
  "负载配置",
  "NMSE(NMSE_WITHOUTDPD)",
  "上下边带ACPR平均值withoutdpd",
  "A段建模精度",
  "AB段泛化精度",
  "C段建模精度",
  "CB段泛化精度",
];

function assertCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function finiteNumber(value, name) {
  const number = Number(value);
  assertCondition(Number.isFinite(number), `${name} must be finite, got ${value}`);
  return number;
}

function validateSource(source) {
  assertCondition(JSON.stringify(source.columns) === JSON.stringify(COLUMNS), "source columns mismatch");
  assertCondition(Array.isArray(source.rows) && source.rows.length === 425, "source must contain 425 rows");
  source.rows.forEach((row, index) => {
    assertCondition(Array.isArray(row) && row.length === COLUMNS.length, `row ${index} width mismatch`);
    assertCondition(finiteNumber(row[0], `state_id row ${index}`) === index, `state_id row ${index} is not ${index}`);
    assertCondition(typeof row[1] === "string" && row[1].length > 0, `load config row ${index} is empty`);
    for (const column of [2, 3, 4, 5, 6, 7]) {
      finiteNumber(row[column], `${COLUMNS[column]} row ${index}`);
    }
  });
}

function applyFormatting(sheet) {
  const all = sheet.getRange("A1:H426");
  all.format.font = { name: "Arial", size: 9, color: "#222222" };
  all.format.verticalAlignment = "center";
  const header = sheet.getRange("A1:H1");
  header.format.fill = "#1F4E78";
  header.format.font = { name: "Arial", size: 9, bold: true, color: "#FFFFFF" };
  header.format.horizontalAlignment = "center";
  header.format.verticalAlignment = "center";
  header.format.wrapText = true;
  header.format.borders = { preset: "all", style: "thin", color: "#FFFFFF" };
  sheet.getRange("A2:A426").format.numberFormat = "0";
  sheet.getRange("C2:H426").format.numberFormat = "0.000000";
  sheet.getRange("A1:H1").format.rowHeight = 34;
  sheet.getRange("A2:H426").format.rowHeight = 18;
  const widths = [12, 48, 24, 30, 22, 22, 22, 22];
  widths.forEach((width, column) => {
    sheet.getRangeByIndexes(0, column, 426, 1).format.columnWidth = width;
  });
  sheet.freezePanes.freezeRows(1);
  const table = sheet.tables.add("A1:H426", true, "StatewiseBestMetricsTable");
  table.showFilterButton = true;
  sheet.showGridLines = false;
}

async function inspectAndValidate(workbook) {
  const inspect = await workbook.inspect({
    kind: "table",
    sheetId: "statewise_best_metrics",
    range: "A1:H6",
    include: "values,formulas",
    tableMaxRows: 6,
    tableMaxCols: 8,
    tableMaxCellChars: 120,
  });
  console.log("TABLE_INSPECT");
  console.log(inspect.ndjson);
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 300 },
    summary: "statewise best metrics formula error scan",
  });
  console.log("FORMULA_ERRORS");
  console.log(errors.ndjson);
  const errorRecords = (errors.ndjson || "")
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => {
      try { return JSON.parse(line); } catch { return { kind: "parse_error" }; }
    })
    .filter((record) => record.kind !== "notice");
  assertCondition(errorRecords.length === 0, "workbook contains formula errors");
  const values = workbook.worksheets.getItem("statewise_best_metrics").getRange("A1:H426").values;
  assertCondition(values.length === 426 && values[0].length === 8, "saved workbook dimensions mismatch");
  assertCondition(JSON.stringify(values[0]) === JSON.stringify(COLUMNS), "saved workbook header mismatch");
  for (let row = 1; row <= 425; row += 1) {
    assertCondition(Number(values[row][0]) === row - 1, `saved state_id mismatch at row ${row + 1}`);
    for (const column of [2, 3, 4, 5, 6, 7]) {
      assertCondition(Number.isFinite(Number(values[row][column])), `saved numeric value invalid at row ${row + 1}, col ${column + 1}`);
    }
  }
  return { rows: 425, columns: 8, first_state_id: values[1][0], last_state_id: values[425][0] };
}

async function main() {
  const [sourcePath, outputPath, previewPath] = process.argv.slice(2);
  if (!sourcePath || !outputPath) throw new Error("usage: node export...mjs <source.json> <output.xlsx> [preview.png]");
  const source = JSON.parse(await fs.readFile(sourcePath, "utf8"));
  validateSource(source);

  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add("statewise_best_metrics");
  sheet.getRange("A1:H426").values = [source.columns, ...source.rows];
  applyFormatting(sheet);
  workbook.recalculate();

  if (previewPath) {
    const preview = await workbook.render({
      sheetName: "statewise_best_metrics",
      range: "A1:H25",
      scale: 1,
      format: "png",
    });
    await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
  }

  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(outputPath);
  const saved = await FileBlob.load(outputPath);
  const reloaded = await SpreadsheetFile.importXlsx(saved);
  const checked = await inspectAndValidate(reloaded);
  console.log(JSON.stringify({ operation: "created_and_reloaded", outputPath, ...checked }));
}

await main();

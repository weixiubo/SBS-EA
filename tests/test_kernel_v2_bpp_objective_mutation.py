from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path

from pairwise_bpp_poc.config import PublicBenchmarkConfig, StageBConfig, StageBBudget, default_config
from pairwise_bpp_poc.kernel_v2 import run_kernel_stage_b
from pairwise_bpp_poc.mocking import build_mock_backends


def _write_bpplib_instance(path: Path, *, capacity: int, items: list[int]) -> None:
    tokens = [str(len(items)), str(capacity), *[str(item) for item in items]]
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")


def _write_bpplib_solutions_workbook(root: Path, names: list[str]) -> None:
    workbook_path = root / "Instances" / "Solutions" / "Solutions.xlsx"
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    shared_entries = ["Name", "Best LB", "Best UB", "Status", "Selected", *names, "Solved"]
    shared_strings = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        "<sst xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
        f"count=\"{len(shared_entries)}\" uniqueCount=\"{len(shared_entries)}\">",
    ]
    for value in shared_entries:
        shared_strings.append(f"<si><t>{value}</t></si>")
    shared_strings.append("</sst>")
    index_by_name = {value: index for index, value in enumerate(shared_entries)}

    def sheet_xml(sheet_names: list[str]) -> str:
        rows = [
            "<row r=\"1\">"
            f"<c r=\"A1\" t=\"s\"><v>{index_by_name['Name']}</v></c>"
            f"<c r=\"B1\" t=\"s\"><v>{index_by_name['Best LB']}</v></c>"
            f"<c r=\"C1\" t=\"s\"><v>{index_by_name['Best UB']}</v></c>"
            f"<c r=\"D1\" t=\"s\"><v>{index_by_name['Status']}</v></c>"
            f"<c r=\"E1\" t=\"s\"><v>{index_by_name['Selected']}</v></c>"
            "</row>"
        ]
        for row_index, name in enumerate(sheet_names, start=2):
            rows.append(
                f"<row r=\"{row_index}\">"
                f"<c r=\"A{row_index}\" t=\"s\"><v>{index_by_name[name]}</v></c>"
                f"<c r=\"B{row_index}\"><v>3</v></c>"
                f"<c r=\"C{row_index}\"><v>3</v></c>"
                f"<c r=\"D{row_index}\" t=\"s\"><v>{index_by_name['Solved']}</v></c>"
                f"<c r=\"E{row_index}\"><v>1</v></c>"
                "</row>"
            )
        return (
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
            "<sheetData>"
            + "".join(rows)
            + "</sheetData></worksheet>"
        )

    with zipfile.ZipFile(workbook_path, "w") as bundle:
        bundle.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>
""",
        )
        bundle.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
""",
        )
        bundle.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Falkenauer" sheetId="1" r:id="rId1"/>
    <sheet name="Scholl" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>
""",
        )
        bundle.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
</Relationships>
""",
        )
        bundle.writestr("xl/sharedStrings.xml", "".join(shared_strings))
        midpoint = len(names) // 2
        bundle.writestr("xl/worksheets/sheet1.xml", sheet_xml(names[:midpoint]))
        bundle.writestr("xl/worksheets/sheet2.xml", sheet_xml(names[midpoint:]))


def _write_small_bpplib_corpus(root_dir: Path) -> None:
    benchmark_root = root_dir / "benchmarks" / "bpplib" / "BPPLIB-main"
    family_specs = {
        "Falkenauer": [f"fal{i:02d}.txt" for i in range(1, 4)],
        "Scholl": [f"sch{i:02d}.txt" for i in range(1, 4)],
    }
    workbook_names: list[str] = []
    for family, names in family_specs.items():
        family_dir = benchmark_root / "Instances" / family
        family_dir.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(names, start=1):
            _write_bpplib_instance(
                family_dir / name,
                capacity=10,
                items=[2, 3, 1 + (index % 3), 1],
            )
            workbook_names.append(name)
    _write_bpplib_solutions_workbook(benchmark_root, workbook_names)


def test_run_kernel_stage_b_llm_mutation_action_supports_bpp_bpplib(tmp_path, monkeypatch):
    _write_small_bpplib_corpus(tmp_path)
    monkeypatch.setenv("POC_BPP_DATASET_KIND", "bpplib")

    base_config = default_config(tmp_path)
    budget = StageBBudget(
        name="budget_tiny",
        population_size=4,
        generations=2,
        elitism=2,
        mutations_per_gen=1,
        crossovers_per_gen=0,
        fresh_samples_per_gen=1,
    )
    config = replace(
        base_config,
        public_benchmarks=PublicBenchmarkConfig(
            bpplib_limit_per_family=3,
            bpplib_smoke_per_family=1,
            bpplib_val_per_family=1,
            bpplib_test_per_family=1,
        ),
        stage_b=StageBConfig(
            calibration_seed=0,
            calibration_budgets=(budget, budget),
            official_seeds=(0,),
        ),
        runtime=replace(base_config.runtime, run_tag="tiny_bpp_bpplib_test"),
    )
    generator_backend, judge_backends = build_mock_backends()
    result = run_kernel_stage_b(
        config=config,
        task="bpp",
        backend_name="mock",
        generator_backend=generator_backend,
        judge_backends=judge_backends,
        seed=0,
        phase="official",
        kernel="llm_mutation_action_v1",
    )

    unified_result = json.loads((result.artifact_dir / "unified_result.json").read_text(encoding="utf-8"))
    assert unified_result["task"] == "bpp"
    assert unified_result["benchmark_suite"] == "bpplib"
    assert unified_result["method_name"] == "llm_mutation_action_v1"

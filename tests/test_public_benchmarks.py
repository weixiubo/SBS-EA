from __future__ import annotations

from pathlib import Path
import zipfile

from pairwise_bpp_poc.config import default_config
from pairwise_bpp_poc.public_benchmarks import (
    summarize_diagnostic_gate,
    summarize_headroom_gate,
    summarize_official_acceptance,
)
from pairwise_bpp_poc.public_bpplib import (
    build_bpplib_dataset,
    bpplib_corpus_summary,
    load_bpplib_instances,
)
from pairwise_bpp_poc.public_jsplib import jsplib_corpus_summary
from pairwise_bpp_poc.public_sygus import sygus_corpus_summary


def _write_bpplib_instance(path: Path, *, capacity: int, items: list[int]) -> None:
    tokens = [str(len(items)), str(capacity), *[str(item) for item in items]]
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")


def _write_bpplib_solutions_workbook(root: Path) -> None:
    workbook_path = root / "Instances" / "Solutions" / "Solutions.xlsx"
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
"""
    workbook = """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Falkenauer" sheetId="1" r:id="rId1"/>
    <sheet name="Scholl" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>
"""
    workbook_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
</Relationships>
"""
    shared_strings = """<?xml version="1.0" encoding="UTF-8"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="10" uniqueCount="10">
  <si><t>Name</t></si>
  <si><t>Best LB</t></si>
  <si><t>Best UB</t></si>
  <si><t>Status</t></si>
  <si><t>Selected</t></si>
  <si><t>a1.txt</t></si>
  <si><t>Solved</t></si>
  <si><t>a2.txt</t></si>
  <si><t>a3.txt</t></si>
  <si><t>a4.txt</t></si>
</sst>
"""
    sheet1 = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1">
      <c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c><c r="D1" t="s"><v>3</v></c><c r="E1" t="s"><v>4</v></c>
    </row>
    <row r="2">
      <c r="A2" t="s"><v>5</v></c><c r="B2"><v>3</v></c><c r="C2"><v>3</v></c><c r="D2" t="s"><v>6</v></c><c r="E2"><v>1</v></c>
    </row>
    <row r="3">
      <c r="A3" t="s"><v>7</v></c><c r="B3"><v>4</v></c><c r="C3"><v>4</v></c><c r="D3" t="s"><v>6</v></c><c r="E3"><v>1</v></c>
    </row>
  </sheetData>
</worksheet>
"""
    sheet2 = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1">
      <c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c><c r="D1" t="s"><v>3</v></c><c r="E1" t="s"><v>4</v></c>
    </row>
    <row r="2">
      <c r="A2" t="s"><v>8</v></c><c r="B2"><v>2</v></c><c r="C2"><v>2</v></c><c r="D2" t="s"><v>6</v></c><c r="E2"><v>1</v></c>
    </row>
    <row r="3">
      <c r="A3" t="s"><v>9</v></c><c r="B3"><v>5</v></c><c r="C3"><v>5</v></c><c r="D3" t="s"><v>6</v></c><c r="E3"><v>1</v></c>
    </row>
  </sheetData>
</worksheet>
"""
    with zipfile.ZipFile(workbook_path, "w") as bundle:
        bundle.writestr("[Content_Types].xml", content_types)
        bundle.writestr("_rels/.rels", rels)
        bundle.writestr("xl/workbook.xml", workbook)
        bundle.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        bundle.writestr("xl/sharedStrings.xml", shared_strings)
        bundle.writestr("xl/worksheets/sheet1.xml", sheet1)
        bundle.writestr("xl/worksheets/sheet2.xml", sheet2)


def _write_jsplib_instance(path: Path, rows: list[list[int]]) -> None:
    jobs = len(rows)
    machines = len(rows[0]) // 2
    payload = [f"{jobs} {machines}"] + [" ".join(str(item) for item in row) for row in rows]
    path.write_text("\n".join(payload) + "\n", encoding="utf-8")


def test_bpplib_loader_and_dataset_builder(tmp_path):
    instances_dir = tmp_path / "Instances"
    for family in ("Falkenauer", "Scholl"):
        family_dir = instances_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        _write_bpplib_instance(family_dir / "a1.txt", capacity=10, items=[2, 3, 4, 1])
        _write_bpplib_instance(family_dir / "a2.txt", capacity=10, items=[5, 1, 2, 1])
        _write_bpplib_instance(family_dir / "a3.txt", capacity=10, items=[4, 4, 1, 1])
    _write_bpplib_solutions_workbook(tmp_path)

    instances = load_bpplib_instances(
        tmp_path,
        include_families=("Falkenauer", "Scholl"),
        limit_per_family=3,
        max_items=10,
    )
    assert len(instances) == 6
    assert instances[0].best_known_bins == 3
    assert instances[1].best_known_bins == 4
    dataset = build_bpplib_dataset(instances, smoke_per_family=1, val_per_family=1, test_per_family=1)
    assert len(dataset["smoke"]) == 2
    assert len(dataset["val"]) == 2
    assert len(dataset["test"]) == 2
    assert max(dataset["smoke"][0].items) <= 1.0
    assert dataset["smoke"][0].best_known_bins == 3


def test_bpplib_summary_reports_missing_corpus(tmp_path):
    config = default_config(tmp_path)
    summary = bpplib_corpus_summary(config, download=False)
    assert summary["official_corpus_available"] is False
    assert summary["paper_ready"] is False


def test_bpplib_summary_reports_download_failure(tmp_path, monkeypatch):
    config = default_config(tmp_path)
    monkeypatch.setattr(
        "pairwise_bpp_poc.public_bpplib.ensure_bpplib_repo",
        lambda _target: (_ for _ in ()).throw(RuntimeError("network blocked")),
    )
    summary = bpplib_corpus_summary(config, download=True)
    assert summary["official_corpus_available"] is False
    assert summary["reason"] == "download_failed"


def test_sygus_summary_uses_preview_when_public_corpus_missing(tmp_path):
    config = default_config(tmp_path)
    summary = sygus_corpus_summary(config, download=False)
    assert summary["official_corpus_available"] is False
    assert summary["preview_mode_active"] is True
    assert summary["preview_summary"]["preview_available"] is True
    assert summary["paper_ready"] is False


def test_sygus_summary_reports_public_files_when_present(tmp_path):
    config = default_config(tmp_path)
    corpus_root = (
        config.runtime.root_dir
        / config.public_benchmarks.root_dir_name
        / config.public_benchmarks.sygus_dir_name
        / "comp"
    )
    corpus_root.mkdir(parents=True, exist_ok=True)
    (corpus_root / "sample.sl").write_text("(set-logic BV)\n", encoding="utf-8")
    summary = sygus_corpus_summary(config, download=False)
    assert summary["official_corpus_available"] is True
    assert summary["preview_mode_active"] is False
    assert summary["paper_ready"] is True


def test_jsplib_summary_reports_missing_corpus(tmp_path):
    config = default_config(tmp_path)
    summary = jsplib_corpus_summary(config, download=False)
    assert summary["official_corpus_available"] is False
    assert summary["paper_ready"] is False


def test_jsplib_summary_reports_download_failure(tmp_path, monkeypatch):
    config = default_config(tmp_path)
    monkeypatch.setattr(
        "pairwise_bpp_poc.public_jsplib.ensure_jsplib_repo",
        lambda _target: (_ for _ in ()).throw(RuntimeError("network blocked")),
    )
    summary = jsplib_corpus_summary(config, download=True)
    assert summary["official_corpus_available"] is False
    assert summary["reason"] == "download_failed"


def test_jsplib_summary_reports_metadata_when_present(tmp_path):
    config = default_config(tmp_path)
    corpus_root = (
        config.runtime.root_dir
        / config.public_benchmarks.root_dir_name
        / config.public_benchmarks.jsplib_dir_name
    )
    instances_dir = corpus_root / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)
    (instances_dir / "ft06").write_text("6 6\n", encoding="utf-8")
    (instances_dir / "la01").write_text("10 5\n", encoding="utf-8")
    (corpus_root / "instances.json").write_text(
        (
            "["
            "{\"name\":\"ft06\",\"jobs\":6,\"machines\":6,\"optimum\":55,\"path\":\"instances/ft06\"},"
            "{\"name\":\"la01\",\"jobs\":10,\"machines\":5,\"optimum\":null,"
            "\"bounds\":{\"lower\":666,\"upper\":666},\"path\":\"instances/la01\"}"
            "]"
        ),
        encoding="utf-8",
    )
    summary = jsplib_corpus_summary(config, download=False)
    assert summary["official_corpus_available"] is True
    assert summary["instance_count"] == 2
    assert summary["family_counts"] == {"ft": 1, "la": 1}
    assert summary["optimum_known_count"] == 1
    assert summary["bounded_only_count"] == 1
    assert summary["missing_file_count"] == 0
    assert summary["default_dataset_counts"] == {"smoke": 0, "val": 0, "test": 0}
    assert summary["paper_ready"] is True


def test_jsplib_summary_reports_stage_a_proxy_when_valid_dataset_present(tmp_path):
    config = default_config(tmp_path)
    corpus_root = (
        config.runtime.root_dir
        / config.public_benchmarks.root_dir_name
        / config.public_benchmarks.jsplib_dir_name
    )
    instances_dir = corpus_root / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("abz01", [[0, 3, 1, 2], [1, 2, 0, 1]]),
        ("abz02", [[0, 2, 1, 4], [1, 1, 0, 3]]),
        ("abz03", [[0, 4, 1, 1], [1, 3, 0, 2]]),
        ("ft01", [[0, 1, 1, 5], [1, 4, 0, 2]]),
        ("ft02", [[0, 2, 1, 3], [1, 5, 0, 1]]),
        ("ft03", [[0, 3, 1, 1], [1, 2, 0, 4]]),
        ("la01", [[0, 2, 1, 2], [1, 3, 0, 1], [0, 4, 1, 1]]),
        ("la02", [[0, 4, 1, 1], [1, 2, 0, 2], [0, 1, 1, 3]]),
        ("la03", [[0, 1, 1, 4], [1, 3, 0, 2], [0, 2, 1, 2]]),
    ]
    metadata_rows: list[str] = []
    for name, rows in specs:
        relative_path = f"instances/{name}"
        _write_jsplib_instance(instances_dir / name, rows)
        metadata_rows.append(
            (
                "{"
                f"\"name\":\"{name}\","
                f"\"jobs\":{len(rows)},"
                f"\"machines\":{len(rows[0]) // 2},"
                "\"optimum\":null,"
                f"\"path\":\"{relative_path}\""
                "}"
            )
        )
    (corpus_root / "instances.json").write_text(
        "[" + ",".join(metadata_rows) + "]",
        encoding="utf-8",
    )

    summary = jsplib_corpus_summary(config, download=False)
    assert summary["official_corpus_available"] is True
    assert summary["default_dataset_counts"] == {"smoke": 3, "val": 3, "test": 3}
    assert summary["parse_failure_count"] == 0
    assert summary["dispatch_preview"]["rule_count"] == 6
    assert summary["stage_a_proxy"]["candidate_count"] == 6
    assert "pointwise_kendall" in summary["stage_a_proxy"]
    assert "pairwise_final_kendall" in summary["stage_a_proxy"]
    assert len(summary["stage_a_proxy"]["pairwise_matches"]) == 15


def test_public_gate_summaries():
    diagnostic = summarize_diagnostic_gate(
        {
            0: {
                "pointwise_kendall": 0.40,
                "pairwise_final_kendall": 0.55,
                "pointwise_spearman": 0.50,
                "pairwise_final_spearman": 0.57,
            },
            1: {
                "pointwise_kendall": 0.42,
                "pairwise_final_kendall": 0.51,
                "pointwise_spearman": 0.49,
                "pairwise_final_spearman": 0.53,
            },
            2: {
                "pointwise_kendall": 0.41,
                "pairwise_final_kendall": 0.48,
                "pointwise_spearman": 0.52,
                "pairwise_final_spearman": 0.53,
            },
        }
    )
    assert diagnostic["ready"] is True

    headroom = summarize_headroom_gate(
        {
            0: {"seed_only_test": 100.0, "oracle_test": 96.0},
            1: {"seed_only_test": 102.0, "oracle_test": 97.0},
            2: {"seed_only_test": 99.0, "oracle_test": 95.0},
        }
    )
    assert headroom["ready"] is True

    acceptance = summarize_official_acceptance(
        {
            0: {"pointwise_test": 98.0, "pairwise_final_test": 96.0},
            1: {"pointwise_test": 100.0, "pairwise_final_test": 99.0},
            2: {"pointwise_test": 101.0, "pairwise_final_test": 97.0},
        }
    )
    assert acceptance["ready"] is True

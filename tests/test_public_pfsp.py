from __future__ import annotations

import zipfile

from sbs_ea.adapters import _parse_vfr_families
from sbs_ea.pfsp import PFSPInstance
from sbs_ea.public_pfsp import (
    _parse_vfr_dat,
    build_taillard_mixed72_dataset,
    build_taillard_mixed96_dataset,
    load_taillard_instances,
    load_vfr_best_known,
    load_vfr_instances,
)


def test_public_taillard_loader_parses_machine_rows(tmp_path) -> None:
    archive = tmp_path / "TestSolutions.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "TestSolutions/taillard/ta001.dat",
            "3 2\n1 2 3\n4 5 6\n",
        )

    instances = load_taillard_instances(archive, best_known={"ta001": 12})

    assert len(instances) == 1
    instance = instances[0]
    assert instance.instance_id == "ta001"
    assert instance.jobs == 3
    assert instance.machines == 2
    assert instance.processing_times == ((1, 4), (2, 5), (3, 6))
    assert instance.best_known_makespan == 12


def test_taillard_mixed96_split_uses_public_family_holdouts(monkeypatch, tmp_path) -> None:
    instances = []
    for jobs, machines, start in ((20, 5, 1), (20, 10, 11)):
        for offset in range(10):
            instances.append(
                PFSPInstance(
                    family=f"taillard_{jobs}x{machines}",
                    split="public",
                    seed=start + offset,
                    instance_id=f"ta{start + offset:03d}",
                    jobs=jobs,
                    machines=machines,
                    processing_times=tuple(tuple(1 for _ in range(machines)) for _ in range(jobs)),
                    best_known_makespan=jobs + machines - 1,
                )
            )

    monkeypatch.setattr("sbs_ea.public_pfsp.TAILLARD_MIXED_FAMILIES", ((20, 5), (20, 10)))
    monkeypatch.setattr("sbs_ea.public_pfsp.ensure_taillard_public_files", lambda _root: tmp_path)
    monkeypatch.setattr("sbs_ea.public_pfsp._load_taillard_upper_bounds", lambda _path: {})
    monkeypatch.setattr("sbs_ea.public_pfsp.load_taillard_instances", lambda _path, best_known: instances)

    dataset = build_taillard_mixed96_dataset(tmp_path)

    assert [len(dataset[split]) for split in ("smoke", "val", "test")] == [2, 2, 16]
    assert dataset["test"][0].instance_id == "test-ta003"
    assert dataset["test"][-1].instance_id == "test-ta020"


def test_taillard_mixed72_uses_larger_development_split(monkeypatch, tmp_path) -> None:
    instances = []
    for jobs, machines, start in ((20, 5, 1), (20, 10, 11)):
        for offset in range(10):
            instances.append(
                PFSPInstance(
                    family=f"taillard_{jobs}x{machines}",
                    split="public",
                    seed=start + offset,
                    instance_id=f"ta{start + offset:03d}",
                    jobs=jobs,
                    machines=machines,
                    processing_times=tuple(tuple(1 for _ in range(machines)) for _ in range(jobs)),
                    best_known_makespan=jobs + machines - 1,
                )
            )

    monkeypatch.setattr("sbs_ea.public_pfsp.TAILLARD_MIXED_FAMILIES", ((20, 5), (20, 10)))
    monkeypatch.setattr("sbs_ea.public_pfsp.ensure_taillard_public_files", lambda _root: tmp_path)
    monkeypatch.setattr("sbs_ea.public_pfsp._load_taillard_upper_bounds", lambda _path: {})
    monkeypatch.setattr("sbs_ea.public_pfsp.load_taillard_instances", lambda _path, best_known: instances)

    dataset = build_taillard_mixed72_dataset(tmp_path)

    assert [len(dataset[split]) for split in ("smoke", "val", "test")] == [4, 4, 12]
    assert dataset["val"][0].instance_id == "val-ta003"
    assert dataset["test"][0].instance_id == "test-ta005"
    assert dataset["test"][-1].instance_id == "test-ta020"


def test_vfr_parser_uses_machine_job_pairs() -> None:
    jobs, machines, rows = _parse_vfr_dat("2 3\n0 5 1 7 2 11\n2 13 0 17 1 19\n")

    assert jobs == 2
    assert machines == 3
    assert rows == ((5, 7, 11), (17, 19, 13))


def test_vfr_loader_filters_transfer_families(monkeypatch, tmp_path) -> None:
    import sbs_ea.public_pfsp as public_pfsp

    payloads = {
        "TestSolutions/vrf/VFR2_2_1_Gap.txt": b"2 2\n0 1 1 2\n1 3 0 4\n",
        "TestSolutions/vrf/VFR3_2_1_Gap.txt": b"3 2\n0 5 1 6\n0 7 1 8\n0 9 1 10\n",
    }

    monkeypatch.setattr(public_pfsp, "_stream_zip_entries", lambda _blob: payloads)
    zip_path = tmp_path / "dummy.zip"
    zip_path.write_bytes(b"unused")

    instances = load_vfr_instances(zip_path, families=((2, 2),))

    assert [item.instance_id for item in instances] == [
        "vfr-transfer-VFR2_2_1_Gap",
    ]
    assert instances[0].processing_times == ((1, 2), (4, 3))


def test_vfr_best_known_loader_parses_semicolon_csv(tmp_path) -> None:
    path = tmp_path / "VFRsmall.csv"
    path.write_text(
        "Name;n;m;LB;UB;Optimal;UBFoundBy;Permutation\nVFR10_5_1;10;5;;695;1;0; 1 2 3 4 5 6 7 8 9 10\n",
        encoding="utf-8",
    )
    assert load_vfr_best_known((path,)) == {"VFR10_5_1": 695}


def test_vfr_transfer_family_parser_accepts_cli_tokens() -> None:
    assert _parse_vfr_families("100x20, 200x40") == ((100, 20), (200, 40))

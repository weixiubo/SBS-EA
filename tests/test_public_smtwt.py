from __future__ import annotations

import zipfile

from pairwise_bpp_poc.public_smtwt import _parse_wt_text, build_wt40_dataset, load_wt40_instances


def test_wt40_parser_reads_three_vectors() -> None:
    processing, weights, due_dates = _parse_wt_text(
        "4\n"
        "3 1 2 5\n"
        "4 10 2 7\n"
        "4 2 8 9\n"
    )

    assert processing == (3, 1, 2, 5)
    assert weights == (4, 10, 2, 7)
    assert due_dates == (4, 2, 8, 9)


def test_wt40_loader_and_split_are_deterministic(monkeypatch, tmp_path) -> None:
    archive = tmp_path / "wt40-1m.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for index in (1, 2, 3, 4, 5, 10):
            bundle.writestr(
                f"1m/wt40-1m-{index}.txt",
                "4\n"
                f"{index} {index + 1} {index + 2} {index + 3}\n"
                "4 10 2 7\n"
                "4 2 8 9\n",
            )

    monkeypatch.setattr("pairwise_bpp_poc.public_smtwt.ensure_wt40_public_file", lambda _root: archive)

    instances = load_wt40_instances(tmp_path)
    assert [item.instance_id for item in instances] == [
        "wt40-1m-1",
        "wt40-1m-2",
        "wt40-1m-3",
        "wt40-1m-4",
        "wt40-1m-5",
        "wt40-1m-10",
    ]

    dataset = build_wt40_dataset(tmp_path, smoke_count=2, val_count=2, test_count=2)
    assert [len(dataset[split]) for split in ("smoke", "val", "test")] == [2, 2, 2]
    assert dataset["smoke"][0].instance_id == "smoke-wt40-1m-1"
    assert dataset["val"][0].instance_id == "val-wt40-1m-3"
    assert dataset["test"][-1].instance_id == "test-wt40-1m-10"

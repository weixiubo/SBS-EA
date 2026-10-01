from __future__ import annotations

from pairwise_bpp_poc.jssp import JSSPInstance, JSSPOperation
from pairwise_bpp_poc.public_jsplib import (
    build_jsplib_dataset,
    build_jsplib_rotation_datasets,
    jsplib_rotation_capacity,
)


def _instance(family: str, name: str) -> JSSPInstance:
    return JSSPInstance(
        family=family,
        split="public",
        seed=0,
        instance_id=name,
        jobs=2,
        machines=2,
        operations=(
            (JSSPOperation(machine_id=0, duration=1), JSSPOperation(machine_id=1, duration=2)),
            (JSSPOperation(machine_id=1, duration=2), JSSPOperation(machine_id=0, duration=1)),
        ),
        optimum=3,
        upper_bound=3,
    )


def test_build_jsplib_dataset_supports_family_rotation():
    instances = [
        _instance("abz", "abz1"),
        _instance("abz", "abz2"),
        _instance("abz", "abz3"),
        _instance("ft", "ft1"),
        _instance("ft", "ft2"),
        _instance("ft", "ft3"),
    ]
    default_dataset = build_jsplib_dataset(instances, smoke_per_family=1, val_per_family=1, test_per_family=1)
    rotated_dataset = build_jsplib_dataset(
        instances,
        smoke_per_family=1,
        val_per_family=1,
        test_per_family=1,
        family_rotation=1,
    )
    assert [row.instance_id for row in default_dataset["smoke"]] == ["smoke-abz-abz1", "smoke-ft-ft1"]
    assert [row.instance_id for row in rotated_dataset["smoke"]] == ["smoke-abz-abz2", "smoke-ft-ft2"]
    assert [row.instance_id for row in rotated_dataset["test"]] == ["test-abz-abz1", "test-ft-ft1"]


def test_build_jsplib_rotation_datasets_uses_min_family_capacity():
    instances = [
        _instance("abz", "abz1"),
        _instance("abz", "abz2"),
        _instance("abz", "abz3"),
        _instance("ft", "ft1"),
        _instance("ft", "ft2"),
        _instance("ft", "ft3"),
    ]
    assert jsplib_rotation_capacity(instances, smoke_per_family=1, val_per_family=1, test_per_family=1) == 3
    datasets = build_jsplib_rotation_datasets(instances, smoke_per_family=1, val_per_family=1, test_per_family=1)
    assert len(datasets) == 3
    assert [row.instance_id for row in datasets[2]["smoke"]] == ["smoke-abz-abz3", "smoke-ft-ft3"]

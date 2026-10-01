from dataclasses import replace

from pairwise_bpp_poc.config import default_config
from pairwise_bpp_poc.registry import RunRegistry


def test_registry_hash_changes_when_stage_a_config_changes(tmp_path):
    registry = RunRegistry(tmp_path / "registry.jsonl")
    base = default_config(tmp_path)
    modified = replace(
        base,
        stage_a=replace(base.stage_a, confidence_floor=0.9),
    )
    left = registry.compute_hash({"stage": "stage_a", "config": base.to_dict()})
    right = registry.compute_hash({"stage": "stage_a", "config": modified.to_dict()})
    assert left != right

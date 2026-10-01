from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .jssp import JSSPInstance, JSSPOperation


GENERATED_JSSP_SPEC_VERSION = "generated_jssp_holdout_v1"


@dataclass(frozen=True)
class GeneratedJSSPSpec:
    family: str
    jobs: int
    machines: int
    duration_mode: str
    seed_offset: int


def generated_jssp_specs() -> tuple[GeneratedJSSPSpec, ...]:
    return (
        GeneratedJSSPSpec(
            family="gen_uniform_10x10",
            jobs=10,
            machines=10,
            duration_mode="uniform",
            seed_offset=11000,
        ),
        GeneratedJSSPSpec(
            family="gen_uniform_15x15",
            jobs=15,
            machines=15,
            duration_mode="uniform",
            seed_offset=15000,
        ),
        GeneratedJSSPSpec(
            family="gen_bimodal_20x10",
            jobs=20,
            machines=10,
            duration_mode="bimodal",
            seed_offset=20100,
        ),
        GeneratedJSSPSpec(
            family="gen_bottleneck_15x15",
            jobs=15,
            machines=15,
            duration_mode="bottleneck",
            seed_offset=15150,
        ),
    )


def build_generated_jssp_holdout(
    *,
    instances_per_family: int = 25,
) -> list[JSSPInstance]:
    instances: list[JSSPInstance] = []
    for spec in generated_jssp_specs():
        for local_index in range(instances_per_family):
            seed = spec.seed_offset + local_index
            instances.append(generate_jssp_instance(spec, seed=seed, local_index=local_index))
    return instances


def generate_jssp_instance(
    spec: GeneratedJSSPSpec,
    *,
    seed: int,
    local_index: int,
) -> JSSPInstance:
    rng = Random(seed)
    bottleneck_machines = _bottleneck_machines(spec, rng)
    job_operations: list[tuple[JSSPOperation, ...]] = []
    for job_id in range(spec.jobs):
        machine_order = list(range(spec.machines))
        rng.shuffle(machine_order)
        job_scale = rng.uniform(0.75, 1.35)
        operations: list[JSSPOperation] = []
        for op_index, machine_id in enumerate(machine_order):
            duration = _sample_duration(
                rng,
                mode=spec.duration_mode,
                job_scale=job_scale,
                machine_id=machine_id,
                operation_index=op_index,
                bottleneck_machines=bottleneck_machines,
            )
            operations.append(JSSPOperation(machine_id=machine_id, duration=duration))
        job_operations.append(tuple(operations))
    operations_tuple = tuple(job_operations)
    return JSSPInstance(
        family=spec.family,
        split="generated_holdout",
        seed=seed,
        instance_id=f"generated-{spec.family}-{local_index:03d}",
        jobs=spec.jobs,
        machines=spec.machines,
        operations=operations_tuple,
        optimum=None,
        lower_bound=_jssp_lower_bound(operations_tuple, spec.machines),
        upper_bound=None,
    )


def _bottleneck_machines(spec: GeneratedJSSPSpec, rng: Random) -> frozenset[int]:
    if spec.duration_mode != "bottleneck":
        return frozenset()
    count = max(1, spec.machines // 5)
    machines = list(range(spec.machines))
    rng.shuffle(machines)
    return frozenset(machines[:count])


def _sample_duration(
    rng: Random,
    *,
    mode: str,
    job_scale: float,
    machine_id: int,
    operation_index: int,
    bottleneck_machines: frozenset[int],
) -> int:
    if mode == "uniform":
        raw = rng.randint(1, 99)
    elif mode == "bimodal":
        raw = rng.randint(70, 140) if rng.random() < 0.25 else rng.randint(1, 35)
    elif mode == "bottleneck":
        if machine_id in bottleneck_machines:
            raw = rng.randint(70, 150)
        else:
            raw = rng.randint(5, 55)
        if operation_index % 4 == 0:
            raw += rng.randint(0, 30)
    else:
        raise ValueError(f"unknown_generated_duration_mode:{mode}")
    return max(1, int(round(raw * job_scale)))


def _jssp_lower_bound(
    operations: tuple[tuple[JSSPOperation, ...], ...],
    machines: int,
) -> int:
    job_loads = [sum(operation.duration for operation in job) for job in operations]
    machine_loads = [0] * machines
    for job in operations:
        for operation in job:
            machine_loads[operation.machine_id] += operation.duration
    return max(max(job_loads, default=0), max(machine_loads, default=0))

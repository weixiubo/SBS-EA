"""Selection-only PFSP probe trajectory used to deduplicate atomic expressions."""

from typing import Any

from .pfsp import PFSPInstance, _append_job_completion, pfsp_ready_options


def collect_probe_states(
    instances: list[PFSPInstance], *, max_states: int = 64
) -> list[list[dict[str, Any]]]:
    states: list[list[dict[str, Any]]] = []
    for instance in instances:
        machine_completion = [0] * instance.machines
        unscheduled = list(range(instance.jobs))
        for _step in range(min(instance.jobs, 8)):
            ready = pfsp_ready_options(instance, unscheduled, machine_completion)
            if len(ready) >= 2:
                states.append(ready)
            choice = min(range(len(ready)), key=lambda index: ready[index]["completion_if_next"])
            job_id = int(ready[choice]["job_id"])
            machine_completion = _append_job_completion(machine_completion, instance.processing_times[job_id])
            unscheduled.remove(job_id)
            if not unscheduled:
                break
            if len(states) >= max_states:
                return states
    return states

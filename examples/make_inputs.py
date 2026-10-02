"""Write explicit tiny dataset and proposal examples to a requested directory."""

import argparse
import hashlib
import json
from pathlib import Path

from sbs_ea.datasets import fixture_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    dataset = {
        "schema": 1,
        "splits": {
            split: [
                {
                    "family": i.family,
                    "instance_id": i.instance_id,
                    "processing_times": i.processing_times,
                    "best_known_makespan": i.best_known_makespan,
                    "reference_source": "NEH fixture upper bound",
                }
                for i in rows
            ]
            for split, rows in fixture_dataset().items()
        },
    }
    proposal = {
        "features": ["total_work", "completion_if_next", "completion_increase", "lookahead_pressure"],
        "interactions": [["total_work", "completion_if_next"]],
    }
    output = {}
    for name, value in (("dataset.json", dataset), ("proposal.json", proposal)):
        path = args.directory / name
        path.write_text(json.dumps(value, indent=2) + "\n")
        output[name] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

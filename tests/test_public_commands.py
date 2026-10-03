import hashlib
import json

import pytest

from sbs_ea.cli import main
from sbs_ea.datasets import fixture_dataset
from sbs_ea.execution import demo, prepare, run
from sbs_ea.settings import read_config


def configuration(tmp_path, **changes):
    raw = {
        "schema": 1,
        "run_id": "test-run",
        "paths": {"runs": str(tmp_path / "runs")},
        "seeds": [77],
        "pool_limit": 12,
        "sources": [{"kind": "all_feature_prior"}],
        "search": {
            "population_size": 12,
            "generations": 4,
            "elite_count": 2,
            "mutation_count": 6,
            "crossover_count": 3,
            "random_count": 1,
            "max_candidate_evaluations": 32,
        },
    }
    raw.update(changes)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    return path


def bound(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def public_dataset():
    return {
        "schema": 1,
        "splits": {
            split: [
                {
                    "family": r.family,
                    "instance_id": r.instance_id,
                    "processing_times": r.processing_times,
                    "best_known_makespan": r.best_known_makespan,
                    "reference_source": "NEH fixture upper bound",
                }
                for r in rows
            ]
            for split, rows in fixture_dataset().items()
        },
    }


def test_demo_executes_both_searches_with_exact_budget_without_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = demo()
    assert (
        result["ea"]["evaluated_candidate_count"] == result["no_operator"]["evaluated_candidate_count"] == 32
    )
    assert result["historical_evidence"] is False
    assert not list(tmp_path.iterdir())


def test_configured_run_and_verified_resume_from_another_directory(tmp_path, monkeypatch):
    path = configuration(tmp_path)
    cwd = tmp_path / "outside"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    config = read_config(path)
    result = run(config)
    assert len(result["rows"]) == 2
    assert all(row["evaluated_candidate_count"] == 32 for row in result["rows"])
    monkeypatch.setattr("sbs_ea.execution.execute", lambda *_: pytest.fail("resume reexecuted search"))
    assert run(config, resume=True) == json.loads(json.dumps(result))
    assert not list(cwd.iterdir())
    with pytest.raises(ValueError, match="already exists"):
        run(config)


def test_tampered_complete_result_and_partial_run_are_rejected(tmp_path):
    config = read_config(configuration(tmp_path))
    run(config)
    (config.output / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        run(config, resume=True)
    (config.output / "complete.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        run(config, resume=True)


def test_failed_search_has_a_failure_record_and_no_completion(tmp_path, monkeypatch):
    config = read_config(configuration(tmp_path))

    def fail(*_):
        raise RuntimeError("injected evaluation failure")

    monkeypatch.setattr("sbs_ea.execution.execute", fail)
    with pytest.raises(RuntimeError, match="injected"):
        run(config)
    assert json.loads((config.output / "failure.json").read_text())["type"] == "RuntimeError"
    assert not (config.output / "complete.json").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": "../escape"},
        {"seeds": [True]},
        {"seeds": []},
        {"seeds": [1, 1]},
        {"modes": ["unsupported"]},
        {"modes": ["population_ea", "population_ea"]},
        {"pool_limit": 0},
        {"schema": 3},
        {"typo": 1},
        {"paths": {"runns": "x"}},
        {"sources": [{"kind": "llm_prior"}]},
        {"sources": [{"kind": "unknown"}]},
        {"search": {"robust_se_weight": float("nan")}},
        {"search": {"mutation_count": -1}},
        {"sources": [{"kind": "all_feature_prior", "include_macros": "no"}]},
    ],
)
def test_invalid_config_fails_before_outputs(tmp_path, changes):
    with pytest.raises(ValueError):
        read_config(configuration(tmp_path, **changes))
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("kind", ["missing", "tampered", "malformed", "split_leak"])
def test_invalid_dataset_fails_before_run_directory(tmp_path, kind):
    data = tmp_path / "data.json"
    content = public_dataset()
    if kind == "malformed":
        content["splits"]["smoke"][0]["processing_times"] = [[0]]
    if kind == "split_leak":
        content["splits"]["test"][0]["instance_id"] = content["splits"]["smoke"][0]["instance_id"]
    data.write_text(json.dumps(content))
    binding = bound(data)
    if kind == "missing":
        data.unlink()
    if kind == "tampered":
        data.write_text("{}")
    config = read_config(configuration(tmp_path, dataset=binding))
    with pytest.raises(ValueError):
        run(config)
    assert not config.output.exists()


def test_hash_bound_external_data_and_cached_model_source(tmp_path):
    data = tmp_path / "data.json"
    data.write_text(json.dumps(public_dataset()))
    proposal = tmp_path / "proposal.json"
    proposal.write_text(
        json.dumps(
            {
                "features": [
                    "total_work",
                    "completion_if_next",
                    "completion_increase",
                    "lookahead_pressure",
                    "head_tail_balance",
                ],
                "interactions": [["total_work", "completion_if_next"]],
            }
        )
    )
    config = read_config(
        configuration(
            tmp_path,
            dataset=bound(data),
            sources=[{"kind": "llm_prior", "proposal": bound(proposal)}, {"kind": "all_feature_prior"}],
        )
    )
    _, prepared, identity = prepare(config)
    assert len(prepared) == 2 and len(prepared[0].pool) == len(prepared[1].pool)
    assert identity["dataset_sha256"] == bound(data)["sha256"]
    result = run(config)
    assert len(result["rows"]) == 4
    assert all(row["model_calls_during_search"] == 0 for row in result["rows"])


def test_empty_model_prior_is_rejected_before_output(tmp_path):
    proposal = tmp_path / "proposal.json"
    proposal.write_text('{"features":[]}')
    config = read_config(
        configuration(tmp_path, sources=[{"kind": "llm_prior", "proposal": bound(proposal)}])
    )
    with pytest.raises(ValueError, match="no valid informative"):
        run(config)
    assert not config.output.exists()


def test_doctor_uses_config_and_cli_resource_precedence(tmp_path, capsys):
    path = configuration(tmp_path)
    override = tmp_path / "override"
    assert main(["doctor", "--config", str(path), "--runs-dir", str(override)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["configured_run_ready"] is True
    assert result["paths"]["runs"] == str(override)
    assert not override.exists()

import json
from pathlib import Path

import pytest

from examples.search_smoke import characterize
from sbs_ea.grammar import safe_eval_expr


def test_search_matches_pre_refactor_baseline():
    expected = json.loads((Path(__file__).parent / "fixtures/pfsp_search_baseline.json").read_text())
    assert json.loads(json.dumps(characterize())) == expected


@pytest.mark.parametrize("expr", ["__import__('os')", "x.__class__", "x[0]", "x / x"])
def test_compiler_rejects_unsafe_or_unprotected_expressions(expr):
    with pytest.raises(ValueError):
        safe_eval_expr(expr, [{"x": 1.0}, {"x": 2.0}])

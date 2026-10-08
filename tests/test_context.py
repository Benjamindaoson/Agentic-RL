import pytest

from agentic_rl_sql.context import fit_schema_to_budget


def test_schema_is_trimmed_but_question_not_removed():
    counter = lambda messages: sum(len(m["content"]) for m in messages)
    make = lambda schema: [{"role": "user", "content": "QUESTION critical\nSCHEMA " + schema}]
    messages, count, truncated = fit_schema_to_budget(make, "x" * 5000, 120, counter)
    assert truncated and count <= 120
    assert "QUESTION critical" in messages[0]["content"]
    assert "TRUNCATED" in messages[0]["content"]


def test_over_budget_question_is_an_error():
    counter = lambda messages: sum(len(m["content"]) for m in messages)
    with pytest.raises(ValueError, match="question/history"):
        fit_schema_to_budget(lambda s: [{"role": "user", "content": "x" * 2000 + s}], "", 100, counter)

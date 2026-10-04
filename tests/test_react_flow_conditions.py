"""ReactFlowParser condition nodes: structured field conditions."""

import pytest

from mesh import NodeRegistry
from mesh.parsers.react_flow import ReactFlowParser
from mesh.utils.errors import GraphValidationError


def _predicate(**cond):
    parser = ReactFlowParser(NodeRegistry())
    node = parser._create_condition_node(
        "c", {"conditions": [{"name": "n", "target": "t", **cond}]}
    )
    return node.conditions[0].predicate


@pytest.mark.parametrize(
    "cond, value, expected",
    [
        ({"field": "kind", "operation": "equal", "value": "a"}, {"kind": "a"}, True),
        ({"field": "kind", "operation": "equal", "value": "a"}, {"kind": "b"}, False),
        ({"field": "kind", "operation": "notEqual", "value": "a"}, {"kind": "b"}, True),
        ({"field": "tags", "operation": "contains", "value": "x"}, {"tags": ["x"]}, True),
        ({"field": "tags", "operation": "contains", "value": "x"}, {}, False),
        ({"field": "missing", "operation": "isEmpty"}, {"missing": []}, True),
        ({"field": "missing", "operation": "isEmpty"}, {}, True),
        ({"field": "missing", "operation": "notEmpty"}, {"missing": ["geo"]}, True),
        ({"field": "a.b", "operation": "equal", "value": 1}, {"a": {"b": 1}}, True),
    ],
)
def test_field_conditions(cond, value, expected):
    assert _predicate(**cond)(value) is expected


def test_unknown_operation_fails_at_parse_time():
    with pytest.raises(GraphValidationError, match="unknown operation"):
        _predicate(field="kind", operation="matches", value="a")


def test_field_condition_requires_a_field():
    with pytest.raises(GraphValidationError, match="needs a 'field'"):
        _predicate(operation="isEmpty")


def test_expression_conditions_still_parse():
    assert _predicate(expression="contains('ok')")({"status": "ok"}) is True

"""Tool: калькулятор (безопасный eval арифметики)."""

from __future__ import annotations

import ast
import operator
from typing import Any

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Вычислить арифметическое выражение. "
            "Поддерживаются + - * / ** // % и скобки. Без переменных и вызовов функций."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Выражение, например: (15*7)+3",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    raise ValueError("Разрешена только арифметика (+ - * / ** // % и скобки)")


def calculator(expression: str) -> dict[str, Any]:
    expr = (expression or "").strip()
    if not expr:
        return {"ok": False, "error": "Пустое expression"}
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_node(tree)
        if value == int(value):
            value = int(value)
        return {"ok": True, "expression": expr, "result": value}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "expression": expr, "error": str(exc)}

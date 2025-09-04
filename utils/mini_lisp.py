"""
A tiny, beginner-friendly Lisp interpreter in pure Python.

Features:
- Numbers (ints, floats), strings, booleans (#t/#f), symbols, lists
- Special forms: quote ('), if, define, set!, lambda, begin, and, or
- Builtins: + - * /, < <= > >=, =, eq?, equal?, cons, car, cdr, list, length,
  null?, list?, symbol?, number?, boolean?, pair?, not, display, newline
- REPL and file execution: `python -m utils.mini_lisp` or `python utils/mini_lisp.py file.lisp`

This module is intentionally small and readable for learning purposes.
"""

from __future__ import annotations

import math
import operator
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union


# Types
class Symbol(str):
    """Represents a Lisp symbol (identifier). Distinct from Python strings used for Lisp strings."""


LispNumber = Union[int, float]
LispBoolean = bool
LispString = str  # Used for string literals
LispList = list
LispAtom = Union[Symbol, LispNumber, LispBoolean, LispString]
LispExpr = Union[LispAtom, LispList]


class Env(dict):
    """Environment mapping symbols to values, with optional outer (parent) env for lexical scoping."""

    def __init__(self, bindings: Optional[Dict[Symbol, Any]] = None, outer: Optional["Env"] = None):
        super().__init__(bindings or {})
        self.outer = outer

    def find(self, key: Symbol) -> "Env":
        if key in self:
            return self
        if self.outer is not None:
            return self.outer.find(key)
        raise NameError(f"unbound symbol: {key}")


def _is_true(value: Any) -> bool:
    return value is not False and value is not None


def standard_env() -> Env:
    """Create the initial environment with standard procedures and constants."""
    env: Dict[Symbol, Any] = {}

    # Math
    env[Symbol("+")] = lambda *args: sum(args)
    env[Symbol("-")] = lambda a, *rest: (-a if not rest else a - sum(rest))
    env[Symbol("*")] = lambda *args: math.prod(args) if args else 1
    env[Symbol("/")] = lambda a, *rest: (1 / a if not rest else _div_seq(a, rest))
    env[Symbol("sqrt")] = math.sqrt
    env[Symbol("pow")] = pow

    # Comparison
    env[Symbol("<")] = lambda a, b: a < b
    env[Symbol("<=")] = lambda a, b: a <= b
    env[Symbol(">")] = lambda a, b: a > b
    env[Symbol(">=")] = lambda a, b: a >= b
    env[Symbol("=")] = lambda a, b: a == b
    env[Symbol("eq?")] = lambda a, b: a is b
    env[Symbol("equal?")] = lambda a, b: a == b

    # Lists
    env[Symbol("cons")] = lambda a, b: [a] + list(b)
    env[Symbol("car")] = lambda x: x[0]
    env[Symbol("cdr")] = lambda x: x[1:]
    env[Symbol("list")] = lambda *args: list(args)
    env[Symbol("length")] = lambda x: len(x)
    env[Symbol("null?")] = lambda x: x == []
    env[Symbol("list?")] = lambda x: isinstance(x, list)
    env[Symbol("symbol?")] = lambda x: isinstance(x, Symbol)
    env[Symbol("number?")] = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    env[Symbol("boolean?")] = lambda x: isinstance(x, bool)
    env[Symbol("pair?")] = lambda x: isinstance(x, list) and len(x) > 0

    # Logic and utilities
    env[Symbol("not")] = lambda x: not _is_true(x)
    env[Symbol("display")] = _builtin_display
    env[Symbol("newline")] = _builtin_newline
    env[Symbol("load")] = _builtin_load

    # Constants
    env[Symbol("#t")] = True
    env[Symbol("#f")] = False

    return Env(env)


def _div_seq(a: LispNumber, rest: Sequence[LispNumber]) -> LispNumber:
    result: float = float(a)
    for r in rest:
        result /= r
    return result


def _builtin_display(*args: Any) -> None:
    text = "".join(_to_display_string(arg) for arg in args)
    print(text, end="")


def _builtin_newline() -> None:
    print("")


def _to_display_string(value: Any) -> str:
    if isinstance(value, list):
        return "(" + " ".join(_to_display_string(v) for v in value) + ")"
    if isinstance(value, bool):
        return "#t" if value else "#f"
    return str(value)


# Tokenizer and parser
def tokenize(source: str) -> List[str]:
    """Convert a Lisp source string into a list of tokens."""
    tokens: List[str] = []
    i = 0
    length = len(source)
    while i < length:
        ch = source[i]
        if ch in (" ", "\n", "\t", "\r"):
            i += 1
            continue
        if ch == ";":
            while i < length and source[i] != "\n":
                i += 1
            continue
        if ch == "(":
            tokens.append("(")
            i += 1
            continue
        if ch == ")":
            tokens.append(")")
            i += 1
            continue
        if ch == "'":
            tokens.append("'")
            i += 1
            continue
        if ch == '"':
            i += 1
            start = i
            buf: List[str] = []
            while i < length and source[i] != '"':
                if source[i] == "\\":
                    i += 1
                    if i < length:
                        esc = source[i]
                        mapping = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}
                        buf.append(mapping.get(esc, esc))
                        i += 1
                        continue
                buf.append(source[i])
                i += 1
            if i >= length or source[i] != '"':
                raise SyntaxError("unterminated string literal")
            i += 1  # skip closing quote
            tokens.append('"' + "".join(buf) + '"')
            continue
        # symbol or number
        start = i
        while i < length and source[i] not in (" ", "\n", "\t", "\r", "(", ")"):
            i += 1
        tokens.append(source[start:i])
    return tokens


def parse(program: str) -> LispExpr:
    return _read_from_tokens(tokenize(program))


def parse_many(program: str) -> List[LispExpr]:
    tokens = tokenize(program)
    exprs: List[LispExpr] = []
    while tokens:
        exprs.append(_read_from_tokens(tokens))
    return exprs


def _read_from_tokens(tokens: List[str]) -> LispExpr:
    if not tokens:
        raise SyntaxError("unexpected EOF while reading")
    token = tokens.pop(0)
    if token == "(":
        lst: List[LispExpr] = []
        while tokens and tokens[0] != ")":
            lst.append(_read_from_tokens(tokens))
        if not tokens:
            raise SyntaxError("missing ')' in list")
        tokens.pop(0)
        return lst
    if token == ")":
        raise SyntaxError("unexpected ')'")
    if token == "'":
        return [Symbol("quote"), _read_from_tokens(tokens)]
    return _atom(token)


def _atom(token: str) -> LispAtom:
    if token.startswith('"') and token.endswith('"') and len(token) >= 2:
        return token[1:-1]
    if token == "#t":
        return True
    if token == "#f":
        return False
    try:
        if token.startswith("0x") or token.startswith("-0x"):
            return int(token, 16)
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return Symbol(token)


@dataclass
class Procedure:
    params: List[Symbol]
    body: List[LispExpr]
    env: Env

    def __call__(self, *args: Any) -> Any:
        if len(args) != len(self.params):
            raise TypeError(f"expected {len(self.params)} args, got {len(args)}")
        local_bindings = {param: arg for param, arg in zip(self.params, args)}
        call_env = Env(local_bindings, self.env)
        result: Any = None
        for expr in self.body:
            result = evaluate(expr, call_env)
        return result


def evaluate(x: LispExpr, env: Env) -> Any:
    if isinstance(x, Symbol):
        return env.find(x)[x]
    if not isinstance(x, list):
        return x
    if len(x) == 0:
        return []

    op = x[0]
    if isinstance(op, list):
        op = evaluate(op, env)
    if isinstance(op, Symbol):
        # Special forms
        if op == Symbol("quote"):
            (_quote, value) = x
            return value
        if op == Symbol("if"):
            (_, test, conseq, alt) = x
            branch = conseq if _is_true(evaluate(test, env)) else alt
            return evaluate(branch, env)
        if op == Symbol("define"):
            if isinstance(x[1], list):
                # (define (name args...) body...)
                _, sig, *body = x
                name = Symbol(sig[0])
                params = [Symbol(p) for p in sig[1:]]
                proc = Procedure(params, body, env)
                env[name] = proc
                return name
            else:
                (_, name, expr) = x
                value = evaluate(expr, env)
                env[name] = value
                return name
        if op == Symbol("set!"):
            (_, name, expr) = x
            env.find(name)[name] = evaluate(expr, env)
            return name
        if op == Symbol("lambda"):
            (_, params_list, *body) = x
            params = [Symbol(p) for p in params_list]
            return Procedure(params, body, env)
        if op == Symbol("begin"):
            result: Any = None
            for expr in x[1:]:
                result = evaluate(expr, env)
            return result
        if op == Symbol("and"):
            for expr in x[1:]:
                val = evaluate(expr, env)
                if not _is_true(val):
                    return False
            return True
        if op == Symbol("or"):
            for expr in x[1:]:
                val = evaluate(expr, env)
                if _is_true(val):
                    return val
            return False

    # Procedure call
    proc = evaluate(op, env) if isinstance(op, Symbol) else op
    args = [evaluate(arg, env) for arg in x[1:]]
    if callable(proc):
        return proc(*args)
    raise TypeError(f"attempt to call non-procedure: {proc}")


def _builtin_load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    exprs = parse_many(src)
    result: Any = None
    env = _GLOBAL_ENV
    for expr in exprs:
        result = evaluate(expr, env)
    return result


def repl() -> None:
    """A simple Read-Eval-Print Loop with multi-line input support."""
    env = _GLOBAL_ENV
    buffer: List[str] = []
    prompt = "lisp> "
    cont_prompt = "...   "
    try:
        while True:
            line = input(prompt if not buffer else cont_prompt)
            buffer.append(line)
            joined = "\n".join(buffer)
            if _paren_balance(joined) > 0:
                continue
            if _paren_balance(joined) < 0:
                print("SyntaxError: too many ')'")
                buffer.clear()
                continue
            try:
                for expr in parse_many(joined):
                    val = evaluate(expr, env)
                    if val is not None:
                        print(_to_display_string(val))
                buffer.clear()
            except Exception as e:  # noqa: BLE001
                print(f"Error: {e}")
                buffer.clear()
    except (EOFError, KeyboardInterrupt):
        print("")


def _paren_balance(s: str) -> int:
    bal = 0
    in_string = False
    escaped = False
    for ch in s:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "(":
            bal += 1
        elif ch == ")":
            bal -= 1
    return bal


_GLOBAL_ENV: Env = standard_env()


def _main(argv: List[str]) -> int:
    if len(argv) > 1:
        path = argv[1]
        try:
            _builtin_load(path)
        except Exception as e:  # noqa: BLE001
            print(f"Error: {e}")
            return 1
        return 0
    repl()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))



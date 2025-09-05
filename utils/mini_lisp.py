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

import importlib
import math
import sys
import time
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
    env_dict: Dict[Symbol, Any] = {}

    # Math
    env_dict[Symbol("+")] = lambda *args: sum(args)
    env_dict[Symbol("-")] = lambda a, *rest: (-a if not rest else a - sum(rest))
    env_dict[Symbol("*")] = lambda *args: math.prod(args) if args else 1
    env_dict[Symbol("/")] = lambda a, *rest: (1 / a if not rest else _div_seq(a, rest))
    env_dict[Symbol("sqrt")] = math.sqrt
    env_dict[Symbol("pow")] = pow

    # Comparison
    env_dict[Symbol("<")] = lambda a, b: a < b
    env_dict[Symbol("<=")] = lambda a, b: a <= b
    env_dict[Symbol(">")] = lambda a, b: a > b
    env_dict[Symbol(">=")] = lambda a, b: a >= b
    env_dict[Symbol("=")] = lambda a, b: a == b
    env_dict[Symbol("eq?")] = lambda a, b: a is b
    env_dict[Symbol("equal?")] = lambda a, b: a == b

    # Lists
    env_dict[Symbol("cons")] = lambda a, b: [a] + list(b)
    env_dict[Symbol("car")] = lambda x: x[0]
    env_dict[Symbol("cdr")] = lambda x: x[1:]
    env_dict[Symbol("list")] = lambda *args: list(args)
    env_dict[Symbol("length")] = lambda x: len(x)
    env_dict[Symbol("null?")] = lambda x: x == []
    env_dict[Symbol("list?")] = lambda x: isinstance(x, list)
    env_dict[Symbol("symbol?")] = lambda x: isinstance(x, Symbol)
    env_dict[Symbol("number?")] = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    env_dict[Symbol("boolean?")] = lambda x: isinstance(x, bool)
    env_dict[Symbol("pair?")] = lambda x: isinstance(x, list) and len(x) > 0

    # Logic and utilities
    env_dict[Symbol("not")] = lambda x: not _is_true(x)
    env_dict[Symbol("display")] = _builtin_display
    env_dict[Symbol("newline")] = _builtin_newline
    env_dict[Symbol("load")] = _builtin_load

    # Constants
    env_dict[Symbol("#t")] = True
    env_dict[Symbol("#f")] = False

    env = Env(env_dict)

    # Advanced utilities (added after Env is created to capture env by closure)
    env[Symbol("simplify")] = _builtin_simplify
    env[Symbol("eval")] = lambda expr: evaluate(expr, env)
    env[Symbol("eval-simplified")] = lambda expr: evaluate(simplify(expr), env)
    env[Symbol("eval-with-limits")] = lambda expr, steps, timeout_ms, do_simplify: evaluate_limited(
        simplify(expr) if do_simplify else expr,
        env,
        max_steps=int(steps) if isinstance(steps, (int, float)) else None,
        time_limit_ms=int(timeout_ms) if isinstance(timeout_ms, (int, float)) else None,
        simplify_before=False,
        trace=_TRACE_DEFAULT,
    )
    env[Symbol("trace-on")] = lambda: _set_trace_default(True)
    env[Symbol("trace-off")] = lambda: _set_trace_default(False)
    env[Symbol("use")] = _builtin_use

    return env


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


def _trace_line(depth: int, expr: LispExpr) -> str:
    indent = "  " * max(depth, 0)
    return f"{indent}{_to_display_string(expr)}"


# --------------------------------------------
# Simplification utilities
# --------------------------------------------

def simplify(expr: LispExpr) -> LispExpr:
    """Symbolically simplify a Lisp expression (pure, no environment).

    Examples of rules:
    - (+ 0 x) -> x, (* 1 x) -> x, (* 0 x) -> 0
    - Constant folding for +, -, *, / when all args are numbers
    - (if #t a b) -> a, (if #f a b) -> b
    - Flatten nested +, * and begin
    """
    return _simplify(expr)


def _simplify(expr: LispExpr) -> LispExpr:
    if not isinstance(expr, list):
        return expr
    if len(expr) == 0:
        return expr

    op = expr[0]
    # Do not simplify inside quotes
    if isinstance(op, Symbol) and op == Symbol("quote"):
        return expr

    # Recursively simplify children (operator may be an expression)
    simp_children = [
        _simplify(c) if (i == 0 and isinstance(c, list)) or i > 0 else c
        for i, c in enumerate(expr)
    ]
    op = simp_children[0]

    # Special forms
    if isinstance(op, Symbol):
        if op == Symbol("if") and len(simp_children) == 4:
            (_, test, conseq, alt) = simp_children
            if isinstance(test, bool):
                return _simplify(conseq if test else alt)
            return [op, test, _simplify(conseq), _simplify(alt)]
        if op == Symbol("begin"):
            items: List[LispExpr] = []
            for part in simp_children[1:]:
                if isinstance(part, list) and len(part) > 0 and isinstance(part[0], Symbol) and part[0] == Symbol("begin"):
                    items.extend(part[1:])
                else:
                    items.append(part)
            if len(items) == 0:
                return []
            if len(items) == 1:
                return _simplify(items[0])
            return [Symbol("begin"), *[ _simplify(i) for i in items ]]
        if op == Symbol("and"):
            parts: List[LispExpr] = []
            for p in simp_children[1:]:
                if p is False:
                    return False
                if p is True:
                    continue
                parts.append(p)
            if not parts:
                return True
            if len(parts) == 1:
                return parts[0]
            return [Symbol("and"), *parts]
        if op == Symbol("or"):
            parts2: List[LispExpr] = []
            for p in simp_children[1:]:
                if p is True:
                    return True
                if p is False:
                    continue
                parts2.append(p)
            if not parts2:
                return False
            if len(parts2) == 1:
                return parts2[0]
            return [Symbol("or"), *parts2]

    # Procedure-like simplifications for arithmetic
    if isinstance(op, Symbol) and op in (Symbol("+"), Symbol("*")):
        args = [ _simplify(a) for a in simp_children[1:] ]
        flat: List[LispExpr] = []
        for a in args:
            if isinstance(a, list) and len(a) > 0 and isinstance(a[0], Symbol) and a[0] == op:
                flat.extend(a[1:])
            else:
                flat.append(a)
        if op == Symbol("+"):
            # remove zeros
            flat = [a for a in flat if not _is_number(a) or not _is_zero(a)]
            if not flat:
                return 0
            if len(flat) == 1:
                return flat[0]
            if all(_is_number(a) for a in flat):
                try:
                    return sum(a for a in flat)  # type: ignore[arg-type]
                except Exception:
                    return [op, *flat]
            return [op, *flat]
        if op == Symbol("*"):
            # zero rule
            if any(_is_number(a) and _is_zero(a) for a in flat):
                return 0
            # remove ones
            flat = [a for a in flat if not (_is_number(a) and _is_one(a))]
            if not flat:
                return 1
            if len(flat) == 1:
                return flat[0]
            if all(_is_number(a) for a in flat):
                try:
                    prod: float = 1
                    for a in flat:
                        prod *= float(a)  # type: ignore[arg-type]
                    # cast back to int if integral
                    if float(prod).is_integer():
                        return int(prod)
                    return prod
                except Exception:
                    return [op, *flat]
            return [op, *flat]

    if isinstance(op, Symbol) and op in (Symbol("-"), Symbol("/")):
        args = [ _simplify(a) for a in simp_children[1:] ]
        if op == Symbol("-"):
            if len(args) == 1 and _is_number(args[0]):
                return -args[0]  # type: ignore[operator]
            if len(args) == 2 and _is_number(args[1]) and _is_zero(args[1]):
                return args[0]
            if all(_is_number(a) for a in args):
                try:
                    a0 = args[0]
                    rest = args[1:]
                    total = float(a0)  # type: ignore[arg-type]
                    for r in rest:
                        total -= float(r)  # type: ignore[arg-type]
                    if float(total).is_integer():
                        return int(total)
                    return total
                except Exception:
                    return [op, *args]
            return [op, *args]
        if op == Symbol("/"):
            if len(args) == 2 and _is_number(args[1]) and _is_one(args[1]):
                return args[0]
            if len(args) == 2 and _is_number(args[0]) and _is_zero(args[0]):
                return 0
            if len(args) == 1 and _is_number(args[0]):
                if args[0] == 0:
                    return [op, *args]
                try:
                    return 1 / float(args[0])  # type: ignore[arg-type]
                except Exception:
                    return [op, *args]
            if all(_is_number(a) for a in args) and len(args) >= 2:
                try:
                    total = float(args[0])  # type: ignore[arg-type]
                    for r in args[1:]:
                        total /= float(r)  # type: ignore[arg-type]
                    if float(total).is_integer():
                        return int(total)
                    return total
                except Exception:
                    return [op, *args]
            return [op, *args]

    # Fallback: rebuild with simplified children
    return [simp_children[0], *[ _simplify(a) for a in simp_children[1:] ]]


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_zero(x: Any) -> bool:
    return _is_number(x) and float(x) == 0.0


def _is_one(x: Any) -> bool:
    return _is_number(x) and float(x) == 1.0


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


# --------------------------------------------
# Evaluation context and errors
# --------------------------------------------

@dataclass
class _EvalContext:
    step_budget: Optional[int]
    deadline_monotonic: Optional[float]
    trace: bool
    stack: List[LispExpr]


class LispError(Exception):
    def __init__(self, message: str, stack: Optional[List[LispExpr]] = None):
        super().__init__(message)
        self.stack = list(stack or [])

    def format_stack(self) -> str:
        if not self.stack:
            return ""
        lines = ["Lisp stack (most recent call last):"]
        for expr in reversed(self.stack[-20:]):
            lines.append("  " + _to_display_string(expr))
        return "\n".join(lines)


class LispTimeoutError(LispError):
    pass


class LispStepLimitError(LispError):
    pass


_CURRENT_EVAL_CONTEXT: Optional[_EvalContext] = None
_TRACE_DEFAULT: bool = False


def _set_trace_default(value: bool) -> None:
    global _TRACE_DEFAULT
    _TRACE_DEFAULT = bool(value)


def evaluate(x: LispExpr, env: Env) -> Any:
    ctx = _CURRENT_EVAL_CONTEXT
    # Budget/time checks and tracing
    if ctx is not None:
        if ctx.deadline_monotonic is not None and time.monotonic() > ctx.deadline_monotonic:
            raise LispTimeoutError("evaluation timed out", stack=list(ctx.stack))
        if ctx.step_budget is not None:
            if ctx.step_budget <= 0:
                raise LispStepLimitError("step budget exceeded", stack=list(ctx.stack))
            ctx.step_budget -= 1
        ctx.stack.append(x)
        if ctx.trace or _TRACE_DEFAULT:
            print(_trace_line(len(ctx.stack) - 1, x))
    try:
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
    except LispError:
        # Already enriched with stack
        raise
    except Exception as e:  # noqa: BLE001
        # Wrap other exceptions with stack if we have a context
        if ctx is not None:
            raise LispError(str(e), stack=list(ctx.stack)) from e
        raise
    finally:
        if ctx is not None and ctx.stack:
            ctx.stack.pop()


def evaluate_limited(
    x: LispExpr,
    env: Env,
    *,
    max_steps: Optional[int] = None,
    time_limit_ms: Optional[int] = None,
    simplify_before: bool = False,
    trace: bool = False,
) -> Any:
    expr = simplify(x) if simplify_before else x
    deadline = None if time_limit_ms is None else time.monotonic() + (time_limit_ms / 1000.0)
    ctx = _EvalContext(step_budget=max_steps, deadline_monotonic=deadline, trace=trace, stack=[])
    global _CURRENT_EVAL_CONTEXT
    prev = _CURRENT_EVAL_CONTEXT
    _CURRENT_EVAL_CONTEXT = ctx
    try:
        return evaluate(expr, env)
    finally:
        _CURRENT_EVAL_CONTEXT = prev


def _builtin_load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    exprs = parse_many(src)
    result: Any = None
    env = _GLOBAL_ENV
    for expr in exprs:
        result = evaluate(expr, env)
    return result


def _builtin_simplify(expr: LispExpr) -> LispExpr:
    return simplify(expr)


def _builtin_use(module_path: str) -> Any:
    module = importlib.import_module(module_path)
    if hasattr(module, "lisp_register") and callable(getattr(module, "lisp_register")):
        getattr(module, "lisp_register")(_GLOBAL_ENV)
        return module_path
    if hasattr(module, "register") and callable(getattr(module, "register")):
        getattr(module, "register")(_GLOBAL_ENV)
        return module_path
    # Fallback: expose module as a symbol
    _GLOBAL_ENV[Symbol(module_path)] = module
    return module_path


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
                    try:
                        val = evaluate(expr, env)
                    except LispError as le:
                        print(f"Error: {le}")
                        st = le.format_stack()
                        if st:
                            print(st)
                        continue
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



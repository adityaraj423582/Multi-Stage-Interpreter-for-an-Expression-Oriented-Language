#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Multi-Stage Interpreter for an Expression-Oriented Language
Single-file implementation.

Language features:
- integers, booleans:  42, true, false
- variables, let bindings:  let x = 1 in x + 2
- lambdas and application:  fun x -> x + 1  (applied as (f 3))
- if-then-else
- binary ops: + - * / == < <= >= > && ||
- prefix not
- multi-stage:
    box E          -> produces a Code value containing E's AST (quote)
    splice(E)      -> only meaningful inside box: evaluate E now and insert resulting AST/literal into quoted AST
    run(E)         -> evaluate Code value E (i.e. evaluate the quoted AST at runtime)
Examples near the bottom.
"""

from __future__ import annotations
import re
import sys
from typing import Any, Dict, List, Tuple, Optional, Union, Callable
import copy

# ---------------------------
# Lexer
# ---------------------------
Token = Tuple[str, str]  # (type, value)

TOKEN_SPEC = [
    ('NUMBER',   r'\d+'),
    ('ID',       r'[A-Za-z_][A-Za-z0-9_]*'),
    ('ARROW',    r'->'),
    ('EQ',       r'=='),
    ('LE',       r'<='),
    ('GE',       r'>='),
    ('AND',      r'&&'),
    ('OR',       r'\|\|'),
    ('NOT',      r'!'),
    ('SKIP',     r'[ \t]+'),
    ('NEWLINE',  r'\n'),
    ('LPAREN',   r'\('),
    ('RPAREN',   r'\)'),
    ('COMMA',    r','),
    ('PLUS',     r'\+'),
    ('MINUS',    r'-'),
    ('TIMES',    r'\*'),
    ('DIV',      r'/'),
    ('LT',       r'<'),
    ('GT',       r'>'),
    ('ASSIGN',   r'='),
    ('SEMICOL',  r';'),
    ('BACK',     r'`'),  # unused
    ('OTHER',    r'.'),
]

TOKEN_REGEX = '|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPEC)
KEYWORDS = {
    'let', 'in', 'fun', 'if', 'then', 'else', 'true', 'false', 'box', 'run', 'splice', 'print'
}

def lex(text: str) -> List[Token]:
    tokens: List[Token] = []
    for mo in re.finditer(TOKEN_REGEX, text):
        kind = mo.lastgroup
        val = mo.group()
        if kind == 'NUMBER':
            tokens.append((kind, val))
        elif kind == 'ID':
            if val in KEYWORDS:
                tokens.append((val.upper(), val))
            else:
                tokens.append((kind, val))
        elif kind == 'SKIP' or kind == 'NEWLINE':
            continue
        elif kind == 'OTHER':
            tokens.append((val, val))
        else:
            tokens.append((kind, val))
    tokens.append(('EOF', ''))
    return tokens

# ---------------------------
# AST
# ---------------------------
class AST:
    pass

class Int(AST):
    def __init__(self, value: int): self.value = value
    def __repr__(self): return f"Int({self.value})"

class Bool(AST):
    def __init__(self, value: bool): self.value = value
    def __repr__(self): return f"Bool({self.value})"

class Var(AST):
    def __init__(self, name: str): self.name = name
    def __repr__(self): return f"Var({self.name})"

class Let(AST):
    def __init__(self, name: str, expr: AST, body: AST):
        self.name = name; self.expr = expr; self.body = body
    def __repr__(self): return f"Let({self.name}, {self.expr}, {self.body})"

class Fun(AST):
    def __init__(self, param: str, body: AST):
        self.param = param; self.body = body
    def __repr__(self): return f"Fun({self.param} -> {self.body})"

class App(AST):
    def __init__(self, fn: AST, arg: AST):
        self.fn = fn; self.arg = arg
    def __repr__(self): return f"App({self.fn}, {self.arg})"

class If(AST):
    def __init__(self, cond: AST, then: AST, els: AST):
        self.cond = cond; self.then = then; self.els = els
    def __repr__(self): return f"If({self.cond}, {self.then}, {self.els})"

class BinOp(AST):
    def __init__(self, op: str, left: AST, right: AST):
        self.op = op; self.left = left; self.right = right
    def __repr__(self): return f"BinOp({self.op}, {self.left}, {self.right})"

class UnOp(AST):
    def __init__(self, op: str, expr: AST):
        self.op = op; self.expr = expr
    def __repr__(self): return f"UnOp({self.op}, {self.expr})"

# Staging nodes
class Box(AST):
    def __init__(self, expr: AST): self.expr = expr
    def __repr__(self): return f"Box({self.expr})"

class Run(AST):
    def __init__(self, expr: AST): self.expr = expr
    def __repr__(self): return f"Run({self.expr})"

class Splice(AST):
    def __init__(self, expr: AST): self.expr = expr
    def __repr__(self): return f"Splice({self.expr})"

# Code wrapper runtime value
class CodeVal:
    def __init__(self, ast: AST):
        self.ast = ast
    def __repr__(self): return f"<Code {self.ast!r}>"

# Function closures
class Closure:
    def __init__(self, param: str, body: AST, env: Dict[str, Any]):
        self.param = param; self.body = body; self.env = env
    def __repr__(self): return f"<Closure {self.param} -> {self.body}>"

# ---------------------------
# Parser (recursive descent)
# ---------------------------
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def next(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def accept(self, *types):
        if self.peek()[0] in types:
            return self.next()
        return None

    def expect(self, typ):
        tok = self.next()
        if tok[0] != typ:
            raise SyntaxError(f"Expected {typ}, got {tok}")
        return tok

    # Grammar (simple):
    # expr ::= letexpr | ifexpr | lambda | orexpr
    # letexpr ::= 'let' ID '=' expr 'in' expr
    # ifexpr ::= 'if' expr 'then' expr 'else' expr
    # lambda ::= 'fun' ID '->' expr
    # orexpr ::= andexpr ( '||' andexpr )*
    # andexpr ::= eqexpr ( '&&' eqexpr )*
    # eqexpr ::= relexpr ( '==' relexpr )*
    # relexpr ::= addexpr ( ('<' | '>' | '<=' | '>=') addexpr)*
    # addexpr ::= mulexpr (('+'|'-') mulexpr)*
    # mulexpr ::= unexpr (('*'|'/') unexpr)*
    # unexpr ::= ('!' | '-') unexpr | primary
    # primary ::= NUMBER | true | false | ID | '(' expr ')' | application | box | run | splice
    # application: primary primary  (we will parse application via left recursion after parsing atom)
    def parse(self) -> AST:
        ast = self.parse_expr()
        if self.peek()[0] != 'EOF':
            raise SyntaxError("Extra input after program")
        return ast

    def parse_expr(self):
        if self.accept('LET'):
            name = self.expect('ID')[1]
            self.expect('ASSIGN')
            expr = self.parse_expr()
            self.expect('IN')
            body = self.parse_expr()
            return Let(name, expr, body)
        if self.accept('IF'):
            cond = self.parse_expr()
            self.expect('THEN')
            th = self.parse_expr()
            self.expect('ELSE')
            el = self.parse_expr()
            return If(cond, th, el)
        if self.accept('FUN'):
            param = self.expect('ID')[1]
            self.expect('ARROW')
            body = self.parse_expr()
            return Fun(param, body)
        return self.parse_or()

    def parse_or(self):
        left = self.parse_and()
        while self.accept('OR'):
            right = self.parse_and()
            left = BinOp('||', left, right)
        return left

    def parse_and(self):
        left = self.parse_eq()
        while self.accept('AND'):
            right = self.parse_eq()
            left = BinOp('&&', left, right)
        return left

    def parse_eq(self):
        left = self.parse_rel()
        while self.accept('EQ'):
            right = self.parse_rel()
            left = BinOp('==', left, right)
        return left

    def parse_rel(self):
        left = self.parse_add()
        while True:
            if self.accept('LT'):
                right = self.parse_add(); left = BinOp('<', left, right)
            elif self.accept('GT'):
                right = self.parse_add(); left = BinOp('>', left, right)
            elif self.accept('LE'):
                right = self.parse_add(); left = BinOp('<=', left, right)
            elif self.accept('GE'):
                right = self.parse_add(); left = BinOp('>=', left, right)
            else:
                break
        return left

    def parse_add(self):
        left = self.parse_mul()
        while True:
            if self.accept('PLUS'):
                right = self.parse_mul(); left = BinOp('+', left, right)
            elif self.accept('MINUS'):
                right = self.parse_mul(); left = BinOp('-', left, right)
            else:
                break
        return left

    def parse_mul(self):
        left = self.parse_un()
        while True:
            if self.accept('TIMES'):
                right = self.parse_un(); left = BinOp('*', left, right)
            elif self.accept('DIV'):
                right = self.parse_un(); left = BinOp('/', left, right)
            else:
                break
        return left

    def parse_un(self):
        if self.accept('NOT'):
            return UnOp('!', self.parse_un())
        if self.accept('MINUS'):
            return UnOp('-', self.parse_un())
        return self.parse_app()

    def parse_app(self):
        # parse atomic and then repeated application: left-associative
        node = self.parse_primary()
        while True:
            # application is indicated by juxtaposition of expressions or explicit parentheses
            # attempt to parse another primary and make App(node, primary)
            # but don't consume keywords that cannot start an expression's primary (we support let/if/fun as expression start)
            tok = self.peek()[0]
            if tok in ('NUMBER','ID','LPAREN','TRUE','FALSE','BOX','RUN','SPLICE','FUN','IF','LET'):
                right = self.parse_primary()
                node = App(node, right)
            else:
                break
        return node

    def parse_primary(self):
        tok = self.peek()
        if tok[0] == 'NUMBER':
            self.next(); return Int(int(tok[1]))
        if tok[0] == 'TRUE':
            self.next(); return Bool(True)
        if tok[0] == 'FALSE':
            self.next(); return Bool(False)
        if tok[0] == 'ID':
            name = tok[1]; self.next(); return Var(name)
        if self.accept('LPAREN'):
            e = self.parse_expr(); self.expect('RPAREN'); return e
        if self.accept('BOX'):
            # box expr
            inner = self.parse_primary()  # box only applies to a single primary for simplicity; user can parenthesize
            return Box(inner)
        if self.accept('RUN'):
            inner = self.parse_primary()
            return Run(inner)
        if self.accept('SPLICE'):
            inner = self.parse_primary()
            return Splice(inner)
        raise SyntaxError(f"Unexpected token in primary: {tok}")

# ---------------------------
# Evaluator / Interpreter
# ---------------------------
class RuntimeError_(Exception):
    pass

def is_truthy(v: Any) -> bool:
    if isinstance(v, bool): return v
    if isinstance(v, int): return v != 0
    return bool(v)

def evaluate(ast: AST, env: Dict[str, Any]) -> Any:
    """
    Evaluate AST with environment env.
    env maps variable names -> runtime values (ints, bools, Closure, CodeVal)
    """
    if isinstance(ast, Int):
        return ast.value
    if isinstance(ast, Bool):
        return ast.value
    if isinstance(ast, Var):
        if ast.name in env:
            return env[ast.name]
        else:
            raise RuntimeError_(f"Unbound variable {ast.name}")
    if isinstance(ast, Let):
        v = evaluate(ast.expr, env)
        newenv = env.copy()
        newenv[ast.name] = v
        return evaluate(ast.body, newenv)
    if isinstance(ast, Fun):
        return Closure(ast.param, ast.body, env.copy())
    if isinstance(ast, App):
        fn = evaluate(ast.fn, env)
        arg = evaluate(ast.arg, env)
        if isinstance(fn, Closure):
            newenv = fn.env.copy()
            newenv[fn.param] = arg
            return evaluate(fn.body, newenv)
        elif callable(fn):
            return fn(arg)
        else:
            raise RuntimeError_(f"Tried to call non-function {fn}")
    if isinstance(ast, If):
        c = evaluate(ast.cond, env)
        if is_truthy(c):
            return evaluate(ast.then, env)
        else:
            return evaluate(ast.els, env)
    if isinstance(ast, BinOp):
        l = evaluate(ast.left, env)
        # short-circuit for && and ||
        if ast.op == '&&':
            if not is_truthy(l): return False
            r = evaluate(ast.right, env); return is_truthy(r)
        if ast.op == '||':
            if is_truthy(l): return True
            r = evaluate(ast.right, env); return is_truthy(r)
        r = evaluate(ast.right, env)
        if ast.op == '+': return l + r
        if ast.op == '-': return l - r
        if ast.op == '*': return l * r
        if ast.op == '/':
            if r == 0: raise RuntimeError_("Division by zero")
            return l // r
        if ast.op == '==': return l == r
        if ast.op == '<': return l < r
        if ast.op == '>': return l > r
        if ast.op == '<=': return l <= r
        if ast.op == '>=': return l >= r
        raise RuntimeError_(f"Unknown binary op {ast.op}")
    if isinstance(ast, UnOp):
        v = evaluate(ast.expr, env)
        if ast.op == '-':
            return -v
        if ast.op == '!':
            return not is_truthy(v)
        raise RuntimeError_(f"Unknown unary op {ast.op}")

    # Staging semantics
    if isinstance(ast, Box):
        # produce a CodeVal by quoting AST, but evaluating Splice nodes immediately
        boxed_ast = process_box(ast.expr, env)
        return CodeVal(boxed_ast)
    if isinstance(ast, Splice):
        # Splice outside of box: evaluate normally
        return evaluate(ast.expr, env)
    if isinstance(ast, Run):
        v = evaluate(ast.expr, env)
        if not isinstance(v, CodeVal):
            raise RuntimeError_("run expects a Code value")
        # Evaluate the code AST in the current environment (allow capture)
        # Note: we make a shallow copy of env to avoid accidental mutations
        return evaluate(v.ast, env.copy())

    raise RuntimeError_(f"Unhandled AST node: {ast}")

def process_box(node: AST, env: Dict[str, Any]) -> AST:
    """
    Walk the AST of the boxed expression and:
    - whenever finding Splice(expr), evaluate expr in the *current* env:
        * if result is CodeVal -> insert its AST
        * if result is int/bool -> insert Int/Bool literal
        * if result is Closure or other -> ERROR for now
    - otherwise, produce AST nodes structurally identical to the input, with children processed recursively.
    This creates a new AST (a quasi-quoted AST) representing the code.
    """
    if isinstance(node, Splice):
        val = evaluate(node.expr, env)
        if isinstance(val, CodeVal):
            return copy.deepcopy(val.ast)
        elif isinstance(val, int):
            return Int(val)
        elif isinstance(val, bool):
            return Bool(val)
        else:
            raise RuntimeError_(f"Cannot splice value of type {type(val)}")
    # copy structure
    if isinstance(node, Int): return Int(node.value)
    if isinstance(node, Bool): return Bool(node.value)
    if isinstance(node, Var): return Var(node.name)
    if isinstance(node, Let):
        return Let(node.name, process_box(node.expr, env), process_box(node.body, env))
    if isinstance(node, Fun):
        return Fun(node.param, process_box(node.body, env))
    if isinstance(node, App):
        return App(process_box(node.fn, env), process_box(node.arg, env))
    if isinstance(node, If):
        return If(process_box(node.cond, env), process_box(node.then, env), process_box(node.els, env))
    if isinstance(node, BinOp):
        return BinOp(node.op, process_box(node.left, env), process_box(node.right, env))
    if isinstance(node, UnOp):
        return UnOp(node.op, process_box(node.expr, env))
    if isinstance(node, Box):
        # nested box: do NOT evaluate inner splices now — treat nested box as literal: recurse but do not evaluate splices in it.
        # For simplicity, we will treat nested box by copying inner structure but treating Splice as literal (i.e., keep Splice nodes)
        return Box(process_box_keep_splice(node.expr))
    if isinstance(node, Run):
        return Run(process_box(node.expr, env))
    raise RuntimeError_(f"Unhandled node in process_box: {node}")

def process_box_keep_splice(node: AST) -> AST:
    """Helper to copy structure while preserving Splice nodes (used for nested boxes)."""
    if isinstance(node, (Int, Bool, Var)): return copy.deepcopy(node)
    if isinstance(node, Splice): return Splice(copy.deepcopy(node.expr))
    if isinstance(node, Let):
        return Let(node.name, process_box_keep_splice(node.expr), process_box_keep_splice(node.body))
    if isinstance(node, Fun):
        return Fun(node.param, process_box_keep_splice(node.body))
    if isinstance(node, App):
        return App(process_box_keep_splice(node.fn), process_box_keep_splice(node.arg))
    if isinstance(node, If):
        return If(process_box_keep_splice(node.cond), process_box_keep_splice(node.then), process_box_keep_splice(node.els))
    if isinstance(node, BinOp):
        return BinOp(node.op, process_box_keep_splice(node.left), process_box_keep_splice(node.right))
    if isinstance(node, UnOp):
        return UnOp(node.op, process_box_keep_splice(node.expr))
    if isinstance(node, Box):
        return Box(process_box_keep_splice(node.expr))
    if isinstance(node, Run):
        return Run(process_box_keep_splice(node.expr))
    raise RuntimeError_(f"Unhandled node in process_box_keep_splice: {node}")

# ---------------------------
# Top-level utilities and REPL
# ---------------------------
def parse_and_eval(src: str, env: Dict[str, Any]) -> Any:
    toks = lex(src)
    p = Parser(toks)
    ast = p.parse()
    #print("AST:", ast)
    return evaluate(ast, env)

def standard_env() -> Dict[str, Any]:
    env: Dict[str, Any] = {}
    # add print as builtin
    def _print(x):
        print(x)
        return 0
    env['print'] = _print
    return env

def demo():
    env = standard_env()
    examples = [
        ("Simple arithmetic", "1 + 2 * 3"),
        ("Let & lambda", "let x = 5 in let f = fun y -> x + y in (f 3)"),
        ("If expression", "if 1 < 2 then 10 else 20"),
        ("Box basic (quote)", "box (1 + 2)"),
        ("Run a boxed expr", "run (box (1 + 2))"),
        ("Splice inside box",
         # Create code for constant 10 by splicing an int computed now
         "let n = 7 in box (3 + splice(n))"),
        ("Splice with Code",
         # Build a code piece and splice it
         "let piece = box (4 * 5) in box (1 + splice(piece))"),
        ("Run with splice and capture",
         # Compose and run
         "let piece = box (4 * 5) in run (box (2 + splice(piece)))"),
        ("Meta-programming example: generate code for adding constant",
         "let make_adder = fun c -> box (fun x -> x + splice(c)) in let code = (make_adder (box (10))) in run(code)"),
    ]
    for title, src in examples:
        try:
            print("==", title, "==")
            val = parse_and_eval(src, env)
            print("=>", val)
        except Exception as e:
            print("Error:", e)
        print()

def repl():
    print("Multi-stage interpreter REPL. Type 'exit' to quit.")
    env = standard_env()
    while True:
        try:
            src = input(">>> ")
        except EOFError:
            print(); break
        if not src: continue
        if src.strip() in ('exit','quit'):
            break
        try:
            val = parse_and_eval(src, env)
            print("=>", val)
        except Exception as e:
            print("Error:", e)

# ---------------------------
# If executed as script: show demo, then repl if interactive
# ---------------------------
if __name__ == "__main__":
    print("Multi-stage interpreter demo\n")
    demo()
    # If run interactively, drop into REPL
    if sys.stdin.isatty():
        repl()


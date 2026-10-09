"""
SignSyntax - End-to-End Compiler Frontend & AST CLI Demonstration
Phase 2: Core Implementation (AST Demonstration)

Usage:
    python tests/demo_ast.py
    python tests/demo_ast.py "YESTERDAY ME WANT MILK"
    python tests/demo_ast.py "WHAT YOU WANT"
    python tests/demo_ast.py "SHE HAVE CAR BIG"
"""

import sys
import os
import json

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.lexer import Lexer
from compiler.parser import Parser


def run_ast_demo(input_text: str) -> None:
    """Executes the full Lexer -> Parser -> AST pipeline and displays structured results."""
    print("=" * 65)
    print("           SIGNSYNTAX COMPILER FRONTEND: AST DEMO")
    print("=" * 65)
    print(f"Input: {input_text}")
    print("-" * 65)

    # 1. Lexical Analysis
    lexer = Lexer()
    lex_result = lexer.tokenize(input_text)

    if lex_result.has_errors:
        print("[LEXICAL ANALYSIS] FAILED")
        for err in lex_result.errors:
            print(f"  [LEXICAL ERROR] {err}")
        print("=" * 65)
        return

    print(f"[LEXICAL ANALYSIS] [OK] ({len(lex_result.tokens)} tokens)")
    token_str = " ".join([f"[{t.lexeme}:{t.token_type}]" for t in lex_result.tokens])
    print(f"  Token Stream: {token_str}")

    # 2. Syntax Analysis & AST Construction
    parser = Parser(lex_result.tokens)
    parse_result = parser.parse()

    if parse_result.has_errors:
        print("\n[SYNTAX ANALYSIS] FAILED")
        for err in parse_result.errors:
            print(f"  [SYNTAX ERROR] {err}")
        print("=" * 65)
        return

    print("\n[SYNTAX ANALYSIS] [OK] (Valid sentence)")

    # 3. Abstract Syntax Tree
    print("\n[ABSTRACT SYNTAX TREE]")
    if parse_result.ast is not None:
        print(parse_result.ast.pretty_print())
        print("\n[AST JSON SERIALIZATION]")
        print(json.dumps(parse_result.ast.to_dict(), indent=2))
    else:
        print("  (No AST generated)")

    print("=" * 65)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sentence = " ".join(sys.argv[1:])
        run_ast_demo(sentence)
    else:
        demo_sentences = [
            "YESTERDAY ME WANT MILK",
            "ME WANT MILK",
            "WHAT YOU WANT",
            "SHE HAVE CAR BIG",
            "ME WANT MILK NOT",
            "YESTERDAY ME WANT MILK NOT",
            "WHAT BOOK",
            "ME MILK WANT",
            "ME DANCE MILK",
        ]
        for s in demo_sentences:
            run_ast_demo(s)
            print()

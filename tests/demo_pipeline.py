"""
SignSyntax - End-to-End Syntax Pipeline Demonstration
Phase 2: Core Implementation (Pipeline Demonstration)

Demonstrates the 3-stage compiler frontend:
  [1] Lexical Analysis (Gloss Tokenization & Symbol Table validation)
  [2] Syntax Analysis (LL(1) Recursive-Descent Parsing)
  [3] Abstract Syntax Tree (Clean Semantic Tree Representation)

Usage:
    python tests/demo_pipeline.py
    python tests/demo_pipeline.py "YESTERDAY ME WANT MILK"
    python tests/demo_pipeline.py "WHAT YOU WANT"
    python tests/demo_pipeline.py "ME MILK WANT"
"""

import sys
import os

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.lexer import Lexer
from compiler.parser import Parser


def run_pipeline(source: str) -> bool:
    """Runs the SignSyntax compiler frontend pipeline on a source string."""
    print("=" * 60)
    print("                 SIGNSYNTAX SYNTAX PIPELINE")
    print("=" * 60)
    print(f"\nINPUT:\n{source}\n")

    # -------------------------------------------------------------------------
    # [1] Lexical Analysis
    # -------------------------------------------------------------------------
    print("[1] LEXICAL ANALYSIS")
    lexer = Lexer()
    lex_result = lexer.tokenize(source)

    if lex_result.has_errors:
        print("  [FAILED] Lexical analysis encountered errors:")
        for err in lex_result.errors:
            print(f"    - {err}")
        print("\n[PIPELINE STOPPED AT LEXICAL ANALYSIS]")
        print("=" * 60)
        return False

    for tok in lex_result.tokens:
        print(f"  [OK] {tok.lexeme} -> {tok.token_type}")

    # -------------------------------------------------------------------------
    # [2] Syntax Analysis
    # -------------------------------------------------------------------------
    print("\n[2] SYNTAX ANALYSIS")
    parser = Parser(lex_result.tokens)
    parse_result = parser.parse()

    if parse_result.has_errors:
        print("  [FAILED] Syntax analysis encountered errors:")
        for err in parse_result.errors:
            print(f"    - {err}")
        print("\n[PIPELINE STOPPED AT SYNTAX ANALYSIS]")
        print("=" * 60)
        return False

    print("  [OK] Valid sentence structure")

    # -------------------------------------------------------------------------
    # [3] Abstract Syntax Tree
    # -------------------------------------------------------------------------
    print("\n[3] ABSTRACT SYNTAX TREE\n")
    if parse_result.ast is not None:
        print(parse_result.ast.pretty_print())
    else:
        print("  (No AST generated)")

    print("\n[OK] Syntax pipeline successful")
    print("=" * 60)
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sentence = " ".join(sys.argv[1:])
        run_pipeline(sentence)
    else:
        demo_sentences = [
            "YESTERDAY ME WANT MILK",
            "ME WANT MILK",
            "WHAT YOU WANT",
            "SHE HAVE CAR BIG",
            "ME WANT MILK NOT",
            "ME MILK WANT",
            "ME",
            "ME WANT MILK BOOK",
            "ME DANCE MILK",
        ]
        for s in demo_sentences:
            run_pipeline(s)
            print()

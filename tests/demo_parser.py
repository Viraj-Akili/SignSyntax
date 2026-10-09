"""
SignSyntax - Syntax Parser CLI Demonstration
Phase 2: Core Implementation (Parser Demonstration)

Usage:
    python tests/demo_parser.py
    python tests/demo_parser.py "YESTERDAY ME WANT MILK"
    python tests/demo_parser.py "WHAT YOU WANT"
    python tests/demo_parser.py "ME MILK WANT"
"""

import sys
import os

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.lexer import Lexer
from compiler.parser import Parser


def run_parser_demo(input_text: str) -> None:
    """Executes the Lexer and Parser on input_text and prints a structured report."""
    print("=" * 60)
    print("                 SIGNSYNTAX PARSER DEMO")
    print("=" * 60)
    print(f"Input: {input_text}")
    print("-" * 60)

    # 1. Lexical Analysis
    lexer = Lexer()
    lex_result = lexer.tokenize(input_text)

    if lex_result.has_errors:
        print("Lexical Analysis: FAILED")
        for err in lex_result.errors:
            print(f"  [LEXICAL ERROR] {err}")
        print("=" * 60)
        return

    print(f"Lexical Analysis: [OK] ({len(lex_result.tokens)} tokens)")
    token_str = " ".join([f"[{t.lexeme}:{t.token_type}]" for t in lex_result.tokens])
    print(f"  Token Stream: {token_str}")

    # 2. Syntax Analysis
    parser = Parser(lex_result.tokens)
    parse_result = parser.parse()

    if parse_result.has_errors:
        print("\nSyntax Analysis: FAILED")
        for err in parse_result.errors:
            print(f"  [SYNTAX ERROR] {err}")
    else:
        print("\nSyntax Analysis: [OK] (Valid sentence)")
        print("\nParse Tree:")
        print(parse_result.tree.display(indent=1))

    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sentence = " ".join(sys.argv[1:])
        run_parser_demo(sentence)
    else:
        # Run a representative suite of examples
        demo_sentences = [
            "YESTERDAY ME WANT MILK",
            "ME WANT MILK",
            "WHAT YOU WANT",
            "SHE HAVE CAR BIG",
            "ME WANT MILK NOT",
            "ME MILK WANT",
            "ME",
            "ME DANCE MILK",
        ]
        for s in demo_sentences:
            run_parser_demo(s)
            print()

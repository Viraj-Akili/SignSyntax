"""
SignSyntax - Lexer Demonstration Script
Phase 1: Compiler Design - Interactive Lexical Analyzer Demonstration

Demonstrates the Lexical Analyzer converting Sign Language Glosses
into typed Token streams using the Symbol Table, with detailed error reporting.
"""

import os
import sys

# Ensure root workspace directory is in sys.path when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.lexer import Lexer


def run_demonstration() -> None:
    lexer = Lexer()

    print("=" * 70)
    print("        SignSyntax: Phase 1 Lexical Analyzer (Lexer) Demo        ")
    print("=" * 70)

    test_cases = [
        ("Example 1: Valid Sentence with Temporal Marker", "YESTERDAY ME WANT MILK"),
        ("Example 2: Unknown Gloss Error Detection", "ME DANCE MILK"),
        ("Example 3: Case-Sensitivity Rejection", "me WANT MILK"),
        ("Example 4: Question / Interrogative Sentence", "WHERE HOME"),
        ("Example 5: Multiple Unknown Glosses", "ME DANCE RUN MILK"),
    ]

    for title, input_text in test_cases:
        print(f"\n--- {title} ---")
        print(f"Input Source : \"{input_text}\"")
        result = lexer.tokenize(input_text)

        if result.tokens:
            print("Tokens Generated:")
            for token in result.tokens:
                print(f"  [Pos {token.position}] Type: {token.token_type:<10} | Lexeme: {token.lexeme:<12} | Meaning: '{token.meaning}'")
        else:
            print("Tokens Generated: (none)")

        if result.has_errors:
            print("Lexical Errors Detected:")
            for err in result.errors:
                print(f"  [ERROR] {err}")
        else:
            print("Lexical Status : [OK] Valid Token Stream")

    print("\n" + "=" * 70)
    print("Demonstration complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_demonstration()

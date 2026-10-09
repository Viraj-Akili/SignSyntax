"""
SignSyntax - Unified Compiler Frontend Demonstration
Phase 2: Core Implementation (Frontend Demonstration for Review 2)

Demonstrates the 4-stage compiler frontend:
  [1] Lexical Analysis (Gloss Tokenization & Symbol Table validation)
  [2] Syntax Analysis (LL(1) Recursive-Descent Parsing)
  [3] AST Construction (Abstract Syntax Tree Hierarchy)
  [4] Semantic Analysis (Category Verification & Meaning Normalization)

Also demonstrates phase isolation on lexical, syntax, and semantic failures.

Usage:
    python tests/demo_frontend.py
    python tests/demo_frontend.py "YESTERDAY ME WANT MILK"
"""

import sys
import os

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.ast import (
    DeclarativeSentenceNode,
    ObjectNode,
    SentenceNode,
    SubjectNode,
    VerbPhraseNode,
)
from compiler.frontend import compile_frontend
from compiler.semantic_analyzer import SemanticAnalyzer


def run_frontend_demo(source: str) -> None:
    """Executes compile_frontend and displays the structured execution report."""
    print("=" * 65)
    print("                SIGNSYNTAX COMPILER FRONTEND")
    print("=" * 65)
    print(f"\nSOURCE:\n{source}\n")

    res = compile_frontend(source)

    # 1. Lexical Analysis
    if res.lexer_result is not None:
        if res.lexer_result.has_errors:
            print("[1] LEXICAL ANALYSIS\n[FAILED]")
            for err in res.lexer_result.errors:
                print(f"  [LEXICAL ERROR] {err}")
            print("\n[STATUS] FRONTEND COMPILATION STOPPED AT LEXICAL ANALYSIS")
            print("=" * 65)
            return
        else:
            print("[1] LEXICAL ANALYSIS\n[OK]")
            print("Tokens:")
            for tok in res.lexer_result.tokens:
                print(f"  [OK] {tok.lexeme} -> {tok.token_type}")
    else:
        print("[1] LEXICAL ANALYSIS\n[SKIPPED]")

    # 2. Syntax Analysis
    if res.parse_result is not None:
        if res.parse_result.has_errors:
            print("\n[2] SYNTAX ANALYSIS\n[FAILED]")
            for err in res.parse_result.errors:
                print(f"  [SYNTAX ERROR] {err}")
            print("\n[STATUS] FRONTEND COMPILATION STOPPED AT SYNTAX ANALYSIS")
            print("=" * 65)
            return
        else:
            print("\n[2] SYNTAX ANALYSIS\n[OK]")
    else:
        print("\n[2] SYNTAX ANALYSIS\n[SKIPPED]")

    # 3. AST Construction
    if res.ast is not None:
        print("\n[3] AST CONSTRUCTION\n[OK]\n")
        print(res.ast.pretty_print())
    else:
        print("\n[3] AST CONSTRUCTION\n[NONE]")

    # 4. Semantic Analysis
    if res.semantic_result is not None:
        if res.semantic_result.has_errors:
            print("\n[4] SEMANTIC ANALYSIS\n[FAILED]")
            for err in res.semantic_result.errors:
                print(f"  [SEMANTIC ERROR] {err}")
            print("\n[STATUS] FRONTEND COMPILATION STOPPED AT SEMANTIC ANALYSIS")
            print("=" * 65)
            return
        else:
            print("\n[4] SEMANTIC ANALYSIS\n[OK]")
            print("\n[5] NORMALIZED SEMANTIC FORM:")
            print(res.semantic_form.display())
            print("\n[STATUS] FRONTEND COMPILATION SUCCESSFUL")
    else:
        print("\n[4] SEMANTIC ANALYSIS\n[SKIPPED]")

    print("=" * 65)


def run_malformed_ast_semantic_demo() -> None:
    """Demonstrates semantic error accumulation on a direct malformed AST."""
    print("\n" + "=" * 65)
    print("   DEMONSTRATION: DIRECT AST MULTI-ERROR SEMANTIC VALIDATION")
    print("=" * 65)
    print("Constructing AST with 2 category violations:")
    print("  1. SubjectNode('MILK') -> NOUN instead of PRONOUN")
    print("  2. ObjectNode('WANT')  -> VERB instead of NOUN\n")

    malformed_ast = SentenceNode(
        body=DeclarativeSentenceNode(
            subject=SubjectNode(value="MILK"),
            verb_phrase=VerbPhraseNode(
                verb="LIKE",
                object=ObjectNode(value="WANT"),
            ),
        )
    )
    print("Malformed AST:")
    print(malformed_ast.pretty_print())

    print("\nRunning SemanticAnalyzer.analyze()...")
    analyzer = SemanticAnalyzer()
    res = analyzer.analyze(malformed_ast)

    print(f"Validation Result: {'[OK]' if res.is_valid else '[FAILED]'}")
    print(f"Errors Collected ({len(res.errors)}):")
    for err in res.errors:
        print(f"  [SEMANTIC ERROR] {err}")
    print("=" * 65)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sentence = " ".join(sys.argv[1:])
        run_frontend_demo(sentence)
    else:
        # 1. Successful compilation
        run_frontend_demo("YESTERDAY ME WANT MILK")
        print()
        # 2. Lexical failure
        run_frontend_demo("ME DANCE MILK")
        print()
        # 3. Syntax failure
        run_frontend_demo("ME MILK WANT")
        print()
        # 4. Semantic failure
        run_malformed_ast_semantic_demo()

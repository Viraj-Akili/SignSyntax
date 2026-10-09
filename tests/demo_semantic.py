"""
SignSyntax - Semantic Analysis & Full Frontend Pipeline Demonstration
Phase 2: Core Implementation (Semantic Analysis Demonstration)

Demonstrates the complete 4-stage compiler frontend:
  [1] Lexical Analysis (Gloss Tokenization & Symbol Table validation)
  [2] Syntax Analysis (LL(1) Recursive-Descent Parsing)
  [3] Abstract Syntax Tree Construction (Clean Semantic Hierarchy)
  [4] Semantic Analysis (Symbol Table Type Checking & Meaning Normalization)

Usage:
    python tests/demo_semantic.py
    python tests/demo_semantic.py "YESTERDAY ME WANT MILK"
    python tests/demo_semantic.py "WHAT YOU WANT"
"""

import sys
import os

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.ast import (
    DeclarativeSentenceNode,
    SentenceNode,
    SubjectNode,
    VerbPhraseNode,
)
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.semantic_analyzer import SemanticAnalyzer


def run_full_pipeline(source: str) -> None:
    """Executes the complete Lexer -> Parser -> AST -> Semantic Analyzer pipeline."""
    print("=" * 65)
    print("             SIGNSYNTAX COMPILER: SEMANTIC PIPELINE")
    print("=" * 65)
    print(f"\nINPUT:\n{source}\n")

    # 1. Lexical Analysis
    print("[1] LEXICAL ANALYSIS")
    lexer = Lexer()
    lex_res = lexer.tokenize(source)
    if lex_res.has_errors:
        print("  [FAILED] Lexical errors:")
        for err in lex_res.errors:
            print(f"    - {err}")
        print("=" * 65)
        return
    for tok in lex_res.tokens:
        print(f"  [OK] {tok.lexeme} -> {tok.token_type}")

    # 2. Syntax Analysis
    print("\n[2] SYNTAX ANALYSIS")
    parser = Parser(lex_res.tokens)
    parse_res = parser.parse()
    if parse_res.has_errors:
        print("  [FAILED] Syntax errors:")
        for err in parse_res.errors:
            print(f"    - {err}")
        print("=" * 65)
        return
    print("  [OK] Valid sentence structure")

    # 3. AST Construction
    print("\n[3] AST CONSTRUCTION")
    ast = parse_res.ast
    if ast is None:
        print("  [FAILED] No AST generated")
        print("=" * 65)
        return
    print("  [OK] Abstract Syntax Tree created:")
    print(ast.pretty_print())

    # 4. Semantic Analysis
    print("\n[4] SEMANTIC ANALYSIS")
    analyzer = SemanticAnalyzer()
    sem_res = analyzer.analyze(ast)

    if sem_res.has_errors:
        print("  [FAILED] Semantic validation errors:")
        for err in sem_res.errors:
            print(f"    - {err}")
        print("=" * 65)
        return

    print("  [OK] Semantic validation successful")
    print("\nSEMANTIC FORM:")
    print(sem_res.semantic_form.display())
    print("\n[STATUS] All frontend compiler stages succeeded!")
    print("=" * 65)


def run_semantic_failure_demo() -> None:
    """Demonstrates a semantic type/category validation failure on a malformed AST."""
    print("\n" + "=" * 65)
    print("       DEMONSTRATION: DIRECT AST SEMANTIC VALIDATION FAILURE")
    print("=" * 65)
    print("Scenario: AST contains SubjectNode with value 'MILK' (NOUN instead of PRONOUN)")

    # Manually constructed invalid AST
    invalid_ast = SentenceNode(
        body=DeclarativeSentenceNode(
            subject=SubjectNode(value="MILK"),  # NOUN violating PRONOUN category
            verb_phrase=VerbPhraseNode(verb="WANT"),
        )
    )

    print("\nConstructed AST:")
    print(invalid_ast.pretty_print())

    print("\nRunning SemanticAnalyzer...")
    analyzer = SemanticAnalyzer()
    result = analyzer.analyze(invalid_ast)

    print(f"Semantic Result Valid: {result.is_valid}")
    print(f"Errors Captured ({len(result.errors)}):")
    for err in result.errors:
        print(f"  [SEMANTIC ERROR] {err}")
    print("=" * 65)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sentence = " ".join(sys.argv[1:])
        run_full_pipeline(sentence)
    else:
        # Default representative demonstration
        run_full_pipeline("YESTERDAY ME WANT MILK")
        print()
        run_full_pipeline("WHAT YOU WANT")
        print()
        run_full_pipeline("SHE HAVE CAR BIG")
        print()
        run_semantic_failure_demo()

"""
SignSyntax - Intermediate Representation (IR) Pipeline Demonstration
Phase 2: Core Implementation (Compiler Intermediate Representation)

Demonstrates the 5-phase compiler frontend pipeline:
  [1] Lexical Analysis (Lexer)
  [2] Syntax Analysis (Parser)
  [3] AST Construction (ASTBuilder)
  [4] Semantic Analysis (SemanticAnalyzer)
  [5] Intermediate Representation (IRGenerator)

Universal 7-bit ASCII rendering for clean output across all terminal environments.
"""

import os
import sys

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.frontend import compile_frontend
from compiler.ir import IRGenerator


def run_pipeline_demo(source: str, title: str) -> None:
    print("=" * 65)
    print(f"  DEMONSTRATION: {title}")
    print("=" * 65)
    print(f"SOURCE:\n{source}\n")

    res = compile_frontend(source)

    # 1. Lexical
    if res.lexer_result is not None and res.lexer_result.is_valid:
        print("[1] LEXICAL ANALYSIS\n[OK]")
        print("Tokens:")
        for t in res.lexer_result.tokens:
            print(f"  [OK] {t.lexeme} -> {t.token_type}")
    else:
        print("[1] LEXICAL ANALYSIS\n[FAILED]")
        if res.lexer_result:
            for err in res.lexer_result.errors:
                print(f"  [LEXICAL ERROR] {err}")

    # 2. Syntax
    if res.parse_result is not None and res.parse_result.is_valid:
        print("\n[2] SYNTAX ANALYSIS\n[OK]")
    elif res.failed_phase == "SYNTAX":
        print("\n[2] SYNTAX ANALYSIS\n[FAILED]")
        for err in res.parse_result.errors:
            print(f"  [SYNTAX ERROR] {err}")

    # 3. AST
    if res.ast is not None:
        print("\n[3] AST CONSTRUCTION\n[OK]")
        print(res.ast.pretty_print())

    # 4. Semantic
    if res.semantic_result is not None and res.semantic_result.is_valid:
        print("\n[4] SEMANTIC ANALYSIS\n[OK]")
        print("\n[5] NORMALIZED SEMANTIC FORM:")
        print(res.semantic_form.display())
    elif res.failed_phase == "SEMANTIC":
        print("\n[4] SEMANTIC ANALYSIS\n[FAILED]")
        for err in res.semantic_result.errors:
            print(f"  [SEMANTIC ERROR] {err}")

    # 5. IR
    if res.ir_program is not None:
        print("\n[6] INTERMEDIATE REPRESENTATION\n[OK]")
        print(res.ir_program.to_text())
        print("\n[STATUS] COMPILATION + IR GENERATION SUCCESSFUL")
    else:
        print("\n[6] INTERMEDIATE REPRESENTATION\n[IR] NOT GENERATED")
        print(f"\n[STATUS] COMPILATION STOPPED AT {res.failed_phase} PHASE")

    print("=" * 65 + "\n")


def main() -> None:
    # 1. Declarative with Time Frame
    run_pipeline_demo("YESTERDAY ME WANT MILK", "Declarative Sentence with Time Context")

    # 2. Question Sentence
    run_pipeline_demo("WHAT YOU WANT", "Interrogative Question")

    # 3. Lexical Failure Example
    run_pipeline_demo("ME DANCE MILK", "Lexical Error (Unknown Gloss 'DANCE')")

    # 4. Syntax Failure Example
    run_pipeline_demo("ME MILK WANT", "Syntax Error (Invalid SOV Word Order)")


if __name__ == "__main__":
    main()

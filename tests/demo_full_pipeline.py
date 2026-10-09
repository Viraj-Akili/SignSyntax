"""
SignSyntax - Complete Compiler Pipeline Demonstration
Phase 2: Core Implementation (Full Pipeline Integration & Verification)

Demonstrates the entire 6-stage SignSyntax compiler system:
  [1] Gloss Input Stream
  [2] Lexical Analysis (Lexer)
  [3] Syntax Analysis (Parser)
  [4] AST Construction (ASTBuilder)
  [5] Semantic Analysis (SemanticAnalyzer)
  [6] Intermediate Representation (IRGenerator)

Includes demonstrations of:
- Full success on declarative sign utterance ('YESTERDAY ME WANT MILK')
- Full success on question sign utterance ('WHAT YOU WANT')
- Phase 1 Lexical failure boundary ('ME DANCE MILK')
- Phase 2 Syntax failure boundary ('ME MILK WANT')

Universal 7-bit ASCII rendering for cross-platform terminal compatibility.
"""

import os
import sys

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.frontend import compile_frontend


def display_pipeline_execution(source: str, description: str) -> None:
    print("=" * 65)
    print("           SIGNSYNTAX -- COMPLETE COMPILER PIPELINE")
    print("=" * 65)
    print(f"DESCRIPTION: {description}")
    print(f"INPUT:\n{source}\n")

    res = compile_frontend(source)

    # [1] Gloss Input
    print("[1] GLOSS INPUT\n[OK]")

    # [2] Lexical Analysis
    if res.lexer_result is not None and res.lexer_result.is_valid:
        print("\n[2] LEXICAL ANALYSIS\n[OK]")
        print("Tokens:")
        for t in res.lexer_result.tokens:
            print(f"  [OK] {t.lexeme:<12} -> {t.token_type}")
    else:
        print("\n[2] LEXICAL ANALYSIS\n[FAILED]")
        if res.lexer_result:
            for err in res.lexer_result.errors:
                print(f"  [LEXICAL ERROR] {err}")
        print("\n[STATUS] PIPELINE STOPPED AT LEXICAL ANALYSIS PHASE")
        print("=" * 65 + "\n")
        return

    # [3] Syntax Analysis
    if res.parse_result is not None and res.parse_result.is_valid:
        print("\n[3] SYNTAX ANALYSIS\n[OK]")
    else:
        print("\n[3] SYNTAX ANALYSIS\n[FAILED]")
        if res.parse_result:
            for err in res.parse_result.errors:
                print(f"  [SYNTAX ERROR] {err}")
        print("\n[STATUS] PIPELINE STOPPED AT SYNTAX ANALYSIS PHASE")
        print("=" * 65 + "\n")
        return

    # [4] AST Construction
    if res.ast is not None:
        print("\n[4] AST CONSTRUCTION\n[OK]")
        print("Abstract Syntax Tree:")
        print(res.ast.pretty_print())
    else:
        print("\n[4] AST CONSTRUCTION\n[NONE]")

    # [5] Semantic Analysis
    if res.semantic_result is not None and res.semantic_result.is_valid:
        print("\n[5] SEMANTIC ANALYSIS\n[OK]")
        print("Normalized Semantic Form:")
        print(res.semantic_form.display())
    else:
        print("\n[5] SEMANTIC ANALYSIS\n[FAILED]")
        if res.semantic_result:
            for err in res.semantic_result.errors:
                print(f"  [SEMANTIC ERROR] {err}")
        print("\n[STATUS] PIPELINE STOPPED AT SEMANTIC ANALYSIS PHASE")
        print("=" * 65 + "\n")
        return

    # [6] Intermediate Representation
    if res.ir_program is not None:
        print("\n[6] INTERMEDIATE REPRESENTATION\n[OK]")
        print(res.ir_program.to_text())
        print("\n[STATUS] COMPLETE COMPILER PIPELINE SUCCESSFUL")
    else:
        print("\n[6] INTERMEDIATE REPRESENTATION\n[SKIPPED / NOT GENERATED]")
        print(f"\n[STATUS] PIPELINE STOPPED AT {res.failed_phase} PHASE")

    print("=" * 65 + "\n")


def main() -> None:
    # 1. Primary Declarative Sentence
    display_pipeline_execution(
        "YESTERDAY ME WANT MILK",
        "Declarative Sentence with Time Marker and Object Entity",
    )

    # 2. Interrogative Question
    display_pipeline_execution(
        "WHAT YOU WANT",
        "Interrogative WH-Question Sentence",
    )

    # 3. Lexical Error Phase Isolation
    display_pipeline_execution(
        "ME DANCE MILK",
        "Lexical Error Isolation (Unknown Gloss 'DANCE')",
    )

    # 4. Syntax Error Phase Isolation
    display_pipeline_execution(
        "ME MILK WANT",
        "Syntax Error Isolation (Invalid SOV Word Order)",
    )


if __name__ == "__main__":
    main()

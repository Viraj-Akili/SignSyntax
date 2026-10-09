"""
SignSyntax - Complete Compiler Frontend Test Suite
Phase 2: Core Implementation (Frontend Pipeline, Semantic Error Handling, & IR Integration)

Tests end-to-end integration across:
  [1] Lexical Analysis (Lexer)
  [2] Syntax Analysis (Parser)
  [3] AST Construction (ASTBuilder)
  [4] Semantic Analysis (SemanticAnalyzer)
  [5] Intermediate Representation (IRGenerator)

Verifies error phase isolation, multiple semantic error accumulation,
normalized semantic form generation, and target-independent IR production.
"""

import pytest

from compiler.ast import (
    DeclarativeSentenceNode,
    ObjectNode,
    SentenceNode,
    SubjectNode,
    TimeNode,
    VerbPhraseNode,
)
from compiler.frontend import FrontendCompiler, FrontendResult, compile_frontend
from compiler.ir import IROpCode, IRProgram
from compiler.semantic_analyzer import (
    QuestionSemanticForm,
    SemanticAnalyzer,
    SemanticForm,
)


# ==============================================================================
# 1. Successful End-to-End Frontend Compilations
# ==============================================================================

@pytest.mark.parametrize("source,expected_subj,expected_action,expected_obj,expected_neg", [
    ("ME WANT MILK", "I", "want", "milk", False),
    ("YESTERDAY ME WANT MILK", "I", "want", "milk", False),
    ("ME WANT MILK NOT", "I", "want", "milk", True),
    ("SHE HAVE CAR BIG", "she", "have", "car", False),
    ("ME GO", "I", "go", None, False),
])
def test_frontend_successful_declaratives(
    source: str,
    expected_subj: str,
    expected_action: str,
    expected_obj: str,
    expected_neg: bool,
) -> None:
    """Tests complete successful frontend processing for declarative sentences."""
    res = compile_frontend(source)

    assert res.is_valid
    assert not res.has_errors
    assert res.failed_phase is None
    assert len(res.errors) == 0

    # Phase 1: Lexical
    assert res.lexer_result is not None
    assert res.lexer_result.is_valid

    # Phase 2: Syntax
    assert res.parse_result is not None
    assert res.parse_result.is_valid

    # Phase 3: AST
    assert res.ast is not None
    assert isinstance(res.ast, SentenceNode)

    # Phase 4: Semantic
    assert res.semantic_result is not None
    assert res.semantic_result.is_valid
    assert isinstance(res.semantic_form, SemanticForm)

    form = res.semantic_form
    assert form.subject == expected_subj
    assert form.action == expected_action
    assert form.object == expected_obj
    assert form.negated == expected_neg

    # Phase 5: Intermediate Representation
    assert res.ir_result is not None
    assert res.ir_result.is_valid
    assert res.ir_program is not None
    assert isinstance(res.ir_program, IRProgram)
    assert res.ir_program.program_type == "DECLARATIVE"
    assert len(res.ir_program) >= 3


@pytest.mark.parametrize("source,expected_wh,expected_subj,expected_action,expected_obj", [
    ("WHAT YOU WANT", "what", "you", "want", None),
    ("WHERE YOU GO", "where", "you", "go", None),
    ("WHAT BOOK", "what", None, None, "book"),
])
def test_frontend_successful_questions(
    source: str,
    expected_wh: str,
    expected_subj: str,
    expected_action: str,
    expected_obj: str,
    expected_neg: bool = False,
) -> None:
    """Tests complete successful frontend processing for interrogative questions."""
    res = compile_frontend(source)

    assert res.is_valid
    assert not res.has_errors
    assert res.failed_phase is None
    assert res.ast is not None
    assert isinstance(res.semantic_form, QuestionSemanticForm)

    q_form = res.semantic_form
    assert q_form.wh == expected_wh
    assert q_form.subject == expected_subj
    assert q_form.action == expected_action
    assert q_form.object == expected_obj

    # Phase 5: Intermediate Representation
    assert res.ir_result is not None
    assert res.ir_result.is_valid
    assert res.ir_program is not None
    assert isinstance(res.ir_program, IRProgram)
    assert res.ir_program.program_type == "QUESTION"
    assert res.ir_program[0].opcode == IROpCode.SET_WH
    assert res.ir_program[0].operand == expected_wh


# ==============================================================================
# 2. Phase Boundary Error Isolation
# ==============================================================================

def test_frontend_lexical_failure_isolation() -> None:
    """
    Verifies that unknown gloss 'DANCE' in 'ME DANCE MILK' fails at Lexical Analysis.
    Parser, AST, Semantic Analyzer, and IRGenerator MUST NOT run.
    """
    res = compile_frontend("ME DANCE MILK")

    assert not res.is_valid
    assert res.has_errors
    assert res.failed_phase == "LEXICAL"
    assert len(res.errors) == 1
    assert "Unknown gloss 'DANCE'" in str(res.errors[0])

    # Downstream phases are skipped
    assert res.lexer_result is not None
    assert res.lexer_result.has_errors
    assert res.parse_result is None
    assert res.ast is None
    assert res.semantic_result is None
    assert res.semantic_form is None
    assert res.ir_result is None
    assert res.ir_program is None


def test_frontend_syntax_failure_isolation() -> None:
    """
    Verifies that invalid token order 'ME MILK WANT' fails at Syntax Analysis.
    Lexer succeeds, Parser fails, AST is None, Semantic Analyzer & IR are not run.
    """
    res = compile_frontend("ME MILK WANT")

    assert not res.is_valid
    assert res.has_errors
    assert res.failed_phase == "SYNTAX"
    assert len(res.errors) == 1
    assert "Expected VERB predicate" in str(res.errors[0])

    # Lexer succeeded
    assert res.lexer_result is not None
    assert res.lexer_result.is_valid
    assert len(res.lexer_result.tokens) == 3

    # Parser failed & downstream skipped
    assert res.parse_result is not None
    assert res.parse_result.has_errors
    assert res.ast is None
    assert res.semantic_result is None
    assert res.semantic_form is None
    assert res.ir_result is None
    assert res.ir_program is None


# ==============================================================================
# 3. Semantic Error Accumulation & Diagnostic Quality
# ==============================================================================

def test_semantic_multiple_error_accumulation() -> None:
    """
    Verifies that SemanticAnalyzer accumulates multiple semantic errors across
    a single malformed AST rather than crashing or halting on the first error.
    """
    # Malformed AST with 3 category violations:
    # 1. TimeNode("ME") -> PRONOUN instead of TIME
    # 2. SubjectNode("MILK") -> NOUN instead of PRONOUN
    # 3. ObjectNode("WANT") -> VERB instead of NOUN
    malformed_ast = SentenceNode(
        body=DeclarativeSentenceNode(
            time=TimeNode(value="ME"),
            subject=SubjectNode(value="MILK"),
            verb_phrase=VerbPhraseNode(
                verb="LIKE",
                object=ObjectNode(value="WANT"),
            ),
        )
    )

    analyzer = SemanticAnalyzer()
    res = analyzer.analyze(malformed_ast)

    assert not res.is_valid
    assert res.has_errors
    assert len(res.errors) == 3
    assert res.semantic_form is None

    # Verify diagnostic message quality for each error
    err_messages = [str(e) for e in res.errors]
    assert any("TimeNode: symbol 'ME' has category PRONOUN; expected TIME" in msg for msg in err_messages)
    assert any("SubjectNode: symbol 'MILK' has category NOUN; expected PRONOUN" in msg for msg in err_messages)
    assert any("ObjectNode: symbol 'WANT' has category VERB; expected NOUN" in msg for msg in err_messages)


def test_semantic_error_no_python_crash() -> None:
    """Verifies that analyzing a malformed AST produces clean error diagnostics with zero crashes."""
    malformed_ast = SentenceNode(
        body=DeclarativeSentenceNode(
            subject=SubjectNode(value="INVALID_GLOSS"),
            verb_phrase=VerbPhraseNode(verb="WANT"),
        )
    )

    analyzer = SemanticAnalyzer()
    res = analyzer.analyze(malformed_ast)

    assert not res.is_valid
    assert len(res.errors) == 1
    assert "Symbol 'INVALID_GLOSS' is not declared in the Symbol Table" in res.errors[0].message


# ==============================================================================
# 4. FrontendResult Serialization & Display
# ==============================================================================

def test_frontend_result_to_dict_and_display() -> None:
    """Tests FrontendResult serialization and ASCII display formatting."""
    res = compile_frontend("YESTERDAY ME WANT MILK")
    d = res.to_dict()

    assert d["source"] == "YESTERDAY ME WANT MILK"
    assert d["is_valid"] is True
    assert d["failed_phase"] is None
    assert d["token_count"] == 4
    assert d["has_ast"] is True
    assert d["semantic_form"]["subject"] == "I"
    assert d["semantic_form"]["action"] == "want"
    assert d["semantic_form"]["object"] == "milk"
    assert d["has_ir"] is True
    assert d["ir_program"]["program_type"] == "DECLARATIVE"
    assert d["ir_program"]["instruction_count"] == 5

    disp = res.display()
    assert "SIGNSYNTAX FRONTEND REPORT" in disp
    assert "[1] Lexical Analysis: [OK]" in disp
    assert "[2] Syntax Analysis: [OK]" in disp
    assert "[3] AST Construction: [OK]" in disp
    assert "[4] Semantic Analysis: [OK]" in disp
    assert "Time: yesterday" in disp
    assert "Subject: I" in disp
    assert "Action: want" in disp
    assert "Object: milk" in disp
    assert "[6] Intermediate Representation: [OK]" in disp
    assert "SET_TIME" in disp
    assert "SET_SUBJECT" in disp
    assert "SET_ACTION" in disp
    assert "SET_OBJECT" in disp
    assert "EMIT_STATEMENT" in disp

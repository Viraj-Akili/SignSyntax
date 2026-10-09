"""
SignSyntax - Full Compiler Pipeline Comprehensive Integration Tests
Phase 2: Core Implementation (Full Pipeline Integration & Verification)

Verifies the complete 5-stage SignSyntax compiler pipeline:
  Gloss Stream -> Lexer -> Parser -> ASTBuilder -> SemanticAnalyzer -> IRGenerator

Tests:
1. End-to-end execution of all valid grammatical patterns.
2. Failure propagation and phase boundary isolation (Lexical, Syntax, Semantic, IR).
3. Absolute determinism across multiple compilation runs.
4. Core architectural invariants.
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
from compiler.ir import IRGenerator, IRInstruction, IROpCode, IRProgram
from compiler.semantic_analyzer import (
    QuestionSemanticForm,
    SemanticAnalyzer,
    SemanticError,
    SemanticForm,
    SemanticResult,
)


# ==============================================================================
# 1. Valid End-to-End Compilation Across Grammar Forms
# ==============================================================================

@pytest.mark.parametrize("source,expected_opcodes", [
    (
        "ME WANT MILK",
        [IROpCode.SET_SUBJECT, IROpCode.SET_ACTION, IROpCode.SET_OBJECT, IROpCode.EMIT_STATEMENT],
    ),
    (
        "YESTERDAY ME WANT MILK",
        [IROpCode.SET_TIME, IROpCode.SET_SUBJECT, IROpCode.SET_ACTION, IROpCode.SET_OBJECT, IROpCode.EMIT_STATEMENT],
    ),
    (
        "ME WANT MILK NOT",
        [IROpCode.SET_SUBJECT, IROpCode.SET_ACTION, IROpCode.SET_OBJECT, IROpCode.SET_NEGATION, IROpCode.EMIT_STATEMENT],
    ),
    (
        "SHE HAVE CAR BIG",
        [IROpCode.SET_SUBJECT, IROpCode.SET_ACTION, IROpCode.SET_OBJECT, IROpCode.SET_ADJECTIVE, IROpCode.EMIT_STATEMENT],
    ),
    (
        "ME GO",
        [IROpCode.SET_SUBJECT, IROpCode.SET_ACTION, IROpCode.EMIT_STATEMENT],
    ),
    (
        "WHAT YOU WANT",
        [IROpCode.SET_WH, IROpCode.SET_SUBJECT, IROpCode.SET_ACTION, IROpCode.EMIT_QUESTION],
    ),
    (
        "WHERE YOU GO",
        [IROpCode.SET_WH, IROpCode.SET_SUBJECT, IROpCode.SET_ACTION, IROpCode.EMIT_QUESTION],
    ),
    (
        "WHAT BOOK",
        [IROpCode.SET_WH, IROpCode.SET_OBJECT, IROpCode.EMIT_QUESTION],
    ),
    (
        "TOMORROW YOU LIKE BOOK NOT",
        [IROpCode.SET_TIME, IROpCode.SET_SUBJECT, IROpCode.SET_ACTION, IROpCode.SET_OBJECT, IROpCode.SET_NEGATION, IROpCode.EMIT_STATEMENT],
    ),
])
def test_full_pipeline_valid_sentences(source: str, expected_opcodes: list) -> None:
    """Verifies that all standard valid sentences successfully compile end-to-end to IR."""
    res = compile_frontend(source)

    # 1. Frontend validation
    assert res.is_valid is True
    assert res.has_errors is False
    assert res.failed_phase is None
    assert len(res.errors) == 0

    # 2. Stage artifacts exist
    assert res.lexer_result is not None and res.lexer_result.is_valid
    assert res.parse_result is not None and res.parse_result.is_valid
    assert res.ast is not None
    assert res.semantic_result is not None and res.semantic_result.is_valid
    assert res.semantic_form is not None
    assert res.ir_program is not None

    # 3. Verify exact opcode sequence
    actual_opcodes = [instr.opcode for instr in res.ir_program.instructions]
    assert actual_opcodes == expected_opcodes


# ==============================================================================
# 2. Failure Propagation and Phase Isolation
# ==============================================================================

def test_pipeline_lexical_failure_propagation() -> None:
    """
    Verifies that unknown gloss 'DANCE' in 'ME DANCE MILK' halts at Phase 1.
    Downstream phases (Parser, AST, Semantic, IR) must NOT execute.
    """
    res = compile_frontend("ME DANCE MILK")

    assert res.is_valid is False
    assert res.failed_phase == "LEXICAL"
    assert res.lexer_result is not None and res.lexer_result.has_errors
    assert res.parse_result is None
    assert res.ast is None
    assert res.semantic_result is None
    assert res.semantic_form is None
    assert res.ir_result is None
    assert res.ir_program is None


@pytest.mark.parametrize("invalid_syntax_source", [
    "ME MILK WANT",       # SOV word order
    "BOOK YOU READ",      # OSV word order
    "GO WE SCHOOL",       # Verb initial
    "ME WANT MILK BOOK",  # Trailing unconsumed token
    "WHAT YESTERDAY",     # Invalid WH question sequence
])
def test_pipeline_syntax_failure_propagation(invalid_syntax_source: str) -> None:
    """
    Verifies that syntax errors halt at Phase 2.
    Lexer succeeds, Parser fails, AST is None, Semantic & IR must NOT execute.
    """
    res = compile_frontend(invalid_syntax_source)

    assert res.is_valid is False
    assert res.failed_phase == "SYNTAX"
    assert res.lexer_result is not None and res.lexer_result.is_valid
    assert res.parse_result is not None and res.parse_result.has_errors
    assert res.ast is None
    assert res.semantic_result is None
    assert res.semantic_form is None
    assert res.ir_result is None
    assert res.ir_program is None


def test_pipeline_semantic_failure_isolation() -> None:
    """
    Verifies that a malformed AST failing Semantic Analysis halts before IR generation.
    """
    malformed_ast = SentenceNode(
        body=DeclarativeSentenceNode(
            subject=SubjectNode(value="MILK"),  # NOUN where PRONOUN expected
            verb_phrase=VerbPhraseNode(verb="WANT"),
        )
    )

    analyzer = SemanticAnalyzer()
    sem_res = analyzer.analyze(malformed_ast)

    assert sem_res.is_valid is False
    assert sem_res.has_errors is True
    assert sem_res.semantic_form is None

    # IR Generator must reject invalid SemanticResult
    ir_gen = IRGenerator()
    ir_res = ir_gen.generate(sem_res)

    assert ir_res.is_valid is False
    assert ir_res.program is None
    assert len(ir_res.errors) > 0


# ==============================================================================
# 3. Determinism Verification
# ==============================================================================

def test_pipeline_compilation_determinism() -> None:
    """
    Verifies that compiling the same input multiple times produces identical:
    - Token sequences
    - AST structures
    - Semantic forms
    - IR instruction sequences
    """
    source = "YESTERDAY ME WANT MILK"
    runs = [compile_frontend(source) for _ in range(5)]

    base = runs[0]
    for other in runs[1:]:
        assert other.is_valid is True
        assert other.to_dict() == base.to_dict()
        assert other.ast.to_dict() == base.ast.to_dict()
        assert other.semantic_form.to_dict() == base.semantic_form.to_dict()
        assert other.ir_program.to_dict() == base.ir_program.to_dict()
        assert other.ir_program.to_text() == base.ir_program.to_text()


# ==============================================================================
# 4. Core Pipeline Invariants
# ==============================================================================

def test_invariant_only_validated_semantics_reaches_ir() -> None:
    """Verifies that IRGenerator refuses to process unverified raw inputs."""
    ir_gen = IRGenerator()

    # Refuse None
    res_none = ir_gen.generate(None)
    assert res_none.is_valid is False

    # Refuse raw string
    res_str = ir_gen.generate("ME WANT MILK")
    assert res_str.is_valid is False

    # Refuse invalid SemanticResult
    bad_sem = SemanticResult(is_valid=False, semantic_form=None, errors=[SemanticError(message="bad", node_type="X")])
    res_bad = ir_gen.generate(bad_sem)
    assert res_bad.is_valid is False


def test_invariant_successful_compilation_exposes_all_artifacts() -> None:
    """Verifies that a valid compilation provides full access to every intermediate representation."""
    res = compile_frontend("SHE HAVE CAR BIG")

    assert res.is_valid is True
    assert res.source == "SHE HAVE CAR BIG"
    assert len(res.lexer_result.tokens) == 4
    assert res.parse_result.tree is not None
    assert res.ast is not None
    assert res.semantic_form is not None
    assert res.ir_program is not None
    assert len(res.ir_program) == 5

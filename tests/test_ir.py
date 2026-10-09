"""
SignSyntax - Intermediate Representation (IR) Unit Tests
Phase 2: Core Implementation (Intermediate Representation)

Validates:
- Correct target-independent IR instruction generation for declarative semantic forms.
- Correct IR generation for question semantic forms.
- Preservation of temporal framing, subject, predicate action, direct object, adjective modifier, and negation.
- Error handling for invalid/missing semantic inputs without Python exceptions.
- Deterministic instruction ordering and serialization.
"""

import pytest

from compiler.ir import (
    IRGenerationResult,
    IRGenerator,
    IRInstruction,
    IROpCode,
    IRProgram,
    IRValidationError,
)
from compiler.semantic_analyzer import (
    QuestionSemanticForm,
    SemanticError,
    SemanticForm,
    SemanticResult,
)


# ==============================================================================
# Declarative IR Tests
# ==============================================================================

def test_ir_declarative_basic_svo_me_want_milk():
    """Validates IR generation for basic SVO: ME WANT MILK."""
    form = SemanticForm(subject="I", action="want", object="milk")
    generator = IRGenerator()
    res = generator.generate(form)

    assert res.is_valid is True
    assert res.program is not None
    assert len(res.program) == 4

    assert res.program[0] == IRInstruction(opcode=IROpCode.SET_SUBJECT, operand="I", comment="Clause agent")
    assert res.program[1] == IRInstruction(opcode=IROpCode.SET_ACTION, operand="want", comment="Core predicate")
    assert res.program[2] == IRInstruction(opcode=IROpCode.SET_OBJECT, operand="milk", comment="Direct object entity")
    assert res.program[3] == IRInstruction(opcode=IROpCode.EMIT_STATEMENT, operand=None, comment="End of declarative statement")


def test_ir_declarative_with_time_yesterday_me_want_milk():
    """Validates IR generation with temporal context: YESTERDAY ME WANT MILK."""
    form = SemanticForm(time="yesterday", subject="I", action="want", object="milk")
    generator = IRGenerator()
    res = generator.generate(form)

    assert res.is_valid is True
    assert res.program is not None
    assert len(res.program) == 5

    assert res.program[0] == IRInstruction(opcode=IROpCode.SET_TIME, operand="yesterday", comment="Temporal context")
    assert res.program[1] == IRInstruction(opcode=IROpCode.SET_SUBJECT, operand="I", comment="Clause agent")
    assert res.program[2] == IRInstruction(opcode=IROpCode.SET_ACTION, operand="want", comment="Core predicate")
    assert res.program[3] == IRInstruction(opcode=IROpCode.SET_OBJECT, operand="milk", comment="Direct object entity")
    assert res.program[4] == IRInstruction(opcode=IROpCode.EMIT_STATEMENT, operand=None, comment="End of declarative statement")


def test_ir_declarative_with_negation_me_want_milk_not():
    """Validates IR generation with negation: ME WANT MILK NOT."""
    form = SemanticForm(subject="I", action="want", object="milk", negated=True)
    generator = IRGenerator()
    res = generator.generate(form)

    assert res.is_valid is True
    assert res.program is not None
    assert len(res.program) == 5

    assert res.program[0].opcode == IROpCode.SET_SUBJECT
    assert res.program[1].opcode == IROpCode.SET_ACTION
    assert res.program[2].opcode == IROpCode.SET_OBJECT
    assert res.program[3].opcode == IROpCode.SET_NEGATION
    assert res.program[3].operand is True
    assert res.program[4].opcode == IROpCode.EMIT_STATEMENT


def test_ir_declarative_with_adjective_she_have_car_big():
    """Validates IR generation with adjective modifier: SHE HAVE CAR BIG."""
    form = SemanticForm(subject="she", action="have", object="car", adjective="big")
    generator = IRGenerator()
    res = generator.generate(form)

    assert res.is_valid is True
    assert res.program is not None
    assert len(res.program) == 5

    assert res.program[0] == IRInstruction(opcode=IROpCode.SET_SUBJECT, operand="she", comment="Clause agent")
    assert res.program[1] == IRInstruction(opcode=IROpCode.SET_ACTION, operand="have", comment="Core predicate")
    assert res.program[2] == IRInstruction(opcode=IROpCode.SET_OBJECT, operand="car", comment="Direct object entity")
    assert res.program[3] == IRInstruction(opcode=IROpCode.SET_ADJECTIVE, operand="big", comment="Object qualifier")
    assert res.program[4] == IRInstruction(opcode=IROpCode.EMIT_STATEMENT, operand=None, comment="End of declarative statement")


def test_ir_declarative_intransitive_me_go():
    """Validates IR generation for intransitive SV: ME GO."""
    form = SemanticForm(subject="I", action="go")
    generator = IRGenerator()
    res = generator.generate(form)

    assert res.is_valid is True
    assert res.program is not None
    assert len(res.program) == 3

    assert res.program[0].opcode == IROpCode.SET_SUBJECT
    assert res.program[1].opcode == IROpCode.SET_ACTION
    assert res.program[2].opcode == IROpCode.EMIT_STATEMENT


# ==============================================================================
# Question IR Tests
# ==============================================================================

def test_ir_question_what_you_want():
    """Validates IR generation for WH-Subject-Verb question: WHAT YOU WANT."""
    form = QuestionSemanticForm(wh="what", subject="you", action="want")
    generator = IRGenerator()
    res = generator.generate(form)

    assert res.is_valid is True
    assert res.program is not None
    assert res.program.program_type == "QUESTION"
    assert len(res.program) == 4

    assert res.program[0] == IRInstruction(opcode=IROpCode.SET_WH, operand="what", comment="Interrogative operator")
    assert res.program[1] == IRInstruction(opcode=IROpCode.SET_SUBJECT, operand="you", comment="Interrogative target/agent")
    assert res.program[2] == IRInstruction(opcode=IROpCode.SET_ACTION, operand="want", comment="Interrogative action")
    assert res.program[3] == IRInstruction(opcode=IROpCode.EMIT_QUESTION, operand=None, comment="End of interrogative clause")


def test_ir_question_where_you_go():
    """Validates IR generation for WHERE YOU GO."""
    form = QuestionSemanticForm(wh="where", subject="you", action="go")
    generator = IRGenerator()
    res = generator.generate(form)

    assert res.is_valid is True
    assert res.program is not None
    assert len(res.program) == 4

    assert res.program[0].opcode == IROpCode.SET_WH
    assert res.program[0].operand == "where"
    assert res.program[1].opcode == IROpCode.SET_SUBJECT
    assert res.program[1].operand == "you"
    assert res.program[2].opcode == IROpCode.SET_ACTION
    assert res.program[2].operand == "go"
    assert res.program[3].opcode == IROpCode.EMIT_QUESTION


def test_ir_question_what_book():
    """Validates IR generation for WH-Object question: WHAT BOOK."""
    form = QuestionSemanticForm(wh="what", object="book")
    generator = IRGenerator()
    res = generator.generate(form)

    assert res.is_valid is True
    assert res.program is not None
    assert len(res.program) == 3

    assert res.program[0] == IRInstruction(opcode=IROpCode.SET_WH, operand="what", comment="Interrogative operator")
    assert res.program[1] == IRInstruction(opcode=IROpCode.SET_OBJECT, operand="book", comment="Queried entity")
    assert res.program[2] == IRInstruction(opcode=IROpCode.EMIT_QUESTION, operand=None, comment="End of interrogative clause")


# ==============================================================================
# IR Error & Validation Tests
# ==============================================================================

def test_ir_reject_none_semantic_input():
    """Validates that IR generator cleanly rejects None input."""
    generator = IRGenerator()
    res = generator.generate(None)

    assert res.is_valid is False
    assert res.program is None
    assert len(res.errors) == 1
    assert "None" in res.errors[0].message


def test_ir_reject_invalid_semantic_result():
    """Validates that IR generator cleanly rejects an invalid SemanticResult."""
    invalid_sem = SemanticResult(
        is_valid=False,
        semantic_form=None,
        errors=[SemanticError(message="Type mismatch", node_type="SubjectNode")],
    )
    generator = IRGenerator()
    res = generator.generate(invalid_sem)

    assert res.is_valid is False
    assert res.program is None
    assert len(res.errors) == 1
    assert "invalid or failing" in res.errors[0].message


def test_ir_unwrap_valid_semantic_result():
    """Validates that IR generator successfully unwraps a valid SemanticResult."""
    valid_sem = SemanticResult(
        is_valid=True,
        semantic_form=SemanticForm(subject="I", action="like", object="coffee"),
        errors=[],
    )
    generator = IRGenerator()
    res = generator.generate(valid_sem)

    assert res.is_valid is True
    assert res.program is not None
    assert len(res.program) == 4
    assert res.program[0].operand == "I"
    assert res.program[1].operand == "like"
    assert res.program[2].operand == "coffee"


def test_ir_reject_unsupported_type():
    """Validates that IR generator rejects arbitrary types."""
    generator = IRGenerator()
    res = generator.generate({"subject": "I", "action": "want"})

    assert res.is_valid is False
    assert res.program is None
    assert len(res.errors) == 1
    assert "Unsupported semantic form type" in res.errors[0].message


def test_ir_reject_missing_subject_in_declarative():
    """Validates that declarative IR generation fails when subject is empty."""
    form = SemanticForm(subject="", action="want")
    generator = IRGenerator()
    res = generator.generate(form)

    assert res.is_valid is False
    assert res.program is None
    assert any("subject" in err.message for err in res.errors)


def test_ir_reject_missing_action_in_declarative():
    """Validates that declarative IR generation fails when action is empty."""
    form = SemanticForm(subject="I", action="")
    generator = IRGenerator()
    res = generator.generate(form)

    assert res.is_valid is False
    assert res.program is None
    assert any("action" in err.message for err in res.errors)


def test_ir_reject_missing_wh_in_question():
    """Validates that question IR generation fails when WH word is empty."""
    form = QuestionSemanticForm(wh="")
    generator = IRGenerator()
    res = generator.generate(form)

    assert res.is_valid is False
    assert res.program is None
    assert any("wh" in err.message for err in res.errors)


# ==============================================================================
# IR Program Representation & Serialization Tests
# ==============================================================================

def test_ir_determinism():
    """Validates that identical semantic inputs always produce identical IR instructions."""
    form = SemanticForm(time="yesterday", subject="I", action="want", object="milk", negated=True)
    generator = IRGenerator()

    res1 = generator.generate(form)
    res2 = generator.generate(form)

    assert res1.program.to_dict() == res2.program.to_dict()
    assert res1.program.to_text() == res2.program.to_text()


def test_ir_program_serialization_and_display():
    """Validates serialization of IRProgram and IRInstruction."""
    program = IRProgram(program_type="DECLARATIVE")
    program.add(IROpCode.SET_SUBJECT, "I", comment="Test agent")
    program.add(IROpCode.SET_ACTION, "go")
    program.add(IROpCode.EMIT_STATEMENT)

    assert len(program) == 3
    dict_repr = program.to_dict()
    assert dict_repr["program_type"] == "DECLARATIVE"
    assert dict_repr["instruction_count"] == 3
    assert dict_repr["instructions"][0]["opcode"] == "SET_SUBJECT"
    assert dict_repr["instructions"][0]["operand"] == "I"

    text = program.to_text()
    assert "SET_SUBJECT" in text
    assert "SET_ACTION" in text
    assert "EMIT_STATEMENT" in text
    assert program.display() == text

"""
SignSyntax - Intermediate Representation (IR) Module
Phase 2: Core Implementation (Intermediate Representation)

The Intermediate Representation is the bridge between the compiler frontend
(Lexer, Parser, AST, Semantic Analyzer) and the future backend / code generator.
It consumes the validated, normalized SemanticForm / QuestionSemanticForm and
produces a target-independent, deterministic sequence of compiler instructions.

Compiler Architecture:
    Gloss Input
         ↓
       Lexer
         ↓
      Tokens
         ↓
      Parser
         ↓
    Parse Tree
         ↓
    ASTBuilder
         ↓
        AST
         ↓
    SemanticAnalyzer
         ↓
    SemanticForm / QuestionSemanticForm
         ↓
    IRGenerator
         ↓
      IRProgram (Target-Independent Instruction Sequence)

Strict Invariants:
- IR is target-independent and deterministic.
- IRGenerator accepts ONLY validated SemanticForm / QuestionSemanticForm.
- IRGenerator NEVER inspects raw source strings, tokens, or raw AST directly.
- IR generation fails cleanly on invalid/absent semantic input without Python crashes.
- IR is NOT final English text; it is an intermediate compiler instruction stream.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union

from compiler.semantic_analyzer import QuestionSemanticForm, SemanticForm, SemanticResult


# ==============================================================================
# IR OpCodes
# ==============================================================================

class IROpCode(str, Enum):
    """
    Enumeration of target-independent Intermediate Representation operation codes.
    """
    # Declarative / Clause Setup
    SET_TIME = "SET_TIME"
    SET_SUBJECT = "SET_SUBJECT"
    SET_ACTION = "SET_ACTION"
    SET_OBJECT = "SET_OBJECT"
    SET_ADJECTIVE = "SET_ADJECTIVE"
    SET_NEGATION = "SET_NEGATION"
    EMIT_STATEMENT = "EMIT_STATEMENT"

    # Interrogative / Question Setup
    SET_WH = "SET_WH"
    EMIT_QUESTION = "EMIT_QUESTION"

    def __str__(self) -> str:
        return self.value


# ==============================================================================
# IR Instruction & Program Representations
# ==============================================================================

@dataclass(frozen=True)
class IRInstruction:
    """
    Represents a single target-independent compiler intermediate instruction.

    Attributes:
        opcode (IROpCode): The operation code.
        operand (Optional[Any]): The semantic value/argument for the operation.
        comment (Optional[str]): Optional human-readable annotation.
    """
    opcode: IROpCode
    operand: Optional[Any] = None
    comment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the instruction to a dictionary."""
        return {
            "opcode": self.opcode.value if isinstance(self.opcode, IROpCode) else str(self.opcode),
            "operand": self.operand,
            "comment": self.comment,
        }

    def __str__(self) -> str:
        """Formatted textual representation of the IR instruction."""
        op_str = self.opcode.value if isinstance(self.opcode, IROpCode) else str(self.opcode)
        if self.operand is not None:
            if isinstance(self.operand, bool):
                operand_str = str(self.operand).lower()
            else:
                operand_str = str(self.operand)
            res = f"{op_str:<16} {operand_str}"
        else:
            res = f"{op_str}"
        
        if self.comment:
            res = f"{res:<32} ; {self.comment}"
        return res


@dataclass
class IRProgram:
    """
    Encapsulates an ordered sequence of IR instructions representing a compiled sign utterance.

    Attributes:
        instructions (List[IRInstruction]): Ordered list of intermediate instructions.
        program_type (str): Type of program ('DECLARATIVE' or 'QUESTION').
    """
    instructions: List[IRInstruction] = field(default_factory=list)
    program_type: str = "DECLARATIVE"

    def add(self, opcode: IROpCode, operand: Optional[Any] = None, comment: Optional[str] = None) -> None:
        """Appends a new instruction to the program."""
        self.instructions.append(IRInstruction(opcode=opcode, operand=operand, comment=comment))

    def __len__(self) -> int:
        return len(self.instructions)

    def __getitem__(self, index: int) -> IRInstruction:
        return self.instructions[index]

    def __iter__(self):
        return iter(self.instructions)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the complete IR program to a dictionary."""
        return {
            "program_type": self.program_type,
            "instruction_count": len(self.instructions),
            "instructions": [instr.to_dict() for instr in self.instructions],
        }

    def to_text(self) -> str:
        """Renders formatted ASCII text of the intermediate representation."""
        lines = [
            "=" * 60,
            "           SIGNSYNTAX INTERMEDIATE REPRESENTATION",
            "=" * 60,
            f"Program Type: {self.program_type}",
            f"Total Instructions: {len(self.instructions)}",
            "-" * 60,
        ]
        for idx, instr in enumerate(self.instructions):
            lines.append(f"IR[{idx:02d}]  {instr}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def display(self) -> str:
        """Alias for to_text() for uniform display API."""
        return self.to_text()


# ==============================================================================
# IR Errors and Validation
# ==============================================================================

@dataclass(frozen=True)
class IRValidationError(Exception):
    """
    Represents an error encountered during IR generation from an invalid semantic form.

    Attributes:
        message (str): Descriptive reason for IR generation failure.
    """
    message: str

    def __str__(self) -> str:
        return f"IR GENERATION ERROR: {self.message}"


@dataclass
class IRGenerationResult:
    """
    Encapsulates the outcome of the IR generation phase.

    Attributes:
        is_valid (bool): True if IR generation succeeded.
        program (Optional[IRProgram]): Generated IRProgram if valid.
        errors (List[IRValidationError]): List of validation errors encountered.
    """
    is_valid: bool
    program: Optional[IRProgram] = None
    errors: List[IRValidationError] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return not self.is_valid or len(self.errors) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "program": self.program.to_dict() if self.program else None,
            "errors": [str(err) for err in self.errors],
        }


# ==============================================================================
# IR Generator
# ==============================================================================

class IRGenerator:
    """
    Translates validated SemanticForm or QuestionSemanticForm instances into
    a target-independent IRProgram.

    Enforces strict compiler boundaries:
    - Inputs must be verified SemanticForm or QuestionSemanticForm instances.
    - Disallows direct access to raw strings, tokens, or unverified ASTs.
    - Deterministically orders instructions for predictable backend consumption.
    """

    def generate(
        self,
        semantic_input: Optional[Union[SemanticForm, QuestionSemanticForm, SemanticResult]],
    ) -> IRGenerationResult:
        """
        Generates an IRProgram from normalized semantic input.

        Args:
            semantic_input: A SemanticForm, QuestionSemanticForm, or SemanticResult.

        Returns:
            IRGenerationResult: Contains the generated IRProgram or structured diagnostics.
        """
        # Unwrap SemanticResult if provided
        if isinstance(semantic_input, SemanticResult):
            if not semantic_input.is_valid or semantic_input.semantic_form is None:
                err = IRValidationError("Cannot generate IR from invalid or failing SemanticResult.")
                return IRGenerationResult(is_valid=False, program=None, errors=[err])
            semantic_input = semantic_input.semantic_form

        # Validate presence of input
        if semantic_input is None:
            err = IRValidationError("Semantic form input is None; IR generation aborted.")
            return IRGenerationResult(is_valid=False, program=None, errors=[err])

        # Dispatch based on semantic form type
        if isinstance(semantic_input, SemanticForm):
            return self._generate_declarative_ir(semantic_input)
        elif isinstance(semantic_input, QuestionSemanticForm):
            return self._generate_question_ir(semantic_input)
        else:
            err = IRValidationError(
                f"Unsupported semantic form type '{type(semantic_input).__name__}'. "
                "Expected SemanticForm or QuestionSemanticForm."
            )
            return IRGenerationResult(is_valid=False, program=None, errors=[err])

    def _generate_declarative_ir(self, form: SemanticForm) -> IRGenerationResult:
        """
        Generates deterministic IR instructions for a declarative sentence.

        Validation rules:
        - subject must be non-empty string.
        - action must be non-empty string.
        """
        errors: List[IRValidationError] = []

        if not form.subject or not str(form.subject).strip():
            errors.append(IRValidationError("Declarative SemanticForm missing required 'subject' field."))

        if not form.action or not str(form.action).strip():
            errors.append(IRValidationError("Declarative SemanticForm missing required 'action' field."))

        if errors:
            return IRGenerationResult(is_valid=False, program=None, errors=errors)

        program = IRProgram(program_type="DECLARATIVE")

        # 1. Temporal frame (if present)
        if form.time is not None:
            program.add(IROpCode.SET_TIME, form.time, comment="Temporal context")

        # 2. Subject
        program.add(IROpCode.SET_SUBJECT, form.subject, comment="Clause agent")

        # 3. Action / Predicate verb
        program.add(IROpCode.SET_ACTION, form.action, comment="Core predicate")

        # 4. Direct object (if present)
        if form.object is not None:
            program.add(IROpCode.SET_OBJECT, form.object, comment="Direct object entity")

        # 5. Adjective modifier (if present)
        if form.adjective is not None:
            program.add(IROpCode.SET_ADJECTIVE, form.adjective, comment="Object qualifier")

        # 6. Negation flag (if True)
        if form.negated:
            program.add(IROpCode.SET_NEGATION, True, comment="Polarity inversion")

        # 7. Final statement emission delimiter
        program.add(IROpCode.EMIT_STATEMENT, None, comment="End of declarative statement")

        return IRGenerationResult(is_valid=True, program=program, errors=[])

    def _generate_question_ir(self, form: QuestionSemanticForm) -> IRGenerationResult:
        """
        Generates deterministic IR instructions for an interrogative question.

        Validation rules:
        - wh must be non-empty string.
        """
        errors: List[IRValidationError] = []

        if not form.wh or not str(form.wh).strip():
            errors.append(IRValidationError("QuestionSemanticForm missing required 'wh' interrogative field."))

        if errors:
            return IRGenerationResult(is_valid=False, program=None, errors=errors)

        program = IRProgram(program_type="QUESTION")

        # 1. WH-interrogative identifier
        program.add(IROpCode.SET_WH, form.wh, comment="Interrogative operator")

        # 2. Subject (if present, e.g., 'you')
        if form.subject is not None:
            program.add(IROpCode.SET_SUBJECT, form.subject, comment="Interrogative target/agent")

        # 3. Action (if present, e.g., 'want', 'go')
        if form.action is not None:
            program.add(IROpCode.SET_ACTION, form.action, comment="Interrogative action")

        # 4. Object (if present, e.g., 'book' in 'WHAT BOOK')
        if form.object is not None:
            program.add(IROpCode.SET_OBJECT, form.object, comment="Queried entity")

        # 5. Final question emission delimiter
        program.add(IROpCode.EMIT_QUESTION, None, comment="End of interrogative clause")

        return IRGenerationResult(is_valid=True, program=program, errors=[])

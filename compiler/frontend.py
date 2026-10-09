"""
SignSyntax - Compiler Frontend Driver Module
Phase 2: Core Implementation (Frontend Pipeline Integration & IR Generation)

This module provides the unified end-to-end frontend driver that coordinates:
  [1] Lexical Analysis (Lexer)
  [2] Syntax Analysis (Parser)
  [3] AST Construction (ASTBuilder)
  [4] Semantic Analysis (SemanticAnalyzer)
  [5] Intermediate Representation (IRGenerator)

Preserves strict compiler phase boundaries:
- Lexical failures halt before Parser execution.
- Syntax failures halt before AST & Semantic Analysis.
- Semantic failures produce structured diagnostics without generating IR.
- Valid inputs produce a fully verified AST, normalized SemanticForm, and target-independent IRProgram.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from compiler.ast import SentenceNode
from compiler.ir import (
    IRGenerationResult,
    IRGenerator,
    IRInstruction,
    IRProgram,
    IRValidationError,
)
from compiler.lexer import Lexer, LexerResult, LexicalError
from compiler.parser import ParseError, Parser, ParseResult
from compiler.semantic_analyzer import (
    QuestionSemanticForm,
    SemanticAnalyzer,
    SemanticError,
    SemanticForm,
    SemanticResult,
)
from compiler.symbol_table import SymbolTable


# ==============================================================================
# Frontend Compilation Result
# ==============================================================================

@dataclass
class FrontendResult:
    """
    Encapsulates the complete outcome of the SignSyntax compiler frontend pipeline.

    Attributes:
        source (str): Original input gloss string.
        is_valid (bool): True if all frontend phases + IR generation succeeded without errors.
        failed_phase (Optional[str]): Name of the phase that failed ('LEXICAL', 'SYNTAX', 'SEMANTIC', 'IR'), or None.
        lexer_result (Optional[LexerResult]): Result from Lexical Analysis.
        parse_result (Optional[ParseResult]): Result from Syntax Analysis.
        ast (Optional[SentenceNode]): Root of the Abstract Syntax Tree if constructed.
        semantic_result (Optional[SemanticResult]): Result from Semantic Analysis.
        semantic_form (Optional[Union[SemanticForm, QuestionSemanticForm]]): Normalized semantic representation.
        ir_result (Optional[IRGenerationResult]): Result from Intermediate Representation generation.
        ir_program (Optional[IRProgram]): Target-independent IR program if generated.
        errors (List[Union[LexicalError, ParseError, SemanticError, IRValidationError]]): Aggregated diagnostics from the failing phase.
    """
    source: str
    is_valid: bool
    failed_phase: Optional[str] = None
    lexer_result: Optional[LexerResult] = None
    parse_result: Optional[ParseResult] = None
    ast: Optional[SentenceNode] = None
    semantic_result: Optional[SemanticResult] = None
    semantic_form: Optional[Union[SemanticForm, QuestionSemanticForm]] = None
    ir_result: Optional[IRGenerationResult] = None
    ir_program: Optional[IRProgram] = None
    errors: List[Union[LexicalError, ParseError, SemanticError, IRValidationError]] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Returns True if one or more compiler errors occurred in any phase."""
        return not self.is_valid or len(self.errors) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the frontend compilation outcome to a dictionary."""
        return {
            "source": self.source,
            "is_valid": self.is_valid,
            "failed_phase": self.failed_phase,
            "token_count": len(self.lexer_result.tokens) if self.lexer_result else 0,
            "has_ast": self.ast is not None,
            "semantic_form": self.semantic_form.to_dict() if self.semantic_form else None,
            "has_ir": self.ir_program is not None,
            "ir_program": self.ir_program.to_dict() if self.ir_program else None,
            "errors": [str(err) for err in self.errors],
        }

    def display(self) -> str:
        """Renders a comprehensive ASCII summary of all compiler frontend stages."""
        lines = [
            "=" * 60,
            "               SIGNSYNTAX FRONTEND REPORT",
            "=" * 60,
            f"Source Input: {self.source}",
            f"Overall Status: {'[OK] VALID' if self.is_valid else f'[FAILED] at {self.failed_phase}'}",
            "-" * 60,
        ]

        # Phase 1: Lexical
        if self.lexer_result is not None:
            if self.lexer_result.has_errors:
                lines.append("[1] Lexical Analysis: [FAILED]")
                for err in self.lexer_result.errors:
                    lines.append(f"    [LEXICAL ERROR] {err}")
            else:
                tok_repr = " ".join([f"[{t.lexeme}:{t.token_type}]" for t in self.lexer_result.tokens])
                lines.append(f"[1] Lexical Analysis: [OK] ({len(self.lexer_result.tokens)} tokens)")
                lines.append(f"    Tokens: {tok_repr}")
        else:
            lines.append("[1] Lexical Analysis: [SKIPPED]")

        # Phase 2: Syntax
        if self.parse_result is not None:
            if self.parse_result.has_errors:
                lines.append("\n[2] Syntax Analysis: [FAILED]")
                for err in self.parse_result.errors:
                    lines.append(f"    [SYNTAX ERROR] {err}")
            else:
                lines.append("\n[2] Syntax Analysis: [OK]")
        else:
            lines.append("\n[2] Syntax Analysis: [SKIPPED]")

        # Phase 3: AST
        if self.ast is not None:
            lines.append("\n[3] AST Construction: [OK]")
            lines.append(self.ast.pretty_print())
        else:
            lines.append("\n[3] AST Construction: [NONE]")

        # Phase 4: Semantic
        if self.semantic_result is not None:
            if self.semantic_result.has_errors:
                lines.append("\n[4] Semantic Analysis: [FAILED]")
                for err in self.semantic_result.errors:
                    lines.append(f"    [SEMANTIC ERROR] {err}")
            else:
                lines.append("\n[4] Semantic Analysis: [OK]")
                lines.append("\n[5] Normalized Semantic Form:")
                lines.append(self.semantic_form.display())
        else:
            lines.append("\n[4] Semantic Analysis: [SKIPPED]")

        # Phase 5: Intermediate Representation
        if self.ir_program is not None:
            lines.append("\n[6] Intermediate Representation: [OK]")
            lines.append(self.ir_program.to_text())
        elif self.ir_result is not None and self.ir_result.has_errors:
            lines.append("\n[6] Intermediate Representation: [FAILED]")
            for err in self.ir_result.errors:
                lines.append(f"    [IR ERROR] {err}")
        else:
            lines.append("\n[6] Intermediate Representation: [SKIPPED / NOT GENERATED]")

        lines.append("=" * 60)
        return "\n".join(lines)


# ==============================================================================
# Frontend Compiler Engine
# ==============================================================================

class FrontendCompiler:
    """
    Unified SignSyntax Compiler Frontend Driver.

    Coordinates the 5 sequential compiler stages and enforces strict phase separation:
    Lexer -> Parser -> ASTBuilder -> SemanticAnalyzer -> IRGenerator
    """

    def __init__(self, symbol_table: Optional[SymbolTable] = None) -> None:
        """
        Initializes the FrontendCompiler.

        Args:
            symbol_table (Optional[SymbolTable]): Shared SymbolTable for lexing and semantics.
        """
        self.symbol_table: SymbolTable = symbol_table if symbol_table is not None else SymbolTable()
        self.lexer: Lexer = Lexer(symbol_table=self.symbol_table)
        self.semantic_analyzer: SemanticAnalyzer = SemanticAnalyzer(symbol_table=self.symbol_table)
        self.ir_generator: IRGenerator = IRGenerator()

    def compile(self, source: str) -> FrontendResult:
        """
        Executes the complete frontend pipeline on the input string.

        Args:
            source (str): Raw sign gloss input string.

        Returns:
            FrontendResult: Detailed report containing outputs and diagnostics from each phase.
        """
        # ---------------------------------------------------------------------
        # Phase 1: Lexical Analysis
        # ---------------------------------------------------------------------
        lex_res = self.lexer.tokenize(source)
        if lex_res.has_errors:
            return FrontendResult(
                source=source,
                is_valid=False,
                failed_phase="LEXICAL",
                lexer_result=lex_res,
                parse_result=None,
                ast=None,
                semantic_result=None,
                semantic_form=None,
                ir_result=None,
                ir_program=None,
                errors=list(lex_res.errors),
            )

        # ---------------------------------------------------------------------
        # Phase 2 & 3: Syntax Analysis & AST Construction
        # ---------------------------------------------------------------------
        parser = Parser(lex_res.tokens)
        parse_res = parser.parse()
        if parse_res.has_errors:
            return FrontendResult(
                source=source,
                is_valid=False,
                failed_phase="SYNTAX",
                lexer_result=lex_res,
                parse_result=parse_res,
                ast=None,
                semantic_result=None,
                semantic_form=None,
                ir_result=None,
                ir_program=None,
                errors=list(parse_res.errors),
            )

        ast = parse_res.ast
        if ast is None:
            # Empty input or unhandled parse state
            return FrontendResult(
                source=source,
                is_valid=False,
                failed_phase="SYNTAX",
                lexer_result=lex_res,
                parse_result=parse_res,
                ast=None,
                semantic_result=None,
                semantic_form=None,
                ir_result=None,
                ir_program=None,
                errors=[ParseError(message="AST construction failed after syntax parse", position=0)],
            )

        # ---------------------------------------------------------------------
        # Phase 4: Semantic Analysis
        # ---------------------------------------------------------------------
        sem_res = self.semantic_analyzer.analyze(ast)
        if sem_res.has_errors or sem_res.semantic_form is None:
            return FrontendResult(
                source=source,
                is_valid=False,
                failed_phase="SEMANTIC",
                lexer_result=lex_res,
                parse_result=parse_res,
                ast=ast,
                semantic_result=sem_res,
                semantic_form=None,
                ir_result=None,
                ir_program=None,
                errors=list(sem_res.errors),
            )

        # ---------------------------------------------------------------------
        # Phase 5: Intermediate Representation (IR) Generation
        # ---------------------------------------------------------------------
        ir_res = self.ir_generator.generate(sem_res.semantic_form)
        if ir_res.has_errors or ir_res.program is None:
            return FrontendResult(
                source=source,
                is_valid=False,
                failed_phase="IR",
                lexer_result=lex_res,
                parse_result=parse_res,
                ast=ast,
                semantic_result=sem_res,
                semantic_form=sem_res.semantic_form,
                ir_result=ir_res,
                ir_program=None,
                errors=list(ir_res.errors),
            )

        # All frontend stages succeeded
        return FrontendResult(
            source=source,
            is_valid=True,
            failed_phase=None,
            lexer_result=lex_res,
            parse_result=parse_res,
            ast=ast,
            semantic_result=sem_res,
            semantic_form=sem_res.semantic_form,
            ir_result=ir_res,
            ir_program=ir_res.program,
            errors=[],
        )


def compile_frontend(source: str, symbol_table: Optional[SymbolTable] = None) -> FrontendResult:
    """
    Convenience function to run the full SignSyntax frontend pipeline on a source string.

    Args:
        source (str): Raw sign gloss input string.
        symbol_table (Optional[SymbolTable]): Optional custom symbol table.

    Returns:
        FrontendResult: Complete compilation outcome including IRProgram.
    """
    return FrontendCompiler(symbol_table=symbol_table).compile(source)

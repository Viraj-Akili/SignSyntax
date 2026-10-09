"""
SignSyntax - Semantic Analyzer Module
Phase 2: Core Implementation (Semantic Analysis & Type Checking)

The Semantic Analyzer is the third phase of the SignSyntax compiler frontend.
It consumes the Abstract Syntax Tree (AST) produced by the Parser/ASTBuilder,
verifies that all terminal symbols are semantically valid within the Symbol Table,
validates that AST nodes match their declared grammatical categories, binds normalized
semantic meanings, and produces a normalized semantic representation.

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
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from compiler.ast import (
    ASTNode,
    AdjectiveNode,
    DeclarativeSentenceNode,
    NegationNode,
    ObjectNode,
    QuestionNode,
    SentenceNode,
    SubjectNode,
    TimeNode,
    VerbPhraseNode,
    WHWordNode,
)
from compiler.symbol_table import Symbol, SymbolTable


# ==============================================================================
# Semantic Errors
# ==============================================================================

@dataclass(frozen=True)
class SemanticError(Exception):
    """
    Represents a semantic violation or type/category mismatch in the AST.

    Attributes:
        message (str): Descriptive explanation of the semantic inconsistency.
        node_type (str): The AST node type where the violation occurred.
        lexeme (Optional[str]): The gloss string causing the violation.
        expected_category (Optional[str]): Expected Symbol Table category.
        found_category (Optional[str]): Actual Symbol Table category found.
    """
    message: str
    node_type: str
    lexeme: Optional[str] = None
    expected_category: Optional[str] = None
    found_category: Optional[str] = None

    def __str__(self) -> str:
        return f"SEMANTIC ERROR in {self.node_type}: {self.message}"


# ==============================================================================
# Normalized Semantic Representations
# ==============================================================================

@dataclass
class SemanticForm:
    """
    Normalized semantic representation for declarative sign sentences.

    Attributes:
        subject (str): Normalized meaning of the subject (e.g., 'I', 'you').
        action (str): Normalized meaning of the verb action (e.g., 'want', 'like').
        time (Optional[str]): Normalized meaning of the temporal frame (e.g., 'yesterday'), or None.
        object (Optional[str]): Normalized meaning of the direct object entity (e.g., 'milk'), or None.
        adjective (Optional[str]): Normalized meaning of the qualifying adjective (e.g., 'big'), or None.
        negated (bool): True if the clause contains a negation particle, False otherwise.
    """
    subject: str
    action: str
    time: Optional[str] = None
    object: Optional[str] = None
    adjective: Optional[str] = None
    negated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes semantic form to a clean structured dictionary."""
        return {
            "form_type": "Declarative",
            "time": self.time,
            "subject": self.subject,
            "action": self.action,
            "object": self.object,
            "adjective": self.adjective,
            "negated": self.negated,
        }

    def display(self) -> str:
        """Renders a readable formatted representation of semantic attributes."""
        lines = [
            f"Time: {self.time if self.time is not None else 'null'}",
            f"Subject: {self.subject}",
            f"Action: {self.action}",
            f"Object: {self.object if self.object is not None else 'null'}",
            f"Adjective: {self.adjective if self.adjective is not None else 'null'}",
            f"Negated: {str(self.negated).lower()}",
        ]
        return "\n".join(lines)


@dataclass
class QuestionSemanticForm:
    """
    Normalized semantic representation for interrogative questions.

    Attributes:
        wh (str): Normalized meaning of the WH-word (e.g., 'what', 'where').
        subject (Optional[str]): Normalized meaning of the subject (e.g., 'you'), or None.
        action (Optional[str]): Normalized meaning of the verb action (e.g., 'want'), or None.
        object (Optional[str]): Normalized meaning of the queried object (e.g., 'book'), or None.
    """
    wh: str
    subject: Optional[str] = None
    action: Optional[str] = None
    object: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes question semantic form to a clean structured dictionary."""
        return {
            "form_type": "Question",
            "wh": self.wh,
            "subject": self.subject,
            "action": self.action,
            "object": self.object,
        }

    def display(self) -> str:
        """Renders a readable formatted representation of question semantic attributes."""
        lines = [
            f"WH-Type: {self.wh}",
            f"Subject: {self.subject if self.subject is not None else 'null'}",
            f"Action: {self.action if self.action is not None else 'null'}",
            f"Object: {self.object if self.object is not None else 'null'}",
        ]
        return "\n".join(lines)


@dataclass
class SemanticResult:
    """
    Encapsulates the result of semantic analysis.

    Attributes:
        semantic_form (Optional[Union[SemanticForm, QuestionSemanticForm]]): Normalized representation if valid.
        errors (List[SemanticError]): List of semantic errors detected during analysis.
        is_valid (bool): True if analysis succeeded with zero semantic errors.
    """
    semantic_form: Optional[Union[SemanticForm, QuestionSemanticForm]] = None
    errors: List[SemanticError] = field(default_factory=list)
    is_valid: bool = True

    @property
    def has_errors(self) -> bool:
        """Returns True if one or more semantic errors occurred."""
        return len(self.errors) > 0 or not self.is_valid


# ==============================================================================
# Semantic Analyzer Engine
# ==============================================================================

class SemanticAnalyzer:
    """
    Hand-written Semantic Analyzer for SignSyntax.

    Performs semantic checking, Symbol Table cross-validation, category verification,
    and semantic attribute binding across Abstract Syntax Tree nodes.
    """

    def __init__(self, symbol_table: Optional[SymbolTable] = None) -> None:
        """
        Initializes the SemanticAnalyzer with the compiler SymbolTable.

        Args:
            symbol_table (Optional[SymbolTable]): Symbol table for identifier resolution.
        """
        self.symbol_table: SymbolTable = symbol_table if symbol_table is not None else SymbolTable()

    def analyze(self, ast: Optional[SentenceNode]) -> SemanticResult:
        """
        Performs semantic analysis on the given AST.

        Args:
            ast (Optional[SentenceNode]): Root AST node from parser / ASTBuilder.

        Returns:
            SemanticResult: Object containing normalized semantic form or semantic errors.
        """
        if ast is None or not isinstance(ast, SentenceNode):
            error = SemanticError(
                message="Cannot perform semantic analysis on empty or invalid AST root node",
                node_type="SentenceNode",
            )
            return SemanticResult(semantic_form=None, errors=[error], is_valid=False)

        errors: List[SemanticError] = []

        if ast.is_declarative:
            form = self._analyze_declarative(ast.body, errors)
        elif ast.is_question:
            form = self._analyze_question(ast.body, errors)
        else:
            errors.append(SemanticError(
                message=f"Unrecognized AST sentence body type: {type(ast.body).__name__}",
                node_type="SentenceNode",
            ))
            return SemanticResult(semantic_form=None, errors=errors, is_valid=False)

        if errors:
            return SemanticResult(semantic_form=None, errors=errors, is_valid=False)

        return SemanticResult(semantic_form=form, errors=[], is_valid=True)

    # -------------------------------------------------------------------------
    # Internal Node Analysis & Symbol Table Verification
    # -------------------------------------------------------------------------

    def _resolve_and_verify(
        self,
        lexeme: str,
        expected_category: str,
        node_type: str,
        errors: List[SemanticError],
    ) -> Optional[Symbol]:
        """
        Validates that a lexeme exists in the Symbol Table and belongs to the expected category.

        Args:
            lexeme (str): The sign gloss string.
            expected_category (str): Expected Symbol Table token_type (e.g., 'PRONOUN', 'VERB').
            node_type (str): The AST node class name for error context.
            errors (List[SemanticError]): Error accumulator.

        Returns:
            Optional[Symbol]: Resolved Symbol entry, or None if validation failed.
        """
        symbol = self.symbol_table.lookup(lexeme)

        if symbol is None:
            err = SemanticError(
                message=f"Symbol '{lexeme}' is not declared in the Symbol Table",
                node_type=node_type,
                lexeme=lexeme,
                expected_category=expected_category,
                found_category=None,
            )
            errors.append(err)
            return None

        if symbol.token_type != expected_category:
            err = SemanticError(
                message=f"symbol '{lexeme}' has category {symbol.token_type}; expected {expected_category}",
                node_type=node_type,
                lexeme=lexeme,
                expected_category=expected_category,
                found_category=symbol.token_type,
            )
            errors.append(err)
            return None

        return symbol

    def _analyze_declarative(
        self,
        decl_node: DeclarativeSentenceNode,
        errors: List[SemanticError],
    ) -> Optional[SemanticForm]:
        """Validates and normalizes a DeclarativeSentenceNode AST."""
        time_meaning: Optional[str] = None
        subject_meaning: str = ""
        action_meaning: str = ""
        object_meaning: Optional[str] = None
        adjective_meaning: Optional[str] = None
        is_negated: bool = False

        # 1. Optional TimeMarker validation (TIME)
        if decl_node.time is not None:
            time_sym = self._resolve_and_verify(
                lexeme=decl_node.time.value,
                expected_category="TIME",
                node_type="TimeNode",
                errors=errors,
            )
            if time_sym is not None:
                time_meaning = time_sym.meaning

        # 2. Mandatory Subject validation (PRONOUN)
        if decl_node.subject is not None:
            subj_sym = self._resolve_and_verify(
                lexeme=decl_node.subject.value,
                expected_category="PRONOUN",
                node_type="SubjectNode",
                errors=errors,
            )
            if subj_sym is not None:
                subject_meaning = subj_sym.meaning
        else:
            errors.append(SemanticError(
                message="Missing required SubjectNode in DeclarativeSentence",
                node_type="DeclarativeSentenceNode",
            ))

        # 3. Mandatory VerbPhrase validation
        if decl_node.verb_phrase is not None:
            vp = decl_node.verb_phrase
            verb_sym = self._resolve_and_verify(
                lexeme=vp.verb,
                expected_category="VERB",
                node_type="VerbPhraseNode",
                errors=errors,
            )
            if verb_sym is not None:
                action_meaning = verb_sym.meaning

            # Optional Object validation (NOUN)
            if vp.object is not None:
                obj_sym = self._resolve_and_verify(
                    lexeme=vp.object.value,
                    expected_category="NOUN",
                    node_type="ObjectNode",
                    errors=errors,
                )
                if obj_sym is not None:
                    object_meaning = obj_sym.meaning

            # Optional Adjective validation (ADJECTIVE)
            if vp.adjective is not None:
                adj_sym = self._resolve_and_verify(
                    lexeme=vp.adjective.value,
                    expected_category="ADJECTIVE",
                    node_type="AdjectiveNode",
                    errors=errors,
                )
                if adj_sym is not None:
                    adjective_meaning = adj_sym.meaning
        else:
            errors.append(SemanticError(
                message="Missing required VerbPhraseNode in DeclarativeSentence",
                node_type="DeclarativeSentenceNode",
            ))

        # 4. Optional Negation validation (NEGATION)
        if decl_node.negation is not None:
            neg_sym = self._resolve_and_verify(
                lexeme=decl_node.negation.value,
                expected_category="NEGATION",
                node_type="NegationNode",
                errors=errors,
            )
            if neg_sym is not None:
                is_negated = True

        if errors:
            return None

        return SemanticForm(
            subject=subject_meaning,
            action=action_meaning,
            time=time_meaning,
            object=object_meaning,
            adjective=adjective_meaning,
            negated=is_negated,
        )

    def _analyze_question(
        self,
        q_node: QuestionNode,
        errors: List[SemanticError],
    ) -> Optional[QuestionSemanticForm]:
        """Validates and normalizes a QuestionNode AST."""
        wh_meaning: str = ""
        subject_meaning: Optional[str] = None
        action_meaning: Optional[str] = None
        object_meaning: Optional[str] = None

        # 1. Mandatory WHWord validation (WH_WORD)
        if q_node.wh_word is not None:
            wh_sym = self._resolve_and_verify(
                lexeme=q_node.wh_word.value,
                expected_category="WH_WORD",
                node_type="WHWordNode",
                errors=errors,
            )
            if wh_sym is not None:
                wh_meaning = wh_sym.meaning
        else:
            errors.append(SemanticError(
                message="Missing required WHWordNode in Question",
                node_type="QuestionNode",
            ))

        # 2. Optional Subject validation (PRONOUN)
        if q_node.subject is not None:
            subj_sym = self._resolve_and_verify(
                lexeme=q_node.subject.value,
                expected_category="PRONOUN",
                node_type="SubjectNode",
                errors=errors,
            )
            if subj_sym is not None:
                subject_meaning = subj_sym.meaning

        # 3. Optional VerbPhrase validation (VERB)
        if q_node.verb_phrase is not None:
            vp = q_node.verb_phrase
            verb_sym = self._resolve_and_verify(
                lexeme=vp.verb,
                expected_category="VERB",
                node_type="VerbPhraseNode",
                errors=errors,
            )
            if verb_sym is not None:
                action_meaning = verb_sym.meaning

            if vp.object is not None:
                obj_sym = self._resolve_and_verify(
                    lexeme=vp.object.value,
                    expected_category="NOUN",
                    node_type="ObjectNode",
                    errors=errors,
                )
                if obj_sym is not None:
                    object_meaning = obj_sym.meaning

        # 4. Optional Direct Object query validation (NOUN)
        if q_node.object is not None:
            obj_sym = self._resolve_and_verify(
                lexeme=q_node.object.value,
                expected_category="NOUN",
                node_type="ObjectNode",
                errors=errors,
            )
            if obj_sym is not None:
                object_meaning = obj_sym.meaning

        if errors:
            return None

        return QuestionSemanticForm(
            wh=wh_meaning,
            subject=subject_meaning,
            action=action_meaning,
            object=object_meaning,
        )

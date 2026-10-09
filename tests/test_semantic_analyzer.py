"""
SignSyntax - Semantic Analyzer Test Suite
Phase 2: Core Implementation (Semantic Analysis Testing)

Tests semantic validation, Symbol Table cross-checking, category conformance,
semantic normalization, and diagnostic error generation on Abstract Syntax Trees.
"""

import pytest

from compiler.ast import (
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
from compiler.lexer import Lexer
from compiler.parser import Parser
from compiler.semantic_analyzer import (
    QuestionSemanticForm,
    SemanticAnalyzer,
    SemanticError,
    SemanticForm,
    SemanticResult,
)


@pytest.fixture
def analyzer() -> SemanticAnalyzer:
    """Fixture providing a fresh SemanticAnalyzer instance."""
    return SemanticAnalyzer()


@pytest.fixture
def lexer() -> Lexer:
    """Fixture providing a fresh Lexer instance."""
    return Lexer()


# ==============================================================================
# 1. Valid Semantic Cases
# ==============================================================================

def test_semantic_svo_me_want_milk(lexer: Lexer, analyzer: SemanticAnalyzer) -> None:
    """Tests semantic analysis on standard SVO sentence: 'ME WANT MILK'."""
    tokens = lexer.tokenize("ME WANT MILK").tokens
    ast = Parser(tokens).parse().ast

    result = analyzer.analyze(ast)
    assert result.is_valid
    assert not result.has_errors
    assert isinstance(result.semantic_form, SemanticForm)

    form = result.semantic_form
    assert form.subject == "I"
    assert form.action == "want"
    assert form.object == "milk"
    assert form.time is None
    assert form.adjective is None
    assert form.negated is False


def test_semantic_time_yesterday_me_want_milk(lexer: Lexer, analyzer: SemanticAnalyzer) -> None:
    """Tests semantic analysis on temporal sentence: 'YESTERDAY ME WANT MILK'."""
    tokens = lexer.tokenize("YESTERDAY ME WANT MILK").tokens
    ast = Parser(tokens).parse().ast

    result = analyzer.analyze(ast)
    assert result.is_valid
    form = result.semantic_form
    assert isinstance(form, SemanticForm)
    assert form.time == "yesterday"
    assert form.subject == "I"
    assert form.action == "want"
    assert form.object == "milk"
    assert form.negated is False


def test_semantic_negation_me_want_milk_not(lexer: Lexer, analyzer: SemanticAnalyzer) -> None:
    """Tests semantic analysis on negated sentence: 'ME WANT MILK NOT'."""
    tokens = lexer.tokenize("ME WANT MILK NOT").tokens
    ast = Parser(tokens).parse().ast

    result = analyzer.analyze(ast)
    assert result.is_valid
    form = result.semantic_form
    assert isinstance(form, SemanticForm)
    assert form.subject == "I"
    assert form.action == "want"
    assert form.object == "milk"
    assert form.negated is True


def test_semantic_adjective_she_have_car_big(lexer: Lexer, analyzer: SemanticAnalyzer) -> None:
    """Tests semantic analysis on qualified object sentence: 'SHE HAVE CAR BIG'."""
    tokens = lexer.tokenize("SHE HAVE CAR BIG").tokens
    ast = Parser(tokens).parse().ast

    result = analyzer.analyze(ast)
    assert result.is_valid
    form = result.semantic_form
    assert isinstance(form, SemanticForm)
    assert form.subject == "she"
    assert form.action == "have"
    assert form.object == "car"
    assert form.adjective == "big"
    assert form.negated is False


def test_semantic_question_what_you_want(lexer: Lexer, analyzer: SemanticAnalyzer) -> None:
    """Tests semantic analysis on WH-question: 'WHAT YOU WANT'."""
    tokens = lexer.tokenize("WHAT YOU WANT").tokens
    ast = Parser(tokens).parse().ast

    result = analyzer.analyze(ast)
    assert result.is_valid
    form = result.semantic_form
    assert isinstance(form, QuestionSemanticForm)
    assert form.wh == "what"
    assert form.subject == "you"
    assert form.action == "want"
    assert form.object is None


def test_semantic_question_where_you_go(lexer: Lexer, analyzer: SemanticAnalyzer) -> None:
    """Tests semantic analysis on WH-question: 'WHERE YOU GO'."""
    tokens = lexer.tokenize("WHERE YOU GO").tokens
    ast = Parser(tokens).parse().ast

    result = analyzer.analyze(ast)
    assert result.is_valid
    form = result.semantic_form
    assert isinstance(form, QuestionSemanticForm)
    assert form.wh == "where"
    assert form.subject == "you"
    assert form.action == "go"


def test_semantic_question_what_book(lexer: Lexer, analyzer: SemanticAnalyzer) -> None:
    """Tests semantic analysis on concise object query: 'WHAT BOOK'."""
    tokens = lexer.tokenize("WHAT BOOK").tokens
    ast = Parser(tokens).parse().ast

    result = analyzer.analyze(ast)
    assert result.is_valid
    form = result.semantic_form
    assert isinstance(form, QuestionSemanticForm)
    assert form.wh == "what"
    assert form.subject is None
    assert form.action is None
    assert form.object == "book"


# ==============================================================================
# 2. Semantic Category Conformance & Validation (Negative Tests on Direct ASTs)
# ==============================================================================

def test_semantic_reject_non_pronoun_subject(analyzer: SemanticAnalyzer) -> None:
    """Rejects SubjectNode containing a non-PRONOUN symbol (e.g. 'MILK' which is NOUN)."""
    ast = SentenceNode(
        body=DeclarativeSentenceNode(
            subject=SubjectNode(value="MILK"),  # NOUN instead of PRONOUN
            verb_phrase=VerbPhraseNode(verb="WANT"),
        )
    )
    result = analyzer.analyze(ast)
    assert not result.is_valid
    assert len(result.errors) == 1
    err = result.errors[0]
    assert "symbol 'MILK' has category NOUN; expected PRONOUN" in err.message
    assert err.node_type == "SubjectNode"
    assert err.expected_category == "PRONOUN"
    assert err.found_category == "NOUN"


def test_semantic_reject_non_verb_in_verb_phrase(analyzer: SemanticAnalyzer) -> None:
    """Rejects VerbPhraseNode containing a non-VERB symbol (e.g. 'ME' which is PRONOUN)."""
    ast = SentenceNode(
        body=DeclarativeSentenceNode(
            subject=SubjectNode(value="YOU"),
            verb_phrase=VerbPhraseNode(verb="ME"),  # PRONOUN instead of VERB
        )
    )
    result = analyzer.analyze(ast)
    assert not result.is_valid
    err = result.errors[0]
    assert "symbol 'ME' has category PRONOUN; expected VERB" in err.message
    assert err.node_type == "VerbPhraseNode"
    assert err.expected_category == "VERB"
    assert err.found_category == "PRONOUN"


def test_semantic_reject_non_noun_object(analyzer: SemanticAnalyzer) -> None:
    """Rejects ObjectNode containing a non-NOUN symbol (e.g. 'WANT' which is VERB)."""
    ast = SentenceNode(
        body=DeclarativeSentenceNode(
            subject=SubjectNode(value="ME"),
            verb_phrase=VerbPhraseNode(
                verb="LIKE",
                object=ObjectNode(value="WANT"),  # VERB instead of NOUN
            ),
        )
    )
    result = analyzer.analyze(ast)
    assert not result.is_valid
    err = result.errors[0]
    assert "symbol 'WANT' has category VERB; expected NOUN" in err.message
    assert err.node_type == "ObjectNode"
    assert err.expected_category == "NOUN"
    assert err.found_category == "VERB"


def test_semantic_reject_non_adjective_modifier(analyzer: SemanticAnalyzer) -> None:
    """Rejects AdjectiveNode containing a non-ADJECTIVE symbol (e.g. 'BOOK' which is NOUN)."""
    ast = SentenceNode(
        body=DeclarativeSentenceNode(
            subject=SubjectNode(value="SHE"),
            verb_phrase=VerbPhraseNode(
                verb="HAVE",
                object=ObjectNode(value="CAR"),
                adjective=AdjectiveNode(value="BOOK"),  # NOUN instead of ADJECTIVE
            ),
        )
    )
    result = analyzer.analyze(ast)
    assert not result.is_valid
    err = result.errors[0]
    assert "symbol 'BOOK' has category NOUN; expected ADJECTIVE" in err.message
    assert err.node_type == "AdjectiveNode"
    assert err.expected_category == "ADJECTIVE"
    assert err.found_category == "NOUN"


def test_semantic_reject_non_time_marker(analyzer: SemanticAnalyzer) -> None:
    """Rejects TimeNode containing a non-TIME symbol (e.g. 'ME' which is PRONOUN)."""
    ast = SentenceNode(
        body=DeclarativeSentenceNode(
            time=TimeNode(value="ME"),  # PRONOUN instead of TIME
            subject=SubjectNode(value="YOU"),
            verb_phrase=VerbPhraseNode(verb="GO"),
        )
    )
    result = analyzer.analyze(ast)
    assert not result.is_valid
    err = result.errors[0]
    assert "symbol 'ME' has category PRONOUN; expected TIME" in err.message
    assert err.node_type == "TimeNode"
    assert err.expected_category == "TIME"


def test_semantic_reject_non_wh_word(analyzer: SemanticAnalyzer) -> None:
    """Rejects WHWordNode containing a non-WH_WORD symbol (e.g. 'YOU' which is PRONOUN)."""
    ast = SentenceNode(
        body=QuestionNode(
            wh_word=WHWordNode(value="YOU"),  # PRONOUN instead of WH_WORD
            subject=SubjectNode(value="HE"),
            verb_phrase=VerbPhraseNode(verb="SEE"),
        )
    )
    result = analyzer.analyze(ast)
    assert not result.is_valid
    err = result.errors[0]
    assert "symbol 'YOU' has category PRONOUN; expected WH_WORD" in err.message
    assert err.node_type == "WHWordNode"
    assert err.expected_category == "WH_WORD"


def test_semantic_reject_non_negation_symbol(analyzer: SemanticAnalyzer) -> None:
    """Rejects NegationNode containing a non-NEGATION symbol (e.g. 'BAD' which is ADJECTIVE)."""
    ast = SentenceNode(
        body=DeclarativeSentenceNode(
            subject=SubjectNode(value="ME"),
            verb_phrase=VerbPhraseNode(verb="WANT"),
            negation=NegationNode(value="BAD"),  # ADJECTIVE instead of NEGATION
        )
    )
    result = analyzer.analyze(ast)
    assert not result.is_valid
    err = result.errors[0]
    assert "symbol 'BAD' has category ADJECTIVE; expected NEGATION" in err.message
    assert err.node_type == "NegationNode"
    assert err.expected_category == "NEGATION"


def test_semantic_reject_unknown_symbol_not_in_symbol_table(analyzer: SemanticAnalyzer) -> None:
    """Rejects an AST node containing an unrecognized symbol not present in Symbol Table."""
    ast = SentenceNode(
        body=DeclarativeSentenceNode(
            subject=SubjectNode(value="UNKNOWN_SUBJECT"),
            verb_phrase=VerbPhraseNode(verb="WANT"),
        )
    )
    result = analyzer.analyze(ast)
    assert not result.is_valid
    err = result.errors[0]
    assert "Symbol 'UNKNOWN_SUBJECT' is not declared in the Symbol Table" in err.message


def test_semantic_analyzer_none_ast(analyzer: SemanticAnalyzer) -> None:
    """Rejects empty/None AST input gracefully."""
    result = analyzer.analyze(None)
    assert not result.is_valid
    assert "Cannot perform semantic analysis on empty or invalid AST" in result.errors[0].message


# ==============================================================================
# 3. Serialization & Display Formatting
# ==============================================================================

def test_semantic_form_to_dict_and_display() -> None:
    """Tests SemanticForm serialization to dict and human-readable display."""
    form = SemanticForm(
        time="yesterday",
        subject="I",
        action="want",
        object="milk",
        adjective="good",
        negated=True,
    )
    d = form.to_dict()
    assert d["form_type"] == "Declarative"
    assert d["time"] == "yesterday"
    assert d["subject"] == "I"
    assert d["action"] == "want"
    assert d["object"] == "milk"
    assert d["adjective"] == "good"
    assert d["negated"] is True

    disp = form.display()
    assert "Time: yesterday" in disp
    assert "Subject: I" in disp
    assert "Action: want" in disp
    assert "Object: milk" in disp
    assert "Adjective: good" in disp
    assert "Negated: true" in disp


def test_question_semantic_form_to_dict_and_display() -> None:
    """Tests QuestionSemanticForm serialization to dict and human-readable display."""
    q_form = QuestionSemanticForm(
        wh="what",
        subject="you",
        action="want",
        object=None,
    )
    d = q_form.to_dict()
    assert d["form_type"] == "Question"
    assert d["wh"] == "what"
    assert d["subject"] == "you"
    assert d["action"] == "want"
    assert d["object"] is None

    disp = q_form.display()
    assert "WH-Type: what" in disp
    assert "Subject: you" in disp
    assert "Action: want" in disp

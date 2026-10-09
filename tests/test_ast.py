"""
SignSyntax - Abstract Syntax Tree (AST) Test Suite
Phase 2: Core Implementation (AST Testing)

Tests all AST node types, ASTBuilder conversion from parse trees, optional field handling,
ASCII tree visualization, dictionary serialization, and semantic value extraction.
"""

import pytest

from compiler.ast import (
    ASTBuilder,
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
    build_ast,
)
from compiler.lexer import Lexer
from compiler.parser import Parser


@pytest.fixture
def lexer() -> Lexer:
    """Fixture providing a fresh Lexer instance."""
    return Lexer()


# ==============================================================================
# 1. Simple Verb / Intransitive Sentences
# ==============================================================================

def test_ast_minimal_declarative_me_go(lexer: Lexer) -> None:
    """Tests AST for intransitive statement: 'ME GO'."""
    lex_res = lexer.tokenize("ME GO")
    parse_res = Parser(lex_res.tokens).parse()
    assert parse_res.is_valid

    ast = parse_res.ast
    assert isinstance(ast, SentenceNode)
    assert ast.is_declarative
    assert not ast.is_question

    body = ast.body
    assert isinstance(body, DeclarativeSentenceNode)
    assert body.time is None
    assert body.negation is None
    assert isinstance(body.subject, SubjectNode)
    assert body.subject.value == "ME"
    assert body.subject.meaning == "I"

    assert isinstance(body.verb_phrase, VerbPhraseNode)
    assert body.verb_phrase.verb == "GO"
    assert body.verb_phrase.verb_meaning == "go"
    assert body.verb_phrase.object is None
    assert body.verb_phrase.adjective is None


# ==============================================================================
# 2. Subject + Verb + Object
# ==============================================================================

def test_ast_svo_me_want_milk(lexer: Lexer) -> None:
    """Tests AST for transitive SVO statement: 'ME WANT MILK'."""
    lex_res = lexer.tokenize("ME WANT MILK")
    parse_res = Parser(lex_res.tokens).parse()
    assert parse_res.is_valid

    ast = parse_res.ast
    assert isinstance(ast, SentenceNode)
    body = ast.body
    assert isinstance(body, DeclarativeSentenceNode)
    assert body.time is None
    assert body.negation is None
    assert body.subject.value == "ME"
    assert body.verb_phrase.verb == "WANT"
    assert body.verb_phrase.verb_meaning == "want"
    assert isinstance(body.verb_phrase.object, ObjectNode)
    assert body.verb_phrase.object.value == "MILK"
    assert body.verb_phrase.object.meaning == "milk"
    assert body.verb_phrase.adjective is None


# ==============================================================================
# 3. Time Marker + Subject + Verb + Object
# ==============================================================================

def test_ast_time_svo_yesterday_me_want_milk(lexer: Lexer) -> None:
    """Tests AST for temporal statement: 'YESTERDAY ME WANT MILK'."""
    lex_res = lexer.tokenize("YESTERDAY ME WANT MILK")
    parse_res = Parser(lex_res.tokens).parse()
    assert parse_res.is_valid

    ast = parse_res.ast
    assert isinstance(ast, SentenceNode)
    body = ast.body
    assert isinstance(body, DeclarativeSentenceNode)
    assert isinstance(body.time, TimeNode)
    assert body.time.value == "YESTERDAY"
    assert body.time.meaning == "yesterday"
    assert body.subject.value == "ME"
    assert body.verb_phrase.verb == "WANT"
    assert body.verb_phrase.object.value == "MILK"
    assert body.negation is None


# ==============================================================================
# 4. Clause-Final Negation
# ==============================================================================

def test_ast_negation_me_want_milk_not(lexer: Lexer) -> None:
    """Tests AST for negated statement: 'ME WANT MILK NOT'."""
    lex_res = lexer.tokenize("ME WANT MILK NOT")
    parse_res = Parser(lex_res.tokens).parse()
    assert parse_res.is_valid

    ast = parse_res.ast
    body = ast.body
    assert isinstance(body, DeclarativeSentenceNode)
    assert body.time is None
    assert body.subject.value == "ME"
    assert body.verb_phrase.verb == "WANT"
    assert body.verb_phrase.object.value == "MILK"
    assert isinstance(body.negation, NegationNode)
    assert body.negation.value == "NOT"


# ==============================================================================
# 5. Time Marker + SVO + Negation
# ==============================================================================

def test_ast_time_and_negation_yesterday_me_want_milk_not(lexer: Lexer) -> None:
    """Tests AST for statement with both time and negation: 'YESTERDAY ME WANT MILK NOT'."""
    lex_res = lexer.tokenize("YESTERDAY ME WANT MILK NOT")
    parse_res = Parser(lex_res.tokens).parse()
    assert parse_res.is_valid

    ast = parse_res.ast
    body = ast.body
    assert isinstance(body, DeclarativeSentenceNode)
    assert isinstance(body.time, TimeNode)
    assert body.time.value == "YESTERDAY"
    assert body.subject.value == "ME"
    assert body.verb_phrase.verb == "WANT"
    assert body.verb_phrase.object.value == "MILK"
    assert isinstance(body.negation, NegationNode)
    assert body.negation.value == "NOT"


# ==============================================================================
# 6. WH-Question with Subject and Predicate
# ==============================================================================

def test_ast_question_what_you_want(lexer: Lexer) -> None:
    """Tests AST for WH-question: 'WHAT YOU WANT'."""
    lex_res = lexer.tokenize("WHAT YOU WANT")
    parse_res = Parser(lex_res.tokens).parse()
    assert parse_res.is_valid

    ast = parse_res.ast
    assert isinstance(ast, SentenceNode)
    assert ast.is_question
    assert not ast.is_declarative

    body = ast.body
    assert isinstance(body, QuestionNode)
    assert isinstance(body.wh_word, WHWordNode)
    assert body.wh_word.value == "WHAT"
    assert body.wh_word.meaning == "what"
    assert isinstance(body.subject, SubjectNode)
    assert body.subject.value == "YOU"
    assert isinstance(body.verb_phrase, VerbPhraseNode)
    assert body.verb_phrase.verb == "WANT"
    assert body.object is None


# ==============================================================================
# 7. WH-Question with Object
# ==============================================================================

def test_ast_question_what_book(lexer: Lexer) -> None:
    """Tests AST for concise WH + object question: 'WHAT BOOK'."""
    lex_res = lexer.tokenize("WHAT BOOK")
    parse_res = Parser(lex_res.tokens).parse()
    assert parse_res.is_valid

    ast = parse_res.ast
    body = ast.body
    assert isinstance(body, QuestionNode)
    assert body.wh_word.value == "WHAT"
    assert body.subject is None
    assert body.verb_phrase is None
    assert isinstance(body.object, ObjectNode)
    assert body.object.value == "BOOK"
    assert body.object.meaning == "book"


# ==============================================================================
# 8. Verb + Object + Adjective Qualification
# ==============================================================================

def test_ast_verb_object_adjective_she_have_car_big(lexer: Lexer) -> None:
    """Tests AST with post-nominal adjective qualification: 'SHE HAVE CAR BIG'."""
    lex_res = lexer.tokenize("SHE HAVE CAR BIG")
    parse_res = Parser(lex_res.tokens).parse()
    assert parse_res.is_valid

    ast = parse_res.ast
    body = ast.body
    assert isinstance(body, DeclarativeSentenceNode)
    assert body.subject.value == "SHE"
    vp = body.verb_phrase
    assert vp.verb == "HAVE"
    assert isinstance(vp.object, ObjectNode)
    assert vp.object.value == "CAR"
    assert isinstance(vp.adjective, AdjectiveNode)
    assert vp.adjective.value == "BIG"
    assert vp.adjective.meaning == "big"


# ==============================================================================
# 9. ASCII Visualization Rendering
# ==============================================================================

def test_ast_ascii_visualization_rendering(lexer: Lexer) -> None:
    """Tests pretty_print() / to_ascii() rendering with tree branch characters."""
    lex_res = lexer.tokenize("YESTERDAY ME WANT MILK")
    parse_res = Parser(lex_res.tokens).parse()
    ast = parse_res.ast

    ascii_str = ast.to_ascii()
    assert "Sentence" in ascii_str
    assert "Time: YESTERDAY" in ascii_str
    assert "Subject: ME" in ascii_str
    assert "VerbPhrase" in ascii_str
    assert "Verb: WANT" in ascii_str
    assert "Object: MILK" in ascii_str

    # Test str(ast) and pretty_print()
    assert str(ast) == ascii_str
    assert ast.pretty_print() == ascii_str


def test_ast_ascii_question_visualization(lexer: Lexer) -> None:
    """Tests to_ascii() on question AST: 'WHAT YOU WANT'."""
    lex_res = lexer.tokenize("WHAT YOU WANT")
    parse_res = Parser(lex_res.tokens).parse()
    ast = parse_res.ast

    ascii_str = ast.to_ascii()
    assert "Question" in ascii_str
    assert "WHWord: WHAT" in ascii_str
    assert "Subject: YOU" in ascii_str
    assert "VerbPhrase" in ascii_str
    assert "Verb: WANT" in ascii_str


# ==============================================================================
# 10. Dictionary Serialization
# ==============================================================================

def test_ast_dictionary_serialization(lexer: Lexer) -> None:
    """Tests to_dict() nested JSON-like serialization."""
    lex_res = lexer.tokenize("YESTERDAY ME WANT MILK NOT")
    parse_res = Parser(lex_res.tokens).parse()
    ast = parse_res.ast

    d = ast.to_dict()
    assert d["node_type"] == "Sentence"
    body = d["body"]
    assert body["node_type"] == "DeclarativeSentence"
    assert body["time"]["node_type"] == "Time"
    assert body["time"]["value"] == "YESTERDAY"
    assert body["subject"]["node_type"] == "Subject"
    assert body["subject"]["value"] == "ME"
    assert body["verb_phrase"]["node_type"] == "VerbPhrase"
    assert body["verb_phrase"]["verb"] == "WANT"
    assert body["verb_phrase"]["object"]["value"] == "MILK"
    assert body["negation"]["node_type"] == "Negation"
    assert body["negation"]["value"] == "NOT"


# ==============================================================================
# 11. Error Behavior / Non-Construction on Invalid Parse
# ==============================================================================

def test_ast_not_created_for_invalid_syntax(lexer: Lexer) -> None:
    """Tests that parse_res.ast is None when syntax error occurs."""
    lex_res = lexer.tokenize("ME MILK WANT")
    parse_res = Parser(lex_res.tokens).parse()
    assert not parse_res.is_valid
    assert parse_res.ast is None


def test_ast_builder_none_input() -> None:
    """Tests that ASTBuilder.build(None) safely returns None."""
    assert ASTBuilder.build(None) is None
    assert build_ast(None) is None

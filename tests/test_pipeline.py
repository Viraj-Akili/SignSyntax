"""
SignSyntax - End-to-End Syntax Pipeline & Error Boundary Test Suite
Phase 2: Core Implementation (Pipeline Hardening & Error Boundaries)

Verifies strict compiler phase separation across:
  [1] Lexical Analysis (Symbol Table validation, case-sensitivity, unknown glosses)
  [2] Syntax Analysis (Grammar productions, LL(1) lookahead, complete stream consumption)
  [3] AST Construction (AST created strictly for valid parses; None for all invalid parses)
"""

import pytest

from compiler.ast import (
    DeclarativeSentenceNode,
    QuestionNode,
    SentenceNode,
    TimeNode,
    VerbPhraseNode,
)
from compiler.lexer import Lexer, LexerResult
from compiler.parser import Parser, ParseResult


@pytest.fixture
def lexer() -> Lexer:
    """Fixture providing a fresh Lexer instance."""
    return Lexer()


# ==============================================================================
# 1. Error Boundary Verification (Task 2)
# ==============================================================================

def test_pipeline_case_1_lexical_error_unknown_gloss(lexer: Lexer) -> None:
    """
    Case 1: 'ME DANCE MILK'
    DANCE is not in Symbol Table vocabulary -> Lexical Error.
    The Parser must NOT receive an invalid token stream.
    """
    lex_res = lexer.tokenize("ME DANCE MILK")
    assert lex_res.has_errors
    assert not lex_res.is_valid
    assert len(lex_res.errors) == 1
    assert lex_res.errors[0].position == 1
    assert "Unknown gloss 'DANCE'" in str(lex_res.errors[0])

    # Pipeline rule: when lexical errors occur, parser is not run or returns empty/invalid
    # If parser is called on partial valid tokens, it fails or does not produce valid AST
    parser = Parser(lex_res.tokens)
    parse_res = parser.parse()
    # 'ME MILK' is invalid syntax (missing verb)
    assert not parse_res.is_valid
    assert parse_res.ast is None


def test_pipeline_case_2_syntax_error_invalid_token_order(lexer: Lexer) -> None:
    """
    Case 2: 'ME MILK WANT'
    All tokens are valid in vocabulary (Lexer OK), but NOUN appears where VERB is expected.
    Lexical Analysis -> SUCCESS
    Syntax Analysis -> FAILURE at position 1
    AST -> None
    """
    lex_res = lexer.tokenize("ME MILK WANT")
    assert lex_res.is_valid
    assert not lex_res.has_errors
    assert len(lex_res.tokens) == 3

    parser = Parser(lex_res.tokens)
    parse_res = parser.parse()
    assert not parse_res.is_valid
    assert parse_res.has_errors
    assert len(parse_res.errors) == 1
    err = parse_res.errors[0]
    assert err.position == 1
    assert "Expected VERB predicate" in err.message
    assert "found NOUN 'MILK'" in err.message
    assert parse_res.ast is None


def test_pipeline_case_3_syntax_error_incomplete_subject_only(lexer: Lexer) -> None:
    """
    Case 3: 'ME'
    Subject pronoun exists (Lexer OK), but mandatory VerbPhrase predicate is missing.
    Lexical Analysis -> SUCCESS
    Syntax Analysis -> FAILURE at position 1 (unexpected EOF)
    AST -> None
    """
    lex_res = lexer.tokenize("ME")
    assert lex_res.is_valid

    parser = Parser(lex_res.tokens)
    parse_res = parser.parse()
    assert not parse_res.is_valid
    assert parse_res.errors[0].position == 1
    assert "Unexpected end of input; expected VERB predicate" in parse_res.errors[0].message
    assert parse_res.ast is None


def test_pipeline_case_4_syntax_error_trailing_unconsumed_token(lexer: Lexer) -> None:
    """
    Case 4: 'ME WANT MILK BOOK'
    All 4 tokens are valid in vocabulary (Lexer OK).
    'ME WANT MILK' forms a valid complete sentence, but 'BOOK' remains unconsumed.
    Lexical Analysis -> SUCCESS
    Syntax Analysis -> FAILURE at position 3
    AST -> None
    """
    lex_res = lexer.tokenize("ME WANT MILK BOOK")
    assert lex_res.is_valid
    assert len(lex_res.tokens) == 4

    parser = Parser(lex_res.tokens)
    parse_res = parser.parse()
    assert not parse_res.is_valid
    assert parse_res.errors[0].position == 3
    assert "Unexpected trailing token 'BOOK' after complete sentence" in parse_res.errors[0].message
    assert parse_res.ast is None


def test_pipeline_case_5_full_success_valid_sentence(lexer: Lexer) -> None:
    """
    Case 5: 'ME WANT MILK'
    Lexical Analysis -> SUCCESS
    Syntax Analysis -> SUCCESS
    AST -> Valid SentenceNode created
    """
    lex_res = lexer.tokenize("ME WANT MILK")
    assert lex_res.is_valid
    assert len(lex_res.tokens) == 3

    parser = Parser(lex_res.tokens)
    parse_res = parser.parse()
    assert parse_res.is_valid
    assert not parse_res.has_errors
    assert parse_res.ast is not None
    assert isinstance(parse_res.ast, SentenceNode)
    assert parse_res.ast.is_declarative
    assert parse_res.ast.body.subject.value == "ME"
    assert parse_res.ast.body.verb_phrase.verb == "WANT"
    assert parse_res.ast.body.verb_phrase.object.value == "MILK"


# ==============================================================================
# 2. Strict AST Invariant Verification (Task 3)
# ==============================================================================

@pytest.mark.parametrize("invalid_source", [
    "ME MILK WANT",
    "BOOK YOU READ",
    "GO WE SCHOOL",
    "ME",
    "YESTERDAY",
    "WANT",
    "WHAT",
    "WHAT YESTERDAY",
    "ME WANT MILK BOOK",
    "ME WANT HAPPY",
    "YESTERDAY ME",
])
def test_ast_never_created_for_invalid_syntax(lexer: Lexer, invalid_source: str) -> None:
    """Guarantees that AST is strictly None for any syntax error."""
    lex_res = lexer.tokenize(invalid_source)
    assert lex_res.is_valid, f"Expected valid lexing for '{invalid_source}'"
    parser = Parser(lex_res.tokens)
    parse_res = parser.parse()
    assert not parse_res.is_valid
    assert parse_res.ast is None


# ==============================================================================
# 3. Comprehensive Edge Cases (Task 5)
# ==============================================================================

def test_edge_case_empty_input(lexer: Lexer) -> None:
    """Edge Case 1: Empty input string."""
    lex_res = lexer.tokenize("")
    assert lex_res.is_valid
    assert len(lex_res.tokens) == 0

    parse_res = Parser(lex_res.tokens).parse()
    assert not parse_res.is_valid
    assert "Empty input sequence" in parse_res.errors[0].message
    assert parse_res.ast is None


def test_edge_case_whitespace_only(lexer: Lexer) -> None:
    """Edge Case 2: Whitespace-only string."""
    lex_res = lexer.tokenize("   \t  \n  ")
    assert lex_res.is_valid
    assert len(lex_res.tokens) == 0

    parse_res = Parser(lex_res.tokens).parse()
    assert not parse_res.is_valid
    assert "Empty input sequence" in parse_res.errors[0].message
    assert parse_res.ast is None


def test_edge_case_subject_only_me(lexer: Lexer) -> None:
    """Edge Case 3: Subject only 'ME'."""
    lex_res = lexer.tokenize("ME")
    parse_res = Parser(lex_res.tokens).parse()
    assert not parse_res.is_valid
    assert parse_res.errors[0].position == 1
    assert parse_res.ast is None


def test_edge_case_time_only_yesterday(lexer: Lexer) -> None:
    """Edge Case 4: Time only 'YESTERDAY'."""
    lex_res = lexer.tokenize("YESTERDAY")
    parse_res = Parser(lex_res.tokens).parse()
    assert not parse_res.is_valid
    assert parse_res.errors[0].position == 1
    assert parse_res.ast is None


def test_edge_case_verb_only_want(lexer: Lexer) -> None:
    """Edge Case 5: Verb only 'WANT'."""
    lex_res = lexer.tokenize("WANT")
    parse_res = Parser(lex_res.tokens).parse()
    assert not parse_res.is_valid
    assert parse_res.errors[0].position == 0
    assert parse_res.ast is None


def test_edge_case_wh_word_only_what(lexer: Lexer) -> None:
    """Edge Case 6: WH-word only 'WHAT'."""
    lex_res = lexer.tokenize("WHAT")
    parse_res = Parser(lex_res.tokens).parse()
    assert not parse_res.is_valid
    assert parse_res.errors[0].position == 1
    assert parse_res.ast is None


def test_edge_case_valid_minimal_sentence_me_go(lexer: Lexer) -> None:
    """Edge Case 7: Valid minimal sentence 'ME GO'."""
    lex_res = lexer.tokenize("ME GO")
    parse_res = Parser(lex_res.tokens).parse()
    assert parse_res.is_valid
    assert parse_res.ast is not None
    assert parse_res.ast.body.subject.value == "ME"
    assert parse_res.ast.body.verb_phrase.verb == "GO"
    assert parse_res.ast.body.verb_phrase.object is None


def test_edge_case_valid_question_what_you_want(lexer: Lexer) -> None:
    """Edge Case 8: Valid question 'WHAT YOU WANT'."""
    lex_res = lexer.tokenize("WHAT YOU WANT")
    parse_res = Parser(lex_res.tokens).parse()
    assert parse_res.is_valid
    assert parse_res.ast is not None
    assert isinstance(parse_res.ast.body, QuestionNode)
    assert parse_res.ast.body.wh_word.value == "WHAT"
    assert parse_res.ast.body.subject.value == "YOU"
    assert parse_res.ast.body.verb_phrase.verb == "WANT"


def test_edge_case_valid_negation_me_want_milk_not(lexer: Lexer) -> None:
    """Edge Case 9: Valid negation 'ME WANT MILK NOT'."""
    lex_res = lexer.tokenize("ME WANT MILK NOT")
    parse_res = Parser(lex_res.tokens).parse()
    assert parse_res.is_valid
    assert parse_res.ast is not None
    assert parse_res.ast.body.negation is not None
    assert parse_res.ast.body.negation.value == "NOT"


def test_edge_case_valid_time_and_negation_yesterday_me_want_milk_not(lexer: Lexer) -> None:
    """Edge Case 10: Valid time + negation 'YESTERDAY ME WANT MILK NOT'."""
    lex_res = lexer.tokenize("YESTERDAY ME WANT MILK NOT")
    parse_res = Parser(lex_res.tokens).parse()
    assert parse_res.is_valid
    assert parse_res.ast is not None
    assert parse_res.ast.body.time is not None
    assert parse_res.ast.body.time.value == "YESTERDAY"
    assert parse_res.ast.body.negation is not None
    assert parse_res.ast.body.negation.value == "NOT"


def test_edge_case_valid_verb_object_adjective_she_have_car_big(lexer: Lexer) -> None:
    """Edge Case 11: Valid post-nominal adjective qualification 'SHE HAVE CAR BIG'."""
    lex_res = lexer.tokenize("SHE HAVE CAR BIG")
    parse_res = Parser(lex_res.tokens).parse()
    assert parse_res.is_valid
    assert parse_res.ast is not None
    vp = parse_res.ast.body.verb_phrase
    assert vp.verb == "HAVE"
    assert vp.object.value == "CAR"
    assert vp.adjective.value == "BIG"


def test_edge_case_valid_sentence_followed_by_extra_token(lexer: Lexer) -> None:
    """Edge Case 12: Valid sentence followed by extra token 'ME WANT MILK BOOK'."""
    lex_res = lexer.tokenize("ME WANT MILK BOOK")
    parse_res = Parser(lex_res.tokens).parse()
    assert not parse_res.is_valid
    assert parse_res.errors[0].position == 3
    assert parse_res.ast is None


def test_edge_case_unknown_lexical_token_me_dance_milk(lexer: Lexer) -> None:
    """Edge Case 13: Unknown lexical token 'ME DANCE MILK'."""
    lex_res = lexer.tokenize("ME DANCE MILK")
    assert lex_res.has_errors
    assert not lex_res.is_valid
    assert "DANCE" in lex_res.errors[0].invalid_lexeme

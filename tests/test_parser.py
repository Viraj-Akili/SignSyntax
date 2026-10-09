"""
SignSyntax - Syntax Analyzer (Parser) Test Suite
Phase 2: Core Implementation (Parser Testing)

Tests all grammar productions, LL(1) lookahead branches, valid sentence forms,
syntax error detection, incomplete inputs, and trailing token rejections.
"""

import pytest

from compiler.lexer import Lexer
from compiler.parser import ParseError, Parser, ParseResult, ParseTreeNode
from compiler.tokens import Token


@pytest.fixture
def lexer() -> Lexer:
    """Fixture providing a fresh Lexer instance."""
    return Lexer()


# ==============================================================================
# Category A: Valid Declarative Sentences
# ==============================================================================

def test_parse_minimal_declarative_verb_only(lexer: Lexer) -> None:
    """Tests minimal declarative sentence: <Subject> <Verb> (e.g., 'ME GO')."""
    lex_res = lexer.tokenize("ME GO")
    assert lex_res.is_valid

    parser = Parser(lex_res.tokens)
    result = parser.parse()

    assert result.is_valid
    assert result.tree is not None
    assert result.tree.rule == "<Sentence>"
    decl_node = result.tree.children[0]
    assert isinstance(decl_node, ParseTreeNode)
    assert decl_node.rule == "<DeclarativeSentence>"
    assert len(decl_node.children) == 2  # Subject, VerbPhrase
    assert decl_node.children[0].rule == "<Subject>"
    assert decl_node.children[1].rule == "<VerbPhrase>"


def test_parse_standard_svo_sentence(lexer: Lexer) -> None:
    """Tests standard SVO sentence: <Subject> <Verb> <Object> (e.g., 'ME WANT MILK')."""
    lex_res = lexer.tokenize("ME WANT MILK")
    assert lex_res.is_valid

    parser = Parser(lex_res.tokens)
    result = parser.parse()

    assert result.is_valid
    assert not result.has_errors
    decl = result.tree.children[0]
    assert len(decl.children) == 2  # Subject, VerbPhrase
    vp = decl.children[1]
    assert len(vp.children) == 2  # Verb, Object


def test_parse_declarative_with_time_marker_past(lexer: Lexer) -> None:
    """Tests sentence with leading past time marker: 'YESTERDAY ME WANT MILK'."""
    lex_res = lexer.tokenize("YESTERDAY ME WANT MILK")
    assert lex_res.is_valid

    parser = Parser(lex_res.tokens)
    result = parser.parse()

    assert result.is_valid
    decl = result.tree.children[0]
    assert len(decl.children) == 3  # TimeMarker, Subject, VerbPhrase
    assert decl.children[0].rule == "<TimeMarker>"
    assert decl.children[0].children[0].lexeme == "YESTERDAY"
    assert decl.children[1].rule == "<Subject>"
    assert decl.children[2].rule == "<VerbPhrase>"


def test_parse_declarative_with_time_marker_present_and_future(lexer: Lexer) -> None:
    """Tests sentences with other time markers: TODAY, TOMORROW, NOW, MORNING, NIGHT."""
    for time_gloss in ["TODAY", "TOMORROW", "NOW", "MORNING", "NIGHT"]:
        lex_res = lexer.tokenize(f"{time_gloss} YOU LIKE BOOK")
        assert lex_res.is_valid
        parser = Parser(lex_res.tokens)
        result = parser.parse()
        assert result.is_valid, f"Failed for time marker: {time_gloss}"


def test_parse_all_pronoun_subjects(lexer: Lexer) -> None:
    """Tests that all 6 formal pronouns parse correctly as subjects."""
    for pronoun in ["ME", "YOU", "HE", "SHE", "WE", "THEY"]:
        lex_res = lexer.tokenize(f"{pronoun} READ BOOK")
        assert lex_res.is_valid
        parser = Parser(lex_res.tokens)
        result = parser.parse()
        assert result.is_valid, f"Failed for pronoun: {pronoun}"


# ==============================================================================
# Category B: Valid Verb + Object Constructions
# ==============================================================================

def test_parse_transitive_verb_object(lexer: Lexer) -> None:
    """Tests transitive verb-object pairs: 'HE DRINK WATER', 'YOU READ BOOK'."""
    test_cases = [
        ("HE DRINK WATER", "DRINK", "WATER"),
        ("YOU READ BOOK", "READ", "BOOK"),
        ("THEY EAT FOOD", "EAT", "FOOD"),
        ("SHE SEE FRIEND", "SEE", "FRIEND"),
        ("WE HAVE MONEY", "HAVE", "MONEY"),
    ]
    for sentence, verb, obj in test_cases:
        lex_res = lexer.tokenize(sentence)
        assert lex_res.is_valid
        parser = Parser(lex_res.tokens)
        result = parser.parse()
        assert result.is_valid, f"Failed parsing: {sentence}"
        vp = result.tree.children[0].children[1]
        assert vp.children[0].children[0].lexeme == verb
        assert vp.children[1].children[0].lexeme == obj


# ==============================================================================
# Category C: Valid Verb + Object + Adjective Constructions
# ==============================================================================

def test_parse_verb_object_adjective_qualification(lexer: Lexer) -> None:
    """Tests post-nominal adjective qualification: 'SHE HAVE CAR BIG', 'ME WANT FOOD GOOD'."""
    test_cases = [
        ("SHE HAVE CAR BIG", "CAR", "BIG"),
        ("ME WANT FOOD GOOD", "FOOD", "GOOD"),
        ("THEY LIKE HOME SMALL", "HOME", "SMALL"),
        ("HE SEE FRIEND HAPPY", "FRIEND", "HAPPY"),
        ("YOU HAVE PHONE BAD", "PHONE", "BAD"),
    ]
    for sentence, obj, adj in test_cases:
        lex_res = lexer.tokenize(sentence)
        assert lex_res.is_valid
        parser = Parser(lex_res.tokens)
        result = parser.parse()
        assert result.is_valid, f"Failed parsing: {sentence}"
        vp = result.tree.children[0].children[1]
        assert len(vp.children) == 3  # Verb, Object, Adjective
        assert vp.children[1].children[0].lexeme == obj
        assert vp.children[2].children[0].lexeme == adj


# ==============================================================================
# Category D: Valid Negation
# ==============================================================================

def test_parse_clause_final_negation(lexer: Lexer) -> None:
    """Tests clause-final negation: 'ME WANT MILK NOT', 'YOU LIKE BOOK NOT'."""
    test_cases = [
        "ME WANT MILK NOT",
        "YOU LIKE BOOK NOT",
        "YESTERDAY SHE GO SCHOOL NOT",
        "NOW WE DRINK WATER NOT",
    ]
    for sentence in test_cases:
        lex_res = lexer.tokenize(sentence)
        assert lex_res.is_valid
        parser = Parser(lex_res.tokens)
        result = parser.parse()
        assert result.is_valid, f"Failed for negated sentence: {sentence}"
        decl = result.tree.children[0]
        neg_node = decl.children[-1]
        assert neg_node.rule == "<Negation>"
        assert neg_node.children[0].lexeme == "NOT"


# ==============================================================================
# Category E: Valid Questions (WH + Subject + VerbPhrase)
# ==============================================================================

def test_parse_wh_questions_with_subject(lexer: Lexer) -> None:
    """Tests WH-questions with subject and predicate: 'WHAT YOU WANT', 'WHERE YOU GO', 'WHO SHE SEE'."""
    test_cases = [
        ("WHAT YOU WANT", "WHAT", "YOU", "WANT"),
        ("WHERE YOU GO", "WHERE", "YOU", "GO"),
        ("WHO SHE SEE", "WHO", "SHE", "SEE"),
        ("WHEN THEY COME", "WHEN", "THEY", "COME"),
        ("WHY HE NEED MONEY", "WHY", "HE", "NEED"),
    ]
    for sentence, wh_word, subject, verb in test_cases:
        lex_res = lexer.tokenize(sentence)
        assert lex_res.is_valid
        parser = Parser(lex_res.tokens)
        result = parser.parse()
        assert result.is_valid, f"Failed for question: {sentence}"
        q_node = result.tree.children[0]
        assert q_node.rule == "<Question>"
        wh_q = q_node.children[0]
        assert wh_q.rule == "<WHQuestion>"
        assert wh_q.children[0].children[0].lexeme == wh_word
        assert wh_q.children[1].children[0].children[0].lexeme == subject


# ==============================================================================
# Category F: Valid WH + Object Questions
# ==============================================================================

def test_parse_wh_questions_with_object(lexer: Lexer) -> None:
    """Tests concise WH + Object questions: 'WHAT BOOK', 'WHERE HOME'."""
    test_cases = [
        ("WHAT BOOK", "WHAT", "BOOK"),
        ("WHERE HOME", "WHERE", "HOME"),
        ("WHO FRIEND", "WHO", "FRIEND"),
        ("WHERE SCHOOL", "WHERE", "SCHOOL"),
        ("WHAT PHONE", "WHAT", "PHONE"),
    ]
    for sentence, wh_word, obj in test_cases:
        lex_res = lexer.tokenize(sentence)
        assert lex_res.is_valid
        parser = Parser(lex_res.tokens)
        result = parser.parse()
        assert result.is_valid, f"Failed for WH+Object: {sentence}"
        wh_q = result.tree.children[0].children[0]
        assert len(wh_q.children) == 2  # WHWord, Object
        assert wh_q.children[0].children[0].lexeme == wh_word
        assert wh_q.children[1].children[0].lexeme == obj


# ==============================================================================
# Category G: Invalid Syntax (Grammar Order Violations)
# ==============================================================================

def test_reject_sov_word_order_me_milk_want(lexer: Lexer) -> None:
    """Tests rejection of SOV order: 'ME MILK WANT' (expected VERB, found NOUN)."""
    lex_res = lexer.tokenize("ME MILK WANT")
    assert lex_res.is_valid
    parser = Parser(lex_res.tokens)
    result = parser.parse()

    assert not result.is_valid
    assert result.has_errors
    assert len(result.errors) == 1
    err = result.errors[0]
    assert err.position == 1
    assert "Expected VERB" in err.message
    assert "found NOUN 'MILK'" in err.message


def test_reject_osv_word_order_book_you_read(lexer: Lexer) -> None:
    """Tests rejection of object-initial declarative sentence: 'BOOK YOU READ'."""
    lex_res = lexer.tokenize("BOOK YOU READ")
    assert lex_res.is_valid
    parser = Parser(lex_res.tokens)
    result = parser.parse()

    assert not result.is_valid
    assert result.errors[0].position == 0
    assert "Expected sentence starting with TIME, PRONOUN, or WH_WORD" in result.errors[0].message


def test_reject_verb_initial_sentence_go_we_school(lexer: Lexer) -> None:
    """Tests rejection of verb-initial sentence: 'GO WE SCHOOL'."""
    lex_res = lexer.tokenize("GO WE SCHOOL")
    assert lex_res.is_valid
    parser = Parser(lex_res.tokens)
    result = parser.parse()

    assert not result.is_valid
    assert result.errors[0].position == 0
    assert "found VERB 'GO'" in result.errors[0].message


def test_reject_adjective_directly_after_verb_without_object(lexer: Lexer) -> None:
    """Tests rejection of adjective directly after verb: 'ME WANT BIG' (requires NOUN first)."""
    lex_res = lexer.tokenize("ME WANT BIG")
    assert lex_res.is_valid
    parser = Parser(lex_res.tokens)
    result = parser.parse()

    assert not result.is_valid
    assert "Adjective 'BIG' cannot directly follow verb 'WANT'" in result.errors[0].message


# ==============================================================================
# Category H: Incomplete Input
# ==============================================================================

def test_reject_incomplete_subject_only_me(lexer: Lexer) -> None:
    """Tests rejection of subject only: 'ME'."""
    lex_res = lexer.tokenize("ME")
    assert lex_res.is_valid
    parser = Parser(lex_res.tokens)
    result = parser.parse()

    assert not result.is_valid
    assert "Unexpected end of input; expected VERB" in result.errors[0].message


def test_reject_incomplete_time_only_yesterday(lexer: Lexer) -> None:
    """Tests rejection of time marker only: 'YESTERDAY'."""
    lex_res = lexer.tokenize("YESTERDAY")
    assert lex_res.is_valid
    parser = Parser(lex_res.tokens)
    result = parser.parse()

    assert not result.is_valid
    assert "Unexpected end of input; expected PRONOUN" in result.errors[0].message


def test_reject_incomplete_time_and_subject_yesterday_me(lexer: Lexer) -> None:
    """Tests rejection of time and subject without predicate: 'YESTERDAY ME'."""
    lex_res = lexer.tokenize("YESTERDAY ME")
    assert lex_res.is_valid
    parser = Parser(lex_res.tokens)
    result = parser.parse()

    assert not result.is_valid
    assert "Unexpected end of input; expected VERB" in result.errors[0].message


def test_reject_incomplete_verb_only_want(lexer: Lexer) -> None:
    """Tests rejection of bare verb without subject: 'WANT'."""
    lex_res = lexer.tokenize("WANT")
    assert lex_res.is_valid
    parser = Parser(lex_res.tokens)
    result = parser.parse()

    assert not result.is_valid
    assert result.errors[0].position == 0
    assert "found VERB 'WANT'" in result.errors[0].message


# ==============================================================================
# Category I: Invalid Question Structures
# ==============================================================================

def test_reject_incomplete_wh_word_only_what(lexer: Lexer) -> None:
    """Tests rejection of bare WH-word: 'WHAT'."""
    lex_res = lexer.tokenize("WHAT")
    assert lex_res.is_valid
    parser = Parser(lex_res.tokens)
    result = parser.parse()

    assert not result.is_valid
    assert "Incomplete WH-question after 'WHAT'" in result.errors[0].message


def test_reject_invalid_token_following_wh_word_what_yesterday(lexer: Lexer) -> None:
    """Tests rejection of invalid follower after WH-word: 'WHAT YESTERDAY'."""
    lex_res = lexer.tokenize("WHAT YESTERDAY")
    assert lex_res.is_valid
    parser = Parser(lex_res.tokens)
    result = parser.parse()

    assert not result.is_valid
    assert "Expected PRONOUN subject or NOUN object after 'WHAT'" in result.errors[0].message


def test_reject_wh_word_with_adjective_where_happy(lexer: Lexer) -> None:
    """Tests rejection of WH-word followed by adjective: 'WHERE HAPPY'."""
    lex_res = lexer.tokenize("WHERE HAPPY")
    assert lex_res.is_valid
    parser = Parser(lex_res.tokens)
    result = parser.parse()

    assert not result.is_valid
    assert "found ADJECTIVE 'HAPPY'" in result.errors[0].message


# ==============================================================================
# Category J: Trailing / Unconsumed Token Detection
# ==============================================================================

def test_reject_trailing_valid_token_me_want_milk_book(lexer: Lexer) -> None:
    """Tests rejection when a valid sentence has trailing valid tokens: 'ME WANT MILK BOOK'."""
    lex_res = lexer.tokenize("ME WANT MILK BOOK")
    assert lex_res.is_valid
    parser = Parser(lex_res.tokens)
    result = parser.parse()

    assert not result.is_valid
    assert len(result.errors) == 1
    err = result.errors[0]
    assert err.position == 3
    assert "Unexpected trailing token 'BOOK' after complete sentence" in err.message


def test_empty_token_stream_handling() -> None:
    """Tests that parsing an empty token list produces an informative syntax error."""
    parser = Parser([])
    result = parser.parse()

    assert not result.is_valid
    assert len(result.errors) == 1
    assert "Empty input sequence" in result.errors[0].message


# ==============================================================================
# Category K: ParseTree Serialization and Display
# ==============================================================================

def test_parse_tree_to_dict_and_display(lexer: Lexer) -> None:
    """Tests serialization of ParseTreeNode to dict and ASCII display string."""
    lex_res = lexer.tokenize("YESTERDAY ME WANT MILK")
    assert lex_res.is_valid
    parser = Parser(lex_res.tokens)
    result = parser.parse()

    assert result.is_valid
    tree_dict = result.tree.to_dict()
    assert tree_dict["rule"] == "<Sentence>"
    assert len(tree_dict["children"]) == 1

    display_str = result.tree.display()
    assert "<Sentence>" in display_str
    assert "<DeclarativeSentence>" in display_str
    assert "<TimeMarker>" in display_str
    assert "<Subject>" in display_str
    assert "<VerbPhrase>" in display_str
    assert "YESTERDAY" in display_str
    assert "MILK" in display_str


def test_parse_error_string_formatting() -> None:
    """Tests __str__ representation of ParseError."""
    err = ParseError(
        message="Expected VERB predicate, found NOUN 'MILK'",
        position=2,
        expected=["VERB"],
    )
    assert str(err) == "SYNTAX ERROR at position 2: Expected VERB predicate, found NOUN 'MILK'"

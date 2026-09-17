"""
Tests for SignSyntax Lexical Analyzer (Lexer)
Phase 1: Compiler Design - Lexical Analysis Verification

Verifies that the hand-written Lexer properly tokenizes glosses from all 7 formal
token categories, correctly consults the Symbol Table for token types and meanings,
tracks zero-based token positions, handles whitespace variations, and reports lexical errors.
"""

import pytest
from compiler.lexer import Lexer, LexicalError
from compiler.symbol_table import Symbol, SymbolTable
from compiler.tokens import Token


@pytest.fixture
def lexer() -> Lexer:
    """Fixture providing a fresh instance of the Lexer with the default 44-gloss SymbolTable."""
    return Lexer()


# ==============================================================================
# 1. Basic Tokenization & Representative Tests
# ==============================================================================

def test_single_token(lexer: Lexer) -> None:
    """Test scanning a single valid token ('ME')."""
    result = lexer.tokenize("ME")
    assert result.is_valid
    assert len(result.tokens) == 1
    assert len(result.errors) == 0

    tok = result.tokens[0]
    assert tok.token_type == "PRONOUN"
    assert tok.lexeme == "ME"
    assert tok.meaning == "I"
    assert tok.position == 0


def test_multiple_valid_glosses(lexer: Lexer) -> None:
    """Test scanning multiple valid glosses ('ME WANT MILK')."""
    tokens, errors = lexer.tokenize("ME WANT MILK")
    assert len(errors) == 0
    assert len(tokens) == 3

    assert [t.token_type for t in tokens] == ["PRONOUN", "VERB", "NOUN"]
    assert [t.lexeme for t in tokens] == ["ME", "WANT", "MILK"]


# ==============================================================================
# 2. Coverage for All 7 Token Categories
# ==============================================================================

def test_pronouns_category(lexer: Lexer) -> None:
    """Verify all 6 pronouns are recognized with PRONOUN type."""
    tokens, errors = lexer.tokenize("ME YOU HE SHE WE THEY")
    assert len(errors) == 0
    assert len(tokens) == 6
    assert all(t.token_type == "PRONOUN" for t in tokens)
    assert [t.meaning for t in tokens] == ["I", "you", "he", "she", "we", "they"]


def test_time_category(lexer: Lexer) -> None:
    """Verify all 6 temporal markers are recognized with TIME type."""
    tokens, errors = lexer.tokenize("YESTERDAY TODAY TOMORROW NOW MORNING NIGHT")
    assert len(errors) == 0
    assert len(tokens) == 6
    assert all(t.token_type == "TIME" for t in tokens)
    assert [t.meaning for t in tokens] == ["yesterday", "today", "tomorrow", "now", "morning", "night"]


def test_verbs_category(lexer: Lexer) -> None:
    """Verify all 10 verbs are recognized with VERB type."""
    tokens, errors = lexer.tokenize("WANT HAVE LIKE GO COME EAT DRINK SEE READ NEED")
    assert len(errors) == 0
    assert len(tokens) == 10
    assert all(t.token_type == "VERB" for t in tokens)
    assert [t.meaning for t in tokens] == [
        "want", "have", "like", "go", "come", "eat", "drink", "see", "read", "need"
    ]


def test_nouns_category(lexer: Lexer) -> None:
    """Verify all 10 nouns are recognized with NOUN type."""
    tokens, errors = lexer.tokenize("MILK WATER FOOD BOOK HOME SCHOOL FRIEND PHONE MONEY CAR")
    assert len(errors) == 0
    assert len(tokens) == 10
    assert all(t.token_type == "NOUN" for t in tokens)
    assert [t.meaning for t in tokens] == [
        "milk", "water", "food", "book", "home", "school", "friend", "phone", "money", "car"
    ]


def test_adjectives_category(lexer: Lexer) -> None:
    """Verify all 6 adjectives are recognized with ADJECTIVE type."""
    tokens, errors = lexer.tokenize("HAPPY SAD GOOD BAD BIG SMALL")
    assert len(errors) == 0
    assert len(tokens) == 6
    assert all(t.token_type == "ADJECTIVE" for t in tokens)
    assert [t.meaning for t in tokens] == ["happy", "sad", "good", "bad", "big", "small"]


def test_wh_words_category(lexer: Lexer) -> None:
    """Verify all 5 interrogatives are recognized with WH_WORD type."""
    tokens, errors = lexer.tokenize("WHAT WHERE WHO WHEN WHY")
    assert len(errors) == 0
    assert len(tokens) == 5
    assert all(t.token_type == "WH_WORD" for t in tokens)
    assert [t.meaning for t in tokens] == ["what", "where", "who", "when", "why"]


def test_negation_category(lexer: Lexer) -> None:
    """Verify negation token NOT is recognized with NEGATION type."""
    tokens, errors = lexer.tokenize("NOT")
    assert len(errors) == 0
    assert len(tokens) == 1
    assert tokens[0].token_type == "NEGATION"
    assert tokens[0].lexeme == "NOT"
    assert tokens[0].meaning == "not"


# ==============================================================================
# 3. Multi-Category Sequences & Complex Sentence Tokenization
# ==============================================================================

def test_long_sequence_yesterday_me_read_book(lexer: Lexer) -> None:
    """Test tokenizing a multi-category sentence: 'YESTERDAY ME READ BOOK'."""
    result = lexer.tokenize("YESTERDAY ME READ BOOK")
    assert result.is_valid
    assert len(result.tokens) == 4

    expected = [
        ("TIME", "YESTERDAY", "yesterday", 0),
        ("PRONOUN", "ME", "I", 1),
        ("VERB", "READ", "read", 2),
        ("NOUN", "BOOK", "book", 3),
    ]

    for tok, (exp_type, exp_lex, exp_meaning, exp_pos) in zip(result.tokens, expected):
        assert tok.token_type == exp_type
        assert tok.lexeme == exp_lex
        assert tok.meaning == exp_meaning
        assert tok.position == exp_pos


def test_interrogative_sequence_who_you_see(lexer: Lexer) -> None:
    """Test tokenizing question sentence: 'WHO YOU SEE'."""
    result = lexer.tokenize("WHO YOU SEE")
    assert result.is_valid
    assert [t.token_type for t in result.tokens] == ["WH_WORD", "PRONOUN", "VERB"]
    assert [t.lexeme for t in result.tokens] == ["WHO", "YOU", "SEE"]


def test_adjective_sequence_she_have_big_car(lexer: Lexer) -> None:
    """Test tokenizing sentence with adjective modifier: 'SHE HAVE BIG CAR'."""
    result = lexer.tokenize("SHE HAVE BIG CAR")
    assert result.is_valid
    assert [t.token_type for t in result.tokens] == ["PRONOUN", "VERB", "ADJECTIVE", "NOUN"]
    assert [t.lexeme for t in result.tokens] == ["SHE", "HAVE", "BIG", "CAR"]


def test_negated_sequence_me_not_want_food(lexer: Lexer) -> None:
    """Test tokenizing negated sentence: 'ME NOT WANT FOOD'."""
    result = lexer.tokenize("ME NOT WANT FOOD")
    assert result.is_valid
    assert [t.token_type for t in result.tokens] == ["PRONOUN", "NEGATION", "VERB", "NOUN"]


# ==============================================================================
# 4. Lexical Error Detection for Unknown & Invalid Glosses
# ==============================================================================

def test_unknown_glosses_dance_apple_run(lexer: Lexer) -> None:
    """Verify that unknown glosses DANCE, APPLE, and RUN produce LexicalErrors."""
    for invalid in ["DANCE", "APPLE", "RUN"]:
        result = lexer.tokenize(f"ME {invalid} MILK")
        assert result.has_errors
        assert len(result.errors) == 1
        assert result.errors[0].invalid_lexeme == invalid
        assert result.errors[0].position == 1
        assert invalid in result.errors[0].message


def test_multiple_unknown_glosses_collected(lexer: Lexer) -> None:
    """Test collecting multiple unknown glosses while still capturing valid tokens."""
    result = lexer.tokenize("ME DANCE RUN APPLE MILK")
    assert len(result.errors) == 3
    assert [e.invalid_lexeme for e in result.errors] == ["DANCE", "RUN", "APPLE"]
    assert [e.position for e in result.errors] == [1, 2, 3]

    assert len(result.tokens) == 2
    assert [t.lexeme for t in result.tokens] == ["ME", "MILK"]
    assert [t.position for t in result.tokens] == [0, 4]


# ==============================================================================
# 5. Case Sensitivity Enforcement
# ==============================================================================

def test_case_sensitivity_lowercase_and_mixed(lexer: Lexer) -> None:
    """Verify lowercase and mixed-case glosses produce lexical errors."""
    res1 = lexer.tokenize("me WANT MILK")
    assert res1.has_errors
    assert res1.errors[0].invalid_lexeme == "me"
    assert res1.errors[0].position == 0

    res2 = lexer.tokenize("Happy YOU")
    assert res2.has_errors
    assert res2.errors[0].invalid_lexeme == "Happy"
    assert res2.errors[0].position == 0

    res3 = lexer.tokenize("sChOoL")
    assert res3.has_errors
    assert res3.errors[0].invalid_lexeme == "sChOoL"


# ==============================================================================
# 6. Formatting, Whitespace, & Position Verification
# ==============================================================================

def test_empty_and_whitespace_input(lexer: Lexer) -> None:
    """Test that empty and whitespace-only inputs return empty lists without crashing."""
    res1 = lexer.tokenize("")
    assert len(res1.tokens) == 0
    assert len(res1.errors) == 0
    assert not res1.has_errors

    res2 = lexer.tokenize("     \t \n  ")
    assert len(res2.tokens) == 0
    assert len(res2.errors) == 0


def test_whitespace_variations(lexer: Lexer) -> None:
    """Test that irregular leading, trailing, and multi-space whitespace is handled cleanly."""
    standard_result = lexer.tokenize("TODAY WE GO SCHOOL")
    varied_result = lexer.tokenize("  TODAY   WE    GO   SCHOOL  ")

    assert varied_result.is_valid
    assert len(varied_result.tokens) == 4
    assert [t.lexeme for t in varied_result.tokens] == [t.lexeme for t in standard_result.tokens]
    assert [t.position for t in varied_result.tokens] == [0, 1, 2, 3]


def test_token_positions_indexing(lexer: Lexer) -> None:
    """Test that token positions accurately reflect sequential 0-based token indices."""
    result = lexer.tokenize("MORNING FRIEND COME HOME HAPPY")
    assert [t.position for t in result.tokens] == [0, 1, 2, 3, 4]


# ==============================================================================
# 7. Symbol Table Consultation Verification (Decoupling)
# ==============================================================================

def test_meanings_and_types_come_from_symbol_table() -> None:
    """Verify Lexer retrieves attributes directly from the Symbol Table without hardcoding."""
    custom_table = SymbolTable(populate_defaults=False)
    custom_table.insert(Symbol(lexeme="ALPHA", token_type="CUSTOM_TYPE", meaning="custom meaning"))

    custom_lexer = Lexer(symbol_table=custom_table)
    result = custom_lexer.tokenize("ALPHA")

    assert result.is_valid
    assert len(result.tokens) == 1
    assert result.tokens[0].token_type == "CUSTOM_TYPE"
    assert result.tokens[0].meaning == "custom meaning"
    assert result.tokens[0].lexeme == "ALPHA"

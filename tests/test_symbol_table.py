"""
Tests for SignSyntax Symbol Table
Phase 1: Compiler Design - Symbol Table Verification (Expanded Formal Vocabulary)
"""

import pytest
from compiler.symbol_table import INITIAL_VOCABULARY, Symbol, SymbolTable


@pytest.fixture
def symbol_table() -> SymbolTable:
    """Provides a freshly initialized SymbolTable populated with the default formal glosses."""
    return SymbolTable()


def test_initial_size(symbol_table: SymbolTable) -> None:
    """Verify symbol table is pre-populated with the complete formal vocabulary (44 entries)."""
    assert symbol_table.size() == len(INITIAL_VOCABULARY)
    assert symbol_table.size() == 44


def test_all_formal_vocabulary_entries(symbol_table: SymbolTable) -> None:
    """Ensure every single defined gloss exists with the correct token type and English meaning."""
    expected = {
        # Pronouns (6)
        "ME": ("PRONOUN", "I"),
        "YOU": ("PRONOUN", "you"),
        "HE": ("PRONOUN", "he"),
        "SHE": ("PRONOUN", "she"),
        "WE": ("PRONOUN", "we"),
        "THEY": ("PRONOUN", "they"),
        # Temporal markers (6)
        "YESTERDAY": ("TIME", "yesterday"),
        "TODAY": ("TIME", "today"),
        "TOMORROW": ("TIME", "tomorrow"),
        "NOW": ("TIME", "now"),
        "MORNING": ("TIME", "morning"),
        "NIGHT": ("TIME", "night"),
        # Verbs (10)
        "WANT": ("VERB", "want"),
        "HAVE": ("VERB", "have"),
        "LIKE": ("VERB", "like"),
        "GO": ("VERB", "go"),
        "COME": ("VERB", "come"),
        "EAT": ("VERB", "eat"),
        "DRINK": ("VERB", "drink"),
        "SEE": ("VERB", "see"),
        "READ": ("VERB", "read"),
        "NEED": ("VERB", "need"),
        # Nouns (10)
        "MILK": ("NOUN", "milk"),
        "WATER": ("NOUN", "water"),
        "FOOD": ("NOUN", "food"),
        "BOOK": ("NOUN", "book"),
        "HOME": ("NOUN", "home"),
        "SCHOOL": ("NOUN", "school"),
        "FRIEND": ("NOUN", "friend"),
        "PHONE": ("NOUN", "phone"),
        "MONEY": ("NOUN", "money"),
        "CAR": ("NOUN", "car"),
        # Adjectives (6)
        "HAPPY": ("ADJECTIVE", "happy"),
        "SAD": ("ADJECTIVE", "sad"),
        "GOOD": ("ADJECTIVE", "good"),
        "BAD": ("ADJECTIVE", "bad"),
        "BIG": ("ADJECTIVE", "big"),
        "SMALL": ("ADJECTIVE", "small"),
        # WH-Words (5)
        "WHAT": ("WH_WORD", "what"),
        "WHERE": ("WH_WORD", "where"),
        "WHO": ("WH_WORD", "who"),
        "WHEN": ("WH_WORD", "when"),
        "WHY": ("WH_WORD", "why"),
        # Negation (1)
        "NOT": ("NEGATION", "not"),
    }

    assert len(expected) == 44

    for lexeme, (token_type, meaning) in expected.items():
        sym = symbol_table.lookup(lexeme)
        assert sym is not None, f"Expected symbol '{lexeme}' not found in table."
        assert sym.token_type == token_type, f"Token type mismatch for '{lexeme}': expected {token_type}, got {sym.token_type}"
        assert sym.meaning == meaning, f"Meaning mismatch for '{lexeme}': expected {meaning}, got {sym.meaning}"


def test_contains_representative_symbols(symbol_table: SymbolTable) -> None:
    """Test contains() returns True for valid representative symbols across categories."""
    assert symbol_table.contains("ME") is True
    assert symbol_table.contains("HAPPY") is True
    assert symbol_table.contains("WHY") is True
    assert symbol_table.contains("WANT") is True
    assert symbol_table.contains("MILK") is True
    assert symbol_table.contains("TODAY") is True
    assert symbol_table.contains("NOT") is True


def test_contains_unknown_symbols(symbol_table: SymbolTable) -> None:
    """Test contains() returns False for unknown vocabulary symbols."""
    assert symbol_table.contains("DANCE") is False
    assert symbol_table.contains("APPLE") is False
    assert symbol_table.contains("RUN") is False


def test_lookup_specific_symbols(symbol_table: SymbolTable) -> None:
    """Test lookup returns accurate Symbol information for specific queries."""
    me = symbol_table.lookup("ME")
    assert me is not None
    assert me.lexeme == "ME"
    assert me.token_type == "PRONOUN"
    assert me.meaning == "I"

    happy = symbol_table.lookup("HAPPY")
    assert happy is not None
    assert happy.lexeme == "HAPPY"
    assert happy.token_type == "ADJECTIVE"
    assert happy.meaning == "happy"

    why = symbol_table.lookup("WHY")
    assert why is not None
    assert why.lexeme == "WHY"
    assert why.token_type == "WH_WORD"
    assert why.meaning == "why"


def test_lookup_unknown_symbols(symbol_table: SymbolTable) -> None:
    """Test lookup() returns None for unknown vocabulary symbols."""
    assert symbol_table.lookup("DANCE") is None
    assert symbol_table.lookup("APPLE") is None


def test_case_sensitivity(symbol_table: SymbolTable) -> None:
    """Test that lookup and contains strictly enforce uppercase case sensitivity."""
    assert symbol_table.contains("ME") is True
    assert symbol_table.contains("me") is False
    assert symbol_table.contains("Me") is False

    assert symbol_table.contains("HAPPY") is True
    assert symbol_table.contains("happy") is False
    assert symbol_table.contains("Happy") is False

    assert symbol_table.lookup("ME") is not None
    assert symbol_table.lookup("me") is None
    assert symbol_table.lookup("Me") is None


def test_duplicate_insertion_rejection(symbol_table: SymbolTable) -> None:
    """Test that inserting an existing symbol raises ValueError."""
    with pytest.raises(ValueError, match="already exists"):
        symbol_table.insert(Symbol(lexeme="ME", token_type="PRONOUN", meaning="I"))

    with pytest.raises(ValueError, match="already exists"):
        symbol_table.insert(lexeme="HAPPY", token_type="ADJECTIVE", meaning="happy")


def test_insert_custom_symbol(symbol_table: SymbolTable) -> None:
    """Test inserting a new custom symbol for future extensibility."""
    initial_size = symbol_table.size()
    new_sym = symbol_table.insert(
        Symbol(lexeme="TREE", token_type="NOUN", meaning="tree")
    )
    assert symbol_table.size() == initial_size + 1
    assert symbol_table.contains("TREE") is True
    assert symbol_table.lookup("TREE") == new_sym


def test_remove_symbol(symbol_table: SymbolTable) -> None:
    """Test removing an existing symbol."""
    initial_size = symbol_table.size()
    assert symbol_table.remove("ME") is True
    assert symbol_table.size() == initial_size - 1
    assert symbol_table.contains("ME") is False
    assert symbol_table.lookup("ME") is None
    assert symbol_table.remove("NON_EXISTENT") is False


def test_clear_table(symbol_table: SymbolTable) -> None:
    """Test that clear() empties the table completely."""
    symbol_table.clear()
    assert symbol_table.size() == 0
    assert symbol_table.contains("ME") is False
    assert symbol_table.lookup("ME") is None


def test_display_and_dump_categories(symbol_table: SymbolTable) -> None:
    """Test that display and dump contain all 7 token categories."""
    output = symbol_table.display()
    assert "Lexeme" in output
    assert "Token Type" in output
    assert "Meaning" in output
    for category in ["PRONOUN", "TIME", "VERB", "NOUN", "ADJECTIVE", "WH_WORD", "NEGATION"]:
        assert category in output
    assert symbol_table.dump() == output

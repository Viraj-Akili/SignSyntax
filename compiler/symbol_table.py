"""
SignSyntax - Symbol Table Module
Phase 1: Compiler Infrastructure

In compiler design, the Symbol Table is a central data structure used to store
information about source language identifiers, keywords, and symbols. 

For SignSyntax, the Symbol Table maintains the recognized sign gloss vocabulary,
their associated token types, and corresponding English semantic meanings.
The Lexical Analyzer will consult this table during tokenization to validate
lexemes and produce typed tokens.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass(frozen=True)
class Symbol:
    """
    Represents a single entry in the SignSyntax Symbol Table.

    Attributes:
        lexeme (str): The unique sign gloss string (e.g., 'ME', 'WANT', 'HAPPY').
        token_type (str): The categorical token type (e.g., 'PRONOUN', 'VERB', 'ADJECTIVE').
        meaning (str): The English translation or semantic concept (e.g., 'I', 'want', 'happy').
    """
    lexeme: str
    token_type: str
    meaning: str

    def __str__(self) -> str:
        return f"Symbol(lexeme='{self.lexeme}', type='{self.token_type}', meaning='{self.meaning}')"


# Formal SignSyntax source vocabulary specification as defined in grammar/vocabulary.md
INITIAL_VOCABULARY: List[Symbol] = [
    # Pronouns (6)
    Symbol(lexeme="ME", token_type="PRONOUN", meaning="I"),
    Symbol(lexeme="YOU", token_type="PRONOUN", meaning="you"),
    Symbol(lexeme="HE", token_type="PRONOUN", meaning="he"),
    Symbol(lexeme="SHE", token_type="PRONOUN", meaning="she"),
    Symbol(lexeme="WE", token_type="PRONOUN", meaning="we"),
    Symbol(lexeme="THEY", token_type="PRONOUN", meaning="they"),

    # Temporal Markers (6)
    Symbol(lexeme="YESTERDAY", token_type="TIME", meaning="yesterday"),
    Symbol(lexeme="TODAY", token_type="TIME", meaning="today"),
    Symbol(lexeme="TOMORROW", token_type="TIME", meaning="tomorrow"),
    Symbol(lexeme="NOW", token_type="TIME", meaning="now"),
    Symbol(lexeme="MORNING", token_type="TIME", meaning="morning"),
    Symbol(lexeme="NIGHT", token_type="TIME", meaning="night"),

    # Verbs (10)
    Symbol(lexeme="WANT", token_type="VERB", meaning="want"),
    Symbol(lexeme="HAVE", token_type="VERB", meaning="have"),
    Symbol(lexeme="LIKE", token_type="VERB", meaning="like"),
    Symbol(lexeme="GO", token_type="VERB", meaning="go"),
    Symbol(lexeme="COME", token_type="VERB", meaning="come"),
    Symbol(lexeme="EAT", token_type="VERB", meaning="eat"),
    Symbol(lexeme="DRINK", token_type="VERB", meaning="drink"),
    Symbol(lexeme="SEE", token_type="VERB", meaning="see"),
    Symbol(lexeme="READ", token_type="VERB", meaning="read"),
    Symbol(lexeme="NEED", token_type="VERB", meaning="need"),

    # Nouns / Objects (10)
    Symbol(lexeme="MILK", token_type="NOUN", meaning="milk"),
    Symbol(lexeme="WATER", token_type="NOUN", meaning="water"),
    Symbol(lexeme="FOOD", token_type="NOUN", meaning="food"),
    Symbol(lexeme="BOOK", token_type="NOUN", meaning="book"),
    Symbol(lexeme="HOME", token_type="NOUN", meaning="home"),
    Symbol(lexeme="SCHOOL", token_type="NOUN", meaning="school"),
    Symbol(lexeme="FRIEND", token_type="NOUN", meaning="friend"),
    Symbol(lexeme="PHONE", token_type="NOUN", meaning="phone"),
    Symbol(lexeme="MONEY", token_type="NOUN", meaning="money"),
    Symbol(lexeme="CAR", token_type="NOUN", meaning="car"),

    # Adjectives (6)
    Symbol(lexeme="HAPPY", token_type="ADJECTIVE", meaning="happy"),
    Symbol(lexeme="SAD", token_type="ADJECTIVE", meaning="sad"),
    Symbol(lexeme="GOOD", token_type="ADJECTIVE", meaning="good"),
    Symbol(lexeme="BAD", token_type="ADJECTIVE", meaning="bad"),
    Symbol(lexeme="BIG", token_type="ADJECTIVE", meaning="big"),
    Symbol(lexeme="SMALL", token_type="ADJECTIVE", meaning="small"),

    # WH-Words / Interrogatives (5)
    Symbol(lexeme="WHAT", token_type="WH_WORD", meaning="what"),
    Symbol(lexeme="WHERE", token_type="WH_WORD", meaning="where"),
    Symbol(lexeme="WHO", token_type="WH_WORD", meaning="who"),
    Symbol(lexeme="WHEN", token_type="WH_WORD", meaning="when"),
    Symbol(lexeme="WHY", token_type="WH_WORD", meaning="why"),

    # Negation (1)
    Symbol(lexeme="NOT", token_type="NEGATION", meaning="not"),
]


class SymbolTable:
    """
    Symbol Table data structure for managing SignSyntax glosses and their attributes.

    Provides constant-time lookup, duplicate rejection, and maintenance
    of vocabulary attributes.
    """

    def __init__(self, populate_defaults: bool = True) -> None:
        """
        Initializes the SymbolTable.

        Args:
            populate_defaults (bool): If True, populates the table with the
                                      formal SignSyntax vocabulary symbols.
        """
        self._symbols: Dict[str, Symbol] = {}

        if populate_defaults:
            for symbol in INITIAL_VOCABULARY:
                self.insert(symbol)

    def insert(
        self,
        symbol: Optional[Symbol] = None,
        *,
        lexeme: Optional[str] = None,
        token_type: Optional[str] = None,
        meaning: Optional[str] = None,
    ) -> Symbol:
        """
        Inserts a new symbol into the symbol table.

        Accepts either a Symbol instance or keyword arguments (lexeme, token_type, meaning).

        Args:
            symbol (Optional[Symbol]): A pre-constructed Symbol instance.
            lexeme (Optional[str]): The gloss identifier.
            token_type (Optional[str]): The token category.
            meaning (Optional[str]): The English meaning.

        Returns:
            Symbol: The inserted Symbol instance.

        Raises:
            ValueError: If the symbol is invalid, incomplete, or already exists.
        """
        if symbol is None:
            if not lexeme or not token_type or not meaning:
                raise ValueError("Must provide either a Symbol instance or lexeme, token_type, and meaning.")
            entry = Symbol(lexeme=lexeme, token_type=token_type, meaning=meaning)
        else:
            entry = symbol

        if entry.lexeme in self._symbols:
            raise ValueError(f"Symbol '{entry.lexeme}' already exists in the symbol table.")

        self._symbols[entry.lexeme] = entry
        return entry

    def lookup(self, lexeme: str) -> Optional[Symbol]:
        """
        Searches for a symbol in the table by its exact lexeme.

        Lookup is strictly case-sensitive.

        Args:
            lexeme (str): The gloss string to search for (e.g., 'ME', 'HAPPY').

        Returns:
            Optional[Symbol]: The Symbol entry if found, otherwise None.
        """
        return self._symbols.get(lexeme, None)

    def contains(self, lexeme: str) -> bool:
        """
        Checks whether a given lexeme exists in the symbol table.

        Args:
            lexeme (str): The gloss string to check.

        Returns:
            bool: True if the lexeme exists, False otherwise.
        """
        return lexeme in self._symbols

    def remove(self, lexeme: str) -> bool:
        """
        Removes a symbol entry from the table.

        Args:
            lexeme (str): The lexeme to remove.

        Returns:
            bool: True if the symbol was found and removed, False otherwise.
        """
        if lexeme in self._symbols:
            del self._symbols[lexeme]
            return True
        return False

    def clear(self) -> None:
        """Clears all symbols from the table."""
        self._symbols.clear()

    def size(self) -> int:
        """
        Returns the number of symbols currently in the table.

        Returns:
            int: The total count of entries.
        """
        return len(self._symbols)

    def get_all_symbols(self) -> List[Symbol]:
        """
        Retrieves all symbols in the table.

        Returns:
            List[Symbol]: List of all Symbol entries.
        """
        return list(self._symbols.values())

    def display(self) -> str:
        """
        Generates a human-readable table representation of all symbol table entries.

        Returns:
            str: Formatted ASCII table.
        """
        header = f"{'Lexeme':<14} | {'Token Type':<14} | {'Meaning':<14}"
        separator = "-" * len(header)
        rows = [header, separator]

        for s in self._symbols.values():
            rows.append(f"{s.lexeme:<14} | {s.token_type:<14} | {s.meaning:<14}")

        return "\n".join(rows)

    def dump(self) -> str:
        """Alias for display() to support standard compiler debugging dumps."""
        return self.display()

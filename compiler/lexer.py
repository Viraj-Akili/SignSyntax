"""
SignSyntax - Lexical Analyzer (Lexer) Module
Phase 1: Compiler Infrastructure

The Lexer (or scanner) is the first phase of the SignSyntax compiler. It accepts a raw
stream of sign language glosses, partitions the text into individual lexemes using whitespace
delimiters, and consults the Symbol Table to convert valid lexemes into typed Token objects.

Any unrecognized, misspelled, or casing-invalid glosses produce detailed LexicalError instances.
"""

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Union

from compiler.symbol_table import SymbolTable
from compiler.tokens import Token


@dataclass(frozen=True)
class LexicalError:
    """
    Represents a lexical error encountered when scanning the input gloss stream.

    Attributes:
        invalid_lexeme (str): The unrecognized or invalid gloss string.
        position (int): The 0-based token index where the invalid lexeme occurred.
        message (str): A descriptive human-readable error message.
    """
    invalid_lexeme: str
    position: int
    message: str

    def __str__(self) -> str:
        return f"LEXICAL ERROR: {self.message}"


@dataclass
class LexerResult:
    """
    Encapsulates the result of a lexical analysis run.

    Supports both object-oriented property access and tuple unpacking:
        result = lexer.tokenize("ME WANT MILK")
        tokens, errors = result

    Attributes:
        tokens (List[Token]): List of successfully identified and typed tokens.
        errors (List[LexicalError]): List of lexical errors collected during scanning.
    """
    tokens: List[Token]
    errors: List[LexicalError]

    @property
    def has_errors(self) -> bool:
        """Returns True if one or more lexical errors occurred."""
        return len(self.errors) > 0

    @property
    def is_valid(self) -> bool:
        """Returns True if tokenization succeeded without any lexical errors."""
        return len(self.errors) == 0

    def __iter__(self) -> Iterator[Union[List[Token], List[LexicalError]]]:
        """Allows unpacking as `tokens, errors = lexer.tokenize(...)`."""
        return iter((self.tokens, self.errors))


class Lexer:
    """
    Hand-written Lexical Analyzer for SignSyntax.

    The Lexer operates on whitespace-delimited uppercase sign glosses, verifying each
    lexeme against the central Symbol Table.
    """

    def __init__(self, symbol_table: Optional[SymbolTable] = None) -> None:
        """
        Initializes the Lexer.

        Args:
            symbol_table (Optional[SymbolTable]): The SymbolTable to consult for valid glosses.
                                                  Defaults to a new standard SymbolTable instance.
        """
        self.symbol_table: SymbolTable = symbol_table if symbol_table is not None else SymbolTable()

    def tokenize(self, source: str) -> LexerResult:
        """
        Performs lexical analysis on the given raw gloss input string.

        Args:
            source (str): Raw input containing whitespace-separated sign glosses.

        Returns:
            LexerResult: Object containing the recognized Tokens and any LexicalErrors.
        """
        if not source or not source.strip():
            return LexerResult(tokens=[], errors=[])

        raw_lexemes = source.split()
        tokens: List[Token] = []
        errors: List[LexicalError] = []

        for position, lexeme in enumerate(raw_lexemes):
            symbol = self.symbol_table.lookup(lexeme)
            if symbol is not None:
                token = Token(
                    token_type=symbol.token_type,
                    lexeme=symbol.lexeme,
                    meaning=symbol.meaning,
                    position=position,
                )
                tokens.append(token)
            else:
                error = LexicalError(
                    invalid_lexeme=lexeme,
                    position=position,
                    message=f"Unknown gloss '{lexeme}' at position {position}",
                )
                errors.append(error)

        return LexerResult(tokens=tokens, errors=errors)

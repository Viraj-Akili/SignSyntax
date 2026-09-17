"""
SignSyntax - Token Representation
Phase 1: Compiler Infrastructure (Lexical Analysis)

A Token is the fundamental lexical unit produced by the Lexical Analyzer (Lexer).
It encapsulates the token category (type), the actual source lexeme, its associated
semantic meaning, and its position in the source input stream.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    """
    Represents a lexical token emitted by the SignSyntax Lexer.

    Attributes:
        token_type (str): The category of the token (e.g., 'PRONOUN', 'VERB', 'NOUN', 'TIME', 'WH_WORD', 'NEGATION').
        lexeme (str): The exact text/gloss recognized in the input stream (e.g., 'ME', 'WANT').
        meaning (str): The English translation or concept associated with the symbol (e.g., 'I', 'want').
        position (int): The 0-based token index indicating where this token appeared in the input sequence.
    """
    token_type: str
    lexeme: str
    meaning: str
    position: int

    def __repr__(self) -> str:
        return f"Token(type='{self.token_type}', lexeme='{self.lexeme}', meaning='{self.meaning}', position={self.position})"

    def __str__(self) -> str:
        return f"Token(type={self.token_type}, lexeme={self.lexeme}, meaning={self.meaning}, position={self.position})"

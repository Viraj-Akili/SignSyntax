"""
SignSyntax - Compiler Package Initialization
Phase 1: Project Foundation & Scaffolding
"""

from compiler.lexer import Lexer, LexerResult, LexicalError
from compiler.symbol_table import Symbol, SymbolTable
from compiler.tokens import Token

__version__ = "0.1.0"

__all__ = [
    "Lexer",
    "LexerResult",
    "LexicalError",
    "Symbol",
    "SymbolTable",
    "Token",
]

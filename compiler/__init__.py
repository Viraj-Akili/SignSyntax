"""
SignSyntax - Compiler Package Initialization
Phase 2: Core Implementation (Lexer, Parser, AST, Semantic Analyzer, IR)
"""

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
from compiler.frontend import FrontendCompiler, FrontendResult, compile_frontend
from compiler.ir import (
    IRGenerationResult,
    IRGenerator,
    IRInstruction,
    IROpCode,
    IRProgram,
    IRValidationError,
)
from compiler.lexer import Lexer, LexerResult, LexicalError
from compiler.parser import ParseError, Parser, ParseResult, ParseTreeNode
from compiler.semantic_analyzer import (
    QuestionSemanticForm,
    SemanticAnalyzer,
    SemanticError,
    SemanticForm,
    SemanticResult,
)
from compiler.symbol_table import Symbol, SymbolTable
from compiler.tokens import Token

__version__ = "0.2.0"

__all__ = [
    "ASTBuilder",
    "ASTNode",
    "AdjectiveNode",
    "DeclarativeSentenceNode",
    "FrontendCompiler",
    "FrontendResult",
    "IRGenerationResult",
    "IRGenerator",
    "IRInstruction",
    "IROpCode",
    "IRProgram",
    "IRValidationError",
    "Lexer",
    "LexerResult",
    "LexicalError",
    "NegationNode",
    "ObjectNode",
    "ParseError",
    "Parser",
    "ParseResult",
    "ParseTreeNode",
    "QuestionNode",
    "QuestionSemanticForm",
    "SemanticAnalyzer",
    "SemanticError",
    "SemanticForm",
    "SemanticResult",
    "SentenceNode",
    "SubjectNode",
    "Symbol",
    "SymbolTable",
    "TimeNode",
    "Token",
    "VerbPhraseNode",
    "WHWordNode",
    "build_ast",
    "compile_frontend",
]

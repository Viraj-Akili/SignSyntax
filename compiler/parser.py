"""
SignSyntax - Syntax Analyzer (Parser) Module
Phase 2: Core Implementation (Syntax Analysis)

This module implements a clean, hand-written recursive-descent LL(1) parser
for the formal SignSyntax language based directly on the frozen BNF grammar in grammar/grammar.bnf.

The parser consumes a sequence of typed Token objects emitted by the Lexer,
verifies syntactic correctness against production rules, ensures the full
token stream is consumed without trailing tokens, and builds a structured
parse tree representation.

Architecture:
    Gloss Input
         ↓
       Lexer
         ↓
      Tokens
         ↓
      Parser
         ↓
    ParseResult (ParseTreeNode / ParseError)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from compiler.ast import SentenceNode, build_ast
from compiler.tokens import Token


@dataclass(frozen=True)
class ParseError(Exception):
    """
    Represents a syntax error encountered during recursive-descent parsing.

    Attributes:
        message (str): Descriptive, student-understandable explanation of the error.
        position (int): 0-based token position in the input stream where the error occurred.
        found_token (Optional[Token]): The actual unexpected token found, or None if EOF.
        expected (Optional[List[str]]): List of expected token types or grammar constructs.
    """
    message: str
    position: int
    found_token: Optional[Token] = None
    expected: Optional[List[str]] = None

    def __str__(self) -> str:
        return f"SYNTAX ERROR at position {self.position}: {self.message}"


@dataclass
class ParseTreeNode:
    """
    Represents a node in the concrete syntax / parse tree produced by the Parser.

    Attributes:
        rule (str): The grammar production rule or non-terminal name (e.g., '<Sentence>', '<VerbPhrase>').
        children (List[Union[ParseTreeNode, Token]]): Child parse nodes or leaf Tokens.
        attributes (Dict[str, Any]): Optional metadata / semantic attributes for demonstration.
    """
    rule: str
    children: List[Union["ParseTreeNode", Token]] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the parse tree to a nested dictionary for visualization and debugging."""
        serialized_children = []
        for child in self.children:
            if isinstance(child, ParseTreeNode):
                serialized_children.append(child.to_dict())
            elif isinstance(child, Token):
                serialized_children.append({
                    "token_type": child.token_type,
                    "lexeme": child.lexeme,
                    "meaning": child.meaning,
                    "position": child.position,
                })
        return {
            "rule": self.rule,
            "children": serialized_children,
            "attributes": self.attributes,
        }

    def display(self, indent: int = 0) -> str:
        """Renders an ASCII tree visualization of the parse tree."""
        prefix = "  " * indent
        lines = [f"{prefix}{self.rule}"]
        for child in self.children:
            if isinstance(child, ParseTreeNode):
                lines.append(child.display(indent + 1))
            elif isinstance(child, Token):
                lines.append(f"{prefix}  [Token: {child.token_type} '{child.lexeme}' (pos={child.position})]")
        return "\n".join(lines)


@dataclass
class ParseResult:
    """
    Encapsulates the result of a syntax parsing operation.

    Supports boolean evaluation (`if parse_result.is_valid: ...`) and error checking.

    Attributes:
        tree (Optional[ParseTreeNode]): The root parse tree node if parsing succeeded, else None.
        errors (List[ParseError]): List of syntax errors collected during parsing.
        is_valid (bool): True if parsing succeeded with zero syntax errors.
        ast (Optional[SentenceNode]): The clean Abstract Syntax Tree root if parsing succeeded, else None.
    """
    tree: Optional[ParseTreeNode] = None
    errors: List[ParseError] = field(default_factory=list)
    is_valid: bool = True
    ast: Optional[SentenceNode] = None

    @property
    def has_errors(self) -> bool:
        """Returns True if one or more syntax errors occurred."""
        return len(self.errors) > 0 or not self.is_valid


class Parser:
    """
    Hand-written recursive-descent LL(1) Parser for SignSyntax.

    Consumes a stream of typed Token objects from the Lexer and verifies syntax
    according to the frozen BNF grammar:

        <Sentence>            ::= <DeclarativeSentence> | <Question>
        <DeclarativeSentence> ::= <TimeMarker>? <Subject> <VerbPhrase> <Negation>?
        <VerbPhrase>          ::= <Verb> | <Verb> <Object> | <Verb> <Object> <Adjective>
        <Question>            ::= <WHQuestion>
        <WHQuestion>          ::= <WHWord> <Subject> <VerbPhrase> | <WHWord> <Object>
        <TimeMarker>          ::= YESTERDAY | TODAY | TOMORROW | NOW | MORNING | NIGHT
        <Subject>             ::= <Pronoun>
        <Pronoun>             ::= ME | YOU | HE | SHE | WE | THEY
        <Verb>                ::= WANT | HAVE | LIKE | GO | COME | EAT | DRINK | SEE | READ | NEED
        <Object>              ::= MILK | WATER | FOOD | BOOK | HOME | SCHOOL | FRIEND | PHONE | MONEY | CAR
        <Adjective>           ::= HAPPY | SAD | GOOD | BAD | BIG | SMALL
        <WHWord>              ::= WHAT | WHERE | WHO | WHEN | WHY
        <Negation>            ::= NOT
    """

    def __init__(self, tokens: List[Token]) -> None:
        """
        Initializes the Parser with a list of tokens from lexical analysis.

        Args:
            tokens (List[Token]): Token stream produced by Lexer.tokenize().
        """
        self.tokens: List[Token] = list(tokens) if tokens is not None else []
        self.cursor: int = 0

    # -------------------------------------------------------------------------
    # Stream Navigation & Token Matching Helpers
    # -------------------------------------------------------------------------

    def is_at_end(self) -> bool:
        """Checks if all tokens in the stream have been consumed."""
        return self.cursor >= len(self.tokens)

    def current_token(self) -> Optional[Token]:
        """Returns the token at the current cursor position, or None if EOF."""
        if self.is_at_end():
            return None
        return self.tokens[self.cursor]

    def peek(self, offset: int = 0) -> Optional[Token]:
        """Lookahead at the token `offset` positions ahead of current cursor."""
        target_index = self.cursor + offset
        if 0 <= target_index < len(self.tokens):
            return self.tokens[target_index]
        return None

    def match(self, *token_types: str) -> bool:
        """Checks if current token matches any of the given token types without consuming."""
        curr = self.current_token()
        if curr is None:
            return False
        return curr.token_type in token_types

    def consume(self, expected_type: str, context_description: Optional[str] = None) -> Token:
        """
        Consumes and returns the current token if its type matches expected_type.
        Otherwise raises a descriptive ParseError.

        Args:
            expected_type (str): The required Token type (e.g., 'VERB', 'PRONOUN').
            context_description (Optional[str]): Human-readable context for error messages.

        Returns:
            Token: The consumed token.

        Raises:
            ParseError: If token does not match or if input ended prematurely.
        """
        if self.is_at_end():
            last_pos = self.tokens[-1].position + 1 if self.tokens else 0
            desc = context_description or f"token of type {expected_type}"
            raise ParseError(
                message=f"Unexpected end of input; expected {desc}",
                position=last_pos,
                found_token=None,
                expected=[expected_type],
            )

        curr = self.tokens[self.cursor]
        if curr.token_type != expected_type:
            desc = context_description or f"token of type {expected_type}"
            raise ParseError(
                message=f"Expected {desc}, found {curr.token_type} '{curr.lexeme}'",
                position=curr.position,
                found_token=curr,
                expected=[expected_type],
            )

        self.cursor += 1
        return curr

    # -------------------------------------------------------------------------
    # Public Entry Point
    # -------------------------------------------------------------------------

    def parse(self) -> ParseResult:
        """
        Parses the complete token stream into a parse tree.

        Enforces that the entire token stream is valid and fully consumed without
        unconsumed trailing tokens.

        Returns:
            ParseResult: Result object containing the parse tree or syntax errors.
        """
        if not self.tokens:
            error = ParseError(
                message="Empty input sequence; expected sentence starting with TIME, PRONOUN, or WH_WORD",
                position=0,
                found_token=None,
                expected=["TIME", "PRONOUN", "WH_WORD"],
            )
            return ParseResult(tree=None, errors=[error], is_valid=False)

        try:
            tree = self.parse_sentence()

            # Verify complete token stream is consumed (no trailing tokens)
            if not self.is_at_end():
                trailing_token = self.current_token()
                pos = trailing_token.position if trailing_token else len(self.tokens)
                lexeme = trailing_token.lexeme if trailing_token else "<EOF>"
                error = ParseError(
                    message=f"Unexpected trailing token '{lexeme}' after complete sentence",
                    position=pos,
                    found_token=trailing_token,
                    expected=["<EOF>"],
                )
                return ParseResult(tree=None, errors=[error], is_valid=False, ast=None)

            ast = build_ast(tree)
            return ParseResult(tree=tree, errors=[], is_valid=True, ast=ast)

        except ParseError as exc:
            return ParseResult(tree=None, errors=[exc], is_valid=False, ast=None)

    # -------------------------------------------------------------------------
    # Grammar Production Parsing Functions
    # -------------------------------------------------------------------------

    def parse_sentence(self) -> ParseTreeNode:
        """
        <Sentence> ::= <DeclarativeSentence> | <Question>

        LL(1) Lookahead:
        - If lookahead is 'TIME' or 'PRONOUN': parse <DeclarativeSentence>
        - If lookahead is 'WH_WORD': parse <Question>
        - Otherwise: raise ParseError
        """
        curr = self.current_token()
        if curr is None:
            last_pos = self.tokens[-1].position + 1 if self.tokens else 0
            raise ParseError(
                message="Unexpected end of input; expected sentence",
                position=last_pos,
                found_token=None,
                expected=["TIME", "PRONOUN", "WH_WORD"],
            )

        if curr.token_type in ("TIME", "PRONOUN"):
            child = self.parse_declarative_sentence()
            return ParseTreeNode(rule="<Sentence>", children=[child])
        elif curr.token_type == "WH_WORD":
            child = self.parse_question()
            return ParseTreeNode(rule="<Sentence>", children=[child])
        else:
            raise ParseError(
                message=f"Expected sentence starting with TIME, PRONOUN, or WH_WORD, found {curr.token_type} '{curr.lexeme}'",
                position=curr.position,
                found_token=curr,
                expected=["TIME", "PRONOUN", "WH_WORD"],
            )

    def parse_declarative_sentence(self) -> ParseTreeNode:
        """
        <DeclarativeSentence> ::= <TimeMarker>? <Subject> <VerbPhrase> <Negation>?
        """
        children: List[Union[ParseTreeNode, Token]] = []

        # 1. Optional <TimeMarker>?
        if self.match("TIME"):
            time_node = self.parse_time_marker()
            children.append(time_node)

        # 2. Mandatory <Subject>
        subject_node = self.parse_subject()
        children.append(subject_node)

        # 3. Mandatory <VerbPhrase>
        vp_node = self.parse_verb_phrase()
        children.append(vp_node)

        # 4. Optional <Negation>?
        if self.match("NEGATION"):
            neg_node = self.parse_negation()
            children.append(neg_node)

        return ParseTreeNode(rule="<DeclarativeSentence>", children=children)

    def parse_verb_phrase(self) -> ParseTreeNode:
        """
        <VerbPhrase> ::= <Verb>
                       | <Verb> <Object>
                       | <Verb> <Object> <Adjective>

        LL(1) Lookahead:
        1. Mandatory <Verb>
        2. If next token is NOUN: consume <Object>
           a. If following token is ADJECTIVE: consume <Adjective>
        3. If next token is ADJECTIVE without NOUN: raise ParseError
        """
        children: List[Union[ParseTreeNode, Token]] = []

        # 1. Mandatory <Verb>
        verb_node = self.parse_verb()
        children.append(verb_node)

        # 2. Check for optional <Object> (NOUN)
        if self.match("NOUN"):
            obj_node = self.parse_object()
            children.append(obj_node)

            # 3. Check for optional <Adjective> (ADJECTIVE) after Object
            if self.match("ADJECTIVE"):
                adj_node = self.parse_adjective()
                children.append(adj_node)
        elif self.match("ADJECTIVE"):
            # An adjective directly following a verb without an object is invalid
            curr = self.current_token()
            verb_lexeme = verb_node.children[0].lexeme if verb_node.children else "VERB"
            raise ParseError(
                message=f"Adjective '{curr.lexeme}' cannot directly follow verb '{verb_lexeme}'; expected NOUN object first",
                position=curr.position,
                found_token=curr,
                expected=["NOUN"],
            )

        return ParseTreeNode(rule="<VerbPhrase>", children=children)

    def parse_question(self) -> ParseTreeNode:
        """
        <Question> ::= <WHQuestion>
        """
        child = self.parse_wh_question()
        return ParseTreeNode(rule="<Question>", children=[child])

    def parse_wh_question(self) -> ParseTreeNode:
        """
        <WHQuestion> ::= <WHWord> <Subject> <VerbPhrase>
                       | <WHWord> <Object>

        LL(1) Lookahead after <WHWord>:
        - If next token is 'PRONOUN': parse <Subject> and <VerbPhrase>
        - If next token is 'NOUN': parse <Object>
        - Otherwise: raise ParseError
        """
        children: List[Union[ParseTreeNode, Token]] = []

        # 1. Mandatory <WHWord>
        wh_node = self.parse_wh_word()
        children.append(wh_node)

        # Lookahead next token
        curr = self.current_token()
        wh_lexeme = wh_node.children[0].lexeme if wh_node.children else "WH_WORD"

        if curr is None:
            last_pos = self.tokens[-1].position + 1 if self.tokens else 0
            raise ParseError(
                message=f"Incomplete WH-question after '{wh_lexeme}'; expected PRONOUN subject or NOUN object",
                position=last_pos,
                found_token=None,
                expected=["PRONOUN", "NOUN"],
            )

        if curr.token_type == "PRONOUN":
            subject_node = self.parse_subject()
            children.append(subject_node)
            vp_node = self.parse_verb_phrase()
            children.append(vp_node)
        elif curr.token_type == "NOUN":
            obj_node = self.parse_object()
            children.append(obj_node)
        else:
            raise ParseError(
                message=f"Expected PRONOUN subject or NOUN object after '{wh_lexeme}', found {curr.token_type} '{curr.lexeme}'",
                position=curr.position,
                found_token=curr,
                expected=["PRONOUN", "NOUN"],
            )

        return ParseTreeNode(rule="<WHQuestion>", children=children)

    # -------------------------------------------------------------------------
    # Terminal & Category Helper Parsing Functions
    # -------------------------------------------------------------------------

    def parse_time_marker(self) -> ParseTreeNode:
        """<TimeMarker> ::= YESTERDAY | TODAY | TOMORROW | NOW | MORNING | NIGHT"""
        token = self.consume("TIME", "TIME marker (e.g., YESTERDAY, TODAY, NOW)")
        return ParseTreeNode(rule="<TimeMarker>", children=[token])

    def parse_subject(self) -> ParseTreeNode:
        """<Subject> ::= <Pronoun>"""
        pronoun_node = self.parse_pronoun()
        return ParseTreeNode(rule="<Subject>", children=[pronoun_node])

    def parse_pronoun(self) -> ParseTreeNode:
        """<Pronoun> ::= ME | YOU | HE | SHE | WE | THEY"""
        token = self.consume("PRONOUN", "PRONOUN subject (e.g., ME, YOU, HE)")
        return ParseTreeNode(rule="<Pronoun>", children=[token])

    def parse_verb(self) -> ParseTreeNode:
        """<Verb> ::= WANT | HAVE | LIKE | GO | COME | EAT | DRINK | SEE | READ | NEED"""
        token = self.consume("VERB", "VERB predicate (e.g., WANT, LIKE, GO)")
        return ParseTreeNode(rule="<Verb>", children=[token])

    def parse_object(self) -> ParseTreeNode:
        """<Object> ::= MILK | WATER | FOOD | BOOK | HOME | SCHOOL | FRIEND | PHONE | MONEY | CAR"""
        token = self.consume("NOUN", "NOUN object (e.g., MILK, BOOK, HOME)")
        return ParseTreeNode(rule="<Object>", children=[token])

    def parse_adjective(self) -> ParseTreeNode:
        """<Adjective> ::= HAPPY | SAD | GOOD | BAD | BIG | SMALL"""
        token = self.consume("ADJECTIVE", "ADJECTIVE modifier (e.g., GOOD, BIG, HAPPY)")
        return ParseTreeNode(rule="<Adjective>", children=[token])

    def parse_wh_word(self) -> ParseTreeNode:
        """<WHWord> ::= WHAT | WHERE | WHO | WHEN | WHY"""
        token = self.consume("WH_WORD", "WH_WORD interrogative (e.g., WHAT, WHERE, WHO)")
        return ParseTreeNode(rule="<WHWord>", children=[token])

    def parse_negation(self) -> ParseTreeNode:
        """<Negation> ::= NOT"""
        token = self.consume("NEGATION", "NEGATION marker 'NOT'")
        return ParseTreeNode(rule="<Negation>", children=[token])

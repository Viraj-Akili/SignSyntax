"""
SignSyntax - Abstract Syntax Tree (AST) Module
Phase 2: Core Implementation (AST Representation & Construction)

This module defines the clean, high-level Abstract Syntax Tree (AST) hierarchy
for the SignSyntax compiler. While the concrete Parse Tree captures every grammar
production rule, the AST captures the essential semantic structure of sign gloss
sentences without grammatical overhead.

Architecture:
    Gloss Input
         ↓
       Lexer
         ↓
      Tokens
         ↓
      Parser
         ↓
    Parse Tree
         ↓
    ASTBuilder
         ↓
      AST (SentenceNode)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from compiler.tokens import Token


# ==============================================================================
# Base AST Node
# ==============================================================================

class ASTNode(ABC):
    """
    Abstract base class for all Abstract Syntax Tree nodes in SignSyntax.
    """

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serializes the AST node and its subtrees into a structured dictionary."""
        pass

    @abstractmethod
    def to_ascii(self, indent: int = 0, is_last: bool = True, prefix: str = "") -> str:
        """Renders a clean ASCII visual representation of the AST node."""
        pass

    def pretty_print(self) -> str:
        """User-friendly alias for to_ascii()."""
        return self.to_ascii()

    def __str__(self) -> str:
        return self.to_ascii()


# ==============================================================================
# Leaf & Semantic Value Nodes
# ==============================================================================

@dataclass
class TimeNode(ASTNode):
    """
    Represents a temporal frame marker in a sentence (e.g., 'YESTERDAY', 'NOW').

    Attributes:
        value (str): The uppercase sign gloss (e.g., 'YESTERDAY').
        meaning (str): The English translation (e.g., 'yesterday').
    """
    value: str
    meaning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "Time",
            "value": self.value,
            "meaning": self.meaning,
        }

    def to_ascii(self, indent: int = 0, is_last: bool = True, prefix: str = "") -> str:
        connector = "\\-- " if is_last else "|-- "
        return f"{prefix}{connector}Time: {self.value}"


@dataclass
class SubjectNode(ASTNode):
    """
    Represents the subject argument (pronoun) of a sentence (e.g., 'ME', 'YOU').

    Attributes:
        value (str): The subject pronoun gloss (e.g., 'ME').
        meaning (str): The English translation (e.g., 'I').
    """
    value: str
    meaning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "Subject",
            "value": self.value,
            "meaning": self.meaning,
        }

    def to_ascii(self, indent: int = 0, is_last: bool = True, prefix: str = "") -> str:
        connector = "\\-- " if is_last else "|-- "
        return f"{prefix}{connector}Subject: {self.value}"


@dataclass
class ObjectNode(ASTNode):
    """
    Represents the direct object / entity argument of a verb phrase (e.g., 'MILK', 'BOOK').

    Attributes:
        value (str): The noun gloss (e.g., 'MILK').
        meaning (str): The English translation (e.g., 'milk').
    """
    value: str
    meaning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "Object",
            "value": self.value,
            "meaning": self.meaning,
        }

    def to_ascii(self, indent: int = 0, is_last: bool = True, prefix: str = "") -> str:
        connector = "\\-- " if is_last else "|-- "
        return f"{prefix}{connector}Object: {self.value}"


@dataclass
class AdjectiveNode(ASTNode):
    """
    Represents an adjective modifier qualifying an object (e.g., 'BIG', 'GOOD').

    Attributes:
        value (str): The adjective gloss (e.g., 'BIG').
        meaning (str): The English translation (e.g., 'big').
    """
    value: str
    meaning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "Adjective",
            "value": self.value,
            "meaning": self.meaning,
        }

    def to_ascii(self, indent: int = 0, is_last: bool = True, prefix: str = "") -> str:
        connector = "\\-- " if is_last else "|-- "
        return f"{prefix}{connector}Adjective: {self.value}"


@dataclass
class NegationNode(ASTNode):
    """
    Represents a clause-level negation particle ('NOT').

    Attributes:
        value (str): The negation gloss (default 'NOT').
    """
    value: str = "NOT"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "Negation",
            "value": self.value,
        }

    def to_ascii(self, indent: int = 0, is_last: bool = True, prefix: str = "") -> str:
        connector = "\\-- " if is_last else "|-- "
        return f"{prefix}{connector}Negation: {self.value}"


@dataclass
class WHWordNode(ASTNode):
    """
    Represents an interrogative inquiry marker in questions (e.g., 'WHAT', 'WHERE').

    Attributes:
        value (str): The WH-word gloss (e.g., 'WHAT').
        meaning (str): The English translation (e.g., 'what').
    """
    value: str
    meaning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "WHWord",
            "value": self.value,
            "meaning": self.meaning,
        }

    def to_ascii(self, indent: int = 0, is_last: bool = True, prefix: str = "") -> str:
        connector = "\\-- " if is_last else "|-- "
        return f"{prefix}{connector}WHWord: {self.value}"


# ==============================================================================
# Phrase & Structural Nodes
# ==============================================================================

@dataclass
class VerbPhraseNode(ASTNode):
    """
    Represents a predicate verb phrase containing a mandatory verb,
    an optional direct object, and an optional qualifying adjective.

    Attributes:
        verb (str): The action / state verb gloss (e.g., 'WANT', 'LIKE').
        verb_meaning (str): English translation of the verb (e.g., 'want').
        object (Optional[ObjectNode]): Optional direct object node.
        adjective (Optional[AdjectiveNode]): Optional post-nominal modifier node.
    """
    verb: str
    verb_meaning: str = ""
    object: Optional[ObjectNode] = None
    adjective: Optional[AdjectiveNode] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "VerbPhrase",
            "verb": self.verb,
            "verb_meaning": self.verb_meaning,
            "object": self.object.to_dict() if self.object else None,
            "adjective": self.adjective.to_dict() if self.adjective else None,
        }

    def to_ascii(self, indent: int = 0, is_last: bool = True, prefix: str = "") -> str:
        connector = "\\-- " if is_last else "|-- "
        child_prefix = prefix + ("    " if is_last else "|   ")
        lines = [f"{prefix}{connector}VerbPhrase"]

        # Verb is always present
        has_children = (self.object is not None) or (self.adjective is not None)
        verb_conn = "|-- " if has_children else "\\-- "
        lines.append(f"{child_prefix}{verb_conn}Verb: {self.verb}")

        if self.object is not None:
            has_adj = self.adjective is not None
            lines.append(self.object.to_ascii(is_last=not has_adj, prefix=child_prefix))

        if self.adjective is not None:
            lines.append(self.adjective.to_ascii(is_last=True, prefix=child_prefix))

        return "\n".join(lines)


@dataclass
class DeclarativeSentenceNode(ASTNode):
    """
    Represents a declarative sign sentence / statement.

    Structure:
        [TimeMarker]? + Subject + VerbPhrase + [Negation]?

    Attributes:
        subject (SubjectNode): The subject pronoun.
        verb_phrase (VerbPhraseNode): The predicate phrase.
        time (Optional[TimeNode]): Optional leading temporal marker.
        negation (Optional[NegationNode]): Optional clause-final negation.
    """
    subject: SubjectNode
    verb_phrase: VerbPhraseNode
    time: Optional[TimeNode] = None
    negation: Optional[NegationNode] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "DeclarativeSentence",
            "time": self.time.to_dict() if self.time else None,
            "subject": self.subject.to_dict(),
            "verb_phrase": self.verb_phrase.to_dict(),
            "negation": self.negation.to_dict() if self.negation else None,
        }

    def to_ascii(self, indent: int = 0, is_last: bool = True, prefix: str = "") -> str:
        children: List[Tuple[str, ASTNode]] = []
        if self.time is not None:
            children.append(("time", self.time))
        children.append(("subject", self.subject))
        children.append(("verb_phrase", self.verb_phrase))
        if self.negation is not None:
            children.append(("negation", self.negation))

        lines = []
        for i, (_, child_node) in enumerate(children):
            child_is_last = (i == len(children) - 1)
            lines.append(child_node.to_ascii(is_last=child_is_last, prefix=prefix))

        return "\n".join(lines)


@dataclass
class QuestionNode(ASTNode):
    """
    Represents an interrogative question sentence.

    Variants:
        1. <WHWord> + Subject + VerbPhrase (e.g., 'WHAT YOU WANT')
        2. <WHWord> + Object (e.g., 'WHAT BOOK', 'WHERE HOME')

    Attributes:
        wh_word (WHWordNode): The interrogative marker (e.g., 'WHAT').
        subject (Optional[SubjectNode]): Optional subject for predicate questions.
        verb_phrase (Optional[VerbPhraseNode]): Optional predicate phrase.
        object (Optional[ObjectNode]): Optional object for concise object queries.
    """
    wh_word: WHWordNode
    subject: Optional[SubjectNode] = None
    verb_phrase: Optional[VerbPhraseNode] = None
    object: Optional[ObjectNode] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "Question",
            "wh_word": self.wh_word.to_dict(),
            "subject": self.subject.to_dict() if self.subject else None,
            "verb_phrase": self.verb_phrase.to_dict() if self.verb_phrase else None,
            "object": self.object.to_dict() if self.object else None,
        }

    def to_ascii(self, indent: int = 0, is_last: bool = True, prefix: str = "") -> str:
        children: List[ASTNode] = [self.wh_word]
        if self.subject is not None:
            children.append(self.subject)
        if self.verb_phrase is not None:
            children.append(self.verb_phrase)
        if self.object is not None:
            children.append(self.object)

        lines = []
        for i, child_node in enumerate(children):
            child_is_last = (i == len(children) - 1)
            lines.append(child_node.to_ascii(is_last=child_is_last, prefix=prefix))

        return "\n".join(lines)


# ==============================================================================
# Root Sentence Node
# ==============================================================================

@dataclass
class SentenceNode(ASTNode):
    """
    Root node of a SignSyntax Abstract Syntax Tree.

    Wraps either a DeclarativeSentenceNode or a QuestionNode.

    Attributes:
        body (Union[DeclarativeSentenceNode, QuestionNode]): The parsed sentence structure.
    """
    body: Union[DeclarativeSentenceNode, QuestionNode]

    @property
    def is_question(self) -> bool:
        """Returns True if the sentence is an interrogative question."""
        return isinstance(self.body, QuestionNode)

    @property
    def is_declarative(self) -> bool:
        """Returns True if the sentence is a declarative statement."""
        return isinstance(self.body, DeclarativeSentenceNode)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": "Sentence",
            "body": self.body.to_dict(),
        }

    def to_ascii(self, indent: int = 0, is_last: bool = True, prefix: str = "") -> str:
        node_name = "Question" if self.is_question else "Sentence"
        lines = [f"{node_name}"]
        body_ascii = self.body.to_ascii(prefix="")
        if body_ascii:
            lines.append(body_ascii)
        return "\n".join(lines)


# ==============================================================================
# AST Construction Builder
# ==============================================================================

class ASTBuilder:
    """
    Constructs an Abstract Syntax Tree (AST) from a concrete ParseTreeNode.

    Extracts meaningful semantic entities from concrete grammar productions,
    omits syntax-only nodes, and returns a clean SentenceNode hierarchy.
    """

    @classmethod
    def build(cls, parse_tree: Any) -> Optional[SentenceNode]:
        """
        Converts a concrete ParseTreeNode root (<Sentence>) into an AST SentenceNode.

        Args:
            parse_tree: Root ParseTreeNode from Parser.parse().

        Returns:
            Optional[SentenceNode]: Root AST node, or None if parse_tree is invalid.
        """
        if parse_tree is None:
            return None

        # Root must be <Sentence>
        if getattr(parse_tree, "rule", None) != "<Sentence>" or not parse_tree.children:
            return None

        child_node = parse_tree.children[0]
        rule = getattr(child_node, "rule", None)

        if rule == "<DeclarativeSentence>":
            decl_ast = cls._build_declarative(child_node)
            return SentenceNode(body=decl_ast)
        elif rule == "<Question>":
            q_ast = cls._build_question(child_node)
            return SentenceNode(body=q_ast)

        return None

    @classmethod
    def _build_declarative(cls, node: Any) -> DeclarativeSentenceNode:
        """Constructs DeclarativeSentenceNode from <DeclarativeSentence> parse tree."""
        time_node: Optional[TimeNode] = None
        subject_node: Optional[SubjectNode] = None
        vp_node: Optional[VerbPhraseNode] = None
        neg_node: Optional[NegationNode] = None

        for child in node.children:
            rule = getattr(child, "rule", None)
            if rule == "<TimeMarker>":
                tok = child.children[0]
                time_node = TimeNode(value=tok.lexeme, meaning=tok.meaning)
            elif rule == "<Subject>":
                pronoun_node = child.children[0]
                tok = pronoun_node.children[0]
                subject_node = SubjectNode(value=tok.lexeme, meaning=tok.meaning)
            elif rule == "<VerbPhrase>":
                vp_node = cls._build_verb_phrase(child)
            elif rule == "<Negation>":
                tok = child.children[0]
                neg_node = NegationNode(value=tok.lexeme)

        if subject_node is None or vp_node is None:
            raise ValueError("Malformed DeclarativeSentence parse tree: missing subject or verb phrase.")

        return DeclarativeSentenceNode(
            subject=subject_node,
            verb_phrase=vp_node,
            time=time_node,
            negation=neg_node,
        )

    @classmethod
    def _build_verb_phrase(cls, node: Any) -> VerbPhraseNode:
        """Constructs VerbPhraseNode from <VerbPhrase> parse tree."""
        verb_str = ""
        verb_meaning = ""
        obj_node: Optional[ObjectNode] = None
        adj_node: Optional[AdjectiveNode] = None

        for child in node.children:
            rule = getattr(child, "rule", None)
            if rule == "<Verb>":
                tok = child.children[0]
                verb_str = tok.lexeme
                verb_meaning = tok.meaning
            elif rule == "<Object>":
                tok = child.children[0]
                obj_node = ObjectNode(value=tok.lexeme, meaning=tok.meaning)
            elif rule == "<Adjective>":
                tok = child.children[0]
                adj_node = AdjectiveNode(value=tok.lexeme, meaning=tok.meaning)

        return VerbPhraseNode(
            verb=verb_str,
            verb_meaning=verb_meaning,
            object=obj_node,
            adjective=adj_node,
        )

    @classmethod
    def _build_question(cls, node: Any) -> QuestionNode:
        """Constructs QuestionNode from <Question> parse tree."""
        wh_q_node = node.children[0]
        wh_word_node: Optional[WHWordNode] = None
        subject_node: Optional[SubjectNode] = None
        vp_node: Optional[VerbPhraseNode] = None
        obj_node: Optional[ObjectNode] = None

        for child in wh_q_node.children:
            rule = getattr(child, "rule", None)
            if rule == "<WHWord>":
                tok = child.children[0]
                wh_word_node = WHWordNode(value=tok.lexeme, meaning=tok.meaning)
            elif rule == "<Subject>":
                pronoun_node = child.children[0]
                tok = pronoun_node.children[0]
                subject_node = SubjectNode(value=tok.lexeme, meaning=tok.meaning)
            elif rule == "<VerbPhrase>":
                vp_node = cls._build_verb_phrase(child)
            elif rule == "<Object>":
                tok = child.children[0]
                obj_node = ObjectNode(value=tok.lexeme, meaning=tok.meaning)

        if wh_word_node is None:
            raise ValueError("Malformed WHQuestion parse tree: missing WHWord.")

        return QuestionNode(
            wh_word=wh_word_node,
            subject=subject_node,
            verb_phrase=vp_node,
            object=obj_node,
        )


def build_ast(parse_tree: Any) -> Optional[SentenceNode]:
    """Helper function to build an AST from a parse tree."""
    return ASTBuilder.build(parse_tree)

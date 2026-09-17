# SignSyntax Source Language Vocabulary Specification

## 1. Source Language Definition

The source language for **SignSyntax** consists of a stream of uppercase sign language **glosses** representing discrete hand gestures or signed concepts. In sign linguistics, a gloss is a conventional uppercase label approximating the meaning of a sign.

In SignSyntax:
- **Case Sensitivity:** Glosses are represented strictly using uppercase ASCII alphabetic identifiers (e.g., `ME`, `MILK`, `WANT`, `HAPPY`).
- **Delimiter:** Lexemes in the source stream are separated by one or more whitespace characters (spaces, tabs, newlines).
- **Lexical Processing:** The front-end lexical analyzer (Lexer) scans the input gloss stream and categorizes each recognized lexeme into a typed `Token`.
- **Error Handling:** Any gloss not defined in this formal vocabulary specification (e.g., `DANCE`, `RUN`, `APPLE`) is flagged as an unknown/invalid lexical item (`LexicalError`).
- **Extensibility:** The vocabulary can be expanded in later phases without altering the underlying compiler architecture.

> **Architecture Principle:**
> *"The formal glosses define the Phase 1 SignSyntax source vocabulary. The initial computer-vision recognition prototype may support a smaller subset of these glosses for reliability."*
> 
> The compiler front-end is designed around this full formal language, while the computer-vision gesture recognition pipeline can incrementally expand its supported classifier subset.

---

## 2. Supported Vocabulary (44 Core Glosses across 7 Categories)

SignSyntax Phase 1 defines the formal source vocabulary organized into **7 grammatical token categories**:

| # | Gloss | Token Type | Meaning / English Equivalent | Category Description |
| :-: | :--- | :--- | :--- | :--- |
| **Pronouns (6)** | | | | |
| 1 | `ME` | `PRONOUN` | I | First-person singular pronoun |
| 2 | `YOU` | `PRONOUN` | you | Second-person singular/plural pronoun |
| 3 | `HE` | `PRONOUN` | he | Third-person singular masculine pronoun |
| 4 | `SHE` | `PRONOUN` | she | Third-person singular feminine pronoun |
| 5 | `WE` | `PRONOUN` | we | First-person plural pronoun |
| 6 | `THEY` | `PRONOUN` | they | Third-person plural pronoun |
| **Temporal Markers (6)** | | | | |
| 7 | `YESTERDAY` | `TIME` | yesterday | Past temporal marker |
| 8 | `TODAY` | `TIME` | today | Present temporal marker |
| 9 | `TOMORROW` | `TIME` | tomorrow | Future temporal marker |
| 10 | `NOW` | `TIME` | now | Immediate present temporal marker |
| 11 | `MORNING` | `TIME` | morning | Day-part temporal marker |
| 12 | `NIGHT` | `TIME` | night | Day-part temporal marker |
| **Verbs (10)** | | | | |
| 13 | `WANT` | `VERB` | want | Modal / volition verb |
| 14 | `HAVE` | `VERB` | have | Possession verb |
| 15 | `LIKE` | `VERB` | like | Preference verb |
| 16 | `GO` | `VERB` | go | Locomotion / directional verb |
| 17 | `COME` | `VERB` | come | Locomotion / arrival verb |
| 18 | `EAT` | `VERB` | eat | Action / consumption verb |
| 19 | `DRINK` | `VERB` | drink | Action / consumption verb |
| 20 | `SEE` | `VERB` | see | Perception verb |
| 21 | `READ` | `VERB` | read | Activity verb |
| 22 | `NEED` | `VERB` | need | Requirement / volition verb |
| **Nouns / Objects (10)** | | | | |
| 23 | `MILK` | `NOUN` | milk | Concrete noun / drink entity |
| 24 | `WATER` | `NOUN` | water | Concrete noun / liquid entity |
| 25 | `FOOD` | `NOUN` | food | Concrete noun / nourishment entity |
| 26 | `BOOK` | `NOUN` | book | Concrete noun / entity object |
| 27 | `HOME` | `NOUN` | home | Location / residence noun |
| 28 | `SCHOOL` | `NOUN` | school | Location / institution noun |
| 29 | `FRIEND` | `NOUN` | friend | Person / social relation noun |
| 30 | `PHONE` | `NOUN` | phone | Concrete noun / technology object |
| 31 | `MONEY` | `NOUN` | money | Concrete noun / currency entity |
| 32 | `CAR` | `NOUN` | car | Concrete noun / vehicle object |
| **Adjectives (6)** | | | | |
| 33 | `HAPPY` | `ADJECTIVE` | happy | State / emotion adjective |
| 34 | `SAD` | `ADJECTIVE` | sad | State / emotion adjective |
| 35 | `GOOD` | `ADJECTIVE` | good | Evaluation / quality adjective |
| 36 | `BAD` | `ADJECTIVE` | bad | Evaluation / quality adjective |
| 37 | `BIG` | `ADJECTIVE` | big | Dimension / size adjective |
| 38 | `SMALL` | `ADJECTIVE` | small | Dimension / size adjective |
| **WH-Words / Interrogatives (5)** | | | | |
| 39 | `WHAT` | `WH_WORD` | what | Interrogative inquiry pronoun |
| 40 | `WHERE` | `WH_WORD` | where | Interrogative locative adverb |
| 41 | `WHO` | `WH_WORD` | who | Interrogative person pronoun |
| 42 | `WHEN` | `WH_WORD` | when | Interrogative temporal adverb |
| 43 | `WHY` | `WH_WORD` | why | Interrogative causal adverb |
| **Negation (1)** | | | | |
| 44 | `NOT` | `NEGATION` | not | Logical negation particle |

---

## 3. Token Categories Summary

1. **`PRONOUN`** (6): Discourse participants acting as grammatical subjects (`ME`, `YOU`, `HE`, `SHE`, `WE`, `THEY`).
2. **`TIME`** (6): Temporal adverbs establishing the tense/temporal frame (`YESTERDAY`, `TODAY`, `TOMORROW`, `NOW`, `MORNING`, `NIGHT`).
3. **`VERB`** (10): Core predicates, actions, states, and relations (`WANT`, `HAVE`, `LIKE`, `GO`, `COME`, `EAT`, `DRINK`, `SEE`, `READ`, `NEED`).
4. **`NOUN`** (10): Entities acting as direct objects or topic arguments (`MILK`, `WATER`, `FOOD`, `BOOK`, `HOME`, `SCHOOL`, `FRIEND`, `PHONE`, `MONEY`, `CAR`).
5. **`ADJECTIVE`** (6): Modifiers expressing qualities, states, or dimensions (`HAPPY`, `SAD`, `GOOD`, `BAD`, `BIG`, `SMALL`).
6. **`WH_WORD`** (5): Interrogative question markers (`WHAT`, `WHERE`, `WHO`, `WHEN`, `WHY`).
7. **`NEGATION`** (1): Modifiers denoting denial or negation of the predicate (`NOT`).

---

## 4. Source Language Sequence Examples

### Lexical Validity vs. Grammatical Validity

In compiler construction, **lexical validity** and **grammatical (syntactic) validity** are distinct phases:
- **Lexically Valid:** Every individual lexeme exists in the formal Symbol Table vocabulary and adheres to uppercase casing rules.
- **Grammatically Valid:** The sequence of tokens satisfies the formal BNF production rules of the grammar.

### Examples:

#### ✅ Valid Source Sequences (Lexically & Grammatically Valid)
- `ME WANT MILK` *(Subject + Verb + Object)*
- `YOU LIKE BOOK` *(Subject + Verb + Object)*
- `YESTERDAY ME READ BOOK` *(Time + Subject + Verb + Object)*
- `TOMORROW WE GO SCHOOL` *(Time + Subject + Verb + Object)*
- `ME DRINK WATER` *(Subject + Verb + Object)*
- `SHE HAVE PHONE` *(Subject + Verb + Object)*
- `ME NOT WANT FOOD` *(Subject + Negation + Verb + Object)*
- `WHAT YOU WANT` *(WH-Question: WHWord + Subject + Verb)*
- `WHERE HOME` *(WH-Question: WHWord + Object)*
- `WHO YOU SEE` *(WH-Question: WHWord + Subject + Verb)*
- `WHEN YOU GO` *(WH-Question: WHWord + Subject + Verb)*

#### ❌ Lexically Invalid Sequences (Contains Unknown Glosses)
- `ME DANCE MILK` $\rightarrow$ Rejected at Lexical Analysis (`DANCE` is not in vocabulary).
- `ME RUN WATER` $\rightarrow$ Rejected at Lexical Analysis (`RUN` is not in vocabulary).
- `ME WANT UNKNOWNWORD` $\rightarrow$ Rejected at Lexical Analysis (`UNKNOWNWORD` is unknown).
- `me want milk` $\rightarrow$ Rejected at Lexical Analysis (lowercase is invalid).

#### ⚠️ Lexically Valid but Syntactically Invalid Sequences
- `MILK ME WANT` $\rightarrow$ Lexically valid (all 3 words exist in vocabulary), but violates the BNF sentence structure rule `<DeclarativeSentence> ::= <TimeMarker>? <Subject> <VerbPhrase> <Negation>?`.

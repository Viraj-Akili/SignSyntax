# SignSyntax Visual Frontend & Demonstration Dashboard

This directory contains the Streamlit-based visual presentation layer for the **SignSyntax Compiler Design** project (Phase 2 Review 2).

---

## 1. Overview & Architecture

The frontend is strictly a **presentation and demonstration layer**. It contains **zero duplicate compiler logic** and directly queries the authoritative compiler entrypoint:

```python
from compiler.frontend import compile_frontend

# Calls the complete 5-stage compiler pipeline
result = compile_frontend(source_text)
```

### Compiler Stages Visualized:
1. **Symbol Table (Stage 0)**: Authoritative 44-sign vocabulary categories and meanings.
2. **Lexical Analysis (Stage 1)**: Token stream with positions, token types, and meanings.
3. **Syntax Analysis (Stage 2)**: LL(1) recursive-descent verification & Concrete Parse Tree.
4. **AST Construction (Stage 3)**: Hierarchical Abstract Syntax Tree.
5. **Semantic Analysis (Stage 4)**: Category type-checking & normalized `SemanticForm`.
6. **Intermediate Representation (Stage 5)**: Target-independent `IRProgram` instruction stream.

---

## 2. Launching the Application

Run the application using Streamlit:

```bash
streamlit run frontend/app.py
```

The dashboard will launch in your default web browser at `http://localhost:8501`.

---

## 3. Features & Capabilities

- **Interactive Gloss Compiler**: Enter any sign gloss sequence or select from pre-configured canonical presets.
- **Dynamic Phase Isolation Indicators**: Clear status cards showing `PASSED`, `FAILED`, or `NOT EXECUTED` across all 6 compiler stages.
- **Stage-by-Stage Inspector**:
  - Interactive Token Table with lexical positions.
  - Expandable Concrete Parse Tree view.
  - Formatted Abstract Syntax Tree view.
  - Normalized Semantic Form key-value display.
  - Linearized Intermediate Representation (IR) instruction listing.
- **Symbol Table Explorer**: Search and inspect all 44 recognized glosses.
- **Computer Vision & Recognition Tab**: Explains landmark extraction, 252D temporal feature vector, and the boundary interface between vision and the compiler.
- **System Architecture Tab**: Full architectural diagram showing input acquisition, compiler frontend, IR, and Phase 3 roadmap.

# SignSyntax: Compiler-Based Sign Language Translation System

**SignSyntax** is a Compiler Design academic project that translates a restricted sign language gloss representation into well-formed English sentences using classic compiler design principles and modern computer vision techniques.

---

## 🏗️ System Architecture Overview

The end-to-end vision for SignSyntax follows a structured compiler pipeline:

```
Sign Language
    ↓
Webcam Input
    ↓
MediaPipe Hands
    ↓
Hand Landmarks
    ↓
Sign Recognition
    ↓
Gloss Tokens
    ↓
Lexical Analyzer (Lexer)
    ↓
Symbol Table
    ↓
Parser (Syntax Analysis)
    ↓
Semantic Analyzer
    ↓
Intermediate Representation (IR)
    ↓
Code Generator
    ↓
English Output
```

---

## 🎯 Phase 1 Focus

In **Phase 1**, the primary goal is establishing the project foundation and implementing the early lexical analysis pipeline:

```
Sign Language / Manual Gloss Input
            ↓
       Gloss Stream
            ↓
          LEXER
            ↓
       Token Stream
            ↑
            │
      SYMBOL TABLE
```

---

## 🔤 Source Language

SignSyntax operates on a formal **sign language gloss representation** as its source language. Each gloss is an uppercase identifier representing a discrete sign gesture (e.g., `ME`, `WANT`, `MILK`, `HAPPY`, `YESTERDAY`).

In **Phase 1**, the formal source language defines a vocabulary of **40 glosses** organized into **7 grammatical token categories** (`PRONOUN`, `TIME`, `VERB`, `NOUN`, `ADJECTIVE`, `WH_WORD`, `NEGATION`), along with a formal **Backus-Naur Form (BNF)** grammar specification.

> **Compiler Architecture Note:**
> The formal SignSyntax source vocabulary represents the full language targeted by the compiler pipeline. The initial computer-vision recognition prototype (OpenCV + MediaPipe) may support a smaller, high-reliability subset of these signs during early testing without requiring compiler redesign.

For detailed specifications, see:
- [`grammar/vocabulary.md`](grammar/vocabulary.md) — 40-gloss formal vocabulary, token categories, and lexical rules.
- [`grammar/grammar.bnf`](grammar/grammar.bnf) — Formal BNF grammar specification for declarative and interrogative sentences.

---

## 🗃️ Symbol Table

In compiler construction, the **Symbol Table** is a foundational data structure that acts as the authoritative lexical vocabulary repository for all source language symbols.

### Why the Compiler Needs It:
- **Lexeme Validation:** Stores valid vocabulary items and prevents unauthorized or misspelled tokens from entering subsequent compiler phases.
- **Attribute Association:** Associates each gloss lexeme with its syntactic token category (e.g., `PRONOUN`, `TIME`, `VERB`, `NOUN`, `ADJECTIVE`, `WH_WORD`, `NEGATION`) and its semantic English translation.
- **Fast Disambiguation:** Allows the Lexical Analyzer (Lexer) to perform constant-time $O(1)$ lookups when scanning input tokens.

In SignSyntax Phase 1:
- The symbol table is initialized with the **40 supported formal glosses**.
- It enforces **strict case-sensitivity** (uppercase glosses only).
- It serves as the single source of truth for the Lexer during lexical analysis.

---

## 🔍 Lexical Analysis

The **Lexical Analyzer (Lexer)** is the first operational compiler phase in SignSyntax. It transforms an unstructured string of sign glosses into a discrete, typed **Token Stream**.

### Lexer Responsibilities:
1. **Lexeme Scanning:** Reads raw input and splits glosses using whitespace delimiters.
2. **Symbol Table Lookup:** Queries the central `SymbolTable` for each lexeme to verify validity and extract token attributes.
3. **Token Creation:** Emits strongly typed `Token` objects containing `token_type`, `lexeme`, `meaning`, and `position`.
4. **Lexical Error Reporting:** Identifies invalid, misspelled, or incorrectly cased glosses (e.g., `DANCE`, `me`), generating detailed `LexicalError` objects while continuing to process remaining lexemes.

### Example Transformation:

```
Input:   YESTERDAY ME WANT MILK
           ↓         ↓    ↓    ↓
Tokens:  TIME    PRONOUN VERB NOUN
```

> **Note:** The Lexer only performs lexical scanning and token categorization. It does **not** perform syntax analysis (parsing grammar rules) or semantic analysis.

---

## 📷 Phase 1 Computer Vision Input

The computer-vision subsystem serves as the physical input and feature extraction layer for SignSyntax:

```
Webcam Frame
    ↓
MediaPipe Hands Detector
    ↓
21 3D Hand Landmarks (x, y, z) per hand (up to 2 hands)
    ↓
Structured Landmark Data Point Stream
```

### Architectural Separation:
* **Feature Extraction vs. Lexical Analysis:** MediaPipe Hands operates purely as an **input and spatial coordinate extraction mechanism**. It does **not** perform sign classification, nor does it perform lexical tokenization.
* **Compiler Pipeline Integration:** The extracted landmarks provide the raw geometric features that a future gesture classifier will map to glosses. The generated glosses then enter the compiler pipeline:

```
Hand / Gesture Input
    ↓
Gloss Generation (Future Classifier)
    ↓
Symbol Table
    ↓
Lexer
    ↓
Token Stream
```

---

## 📊 Phase 1 Dataset Collection

SignSyntax provides a specialized landmark dataset collector for recording labeled training samples directly into structured CSV files:

```
Webcam Frame
    ↓
MediaPipe Hands (1 Hand Detected)
    ↓
21 3D Landmarks (x, y, z)
    ↓
Validated Gloss Label (Symbol Table check)
    ↓
CSV Dataset Record (dataset/<GLOSS>/samples.csv)
```

### Dataset Schema (64 Columns):
```
label,x0,y0,z0,x1,y1,z1,...,x20,y20,z20
```

### Collector Operation:
- **Strict Validation:** Only accepts labels defined in the formal 44-gloss `SymbolTable` (e.g., `ME`, `YOU`, `WANT`, `MILK`). Rejects invalid labels like `DANCE`.
- **Quality Checks:** Requires exactly 1 hand in frame before persisting a sample.
- **Purpose:** Collects raw spatial data for future classifier training without implementing ML models in Phase 1.

To run the interactive collector:
```bash
# Launch with prompt
python recognition/dataset_collector.py

# Or pass gloss label directly
python recognition/dataset_collector.py ME
```

---

## 📹 WLASL Video Dataset Integration

To scale beyond manual single-frame sample collection, SignSyntax integrates a subset of the public **WLASL (Word-Level American Sign Language)** video dataset for temporal, multi-hand landmark extraction.

### Architectural Highlights:
- **Temporal Sequences:** Rather than a static frame, each sample represents an ordered sequence of $T$ video frames.
- **Multi-Hand Support:** Extracts up to two hands per frame using MediaPipe Hands.
- **Consistent Feature Representation:** Exactly **126 float features per frame**:
  - **Slot 0 (Indices 0..62):** Left Hand (21 landmarks $\times$ 3 coords $x, y, z$). Padded with 63 zeros (`0.0`) if absent.
  - **Slot 1 (Indices 63..125):** Right Hand (21 landmarks $\times$ 3 coords $x, y, z$). Padded with 63 zeros (`0.0`) if absent.
- **Decoupled Architecture:** WLASL video processing and sequence storage (`dataset/wlasl_sequences/`) are completely decoupled from the manual single-frame CSV collector (`dataset/<GLOSS>/samples.csv`).
- **No Phase 2 Scope Creep:** Only handles video sequence parsing and landmark feature extraction. No machine learning classifiers (k-NN, SVM, neural networks) are trained in Phase 1.

### Target Prototype Gloss Availability in WLASL:
Running the metadata inspector reveals that all 7 target prototype glosses are present in WLASL v0.3:

| SignSyntax Gloss | WLASL Gloss Key | Available Samples |
|:-----------------|:----------------|:------------------|
| `ME`             | `'me'`          | 9                 |
| `YOU`            | `'you'`         | 16                |
| `WANT`           | `'want'`        | 19                |
| `MILK`           | `'milk'`        | 14                |
| `BOOK`           | `'book'`        | 40                |
| `HOME`           | `'home'`        | 16                |
| `NOT`            | `'not'`         | 14                |
| **Total**        |                 | **128 samples**   |

### WLASL CLI Tools:

```bash
# 1. Inspect metadata availability for target glosses (or specific glosses)
python recognition/wlasl_inspector.py
python recognition/wlasl_inspector.py BOOK MILK WANT

# 2. Download and process the prototype subset (up to 10 videos per gloss)
python recognition/wlasl_downloader.py --max-samples 10

# 3. Process an individual video into a 126-dim temporal landmark sequence JSON
python recognition/wlasl_processor.py --video path/to/sample.mp4 --gloss BOOK
```

### Dataset License & Citation:
The WLASL dataset is distributed under the **Computational Use of Data Agreement (C-UDA)**.
Reference:
> Dongxu Li, Cristian Rodriguez, Xin Yu, and Hongdong Li. *"Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison"*, Winter Conference on Applications of Computer Vision (WACV), 2020.
> Repository: [https://github.com/dxli94/WLASL](https://github.com/dxli94/WLASL)

---

## 🤖 Phase 1 Explainable Gesture Recognition (k-NN)

SignSyntax Phase 1 includes a simple, explainable **k-Nearest Neighbors (k-NN)** gesture classification pipeline to connect the vision input subsystem directly to the compiler frontend.

### 1. Statistical Feature Extraction (252 Features)
Because video sequences have variable duration ($T$ frames), temporal sequences are aggregated into a fixed-length feature vector using transparent statistical moments:
* **Features 0..125 (126 features):** Temporal Mean $\mu = \frac{1}{T}\sum_{t=1}^T \mathbf{x}_t$ for each coordinate.
* **Features 126..251 (126 features):** Temporal Standard Deviation $\sigma = \sqrt{\frac{1}{T}\sum_{t=1}^T (\mathbf{x}_t - \mu)^2}$.
* **Total Dimensionality:** **252 features** per video sequence sample.

### 2. Explainable k-NN Architecture:
* **Distance Metric:** Euclidean distance ($L_2$ norm) over standardized features (`StandardScaler`).
* **Inference:** Top-$k$ nearest neighbor voting with confidence scoring ($k=3$).
* **Leakage Prevention:** Stratified train/test splitting operates strictly at the sequence/video level (entire video in train or test).
* **Viva Transparency:** 100% explainable mathematical formulation suitable for Compiler Design oral examinations.

### 3. Compiler Integration Flow:
```
Landmark Sequence (T x 126)
       ↓
Feature Extractor (252-dim Mean + Std)
       ↓
Gesture Classifier (k-NN)
       ↓
Predicted Gloss (e.g. "YOU")
       ↓
Symbol Table Lookup (Case-sensitive check & attribute extraction)
       ↓
Lexical Analyzer (Lexer)
       ↓
Token Stream: [Token(type='PRONOUN', lexeme='YOU', meaning='you', position=0)]
```

### 4. Vocabulary Scope & Academic Prototype Boundaries:
* **Formal Compiler Vocabulary (44 Glosses):** The complete language targeted by the grammar and symbol table.
* **Recognition Prototype Subset (7 Glosses):** `ME`, `YOU`, `WANT`, `MILK`, `BOOK`, `HOME`, `NOT`.
* **Prototype Disclaimer:** This milestone is an academic demonstration of compiler frontend integration with computer vision features. It does not claim full commercial sign language translation coverage.

### 5. CLI Execution Commands:

```bash
# 1. Train the explainable gesture classifier on the WLASL sequence dataset
python recognition/gesture_classifier.py --k 3

# 2. Run prediction on a single landmark sequence JSON
python recognition/predict_gesture.py --sequence dataset/wlasl_sequences/BOOK/69241.json

# 3. Run the complete end-to-end Phase 1 demonstration pipeline
python recognition/phase1_demo.py
```

---

## 📁 Project Structure

```
SignSyntax/
├── compiler/                   # Core compiler pipeline
│   ├── __init__.py
│   ├── lexer.py                # Hand-written Lexical Analyzer & LexicalError
│   ├── symbol_table.py         # 44-gloss Symbol Table and Symbol entry classes
│   └── tokens.py               # Token dataclass representation
├── recognition/                # Vision input and dataset modules
│   ├── __init__.py
│   ├── dataset_collector.py    # Manual single-frame landmark CSV collector
│   ├── gesture_classifier.py   # Explainable k-NN trainer & feature extractor (252-dim)
│   ├── hand_tracker.py         # MediaPipe Hands tracking & landmark extraction
│   ├── phase1_demo.py          # End-to-end demo (Vision -> k-NN -> SymbolTable -> Lexer)
│   ├── predict_gesture.py      # Gesture inference engine with Symbol Table lookup
│   ├── wlasl_downloader.py     # WLASL subset downloader and pipeline driver
│   ├── wlasl_inspector.py      # WLASL metadata inspection & gloss verification
│   └── wlasl_processor.py      # WLASL video processing & temporal sequence pipeline
├── grammar/                    # Formal grammar specifications and vocabulary definitions
│   ├── grammar.bnf             # BNF grammar rules
│   └── vocabulary.md           # 44-gloss source vocabulary and token mappings
├── dataset/                    # Recorded landmark samples & sequences
│   ├── ME/samples.csv          # Manually collected single-frame samples
│   ├── YOU/samples.csv         # Manually collected single-frame samples
│   ├── WLASL_v0.3.json         # Cached WLASL dataset metadata
│   ├── wlasl_raw/              # Downloaded subset video files (MP4/MOV)
│   └── wlasl_sequences/        # Extracted temporal multi-hand JSON sequences
├── models/                     # Serialized gesture recognition models
│   ├── gesture_knn.json        # Trained k-NN weights, classes, and scaler parameters
│   └── gesture_knn.pkl         # Pickle serialized model
├── tests/                      # Comprehensive unit test suite (82 tests)
│   ├── __init__.py
│   ├── demo_hand_tracker.py    # Live webcam 21-landmark tracking demonstration
│   ├── demo_lexer.py           # Interactive CLI demonstration for Lexer
│   ├── test_dataset_collector.py # Unit tests for manual dataset collector
│   ├── test_gesture_classifier.py # Unit tests for k-NN, 252-dim features & integration
│   ├── test_hand_tracker.py    # Unit tests for hand tracker & landmark data
│   ├── test_lexer.py           # Unit tests for Lexer
│   ├── test_symbol_table.py    # Unit tests for Symbol Table
│   ├── test_wlasl_downloader.py # Unit tests for WLASL subset downloader
│   └── test_wlasl_processor.py # Unit tests for WLASL inspection & video processor
├── main.py                     # Application entry point
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Test Suite

```bash
python -m pytest -v
```

### 4. Run the Phase 1 End-to-End Demo

```bash
python recognition/phase1_demo.py
```

### 5. Run the Application Entry Point

```bash
python main.py
```



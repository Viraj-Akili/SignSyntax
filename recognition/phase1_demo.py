"""
SignSyntax - Phase 1 End-to-End Demonstration Script
Compiler Design Academic Project: Translating Sign Language Gloss into English

This script demonstrates the complete, working Phase 1 pipeline:
1. Input: Temporal Multi-Hand Landmark Sequences (WLASL dataset sequences)
2. Feature Extraction: 252-dimensional Statistical Features (Mean + Std per landmark coordinate)
3. Vision Recognition: Explainable k-NN Gesture Classifier
4. Compiler Validation: Authoritative Symbol Table Lookup
5. Lexical Analysis: Hand-written Lexer Tokenization

Architecture Demonstrated:
   Landmark Sequence
          ↓
   Gesture Classifier (k-NN)
          ↓
   Predicted Gloss (e.g., "YOU")
          ↓
   Symbol Table Lookup
          ↓
   Lexical Analyzer (Lexer)
          ↓
   Token Stream: [Token(type=PRONOUN, lexeme='YOU', meaning='you', pos=0)]

Phase 1 Boundary Notice:
This is strictly the Phase 1 milestone. No parser, AST, semantic analyzer, IR, code generator,
or TTS is implemented.
"""

import glob
import os
import sys
from typing import List, Optional

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.lexer import Lexer
from compiler.symbol_table import SymbolTable
from recognition.gesture_classifier import DEFAULT_MODEL_PATH
from recognition.predict_gesture import GestureIntegrationError, GesturePredictor
from recognition.wlasl_processor import DEFAULT_SEQUENCES_DIR


def run_phase1_demo(
    sample_sequence_paths: Optional[List[str]] = None,
    model_path: str = DEFAULT_MODEL_PATH,
) -> None:
    """
    Executes the complete end-to-end Phase 1 demonstration.

    Args:
        sample_sequence_paths (Optional[List[str]]): List of sequence JSON paths to test.
        model_path (str): Path to trained gesture classifier.
    """
    print("=" * 80)
    print("           SignSyntax: Phase 1 End-to-End Demonstration Pipeline           ")
    print("           (Vision Input -> Gesture k-NN -> Symbol Table -> Lexer)          ")
    print("=" * 80)

    # 1. Initialize components
    print("\n[STEP 1] Initializing Compiler and Vision Components...")
    st = SymbolTable()
    print(f"  - Symbol Table loaded with {st.size()} formal language symbols.")

    lexer = Lexer(st)
    print("  - Hand-written Lexer initialized.")

    predictor = GesturePredictor(model_path=model_path, symbol_table=st)
    print(f"  - Gesture Classifier loaded from '{model_path}'.")

    # 2. Find sample sequences if none provided
    if not sample_sequence_paths:
        sample_sequence_paths = []
        for gloss in ["YOU", "WANT", "MILK", "BOOK", "HOME", "NOT", "ME"]:
            pattern = os.path.join(DEFAULT_SEQUENCES_DIR, gloss, "*.json")
            files = glob.glob(pattern)
            if files:
                sample_sequence_paths.append(files[0])

    if not sample_sequence_paths:
        print("\nNo sequence JSON files found in dataset/wlasl_sequences/. Run download & processing first.")
        return

    print(f"\n[STEP 2] Running Recognition & Lexical Pipeline on {len(sample_sequence_paths)} Video Sequences:\n")

    recognized_glosses: List[str] = []

    for idx, seq_path in enumerate(sample_sequence_paths, 1):
        filename = os.path.basename(seq_path)
        parent_dir = os.path.basename(os.path.dirname(seq_path))
        print("-" * 80)
        print(f"Test Sample #{idx}: [{parent_dir}/{filename}]")

        # Step 2a: Predict gesture
        result = predictor.predict_sequence(seq_path)
        print(f"  [Vision Layer]   Input Frames    : {result.frame_count} frames (126 features/frame)")
        print(f"  [Vision Layer]   Extracted Vector: 252 statistical features (mean + std)")
        print(f"  [Classifier]     Predicted Gloss : {result.predicted_gloss} (Confidence: {result.confidence * 100:.1f}%)")

        # Step 2b: Symbol Table check
        print(f"  [Symbol Table]   Lookup Result   : Valid symbol found")
        print(f"                   Token Category  : {result.token_type}")
        print(f"                   English Meaning : {result.meaning}")

        # Step 2c: Lexer Tokenization
        lexer_res = lexer.tokenize(result.predicted_gloss)
        if lexer_res.tokens:
            token = lexer_res.tokens[0]
            print(f"  [Lexical Layer]  Lexer Token     : Token(type='{token.token_type}', lexeme='{token.lexeme}', meaning='{token.meaning}', pos={token.position})")
        else:
            print(f"  [Lexical Layer]  Lexer Error     : {lexer_res.errors}")

        recognized_glosses.append(result.predicted_gloss)

    # 3. Demonstrate sentence chaining through Lexer
    print("\n" + "=" * 80)
    print("      [STEP 3] Composing Multi-Gesture Token Stream Through Lexer       ")
    print("=" * 80)

    composed_sentence = " ".join(recognized_glosses[:3]) if len(recognized_glosses) >= 3 else " ".join(recognized_glosses)
    print(f"Composed Gloss Sequence : {composed_sentence}")

    result_stream = lexer.tokenize(composed_sentence)
    print(f"\nLexer Token Stream Output ({len(result_stream.tokens)} tokens):")
    for tok in result_stream.tokens:
        print(f"  -> [{tok.token_type:<10}] lexeme='{tok.lexeme:<6}' meaning='{tok.meaning}' (pos={tok.position})")

    if result_stream.has_errors:
        print(f"Lexical Errors: {result_stream.errors}")
    else:
        print("\nAll tokens verified valid in Symbol Table. Ready for Phase 2 Syntax Parsing.")

    print("=" * 80)


def main() -> None:
    """CLI entrypoint for running Phase 1 demo."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SignSyntax Phase 1: End-to-End Gesture to Lexer Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Optional paths to specific sequence JSON files to test",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained classifier model (default: models/gesture_knn.json)",
    )

    args = parser.parse_args()
    run_phase1_demo(sample_sequence_paths=args.sequences, model_path=args.model)


if __name__ == "__main__":
    main()

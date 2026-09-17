"""
SignSyntax - Gesture Prediction & Symbol Table Integration Module
Phase 1: Computer Vision Input & Gesture Inference

This module loads the trained k-NN gesture classifier, extracts 252-dimensional statistical
features from a temporal landmark sequence, predicts the sign gloss, validates the gloss
against the authoritative SignSyntax Symbol Table, and associates compiler token attributes.

Key Guarantees:
1. Deterministic Inference: Extracts mean and std across time matching training feature space.
2. Symbol Table Verification: Validates predicted gloss against the 44-gloss SymbolTable.
3. Transparent Integration Error: Explicitly reports if a predicted label is outside the Symbol Table.
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.symbol_table import SymbolTable
from recognition.gesture_classifier import (
    DEFAULT_MODEL_PATH,
    KNearestNeighborsClassifier,
    extract_sequence_features,
)
from recognition.wlasl_processor import LandmarkSequenceSample


class GestureIntegrationError(Exception):
    """Raised when a predicted gesture gloss fails validation against the Symbol Table."""
    pass


@dataclass
class GesturePredictionResult:
    """
    Structured outcome of gesture classification and Symbol Table lookup.

    Attributes:
        predicted_gloss (str): Recognized uppercase gloss (e.g. 'ME', 'YOU', 'WANT').
        confidence (float): Classification confidence score (0.0 to 1.0).
        token_type (str): Syntactic token category from Symbol Table (e.g. 'PRONOUN', 'VERB').
        meaning (str): English semantic meaning from Symbol Table (e.g. 'I / me', 'want / desire').
        nearest_neighbors (List[Tuple[str, float]]): Top-k nearest neighbors with distances.
        source_video_id (str): Video ID if available.
        frame_count (int): Number of frames in the input sequence.
    """
    predicted_gloss: str
    confidence: float
    token_type: str
    meaning: str
    nearest_neighbors: List[Tuple[str, float]]
    source_video_id: str = ""
    frame_count: int = 0

    def display(self) -> str:
        """Formats prediction result as an informative ASCII summary."""
        lines = [
            "=" * 65,
            "         SignSyntax: Phase 1 Gesture Prediction Result          ",
            "=" * 65,
            f"Predicted Gloss : {self.predicted_gloss}",
            f"Confidence Score: {self.confidence * 100:.1f}%",
            f"Token Category  : {self.token_type}",
            f"English Meaning : {self.meaning}",
            f"Source Video ID : {self.source_video_id or 'N/A'} ({self.frame_count} frames)",
            "-" * 65,
            "Top-k Nearest Neighbors (Class, Distance):",
        ]
        for idx, (lbl, dist) in enumerate(self.nearest_neighbors, 1):
            lines.append(f"  {idx}. {lbl:<8} (Euclidean dist: {dist:.4f})")
        lines.append("=" * 65)
        return "\n".join(lines)


class GesturePredictor:
    """
    Inference engine for predicting sign glosses from temporal landmark sequences.

    Attributes:
        model_path (str): Path to saved gesture classifier model.
        symbol_table (SymbolTable): Authoritative SignSyntax Symbol Table.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        symbol_table: Optional[SymbolTable] = None,
    ) -> None:
        self.model_path = model_path
        self.symbol_table = symbol_table if symbol_table is not None else SymbolTable()
        self._model: Optional[KNearestNeighborsClassifier] = None
        self._load_model()

    def _load_model(self) -> None:
        """Loads classifier from disk."""
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"Trained gesture model not found at '{self.model_path}'. "
                f"Please run 'python recognition/gesture_classifier.py' to train the model."
            )
        self._model = KNearestNeighborsClassifier.load(self.model_path)

    def predict_sequence(
        self,
        sequence: Union[str, Dict[str, Any], LandmarkSequenceSample, List[List[float]], np.ndarray],
    ) -> GesturePredictionResult:
        """
        Predicts sign gloss from a sequence JSON path, dictionary, LandmarkSequenceSample, or frame list.

        Args:
            sequence: Input temporal sequence in any supported representation.

        Returns:
            GesturePredictionResult: Full prediction and Symbol Table metadata.

        Raises:
            GestureIntegrationError: If predicted gloss is not in the Symbol Table.
        """
        if self._model is None:
            self._load_model()

        video_id = ""
        frame_count = 0

        # Parse input into raw frame vectors
        if isinstance(sequence, str):
            # File path
            sample = LandmarkSequenceSample.load_json(sequence)
            frames = sample.frames
            video_id = sample.video_id
            frame_count = sample.frame_count
        elif isinstance(sequence, dict):
            sample = LandmarkSequenceSample.from_dict(sequence)
            frames = sample.frames
            video_id = sample.video_id
            frame_count = sample.frame_count
        elif isinstance(sequence, LandmarkSequenceSample):
            frames = sequence.frames
            video_id = sequence.video_id
            frame_count = sequence.frame_count
        else:
            frames = sequence
            frame_count = len(frames)

        # Step 1: Extract 252-dimensional features
        features = extract_sequence_features(frames)

        # Step 2: k-NN inference
        predicted_gloss, confidence, neighbors = self._model.predict_single(features)

        # Step 3: Validate against authoritative Symbol Table
        symbol = self.symbol_table.lookup(predicted_gloss)
        if symbol is None:
            raise GestureIntegrationError(
                f"Integration Error: Predicted gloss '{predicted_gloss}' is not defined "
                f"in the authoritative 44-gloss SignSyntax Symbol Table."
            )

        return GesturePredictionResult(
            predicted_gloss=predicted_gloss,
            confidence=confidence,
            token_type=symbol.token_type,
            meaning=symbol.meaning,
            nearest_neighbors=neighbors,
            source_video_id=video_id,
            frame_count=frame_count,
        )


def main() -> None:
    """CLI entry point for predicting sign gloss from a sequence JSON file."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SignSyntax Phase 1: Predict Sign Gloss from Landmark Sequence JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help="Path to processed landmark sequence JSON file (e.g. dataset/wlasl_sequences/ME/35544.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained classifier model (default: models/gesture_knn.json)",
    )

    args = parser.parse_args()

    predictor = GesturePredictor(model_path=args.model)
    result = predictor.predict_sequence(args.sequence)
    print("\n" + result.display())


if __name__ == "__main__":
    main()

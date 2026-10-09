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
from dataclasses import dataclass, field
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
        predicted_gloss (str): Recognized uppercase gloss (e.g. 'ME', 'YOU', 'BOOK', etc.).
        confidence (float): Classification confidence score (0.0 to 1.0).
        token_type (str): Syntactic token category from Symbol Table.
        meaning (str): English semantic meaning from Symbol Table.
        nearest_neighbors (List[Tuple[str, float]]): Top-k nearest neighbors with distances.
        source_video_id (str): Video ID if available.
        frame_count (int): Number of frames in the input sequence.
        features (Optional[np.ndarray]): Extracted 252-dim feature vector.
        vote_counts (Dict[str, int]): Distribution of k-NN neighbor votes across classes.
        is_recognized (bool): Always True in closed-set recognition.
        rejection_reason (Optional[str]): None in closed-set recognition.
    """
    predicted_gloss: str
    confidence: float
    token_type: str
    meaning: str
    nearest_neighbors: List[Tuple[str, float]]
    source_video_id: str = ""
    frame_count: int = 0
    features: Optional[np.ndarray] = None
    vote_counts: Dict[str, int] = field(default_factory=dict)
    is_recognized: bool = True
    rejection_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.vote_counts is None:
            self.vote_counts = {}

    def __getattr__(self, name: str) -> Any:
        # Fallback for older deserialized or cached instances
        if name == "vote_counts":
            return {}
        if name == "is_recognized":
            return True
        if name == "rejection_reason":
            return None
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

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
        ]
        if self.features is not None:
            feat_arr = np.asarray(self.features)
            lines.extend([
                f"Feature Dim     : {feat_arr.shape}",
                f"Feature Stats   : min={feat_arr.min():.4f}, max={feat_arr.max():.4f}, mean={feat_arr.mean():.4f}, std={feat_arr.std():.4f}",
            ])
        if self.vote_counts:
            lines.append(f"k-NN Votes      : {dict(self.vote_counts)}")
        lines.extend([
            "-" * 65,
            "Top-k Nearest Neighbors (Class, Distance):",
        ])
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
        distance_threshold: Optional[float] = None,
    ) -> None:
        self.model_path = model_path
        self.symbol_table = symbol_table if symbol_table is not None else SymbolTable()
        self.distance_threshold = distance_threshold
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

        # Step 2: k-NN inference (closed-set: majority vote among top-k neighbors)
        predicted_gloss, confidence, neighbors = self._model.predict_single(features)

        # Compute k-NN vote counts
        vote_counts: Dict[str, int] = {}
        for lbl, _ in neighbors:
            vote_counts[lbl] = vote_counts.get(lbl, 0) + 1

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
            features=features,
            vote_counts=vote_counts,
            is_recognized=True,
            rejection_reason=None,
        )

    def predict_single(
        self,
        feature_vector: Union[List[float], np.ndarray],
    ) -> GesturePredictionResult:
        """
        Predicts sign gloss directly from a pre-extracted 252-dimensional feature vector.

        Args:
            feature_vector: 252-dimensional feature vector (mean + std over time).

        Returns:
            GesturePredictionResult: Full prediction, vote counts, and Symbol Table metadata.

        Raises:
            GestureIntegrationError: If predicted gloss is not in the Symbol Table.
        """
        if self._model is None:
            self._load_model()

        feat_arr = np.asarray(feature_vector, dtype=np.float32)
        predicted_gloss, confidence, neighbors = self._model.predict_single(feat_arr)

        vote_counts: Dict[str, int] = {}
        for lbl, _ in neighbors:
            vote_counts[lbl] = vote_counts.get(lbl, 0) + 1

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
            source_video_id="",
            frame_count=0,
            features=feat_arr,
            vote_counts=vote_counts,
            is_recognized=True,
            rejection_reason=None,
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

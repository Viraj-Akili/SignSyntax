"""
SignSyntax - Gesture Classifier Module
Phase 1: Computer Vision Input & Gesture Classification

This module implements a simple, explainable k-Nearest Neighbors (k-NN) classifier for
sign gesture recognition on temporal hand-landmark sequences extracted from WLASL.

Key Architectural Guarantees:
1. Fixed-Length Feature Extraction:
   - Aggregates variable-length temporal sequences (T x 126) into exactly 252 features:
     * Mean of each of the 126 landmark coordinate features across all frames (126 features)
     * Standard deviation of each of the 126 features across all frames (126 features)
     * Total dimensionality: 126 + 126 = 252 features per video sequence.
2. Leakage Prevention:
   - Train/test splitting operates strictly at the sequence/video level (entire video in train or test).
3. Explainable k-NN:
   - Uses Euclidean distance and majority voting with confidence scores.
   - 100% transparent and explainable for Compiler Design viva presentations.
4. Model Persistence:
   - Saves trained model, scaler parameters, and metadata under models/.

Citation & Attribution:
WLASL is provided under the Computational Use of Data Agreement (C-UDA).
Reference: Dongxu Li et al., WACV 2020. https://github.com/dxli94/WLASL
"""

import glob
import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.symbol_table import SymbolTable
from recognition.wlasl_inspector import DEFAULT_PROTOTYPE_GLOSSES
from recognition.wlasl_processor import FRAME_FEATURE_DIM, LandmarkSequenceSample

# Constants
FEATURE_DIM_PER_FRAME = 126  # 2 hands x 21 landmarks x 3 coords
AGGREGATED_FEATURE_DIM = FEATURE_DIM_PER_FRAME * 2  # 252 (mean + std)

DEFAULT_SEQUENCES_DIR = os.path.join("dataset", "wlasl_sequences")
DEFAULT_MODELS_DIR = "models"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODELS_DIR, "gesture_knn.json")
DEFAULT_MODEL_PKL_PATH = os.path.join(DEFAULT_MODELS_DIR, "gesture_knn.pkl")


def extract_sequence_features(frames: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """
    Extracts a fixed-length 252-dimensional feature vector from a variable-length temporal sequence.

    Aggregation method:
    - Mean across time for each of the 126 landmark features (features 0..125)
    - Standard deviation across time for each of the 126 landmark features (features 126..251)

    Args:
        frames (Union[List[List[float]], np.ndarray]): Sequence of T frames, each having 126 features.

    Returns:
        np.ndarray: 1D array of shape (252,) with dtype float32.

    Raises:
        ValueError: If sequence is empty or frame feature dimension is invalid.
    """
    if frames is None or len(frames) == 0:
        raise ValueError("Cannot extract features from an empty sequence.")

    arr = np.array(frames, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != FEATURE_DIM_PER_FRAME:
        raise ValueError(
            f"Expected frame feature dimension of {FEATURE_DIM_PER_FRAME}, got shape {arr.shape}."
        )

    # Calculate mean and standard deviation along temporal axis (axis 0)
    mean_vec = np.mean(arr, axis=0)  # Shape: (126,)

    if arr.shape[0] > 1:
        std_vec = np.std(arr, axis=0)  # Shape: (126,)
    else:
        std_vec = np.zeros(FEATURE_DIM_PER_FRAME, dtype=np.float32)

    # Concatenate mean and std
    features = np.concatenate([mean_vec, std_vec], axis=0).astype(np.float32)
    return features


@dataclass
class StandardScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.

    z = (x - mean) / (std + eps)
    """
    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None
    eps: float = 1e-8

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Computes the mean and std to be used for later scaling."""
        X_arr = np.array(X, dtype=np.float32)
        self.mean_ = np.mean(X_arr, axis=0)
        self.scale_ = np.std(X_arr, axis=0)
        # Avoid division by zero for constant features
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardizes features using computed mean and scale."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform.")
        X_arr = np.array(X, dtype=np.float32)
        return (X_arr - self.mean_) / (self.scale_ + self.eps)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits to data, then transforms it."""
        return self.fit(X).transform(X)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes scaler to dictionary."""
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else [],
            "scale": self.scale_.tolist() if self.scale_ is not None else [],
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardScaler":
        """Deserializes scaler from dictionary."""
        mean = np.array(data.get("mean", []), dtype=np.float32) if data.get("mean") else None
        scale = np.array(data.get("scale", []), dtype=np.float32) if data.get("scale") else None
        eps = float(data.get("eps", 1e-8))
        return cls(mean_=mean, scale_=scale, eps=eps)


@dataclass
class ClassificationReport:
    """Detailed multi-class evaluation report."""
    accuracy: float
    classes: List[str]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    per_class_support: Dict[str, int]
    confusion_matrix: Dict[str, Dict[str, int]]
    total_samples: int

    def format_table(self) -> str:
        """Renders classification report as a clean ASCII table."""
        lines = [
            "=" * 70,
            "              SignSyntax: Phase 1 Gesture Classifier Report           ",
            "=" * 70,
            f"{'Class (Gloss)':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}",
            "-" * 70,
        ]

        for cls_name in self.classes:
            prec = self.per_class_precision.get(cls_name, 0.0)
            rec = self.per_class_recall.get(cls_name, 0.0)
            f1 = self.per_class_f1.get(cls_name, 0.0)
            sup = self.per_class_support.get(cls_name, 0)
            lines.append(f"{cls_name:<15} | {prec:<10.2f} | {rec:<10.2f} | {f1:<10.2f} | {sup:<8}")

        lines.append("-" * 70)
        lines.append(f"Overall Accuracy: {self.accuracy * 100:.2f}% (on {self.total_samples} test samples)")
        lines.append("=" * 70)
        return "\n".join(lines)


class KNearestNeighborsClassifier:
    """
    Explainable k-Nearest Neighbors Classifier using Euclidean distance and voting.

    Attributes:
        k (int): Number of nearest neighbors to consider (default: 3).
        metric (str): Distance metric ('euclidean').
    """

    def __init__(self, k: int = 3, metric: str = "euclidean") -> None:
        self.k = k
        self.metric = metric
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[List[str]] = None
        self.classes_: List[str] = []
        self.scaler: Optional[StandardScaler] = None

    def fit(self, X: np.ndarray, y: List[str], scale: bool = True) -> "KNearestNeighborsClassifier":
        """
        Fits the k-NN classifier on training feature matrix X and labels y.

        Args:
            X (np.ndarray): Training feature matrix of shape (N, 252).
            y (List[str]): Training gloss labels of length N.
            scale (bool): Whether to fit and apply standard scaling.
        """
        X_arr = np.array(X, dtype=np.float32)
        if len(X_arr) != len(y):
            raise ValueError(f"X and y must have same length, got {len(X_arr)} and {len(y)}.")
        if len(X_arr) == 0:
            raise ValueError("Training dataset cannot be empty.")

        if scale:
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(X_arr)
        else:
            self.scaler = None
            self.X_train = X_arr

        self.y_train = [str(label).upper() for label in y]
        self.classes_ = sorted(list(set(self.y_train)))

        # Constrain k to at most number of training samples
        if self.k > len(self.X_train):
            self.k = max(1, len(self.X_train))

        return self

    def predict_single(self, feature_vector: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predicts gloss label for a single 252-dimensional feature vector.

        Args:
            feature_vector (np.ndarray): Feature vector of shape (252,).

        Returns:
            Tuple[str, float, List[Tuple[str, float]]]:
                - Predicted uppercase gloss label.
                - Confidence probability (0.0 to 1.0).
                - List of top-k nearest neighbors: [(label, distance), ...].
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Classifier must be fitted before predict.")

        x = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != AGGREGATED_FEATURE_DIM:
            raise ValueError(f"Expected feature dim {AGGREGATED_FEATURE_DIM}, got {x.shape[1]}.")

        if self.scaler is not None:
            x = self.scaler.transform(x)

        # Compute Euclidean distances to all training points
        diff = self.X_train - x
        distances = np.sqrt(np.sum(diff ** 2, axis=1))  # Shape: (N,)

        # Find indices of top-k nearest neighbors
        k_val = min(self.k, len(distances))
        nearest_indices = np.argsort(distances)[:k_val]

        neighbor_labels = [self.y_train[i] for i in nearest_indices]
        neighbor_distances = [float(distances[i]) for i in nearest_indices]
        neighbors_info = list(zip(neighbor_labels, neighbor_distances))

        # Majority voting
        label_counts: Dict[str, int] = {}
        for lbl in neighbor_labels:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        # Break ties using minimum distance sum
        sorted_candidates = sorted(
            label_counts.keys(),
            key=lambda l: (
                -label_counts[l],
                sum(d for lbl, d in neighbors_info if lbl == l),
            ),
        )

        predicted_label = sorted_candidates[0]
        confidence = float(label_counts[predicted_label] / k_val)

        return predicted_label, confidence, neighbors_info

    def predict(self, X: np.ndarray) -> List[str]:
        """Predicts labels for multiple feature vectors."""
        X_arr = np.array(X, dtype=np.float32)
        predictions = []
        for i in range(len(X_arr)):
            pred, _, _ = self.predict_single(X_arr[i])
            predictions.append(pred)
        return predictions

    def evaluate(self, X_test: np.ndarray, y_test: List[str]) -> ClassificationReport:
        """Evaluates classifier on test dataset and generates a comprehensive report."""
        y_test_clean = [str(l).upper() for l in y_test]
        predictions = self.predict(X_test)

        total = len(y_test_clean)
        correct = sum(1 for p, y in zip(predictions, y_test_clean) if p == y)
        accuracy = correct / total if total > 0 else 0.0

        all_classes = sorted(list(set(self.classes_ + y_test_clean)))

        confusion: Dict[str, Dict[str, int]] = {
            c1: {c2: 0 for c2 in all_classes} for c1 in all_classes
        }
        for pred, actual in zip(predictions, y_test_clean):
            confusion[actual][pred] += 1

        precision_dict: Dict[str, float] = {}
        recall_dict: Dict[str, float] = {}
        f1_dict: Dict[str, float] = {}
        support_dict: Dict[str, int] = {}

        for c in all_classes:
            tp = confusion[c][c]
            fp = sum(confusion[other][c] for other in all_classes if other != c)
            fn = sum(confusion[c][other] for other in all_classes if other != c)
            sup = sum(confusion[c].values())

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            precision_dict[c] = prec
            recall_dict[c] = rec
            f1_dict[c] = f1
            support_dict[c] = sup

        return ClassificationReport(
            accuracy=accuracy,
            classes=all_classes,
            per_class_precision=precision_dict,
            per_class_recall=recall_dict,
            per_class_f1=f1_dict,
            per_class_support=support_dict,
            confusion_matrix=confusion,
            total_samples=total,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes classifier state to a JSON-compatible dictionary."""
        return {
            "model_type": "KNearestNeighborsClassifier",
            "k": self.k,
            "metric": self.metric,
            "classes": self.classes_,
            "feature_dim": AGGREGATED_FEATURE_DIM,
            "training_samples_count": len(self.X_train) if self.X_train is not None else 0,
            "X_train": self.X_train.tolist() if self.X_train is not None else [],
            "y_train": self.y_train if self.y_train is not None else [],
            "scaler": self.scaler.to_dict() if self.scaler is not None else None,
        }

    def save(self, json_path: str = DEFAULT_MODEL_PATH, pkl_path: Optional[str] = DEFAULT_MODEL_PKL_PATH) -> None:
        """Saves model to JSON and optionally pickle format."""
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

        if pkl_path:
            os.makedirs(os.path.dirname(os.path.abspath(pkl_path)), exist_ok=True)
            with open(pkl_path, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KNearestNeighborsClassifier":
        """Deserializes classifier from dictionary."""
        k = int(data.get("k", 3))
        metric = str(data.get("metric", "euclidean"))
        model = cls(k=k, metric=metric)

        model.classes_ = data.get("classes", [])
        if data.get("X_train"):
            model.X_train = np.array(data["X_train"], dtype=np.float32)
        if data.get("y_train"):
            model.y_train = data["y_train"]
        if data.get("scaler"):
            model.scaler = StandardScaler.from_dict(data["scaler"])
        return model

    @classmethod
    def load(cls, filepath: str = DEFAULT_MODEL_PATH) -> "KNearestNeighborsClassifier":
        """Loads classifier from JSON or PKL file."""
        if filepath.endswith(".pkl"):
            with open(filepath, "rb") as f:
                return pickle.load(f)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def load_dataset_from_sequences(
    sequences_dir: str = DEFAULT_SEQUENCES_DIR,
    target_glosses: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Loads all JSON sequence files from the dataset directory and extracts 252-dim features.

    Args:
        sequences_dir (str): Base directory containing <GLOSS>/<video_id>.json.
        target_glosses (Optional[List[str]]): Filter for specific glosses.

    Returns:
        Tuple[np.ndarray, List[str], List[str]]:
            - X: Feature matrix of shape (N, 252).
            - y: List of N uppercase gloss labels.
            - video_ids: List of N video ID strings.
    """
    if target_glosses is None:
        target_glosses = DEFAULT_PROTOTYPE_GLOSSES

    target_set = {g.upper() for g in target_glosses}

    features_list: List[np.ndarray] = []
    labels_list: List[str] = []
    video_ids_list: List[str] = []

    pattern = os.path.join(sequences_dir, "*", "*.json")
    json_files = glob.glob(pattern)

    for filepath in sorted(json_files):
        try:
            sample = LandmarkSequenceSample.load_json(filepath)
            if sample.gloss.upper() in target_set:
                feat = extract_sequence_features(sample.frames)
                features_list.append(feat)
                labels_list.append(sample.gloss.upper())
                video_ids_list.append(sample.video_id)
        except Exception as e:
            print(f"Warning: Failed to load sequence from '{filepath}': {e}")

    if not features_list:
        return np.zeros((0, AGGREGATED_FEATURE_DIM), dtype=np.float32), [], []

    X = np.stack(features_list, axis=0).astype(np.float32)
    return X, labels_list, video_ids_list


def stratified_sequence_train_test_split(
    X: np.ndarray,
    y: List[str],
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Splits video sequences into train and test partitions in a stratified manner.

    Guarantees:
    - Entire video sequences remain in either train or test (no frame leakage).
    - Every class with >= 2 samples is represented in both train and test.
    - Reproducible via random_state.

    Args:
        X (np.ndarray): Feature matrix (N, 252).
        y (List[str]): Labels of length N.
        test_size (float): Proportion of samples to assign to test set.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
            - X_train, X_test, y_train, y_test
    """
    rng = np.random.RandomState(random_state)

    # Group indices by class label
    class_indices: Dict[str, List[int]] = {}
    for idx, label in enumerate(y):
        class_indices.setdefault(label, []).append(idx)

    train_indices: List[int] = []
    test_indices: List[int] = []

    for label, indices in sorted(class_indices.items()):
        indices_shuffled = rng.permutation(indices).tolist()
        n_samples = len(indices_shuffled)

        if n_samples == 1:
            # If only 1 sample exists, place in train
            train_indices.extend(indices_shuffled)
        else:
            n_test = max(1, int(round(n_samples * test_size)))
            # Ensure at least 1 sample remains in train
            if n_test >= n_samples:
                n_test = n_samples - 1

            test_indices.extend(indices_shuffled[:n_test])
            train_indices.extend(indices_shuffled[n_test:])

    # Shuffle final train and test splits
    train_indices = rng.permutation(train_indices).tolist()
    test_indices = rng.permutation(test_indices).tolist()

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, X_test, y_train, y_test


def train_gesture_classifier(
    sequences_dir: str = DEFAULT_SEQUENCES_DIR,
    target_glosses: Optional[List[str]] = None,
    k: int = 3,
    test_size: float = 0.25,
    random_state: int = 42,
    save_model: bool = True,
    output_model_path: str = DEFAULT_MODEL_PATH,
) -> Tuple[KNearestNeighborsClassifier, ClassificationReport]:
    """
    Complete training pipeline for the SignSyntax Phase 1 Gesture Classifier.

    1. Loads processed WLASL landmark JSON sequences
    2. Extracts 252-dimensional features
    3. Performs stratified train/test split at the sequence level
    4. Fits StandardScaler and k-NN classifier
    5. Evaluates on test split and prints report
    6. Saves trained model and scaler to models/

    Args:
        sequences_dir (str): Path to sequences directory.
        target_glosses (Optional[List[str]]): Target prototype glosses.
        k (int): Number of neighbors.
        test_size (float): Test split ratio.
        random_state (int): Random seed.
        save_model (bool): Whether to save model to disk.
        output_model_path (str): Destination path for model JSON.

    Returns:
        Tuple[KNearestNeighborsClassifier, ClassificationReport]: Trained model and evaluation report.
    """
    if target_glosses is None:
        target_glosses = DEFAULT_PROTOTYPE_GLOSSES

    print("=" * 75)
    print("      SignSyntax: Phase 1 Gesture Classifier Training Pipeline      ")
    print("=" * 75)

    # Step 1: Load sequences
    X, y, vids = load_dataset_from_sequences(sequences_dir, target_glosses)
    print(f"Loaded {len(X)} video sequences across {len(set(y))} glosses.")
    print(f"Feature representation: 126 mean + 126 std = {X.shape[1]} features per sequence.")

    # Class distribution
    class_counts: Dict[str, int] = {}
    for lbl in y:
        class_counts[lbl] = class_counts.get(lbl, 0) + 1
    print("\nDataset Class Distribution:")
    for lbl, count in sorted(class_counts.items()):
        print(f"  - {lbl:<6}: {count} sequences")

    # Step 2: Stratified Split
    X_train, X_test, y_train, y_test = stratified_sequence_train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\nTrain/Test Split (Sequence-Level, No Frame Leakage):")
    print(f"  - Training samples : {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  - Testing samples  : {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    # Step 3: Train k-NN
    print(f"\nFitting k-NN Classifier (k={k}, metric=euclidean, feature_scaling=StandardScaler)...")
    clf = KNearestNeighborsClassifier(k=k)
    clf.fit(X_train, y_train, scale=True)

    # Step 4: Evaluate
    report = clf.evaluate(X_test, y_test)
    print("\n" + report.format_table())

    # Step 5: Save model
    if save_model:
        clf.save(json_path=output_model_path)
        print(f"\nTrained model successfully saved to: {output_model_path}")

    return clf, report


def main() -> None:
    """CLI entrypoint for training the gesture classifier."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SignSyntax Phase 1: Train Explainable Gesture k-NN Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sequences-dir",
        type=str,
        default=DEFAULT_SEQUENCES_DIR,
        help="Path to processed sequence JSONs directory (default: dataset/wlasl_sequences)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of nearest neighbors for k-NN (default: 3)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test partition size ratio (default: 0.25)",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Output model JSON filepath (default: models/gesture_knn.json)",
    )

    args = parser.parse_args()

    train_gesture_classifier(
        sequences_dir=args.sequences_dir,
        k=args.k,
        test_size=args.test_size,
        output_model_path=args.output_model,
    )


if __name__ == "__main__":
    main()

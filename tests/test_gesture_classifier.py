"""
Unit tests for the Phase 1 Gesture Classifier, Feature Extraction, Predictor, and Lexer Integration.
Phase 1: Computer Vision Input & Gesture Classification

Tests verify:
1. Feature extraction produces exactly 252 statistical features (126 mean + 126 std).
2. Single-frame sequence handling (zero std).
3. Error handling on invalid/empty sequences.
4. StandardScaler normalization and serialization.
5. k-NN training, prediction, neighbor retrieval, and confidence calculation.
6. Stratified train/test split at sequence level (zero frame leakage).
7. Model persistence (JSON and PKL serialization roundtrip).
8. GesturePredictor integration with SymbolTable and Lexer.
9. Raising GestureIntegrationError when a predicted gloss is outside the Symbol Table.
"""

import json
import os
import tempfile
import pytest
import numpy as np

from compiler.lexer import Lexer
from compiler.symbol_table import SymbolTable
from recognition.gesture_classifier import (
    AGGREGATED_FEATURE_DIM,
    FEATURE_DIM_PER_FRAME,
    KNearestNeighborsClassifier,
    StandardScaler,
    extract_sequence_features,
    load_dataset_from_sequences,
    stratified_sequence_train_test_split,
)
from recognition.predict_gesture import (
    GestureIntegrationError,
    GesturePredictionResult,
    GesturePredictor,
)
from recognition.wlasl_processor import LandmarkSequenceSample


# =========================================================================
# 1. Feature Extraction Tests
# =========================================================================

def test_extract_sequence_features_dimension():
    """Verifies that variable-length frame sequences always yield 252 features."""
    # 5 frames x 126 features
    frames_5 = [[float(i + j) for i in range(126)] for j in range(5)]
    feat_5 = extract_sequence_features(frames_5)
    assert isinstance(feat_5, np.ndarray)
    assert feat_5.shape == (252,)
    assert feat_5.dtype == np.float32

    # 15 frames x 126 features
    frames_15 = [[float(i * 0.1 + j) for i in range(126)] for j in range(15)]
    feat_15 = extract_sequence_features(frames_15)
    assert feat_15.shape == (252,)


def test_extract_sequence_features_mean_and_std_values():
    """Verifies correct calculation of temporal mean (first 126) and std (last 126)."""
    # 2 frames
    f1 = [10.0] * 126
    f2 = [20.0] * 126
    feat = extract_sequence_features([f1, f2])

    # Mean must be 15.0 for all 126
    assert np.allclose(feat[:126], 15.0)
    # Std must be 5.0 for all 126
    assert np.allclose(feat[126:], 5.0)


def test_extract_sequence_features_single_frame():
    """Verifies that a 1-frame sequence yields standard deviation of 0.0."""
    single_frame = [[3.14] * 126]
    feat = extract_sequence_features(single_frame)
    assert feat.shape == (252,)
    assert np.allclose(feat[:126], 3.14)
    assert np.allclose(feat[126:], 0.0)


def test_extract_sequence_features_empty_sequence():
    """Verifies ValueError when passing an empty sequence."""
    with pytest.raises(ValueError, match="empty sequence"):
        extract_sequence_features([])


def test_extract_sequence_features_invalid_dim():
    """Verifies ValueError when frame width is not 126."""
    with pytest.raises(ValueError, match="Expected frame feature dimension"):
        extract_sequence_features([[1.0, 2.0, 3.0]])


# =========================================================================
# 2. StandardScaler Tests
# =========================================================================

def test_standard_scaler_fit_transform():
    """Verifies feature standardization to zero mean and unit variance."""
    X = np.array([
        [10.0, 20.0],
        [20.0, 40.0],
        [30.0, 60.0],
    ], dtype=np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(np.mean(X_scaled, axis=0), 0.0, atol=1e-6)
    assert np.allclose(np.std(X_scaled, axis=0), 1.0, atol=1e-5)


def test_standard_scaler_constant_feature():
    """Verifies that features with zero variance do not cause division by zero."""
    X = np.array([[5.0], [5.0], [5.0]], dtype=np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert not np.isnan(X_scaled).any()
    assert np.allclose(X_scaled, 0.0)


def test_standard_scaler_dict_serialization():
    """Verifies JSON roundtrip for scaler parameters."""
    X = np.random.randn(10, 5).astype(np.float32)
    scaler = StandardScaler().fit(X)
    d = scaler.to_dict()

    reconstructed = StandardScaler.from_dict(d)
    assert np.allclose(scaler.mean_, reconstructed.mean_)
    assert np.allclose(scaler.scale_, reconstructed.scale_)


# =========================================================================
# 3. k-NN Classifier Tests
# =========================================================================

def test_knn_fit_and_predict_basic():
    """Verifies training and prediction on simple synthetic clustered data."""
    # Class A centered at 0, Class B centered at 10
    rng = np.random.RandomState(42)
    X_A = rng.normal(0.0, 0.5, size=(5, 252)).astype(np.float32)
    X_B = rng.normal(10.0, 0.5, size=(5, 252)).astype(np.float32)

    X = np.vstack([X_A, X_B])
    y = ["ME"] * 5 + ["YOU"] * 5

    clf = KNearestNeighborsClassifier(k=3).fit(X, y, scale=True)

    # Test point near 0
    test_A = np.zeros(252, dtype=np.float32)
    pred_A, conf_A, neighbors_A = clf.predict_single(test_A)
    assert pred_A == "ME"
    assert conf_A == 1.0
    assert len(neighbors_A) == 3

    # Test point near 10
    test_B = np.full(252, 10.0, dtype=np.float32)
    pred_B, conf_B, _ = clf.predict_single(test_B)
    assert pred_B == "YOU"
    assert conf_B == 1.0


def test_knn_evaluation_report():
    """Verifies evaluation metric computations (accuracy, precision, recall, F1)."""
    X_train = np.array([[0.0] * 252, [10.0] * 252], dtype=np.float32)
    y_train = ["ME", "YOU"]

    clf = KNearestNeighborsClassifier(k=1).fit(X_train, y_train, scale=False)

    X_test = np.array([[0.1] * 252, [9.9] * 252], dtype=np.float32)
    y_test = ["ME", "YOU"]

    report = clf.evaluate(X_test, y_test)
    assert report.accuracy == 1.0
    assert report.per_class_precision["ME"] == 1.0
    assert report.per_class_recall["YOU"] == 1.0
    assert "Overall Accuracy: 100.00%" in report.format_table()


def test_knn_model_serialization_roundtrip():
    """Verifies saving and loading of the trained k-NN model to/from JSON."""
    X = np.random.randn(8, 252).astype(np.float32)
    y = ["BOOK", "BOOK", "HOME", "HOME", "NOT", "NOT", "WANT", "WANT"]

    clf = KNearestNeighborsClassifier(k=3).fit(X, y, scale=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "model.json")
        clf.save(json_path=json_path, pkl_path=None)

        loaded = KNearestNeighborsClassifier.load(json_path)
        assert loaded.k == 3
        assert sorted(loaded.classes_) == sorted(clf.classes_)
        assert np.allclose(loaded.X_train, clf.X_train)
        assert loaded.y_train == clf.y_train

        # Predictions must be identical
        test_vec = np.random.randn(252).astype(np.float32)
        p1, c1, _ = clf.predict_single(test_vec)
        p2, c2, _ = loaded.predict_single(test_vec)
        assert p1 == p2
        assert c1 == c2


# =========================================================================
# 4. Stratified Split Tests
# =========================================================================

def test_stratified_sequence_split_no_leakage():
    """Verifies that sequence-level splitting preserves every class and has zero leakage."""
    X = np.arange(14 * 252).reshape(14, 252).astype(np.float32)
    # 7 classes with 2 samples each
    y = ["ME", "ME", "YOU", "YOU", "WANT", "WANT", "MILK", "MILK", "BOOK", "BOOK", "HOME", "HOME", "NOT", "NOT"]

    X_train, X_test, y_train, y_test = stratified_sequence_train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # Must have 7 samples in train and 7 in test (1 per class)
    assert len(X_train) == 7
    assert len(X_test) == 7
    assert sorted(y_train) == sorted(["ME", "YOU", "WANT", "MILK", "BOOK", "HOME", "NOT"])
    assert sorted(y_test) == sorted(["ME", "YOU", "WANT", "MILK", "BOOK", "HOME", "NOT"])


# =========================================================================
# 5. Predictor & Compiler Integration Tests
# =========================================================================

def test_gesture_predictor_with_symbol_table_and_lexer():
    """Verifies end-to-end integration from landmark sequence to Lexer Token."""
    # Build synthetic sample sequence
    frames = [[float(i * 0.05)] * 126 for i in range(10)]
    sample = LandmarkSequenceSample(
        gloss="WANT",
        wlasl_gloss="want",
        video_id="test_want",
        fps=30.0,
        frame_count=10,
        frames=frames,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal trained model
        X_train = np.vstack([
            extract_sequence_features(frames),
            extract_sequence_features([[0.0] * 126 for _ in range(5)]),
        ])
        y_train = ["WANT", "NOT"]
        clf = KNearestNeighborsClassifier(k=1).fit(X_train, y_train, scale=True)

        model_path = os.path.join(tmpdir, "model.json")
        clf.save(model_path, pkl_path=None)

        seq_path = os.path.join(tmpdir, "sample.json")
        sample.save_json(seq_path)

        # Run Predictor
        st = SymbolTable()
        predictor = GesturePredictor(model_path=model_path, symbol_table=st)
        result = predictor.predict_sequence(seq_path)

        assert result.predicted_gloss == "WANT"
        assert result.token_type == "VERB"
        assert result.meaning == "want"
        assert result.confidence == 1.0

        # Feed to Lexer
        lexer = Lexer(st)
        lexer_res = lexer.tokenize(result.predicted_gloss)
        assert len(lexer_res.tokens) == 1
        assert not lexer_res.has_errors
        assert lexer_res.tokens[0].token_type == "VERB"
        assert lexer_res.tokens[0].lexeme == "WANT"
        assert lexer_res.tokens[0].meaning == "want"


def test_gesture_predictor_integration_error_on_unknown_gloss():
    """Verifies that GestureIntegrationError is raised if predicted gloss is not in Symbol Table."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Model trained on unknown gloss 'UNKNOWN_SIGN'
        X_train = np.random.randn(2, 252).astype(np.float32)
        y_train = ["UNKNOWN_SIGN", "UNKNOWN_SIGN"]
        clf = KNearestNeighborsClassifier(k=1).fit(X_train, y_train, scale=False)

        model_path = os.path.join(tmpdir, "model.json")
        clf.save(model_path, pkl_path=None)

        predictor = GesturePredictor(model_path=model_path, symbol_table=SymbolTable())

        dummy_frames = [[0.0] * 126 for _ in range(3)]
        with pytest.raises(GestureIntegrationError, match="Integration Error"):
            predictor.predict_sequence(dummy_frames)


def test_gesture_prediction_result_vote_counts_safe_default():
    """Verifies that GesturePredictionResult safely provides vote_counts by default."""
    res = GesturePredictionResult(
        predicted_gloss="BOOK",
        confidence=0.8,
        token_type="NOUN",
        meaning="book",
        nearest_neighbors=[("BOOK", 0.1), ("BOOK", 0.2), ("MILK", 0.5)],
    )
    assert hasattr(res, "vote_counts")
    assert isinstance(res.vote_counts, dict)
    assert res.vote_counts == {}
    assert res.predicted_gloss == "BOOK"
    assert res.confidence == 0.8
    assert len(res.nearest_neighbors) == 3


def test_gesture_predictor_predict_single_and_vote_counts():
    """Verifies that predict_single and predict_sequence accurately populate vote_counts and features."""
    with tempfile.TemporaryDirectory() as tmpdir:
        X_train = np.array([
            [1.0] * 252,
            [1.1] * 252,
            [5.0] * 252,
        ], dtype=np.float32)
        y_train = ["BOOK", "BOOK", "MILK"]
        clf = KNearestNeighborsClassifier(k=3).fit(X_train, y_train, scale=False)

        model_path = os.path.join(tmpdir, "model.json")
        clf.save(model_path, pkl_path=None)

        predictor = GesturePredictor(model_path=model_path, symbol_table=SymbolTable())

        # Test predict_single directly with feature vector
        test_feat = np.array([1.05] * 252, dtype=np.float32)
        res_single = predictor.predict_single(test_feat)

        assert res_single.predicted_gloss == "BOOK"
        assert res_single.token_type == "NOUN"
        assert res_single.vote_counts == {"BOOK": 2, "MILK": 1}
        assert res_single.features is not None
        assert len(res_single.nearest_neighbors) == 3

        # Test predict_sequence with synthetic frames
        dummy_frames = [[1.05] * 126 for _ in range(5)]
        res_seq = predictor.predict_sequence(dummy_frames)

        assert res_seq.predicted_gloss == "BOOK"
        assert isinstance(res_seq.vote_counts, dict)
        assert res_seq.vote_counts.get("BOOK") == 2
        assert res_seq.vote_counts.get("MILK") == 1
        assert res_seq.features is not None


def test_gesture_predictor_closed_set_prediction():
    """Verifies closed-set behavior: predictor deterministically classifies to nearest prototype and never returns UNKNOWN."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train a toy model with prototype centered around 1.0
        X_train = np.array([
            [1.0] * 252,
            [1.1] * 252,
            [1.2] * 252,
        ], dtype=np.float32)
        y_train = ["BOOK", "BOOK", "BOOK"]
        clf = KNearestNeighborsClassifier(k=3).fit(X_train, y_train, scale=False)

        model_path = os.path.join(tmpdir, "model.json")
        clf.save(model_path, pkl_path=None)

        predictor = GesturePredictor(model_path=model_path)

        # 1. In-distribution input -> predicts BOOK
        in_distribution = np.array([1.05] * 252, dtype=np.float32)
        res_ok = predictor.predict_single(in_distribution)
        assert res_ok.predicted_gloss == "BOOK"
        assert res_ok.token_type == "NOUN"
        assert res_ok.is_recognized is True

        # 2. Far input -> closed-set still predicts nearest class (BOOK), NEVER UNKNOWN
        far_input = np.array([50.0] * 252, dtype=np.float32)
        res_far = predictor.predict_single(far_input)
        assert res_far.predicted_gloss == "BOOK"
        assert res_far.predicted_gloss != "UNKNOWN"
        assert res_far.token_type == "NOUN"
        assert res_far.is_recognized is True

        # 3. Synthetic frames in predict_sequence -> closed-set predicts BOOK, NEVER UNKNOWN
        zero_frames = [[0.0] * 126 for _ in range(5)]
        res_zero = predictor.predict_sequence(zero_frames)
        assert res_zero.predicted_gloss == "BOOK"
        assert res_zero.predicted_gloss != "UNKNOWN"
        assert res_zero.is_recognized is True



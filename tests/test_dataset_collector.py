"""
Tests for SignSyntax Landmark Dataset Collector
Phase 1: Computer Vision Input & Dataset Collection Verification

Validates CSV schema generation, landmark flattening, label validation against
the Symbol Table, and sample persistence without requiring camera hardware.
"""

import os
from typing import List
import pytest
from compiler.symbol_table import SymbolTable
from recognition.dataset_collector import (
    DatasetCollector,
    count_existing_samples,
    flatten_landmarks,
    generate_csv_header,
    get_dataset_csv_path,
    save_landmark_sample,
    validate_label,
)
from recognition.hand_tracker import HandLandmarks, LandmarkPoint


@pytest.fixture
def sample_hand() -> HandLandmarks:
    """Fixture generating a synthetic HandLandmarks object with exactly 21 points."""
    points = [
        LandmarkPoint(index=i, x=float(i) * 0.04, y=0.50 + float(i) * 0.01, z=-0.02)
        for i in range(21)
    ]
    return HandLandmarks(handedness="Right", landmarks=points)


def test_generate_csv_header() -> None:
    """Verify CSV header has exactly 64 columns and correct landmark coordinate names."""
    header = generate_csv_header()
    assert len(header) == 64
    assert header[0] == "label"
    assert header[1:4] == ["x0", "y0", "z0"]
    assert header[4:7] == ["x1", "y1", "z1"]
    assert header[-3:] == ["x20", "y20", "z20"]


def test_flatten_landmarks_valid(sample_hand: HandLandmarks) -> None:
    """Verify landmark flattening produces a 64-element row matching schema."""
    row = flatten_landmarks("ME", sample_hand)
    assert len(row) == 64
    assert row[0] == "ME"
    # First landmark (x0, y0, z0)
    assert row[1] == pytest.approx(0.0)
    assert row[2] == pytest.approx(0.50)
    assert row[3] == pytest.approx(-0.02)


def test_flatten_landmarks_invalid_count() -> None:
    """Verify ValueError is raised if hand does not have exactly 21 landmarks."""
    short_points = [LandmarkPoint(index=0, x=0.1, y=0.1, z=0.0)]
    invalid_hand = HandLandmarks(handedness="Right", landmarks=short_points)

    with pytest.raises(ValueError, match="exactly 21 landmarks"):
        flatten_landmarks("ME", invalid_hand)


def test_validate_label_acceptance() -> None:
    """Verify acceptance of valid formal glosses from the Symbol Table."""
    st = SymbolTable()
    for valid_label in ["ME", "YOU", "WANT", "MILK", "BOOK", "HOME", "NOT", "HAPPY", "WHY", "TODAY"]:
        assert validate_label(valid_label, st) is True


def test_validate_label_rejection() -> None:
    """Verify rejection of invalid, lowercase, or unknown glosses."""
    st = SymbolTable()
    for invalid_label in ["DANCE", "APPLE", "RUN", "me", "Me", "Milk", "", None]:
        assert validate_label(invalid_label, st) is False


def test_get_dataset_csv_path() -> None:
    """Verify standardized dataset path formatting."""
    path = get_dataset_csv_path("ME", base_dir="custom_dataset")
    expected = os.path.join("custom_dataset", "ME", "samples.csv")
    assert path == expected


def test_save_and_count_samples(tmp_path, sample_hand: HandLandmarks) -> None:
    """Verify saving samples creates header and appends rows correctly."""
    csv_file = str(tmp_path / "ME" / "samples.csv")

    assert count_existing_samples(csv_file) == 0

    row1 = flatten_landmarks("ME", sample_hand)
    assert save_landmark_sample(csv_file, row1) is True
    assert count_existing_samples(csv_file) == 1

    row2 = flatten_landmarks("ME", sample_hand)
    assert save_landmark_sample(csv_file, row2) is True
    assert count_existing_samples(csv_file) == 2


def test_save_sample_invalid_row_length(tmp_path) -> None:
    """Verify saving fails if row does not contain 64 elements."""
    csv_file = str(tmp_path / "ME" / "samples.csv")
    with pytest.raises(ValueError, match="64 elements"):
        save_landmark_sample(csv_file, ["ME", 1.0, 2.0])


def test_collector_initialization_valid(tmp_path) -> None:
    """Verify DatasetCollector initializes with a valid label."""
    collector = DatasetCollector(label="WANT", dataset_dir=str(tmp_path))
    assert collector.label == "WANT"
    assert collector.sample_count == 0
    assert collector.csv_path == os.path.join(str(tmp_path), "WANT", "samples.csv")


def test_collector_initialization_invalid() -> None:
    """Verify DatasetCollector raises ValueError on invalid label."""
    with pytest.raises(ValueError, match="Invalid gloss label"):
        DatasetCollector(label="DANCE")


def test_collector_capture_sample_zero_hands(tmp_path) -> None:
    """Verify capture_sample rejects when no hands are in frame."""
    collector = DatasetCollector(label="MILK", dataset_dir=str(tmp_path))
    success, msg = collector.capture_sample([])
    assert success is False
    assert "No hand detected" in msg
    assert collector.sample_count == 0


def test_collector_capture_sample_multiple_hands(tmp_path, sample_hand: HandLandmarks) -> None:
    """Verify capture_sample rejects when multiple hands are in frame."""
    collector = DatasetCollector(label="MILK", dataset_dir=str(tmp_path))
    success, msg = collector.capture_sample([sample_hand, sample_hand])
    assert success is False
    assert "Multiple hands detected" in msg
    assert collector.sample_count == 0


def test_collector_capture_sample_single_hand(tmp_path, sample_hand: HandLandmarks) -> None:
    """Verify capture_sample records data when exactly one hand is present."""
    collector = DatasetCollector(label="MILK", dataset_dir=str(tmp_path))
    success, msg = collector.capture_sample([sample_hand])
    assert success is True
    assert "captured and saved" in msg
    assert collector.sample_count == 1
    assert count_existing_samples(collector.csv_path) == 1

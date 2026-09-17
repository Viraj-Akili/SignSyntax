"""
Tests for SignSyntax Hand Tracker
Phase 1: Computer Vision Input Layer Verification

Validates data structures, tracker initialization, and offline frame processing
without requiring physical camera hardware.
"""

import numpy as np
import pytest
from recognition.hand_tracker import HandLandmarks, HandTracker, LandmarkPoint


def test_landmark_point_structure() -> None:
    """Verify LandmarkPoint properties, tuple conversion, and string representation."""
    pt = LandmarkPoint(index=0, x=0.25, y=0.50, z=-0.05)
    assert pt.index == 0
    assert pt.x == 0.25
    assert pt.y == 0.50
    assert pt.z == -0.05
    assert pt.to_tuple() == (0.25, 0.50, -0.05)
    assert "LM#00" in str(pt)


def test_hand_landmarks_structure() -> None:
    """Verify HandLandmarks aggregates exactly 21 LandmarkPoint objects."""
    points = [
        LandmarkPoint(index=i, x=float(i) / 20.0, y=0.5, z=0.0)
        for i in range(21)
    ]
    hand = HandLandmarks(handedness="Right", landmarks=points)

    assert len(hand) == 21
    assert hand.handedness == "Right"
    assert len(hand.get_coordinates()) == 21
    assert hand.get_coordinates()[0] == (0.0, 0.5, 0.0)
    assert hand.get_coordinates()[-1] == (1.0, 0.5, 0.0)

    # Test iteration
    extracted = list(hand)
    assert len(extracted) == 21
    assert extracted[5].index == 5


def test_hand_tracker_initialization() -> None:
    """Verify HandTracker initializes with custom configuration."""
    tracker = HandTracker(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    assert tracker.static_image_mode is True
    assert tracker.max_num_hands == 1
    assert tracker.min_detection_confidence == 0.7
    assert tracker.min_tracking_confidence == 0.7
    tracker.close()


def test_hand_tracker_process_synthetic_blank_frame() -> None:
    """Test frame processing with synthetic blank image (no hands present)."""
    tracker = HandTracker(static_image_mode=True, max_num_hands=2)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    hands, annotated = tracker.process_frame(blank_frame, draw=True)

    assert isinstance(hands, list)
    assert len(hands) == 0
    assert annotated.shape == (480, 640, 3)
    tracker.close()


def test_hand_tracker_context_manager() -> None:
    """Verify HandTracker safely closes when used as a context manager."""
    with HandTracker() as tracker:
        assert tracker is not None

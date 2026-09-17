"""
Unit tests for the WLASL dataset inspection and video sequence processor modules.
Phase 1: Computer Vision Input & WLASL Integration

Tests verify:
1. WLASL metadata parsing and gloss matching
2. Target label filtering and sample counting
3. Multi-hand slot assignment and 126-dimensional frame padding
4. Missing hand handling (left missing, right missing, both missing)
5. Temporal landmark sequence serialization (JSON, dict, NumPy)
6. Video frame processing using synthetic test video frames (no webcam required)
"""

import json
import os
import tempfile
import pytest
import numpy as np

from recognition.hand_tracker import HandLandmarks, LandmarkPoint
from recognition.wlasl_inspector import (
    DEFAULT_PROTOTYPE_GLOSSES,
    WLASLGlossReport,
    WLASLInstance,
    display_wlasl_report,
    inspect_wlasl_glosses,
    parse_wlasl_entry,
)
from recognition.wlasl_processor import (
    FRAME_FEATURE_DIM,
    HAND_FEATURE_DIM,
    LandmarkSequenceSample,
    format_frame_landmarks,
    get_sequence_output_path,
    process_video_frames,
    process_video_file_to_sample,
    process_and_save_video,
)


# Helper to build a mock 21-landmark hand
def _create_mock_hand(handedness: str = "Right", offset: float = 0.0) -> HandLandmarks:
    landmarks = [
        LandmarkPoint(
            index=i,
            x=round(0.01 * i + offset, 4),
            y=round(0.02 * i + offset, 4),
            z=round(0.03 * i + offset, 4),
        )
        for i in range(21)
    ]
    return HandLandmarks(handedness=handedness, landmarks=landmarks)


# =========================================================================
# 1. WLASL Metadata & Inspector Tests
# =========================================================================

def test_parse_wlasl_entry_valid():
    """Verifies parsing of a single WLASL dictionary entry into WLASLInstance objects."""
    entry = {
        "gloss": "book",
        "instances": [
            {
                "instance_id": 0,
                "video_id": "06041",
                "url": "https://example.com/video1.mp4",
                "frame_start": 1,
                "frame_end": 45,
                "split": "train",
                "signer_id": 12,
            },
            {
                "instance_id": 1,
                "video_id": "06042",
                "url": "https://example.com/video2.mp4",
                "frame_start": 10,
                "frame_end": 50,
                "split": "val",
                "signer_id": 14,
            },
        ],
    }

    gloss_name, instances = parse_wlasl_entry(entry)
    assert gloss_name == "book"
    assert len(instances) == 2

    assert instances[0].video_id == "06041"
    assert instances[0].url == "https://example.com/video1.mp4"
    assert instances[0].frame_start == 1
    assert instances[0].frame_end == 45
    assert instances[0].split == "train"
    assert instances[0].signer_id == 12

    assert instances[1].video_id == "06042"
    assert instances[1].split == "val"


def test_parse_wlasl_entry_empty_instances():
    """Verifies parsing entry with no instances."""
    entry = {"gloss": "empty_sign", "instances": []}
    gloss_name, instances = parse_wlasl_entry(entry)
    assert gloss_name == "empty_sign"
    assert len(instances) == 0


def test_inspect_wlasl_glosses_matching():
    """Verifies matching uppercase SignSyntax glosses against lowercase WLASL metadata."""
    metadata = [
        {
            "gloss": "me",
            "instances": [{"video_id": "v1", "url": "u1", "frame_start": 1, "frame_end": 30}],
        },
        {
            "gloss": "want",
            "instances": [
                {"video_id": "v2", "url": "u2", "frame_start": 1, "frame_end": 40},
                {"video_id": "v3", "url": "u3", "frame_start": 1, "frame_end": 50},
            ],
        },
        {
            "gloss": "unrelated",
            "instances": [{"video_id": "v4", "url": "u4", "frame_start": 1, "frame_end": 20}],
        },
    ]

    requested = ["ME", "WANT", "MILK"]
    reports = inspect_wlasl_glosses(requested, metadata)

    assert "ME" in reports
    assert reports["ME"].is_available is True
    assert reports["ME"].sample_count == 1
    assert reports["ME"].matched_wlasl_gloss == "me"

    assert "WANT" in reports
    assert reports["WANT"].is_available is True
    assert reports["WANT"].sample_count == 2
    assert reports["WANT"].matched_wlasl_gloss == "want"

    assert "MILK" in reports
    assert reports["MILK"].is_available is False
    assert reports["MILK"].sample_count == 0
    assert reports["MILK"].matched_wlasl_gloss is None


def test_display_wlasl_report_formatting():
    """Verifies ASCII report generation contains expected summary information."""
    reports = {
        "ME": WLASLGlossReport(
            requested_gloss="ME",
            matched_wlasl_gloss="me",
            is_available=True,
            sample_count=9,
            instances=[],
        ),
        "NOT": WLASLGlossReport(
            requested_gloss="NOT",
            matched_wlasl_gloss="not",
            is_available=True,
            sample_count=14,
            instances=[],
        ),
        "MISSING": WLASLGlossReport(
            requested_gloss="MISSING",
            matched_wlasl_gloss=None,
            is_available=False,
            sample_count=0,
            instances=[],
        ),
    }

    report_text = display_wlasl_report(reports)
    assert "ME" in report_text
    assert "NOT" in report_text
    assert "MISSING" in report_text
    assert "[AVAILABLE]" in report_text
    assert "[NOT FOUND]" in report_text
    assert "2/3 requested glosses found in WLASL" in report_text


# =========================================================================
# 2. Multi-Hand Frame Formatting & Missing Hand Handling Tests
# =========================================================================

def test_format_frame_landmarks_no_hands():
    """Verifies that an empty hand detection yields 126 zeros."""
    frame_vector = format_frame_landmarks([])
    assert len(frame_vector) == FRAME_FEATURE_DIM  # 126
    assert all(val == 0.0 for val in frame_vector)


def test_format_frame_landmarks_left_hand_only():
    """Verifies slot mapping when only Left hand is present (Right hand zero-padded)."""
    left_hand = _create_mock_hand(handedness="Left", offset=0.1)
    frame_vector = format_frame_landmarks([left_hand])

    assert len(frame_vector) == 126

    # Slot 0 (Left hand, indices 0..62) must be non-zero
    left_slot = frame_vector[:63]
    assert any(val != 0.0 for val in left_slot)
    assert left_slot[0] == left_hand.landmarks[0].x
    assert left_slot[1] == left_hand.landmarks[0].y
    assert left_slot[2] == left_hand.landmarks[0].z

    # Slot 1 (Right hand, indices 63..125) must be all zeros
    right_slot = frame_vector[63:]
    assert all(val == 0.0 for val in right_slot)


def test_format_frame_landmarks_right_hand_only():
    """Verifies slot mapping when only Right hand is present (Left hand zero-padded)."""
    right_hand = _create_mock_hand(handedness="Right", offset=0.2)
    frame_vector = format_frame_landmarks([right_hand])

    assert len(frame_vector) == 126

    # Slot 0 (Left hand, indices 0..62) must be all zeros
    left_slot = frame_vector[:63]
    assert all(val == 0.0 for val in left_slot)

    # Slot 1 (Right hand, indices 63..125) must be non-zero
    right_slot = frame_vector[63:]
    assert any(val != 0.0 for val in right_slot)
    assert right_slot[0] == right_hand.landmarks[0].x
    assert right_slot[1] == right_hand.landmarks[0].y
    assert right_slot[2] == right_hand.landmarks[0].z


def test_format_frame_landmarks_both_hands():
    """Verifies slot mapping when both Left and Right hands are detected."""
    left_hand = _create_mock_hand(handedness="Left", offset=0.1)
    right_hand = _create_mock_hand(handedness="Right", offset=0.5)

    frame_vector = format_frame_landmarks([left_hand, right_hand])
    assert len(frame_vector) == 126

    left_slot = frame_vector[:63]
    right_slot = frame_vector[63:]

    assert any(val != 0.0 for val in left_slot)
    assert any(val != 0.0 for val in right_slot)
    assert left_slot[0] == left_hand.landmarks[0].x
    assert right_slot[0] == right_hand.landmarks[0].x


# =========================================================================
# 3. LandmarkSequenceSample Serialization & Structure Tests
# =========================================================================

def test_landmark_sequence_sample_dict_and_json_roundtrip():
    """Verifies full roundtrip serialization to/from JSON."""
    frames = [
        [float(i + j * 0.01) for i in range(126)]
        for j in range(5)
    ]

    sample = LandmarkSequenceSample(
        gloss="WANT",
        wlasl_gloss="want",
        video_id="63214",
        fps=25.0,
        frame_count=5,
        frames=frames,
        source_url="https://example.com/v.mp4",
        split="train",
        signer_id=7,
    )

    # Dictionary representation
    d = sample.to_dict()
    assert d["gloss"] == "WANT"
    assert d["frame_count"] == 5
    assert len(d["frames"]) == 5
    assert len(d["frames"][0]) == 126

    # JSON roundtrip
    json_str = sample.to_json()
    reconstructed = LandmarkSequenceSample.from_json(json_str)

    assert reconstructed.gloss == sample.gloss
    assert reconstructed.wlasl_gloss == sample.wlasl_gloss
    assert reconstructed.video_id == sample.video_id
    assert reconstructed.fps == sample.fps
    assert reconstructed.frame_count == sample.frame_count
    assert reconstructed.frames == sample.frames
    assert reconstructed.split == sample.split


def test_landmark_sequence_sample_to_numpy():
    """Verifies conversion of temporal frames to a (T, 126) NumPy array."""
    frames = [[float(i) for i in range(126)] for _ in range(10)]
    sample = LandmarkSequenceSample(
        gloss="BOOK",
        wlasl_gloss="book",
        video_id="06041",
        fps=30.0,
        frame_count=10,
        frames=frames,
    )

    arr = sample.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (10, 126)
    assert arr.dtype == np.float32
    assert arr[0, 0] == 0.0
    assert arr[0, 125] == 125.0


def test_landmark_sequence_sample_save_and_load_file():
    """Verifies file saving and loading of LandmarkSequenceSample."""
    frames = [[0.5] * 126 for _ in range(3)]
    sample = LandmarkSequenceSample(
        gloss="MILK",
        wlasl_gloss="milk",
        video_id="37119",
        fps=30.0,
        frame_count=3,
        frames=frames,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "MILK", "37119.json")
        sample.save_json(filepath)

        assert os.path.isfile(filepath)
        loaded = LandmarkSequenceSample.load_json(filepath)

        assert loaded.gloss == "MILK"
        assert loaded.video_id == "37119"
        assert loaded.frame_count == 3
        assert len(loaded.frames) == 3
        assert loaded.frames[0] == [0.5] * 126


def test_get_sequence_output_path():
    """Verifies standardized sequence JSON path construction."""
    path = get_sequence_output_path("want", "12345", base_dir="custom_dir")
    assert path == os.path.join("custom_dir", "WANT", "12345.json")


# =========================================================================
# 4. Mock Video Frame Processing Pipeline Tests
# =========================================================================

def _create_synthetic_video_file(filepath: str, num_frames: int = 15, width: int = 320, height: int = 240) -> None:
    """Creates a temporary synthetic MP4 video file using OpenCV for tests."""
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filepath, fourcc, 30.0, (width, height))
    for i in range(num_frames):
        # Draw gradient frame with some shapes
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (i * 15 % 255, (255 - i * 15) % 255, 128)
        out.write(frame)
    out.release()


def test_process_video_frames_with_synthetic_video():
    """Verifies video reading and frame-by-frame feature extraction without requiring a webcam."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test_sign.mp4")
        _create_synthetic_video_file(video_path, num_frames=10)

        frames, fps, total_frames = process_video_frames(
            video_path=video_path,
            frame_start=1,
            frame_end=8,
        )

        assert len(frames) == 8
        assert total_frames == 10
        assert fps == 30.0
        # Each frame vector must be exactly 126 elements
        for fvec in frames:
            assert len(fvec) == 126


def test_process_and_save_video_synthetic_pipeline():
    """Verifies end-to-end processing and JSON saving using a synthetic video."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "home_sample.mp4")
        _create_synthetic_video_file(video_path, num_frames=6)

        out_dir = os.path.join(tmpdir, "wlasl_sequences")
        sample, saved_path = process_and_save_video(
            video_path=video_path,
            gloss="HOME",
            wlasl_gloss="home",
            video_id="home_01",
            output_dir=out_dir,
            frame_start=1,
            frame_end=5,
        )

        assert sample.gloss == "HOME"
        assert sample.frame_count == 5
        assert os.path.isfile(saved_path)
        assert saved_path == os.path.join(out_dir, "HOME", "home_01.json")

        # Load back and verify
        loaded = LandmarkSequenceSample.load_json(saved_path)
        assert loaded.gloss == "HOME"
        assert loaded.frame_count == 5
        assert len(loaded.frames) == 5
        assert len(loaded.frames[0]) == 126

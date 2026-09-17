"""
SignSyntax - WLASL Video Dataset Processor Module
Phase 1: Computer Vision Input & Temporal Sequence Processing

This module processes Word-Level American Sign Language (WLASL) video sequences into
consistent temporal hand-landmark sequences using OpenCV and MediaPipe Hands.

Key Architectural Guarantees:
1. Multi-Hand Support: Extracts up to two hands per frame.
2. Fixed Feature Representation: Exactly 126 float values per frame:
   - Slot 0 (Left Hand): 21 landmarks x 3 coordinates (x, y, z) = 63 floats (indices 0..62)
   - Slot 1 (Right Hand): 21 landmarks x 3 coordinates (x, y, z) = 63 floats (indices 63..125)
3. Missing Hand Consistency: Missing hands are consistently padded with 0.0 floats.
4. Temporal Ordering: Preserves frame-by-frame time sequence across the video sample.
5. Decoupled Storage: Saves sequence samples to dataset/wlasl_sequences/<GLOSS>/<video_id>.json,
   keeping existing manual single-frame CSV datasets completely separate.

Citation & Attribution:
WLASL is provided under the Computational Use of Data Agreement (C-UDA).
Reference: Dongxu Li et al., "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison", WACV 2020.
https://github.com/dxli94/WLASL
"""

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import cv2
    import numpy as np
    _VISION_AVAILABLE = True
except ImportError:
    _VISION_AVAILABLE = False

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.symbol_table import SymbolTable
from recognition.hand_tracker import HandLandmarks, HandTracker, LandmarkPoint


# Constants
NUM_LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3  # (x, y, z)
HAND_FEATURE_DIM = NUM_LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 63
NUM_HAND_SLOTS = 2  # Left Hand (slot 0), Right Hand (slot 1)
FRAME_FEATURE_DIM = HAND_FEATURE_DIM * NUM_HAND_SLOTS  # 126

DEFAULT_SEQUENCES_DIR = os.path.join("dataset", "wlasl_sequences")
DEFAULT_VIDEOS_DIR = os.path.join("dataset", "wlasl_videos")


def format_frame_landmarks(detected_hands: List[HandLandmarks]) -> List[float]:
    """
    Formats detected hand landmarks in a frame into a standardized 126-dimensional float vector.

    Slots layout:
    - Indices 0..62: Left Hand (21 landmarks x 3 coords: x, y, z)
    - Indices 63..125: Right Hand (21 landmarks x 3 coords: x, y, z)

    Missing hands are padded with 63 zeros (0.0).

    Args:
        detected_hands (List[HandLandmarks]): Detected hands from HandTracker.

    Returns:
        List[float]: Exactly 126 float values representing both hand slots.
    """
    left_slot: List[float] = [0.0] * HAND_FEATURE_DIM
    right_slot: List[float] = [0.0] * HAND_FEATURE_DIM

    left_assigned = False
    right_assigned = False

    for hand in detected_hands:
        if len(hand.landmarks) != NUM_LANDMARKS_PER_HAND:
            continue

        # Extract 63 coordinates
        coords: List[float] = []
        for lm in hand.landmarks:
            coords.extend([float(lm.x), float(lm.y), float(lm.z)])

        # Assign based on handedness
        if hand.handedness == "Left" and not left_assigned:
            left_slot = coords
            left_assigned = True
        elif hand.handedness == "Right" and not right_assigned:
            right_slot = coords
            right_assigned = True
        elif not left_assigned:
            left_slot = coords
            left_assigned = True
        elif not right_assigned:
            right_slot = coords
            right_assigned = True

    return left_slot + right_slot


@dataclass
class LandmarkSequenceSample:
    """
    Represents a full temporal sequence of multi-hand landmark frames for a sign video sample.

    Attributes:
        gloss (str): Uppercase SignSyntax gloss (e.g. 'WANT', 'BOOK').
        wlasl_gloss (str): WLASL gloss identifier (e.g. 'want', 'book').
        video_id (str): WLASL video identification string.
        fps (float): Video framerate.
        frame_count (int): Number of frames in the temporal sequence.
        feature_dim_per_frame (int): Dimensionality per frame (default: 126).
        frames (List[List[float]]): List of frame vectors, each having length 126.
        source_url (str): Source video URL if available.
        split (str): Dataset split ('train', 'val', 'test').
        signer_id (int): Signer identifier.
    """
    gloss: str
    wlasl_gloss: str
    video_id: str
    fps: float
    frame_count: int
    frames: List[List[float]]
    feature_dim_per_frame: int = FRAME_FEATURE_DIM
    source_url: str = ""
    split: str = "train"
    signer_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Converts sample to a JSON-serializable dictionary."""
        return {
            "gloss": self.gloss,
            "wlasl_gloss": self.wlasl_gloss,
            "video_id": self.video_id,
            "fps": float(self.fps),
            "frame_count": int(self.frame_count),
            "feature_dim_per_frame": int(self.feature_dim_per_frame),
            "source_url": self.source_url,
            "split": self.split,
            "signer_id": int(self.signer_id),
            "schema": {
                "hand_slots": ["left_hand", "right_hand"],
                "features_per_hand": HAND_FEATURE_DIM,
                "total_features_per_frame": FRAME_FEATURE_DIM,
            },
            "frames": self.frames,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializes sample to a JSON formatted string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_numpy(self) -> "np.ndarray":
        """
        Converts the sequence of frames to a NumPy array of shape (T, 126).

        Returns:
            np.ndarray: Array with dtype float32 and shape (frame_count, 126).
        """
        if not _VISION_AVAILABLE:
            raise RuntimeError("NumPy is required for to_numpy().")
        if not self.frames:
            return np.zeros((0, self.feature_dim_per_frame), dtype=np.float32)
        return np.array(self.frames, dtype=np.float32)

    def save_json(self, filepath: str) -> None:
        """
        Saves the sample sequence to a JSON file.

        Args:
            filepath (str): Destination file path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LandmarkSequenceSample":
        """Constructs a LandmarkSequenceSample from a dictionary."""
        return cls(
            gloss=str(data.get("gloss", "")),
            wlasl_gloss=str(data.get("wlasl_gloss", "")),
            video_id=str(data.get("video_id", "")),
            fps=float(data.get("fps", 30.0)),
            frame_count=int(data.get("frame_count", len(data.get("frames", [])))),
            feature_dim_per_frame=int(data.get("feature_dim_per_frame", FRAME_FEATURE_DIM)),
            frames=data.get("frames", []),
            source_url=str(data.get("source_url", "")),
            split=str(data.get("split", "train")),
            signer_id=int(data.get("signer_id", 0)),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "LandmarkSequenceSample":
        """
        Constructs a LandmarkSequenceSample from a JSON formatted string.

        Args:
            json_str (str): JSON string representation.

        Returns:
            LandmarkSequenceSample: Deserialized sample object.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def load_json(cls, filepath: str) -> "LandmarkSequenceSample":
        """
        Loads a LandmarkSequenceSample from a JSON file.

        Args:
            filepath (str): Path to JSON sequence file.

        Returns:
            LandmarkSequenceSample: Loaded sample object.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)



def process_video_frames(
    video_path: str,
    tracker: Optional[HandTracker] = None,
    frame_start: int = 1,
    frame_end: int = -1,
    max_frames: Optional[int] = None,
) -> Tuple[List[List[float]], float, int]:
    """
    Processes an input video file, extracting 126-dimensional multi-hand landmarks for each frame.

    Args:
        video_path (str): Path to the video file (e.g. MP4, AVI).
        tracker (Optional[HandTracker]): MediaPipe HandTracker instance.
        frame_start (int): 1-based start frame index (WLASL annotation standard).
        frame_end (int): 1-based end frame index (-1 means until end of video).
        max_frames (Optional[int]): Maximum number of frames to process.

    Returns:
        Tuple[List[List[float]], float, int]:
            - frames: Ordered list of 126-dimensional frame feature vectors.
            - fps: Video framerate.
            - total_video_frames: Total number of frames in the source video.
    """
    if not _VISION_AVAILABLE:
        raise RuntimeError("OpenCV and MediaPipe are required to process video frames.")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: '{video_path}'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: '{video_path}'")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if tracker is None:
        tracker = HandTracker(static_image_mode=False, max_num_hands=2)

    frames: List[List[float]] = []
    current_frame_idx = 1  # 1-based index

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Respect WLASL frame range annotations
            if current_frame_idx < frame_start:
                current_frame_idx += 1
                continue

            if frame_end > 0 and current_frame_idx > frame_end:
                break

            # Extract hand landmarks without drawing for speed
            hands, _ = tracker.process_frame(frame, draw=False)
            frame_vector = format_frame_landmarks(hands)
            frames.append(frame_vector)

            if max_frames and len(frames) >= max_frames:
                break

            current_frame_idx += 1
    finally:
        cap.release()

    return frames, fps, total_video_frames


def process_video_file_to_sample(
    video_path: str,
    gloss: str,
    wlasl_gloss: str = "",
    video_id: str = "",
    frame_start: int = 1,
    frame_end: int = -1,
    split: str = "train",
    source_url: str = "",
    signer_id: int = 0,
    tracker: Optional[HandTracker] = None,
) -> LandmarkSequenceSample:
    """
    Processes a video file and bundles the extracted landmarks into a LandmarkSequenceSample.

    Args:
        video_path (str): Path to local video file.
        gloss (str): Uppercase SignSyntax gloss (e.g., 'ME', 'BOOK').
        wlasl_gloss (str): WLASL gloss string (e.g. 'me', 'book').
        video_id (str): Video identification code.
        frame_start (int): 1-based starting frame index.
        frame_end (int): 1-based ending frame index.
        split (str): Dataset split ('train', 'val', 'test').
        source_url (str): Source URL.
        signer_id (int): Signer ID.
        tracker (Optional[HandTracker]): MediaPipe hand tracker.

    Returns:
        LandmarkSequenceSample: Structured sequence sample with ordered 126-dim frames.
    """
    if not wlasl_gloss:
        wlasl_gloss = gloss.lower()
    if not video_id:
        video_id = os.path.splitext(os.path.basename(video_path))[0]

    frames, fps, _ = process_video_frames(
        video_path=video_path,
        tracker=tracker,
        frame_start=frame_start,
        frame_end=frame_end,
    )

    return LandmarkSequenceSample(
        gloss=gloss.upper(),
        wlasl_gloss=wlasl_gloss.lower(),
        video_id=video_id,
        fps=fps,
        frame_count=len(frames),
        frames=frames,
        feature_dim_per_frame=FRAME_FEATURE_DIM,
        source_url=source_url,
        split=split,
        signer_id=signer_id,
    )


def get_sequence_output_path(
    gloss: str,
    video_id: str,
    base_dir: str = DEFAULT_SEQUENCES_DIR,
) -> str:
    """
    Computes standard destination path for saving a processed sequence JSON.

    Format: <base_dir>/<GLOSS>/<video_id>.json

    Args:
        gloss (str): SignSyntax gloss label.
        video_id (str): Video ID string.
        base_dir (str): Base output directory.

    Returns:
        str: Destination JSON file path.
    """
    return os.path.join(base_dir, gloss.upper(), f"{video_id}.json")


def process_and_save_video(
    video_path: str,
    gloss: str,
    wlasl_gloss: str = "",
    video_id: str = "",
    output_dir: str = DEFAULT_SEQUENCES_DIR,
    frame_start: int = 1,
    frame_end: int = -1,
    split: str = "train",
    source_url: str = "",
    signer_id: int = 0,
    tracker: Optional[HandTracker] = None,
) -> Tuple[LandmarkSequenceSample, str]:
    """
    Processes a video file and saves the resulting LandmarkSequenceSample JSON to output directory.

    Args:
        video_path (str): Path to input video.
        gloss (str): SignSyntax gloss (e.g. 'WANT').
        wlasl_gloss (str): WLASL gloss.
        video_id (str): Unique video ID.
        output_dir (str): Base output sequences directory.
        frame_start (int): Start frame (1-based).
        frame_end (int): End frame (1-based).
        split (str): Split label.
        source_url (str): Source URL.
        signer_id (int): Signer ID.
        tracker (Optional[HandTracker]): Hand tracker instance.

    Returns:
        Tuple[LandmarkSequenceSample, str]: Created sample object and saved JSON file path.
    """
    sample = process_video_file_to_sample(
        video_path=video_path,
        gloss=gloss,
        wlasl_gloss=wlasl_gloss,
        video_id=video_id,
        frame_start=frame_start,
        frame_end=frame_end,
        split=split,
        source_url=source_url,
        signer_id=signer_id,
        tracker=tracker,
    )

    out_path = get_sequence_output_path(gloss, sample.video_id, base_dir=output_dir)
    sample.save_json(out_path)
    return sample, out_path


def main() -> None:
    """CLI entrypoint for processing videos into landmark sequences."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SignSyntax Phase 1: WLASL Video Dataset Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to an individual video file to process",
    )
    parser.add_argument(
        "--gloss",
        type=str,
        help="SignSyntax gloss label for the video (e.g. 'BOOK', 'WANT')",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default="",
        help="Custom video ID (defaults to filename basename)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_SEQUENCES_DIR,
        help="Directory to save extracted JSON landmark sequences",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=1,
        help="Starting frame index (1-based)",
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        default=-1,
        help="Ending frame index (-1 for end of video)",
    )

    args = parser.parse_args()

    if not args.video or not args.gloss:
        parser.print_help()
        print("\nExample usage:")
        print("  python recognition/wlasl_processor.py --video sample.mp4 --gloss BOOK")
        sys.exit(0)

    st = SymbolTable()
    if not st.contains(args.gloss.upper()):
        print(f"Warning: '{args.gloss}' is not in the formal 44-gloss SignSyntax Symbol Table.")

    print(f"Processing video: {args.video} -> Gloss: {args.gloss.upper()} ...")
    sample, path = process_and_save_video(
        video_path=args.video,
        gloss=args.gloss.upper(),
        video_id=args.video_id,
        output_dir=args.output_dir,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
    )

    print(f"Success! Processed {sample.frame_count} frames (dim: {sample.feature_dim_per_frame})")
    print(f"Saved sequence to: {path}")


if __name__ == "__main__":
    main()

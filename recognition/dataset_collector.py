"""
SignSyntax - Landmark Dataset Collector Module
Phase 1: Computer Vision Input & Dataset Collection

This module captures normalized 3D hand landmark coordinates (21 landmarks x 3 axes = 63 features)
for a user-specified sign gloss label using OpenCV and MediaPipe Hands, saving flattened samples
into labeled CSV files under the dataset/ directory.

Architectural Notice:
This module is strictly for dataset recording and storage. It does NOT perform gesture classification,
machine learning inference, or lexical parsing.
"""

import csv
import os
import sys
from typing import List, Optional, Tuple, Union

try:
    import cv2
    import numpy as np
    _CV_AVAILABLE = True
except ImportError:
    _CV_AVAILABLE = False

from compiler.symbol_table import SymbolTable
from recognition.hand_tracker import HandLandmarks, HandTracker


def generate_csv_header() -> List[str]:
    """
    Generates the standard 64-column CSV header for hand landmark samples.

    Header schema: label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20

    Returns:
        List[str]: List of 64 column header names.
    """
    header = ["label"]
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    return header


def validate_label(label: str, symbol_table: Optional[SymbolTable] = None) -> bool:
    """
    Validates whether a given gloss label exists in the authoritative SignSyntax Symbol Table.

    Validation is strictly case-sensitive.

    Args:
        label (str): Gloss string (e.g., 'ME', 'WANT', 'MILK').
        symbol_table (Optional[SymbolTable]): SymbolTable instance to query against.

    Returns:
        bool: True if the label is a valid vocabulary item, False otherwise.
    """
    if not label or not isinstance(label, str):
        return False
    table = symbol_table if symbol_table is not None else SymbolTable()
    return table.contains(label)


def get_dataset_csv_path(label: str, base_dir: str = "dataset") -> str:
    """
    Computes the standardized file path for a given gloss's dataset CSV file.

    Args:
        label (str): Valid sign gloss label.
        base_dir (str): Base dataset directory path (default: 'dataset').

    Returns:
        str: Path formatted as '<base_dir>/<label>/samples.csv'.
    """
    return os.path.join(base_dir, label, "samples.csv")


def flatten_landmarks(label: str, hand: HandLandmarks) -> List[Union[str, float]]:
    """
    Flattens a HandLandmarks instance into a 64-element row matching the CSV schema.

    Args:
        label (str): Gloss label associated with the sample.
        hand (HandLandmarks): HandLandmarks object containing exactly 21 landmarks.

    Returns:
        List[Union[str, float]]: Row starting with label followed by 63 float coordinates.

    Raises:
        ValueError: If hand does not contain exactly 21 landmarks.
    """
    if len(hand.landmarks) != 21:
        raise ValueError(f"Hand must contain exactly 21 landmarks, got {len(hand.landmarks)}.")

    row: List[Union[str, float]] = [label]
    for lm in hand.landmarks:
        row.extend([lm.x, lm.y, lm.z])
    return row


def save_landmark_sample(csv_path: str, row: List[Union[str, float]]) -> bool:
    """
    Appends a single flattened landmark sample row to the designated CSV file.
    Creates parent directories and CSV header if the file does not already exist.

    Args:
        csv_path (str): Target CSV file path.
        row (List[Union[str, float]]): 64-element flattened sample row.

    Returns:
        bool: True if saved successfully.

    Raises:
        ValueError: If row does not have exactly 64 elements.
    """
    if len(row) != 64:
        raise ValueError(f"Sample row must contain exactly 64 elements (1 label + 63 coords), got {len(row)}.")

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(generate_csv_header())
        writer.writerow(row)

    return True


def count_existing_samples(csv_path: str) -> int:
    """
    Counts the number of recorded samples in the given CSV file.

    Args:
        csv_path (str): Path to the sample CSV file.

    Returns:
        int: Number of data rows (excluding header), or 0 if file does not exist.
    """
    if not os.path.isfile(csv_path):
        return 0

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        # Exclude header row if present
        return max(0, len(rows) - 1) if rows else 0


class DatasetCollector:
    """
    Coordinates capturing and persisting landmark samples for a specified sign gloss.
    """

    def __init__(
        self,
        label: str,
        dataset_dir: str = "dataset",
        symbol_table: Optional[SymbolTable] = None,
    ) -> None:
        """
        Initializes the collector for a given sign gloss label.

        Args:
            label (str): Formal gloss label (e.g., 'ME', 'YOU', 'WANT').
            dataset_dir (str): Root dataset folder.
            symbol_table (Optional[SymbolTable]): Symbol table instance to validate gloss.

        Raises:
            ValueError: If the label is not in the formal SignSyntax Symbol Table.
        """
        self.symbol_table = symbol_table if symbol_table is not None else SymbolTable()

        if not validate_label(label, self.symbol_table):
            raise ValueError(
                f"Invalid gloss label '{label}'. Label must be a formal uppercase gloss "
                f"defined in the SignSyntax Symbol Table."
            )

        self.label = label
        self.dataset_dir = dataset_dir
        self.csv_path = get_dataset_csv_path(label, base_dir=dataset_dir)
        self.sample_count = count_existing_samples(self.csv_path)

    def capture_sample(self, hands: List[HandLandmarks]) -> Tuple[bool, str]:
        """
        Validates detected hands and records one sample if exactly one hand is present.

        Args:
            hands (List[HandLandmarks]): Detected hands from tracker.

        Returns:
            Tuple[bool, str]: (Success boolean, Status message).
        """
        if len(hands) == 0:
            return False, "No hand detected. Please position one hand in frame."

        if len(hands) > 1:
            return False, f"Multiple hands detected ({len(hands)}). Only 1 hand allowed per sample."

        hand = hands[0]
        row = flatten_landmarks(self.label, hand)
        save_landmark_sample(self.csv_path, row)
        self.sample_count += 1
        return True, f"Sample #{self.sample_count} captured and saved to {self.csv_path}."

    def run_interactive_collection(self, camera_index: int = 0) -> None:
        """
        Runs an interactive OpenCV webcam session for recording samples.

        Keyboard Controls:
            SPACE     : Capture current hand sample.
            Q / ESC   : Exit session.
        """
        if not _CV_AVAILABLE:
            print("Error: OpenCV and MediaPipe are required for interactive collection.")
            return

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not access webcam at index {camera_index}.")
            return

        tracker = HandTracker(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        status_msg = "Ready. Position one hand and press SPACE."
        status_color = (255, 255, 255)

        print("=" * 65)
        print(f"       SignSyntax Dataset Collector - Recording: {self.label}")
        print("=" * 65)
        print(f"  Target CSV         : {self.csv_path}")
        print(f"  Existing Samples   : {self.sample_count}")
        print("  Controls:")
        print("    [SPACE]   -> Capture 1 landmark sample (requires 1 hand)")
        print("    [Q / ESC] -> Exit collection")
        print("=" * 65 + "\n")

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    continue

                frame = cv2.flip(frame, 1)
                hands, annotated = tracker.process_frame(frame, draw=True)

                # Overlay HUD Information
                cv2.putText(
                    annotated,
                    "SignSyntax Dataset Collector",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 200, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"Label: {self.label}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    f"Samples collected: {self.sample_count}",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    "Press SPACE to capture | Press Q or ESC to exit",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
                # Display current status / capture feedback
                cv2.putText(
                    annotated,
                    status_msg,
                    (20, annotated.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("SignSyntax - Dataset Collector", annotated)

                key = cv2.waitKey(1) & 0xFF

                # SPACE key -> Capture
                if key == 32:
                    success_cap, msg = self.capture_sample(hands)
                    status_msg = msg
                    if success_cap:
                        status_color = (0, 255, 0)
                        print(f"[{self.label}] {msg}")
                    else:
                        status_color = (0, 0, 255)
                        print(f"[{self.label}] ⚠️ {msg}")

                # Q or ESC -> Exit
                elif key in [ord("q"), ord("Q"), 27]:
                    print("\nExiting dataset collection session...")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            tracker.close()
            print(f"Session finished. Total '{self.label}' samples: {self.sample_count}.")


def main() -> None:
    """CLI entrypoint for standalone dataset collector invocation."""
    st = SymbolTable()

    if len(sys.argv) > 1:
        label = sys.argv[1].strip()
    else:
        print("=" * 60)
        print("          SignSyntax Phase 1 Dataset Collector           ")
        print("=" * 60)
        print("Available formal glosses include: ME, YOU, WANT, MILK, BOOK, HOME, NOT, etc.\n")
        label = input("Enter formal gloss label to collect: ").strip()

    if not validate_label(label, st):
        print(f"\n❌ Error: '{label}' is not a valid formal SignSyntax gloss.")
        print("Please enter a valid uppercase gloss from the Symbol Table (e.g., ME, YOU, WANT, MILK).")
        sys.exit(1)

    collector = DatasetCollector(label=label, symbol_table=st)
    collector.run_interactive_collection()


if __name__ == "__main__":
    main()

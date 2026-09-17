"""
SignSyntax - Hand Tracker Module
Phase 1: Computer Vision Input Layer

This module provides the low-level vision interface using MediaPipe Hands and OpenCV.
It captures webcam frames, extracts 21 3D landmarks for detected hands, draws skeletal
connections on video frames, and outputs structured landmark data.

Architectural Notice:
This module represents the input/feature-extraction layer. It is decoupled from the
compiler modules (Lexer, Symbol Table) and does not perform sign classification or
lexical parsing.
"""

from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Tuple, Union

try:
    import cv2
    import mediapipe as mp
    import numpy as np
    _VISION_DEPS_AVAILABLE = True
except ImportError:
    _VISION_DEPS_AVAILABLE = False


@dataclass(frozen=True)
class LandmarkPoint:
    """
    Represents a single 3D hand landmark detected by MediaPipe.

    Attributes:
        index (int): Landmark index (0 to 20).
        x (float): Normalized horizontal coordinate (0.0 to 1.0).
        y (float): Normalized vertical coordinate (0.0 to 1.0).
        z (float): Normalized depth coordinate relative to wrist.
    """
    index: int
    x: float
    y: float
    z: float

    def to_tuple(self) -> Tuple[float, float, float]:
        """Returns landmark coordinates as an (x, y, z) tuple."""
        return (self.x, self.y, self.z)

    def __str__(self) -> str:
        return f"LM#{self.index:02d}(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"


@dataclass(frozen=True)
class HandLandmarks:
    """
    Represents the full set of 21 landmarks detected for a single hand.

    Attributes:
        handedness (str): 'Left', 'Right', or 'Unknown'.
        landmarks (List[LandmarkPoint]): Exactly 21 landmarks for the hand.
    """
    handedness: str
    landmarks: List[LandmarkPoint]

    def get_coordinates(self) -> List[Tuple[float, float, float]]:
        """
        Retrieves raw (x, y, z) coordinate tuples for all 21 landmarks.

        Returns:
            List[Tuple[float, float, float]]: 21 (x, y, z) coordinate tuples.
        """
        return [lm.to_tuple() for lm in self.landmarks]

    def __len__(self) -> int:
        return len(self.landmarks)

    def __iter__(self) -> Iterator[LandmarkPoint]:
        return iter(self.landmarks)


class HandTracker:
    """
    MediaPipe Hands wrapper for capturing and tracking hand landmarks in real-time.

    Attributes:
        max_num_hands (int): Maximum number of hands to detect simultaneously (default: 2).
        min_detection_confidence (float): Minimum detection confidence threshold.
        min_tracking_confidence (float): Minimum tracking confidence threshold.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Initializes the MediaPipe Hands detector.
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        if _VISION_DEPS_AVAILABLE:
            self._mp_hands = mp.solutions.hands
            self._mp_drawing = mp.solutions.drawing_utils
            self._mp_drawing_styles = mp.solutions.drawing_styles

            self._hands = self._mp_hands.Hands(
                static_image_mode=self.static_image_mode,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        else:
            self._mp_hands = None
            self._mp_drawing = None
            self._mp_drawing_styles = None
            self._hands = None

    def process_frame(
        self,
        frame: "np.ndarray",
        draw: bool = True,
    ) -> Tuple[List[HandLandmarks], "np.ndarray"]:
        """
        Processes a single BGR video frame to detect hands and extract 21 landmarks.

        Args:
            frame (np.ndarray): Input video frame in BGR format.
            draw (bool): If True, draws skeletal landmarks and connections on the returned image.

        Returns:
            Tuple[List[HandLandmarks], np.ndarray]:
                - List of HandLandmarks objects for all detected hands.
                - Annotated image (or unmodified image if draw=False).
        """
        if not _VISION_DEPS_AVAILABLE or self._hands is None:
            raise RuntimeError("OpenCV and MediaPipe must be installed to process frames.")

        annotated_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)

        detected_hands: List[HandLandmarks] = []

        if results.multi_hand_landmarks:
            for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                # Extract handedness label if available
                handedness_label = "Unknown"
                if results.multi_handedness and idx < len(results.multi_handedness):
                    classification = results.multi_handedness[idx].classification[0]
                    handedness_label = classification.label

                # Extract 21 landmarks
                points: List[LandmarkPoint] = []
                for lm_idx, lm in enumerate(hand_lms.landmark):
                    points.append(
                        LandmarkPoint(
                            index=lm_idx,
                            x=float(lm.x),
                            y=float(lm.y),
                            z=float(lm.z),
                        )
                    )

                detected_hands.append(
                    HandLandmarks(
                        handedness=handedness_label,
                        landmarks=points,
                    )
                )

                if draw and self._mp_drawing and self._mp_hands:
                    self._mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_lms,
                        self._mp_hands.HAND_CONNECTIONS,
                        self._mp_drawing_styles.get_default_hand_landmarks_style(),
                        self._mp_drawing_styles.get_default_hand_connections_style(),
                    )

        return detected_hands, annotated_frame

    def run_live_tracking(
        self,
        camera_index: int = 0,
        window_name: str = "SignSyntax - Phase 1 Hand Tracker",
    ) -> None:
        """
        Launches an interactive live webcam tracking session.

        Press 'Q' or 'ESC' to exit the session cleanly.

        Args:
            camera_index (int): Camera device index (default: 0).
            window_name (str): Title of the OpenCV window.
        """
        if not _VISION_DEPS_AVAILABLE:
            print("Error: opencv-python and mediapipe are required for live tracking.")
            return

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open webcam at camera index {camera_index}.")
            return

        print("=" * 60)
        print("    SignSyntax: Phase 1 MediaPipe Hand Tracking Active    ")
        print("=" * 60)
        print("  - Detecting up to 2 hands (21 landmarks each)")
        print("  - Press 'Q' or 'ESC' in the window to exit")
        print("=" * 60 + "\n")

        frame_count = 0

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Warning: Empty frame received from webcam.")
                    continue

                # Flip horizontally for natural mirror display
                frame = cv2.flip(frame, 1)

                # Process hand landmarks
                hands, annotated = self.process_frame(frame, draw=True)
                frame_count += 1

                # Display HUD overlay
                h_count = len(hands)
                hud_text = f"Hands Detected: {h_count}"
                cv2.putText(
                    annotated,
                    hud_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0) if h_count > 0 else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    "Press 'Q' or 'ESC' to exit",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )

                # Print terminal summary periodically when hands are present
                if hands and (frame_count % 15 == 0):
                    print(f"[{frame_count:04d}] Detected {len(hands)} hand(s):")
                    for h_idx, hand in enumerate(hands):
                        wrist = hand.landmarks[0]
                        index_tip = hand.landmarks[8]
                        print(
                            f"  Hand #{h_idx+1} ({hand.handedness}): "
                            f"Wrist({wrist.x:.2f}, {wrist.y:.2f}) | "
                            f"IndexTip({index_tip.x:.2f}, {index_tip.y:.2f}) | "
                            f"21 Landmarks Extracted"
                        )

                cv2.imshow(window_name, annotated)

                # Check for exit key (Q or ESC)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), ord("Q"), 27]:
                    print("\nExiting hand tracker demonstration...")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.close()
            print("Webcam released and windows closed cleanly.")

    def close(self) -> None:
        """Releases underlying MediaPipe Hands resources."""
        if self._hands is not None:
            self._hands.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

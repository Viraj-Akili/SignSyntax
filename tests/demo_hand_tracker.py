"""
SignSyntax - Hand Tracker Live Webcam Demonstration
Phase 1: Computer Vision Input Layer

Demonstrates real-time 21-landmark hand tracking using MediaPipe Hands and OpenCV.
"""

import os
import sys

# Ensure root workspace directory is in sys.path when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recognition.hand_tracker import HandTracker


def main() -> None:
    print("Starting SignSyntax Phase 1 Hand Tracking Demo...")
    tracker = HandTracker(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    tracker.run_live_tracking(camera_index=0)


if __name__ == "__main__":
    main()

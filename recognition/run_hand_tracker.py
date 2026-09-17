"""
SignSyntax - Standalone Hand Tracker Webcam Demo
Phase 1: Computer Vision Input Demonstration

This script provides a standalone interactive webcam demonstration for Phase 1.
It captures live video from the webcam, processes each frame with MediaPipe Hands
via HandTracker, renders skeletal landmarks, and displays an on-screen HUD
with hand counts.

Usage:
    python recognition/run_hand_tracker.py
    or
    python -m recognition.run_hand_tracker
"""

import os
import sys

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

from recognition.hand_tracker import HandTracker


def run_webcam_demo(
    camera_index: int = 0,
    window_name: str = "SignSyntax - Hand Tracking",
) -> None:
    """
    Runs the standalone webcam hand tracking demo.

    Args:
        camera_index (int): Camera device index (default: 0).
        window_name (str): Title of the OpenCV display window.
    """
    if not _CV2_AVAILABLE:
        print("Error: opencv-python is required to run the webcam demo.")
        return

    print("=" * 65)
    print("       SignSyntax: Phase 1 Hand Tracking Webcam Demo       ")
    print("=" * 65)
    print("  - Initializing HandTracker (MediaPipe Hands)...")
    print("  - Press 'Q' or 'ESC' in the video window to exit.")
    print("=" * 65 + "\n")

    tracker = HandTracker()
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open webcam at camera index {camera_index}.")
        tracker.close()
        return

    frame_count = 0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Warning: Failed to capture frame from webcam.")
                continue

            # Mirror the frame horizontally for natural user interaction
            frame = cv2.flip(frame, 1)

            # Process frame using HandTracker
            hands, annotated_frame = tracker.process_frame(frame, draw=True)
            frame_count += 1
            num_hands = len(hands)

            # Render HUD (Heads-Up Display)
            hud_color = (0, 255, 0) if num_hands > 0 else (0, 0, 255)
            hud_text = f"Hands Detected: {num_hands}"
            cv2.putText(
                annotated_frame,
                hud_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                hud_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                "SignSyntax Phase 1 CV Demo | Press 'Q' or 'ESC' to exit",
                (20, annotated_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            # Display processed frame
            cv2.imshow(window_name, annotated_frame)

            # Check for exit key (Q or ESC)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("\nExit key pressed. Stopping webcam stream...")
                break
    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print("Webcam released and OpenCV windows closed successfully.")


def main() -> None:
    """CLI entrypoint for standalone webcam tracking."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SignSyntax Phase 1: Standalone Webcam Hand Tracker Demo"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    args = parser.parse_args()
    run_webcam_demo(camera_index=args.camera)


if __name__ == "__main__":
    main()

"""
SignSyntax - Recognition Package Initialization
Handles vision-based sign language recognition (MediaPipe & Landmark processing)
and dataset sample collection.
"""

from recognition.dataset_collector import (
    DatasetCollector,
    count_existing_samples,
    flatten_landmarks,
    generate_csv_header,
    get_dataset_csv_path,
    save_landmark_sample,
    validate_label,
)
from recognition.hand_tracker import HandLandmarks, HandTracker, LandmarkPoint
from recognition.gesture_classifier import (
    AGGREGATED_FEATURE_DIM,
    ClassificationReport,
    KNearestNeighborsClassifier,
    StandardScaler,
    extract_sequence_features,
    load_dataset_from_sequences,
    stratified_sequence_train_test_split,
    train_gesture_classifier,
)
from recognition.phase1_demo import run_phase1_demo
from recognition.predict_gesture import (
    GestureIntegrationError,
    GesturePredictionResult,
    GesturePredictor,
)
from recognition.wlasl_downloader import (
    DEFAULT_RAW_DIR,
    DownloadResult,
    ProcessingResult,
    display_pipeline_summary,
    download_single_video,
    download_wlasl_subset,
    process_downloaded_subset,
)
from recognition.wlasl_inspector import (
    DEFAULT_PROTOTYPE_GLOSSES,
    WLASLGlossReport,
    WLASLInstance,
    display_wlasl_report,
    inspect_wlasl_glosses,
    load_or_fetch_wlasl_metadata,
    parse_wlasl_entry,
)
from recognition.wlasl_processor import (
    FRAME_FEATURE_DIM,
    HAND_FEATURE_DIM,
    LandmarkSequenceSample,
    format_frame_landmarks,
    get_sequence_output_path,
    process_and_save_video,
    process_video_file_to_sample,
    process_video_frames,
)

__version__ = "0.1.0"

__all__ = [
    "HandTracker",
    "HandLandmarks",
    "LandmarkPoint",
    "DatasetCollector",
    "generate_csv_header",
    "flatten_landmarks",
    "validate_label",
    "get_dataset_csv_path",
    "save_landmark_sample",
    "count_existing_samples",
    "WLASLInstance",
    "WLASLGlossReport",
    "load_or_fetch_wlasl_metadata",
    "parse_wlasl_entry",
    "inspect_wlasl_glosses",
    "display_wlasl_report",
    "DEFAULT_PROTOTYPE_GLOSSES",
    "FRAME_FEATURE_DIM",
    "HAND_FEATURE_DIM",
    "LandmarkSequenceSample",
    "format_frame_landmarks",
    "get_sequence_output_path",
    "process_video_frames",
    "process_video_file_to_sample",
    "process_and_save_video",
    "DownloadResult",
    "ProcessingResult",
    "download_single_video",
    "download_wlasl_subset",
    "process_downloaded_subset",
    "display_pipeline_summary",
    "DEFAULT_RAW_DIR",
    "AGGREGATED_FEATURE_DIM",
    "StandardScaler",
    "KNearestNeighborsClassifier",
    "ClassificationReport",
    "extract_sequence_features",
    "load_dataset_from_sequences",
    "stratified_sequence_train_test_split",
    "train_gesture_classifier",
    "GesturePredictor",
    "GesturePredictionResult",
    "GestureIntegrationError",
    "run_phase1_demo",
]




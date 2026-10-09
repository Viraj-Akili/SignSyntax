"""
SignSyntax - WLASL 7-Class Dataset Builder & Retraining Pipeline
Phase 1: Computer Vision Input & Gesture Classification

Reproducible, end-to-end pipeline that:
1. Parses WLASL_v0.3.json metadata for the 7 target glosses
   (BOOK, HOME, ME, MILK, NOT, WANT, YOU).
2. Downloads up to --max-per-gloss videos per class from WLASL source URLs
   into dataset/wlasl_raw/<GLOSS>/<video_id>.mp4.
3. Extracts 126-dimensional multi-hand landmark sequences per frame using
   the existing HandTracker and stores them as JSON in
   dataset/wlasl_sequences/<GLOSS>/<video_id>.json.
4. Retrains the k-NN classifier (k=3, StandardScaler, Euclidean distance)
   on the full expanded dataset with a fixed random seed (42).
5. Evaluates on a held-out stratified test split and prints:
   - Per-class accuracy, precision, recall, F1
   - Confusion matrix
   - Prediction distribution
6. Saves the retrained model to models/gesture_knn.json and .pkl.

Usage:
    python recognition/build_wlasl_7class_dataset.py
    python recognition/build_wlasl_7class_dataset.py --max-per-gloss 15 --skip-download
    python recognition/build_wlasl_7class_dataset.py --skip-download --skip-processing

Guarantees:
- Training and inference share identical preprocessing
  (HandTracker -> 126-dim frames -> mean/std -> 252-dim -> StandardScaler -> k-NN).
- Video failures are reported but do not halt the pipeline.
- Existing sequence JSONs are preserved unless overwritten by reprocessing.
- Fixed random_state=42 ensures reproducibility.

Citation & Attribution:
WLASL is provided under the Computational Use of Data Agreement (C-UDA).
Reference: Dongxu Li et al., WACV 2020. https://github.com/dxli94/WLASL
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recognition.wlasl_inspector import (
    DEFAULT_PROTOTYPE_GLOSSES,
    WLASLGlossReport,
    WLASLInstance,
    display_wlasl_report,
    inspect_wlasl_glosses,
    load_or_fetch_wlasl_metadata,
)
from recognition.wlasl_downloader import (
    download_wlasl_subset,
    process_downloaded_subset,
    display_pipeline_summary,
    DownloadResult,
)
from recognition.gesture_classifier import (
    AGGREGATED_FEATURE_DIM,
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_PKL_PATH,
    DEFAULT_SEQUENCES_DIR,
    KNearestNeighborsClassifier,
    load_dataset_from_sequences,
    stratified_sequence_train_test_split,
    train_gesture_classifier,
)


# == Constants ==================================================================
TARGET_GLOSSES = ["BOOK", "HOME", "ME", "MILK", "NOT", "WANT", "YOU"]
DEFAULT_RAW_DIR = os.path.join("dataset", "wlasl_raw")
RANDOM_STATE = 42
K_NEIGHBORS = 3
TEST_SIZE = 0.25


def banner(title: str) -> None:
    """Prints a formatted banner."""
    width = 78
    print()
    print("=" * width)
    print(f"  {title}".center(width))
    print("=" * width)


# ===============================================================================
#  STEP 1: METADATA INSPECTION
# ===============================================================================

def step1_inspect_metadata(
    metadata: List[Dict[str, Any]],
) -> Dict[str, WLASLGlossReport]:
    """Inspects WLASL metadata and prints availability report for all 7 glosses."""
    banner("STEP 1 / 5 -- WLASL Metadata Inspection")

    reports = inspect_wlasl_glosses(TARGET_GLOSSES, metadata)
    print(display_wlasl_report(reports))

    total_available = sum(1 for r in reports.values() if r.is_available)
    total_instances = sum(r.sample_count for r in reports.values())
    print(f"\n  Available: {total_available}/{len(TARGET_GLOSSES)} glosses")
    print(f"  Total WLASL instances across all glosses: {total_instances}")

    return reports


# ===============================================================================
#  STEP 2: DOWNLOAD VIDEOS
# ===============================================================================

def step2_download_videos(
    metadata: List[Dict[str, Any]],
    raw_dir: str = DEFAULT_RAW_DIR,
    max_per_gloss: int = 10,
) -> Dict[str, List[DownloadResult]]:
    """Downloads up to max_per_gloss videos per gloss from WLASL source URLs."""
    banner("STEP 2 / 5 -- Video Download")

    dl_results = download_wlasl_subset(
        target_glosses=TARGET_GLOSSES,
        raw_dir=raw_dir,
        max_per_gloss=max_per_gloss,
        metadata=metadata,
    )

    # Summary
    total_success = sum(
        sum(1 for d in dl_list if d.success)
        for dl_list in dl_results.values()
    )
    total_failed = sum(
        sum(1 for d in dl_list if not d.success)
        for dl_list in dl_results.values()
    )
    print(f"\n  Total downloaded: {total_success}")
    print(f"  Total failed:     {total_failed}")

    return dl_results


# ===============================================================================
#  STEP 3: PROCESS VIDEOS INTO LANDMARK SEQUENCES
# ===============================================================================

def step3_process_videos(
    dl_results: Dict[str, List[DownloadResult]],
    metadata: List[Dict[str, Any]],
    sequences_dir: str = DEFAULT_SEQUENCES_DIR,
) -> None:
    """Processes downloaded videos into 126-dim landmark sequences."""
    banner("STEP 3 / 5 -- Landmark Sequence Extraction (126-dim/frame)")

    proc_results = process_downloaded_subset(
        download_results=dl_results,
        metadata=metadata,
        output_sequences_dir=sequences_dir,
    )

    print("\n" + display_pipeline_summary(dl_results, proc_results))

    return proc_results


def step3_process_from_raw(
    raw_dir: str = DEFAULT_RAW_DIR,
    sequences_dir: str = DEFAULT_SEQUENCES_DIR,
    metadata: Optional[List[Dict[str, Any]]] = None,
    force_reprocess: bool = False,
) -> None:
    """
    Processes all raw video files in wlasl_raw/ into sequence JSONs
    in wlasl_sequences/ using the corrected HandTracker pipeline.
    """
    banner("STEP 3 / 5 -- Processing Raw Videos")

    try:
        from recognition.hand_tracker import HandTracker
        from recognition.wlasl_processor import process_and_save_video
    except ImportError as e:
        print(f"  SKIP: Required libraries not available ({e})")
        return

    if metadata is None:
        metadata = load_or_fetch_wlasl_metadata()

    # Build instance metadata index
    instance_meta: Dict[str, Dict[str, Any]] = {}
    for entry in metadata:
        for inst in entry.get("instances", []):
            vid = str(inst.get("video_id", ""))
            instance_meta[vid] = inst

    tracker = None  # Lazy init
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for gloss in TARGET_GLOSSES:
        gloss_raw_dir = os.path.join(raw_dir, gloss)
        if not os.path.isdir(gloss_raw_dir):
            continue

        video_files = sorted([
            f for f in os.listdir(gloss_raw_dir)
            if f.endswith((".mp4", ".mov", ".webm", ".avi"))
        ])

        for vf in video_files:
            video_id = os.path.splitext(vf)[0]
            seq_path = os.path.join(sequences_dir, gloss, f"{video_id}.json")

            if os.path.isfile(seq_path) and not force_reprocess:
                skipped_count += 1
                continue

            # Lazy init tracker
            if tracker is None:
                print("  Initializing HandTracker...")
                tracker = HandTracker(static_image_mode=False, max_num_hands=2)

            video_path = os.path.join(gloss_raw_dir, vf)
            meta = instance_meta.get(video_id, {})
            f_start = int(meta.get("frame_start", 1))
            f_end = int(meta.get("frame_end", -1))
            split = str(meta.get("split", "train"))
            signer = int(meta.get("signer_id", 0))

            print(f"  -> Processing {gloss}/{vf} (frames {f_start}..{f_end}): ", end="", flush=True)

            try:
                sample, saved_path = process_and_save_video(
                    video_path=video_path,
                    gloss=gloss,
                    wlasl_gloss=gloss.lower(),
                    video_id=video_id,
                    output_dir=sequences_dir,
                    frame_start=f_start,
                    frame_end=f_end,
                    split=split,
                    signer_id=signer,
                    tracker=tracker,
                    require_hands=True,
                )
                print(f"DONE ({sample.frame_count} frames)")
                processed_count += 1
            except Exception as e:
                print(f"FAILED: {e}")
                failed_count += 1

    print(f"\n  Newly processed: {processed_count}")
    print(f"  Already existed: {skipped_count}")
    print(f"  Failed:          {failed_count}")


# ===============================================================================
#  STEP 4: DATASET INVENTORY
# ===============================================================================

def step4_dataset_inventory(
    sequences_dir: str = DEFAULT_SEQUENCES_DIR,
) -> Tuple[int, Dict[str, int]]:
    """Reports the current dataset inventory from wlasl_sequences/ including zero-frame count."""
    banner("STEP 4 / 5 -- Dataset Inventory & Zero-Frame Audit")

    class_counts: Dict[str, int] = {}
    total_sequences = 0
    total_all_frames = 0
    total_zero_frames = 0

    import glob
    print(f"\n  {'CLASS':<8} | {'SEQUENCES':<9} | {'TOTAL FRAMES':<12} | {'MIN':<5} | {'MAX':<5} | {'AVG FRAMES':<10} | {'ZERO FRAMES'}")
    print(f"  {'-'*8}-+-{'-'*9}-+-{'-'*12}-+-{'-'*5}-+-{'-'*5}-+-{'-'*10}-+-{'-'*11}")

    for gloss in TARGET_GLOSSES:
        gloss_dir = os.path.join(sequences_dir, gloss)
        seq_files = glob.glob(os.path.join(gloss_dir, "*.json")) if os.path.isdir(gloss_dir) else []
        count = len(seq_files)
        class_counts[gloss] = count
        total_sequences += count

        total_frames = 0
        zero_frames = 0
        frame_counts = []
        for sf in seq_files:
            try:
                with open(sf, "r", encoding="utf-8") as f:
                    sample_data = json.load(f)
                f_list = sample_data.get("frames", [])
                frame_counts.append(len(f_list))
                total_frames += len(f_list)
                zero_frames += sum(1 for fvec in f_list if all(v == 0.0 for v in fvec))
            except Exception:
                pass

        min_f = min(frame_counts) if frame_counts else 0
        max_f = max(frame_counts) if frame_counts else 0
        avg_f = (total_frames / count) if count > 0 else 0.0
        total_all_frames += total_frames
        total_zero_frames += zero_frames

        print(f"  {gloss:<8} | {count:<9} | {total_frames:<12} | {min_f:<5} | {max_f:<5} | {avg_f:<10.1f} | {zero_frames}")

    print(f"  {'-'*8}-+-{'-'*9}-+-{'-'*12}-+-{'-'*5}-+-{'-'*5}-+-{'-'*10}-+-{'-'*11}")
    avg_total = (total_all_frames / total_sequences) if total_sequences > 0 else 0.0
    print(f"  {'TOTAL':<8} | {total_sequences:<9} | {total_all_frames:<12} |       |       | {avg_total:<10.1f} | {total_zero_frames}")

    return total_sequences, class_counts


# ===============================================================================
#  STEP 5: TRAIN & EVALUATE CLASSIFIER
# ===============================================================================

def step5_train_and_evaluate(
    sequences_dir: str = DEFAULT_SEQUENCES_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
    k: int = K_NEIGHBORS,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[Optional[KNearestNeighborsClassifier], Optional[Any]]:
    """Trains k-NN on the 7-class dataset and evaluates on a held-out split."""
    banner("STEP 5 / 5 -- Train & Evaluate k-NN Classifier")

    # Load all sequences
    X, y, video_ids = load_dataset_from_sequences(sequences_dir, TARGET_GLOSSES)

    if len(X) == 0:
        print("  ERROR: No sequences found. Cannot train.")
        return None, None

    print(f"\n  Loaded {len(X)} total sequences across {len(set(y))} classes.")
    print(f"  Feature dimensionality: {X.shape[1]} (126 mean + 126 std)")

    # Class distribution
    class_counts: Dict[str, int] = {}
    for label in y:
        class_counts[label] = class_counts.get(label, 0) + 1

    print(f"\n  Class Distribution:")
    for gloss in sorted(class_counts.keys()):
        count = class_counts[gloss]
        pct = count / len(y) * 100
        print(f"    {gloss:<8}: {count:>3} sequences ({pct:5.1f}%)")

    # Stratified split
    X_train, X_test, y_train, y_test = stratified_sequence_train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    train_classes = set(y_train)
    test_classes = set(y_test)

    print(f"\n  Train/Test Split (random_state={random_state}, test_size={test_size}):")
    print(f"    Training:  {len(X_train)} samples ({len(train_classes)} classes)")
    print(f"    Testing:   {len(X_test)} samples ({len(test_classes)} classes)")

    # Train class breakdown
    train_dist: Dict[str, int] = {}
    for label in y_train:
        train_dist[label] = train_dist.get(label, 0) + 1
    test_dist: Dict[str, int] = {}
    for label in y_test:
        test_dist[label] = test_dist.get(label, 0) + 1

    print(f"\n    {'Gloss':<8} | {'Train':<6} | {'Test':<6}")
    print(f"    {'-'*8}-+-{'-'*6}-+-{'-'*6}")
    for gloss in sorted(class_counts.keys()):
        tr = train_dist.get(gloss, 0)
        te = test_dist.get(gloss, 0)
        print(f"    {gloss:<8} | {tr:<6} | {te:<6}")

    # Train k-NN
    print(f"\n  Fitting k-NN (k={k}, metric=euclidean, scaling=StandardScaler)...")
    clf = KNearestNeighborsClassifier(k=k)
    clf.fit(X_train, y_train, scale=True)

    # Evaluate
    report = clf.evaluate(X_test, y_test)
    print("\n" + report.format_table())

    # Confusion matrix
    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    all_classes = sorted(report.classes)
    header = "         " + "  ".join(f"{c[:4]:>5}" for c in all_classes)
    print(f"    {header}")
    for actual in all_classes:
        row_vals = "  ".join(
            f"{report.confusion_matrix[actual].get(pred, 0):>5}"
            for pred in all_classes
        )
        print(f"    {actual:<8} {row_vals}")

    # Per-class prediction distribution on test set
    predictions = clf.predict(X_test)
    pred_dist: Dict[str, int] = {}
    for p in predictions:
        pred_dist[p] = pred_dist.get(p, 0) + 1

    print(f"\n  Prediction Distribution (test set):")
    for gloss in sorted(pred_dist.keys()):
        count = pred_dist[gloss]
        pct = count / len(predictions) * 100
        print(f"    {gloss:<8}: {count:>3} predictions ({pct:5.1f}%)")

    # Save model
    print(f"\n  Saving model to {model_path} ...")
    clf.save(json_path=model_path, pkl_path=DEFAULT_MODEL_PKL_PATH)
    print(f"  [OK] Model saved successfully.")
    print(f"    JSON: {model_path}")
    print(f"    PKL:  {DEFAULT_MODEL_PKL_PATH}")
    print(f"    Classes: {clf.classes_}")
    print(f"    Training samples: {len(clf.X_train)}")

    return clf, report


# ===============================================================================
#  MAIN PIPELINE
# ===============================================================================

def run_pipeline(
    max_per_gloss: int = 10,
    raw_dir: str = DEFAULT_RAW_DIR,
    sequences_dir: str = DEFAULT_SEQUENCES_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
    skip_download: bool = False,
    skip_processing: bool = False,
    skip_training: bool = False,
    force_reprocess: bool = False,
) -> None:
    """Runs the complete 7-class WLASL dataset building and retraining pipeline."""

    banner("SignSyntax: 7-Class WLASL Dataset Builder & Retrainer")
    print(f"  Target glosses: {', '.join(TARGET_GLOSSES)}")
    print(f"  Max per gloss:  {max_per_gloss}")
    print(f"  Raw dir:        {raw_dir}")
    print(f"  Sequences dir:  {sequences_dir}")
    print(f"  Model path:     {model_path}")
    print(f"  Skip download:  {skip_download}")
    print(f"  Skip process:   {skip_processing}")
    print(f"  Skip training:  {skip_training}")
    print(f"  Force reprocess:{force_reprocess}")

    start_time = time.time()

    # Load metadata
    print("\n  Loading WLASL metadata...")
    metadata = load_or_fetch_wlasl_metadata()
    print(f"  Loaded {len(metadata)} total WLASL glosses.")

    # Step 1: Inspect
    reports = step1_inspect_metadata(metadata)

    # Step 2: Download
    if not skip_download:
        dl_results = step2_download_videos(metadata, raw_dir, max_per_gloss)

        # Step 3: Process
        if not skip_processing:
            step3_process_videos(dl_results, metadata, sequences_dir)
    else:
        print("\n  [SKIPPED] Step 2 -- Video download")
        if not skip_processing:
            # Process any raw videos that don't have sequences yet
            step3_process_from_raw(raw_dir, sequences_dir, metadata, force_reprocess=force_reprocess)
        else:
            print("  [SKIPPED] Step 3 -- Video processing")

    # Step 4: Inventory
    total_seqs, class_counts = step4_dataset_inventory(sequences_dir)

    # Step 5: Train
    if not skip_training and total_seqs > 0:
        clf, report = step5_train_and_evaluate(
            sequences_dir=sequences_dir,
            model_path=model_path,
        )
    else:
        if skip_training:
            print("\n  [SKIPPED] Step 5 -- Training")
        elif total_seqs == 0:
            print("\n  [SKIPPED] Step 5 -- No sequences available for training")

    elapsed = time.time() - start_time

    banner("Pipeline Complete")
    print(f"  Total sequences: {total_seqs}")
    print(f"  Classes:         {len([c for c in class_counts.values() if c > 0])}/{len(TARGET_GLOSSES)}")
    print(f"  Elapsed time:    {elapsed:.1f}s")
    print()


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="SignSyntax: 7-Class WLASL Dataset Builder & Retrainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: download + process + train
  python recognition/build_wlasl_7class_dataset.py

  # Download more videos (up to 15 per class)
  python recognition/build_wlasl_7class_dataset.py --max-per-gloss 15

  # Skip download, just reprocess and retrain
  python recognition/build_wlasl_7class_dataset.py --skip-download --force-reprocess

  # Only retrain from existing sequences (no download or processing)
  python recognition/build_wlasl_7class_dataset.py --skip-download --skip-processing

  # Only inspect and inventory (no training)
  python recognition/build_wlasl_7class_dataset.py --skip-download --skip-processing --skip-training
""",
    )
    parser.add_argument(
        "--max-per-gloss",
        type=int,
        default=10,
        help="Maximum videos to download per gloss (default: 10)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=DEFAULT_RAW_DIR,
        help="Directory for raw video files (default: dataset/wlasl_raw)",
    )
    parser.add_argument(
        "--sequences-dir",
        type=str,
        default=DEFAULT_SEQUENCES_DIR,
        help="Directory for sequence JSONs (default: dataset/wlasl_sequences)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Output model JSON path (default: models/gesture_knn.json)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip video downloading (use existing raw files)",
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip video processing (use existing sequence JSONs)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip classifier training (only inspect and inventory)",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing of raw videos even if sequence JSONs exist",
    )

    args = parser.parse_args()

    run_pipeline(
        max_per_gloss=args.max_per_gloss,
        raw_dir=args.raw_dir,
        sequences_dir=args.sequences_dir,
        model_path=args.model_path,
        skip_download=args.skip_download,
        skip_processing=args.skip_processing,
        skip_training=args.skip_training,
        force_reprocess=args.force_reprocess,
    )


if __name__ == "__main__":
    main()

"""
SignSyntax - WLASL Video Dataset Downloader & Subset Preparer
Phase 1: Computer Vision Input & WLASL Dataset Integration

This module downloads a small, manageable representative subset of WLASL sign video files
(up to 10 samples per target prototype gloss) into dataset/wlasl_raw/<GLOSS>/<video_id>.mp4
and processes them into 126-dimensional temporal landmark sequences via wlasl_processor.py.

Key Guarantees:
1. Selective Subset: Downloads only required glosses (ME, YOU, WANT, MILK, BOOK, HOME, NOT),
   never downloading the entire WLASL dataset.
2. Error Transparency: Non-silent reporting of every HTTP error, timeout, or invalid video file.
3. Decoupled Pipeline: Videos are saved to dataset/wlasl_raw/ and landmark sequences to
   dataset/wlasl_sequences/, keeping manual single-frame CSV datasets (ME, YOU) untouched.
4. Temporal Multi-Hand: Preserves 126-dim features per frame and temporal sequence structure.

Citation & Attribution:
WLASL is provided under the Computational Use of Data Agreement (C-UDA).
Reference: Dongxu Li et al., WACV 2020. https://github.com/dxli94/WLASL
"""

import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.symbol_table import SymbolTable
from recognition.hand_tracker import HandTracker
from recognition.wlasl_inspector import DEFAULT_PROTOTYPE_GLOSSES, load_or_fetch_wlasl_metadata, parse_wlasl_entry
from recognition.wlasl_processor import (
    DEFAULT_SEQUENCES_DIR,
    DEFAULT_VIDEOS_DIR,
    LandmarkSequenceSample,
    process_and_save_video,
)

DEFAULT_RAW_DIR = os.path.join("dataset", "wlasl_raw")


@dataclass
class DownloadResult:
    """Represents the outcome of a single video download attempt."""
    gloss: str
    video_id: str
    url: str
    success: bool
    filepath: str
    error_message: str = ""
    bytes_downloaded: int = 0


@dataclass
class ProcessingResult:
    """Represents the outcome of processing a downloaded video into landmark sequence JSON."""
    gloss: str
    video_id: str
    video_path: str
    success: bool
    sequence_path: str
    frame_count: int = 0
    error_message: str = ""


def download_single_video(
    url: str,
    output_path: str,
    timeout: int = 15,
) -> Tuple[bool, int, str]:
    """
    Downloads a single video file from a URL to output_path.

    Args:
        url (str): Source URL of the video file.
        output_path (str): Destination path on the local filesystem.
        timeout (int): Timeout in seconds.

    Returns:
        Tuple[bool, int, str]: (success, bytes_downloaded, error_message)
    """
    if os.path.isfile(output_path) and os.path.getsize(output_path) > 1024:
        # Already downloaded
        return True, os.path.getsize(output_path), "Cached"

    # Only attempt direct HTTP/HTTPS media URLs (skip raw SWF or non-HTTP unless yt-dlp exists)
    if not url.startswith("http://") and not url.startswith("https://"):
        return False, 0, f"Unsupported URL scheme: {url}"

    if url.endswith(".swf"):
        return False, 0, "Flash SWF files are not supported as MP4 video streams"

    # Handle YouTube URLs
    if "youtube.com" in url or "youtu.be" in url:
        try:
            import yt_dlp
            ydl_opts = {
                "outtmpl": output_path,
                "format": "mp4[height<=480]/best[ext=mp4]/best",
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                return True, os.path.getsize(output_path), ""
            return False, 0, "YouTube download produced no file"
        except ImportError:
            return False, 0, "YouTube URL skipped (yt_dlp not installed)"
        except Exception as e:
            return False, 0, f"YouTube download error: {str(e)}"

    # Direct media URL download with SSL context
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }

    req = urllib.request.Request(url, headers=headers)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as response:
            data = response.read()
            if len(data) < 1024:
                return False, len(data), f"File too small ({len(data)} bytes, likely an error page)"
            with open(output_path, "wb") as f:
                f.write(data)
            return True, len(data), ""
    except urllib.error.HTTPError as e:
        return False, 0, f"HTTP Error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return False, 0, f"URL Error: {e.reason}"
    except TimeoutError:
        return False, 0, f"Connection timed out after {timeout}s"
    except Exception as e:
        return False, 0, f"Unexpected error: {str(e)}"


def download_wlasl_subset(
    target_glosses: List[str] = DEFAULT_PROTOTYPE_GLOSSES,
    raw_dir: str = DEFAULT_RAW_DIR,
    max_per_gloss: int = 10,
    metadata: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[DownloadResult]]:
    """
    Downloads up to max_per_gloss videos for each requested gloss from WLASL metadata.

    Args:
        target_glosses (List[str]): List of uppercase SignSyntax glosses.
        raw_dir (str): Directory where raw videos will be stored (<raw_dir>/<GLOSS>/<video_id>.mp4).
        max_per_gloss (int): Maximum successful video downloads per gloss.
        metadata (Optional[List[Dict[str, Any]]]): Preloaded WLASL metadata.

    Returns:
        Dict[str, List[DownloadResult]]: Download attempt results grouped by gloss.
    """
    if metadata is None:
        metadata = load_or_fetch_wlasl_metadata()

    # Index instances by lowercase gloss
    wlasl_dict: Dict[str, List[Dict[str, Any]]] = {}
    for entry in metadata:
        g = entry.get("gloss", "").lower().strip()
        wlasl_dict[g] = entry.get("instances", [])

    results_by_gloss: Dict[str, List[DownloadResult]] = {}

    print("\n" + "=" * 75)
    print("       SignSyntax: Downloading WLASL Prototype Subset Videos        ")
    print("=" * 75)

    for req_gloss in target_glosses:
        normalized = req_gloss.lower().strip()
        instances = wlasl_dict.get(normalized, [])
        results_by_gloss[req_gloss] = []

        print(f"\n[GLOSS: {req_gloss}] Found {len(instances)} instances in WLASL metadata. Target: up to {max_per_gloss} videos.")

        successful_downloads = 0

        for inst in instances:
            if successful_downloads >= max_per_gloss:
                break

            video_id = str(inst.get("video_id", ""))
            url = str(inst.get("url", ""))

            # Choose file extension
            ext = ".mp4"
            if url.endswith(".mov"):
                ext = ".mov"
            elif url.endswith(".webm"):
                ext = ".webm"

            dest_path = os.path.join(raw_dir, req_gloss, f"{video_id}{ext}")

            print(f"  -> Attempting video {video_id} ({url[:60]}...): ", end="", flush=True)

            success, size_bytes, err = download_single_video(url, dest_path)

            if success:
                successful_downloads += 1
                print(f"SUCCESS ({size_bytes // 1024} KB)")
                results_by_gloss[req_gloss].append(
                    DownloadResult(
                        gloss=req_gloss,
                        video_id=video_id,
                        url=url,
                        success=True,
                        filepath=dest_path,
                        bytes_downloaded=size_bytes,
                    )
                )
            else:
                print(f"FAILED: {err}")
                results_by_gloss[req_gloss].append(
                    DownloadResult(
                        gloss=req_gloss,
                        video_id=video_id,
                        url=url,
                        success=False,
                        filepath=dest_path,
                        error_message=err,
                    )
                )

        print(f"  Summary for {req_gloss}: {successful_downloads}/{min(len(instances), max_per_gloss)} videos obtained.")

    return results_by_gloss


def process_downloaded_subset(
    download_results: Dict[str, List[DownloadResult]],
    metadata: Optional[List[Dict[str, Any]]] = None,
    output_sequences_dir: str = DEFAULT_SEQUENCES_DIR,
) -> Dict[str, List[ProcessingResult]]:
    """
    Processes all successfully downloaded videos into 126-dimensional multi-hand landmark sequences.

    Args:
        download_results (Dict[str, List[DownloadResult]]): Mapping of gloss to download results.
        metadata (Optional[List[Dict[str, Any]]]): WLASL metadata to retrieve frame_start and frame_end.
        output_sequences_dir (str): Base destination directory for sequence JSONs.

    Returns:
        Dict[str, List[ProcessingResult]]: Processing results per gloss.
    """
    if metadata is None:
        metadata = load_or_fetch_wlasl_metadata()

    # Index metadata instances by video_id
    instance_meta: Dict[str, Dict[str, Any]] = {}
    for entry in metadata:
        for inst in entry.get("instances", []):
            vid = str(inst.get("video_id", ""))
            instance_meta[vid] = inst

    tracker = HandTracker(static_image_mode=False, max_num_hands=2)
    proc_results: Dict[str, List[ProcessingResult]] = {}

    print("\n" + "=" * 75)
    print("      SignSyntax: Extracting Multi-Hand Landmark Sequences (126-dim)     ")
    print("=" * 75)

    for gloss, downloads in download_results.items():
        proc_results[gloss] = []
        valid_downloads = [d for d in downloads if d.success and os.path.isfile(d.filepath)]

        print(f"\n[PROCESSING GLOSS: {gloss}] ({len(valid_downloads)} videos to process)")

        for d in valid_downloads:
            meta = instance_meta.get(d.video_id, {})
            f_start = int(meta.get("frame_start", 1))
            f_end = int(meta.get("frame_end", -1))
            split = str(meta.get("split", "train"))
            signer = int(meta.get("signer_id", 0))

            print(f"  -> Processing video {d.video_id} (frames {f_start}..{f_end}): ", end="", flush=True)

            try:
                sample, saved_path = process_and_save_video(
                    video_path=d.filepath,
                    gloss=gloss,
                    wlasl_gloss=gloss.lower(),
                    video_id=d.video_id,
                    output_dir=output_sequences_dir,
                    frame_start=f_start,
                    frame_end=f_end,
                    split=split,
                    source_url=d.url,
                    signer_id=signer,
                    tracker=tracker,
                )
                print(f"DONE ({sample.frame_count} frames extracted -> {saved_path})")
                proc_results[gloss].append(
                    ProcessingResult(
                        gloss=gloss,
                        video_id=d.video_id,
                        video_path=d.filepath,
                        success=True,
                        sequence_path=saved_path,
                        frame_count=sample.frame_count,
                    )
                )
            except Exception as e:
                print(f"FAILED: {str(e)}")
                proc_results[gloss].append(
                    ProcessingResult(
                        gloss=gloss,
                        video_id=d.video_id,
                        video_path=d.filepath,
                        success=False,
                        sequence_path="",
                        error_message=str(e),
                    )
                )

    return proc_results


def display_pipeline_summary(
    download_results: Dict[str, List[DownloadResult]],
    proc_results: Dict[str, List[ProcessingResult]],
) -> str:
    """
    Formats the complete download & processing pipeline results into an ASCII report.

    Args:
        download_results: Download results.
        proc_results: Processing results.

    Returns:
        str: Summary table.
    """
    lines: List[str] = [
        "=" * 85,
        "         SignSyntax: WLASL Prototype Subset Preparation Summary Report          ",
        "=" * 85,
        f"{'Gloss':<10} | {'Downloaded':<12} | {'Processed':<12} | {'Failed DL':<12} | {'Failed Proc':<12} | {'Total Frames':<12}",
        "-" * 85,
    ]

    total_downloaded = 0
    total_processed = 0
    total_failed_dl = 0
    total_failed_proc = 0
    total_frames = 0

    for gloss in download_results:
        dl_list = download_results.get(gloss, [])
        pr_list = proc_results.get(gloss, [])

        num_dl = sum(1 for d in dl_list if d.success)
        num_fail_dl = sum(1 for d in dl_list if not d.success)
        num_pr = sum(1 for p in pr_list if p.success)
        num_fail_pr = sum(1 for p in pr_list if not p.success)
        frames = sum(p.frame_count for p in pr_list if p.success)

        total_downloaded += num_dl
        total_failed_dl += num_fail_dl
        total_processed += num_pr
        total_failed_proc += num_fail_pr
        total_frames += frames

        lines.append(
            f"{gloss:<10} | {num_dl:<12} | {num_pr:<12} | {num_fail_dl:<12} | {num_fail_pr:<12} | {frames:<12}"
        )

    lines.append("-" * 85)
    lines.append(
        f"{'TOTAL':<10} | {total_downloaded:<12} | {total_processed:<12} | {total_failed_dl:<12} | {total_failed_proc:<12} | {total_frames:<12}"
    )
    lines.append("=" * 85)
    return "\n".join(lines)


def main() -> None:
    """CLI entry point for downloading and processing the WLASL prototype subset."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SignSyntax Phase 1: WLASL Prototype Subset Downloader & Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--glosses",
        type=str,
        default=",".join(DEFAULT_PROTOTYPE_GLOSSES),
        help="Comma-separated list of SignSyntax glosses (default: ME,YOU,WANT,MILK,BOOK,HOME,NOT)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum videos to download and process per gloss (default: 10)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=DEFAULT_RAW_DIR,
        help="Directory to store raw video files (default: dataset/wlasl_raw)",
    )
    parser.add_argument(
        "--sequences-dir",
        type=str,
        default=DEFAULT_SEQUENCES_DIR,
        help="Directory to store processed sequence JSONs (default: dataset/wlasl_sequences)",
    )

    args = parser.parse_args()
    target_glosses = [g.strip().upper() for g in args.glosses.split(",") if g.strip()]

    print("Loading WLASL metadata...")
    metadata = load_or_fetch_wlasl_metadata()

    # Step 1: Download subset
    dl_results = download_wlasl_subset(
        target_glosses=target_glosses,
        raw_dir=args.raw_dir,
        max_per_gloss=args.max_samples,
        metadata=metadata,
    )

    # Step 2: Process videos into landmark sequences
    proc_results = process_downloaded_subset(
        download_results=dl_results,
        metadata=metadata,
        output_sequences_dir=args.sequences_dir,
    )

    # Step 3: Print summary report
    print("\n" + display_pipeline_summary(dl_results, proc_results))


if __name__ == "__main__":
    main()

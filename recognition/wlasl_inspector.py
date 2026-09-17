"""
SignSyntax - WLASL Dataset Metadata Inspector
Phase 1: Computer Vision Input & WLASL Dataset Integration

Inspects the official WLASL (Word-Level American Sign Language) dataset metadata (WLASL_v0.3.json)
to discover and verify which SignSyntax glosses are present in WLASL, report the exact instance counts,
and extract video IDs / URLs without downloading full video archives.

Citation & Attribution:
WLASL is provided under the Computational Use of Data Agreement (C-UDA).
Reference: Dongxu Li et al., "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison", WACV 2020.
https://github.com/dxli94/WLASL
"""

import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure root workspace directory is in sys.path when executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from compiler.symbol_table import SymbolTable

# Official raw URL for WLASL v0.3 metadata annotations
WLASL_METADATA_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
DEFAULT_METADATA_CACHE = os.path.join("dataset", "WLASL_v0.3.json")

# Default target prototype glosses
DEFAULT_PROTOTYPE_GLOSSES = ["ME", "YOU", "WANT", "MILK", "BOOK", "HOME", "NOT"]


@dataclass(frozen=True)
class WLASLInstance:
    """
    Represents a single video sample instance in WLASL.

    Attributes:
        instance_id (int): Unique instance index.
        video_id (str): Video identification hash.
        url (str): Source video URL (YouTube, ASL-LEX, etc.).
        frame_start (int): Starting frame index (1-based in WLASL).
        frame_end (int): Ending frame index (-1 if entire video).
        split (str): Dataset split ('train', 'val', 'test').
        signer_id (int): Signer ID identifier.
    """
    instance_id: int
    video_id: str
    url: str
    frame_start: int
    frame_end: int
    split: str
    signer_id: int


@dataclass(frozen=True)
class WLASLGlossReport:
    """
    Summary report for a requested SignSyntax gloss in WLASL.

    Attributes:
        requested_gloss (str): The uppercase SignSyntax gloss (e.g. 'BOOK', 'ME').
        matched_wlasl_gloss (Optional[str]): The exact gloss key in WLASL (e.g. 'book').
        is_available (bool): True if present in WLASL.
        sample_count (int): Number of available video instances.
        instances (List[WLASLInstance]): List of video sample metadata.
    """
    requested_gloss: str
    matched_wlasl_gloss: Optional[str]
    is_available: bool
    sample_count: int
    instances: List[WLASLInstance]


def load_or_fetch_wlasl_metadata(
    cache_path: str = DEFAULT_METADATA_CACHE,
    url: str = WLASL_METADATA_URL,
    force_download: bool = False,
) -> List[Dict[str, Any]]:
    """
    Loads WLASL metadata from local cache or downloads it from GitHub if absent.

    Args:
        cache_path (str): Local path to cache WLASL_v0.3.json.
        url (str): Remote URL if cache is missing.
        force_download (bool): If True, re-downloads even if cached.

    Returns:
        List[Dict[str, Any]]: Parsed WLASL metadata JSON list.
    """
    if os.path.isfile(cache_path) and not force_download:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"Fetching WLASL metadata from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (SignSyntax Compiler Design Project)"})
    with urllib.request.urlopen(req, timeout=30) as response:
        content = response.read().decode("utf-8")

    data = json.loads(content)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Cached WLASL metadata to '{cache_path}' ({len(data)} total glosses).")

    return data


def parse_wlasl_entry(entry: Dict[str, Any]) -> Tuple[str, List[WLASLInstance]]:
    """
    Parses a raw WLASL metadata dictionary entry into structured instances.

    Args:
        entry (Dict[str, Any]): Raw dictionary entry from WLASL_v0.3.json.

    Returns:
        Tuple[str, List[WLASLInstance]]: Gloss name and list of parsed WLASLInstance objects.
    """
    gloss_name = entry.get("gloss", "")
    instances: List[WLASLInstance] = []

    for idx, inst in enumerate(entry.get("instances", [])):
        instances.append(
            WLASLInstance(
                instance_id=idx,
                video_id=str(inst.get("video_id", "")),
                url=str(inst.get("url", "")),
                frame_start=int(inst.get("frame_start", 1)),
                frame_end=int(inst.get("frame_end", -1)),
                split=str(inst.get("split", "train")),
                signer_id=int(inst.get("signer_id", 0)),
            )
        )

    return gloss_name, instances


def inspect_wlasl_glosses(
    requested_glosses: List[str],
    metadata: List[Dict[str, Any]],
) -> Dict[str, WLASLGlossReport]:
    """
    Inspects WLASL metadata for a list of requested uppercase SignSyntax glosses.

    Performs case-insensitive matching between SignSyntax glosses and WLASL gloss entries.

    Args:
        requested_glosses (List[str]): List of SignSyntax glosses (e.g., ['ME', 'YOU', 'WANT', 'MILK']).
        metadata (List[Dict[str, Any]]): Parsed WLASL_v0.3.json content.

    Returns:
        Dict[str, WLASLGlossReport]: Mapping from requested gloss to WLASLGlossReport.
    """
    # Index WLASL entries by lowercase gloss key
    wlasl_index: Dict[str, Tuple[str, List[WLASLInstance]]] = {}
    for entry in metadata:
        gloss_name, instances = parse_wlasl_entry(entry)
        wlasl_index[gloss_name.lower().strip()] = (gloss_name, instances)

    reports: Dict[str, WLASLGlossReport] = {}

    for req_gloss in requested_glosses:
        normalized = req_gloss.lower().strip()
        if normalized in wlasl_index:
            matched_name, instances = wlasl_index[normalized]
            reports[req_gloss] = WLASLGlossReport(
                requested_gloss=req_gloss,
                matched_wlasl_gloss=matched_name,
                is_available=True,
                sample_count=len(instances),
                instances=instances,
            )
        else:
            reports[req_gloss] = WLASLGlossReport(
                requested_gloss=req_gloss,
                matched_wlasl_gloss=None,
                is_available=False,
                sample_count=0,
                instances=[],
            )

    return reports


def display_wlasl_report(reports: Dict[str, WLASLGlossReport]) -> str:
    """
    Formats the inspection report into an ASCII summary table.

    Args:
        reports (Dict[str, WLASLGlossReport]): Dictionary of gloss reports.

    Returns:
        str: Formatted report table.
    """
    lines: List[str] = [
        "=" * 75,
        "          SignSyntax: WLASL Dataset Metadata Inspection Report         ",
        "=" * 75,
        f"{'SignSyntax Gloss':<18} | {'Status':<12} | {'WLASL Gloss':<15} | {'Samples':<8}",
        "-" * 75,
    ]

    available_count = 0
    total_samples = 0

    for gloss, rep in reports.items():
        if rep.is_available:
            status = "[AVAILABLE]"
            wlasl_name = f"'{rep.matched_wlasl_gloss}'"
            count_str = str(rep.sample_count)
            available_count += 1
            total_samples += rep.sample_count
        else:
            status = "[NOT FOUND]"
            wlasl_name = "-"
            count_str = "0"

        lines.append(f"{gloss:<18} | {status:<12} | {wlasl_name:<15} | {count_str:<8}")

    lines.append("-" * 75)
    lines.append(f"Summary: {available_count}/{len(reports)} requested glosses found in WLASL ({total_samples} total video samples).")
    lines.append("=" * 75)
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint for running WLASL metadata inspection."""
    st = SymbolTable()

    target_glosses = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_PROTOTYPE_GLOSSES

    print("Loading WLASL metadata...")
    metadata = load_or_fetch_wlasl_metadata()
    reports = inspect_wlasl_glosses(target_glosses, metadata)
    print("\n" + display_wlasl_report(reports))


if __name__ == "__main__":
    main()

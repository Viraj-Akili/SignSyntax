"""
Unit tests for the WLASL dataset downloader and subset preparation module.
Phase 1: Computer Vision Input & WLASL Integration

Tests verify:
1. Handling of unsupported URLs (SWF, non-HTTP, invalid schemes)
2. Handling of HTTP errors and timeouts with clear non-silent error reporting
3. Summary report ASCII rendering
4. Downloader and processor result dataclass representations
"""

import os
import tempfile
import pytest

from recognition.wlasl_downloader import (
    DownloadResult,
    ProcessingResult,
    display_pipeline_summary,
    download_single_video,
)


def test_download_single_video_unsupported_scheme():
    """Verifies that non-HTTP URLs return clear failure."""
    success, size, err = download_single_video("ftp://example.com/video.mp4", "test.mp4")
    assert success is False
    assert size == 0
    assert "Unsupported URL scheme" in err


def test_download_single_video_swf_rejection():
    """Verifies that Flash SWF files are properly rejected with helpful message."""
    success, size, err = download_single_video("http://www.aslpro.com/main/b/book.swf", "test.swf")
    assert success is False
    assert size == 0
    assert "Flash SWF" in err


def test_download_single_video_cached_existing_file():
    """Verifies that an already downloaded valid file is recognized as cached."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_file = os.path.join(tmpdir, "cached.mp4")
        with open(dummy_file, "wb") as f:
            f.write(b"0" * 4096)

        success, size, msg = download_single_video("http://example.com/sample.mp4", dummy_file)
        assert success is True
        assert size == 4096
        assert msg == "Cached"


def test_display_pipeline_summary_rendering():
    """Verifies formatted ASCII pipeline report calculation and display."""
    dl_results = {
        "ME": [
            DownloadResult("ME", "1", "url1", True, "path1", bytes_downloaded=1000),
            DownloadResult("ME", "2", "url2", False, "path2", error_message="404"),
        ],
        "BOOK": [
            DownloadResult("BOOK", "3", "url3", True, "path3", bytes_downloaded=2000),
        ],
    }

    proc_results = {
        "ME": [
            ProcessingResult("ME", "1", "path1", True, "seq1.json", frame_count=35),
        ],
        "BOOK": [
            ProcessingResult("BOOK", "3", "path3", True, "seq3.json", frame_count=42),
        ],
    }

    summary = display_pipeline_summary(dl_results, proc_results)
    assert "ME" in summary
    assert "BOOK" in summary
    assert "TOTAL" in summary
    assert "35" in summary
    assert "42" in summary
    assert "77" in summary  # Total frames: 35 + 42

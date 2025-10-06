"""Utility functions for the bookwiki application."""

import inspect
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional


class PerformanceTimer:
    """Context manager for timing operations and calling back on slow operations.

    Usage:
        def my_callback(op_type, op_detail, filename, lineno, elapsed_ms):
            print(f"Slow {op_type}: {elapsed_ms}ms at {filename}:{lineno}")

        with PerformanceTimer(
            operation_type="QUERY",
            operation_detail="SELECT * FROM users",
            threshold_ms=100.0,
            skip_frames=0,
            callback=my_callback,
        ):
            # perform operation
            pass
    """

    def __init__(
        self,
        operation_type: str,
        operation_detail: str,
        threshold_ms: float,
        skip_frames: int,
        callback: Callable[[str, str, str, int, float], None],
    ):
        """Initialize the performance timer.

        Args:
            operation_type: Type of operation (e.g., "QUERY", "TRANSACTION")
            operation_detail: Details about the operation (e.g., SQL query text)
            threshold_ms: Threshold in milliseconds for slow operation warnings
            skip_frames: Additional stack frames to skip beyond timer internals
            callback: Function called when operation exceeds threshold
        """
        self.operation_type = operation_type
        self.operation_detail = operation_detail
        self.threshold_ms = threshold_ms
        self.skip_frames = skip_frames
        self.callback = callback
        self.start_time: Optional[float] = None
        self.filename: str = "unknown"
        self.lineno: int = 0

    def _get_caller_info(self) -> tuple[str, int]:
        """Get caller information for logging.

        Returns:
            Tuple of (filename, line_number)
        """
        frame = inspect.currentframe()
        # Always skip this method and __enter__ (+2), plus any additional requested
        frames_to_skip = self.skip_frames + 2

        for _ in range(frames_to_skip):
            if frame and frame.f_back:
                frame = frame.f_back
            else:
                return "unknown", 0

        if frame:
            return frame.f_code.co_filename, frame.f_lineno
        return "unknown", 0

    def __enter__(self) -> "PerformanceTimer":
        """Enter the context manager and start timing."""
        self.filename, self.lineno = self._get_caller_info()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager and check if operation was slow."""
        if self.start_time is None:
            return

        elapsed_time = (time.perf_counter() - self.start_time) * 1000  # Convert to ms

        if elapsed_time > self.threshold_ms:
            self.callback(
                self.operation_type,
                self.operation_detail,
                self.filename,
                self.lineno,
                elapsed_time,
            )


def utc_now() -> datetime:
    """Get the current time in UTC with timezone awareness."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Get the current time in UTC as an ISO format string."""
    return utc_now().isoformat()


@dataclass(frozen=True)
class WikiLink:
    """Represents a wiki link extracted from markdown text."""

    display_text: str
    target: str  # Original target from the markdown
    slug: str  # Extracted slug


def extract_slug_from_target(target: str) -> str:
    """Extract the slug from a link target.

    Args:
        target: The link target (could be a path, URL, or just a slug)

    Returns:
        The extracted slug (final component after stripping paths)
    """
    # Strip trailing slashes
    slug = target.rstrip("/")

    # Extract everything after the last remaining slash
    if "/" in slug:
        slug = slug.split("/")[-1]

    return slug


def extract_wiki_links(text: str) -> list[WikiLink]:
    """Extract wiki links from markdown text.

    Args:
        text: Markdown text containing links

    Returns:
        List of WikiLink objects for all markdown links.
        The slug is extracted from the target by taking the final path component.
    """
    if not text:
        return []

    # Pattern matches [display text](target) format
    link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
    matches = re.findall(link_pattern, text)

    wiki_links = []
    for display_text, target in matches:
        slug = extract_slug_from_target(target)
        wiki_links.append(WikiLink(display_text=display_text, target=target, slug=slug))

    return wiki_links

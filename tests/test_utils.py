"""Tests for bookwiki utilities."""

import time
from datetime import datetime, timedelta, timezone

from bookwiki.utils import utc_now, utc_now_iso


def test_utc_now() -> None:
    """Test that utc_now returns current UTC time with timezone awareness."""
    before = datetime.now(timezone.utc)
    result = utc_now()
    after = datetime.now(timezone.utc)

    # Check it's between before and after
    assert before <= result <= after

    # Check it has UTC timezone
    assert result.tzinfo == timezone.utc

    # Check it's a datetime object
    assert isinstance(result, datetime)


def test_utc_now_iso() -> None:
    """Test that utc_now_iso returns ISO format string."""
    before = datetime.now(timezone.utc)
    result = utc_now_iso()
    after = datetime.now(timezone.utc)

    # Check it's a string
    assert isinstance(result, str)

    # Check it's in ISO format
    assert "T" in result  # ISO format includes T separator
    assert result.endswith("+00:00")  # UTC timezone suffix

    # Parse it back and verify it's between before and after
    parsed = datetime.fromisoformat(result)
    assert before <= parsed <= after
    assert parsed.tzinfo == timezone.utc


def test_utc_now_consistency() -> None:
    """Test that utc_now and utc_now_iso are consistent."""
    # Get both values close together
    dt = utc_now()
    iso = utc_now_iso()

    # Parse the ISO string
    parsed = datetime.fromisoformat(iso)

    # They should be very close (within 1 second)
    diff = abs((parsed - dt).total_seconds())
    assert diff < 1.0


def test_utc_now_timezone_aware() -> None:
    """Test that utc_now always returns timezone-aware datetime."""
    dt = utc_now()

    # Should not be naive
    assert dt.tzinfo is not None
    assert dt.utcoffset() is not None

    # Should be UTC specifically
    assert dt.utcoffset() == timedelta(0)
    assert dt.tzinfo == timezone.utc


def test_utc_now_iso_format() -> None:
    """Test the specific format of utc_now_iso."""
    iso_str = utc_now_iso()

    # Check format components
    parts = iso_str.split("T")
    assert len(parts) == 2

    date_part = parts[0]
    time_part = parts[1]

    # Date should be YYYY-MM-DD
    assert len(date_part) == 10
    assert date_part[4] == "-"
    assert date_part[7] == "-"

    # Time should end with timezone
    assert time_part.endswith("+00:00")

    # Should have microseconds
    assert "." in time_part


def test_utc_functions_multiple_calls() -> None:
    """Test that multiple calls return increasing times."""
    times = []
    for _ in range(5):
        times.append(utc_now())
        # Small delay to ensure different times
        time.sleep(0.001)

    # Each time should be >= the previous
    for i in range(1, len(times)):
        assert times[i] >= times[i - 1]


def test_utc_now_iso_roundtrip() -> None:
    """Test that ISO string can be round-tripped."""
    # Get original time
    original = utc_now()

    # Convert to ISO
    iso_str = original.isoformat()

    # Parse back
    parsed = datetime.fromisoformat(iso_str)

    # Should be identical
    assert parsed == original
    assert parsed.tzinfo == original.tzinfo

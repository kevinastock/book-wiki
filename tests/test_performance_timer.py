"""Tests for PerformanceTimer utility."""

import time
from unittest.mock import Mock

import pytest

from bookwiki.utils import PerformanceTimer


class TestPerformanceTimer:
    """Test the PerformanceTimer context manager."""

    def test_fast_operation_no_callback(self) -> None:
        """Test that fast operations don't trigger the callback."""
        callback = Mock()

        with PerformanceTimer(
            operation_type="TEST",
            operation_detail="fast operation",
            threshold_ms=100.0,
            skip_frames=0,
            callback=callback,
        ):
            # Fast operation - no sleep
            pass

        # Callback should not have been called
        callback.assert_not_called()

    def test_slow_operation_triggers_callback(self) -> None:
        """Test that slow operations trigger the callback with correct parameters."""
        callback = Mock()

        with PerformanceTimer(
            operation_type="TEST",
            operation_detail="slow operation",
            threshold_ms=50.0,  # Low threshold to ensure we exceed it
            skip_frames=0,
            callback=callback,
        ):
            time.sleep(0.06)  # 60ms - should exceed threshold

        # Callback should have been called once
        callback.assert_called_once()

        # Check the callback arguments
        args = callback.call_args[0]
        assert len(args) == 5

        operation_type, operation_detail, filename, lineno, elapsed_ms = args
        assert operation_type == "TEST"
        assert operation_detail == "slow operation"
        assert "test_performance_timer.py" in filename
        assert isinstance(lineno, int)
        assert lineno > 0
        assert elapsed_ms >= 50.0  # Should be at least the threshold

    def test_empty_operation_detail(self) -> None:
        """Test that empty operation detail is passed correctly."""
        callback = Mock()

        with PerformanceTimer(
            operation_type="TRANSACTION",
            operation_detail="",
            threshold_ms=10.0,
            skip_frames=0,
            callback=callback,
        ):
            time.sleep(0.02)  # 20ms

        callback.assert_called_once()
        args = callback.call_args[0]
        operation_type, operation_detail, filename, lineno, elapsed_ms = args
        assert operation_type == "TRANSACTION"
        assert operation_detail == ""

    def test_frame_skipping(self) -> None:
        """Test that frame skipping works correctly."""
        callback = Mock()

        def helper_function() -> None:
            with PerformanceTimer(
                operation_type="TEST",
                operation_detail="frame test",
                threshold_ms=10.0,
                skip_frames=1,  # Skip the helper_function frame
                callback=callback,
            ):
                time.sleep(0.02)

        helper_function()

        callback.assert_called_once()
        args = callback.call_args[0]
        _, _, filename, lineno, _ = args

        # Should point to this test method, not the helper function
        assert "test_performance_timer.py" in filename
        # The line number should be where we call helper_function(), not inside it

    def test_nested_frame_skipping(self) -> None:
        """Test frame skipping with multiple nested functions."""
        callback = Mock()

        def level2() -> None:
            with PerformanceTimer(
                operation_type="NESTED",
                operation_detail="deep call",
                threshold_ms=10.0,
                skip_frames=2,  # Skip level2 and level1
                callback=callback,
            ):
                time.sleep(0.02)

        def level1() -> None:
            level2()

        level1()

        callback.assert_called_once()
        args = callback.call_args[0]
        _, _, filename, lineno, _ = args
        assert "test_performance_timer.py" in filename

    def test_different_thresholds(self) -> None:
        """Test that different thresholds work correctly."""
        callback1 = Mock()
        callback2 = Mock()

        # First timer with high threshold - should not trigger
        with PerformanceTimer(
            operation_type="HIGH_THRESHOLD",
            operation_detail="test",
            threshold_ms=1000.0,  # Very high threshold
            skip_frames=0,
            callback=callback1,
        ):
            time.sleep(0.01)  # Only 10ms

        # Second timer with low threshold - should trigger
        with PerformanceTimer(
            operation_type="LOW_THRESHOLD",
            operation_detail="test",
            threshold_ms=5.0,  # Very low threshold
            skip_frames=0,
            callback=callback2,
        ):
            time.sleep(0.01)  # 10ms - should exceed 5ms threshold

        callback1.assert_not_called()
        callback2.assert_called_once()

    def test_exception_during_operation(self) -> None:
        """Test that exceptions don't prevent timing measurement."""
        callback = Mock()

        with (
            pytest.raises(ValueError, match="test exception"),
            PerformanceTimer(
                operation_type="EXCEPTION_TEST",
                operation_detail="operation that fails",
                threshold_ms=10.0,
                skip_frames=0,
                callback=callback,
            ),
        ):
            time.sleep(0.02)  # Ensure we exceed threshold
            raise ValueError("test exception")

        # Callback should still be called despite the exception
        callback.assert_called_once()
        args = callback.call_args[0]
        operation_type, operation_detail, _, _, elapsed_ms = args
        assert operation_type == "EXCEPTION_TEST"
        assert operation_detail == "operation that fails"
        assert elapsed_ms >= 10.0

    def test_multiple_timers(self) -> None:
        """Test that multiple timers work independently."""
        callback1 = Mock()
        callback2 = Mock()

        # First timer - fast operation
        with PerformanceTimer(
            operation_type="FAST",
            operation_detail="quick",
            threshold_ms=100.0,
            skip_frames=0,
            callback=callback1,
        ):
            time.sleep(0.01)

        # Second timer - slow operation
        with PerformanceTimer(
            operation_type="SLOW",
            operation_detail="lengthy",
            threshold_ms=20.0,
            skip_frames=0,
            callback=callback2,
        ):
            time.sleep(0.03)

        callback1.assert_not_called()
        callback2.assert_called_once()

        # Verify the slow timer got the right data
        args = callback2.call_args[0]
        operation_type, operation_detail, _, _, _ = args
        assert operation_type == "SLOW"
        assert operation_detail == "lengthy"

    def test_callback_receives_correct_types(self) -> None:
        """Test that callback receives arguments of the correct types."""
        callback = Mock()

        with PerformanceTimer(
            operation_type="TYPE_TEST",
            operation_detail="checking types",
            threshold_ms=10.0,
            skip_frames=0,
            callback=callback,
        ):
            time.sleep(0.02)

        callback.assert_called_once()
        args = callback.call_args[0]
        operation_type, operation_detail, filename, lineno, elapsed_ms = args

        assert isinstance(operation_type, str)
        assert isinstance(operation_detail, str)
        assert isinstance(filename, str)
        assert isinstance(lineno, int)
        assert isinstance(elapsed_ms, float)

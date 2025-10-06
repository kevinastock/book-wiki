"""Comprehensive tests for bookwiki.db module."""

import logging
import os
import sqlite3
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from bookwiki.db import SafeConnection, TimedCursor, connect_db


class TestSafeConnection:
    """Test the SafeConnection class functionality."""

    def test_safe_connection_initialization(self) -> None:
        """Test SafeConnection initialization."""
        with tempfile.NamedTemporaryFile() as temp_file:
            conn = SafeConnection(temp_file.name)
            assert hasattr(conn, "_lock")
            assert hasattr(conn, "_acquired")
            assert conn._acquired is False
            conn.close()

    def test_nested_transaction_prevention(self) -> None:
        """Test that nested transactions are prevented."""
        with tempfile.NamedTemporaryFile() as temp_file:
            conn = SafeConnection(temp_file.name)
            conn.executescript("CREATE TABLE test (id INTEGER);")

            with conn.transaction_cursor():  # noqa: SIM117
                # This should raise an error because we're already in a transaction
                with (
                    pytest.raises(
                        RuntimeError, match="Nested transactions are not allowed"
                    ),
                    conn.transaction_cursor(),
                ):
                    pass

            conn.close()

    def test_transaction_cursor_basic_functionality(self) -> None:
        """Test basic transaction cursor functionality."""
        with tempfile.NamedTemporaryFile() as temp_file:
            conn = SafeConnection(temp_file.name)
            conn.executescript("CREATE TABLE test (id INTEGER);")

            # Test successful transaction
            with conn.transaction_cursor() as cursor:
                cursor.execute("INSERT INTO test (id) VALUES (1)")
                cursor.execute("SELECT COUNT(*) FROM test")
                count = cursor.fetchone()[0]
                assert count == 1

            # Verify data persisted
            with conn.transaction_cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM test")
                count = cursor.fetchone()[0]
                assert count == 1

            conn.close()

    def test_transaction_rollback_on_exception(self) -> None:
        """Test transaction rollback when exception occurs."""
        with tempfile.NamedTemporaryFile() as temp_file:
            conn = SafeConnection(temp_file.name)
            conn.executescript("CREATE TABLE test (id INTEGER);")

            # Insert initial data
            with conn.transaction_cursor() as cursor:
                cursor.execute("INSERT INTO test (id) VALUES (1)")

            # Attempt transaction that fails
            with pytest.raises(RuntimeError), conn.transaction_cursor() as cursor:
                cursor.execute("INSERT INTO test (id) VALUES (2)")
                raise RuntimeError("Test exception")

            # Verify rollback occurred
            with conn.transaction_cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM test")
                count = cursor.fetchone()[0]
                assert count == 1  # Only the first insert should remain

            conn.close()

    def test_acquired_flag_reset_after_exception(self) -> None:
        """Test that _acquired flag is properly reset after exceptions."""
        with tempfile.NamedTemporaryFile() as temp_file:
            conn = SafeConnection(temp_file.name)
            conn.executescript("CREATE TABLE test (id INTEGER);")

            # Cause an exception in transaction
            with pytest.raises(RuntimeError), conn.transaction_cursor():
                raise RuntimeError("Test exception")

            # Verify we can start a new transaction (flag was reset)
            with conn.transaction_cursor() as cursor:
                cursor.execute("INSERT INTO test (id) VALUES (1)")
                cursor.execute("SELECT COUNT(*) FROM test")
                assert cursor.fetchone()[0] == 1

            conn.close()

    def test_cursor_close_on_exception(self) -> None:
        """Test that cursors are properly closed even when exceptions occur."""
        with tempfile.NamedTemporaryFile() as temp_file:
            conn = SafeConnection(temp_file.name)
            conn.executescript("CREATE TABLE test (id INTEGER);")

            saved_cursor = None

            # Save cursor reference and cause exception
            with pytest.raises(RuntimeError), conn.transaction_cursor() as cursor:
                saved_cursor = cursor
                cursor.execute("INSERT INTO test (id) VALUES (1)")
                raise RuntimeError("Test exception")

            # Cursor should be closed after context exit
            with pytest.raises(sqlite3.ProgrammingError):
                assert saved_cursor is not None
                saved_cursor.execute("SELECT 1")

            conn.close()


class TestConnectDb:
    """Test the connect_db function."""

    def test_connect_db_with_memory_database(self) -> None:
        """Test connecting to in-memory database."""
        conn = connect_db(":memory:")
        assert isinstance(conn, SafeConnection)
        assert conn.row_factory == sqlite3.Row
        conn.close()

    def test_connect_db_with_file_database(self) -> None:
        """Test connecting to file database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            conn = connect_db(temp_path)
            assert isinstance(conn, SafeConnection)
            assert conn.row_factory == sqlite3.Row

            # Verify schema was initialized
            with conn.transaction_cursor() as cursor:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                # Should have tables from schema.sql
                assert len(tables) > 0

            conn.close()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_connect_db_with_pathlib_path(self) -> None:
        """Test connecting with pathlib.Path object."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            conn = connect_db(temp_path)
            assert isinstance(conn, SafeConnection)
            conn.close()
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_connect_db_threadsafety_check(self) -> None:
        """Test that connect_db checks SQLite threadsafety."""
        with (
            patch("bookwiki.db.sqlite3.threadsafety", 1),
            pytest.raises(
                Exception,
                match="This seems bad. Something wants to use multiple threads.",
            ),
        ):
            connect_db(":memory:")

    def test_connect_db_schema_initialization(self) -> None:
        """Test that schema is properly initialized."""
        conn = connect_db(":memory:")

        # Check that expected tables exist
        with conn.transaction_cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

            # Verify key tables from schema exist
            expected_tables = {
                "conversation",
                "block",
                "chapter",
                "prompt",
                "wiki_page",
                "wiki_name",
                "wiki_page_name",
                "configuration",
            }
            assert expected_tables.issubset(tables)

        conn.close()

    def test_connect_db_atexit_registration(self) -> None:
        """Test that atexit cleanup function is registered."""
        with patch("atexit.register") as mock_register:
            conn = connect_db(":memory:")

            # Verify atexit.register was called
            mock_register.assert_called_once()

            # Verify the registered function is callable
            registered_func = mock_register.call_args[0][0]
            assert callable(registered_func)

            conn.close()

    def test_atexit_cleanup_function_behavior(self) -> None:
        """Test the atexit cleanup function handles exceptions gracefully."""
        cleanup_func = None

        with patch("atexit.register") as mock_register:
            conn = connect_db(":memory:")
            cleanup_func = mock_register.call_args[0][0]
            conn.close()

        # Calling cleanup function should not raise exception
        # (it should suppress LookupError and other exceptions)
        try:
            cleanup_func()  # Should not raise even if connection is already closed
        except Exception as e:
            pytest.fail(f"Cleanup function should not raise exceptions: {e}")

    def test_connect_db_row_factory_setting(self) -> None:
        """Test that row factory is properly set to sqlite3.Row."""
        conn = connect_db(":memory:")

        assert conn.row_factory == sqlite3.Row

        # Test that it actually works
        with conn.transaction_cursor() as cursor:
            cursor.execute("SELECT 1 as test_value")
            row = cursor.fetchone()
            assert row["test_value"] == 1
            assert row[0] == 1

        conn.close()

    def test_connect_db_autocommit_setting(self) -> None:
        """Test that autocommit is set to False."""
        conn = connect_db(":memory:")

        # SQLite autocommit behavior: when autocommit=False,
        # we need explicit transactions
        with conn.transaction_cursor() as cursor:
            cursor.execute("CREATE TABLE test (id INTEGER)")
            cursor.execute("INSERT INTO test VALUES (1)")

        # Should be committed after transaction_cursor context
        with conn.transaction_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM test")
            assert cursor.fetchone()[0] == 1

        conn.close()


class TestThreadSafety:
    """Test thread safety of SafeConnection."""

    def test_concurrent_transactions(self) -> None:
        """Test that concurrent transactions work correctly."""
        conn = connect_db(":memory:")

        results = []
        errors = []

        def worker(worker_id: int) -> None:
            try:
                for i in range(5):
                    with conn.transaction_cursor() as cursor:
                        cursor.execute(
                            "INSERT INTO conversation "
                            "(total_input_tokens, total_output_tokens) VALUES (?, ?)",
                            (worker_id * 10 + i, worker_id * 5 + i),
                        )
                        cursor.execute("SELECT last_insert_rowid()")
                        row_id = cursor.fetchone()[0]
                        results.append((worker_id, i, row_id))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Start multiple threads
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 15  # 3 workers * 5 operations each

        # Verify all records exist
        with conn.transaction_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM conversation")
            assert cursor.fetchone()[0] == 15

        conn.close()

    def test_thread_isolation(self) -> None:
        """Test that threads don't interfere with each other's transactions."""
        conn = connect_db(":memory:")

        results = {}
        errors = []

        def worker(worker_id: int) -> None:
            try:
                with conn.transaction_cursor() as cursor:
                    # Insert data for this thread
                    cursor.execute(
                        "INSERT INTO conversation "
                        "(total_input_tokens, total_output_tokens) VALUES (?, ?)",
                        (worker_id, worker_id),
                    )
                    # Give other threads a chance to run
                    threading.Event().wait(0.001)

                    # Check what this thread sees within its transaction
                    cursor.execute("SELECT COUNT(*) FROM conversation")
                    results[worker_id] = cursor.fetchone()[0]
            except Exception as e:
                errors.append((worker_id, str(e)))

        threads = [
            threading.Thread(target=worker, args=(1,)),
            threading.Thread(target=worker, args=(2,)),
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=5.0)  # Add timeout to prevent hanging

        # Check no errors occurred
        assert len(errors) == 0, f"Thread errors: {errors}"

        # Each thread should see at least its own insert
        assert all(count >= 1 for count in results.values())

        # Final count should be 2
        with conn.transaction_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM conversation")
            assert cursor.fetchone()[0] == 2

        conn.close()


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_multiple_cursor_contexts(self) -> None:
        """Test using multiple cursor contexts sequentially."""
        conn = connect_db(":memory:")

        # First context
        with conn.transaction_cursor() as cursor1:
            cursor1.execute(
                "INSERT INTO conversation "
                "(total_input_tokens, total_output_tokens) VALUES (1, 1)"
            )

        # Second context (should work fine)
        with conn.transaction_cursor() as cursor2:
            cursor2.execute(
                "INSERT INTO conversation "
                "(total_input_tokens, total_output_tokens) VALUES (2, 2)"
            )
            cursor2.execute("SELECT COUNT(*) FROM conversation")
            assert cursor2.fetchone()[0] == 2

        conn.close()

    def test_database_connection_error_handling(self) -> None:
        """Test handling of database connection errors."""
        # Try to connect to invalid path
        with pytest.raises(sqlite3.OperationalError):
            connect_db("/invalid/path/that/does/not/exist.db")

    def test_schema_loading_with_missing_file(self) -> None:
        """Test behavior when schema file is missing."""
        # This tests the importlib.resources usage
        with patch("bookwiki.db.files") as mock_files:
            mock_path = Mock()
            mock_path.read_text.side_effect = FileNotFoundError("Schema file not found")
            mock_files.return_value.__truediv__.return_value = mock_path

            with pytest.raises(FileNotFoundError):
                connect_db(":memory:")

    def test_connection_with_bytes_path(self) -> None:
        """Test connecting with bytes path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            temp_path = temp_file.name.encode("utf-8")

        try:
            conn = connect_db(temp_path)
            assert isinstance(conn, SafeConnection)
            conn.close()
        finally:
            if os.path.exists(temp_path.decode("utf-8")):
                os.unlink(temp_path.decode("utf-8"))


class TestIntegrationWithExistingCode:
    """Test integration with existing codebase patterns."""

    def test_safeconnection_with_existing_models(self, temp_db: SafeConnection) -> None:
        """Test SafeConnection works with existing model patterns."""
        # This uses the temp_db fixture to ensure compatibility
        with temp_db.transaction_cursor() as cursor:
            # Test basic conversation creation (from existing tests)
            from bookwiki.models.block import Block
            from bookwiki.models.conversation import Conversation

            conversation = Conversation.create(cursor)
            # Update token counts to match expected test values
            cursor.execute(
                """UPDATE conversation SET total_input_tokens = ?,
                   total_output_tokens = ? WHERE id = ?""",
                (100, 50, conversation.id),
            )

            # Test block creation using model method
            Block.create_text(
                cursor=cursor,
                conversation_id=conversation.id,
                generation=conversation.current_generation,
                role="user",
                text="Test message",
            )

            # Verify relationship works
            cursor.execute(
                "SELECT COUNT(*) FROM block WHERE conversation = ?", (conversation.id,)
            )
            assert cursor.fetchone()[0] == 1


class TestTimedCursor:
    """Test the TimedCursor functionality for slow query detection."""

    def test_timed_cursor_is_used_by_default(self) -> None:
        """Test that SafeConnection returns TimedCursor instances."""
        conn = connect_db(":memory:")

        # Test that cursor() returns a TimedCursor
        cursor = conn.cursor()
        assert isinstance(cursor, TimedCursor)
        cursor.close()

        # Test that transaction_cursor also uses TimedCursor
        with conn.transaction_cursor() as cursor:  # type: ignore[assignment]
            # mypy sees this as sqlite3.Cursor but at runtime it's TimedCursor
            assert isinstance(cursor, sqlite3.Cursor)
            assert type(cursor).__name__ == "TimedCursor"

        conn.close()

    def test_timed_cursor_threshold_configurable(self) -> None:
        """Test that the slow query threshold is configurable."""
        conn = connect_db(":memory:")

        # Check default threshold
        cursor = conn.cursor()
        assert cursor.slow_query_threshold_ms == 100.0
        cursor.close()

        # Modify threshold
        original_threshold = TimedCursor.slow_query_threshold_ms
        try:
            TimedCursor.slow_query_threshold_ms = 50.0
            cursor = conn.cursor()
            assert cursor.slow_query_threshold_ms == 50.0
            cursor.close()
        finally:
            # Restore original threshold
            TimedCursor.slow_query_threshold_ms = original_threshold

        conn.close()

    def test_slow_query_detection(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that slow queries are detected and reported."""
        conn = connect_db(":memory:")

        # Save original threshold and set a very low one for testing
        original_threshold = TimedCursor.slow_query_threshold_ms
        TimedCursor.slow_query_threshold_ms = (
            0.01  # 0.01ms - almost everything will be "slow"
        )

        try:
            with caplog.at_level(logging.WARNING), conn.transaction_cursor() as cursor:
                # This query should trigger the slow query warning
                cursor.execute("CREATE TABLE test_slow (id INTEGER, data TEXT)")
                cursor.execute("INSERT INTO test_slow VALUES (1, 'test')")

            # Check that slow query warning was logged
            assert any("SLOW QUERY" in record.message for record in caplog.records)
            assert any("ms)" in record.message for record in caplog.records)
            assert any("test_db.py" in record.message for record in caplog.records)

        finally:
            # Restore original threshold
            TimedCursor.slow_query_threshold_ms = original_threshold

        conn.close()

    def test_fast_query_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that fast queries don't trigger warnings."""
        conn = connect_db(":memory:")

        # Keep the default threshold (100ms)
        with caplog.at_level(logging.WARNING), conn.transaction_cursor() as cursor:
            # Simple queries should be fast enough
            cursor.execute("SELECT 1")
            cursor.execute("SELECT 2 + 2")

        # Check that no slow query warning was logged
        assert not any("SLOW QUERY" in record.message for record in caplog.records)

        conn.close()

    def test_execute_methods_return_cursor(self) -> None:
        """Test that execute methods return the cursor for chaining."""
        conn = connect_db(":memory:")

        with conn.transaction_cursor() as cursor:
            # Test execute returns cursor
            result = cursor.execute("SELECT 1 as value")
            assert result is cursor
            assert cursor.fetchone()["value"] == 1

            # Test executemany returns cursor
            cursor.execute("CREATE TABLE test_many (id INTEGER)")
            result = cursor.executemany(
                "INSERT INTO test_many VALUES (?)", [(1,), (2,), (3,)]
            )
            assert result is cursor

            # Test executescript returns cursor
            result = cursor.executescript("""
                CREATE TABLE test_script (id INTEGER);
                INSERT INTO test_script VALUES (1);
            """)
            assert result is cursor

        conn.close()

    def test_slow_query_location_tracking(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that slow query reports show correct file and line number."""
        conn = connect_db(":memory:")

        # Set very low threshold
        original_threshold = TimedCursor.slow_query_threshold_ms
        TimedCursor.slow_query_threshold_ms = 0.01

        try:
            with caplog.at_level(logging.WARNING), conn.transaction_cursor() as cursor:
                # Execute from a specific line to check location tracking
                cursor.execute(
                    "CREATE TABLE location_test (id INTEGER)"
                )  # This line number will be reported

            # Check that slow query warning was logged with location
            slow_query_records = [
                r for r in caplog.records if "SLOW QUERY" in r.message
            ]
            assert len(slow_query_records) > 0
            # Should contain file:line format
            assert any("test_db.py:" in r.message for r in slow_query_records)
            # The line number should be present
            import re

            for record in slow_query_records:
                match = re.search(r"test_db\.py:(\d+)", record.message)
                if match:
                    line_number = int(match.group(1))
                    assert line_number > 0  # Should be a valid line number
                    break
            else:
                raise AssertionError("No line number found in slow query logs")

        finally:
            TimedCursor.slow_query_threshold_ms = original_threshold

        conn.close()

    def test_slow_transaction_detection(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that slow transactions are detected with correct location."""
        import time

        conn = connect_db(":memory:")

        with caplog.at_level(logging.WARNING):
            # This transaction should be fast (no warning)
            with conn.transaction_cursor() as cursor:
                cursor.execute("SELECT 1")

            # Clear any existing logs
            caplog.clear()

            # This transaction should be slow (warning expected)
            with conn.transaction_cursor() as cursor:
                cursor.execute("CREATE TABLE test_table (id INTEGER)")
                # Sleep to make the transaction slow
                time.sleep(0.11)  # 110ms, over the 100ms threshold

        # Check that slow transaction warning was logged
        slow_transaction_records = [
            r for r in caplog.records if "SLOW TRANSACTION" in r.message
        ]
        assert len(slow_transaction_records) > 0

        # Check that the location points to the test file, not db.py
        messages = [r.message for r in slow_transaction_records]
        assert any("test_db.py" in r.message for r in slow_transaction_records), (
            f"Expected test_db.py in log messages, got: {messages}"
        )

        conn.close()

    def test_sql_truncation_in_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that long SQL queries are truncated in warnings."""
        conn = connect_db(":memory:")

        # Set very low threshold
        original_threshold = TimedCursor.slow_query_threshold_ms
        TimedCursor.slow_query_threshold_ms = 0.01

        try:
            with caplog.at_level(logging.WARNING), conn.transaction_cursor() as cursor:
                # Create a very long SQL query
                long_sql = "SELECT " + ", ".join(
                    [f"'{i}' as col_{i}" for i in range(50)]
                )
                cursor.execute(long_sql)

            # Check that slow query warning was logged
            slow_query_records = [
                r for r in caplog.records if "SLOW QUERY" in r.message
            ]
            assert len(slow_query_records) > 0
            # Should be truncated with ellipsis
            assert any("..." in r.message for r in slow_query_records)
            # Should not contain the full query
            assert not any("col_49" in r.message for r in slow_query_records)

        finally:
            TimedCursor.slow_query_threshold_ms = original_threshold

        conn.close()

    def test_multiline_sql_formatting(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that multiline SQL is formatted as single line in warnings."""
        conn = connect_db(":memory:")

        # Set very low threshold
        original_threshold = TimedCursor.slow_query_threshold_ms
        TimedCursor.slow_query_threshold_ms = 0.01

        try:
            with caplog.at_level(logging.WARNING), conn.transaction_cursor() as cursor:
                # Execute multiline SQL
                cursor.execute("""
                    CREATE TABLE
                        multiline_test
                    (id INTEGER,
                     data TEXT)
                """)

            # Check that slow query warning was logged
            slow_query_records = [
                r for r in caplog.records if "SLOW QUERY" in r.message
            ]
            assert len(slow_query_records) > 0
            # Check that newlines were replaced with spaces
            for record in slow_query_records:
                # The SQL should be on a single line (no newlines)
                assert "\n" not in record.message
                # Should have spaces where newlines were
                assert "CREATE TABLE" in record.message
                assert "multiline_test" in record.message

        finally:
            TimedCursor.slow_query_threshold_ms = original_threshold

        conn.close()

    def test_executemany_timing(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that executemany operations are also timed."""
        conn = connect_db(":memory:")

        # Set very low threshold
        original_threshold = TimedCursor.slow_query_threshold_ms
        TimedCursor.slow_query_threshold_ms = 0.01

        try:
            with caplog.at_level(logging.WARNING), conn.transaction_cursor() as cursor:
                cursor.execute("CREATE TABLE batch_test (id INTEGER, value TEXT)")

                # Clear any previous logs
                caplog.clear()

                # Execute many inserts
                data = [(i, f"value_{i}") for i in range(100)]
                cursor.executemany("INSERT INTO batch_test VALUES (?, ?)", data)

            # Check that slow query warning was logged
            assert any("SLOW QUERY" in r.message for r in caplog.records)
            assert any("INSERT INTO batch_test" in r.message for r in caplog.records)

        finally:
            TimedCursor.slow_query_threshold_ms = original_threshold

        conn.close()

    def test_executescript_timing(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that executescript operations are also timed."""
        conn = connect_db(":memory:")

        # Set very low threshold
        original_threshold = TimedCursor.slow_query_threshold_ms
        TimedCursor.slow_query_threshold_ms = 0.01

        try:
            with caplog.at_level(logging.WARNING), conn.transaction_cursor() as cursor:
                # Clear any previous logs
                caplog.clear()

                cursor.executescript("""
                    CREATE TABLE script_test1 (id INTEGER);
                    CREATE TABLE script_test2 (id INTEGER);
                    INSERT INTO script_test1 VALUES (1), (2), (3);
                    INSERT INTO script_test2 VALUES (4), (5), (6);
                """)

            # Check that slow query warning was logged
            assert any("SLOW QUERY" in r.message for r in caplog.records)
            assert any("CREATE TABLE script_test1" in r.message for r in caplog.records)

        finally:
            TimedCursor.slow_query_threshold_ms = original_threshold

        conn.close()

"""Database operations for the bookwiki application."""

import atexit
import contextlib
import logging
import os
import re
import sqlite3
import threading
from importlib.resources import files
from typing import Any, Callable, Iterator, TypeVar, overload

from bookwiki.utils import PerformanceTimer

_CursorT = TypeVar("_CursorT", bound=sqlite3.Cursor)

logger = logging.getLogger(__name__)


def _log_slow_operation(
    operation_type: str,
    operation_detail: str,
    filename: str,
    lineno: int,
    elapsed_ms: float,
) -> None:
    """Log slow database operations."""
    location = f"{filename}:{lineno}"
    if operation_detail:
        logger.warning(
            "SLOW %s (%.1fms) at %s - %s",
            operation_type,
            elapsed_ms,
            location,
            operation_detail,
        )
    else:
        logger.warning(
            "SLOW %s (%.1fms) at %s",
            operation_type,
            elapsed_ms,
            location,
        )


class TimedCursor(sqlite3.Cursor):
    """A cursor that tracks execution time and reports slow queries."""

    # Class variable for slow query threshold in milliseconds
    slow_query_threshold_ms: float = 100.0

    def __init__(self, connection: sqlite3.Connection) -> None:
        super().__init__(connection)
        self._connection = connection

    def _get_timer(self, sql: str) -> PerformanceTimer:
        """Get a PerformanceTimer for timing SQL execution."""
        # Prepare SQL preview for logging
        sql_preview = re.sub(r"\s+", " ", sql.strip())
        sql_preview = (
            sql_preview[:100] + "..." if len(sql_preview) > 100 else sql_preview
        )

        # Skip 1 frame: the execute* method (to get to caller)
        return PerformanceTimer(
            operation_type="QUERY",
            operation_detail=sql_preview,
            threshold_ms=self.slow_query_threshold_ms,
            skip_frames=1,
            callback=_log_slow_operation,
        )

    def execute(self, sql: str, parameters: Any = ()) -> "TimedCursor":
        with self._get_timer(sql):
            result = super().execute(sql, parameters)
            return result  # type: ignore[return-value]

    def executemany(self, sql: str, parameters: Any) -> "TimedCursor":
        with self._get_timer(sql):
            result = super().executemany(sql, parameters)
            return result  # type: ignore[return-value]

    def executescript(self, sql: str) -> "TimedCursor":
        with self._get_timer(sql):
            result = super().executescript(sql)
            return result  # type: ignore[return-value]


class SafeConnection(sqlite3.Connection):
    _lock: threading.RLock
    _acquired: bool

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._lock = threading.RLock()
        self._acquired = False

    @overload
    def cursor(self, factory: None = None) -> TimedCursor: ...

    @overload
    def cursor(self, factory: Callable[[sqlite3.Connection], _CursorT]) -> _CursorT: ...

    def cursor(self, factory: Callable[[sqlite3.Connection], Any] | None = None) -> Any:
        """Create a cursor using TimedCursor as the factory."""
        if factory is None:
            return super().cursor(TimedCursor)
        return super().cursor(factory)

    @contextlib.contextmanager
    def transaction_cursor(
        self, threshold_ms: float = 100.0
    ) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            # Check for nested transactions
            # (only the same thread can re-enter due to RLock)
            if self._acquired:
                raise RuntimeError("Nested transactions are not allowed")

            # Mark as acquired
            self._acquired = True

            # Use PerformanceTimer for transaction timing
            # Skip 2 frames: this method and contextlib wrapper
            with (
                PerformanceTimer(
                    operation_type="TRANSACTION",
                    operation_detail="",  # No detail needed for transactions
                    threshold_ms=threshold_ms,
                    skip_frames=2,
                    callback=_log_slow_operation,
                ),
                self,
            ):
                cur = self.cursor()
                try:
                    yield cur
                finally:
                    with contextlib.suppress(Exception):
                        cur.close()

                    # Release the acquired flag
                    self._acquired = False


def connect_db(path: str | bytes | os.PathLike) -> SafeConnection:
    if sqlite3.threadsafety != 3:
        raise Exception("This seems bad. Something wants to use multiple threads.")

    con = sqlite3.connect(
        path,
        check_same_thread=False,
        autocommit=False,
        factory=SafeConnection,
    )

    # Set row factory globally for all cursors
    con.row_factory = sqlite3.Row

    # Initialize database schema from schema.sql
    # Use executescript which auto-commits the transaction
    schema_sql = (files("bookwiki.data") / "schema.sql").read_text()
    con.executescript(schema_sql)
    con.commit()  # Ensure schema is committed

    def _close_db() -> None:
        with contextlib.suppress(LookupError):
            con.close()

    atexit.register(_close_db)

    return con

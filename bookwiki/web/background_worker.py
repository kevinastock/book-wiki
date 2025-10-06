"""Background worker thread for web application."""

import logging
import threading
import time
from enum import Enum

from bookwiki.processor import Processor

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Status states for the background worker."""

    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETE = "complete"  # When all processing is finished
    DEAD = "dead"


class BackgroundWorker:
    """Background worker thread that can be paused and resumed."""

    def __init__(self, processor: Processor) -> None:
        """Initialize the background worker.

        Args:
            processor: Processor instance to handle conversation logic
        """
        self.processor = processor
        self._running_event = threading.Event()
        self._stop_event = threading.Event()
        # Start paused (running_event not set)
        self._thread: threading.Thread | None = None
        self._is_complete = False
        logger.info("Background worker initialized (paused)")

    def pause(self) -> None:
        """Pause the background worker processing."""
        self._running_event.clear()
        logger.info("Background worker paused")

    def resume(self) -> None:
        """Resume the background worker processing."""
        # Start the thread if it hasn't been started yet
        if self._thread is None:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            logger.info("Background worker thread started")

        self._running_event.set()
        logger.info("Background worker resumed")

    def get_status(self) -> WorkerStatus:
        """Get the current status of the background worker."""
        if self._thread is None:
            return WorkerStatus.INITIALIZED
        elif not self._thread.is_alive():
            return WorkerStatus.DEAD
        elif self._is_complete:
            return WorkerStatus.COMPLETE
        elif not self._running_event.is_set():
            return WorkerStatus.PAUSED
        else:
            return WorkerStatus.RUNNING

    def kill(self) -> None:
        """Kill the background worker thread.

        This is pretty jank and really just exists so I can check the frontend."""
        # TODO: Remove this and the stop event.
        logger.info("Killing background worker thread")
        # Set stop event to signal thread to exit
        self._stop_event.set()
        # Set running so we get to the stop check
        self._running_event.set()
        logger.info("Background worker thread killed")

    def _run_loop(self) -> None:
        """Main processing loop for the background worker."""
        logger.info("Background worker thread started")

        while True:
            # Wait until running event is set (not paused)
            self._running_event.wait()

            # Check if we should exit.
            if self._stop_event.is_set():
                logger.info("Shutting down background thread")
                return

            # Process waiting conversations
            self.processor.process_waiting_conversations()

            # Try to advance chapter if needed
            if not self.processor.advance_chapter_if_needed():
                logger.info("All processing complete - marking worker as complete")
                self._is_complete = True
                self.pause()
                continue

            # Process sendable conversations
            self.processor.process_sendable_conversations()

            # Sleep for 10 seconds
            time.sleep(10)

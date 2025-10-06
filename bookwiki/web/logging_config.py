"""Logging configuration for the bookwiki web application."""

import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional


def configure_web_logging(
    log_dir: str = "logs",
    log_prefix: str = "bookwiki_web",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 100,
    console_level: Optional[int] = None,
    file_level: int = logging.DEBUG,
) -> Path:
    """Configure logging for the web application with rotation.

    Args:
        log_dir: Directory to store log files
        log_prefix: Prefix for log file names
        max_bytes: Maximum size of each log file before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 100)
        console_level: Console logging level (None to disable console logging)
        file_level: File logging level (default: DEBUG)

    Returns:
        Path to the main log file
    """
    # Create logs directory if it doesn't exist
    logs_path = Path(log_dir)
    logs_path.mkdir(exist_ok=True)

    # Generate log file path
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = logs_path / f"{log_prefix}_{timestamp}.log"

    # Clear any existing handlers on the root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set root logger to capture all levels
    root_logger.setLevel(logging.DEBUG)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - "
        "[%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    simple_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)

    # Console handler (optional)
    if console_level is not None:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)

    # Configure specific loggers
    # Reduce verbosity for third-party libraries
    for logger_name in ["werkzeug", "httpx", "httpcore"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)

    # Log the start of the session
    logging.info(f"Web logging session started. Log file: {log_file}")
    mb_size = max_bytes / (1024 * 1024)
    logging.info(
        f"Log rotation: {mb_size:.1f}MB per file, keeping {backup_count} backups"
    )

    return log_file


def get_app_logger() -> logging.Logger:
    """Get the main application logger.

    Returns:
        Logger configured for application-level events
    """
    return logging.getLogger("bookwiki.web.app")

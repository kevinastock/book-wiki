"""Flask web application for bookwiki."""

import argparse
import logging
import os
import sys
from typing import NoReturn, Tuple

from flask import Flask, render_template, request
from jinja2 import StrictUndefined

from bookwiki.db import SafeConnection, connect_db
from bookwiki.impls.openai import OpenAILLMService
from bookwiki.models.configuration import Configuration
from bookwiki.processor import Processor
from bookwiki.tools import get_all_tools
from bookwiki.web.background_worker import BackgroundWorker, WorkerStatus
from bookwiki.web.filters import register_filters
from bookwiki.web.logging_config import configure_web_logging, get_app_logger


def create_app(db_path: str) -> Flask:
    """Create and configure the Flask application.

    Args:
        db_path: Path to the SQLite database

    Returns:
        Configured Flask application
    """
    app = Flask(__name__)

    # Preserve my sanity for a little bit longer.
    app.jinja_env.undefined = StrictUndefined

    # Set secret key for sessions (required for flash messages)
    # In production, this should come from environment variable
    app.config["SECRET_KEY"] = os.environ.get(
        "FLASK_SECRET_KEY", "dev-secret-key-change-in-production"
    )

    # Get logger for app events
    logger = get_app_logger()

    # Setup custom jinja filters
    register_filters(app)

    # Store database connection in app config
    logger.info(f"Connecting to database at {db_path}")
    database = connect_db(db_path)
    app.config["DATABASE"] = database

    # Create LLM service with configuration from database
    db_connection = app.config["DATABASE"]
    with db_connection.transaction_cursor() as cursor:
        llm_service = OpenAILLMService(
            model=Configuration.get_openai_model(cursor),
            service_tier=Configuration.get_openai_service_tier(cursor),
            verbosity=Configuration.get_openai_verbosity(cursor),
            reasoning_effort=Configuration.get_openai_reasoning_effort(cursor),
            timeout_minutes=Configuration.get_openai_timeout_minutes(cursor),
            compression_threshold=Configuration.get_openai_compression_threshold(
                cursor
            ),
            system_message=Configuration.get_system_prompt(cursor),
            tools=get_all_tools(),
        )
    app.config["LLM_SERVICE"] = llm_service

    # Create processor
    processor = Processor(database, llm_service)
    app.config["PROCESSOR"] = processor

    # Initialize background worker
    background_worker = BackgroundWorker(processor)
    app.config["BACKGROUND_WORKER"] = background_worker

    @app.before_request
    def before_request() -> None:
        """Log request info."""
        # Log request details
        logger.debug(
            f"Request: {request.method} {request.path} from {request.remote_addr}"
        )

    @app.after_request
    def after_request(response):  # type: ignore[no-untyped-def]
        """Log response status."""
        logger.debug(
            f"Response: {response.status_code} for {request.method} {request.path}"
        )
        return response

    # Add context processor to make background worker status available to all templates
    @app.context_processor
    def inject_background_worker_status():  # type: ignore[no-untyped-def]
        """Inject background worker status into all template contexts."""
        worker = app.config.get("BACKGROUND_WORKER")
        if worker:
            status = worker.get_status()
            return {
                "bg_worker_status": status,
                "bg_worker_status_value": status.value,
            }
        return {
            "bg_worker_status": WorkerStatus.DEAD,
            "bg_worker_status_value": WorkerStatus.DEAD.value,
        }

    # Register routes
    from bookwiki.web.routes.chapters import chapters_bp
    from bookwiki.web.routes.config import config_bp
    from bookwiki.web.routes.conversations import conversations_bp
    from bookwiki.web.routes.feedback import feedback_bp
    from bookwiki.web.routes.index import index_bp
    from bookwiki.web.routes.prompts import prompts_bp
    from bookwiki.web.routes.tools import tools_bp
    from bookwiki.web.routes.wiki import wiki_bp

    app.register_blueprint(index_bp)
    app.register_blueprint(chapters_bp)
    app.register_blueprint(config_bp)
    app.register_blueprint(conversations_bp)
    app.register_blueprint(feedback_bp)
    app.register_blueprint(prompts_bp)
    app.register_blueprint(tools_bp)
    app.register_blueprint(wiki_bp)

    # Error handlers
    @app.errorhandler(404)
    def page_not_found(error: Exception) -> Tuple[str, int]:
        """Handle 404 errors."""
        logger.warning(f"404 error: {request.path}")
        return render_template("404.html", error=error), 404

    @app.errorhandler(500)
    def internal_error(error: Exception) -> Tuple[str, int]:
        """Handle 500 errors."""
        logger.error(f"Internal server error: {error}", exc_info=True)
        return render_template("500.html", error=error), 500

    @app.errorhandler(Exception)
    def handle_exception(error: Exception) -> Tuple[str, int]:
        """Handle unhandled exceptions."""
        logger.error(f"Unhandled exception: {error}", exc_info=True)
        return render_template("500.html", error=error), 500

    # Add cleanup handler for graceful shutdown
    @app.teardown_appcontext
    def cleanup(error):  # type: ignore[no-untyped-def]
        """Clean up resources when app context tears down."""
        if error:
            logger.error(f"App context error: {error}")

    logger.info("Flask app initialized successfully")
    return app


def get_db() -> SafeConnection:
    """Get database connection from Flask app context."""
    from flask import current_app

    return current_app.config["DATABASE"]  # type: ignore[no-any-return]


def get_background_worker() -> BackgroundWorker:
    """Get background worker from Flask app context."""
    from flask import current_app

    return current_app.config["BACKGROUND_WORKER"]  # type: ignore[no-any-return]


def get_llm_service() -> OpenAILLMService:
    """Get LLM service from Flask app context."""
    from flask import current_app

    return current_app.config["LLM_SERVICE"]  # type: ignore[no-any-return]


def get_processor() -> Processor:
    """Get processor from Flask app context."""
    from flask import current_app

    return current_app.config["PROCESSOR"]  # type: ignore[no-any-return]


def main() -> NoReturn:
    """Entry point for the web server."""
    parser = argparse.ArgumentParser(description="Start the bookwiki web server.")
    parser.add_argument(
        "--db",
        default="bookwiki.db",
        help="Path to the SQLite database (default: bookwiki.db)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable development mode (debug + auto-reload on bookwiki/ changes)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (can be repeated: -v for INFO, -vv for DEBUG)",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Quiet mode (only show errors)"
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for log files (default: logs)",
    )

    args = parser.parse_args()

    # Configure logging based on verbosity
    if args.quiet:
        console_level = logging.ERROR
    elif args.verbose == 0:
        console_level = logging.WARNING
    elif args.verbose == 1:
        console_level = logging.INFO
    else:
        console_level = logging.DEBUG

    # Configure web logging with rotation
    log_file = configure_web_logging(
        log_dir=args.log_dir,
        console_level=console_level,
        file_level=logging.DEBUG,  # Always capture everything in files
    )

    logger = get_app_logger()
    logger.info("=" * 60)
    logger.info("Starting bookwiki web server")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Console logging level: {logging.getLevelName(console_level)}")

    try:
        app = create_app(args.db)
        logger.info(f"Starting web server on {args.host}:{args.port}")
        logger.info(f"Using database: {args.db}")
        if not os.path.exists(args.db):
            logger.info("Created new database. Upload chapters via /config")
        # Set debug mode if --debug or --dev flag is used
        debug_mode = args.debug or args.dev

        # Set up extra files to watch if --dev flag is used
        extra_files = ["bookwiki/"] if args.dev else None

        app.run(
            host=args.host, port=args.port, debug=debug_mode, extra_files=extra_files
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error starting web server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

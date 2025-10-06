"""Configuration management routes."""

import bz2
import json
import logging
import os
import tempfile
from sqlite3 import Cursor
from typing import Callable

from flask import Blueprint, flash, redirect, render_template, request, url_for
from werkzeug.wrappers import Response

from bookwiki.config_enums import (
    OpenAIModel,
    OpenAIReasoningEffort,
    OpenAIServiceTier,
    OpenAIVerbosity,
)
from bookwiki.impls.openai import OpenAILLMService
from bookwiki.models import Chapter
from bookwiki.models.configuration import Configuration
from bookwiki.web.app import get_background_worker, get_db, get_llm_service
from bookwiki.web.background_worker import WorkerStatus

logger = logging.getLogger(__name__)

config_bp = Blueprint("config", __name__, url_prefix="/config")

ALLOWED_EXTENSIONS = {"bz2", "tar.bz2"}


def allowed_file(filename: str) -> bool:
    """Check if a filename has an allowed extension."""
    return any(filename.endswith(f".{ext}") for ext in ALLOWED_EXTENSIONS)


@config_bp.route("/")
def index() -> str:
    """Display configuration page."""
    with get_db().transaction_cursor() as cursor:
        chapter_count = Chapter.get_chapter_count(cursor)

        # Get current OpenAI configuration
        openai_config = {
            "model": Configuration.get_openai_model(cursor),
            "verbosity": Configuration.get_openai_verbosity(cursor),
            "reasoning_effort": Configuration.get_openai_reasoning_effort(cursor),
            "service_tier": Configuration.get_openai_service_tier(cursor),
            "timeout_minutes": Configuration.get_openai_timeout_minutes(cursor),
            "compression_threshold": Configuration.get_openai_compression_threshold(
                cursor
            ),
        }

        # Get prompts
        prompts = {
            "system_prompt": Configuration.get_system_prompt(cursor),
            "chapter_prompt": Configuration.get_chapter_prompt(cursor),
            "compress_prompt": Configuration.get_compress_prompt(cursor),
        }

    # Get available enum options
    enum_options = {
        "models": list(OpenAIModel),
        "verbosity_levels": list(OpenAIVerbosity),
        "reasoning_efforts": list(OpenAIReasoningEffort),
        "service_tiers": list(OpenAIServiceTier),
    }

    return render_template(
        "config/index.html",
        chapter_count=chapter_count,
        openai_config=openai_config,
        enum_options=enum_options,
        prompts=prompts,
    )


@config_bp.route("/upload_chapters", methods=["POST"])
def upload_chapters() -> Response:
    """Handle chapter file upload and database initialization."""

    # Check if file was uploaded
    if "file" not in request.files:
        flash("No file uploaded", "error")
        return redirect(url_for("config.index"))

    file = request.files["file"]

    # Check if file was selected
    if file.filename == "":
        flash("No file selected", "error")
        return redirect(url_for("config.index"))

    # Check file extension
    if not file or not file.filename or not allowed_file(file.filename):
        flash("Invalid file type. Please upload a .bz2 or .tar.bz2 file", "error")
        return redirect(url_for("config.index"))

    # Save the uploaded file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bz2") as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name

        # Load and parse the JSON data
        try:
            with bz2.open(tmp_path, "rt", encoding="UTF-8") as f:
                chapters_data = json.load(f)
        except Exception as e:
            flash(f"Error reading file: {str(e)}", "error")
            return redirect(url_for("config.index"))
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

        # Validate the data structure
        if not isinstance(chapters_data, list):
            flash("Invalid file format: expected a JSON array of chapters", "error")
            return redirect(url_for("config.index"))

        if not chapters_data:
            flash("File contains no chapters", "error")
            return redirect(url_for("config.index"))

        # Check that each chapter has required fields
        for i, chapter in enumerate(chapters_data):
            if not isinstance(chapter, dict):
                flash(f"Invalid chapter at index {i}: expected an object", "error")
                return redirect(url_for("config.index"))
            if "name" not in chapter or "text" not in chapter:
                flash(
                    f"Chapter at index {i} missing required fields (name, text)",
                    "error",
                )
                return redirect(url_for("config.index"))

        with get_db().transaction_cursor() as cursor:
            # Check if chapters already exist
            chapter_count = Chapter.get_chapter_count(cursor)

            if chapter_count > 0:
                msg = f"Database already contains {chapter_count} chapters. "
                msg += "Clear existing chapters first."
                flash(msg, "error")
                return redirect(url_for("config.index"))

            # Insert chapters into the database
            for i, chapter_data in enumerate(chapters_data):
                Chapter.add_chapter(
                    cursor=cursor,
                    chapter_id=i,
                    name=chapter_data["name"],
                    text=chapter_data["text"],
                )

        flash(f"Successfully loaded {len(chapters_data)} chapters", "success")
        logger.info(f"Loaded {len(chapters_data)} chapters from uploaded file")

    except Exception as e:
        logger.error(f"Error processing chapter upload: {e}")
        flash(f"Error processing file: {str(e)}", "error")
        return redirect(url_for("config.index"))

    return redirect(url_for("config.index"))


@config_bp.route("/toggle_worker", methods=["POST"])
def toggle_worker() -> Response:
    """Toggle the background worker between paused and running states."""
    worker = get_background_worker()
    status = worker.get_status()

    if status == WorkerStatus.PAUSED:
        worker.resume()
        flash("Background worker resumed", "success")
        logger.info("Background worker resumed via web interface")
    elif status == WorkerStatus.RUNNING:
        worker.pause()
        flash("Background worker paused", "success")
        logger.info("Background worker paused via web interface")
    elif status == WorkerStatus.INITIALIZED:
        worker.resume()
        flash("Background worker started and resumed", "success")
        logger.info("Background worker started via web interface")
    else:
        flash(f"Cannot toggle worker in {status.value} state", "error")
        logger.warning(f"Attempted to toggle worker in {status.value} state")

    return redirect(url_for("config.index"))


# TODO: remove this eventually, it's just for testing the frontend.
@config_bp.route("/kill_worker", methods=["POST"])
def kill_worker() -> Response:
    """Kill the background worker thread."""
    worker = get_background_worker()
    worker.kill()
    flash("Background worker killed", "warning")
    logger.info("Background worker killed via web interface")
    return redirect(url_for("config.index"))


@config_bp.route("/update_openai", methods=["POST"])
def update_openai() -> Response:
    """Update OpenAI configuration settings."""

    try:
        # Get form values
        model_value = request.form.get("model")
        verbosity_value = request.form.get("verbosity")
        reasoning_effort_value = request.form.get("reasoning_effort")
        service_tier_value = request.form.get("service_tier")
        timeout_minutes_value = request.form.get("timeout_minutes")
        compression_threshold_value = request.form.get("compression_threshold")

        # Validate and convert to enums
        try:
            model = OpenAIModel(model_value) if model_value else None
            verbosity = OpenAIVerbosity(verbosity_value) if verbosity_value else None
            reasoning_effort = (
                OpenAIReasoningEffort(reasoning_effort_value)
                if reasoning_effort_value
                else None
            )
            service_tier = (
                OpenAIServiceTier(service_tier_value) if service_tier_value else None
            )
            timeout_minutes = None
            if timeout_minutes_value:
                timeout_minutes = int(timeout_minutes_value)
                if timeout_minutes < 5 or timeout_minutes > 1440:
                    raise ValueError("Timeout must be between 5 and 1440 minutes")
            compression_threshold = None
            if compression_threshold_value:
                compression_threshold = int(compression_threshold_value)
                if compression_threshold < 1000 or compression_threshold > 1000000:
                    raise ValueError(
                        "Compression threshold must be between "
                        "1,000 and 1,000,000 tokens"
                    )
        except ValueError as e:
            flash(f"Invalid configuration value: {e}", "error")
            return redirect(url_for("config.index"))

        # Update database configuration
        with get_db().transaction_cursor() as cursor:
            if model:
                Configuration.set_openai_model(cursor, model)
            if verbosity:
                Configuration.set_openai_verbosity(cursor, verbosity)
            if reasoning_effort:
                Configuration.set_openai_reasoning_effort(cursor, reasoning_effort)
            if service_tier:
                Configuration.set_openai_service_tier(cursor, service_tier)
            if timeout_minutes is not None:
                Configuration.set_openai_timeout_minutes(cursor, timeout_minutes)
            if compression_threshold is not None:
                Configuration.set_openai_compression_threshold(
                    cursor, compression_threshold
                )

        # Update the running LLM service if it's an OpenAI service
        llm_service = get_llm_service()
        if isinstance(llm_service, OpenAILLMService):
            if model:
                llm_service.set_model(model)
            if verbosity:
                llm_service.set_verbosity(verbosity)
            if reasoning_effort:
                llm_service.set_reasoning_effort(reasoning_effort)
            if service_tier:
                llm_service.set_service_tier(service_tier)
            if timeout_minutes is not None:
                llm_service.set_timeout_minutes(timeout_minutes)
            if compression_threshold is not None:
                llm_service.set_compression_threshold(compression_threshold)

        flash("OpenAI configuration updated successfully", "success")
        logger.info("OpenAI configuration updated via web interface")

    except Exception as e:
        logger.error(f"Error updating OpenAI configuration: {e}")
        flash(f"Error updating configuration: {str(e)}", "error")

    return redirect(url_for("config.index"))


def _update_prompt(
    form_field: str,
    prompt_name: str,
    setter_func: Callable[[Cursor, str], None],
    update_llm_system: bool = False,
) -> Response:
    """Generic helper to update prompt configuration."""
    try:
        prompt_value = request.form.get(form_field)
        if not prompt_value:
            flash(f"{prompt_name} cannot be empty", "error")
            return redirect(url_for("config.index"))

        with get_db().transaction_cursor() as cursor:
            setter_func(cursor, prompt_value)

        # Update the running LLM service if it's the system prompt
        if update_llm_system:
            llm_service = get_llm_service()
            if isinstance(llm_service, OpenAILLMService):
                llm_service.system_message = prompt_value

        flash(f"{prompt_name} updated successfully", "success")
        logger.info(f"{prompt_name} updated via web interface")

    except Exception as e:
        logger.error(f"Error updating {prompt_name}: {e}")
        flash(f"Error updating prompt: {str(e)}", "error")

    return redirect(url_for("config.index"))


@config_bp.route("/update_system_prompt", methods=["POST"])
def update_system_prompt() -> Response:
    """Update the system prompt configuration."""
    return _update_prompt(
        "system_prompt",
        "System prompt",
        Configuration.set_system_prompt,
        update_llm_system=True,
    )


@config_bp.route("/update_chapter_prompt", methods=["POST"])
def update_chapter_prompt() -> Response:
    """Update the chapter prompt configuration."""
    return _update_prompt(
        "chapter_prompt",
        "Chapter prompt",
        Configuration.set_chapter_prompt,
    )


@config_bp.route("/update_compress_prompt", methods=["POST"])
def update_compress_prompt() -> Response:
    """Update the compress prompt configuration."""
    return _update_prompt(
        "compress_prompt",
        "Compress prompt",
        Configuration.set_compress_prompt,
    )

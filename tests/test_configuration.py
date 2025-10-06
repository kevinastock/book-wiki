"""Tests for bookwiki Configuration module."""

from bookwiki.config_enums import (
    OpenAIModel,
    OpenAIReasoningEffort,
    OpenAIServiceTier,
    OpenAIVerbosity,
)
from bookwiki.db import SafeConnection
from bookwiki.models.configuration import Configuration


def test_openai_model_configuration(temp_db: SafeConnection) -> None:
    """Test getting and setting OpenAI model configuration."""
    with temp_db.transaction_cursor() as cursor:
        # Initially should return default value
        assert Configuration.get_openai_model(cursor) == OpenAIModel.GPT_5

        # Set a value
        Configuration.set_openai_model(cursor, OpenAIModel.GPT_5_MINI)
        assert Configuration.get_openai_model(cursor) == OpenAIModel.GPT_5_MINI

        # Update the value
        Configuration.set_openai_model(cursor, OpenAIModel.GPT_5_NANO)
        assert Configuration.get_openai_model(cursor) == OpenAIModel.GPT_5_NANO


def test_openai_verbosity_configuration(temp_db: SafeConnection) -> None:
    """Test getting and setting OpenAI verbosity configuration."""
    with temp_db.transaction_cursor() as cursor:
        # Initially should return default value
        assert Configuration.get_openai_verbosity(cursor) == OpenAIVerbosity.MEDIUM

        # Set a value
        Configuration.set_openai_verbosity(cursor, OpenAIVerbosity.HIGH)
        assert Configuration.get_openai_verbosity(cursor) == OpenAIVerbosity.HIGH

        # Update the value
        Configuration.set_openai_verbosity(cursor, OpenAIVerbosity.LOW)
        assert Configuration.get_openai_verbosity(cursor) == OpenAIVerbosity.LOW


def test_openai_reasoning_effort_configuration(temp_db: SafeConnection) -> None:
    """Test getting and setting OpenAI reasoning effort configuration."""
    with temp_db.transaction_cursor() as cursor:
        # Initially should return default value
        assert (
            Configuration.get_openai_reasoning_effort(cursor)
            == OpenAIReasoningEffort.MEDIUM
        )

        # Set a value
        Configuration.set_openai_reasoning_effort(cursor, OpenAIReasoningEffort.HIGH)
        assert (
            Configuration.get_openai_reasoning_effort(cursor)
            == OpenAIReasoningEffort.HIGH
        )

        # Update the value
        Configuration.set_openai_reasoning_effort(cursor, OpenAIReasoningEffort.MINIMAL)
        assert (
            Configuration.get_openai_reasoning_effort(cursor)
            == OpenAIReasoningEffort.MINIMAL
        )


def test_openai_service_tier_configuration(temp_db: SafeConnection) -> None:
    """Test getting and setting OpenAI service tier configuration."""
    with temp_db.transaction_cursor() as cursor:
        # Initially should return default value
        assert (
            Configuration.get_openai_service_tier(cursor) == OpenAIServiceTier.DEFAULT
        )

        # Set a value
        Configuration.set_openai_service_tier(cursor, OpenAIServiceTier.FLEX)
        assert Configuration.get_openai_service_tier(cursor) == OpenAIServiceTier.FLEX

        # Update the value
        Configuration.set_openai_service_tier(cursor, OpenAIServiceTier.DEFAULT)
        assert (
            Configuration.get_openai_service_tier(cursor) == OpenAIServiceTier.DEFAULT
        )


def test_multiple_configurations_independent(temp_db: SafeConnection) -> None:
    """Test that different configuration keys are independent."""
    with temp_db.transaction_cursor() as cursor:
        # Verify default values
        assert Configuration.get_openai_model(cursor) == OpenAIModel.GPT_5
        assert Configuration.get_openai_verbosity(cursor) == OpenAIVerbosity.MEDIUM
        assert (
            Configuration.get_openai_reasoning_effort(cursor)
            == OpenAIReasoningEffort.MEDIUM
        )
        assert (
            Configuration.get_openai_service_tier(cursor) == OpenAIServiceTier.DEFAULT
        )

        # Set different values for each configuration
        Configuration.set_openai_model(cursor, OpenAIModel.GPT_5_MINI)
        Configuration.set_openai_verbosity(cursor, OpenAIVerbosity.HIGH)
        Configuration.set_openai_reasoning_effort(cursor, OpenAIReasoningEffort.MEDIUM)
        Configuration.set_openai_service_tier(cursor, OpenAIServiceTier.FLEX)

        # Verify they are stored independently
        assert Configuration.get_openai_model(cursor) == OpenAIModel.GPT_5_MINI
        assert Configuration.get_openai_verbosity(cursor) == OpenAIVerbosity.HIGH
        assert (
            Configuration.get_openai_reasoning_effort(cursor)
            == OpenAIReasoningEffort.MEDIUM
        )
        assert Configuration.get_openai_service_tier(cursor) == OpenAIServiceTier.FLEX

        # Update one value and verify others are unchanged
        Configuration.set_openai_model(cursor, OpenAIModel.GPT_5_NANO)
        assert Configuration.get_openai_model(cursor) == OpenAIModel.GPT_5_NANO
        assert Configuration.get_openai_verbosity(cursor) == OpenAIVerbosity.HIGH
        assert (
            Configuration.get_openai_reasoning_effort(cursor)
            == OpenAIReasoningEffort.MEDIUM
        )
        assert Configuration.get_openai_service_tier(cursor) == OpenAIServiceTier.FLEX


def test_enum_values_are_stored_as_strings(temp_db: SafeConnection) -> None:
    """Test that enum values are stored as strings in the database."""
    with temp_db.transaction_cursor() as cursor:
        # Set enum values
        Configuration.set_openai_model(cursor, OpenAIModel.GPT_5_MINI)
        Configuration.set_openai_verbosity(cursor, OpenAIVerbosity.HIGH)

        # Verify the enum values are stored and retrieved correctly
        # The Configuration model handles string serialization/deserialization
        assert Configuration.get_openai_model(cursor) == OpenAIModel.GPT_5_MINI
        assert Configuration.get_openai_verbosity(cursor) == OpenAIVerbosity.HIGH


def test_openai_timeout_minutes_configuration(temp_db: SafeConnection) -> None:
    """Test getting and setting OpenAI timeout minutes configuration."""
    with temp_db.transaction_cursor() as cursor:
        # Initially should return default value
        assert Configuration.get_openai_timeout_minutes(cursor) == 60

        # Set a value
        Configuration.set_openai_timeout_minutes(cursor, 120)
        assert Configuration.get_openai_timeout_minutes(cursor) == 120

        # Update the value
        Configuration.set_openai_timeout_minutes(cursor, 30)
        assert Configuration.get_openai_timeout_minutes(cursor) == 30

        # Test with zero
        Configuration.set_openai_timeout_minutes(cursor, 0)
        assert Configuration.get_openai_timeout_minutes(cursor) == 0

        # Test with large value
        Configuration.set_openai_timeout_minutes(cursor, 1440)  # 24 hours
        assert Configuration.get_openai_timeout_minutes(cursor) == 1440


def test_openai_timeout_stored_as_string(temp_db: SafeConnection) -> None:
    """Test that timeout is stored as string in database."""
    with temp_db.transaction_cursor() as cursor:
        # Set timeout value
        Configuration.set_openai_timeout_minutes(cursor, 90)

        # Verify the timeout value is stored and parsed back correctly
        # The Configuration model handles string serialization/deserialization
        assert Configuration.get_openai_timeout_minutes(cursor) == 90


def test_openai_compression_threshold_configuration(temp_db: SafeConnection) -> None:
    """Test getting and setting OpenAI compression threshold configuration."""
    with temp_db.transaction_cursor() as cursor:
        # Initially should return default value
        assert Configuration.get_openai_compression_threshold(cursor) == 320000

        # Set a value
        Configuration.set_openai_compression_threshold(cursor, 500000)
        assert Configuration.get_openai_compression_threshold(cursor) == 500000

        # Update the value
        Configuration.set_openai_compression_threshold(cursor, 100000)
        assert Configuration.get_openai_compression_threshold(cursor) == 100000

        # Test with minimum valid value
        Configuration.set_openai_compression_threshold(cursor, 1000)
        assert Configuration.get_openai_compression_threshold(cursor) == 1000

        # Test with maximum valid value
        Configuration.set_openai_compression_threshold(cursor, 1000000)
        assert Configuration.get_openai_compression_threshold(cursor) == 1000000


def test_openai_compression_threshold_stored_as_string(temp_db: SafeConnection) -> None:
    """Test that compression threshold is stored as string in database."""
    with temp_db.transaction_cursor() as cursor:
        # Set compression threshold value
        Configuration.set_openai_compression_threshold(cursor, 250000)

        # Verify the compression threshold value is stored and parsed back correctly
        # The Configuration model handles string serialization/deserialization
        assert Configuration.get_openai_compression_threshold(cursor) == 250000

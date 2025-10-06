# Set shell to bash and source venv before each command
set shell := ["bash", "-c", "source .venv/bin/activate && bash -c \"$0\""]

# Run all checks (formatting, linting, type checking, tests)
lint: format-check lint-check type-check test

# Run all checks in CI mode (no auto-fixes)
ci: lint

# Apply auto-fixes and then run all checks
fix: format lint-fix lint

# Format code with ruff format
format:
    ruff format . >&2 || exit 2

# Check formatting without applying changes
format-check:
    ruff format --check . >&2 || exit 2

# Run Ruff linter with auto-fixes
lint-fix:
    ruff check --fix . >&2 || exit 2

# Run Ruff linter (check only)
lint-check:
    ruff check . >&2 || exit 2

# Run type checking with MyPy
type-check:
    mypy . >&2 || exit 2

# Run tests with pytest
test:
    pytest tests/ >&2 || exit 2

# Run tests with coverage
test-cov:
    pytest tests/ --cov=bookwiki >&2 || exit 2

# Start the web server
run db="bookwiki.db" port="5000":
    bookwiki --db {{db}} --port {{port}} --dev

# Run a demo script
demo script="llm" *args="":
    bookwiki-demo-{{script}} {{args}}

# Clean up generated files
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name ".coverage" -delete
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    find . -type d -name ".mypy_cache" -exec rm -rf {} +
    find . -type d -name ".ruff_cache" -exec rm -rf {} +

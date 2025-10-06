import os
from typing import Generator

import pytest
from flask import Flask, g
from flask.testing import FlaskClient

from bookwiki.db import SafeConnection, connect_db
from bookwiki.web.filters import register_filters


@pytest.fixture
def temp_db() -> Generator[SafeConnection, None, None]:
    """Create an in-memory database with schema from schema.sql."""
    conn = connect_db(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def web_app(temp_db: SafeConnection) -> Flask:
    """Create a configured Flask test app."""
    app = Flask(
        __name__,
        template_folder=os.path.join(
            os.path.dirname(__file__), "..", "bookwiki", "web", "templates"
        ),
        static_folder=os.path.join(
            os.path.dirname(__file__), "..", "bookwiki", "web", "static"
        ),
    )
    app.config["DATABASE"] = temp_db
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret-key"

    # Register jinja filters
    register_filters(app)

    @app.before_request
    def before_request() -> None:
        """Make database available to all requests."""
        g.db = temp_db

    # Register all routes
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

    return app


@pytest.fixture
def web_client(web_app: Flask) -> Generator[FlaskClient, None, None]:
    """Create a Flask test client."""
    with web_app.test_client() as client:
        yield client

-- All datetime fields stored as TEXT in ISO 8601 format (UTC)
-- JSON fields stored as TEXT with JSON validation where supported

-- Block table: Stores all conversation blocks (messages, tool uses, responses)
CREATE TABLE IF NOT EXISTS block (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation INTEGER NOT NULL,
    create_time TEXT NOT NULL,
    generation INTEGER NOT NULL,
    tool_name TEXT,
    tool_use_id TEXT,
    tool_params TEXT CHECK (tool_params IS NULL OR json_valid(tool_params)),
    tool_response TEXT,
    text_role TEXT,
    text_body TEXT,
    sent BOOLEAN NOT NULL DEFAULT 0,
    errored BOOLEAN NOT NULL DEFAULT 0,
    FOREIGN KEY (conversation) REFERENCES conversation(id)
);

-- Conversation table: Tracks conversation sessions
CREATE TABLE IF NOT EXISTS conversation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    previously TEXT,  -- Opaque string for LLM service (history or reference ID)
    parent_block INTEGER,
    total_input_tokens INTEGER NOT NULL DEFAULT 0,
    total_output_tokens INTEGER NOT NULL DEFAULT 0,
    current_tokens INTEGER NOT NULL DEFAULT 0,
    current_generation INTEGER NOT NULL DEFAULT 0,
    waiting_on_id TEXT,  -- Optional ID that blocks this conversation
    FOREIGN KEY (parent_block) REFERENCES block(id)
);

-- Add foreign key constraint from block to conversation (circular reference)
-- SQLite doesn't support ALTER TABLE ADD CONSTRAINT, so this is handled by the FK above

-- Chapter table: Source material
CREATE TABLE IF NOT EXISTS chapter (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE CHECK (json_valid(name)),  -- JSON array of strings, must be unique
    text TEXT NOT NULL,
    conversation_id INTEGER DEFAULT NULL,  -- FK to conversation that started processing this chapter
    chapter_summary_page_id INTEGER DEFAULT NULL,  -- FK to wiki_page that summarizes this chapter
    FOREIGN KEY (conversation_id) REFERENCES conversation(id),
    FOREIGN KEY (chapter_summary_page_id) REFERENCES wiki_page(id)
);

-- Prompt table: Reusable prompt templates
CREATE TABLE IF NOT EXISTS prompt (
    key TEXT NOT NULL,
    create_time TEXT NOT NULL,
    create_block INTEGER NOT NULL,
    summary TEXT NOT NULL,
    template TEXT NOT NULL,
    PRIMARY KEY (key, create_time),
    FOREIGN KEY (create_block) REFERENCES block(id)
);

-- WikiPage table: Generated wiki content
CREATE TABLE IF NOT EXISTS wiki_page (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter INTEGER NOT NULL,
    slug TEXT NOT NULL,
    create_time TEXT NOT NULL,
    create_block INTEGER NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    body TEXT NOT NULL,
    UNIQUE (slug, chapter, create_time),
    FOREIGN KEY (chapter) REFERENCES chapter(id),
    FOREIGN KEY (create_block) REFERENCES block(id)
);

-- WikiPageCurrent table: simplify queries of wiki state at chapter
CREATE TABLE IF NOT EXISTS wiki_page_current (
    chapter INTEGER NOT NULL,
    slug TEXT NOT NULL,
    wiki_page INTEGER NOT NULL,
    PRIMARY KEY (chapter, slug),
    FOREIGN KEY (chapter) REFERENCES chapter(id),
    FOREIGN KEY (wiki_page) REFERENCES wiki_page(id)
);

-- WikiName table: Normalized names
CREATE TABLE IF NOT EXISTS wiki_name (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

-- WikiPageName junction table: Many-to-many relationship between wiki pages and names
CREATE TABLE IF NOT EXISTS wiki_page_name (
    wiki_page_id INTEGER NOT NULL,
    wiki_name_id INTEGER NOT NULL,
    PRIMARY KEY (wiki_page_id, wiki_name_id),
    FOREIGN KEY (wiki_page_id) REFERENCES wiki_page(id),
    FOREIGN KEY (wiki_name_id) REFERENCES wiki_name(id)
);

-- Configuration table: Key-value configuration storage
CREATE TABLE IF NOT EXISTS configuration (
    key TEXT NOT NULL PRIMARY KEY,
    value TEXT NOT NULL
);

-- Create indices for common queries (to be optimized later based on usage patterns)
CREATE INDEX IF NOT EXISTS id_block_conversation ON block(conversation);
CREATE INDEX IF NOT EXISTS id_block_sent ON block(sent);
CREATE INDEX IF NOT EXISTS id_block_tool_unresponded ON block(tool_name, tool_response, create_time);
CREATE INDEX IF NOT EXISTS id_conversation_parent_block ON conversation(parent_block);
CREATE INDEX IF NOT EXISTS id_conversation_waiting_on_id ON conversation(waiting_on_id);
CREATE INDEX IF NOT EXISTS id_wiki_page_chapter ON wiki_page(chapter);
CREATE INDEX IF NOT EXISTS id_wiki_page_slug ON wiki_page(slug);
CREATE INDEX IF NOT EXISTS id_wiki_page_name_wiki_name_id ON wiki_page_name(wiki_name_id);
CREATE INDEX IF NOT EXISTS id_chapter_conversation_id ON chapter(conversation_id);

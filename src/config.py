"""Configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set.")

_user = os.environ.get("POSTGRES_USER", "postgres")
_password = os.environ.get("POSTGRES_PASSWORD", "postgres")
_host = os.environ.get("POSTGRES_HOST", "localhost")
_port = os.environ.get("POSTGRES_PORT", "5432")
_db = os.environ.get("POSTGRES_DB", "semantic_search")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"postgresql+psycopg://{_user}:{_password}@{_host}:{_port}/{_db}",
)

COLLECTION_NAME = "documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SEARCH_TOP_K = 10
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-nano"

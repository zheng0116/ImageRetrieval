from __future__ import annotations

import os
import uvicorn
from dotenv import load_dotenv
from retrieval import Initializer
from retrieval import set_logger

logger = set_logger("main", "INFO")
TRUE_VALUES = {"1", "true", "yes", "on"}

load_dotenv(verbose=True)


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in TRUE_VALUES


def parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid integer config value '{value}', fallback to {default}")
        return default


embed_model = os.getenv("EMBED_MODEL")
if not embed_model:
    embed_model = "siglip2_base"

config = {
    "server_url": os.getenv("SERVER_URL", "0.0.0.0"),
    "server_port": parse_int(os.getenv("SERVER_PORT"), 5999),
    "database_folder": os.getenv("DATABASE_FOLDER", "./quary"),
    "model_size": os.getenv("MODEL_SIZE", "small"),
    "model_path": os.getenv("MODEL_PATH"),
    "embed_model": embed_model.strip().lower(),
    "top_k": parse_int(os.getenv("TOP_K"), 20),
}

initializer = Initializer(config)
app = initializer.initialize()

if __name__ == "__main__":
    uvicorn.run(app, host=config["server_url"], port=config["server_port"])

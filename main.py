import os
import uvicorn
from dotenv import load_dotenv
from retrieval import set_logger
from retrieval import Initializer

logger = set_logger("main", "INFO")

load_dotenv(verbose=True)

config = {
    "server_url": os.getenv("SERVER_URL"),
    "server_port": int(os.getenv("SERVER_PORT")),
    "database_folder": os.getenv("DATABASE_FOLDER"),
    "model_size": os.getenv("MODEL_SIZE"),
    "model_path": os.getenv("MODEL_PATH"),
    "overall_model": os.getenv(
        "OVERALL_MODEL", "False"
    ),  # based on clip model achive overall retrieval,deafult is False
    "top_k": int(os.getenv("TOP_K", 20)),
}

initializer = Initializer(config)
app = initializer.initialize()

if __name__ == "__main__":
    uvicorn.run(app, host=config["server_url"], port=config["server_port"])

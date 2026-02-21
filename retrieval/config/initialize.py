from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from retrieval import Clip, Dinov2, ImagePreprocessor, RetrievalProcessor, Routes

from .logger import set_logger

logger = set_logger("initialize", "INFO")

MULTIMODAL_MODEL_REGISTRY = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip2_base": "google/siglip2-base-patch16-224",
    "mobileclip2_s0": "apple/MobileCLIP2-S0",
}


class Initializer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.app = FastAPI()
        self.preprocessor = None
        self.retriever = None
        self.routes = None

    def initialize(self) -> FastAPI:
        self.setup_static_files()
        self.initialize_components()
        self.setup_routes()
        self.setup_root_route()
        return self.app

    def setup_static_files(self) -> None:
        self.app.mount("/static", StaticFiles(directory="static"), name="static")

    def initialize_components(self) -> None:
        logger.info(
            f"Initializing system with database folder: {self.config['database_folder']}"
        )
        logger.info(f"Model size: {self.config['model_size']}")
        logger.info(f"Model path: {self.config['model_path']}")
        logger.info(f"Embedding model: {self.config['embed_model']}")
        dinov2_loader = Dinov2(
            model_size=self.config["model_size"], model_path=self.config["model_path"]
        )
        self.preprocessor = ImagePreprocessor()
        multimodal_model_key = (
            "clip"
            if self.config["embed_model"] == "dinov2"
            else self.config["embed_model"]
        )
        model_name = MULTIMODAL_MODEL_REGISTRY.get(multimodal_model_key)
        if model_name is None:
            raise ValueError(
                f"Unsupported embed model '{self.config['embed_model']}'. "
                f"Supported: dinov2, {', '.join(MULTIMODAL_MODEL_REGISTRY.keys())}"
            )

        clip_loader = Clip(model_name=model_name)

        self.retriever = RetrievalProcessor(
            dinov2_loader,
            clip_loader,
            self.preprocessor,
            self.config["database_folder"],
            self.config["embed_model"],
            multimodal_model_key,
            self.config["top_k"],
        )

    def setup_routes(self) -> None:
        self.routes = Routes(self.retriever, self.config["database_folder"])
        self.app.include_router(self.routes.router)

    def setup_root_route(self) -> None:
        @self.app.get("/")
        async def read_root():
            return FileResponse("static/index.html")

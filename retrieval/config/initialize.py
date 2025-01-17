from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from retrieval import Dinov2, ImagePreprocessor, RetrievalProcessor, Clip, Routes
from .logger import set_logger

logger = set_logger("initialize", "INFO")


class Initializer:
    def __init__(self, config: dict):
        self.config = config
        self.app = FastAPI()
        self.preprocessor = None
        self.retriever = None
        self.routes = None

    def initialize(self):
        self._setup_static_files()
        self._initialize_components()
        self._setup_routes()
        self._setup_root_route()
        return self.app

    def _setup_static_files(self):
        self.app.mount("/static", StaticFiles(directory="static"), name="static")

    def _initialize_components(self):
        """Initialize model components and processors"""
        logger.info(
            f"Initializing system with database folder: {self.config['database_folder']}"
        )
        logger.info(f"Model size: {self.config['model_size']}")
        logger.info(f"Model path: {self.config['model_path']}")

        dinov2_loader = Dinov2(
            model_size=self.config["model_size"], model_path=self.config["model_path"]
        )
        self.preprocessor = ImagePreprocessor()
        clip_loader = Clip()

        self.retriever = RetrievalProcessor(
            dinov2_loader,
            clip_loader,
            self.preprocessor,
            self.config["database_folder"],
        )

    def _setup_routes(self):
        """Initialize and setup routes"""
        self.routes = Routes(self.retriever, self.config["database_folder"])
        self.app.include_router(self.routes.router)

    def _setup_root_route(self):
        """Setup the root route for the application"""

        @self.app.get("/")
        async def read_root():
            return FileResponse("static/index.html")

import shutil
from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import FileResponse
from pathlib import Path
from .logger import set_logger

logger = set_logger("router", "INFO")


class Routes:
    def __init__(self, retriever, database_folder):
        self.retriever = retriever
        self.database_folder = database_folder
        self.router = APIRouter()
        self.setup_routes()

    def setup_routes(self):
        @self.router.post("/retrieve")
        async def retrieve_images(request: Request, file: UploadFile = File(...)):
            temp_file = Path("temp_image.jpg")
            with temp_file.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            try:
                results = self.retriever.image_to_image_retrieve(temp_file)
                temp_file.unlink()
                base_url = str(request.base_url)
                full_results = [
                    {
                        "url": f"{base_url}images/{Path(path).name}",
                        "similarity": similarity,
                    }
                    for path, similarity in results
                ]
                return {"results": full_results}
            except Exception as e:
                logger.error(f"Error during retrieval: {str(e)}")
                return {"error": str(e)}

        @self.router.get("/images/{image_name}")
        async def get_image(image_name: str):
            image_path = Path(self.database_folder) / image_name
            if not image_path.exists():
                return {"error": "Image not found"}
            return FileResponse(image_path)

        @self.router.post("/text_retrieve")
        async def text_retrieve(text_query: dict):
            try:
                results = self.retriever.text_to_image_retrieve(text_query["text"])
                return {
                    "results": [
                        {"url": f"/images/{Path(path).name}", "similarity": similarity}
                        for path, similarity in results
                    ]
                }
            except Exception as e:
                return {"error": str(e)}

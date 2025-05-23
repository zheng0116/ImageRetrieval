import shutil
from fastapi import APIRouter, UploadFile, File, Request, HTTPException
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

        @self.router.post("/add_image")
        async def add_image(file: UploadFile = File(...)):
            try:
                file_path = Path(self.database_folder) / file.filename
                with file_path.open("wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                self.retriever.add_images([file_path])

                return {"message": f"Image {file.filename} added successfully"}
            except Exception as e:
                logger.error(f"Error adding image: {str(e)}")
                return {"error": str(e)}

        @self.router.delete("/delete_image/{image_name}")
        async def delete_image(image_name: str):
            """Delete an image from the database"""
            try:
                image_path = Path(self.database_folder) / image_name
                if not image_path.exists():
                    raise HTTPException(status_code=404, detail="Image not found")
                self.retriever.delete_image(image_path)
                image_path.unlink()

                return {"message": f"Image {image_name} deleted successfully"}
            except Exception as e:
                logger.error(f"Error deleting image: {str(e)}")
                return {"error": str(e)}

        @self.router.get("/database_stats")
        async def get_database_stats():
            try:
                stats = self.retriever.get_database_stats()
                return stats
            except Exception as e:
                logger.error(f"Error getting database stats: {str(e)}")
                return {"error": str(e)}

        @self.router.post("/rebuild_database")
        async def rebuild_database():
            try:
                self.retriever.check_features()
                return {"message": "Database rebuilt successfully"}
            except Exception as e:
                logger.error(f"Error rebuilding database: {str(e)}")
                return {"error": str(e)}

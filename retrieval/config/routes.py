from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .logger import set_logger

logger = set_logger("router", "INFO")


class TextRetrieveRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text query for image retrieval")


class Routes:
    def __init__(self, retriever: Any, database_folder: str):
        self.retriever = retriever
        self.database_folder = Path(database_folder).resolve()
        self.router = APIRouter()
        self.setup_routes()

    def _safe_filename(self, name: str) -> str:
        filename = Path(name or "").name
        if not filename:
            raise HTTPException(status_code=400, detail="Invalid file name")
        if filename != name:
            raise HTTPException(status_code=400, detail="Invalid file path")
        return filename

    def _resolve_db_file(self, image_name: str) -> Path:
        safe_name = self._safe_filename(image_name)
        image_path = (self.database_folder / safe_name).resolve()
        if image_path.parent != self.database_folder:
            raise HTTPException(status_code=400, detail="Invalid file path")
        return image_path

    def setup_routes(self):
        @self.router.post("/retrieve")
        async def retrieve_images(request: Request, file: UploadFile = File(...)):
            suffix = Path(file.filename or "").suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                temp_file = Path(tmp_file.name)

            try:
                results = self.retriever.image_to_image_retrieve(temp_file)
                base_url = str(request.base_url)
                full_results = [
                    {
                        "url": f"{base_url}images/{Path(path).name}",
                        "similarity": similarity,
                    }
                    for path, similarity in results
                ]
                return {"results": full_results}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error during retrieval: {str(e)}")
                raise HTTPException(status_code=500, detail="Image retrieval failed")
            finally:
                temp_file.unlink(missing_ok=True)

        @self.router.get("/images/{image_name}")
        async def get_image(image_name: str):
            image_path = self._resolve_db_file(image_name)
            if not image_path.exists():
                raise HTTPException(status_code=404, detail="Image not found")
            return FileResponse(image_path)

        @self.router.post("/text_retrieve")
        async def text_retrieve(text_query: TextRetrieveRequest):
            try:
                query_text = text_query.text.strip()
                results = self.retriever.text_to_image_retrieve(query_text)
                return {
                    "results": [
                        {"url": f"/images/{Path(path).name}", "similarity": similarity}
                        for path, similarity in results
                    ]
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error in text retrieval: {str(e)}")
                raise HTTPException(status_code=500, detail="Text retrieval failed")

        @self.router.post("/add_image")
        async def add_image(file: UploadFile = File(...)):
            try:
                safe_name = self._safe_filename(file.filename or "")
                file_path = self._resolve_db_file(safe_name)
                with file_path.open("wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                self.retriever.add_images([file_path])

                return {"message": f"Image {safe_name} added successfully"}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error adding image: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to add image")

        @self.router.delete("/delete_image/{image_name}")
        async def delete_image(image_name: str):
            try:
                image_path = self._resolve_db_file(image_name)
                if not image_path.exists():
                    raise HTTPException(status_code=404, detail="Image not found")
                self.retriever.delete_image(image_path)
                image_path.unlink()

                return {"message": f"Image {image_path.name} deleted successfully"}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error deleting image: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to delete image")

        @self.router.get("/database_stats")
        async def get_database_stats():
            try:
                return self.retriever.get_database_stats()
            except Exception as e:
                logger.error(f"Error getting database stats: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to get database stats")

        @self.router.post("/rebuild_database")
        async def rebuild_database():
            try:
                self.retriever.check_features()
                return {"message": "Database rebuilt successfully"}
            except Exception as e:
                logger.error(f"Error rebuilding database: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to rebuild database")

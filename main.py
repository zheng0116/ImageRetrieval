import shutil
import argparse
import logging
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
from model.Diniv2 import ModelLoader
from utils.image_preprocessor import ImagePreprocessor
from utils.retrieval_processor import RetrievalProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

model_loader = None
preprocessor = None
retriever = None
database_folder = None


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return Path("static/index.html").read_text()


@app.post("/retrieve")
async def retrieve_images(request: Request, file: UploadFile = File(...)):
    temp_file = Path("temp_image.jpg")
    with temp_file.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        results = retriever.retrieve(temp_file)
        temp_file.unlink()
        base_url = str(request.base_url)
        full_results = [
            {"url": f"{base_url}images/{Path(path).name}", "similarity": similarity}
            for path, similarity in results
        ]

        return {"results": full_results}
    except Exception as e:
        logger.error(f"Error during retrieval: {str(e)}")
        return {"error": str(e)}


@app.get("/images/{image_name}")
async def get_image(image_name: str):
    image_path = Path(database_folder) / image_name
    if not image_path.exists():
        return {"error": "Image not found"}
    return FileResponse(image_path)


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv2 Image Retrieval System")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./Dinov2_model/dinov2-small",
        help="Path to the DINOv2 model",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "base", "large", "giant"],
        help="Size of the DINOv2 model",
    )
    parser.add_argument(
        "--database_folder",
        type=str,
        default="./quary",
        help="Path to the image database folder",
    )
    return parser.parse_args()


def initialize_system(args):
    global model_loader, preprocessor, retriever, database_folder
    logger.info(f"Initializing system with database folder: {args.database_folder}")
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Model path: {args.model_path}")

    database_folder = args.database_folder
    model_loader = ModelLoader(model_size=args.model_size, model_path=args.model_path)
    preprocessor = ImagePreprocessor()
    retriever = RetrievalProcessor(model_loader, preprocessor, database_folder)


if __name__ == "__main__":
    args = parse_args()
    initialize_system(args)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

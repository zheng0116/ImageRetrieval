# Image Retrieval System
 <strong>[中文](./README.md) |
    English</strong>
## 🌟 Introduction
This project is an image retrieval system based on DINOv2 and CLIP models. It uses Chroma vector database to support both text-to-image and image-to-image retrieval.

## Todo
- [x] support embedding database
- [ ] using rust language achieve
- [ ] support different models to extract image features and text features
- [ ] support rpc agreement
## 🚀 Features
- Image feature extraction using the DINOv2 model for image-to-image search
- Text-to-image search powered by CLIP model
- Support for different sizes of DINOv2 models (small, base, large, giant)
- Image retrieval based on cosine similarity
- Web interface built with FastAPI
- Uses Chroma vector database to retrieval images

## User Interface

![DINOv2 Image Retrieval System Interface](./images/web.png)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/zheng0116/ImageRetrieval.git
cd ImageRetrieval
```

2. Install dependencies:

```bash
sh run.sh install
```

3. Download the Dinov2 weights:
```bash
https://pan.baidu.com/s/1fBVgg_o8PTFEu_2vtLY25Q
Extraction code: f9ww
```

4. Configure environment variables:
Create a `.env` file in the root directory with the following content:
```bash
SERVER_URL="0.0.0.0"
SERVER_PORT=5999
MODEL_PATH="./Dinov2_model/dinov2-small"
MODEL_SIZE="small"
DATABASE_FOLDER="./quary"
```

## Usage

1. Prepare your image database by placing images in the `quary` folder (or specify a custom folder in .env).

2. Run the application:

```bash
sh run.sh start
```

3. Open a web browser and navigate to `http://localhost:5999`.

4. Search Methods:
   - Image-to-Image: Upload an image and click "Start Search"
   - Text-to-Image: Enter text description in the search box and click "Text Search"

## Configuration

You can configure the following parameters in your .env file:

- `SERVER_URL`: Server IP address (default: "0.0.0.0")
- `SERVER_PORT`: Server port number (default: 5999)
- `MODEL_PATH`: Path to the DINOv2 model (default: "./Dinov2_model/dinov2-small")
- `MODEL_SIZE`: Size of the DINOv2 model (choices: small, base, large, giant; default: small)
- `DATABASE_FOLDER`: Path to the image database folder (default: "./quary")

## Project Structure

- `main.py`: Main application file with FastAPI server
- `retrieval/services/model/Diniv2.py`: DINOv2 model loader
- `retrieval/services/model/clip.py`: CLIP model loader
- `retrieval/services/utils/image_process.py`: Image preprocessing 
- `retrieval/services/retrieval.py`: Image retrieval logic
- `static/index.html`: Web interface
- `config` : Logging configuration, model initialization and routing configuration
## Requirements

- Python 3.7+
- chroma
- FastAPI
- Uvicorn
- PyTorch
- Transformers
- Pillow
- NumPy
- tqdm

## License

[MIT License](LICENSE)

## Acknowledgements

- [DINOv2](https://github.com/facebookresearch/dinov2) by Facebook Research
- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [FastAPI](https://fastapi.tiangolo.com/)
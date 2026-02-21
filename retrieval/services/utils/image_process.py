from pathlib import Path
from typing import Union

from PIL import Image

ImageInput = Union[str, Path, Image.Image]


class ImagePreprocessor:
    def __init__(self, size: int = 224) -> None:
        self.size = size

    def preprocess(self, image: ImageInput) -> Image.Image:
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                return img.convert("RGB").resize((self.size, self.size))
        elif not isinstance(image, Image.Image):
            raise TypeError(
                "Image must be a PIL Image object, a string path, or a Path object"
            )

        return image.convert("RGB").resize((self.size, self.size))

from PIL import Image
from pathlib import Path


class ImagePreprocessor:
    def __init__(self, size=224):
        self.size = size

    def preprocess(self, image):
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif not isinstance(image, Image.Image):
            raise TypeError(
                "Image must be a PIL Image object, a string path, or a Path object"
            )

        return image.convert("RGB").resize((self.size, self.size))

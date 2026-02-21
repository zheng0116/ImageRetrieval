from typing import Any

import torch
from transformers import AutoModel, AutoProcessor

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


class Clip:
    def __init__(self, model_name: str = CLIP_MODEL_NAME) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(
            self.device
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        return features / features.norm(dim=-1, keepdim=True)

    def _extract_from_output(self, outputs: Any, field_name: str) -> torch.Tensor:
        if hasattr(outputs, field_name):
            return getattr(outputs, field_name)
        raise ValueError(f"Model {self.model_name} does not provide {field_name}.")

    def extract_image_features(self, image: Any):
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            if hasattr(self.model, "get_image_features"):
                image_features = self.model.get_image_features(**inputs)
            else:
                outputs = self.model(**inputs)
                image_features = self._extract_from_output(outputs, "image_embeds")

            image_features = self._normalize(image_features)
            return image_features.squeeze().cpu().numpy()

    def extract_text_features(self, text: str):
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(
                self.device
            )
            if hasattr(self.model, "get_text_features"):
                text_features = self.model.get_text_features(**inputs)
            else:
                outputs = self.model(**inputs)
                text_features = self._extract_from_output(outputs, "text_embeds")

            text_features = self._normalize(text_features)
            return text_features.squeeze().cpu().numpy()

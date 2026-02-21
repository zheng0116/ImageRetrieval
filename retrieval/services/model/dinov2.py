from __future__ import annotations

from typing import Any

import torch
from transformers import AutoImageProcessor, AutoModel


class Dinov2:
    def __init__(self, model_size: str = "base", model_path: str | None = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor = self.load_model(model_size, model_path)

    def load_model(
        self, model_size: str, model_path: str | None
    ) -> tuple[Any, AutoImageProcessor]:
        model_name = f"facebook/dinov2-{model_size}"
        if model_path:
            model = AutoModel.from_pretrained(model_path)
            processor = AutoImageProcessor.from_pretrained(model_path)
        else:
            model = AutoModel.from_pretrained(model_name)
            processor = AutoImageProcessor.from_pretrained(model_name)

        model = model.to(self.device)
        model.eval()
        return model, processor

    def __call__(self, image: Any):
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze().cpu().numpy()

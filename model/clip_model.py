import torch
from transformers import CLIPProcessor, CLIPModel


class CLIPLoader:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    def extract_image_features(self, image):
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            image_features = self.model.get_image_features(**inputs)
        return image_features.squeeze().cpu().numpy()

    def extract_text_features(self, text):
        with torch.no_grad():
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            text_features = self.model.get_text_features(**inputs)
        return text_features.squeeze().cpu().numpy()

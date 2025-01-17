import torch
from transformers import CLIPProcessor, CLIPModel


class Clip:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    def extract_image_features(self, image):
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.squeeze().cpu().numpy()

    def extract_text_features(self, text):
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(
                self.device
            )
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.squeeze().cpu().numpy()

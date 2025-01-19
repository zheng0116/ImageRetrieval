import os
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from joblib import dump, load
from ..config.logger import set_logger

logger = set_logger("retrieval", "INFO")


class RetrievalProcessor:
    def __init__(self, Dinov2_loader, clip_loader, preprocessor, database_folder):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Dinov2 = Dinov2_loader
        self.clip_loader = clip_loader
        self.preprocessor = preprocessor
        self.database_folder = Path(database_folder)
        self.dinov2_cache_path = self.database_folder / "dinov2_features_cache.joblib"
        self.clip_cache_path = self.database_folder / "clip_features_cache.joblib"
        self.dinov2_features, self.database_paths = self.get_dinov2_features()
        self.clip_features = self.get_clip_features()
        self.cos = nn.CosineSimilarity(dim=0)

    def get_image_files(self):
        """Get all image files from the database folder"""
        support_formats = ("*.jpg", "*.jpeg", "*.png")
        image_paths = [
            path
            for pattern in support_formats
            for path in self.database_folder.glob(pattern)
        ]
        logger.info(f"Contents of {self.database_folder}:")
        for item in os.listdir(self.database_folder):
            full_path = self.database_folder / item
            if full_path.is_dir():
                logger.info(f"  Directory: {item}")
                for subitem in os.listdir(full_path):
                    logger.info(f"    {subitem}")
            else:
                logger.info(f"  File: {item}")
        return image_paths

    def compute_dinov2_features(self, img_paths):
        """Compute DINOv2 features for a list of images"""
        features = []
        for img_path in tqdm(img_paths, desc="Computing DINOv2 features"):
            try:
                img = self.preprocessor.preprocess(img_path)
                with torch.no_grad():
                    feature = torch.tensor(self.Dinov2(img)).to(self.device)
                    feature = feature / torch.norm(feature)
                features.append(feature)
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {str(e)}")
        return torch.stack(features)

    def get_dinov2_features(self):
        """Get DINOv2 features from cache if available, otherwise compute them"""
        if self.dinov2_cache_path.exists():
            logger.info("Loading DINOv2 features from cache")
            try:
                features, paths = load(self.dinov2_cache_path, mmap_mode="r")
                logger.info(f"Loaded {len(features)} DINOv2 features from cache")
            except Exception as e:
                logger.error(f"Error loading cache: {str(e)}")
                features, paths = self._compute_and_save_dinov2_features()
        else:
            logger.info("Cache not found. Computing DINOv2 features from images")
            features, paths = self._compute_and_save_dinov2_features()

        if len(features) == 0:
            raise ValueError("No features available in the database.")
        return features, paths

    def _compute_and_save_dinov2_features(self):
        """Helper method to compute and cache DINOv2 features"""
        img_paths = self.get_image_files()
        if len(img_paths) == 0:
            raise ValueError(
                f"No images found in the database folder: {self.database_folder}"
            )
        features = self.compute_dinov2_features(img_paths)
        paths = img_paths
        logger.info(f"Computed DINOv2 features from {len(features)} images")

        try:
            dump((features, paths), self.dinov2_cache_path, compress=3)
            logger.info("Successfully cached DINOv2 features")
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")

        return features, paths

    def get_clip_features(self):
        """Get CLIP features from cache if available, otherwise compute them"""
        if self.clip_cache_path.exists():
            logger.info("Loading CLIP features from cache")
            try:
                features = load(self.clip_cache_path, mmap_mode="r")
            except Exception as e:
                logger.error(f"Error loading CLIP cache: {str(e)}")
                features = self._compute_and_save_clip_features()
        else:
            logger.info("Computing CLIP features from images")
            features = self._compute_and_save_clip_features()
        return features

    def _compute_and_save_clip_features(self):
        """Helper method to compute and cache CLIP features"""
        features = []
        for img_path in tqdm(self.database_paths, desc="Computing CLIP features"):
            img = self.preprocessor.preprocess(img_path)
            with torch.no_grad():
                feature = torch.tensor(self.clip_loader.extract_image_features(img)).to(
                    self.device
                )
                feature = feature / torch.norm(feature)
            features.append(feature)
        features = torch.stack(features)

        try:
            dump(features, self.clip_cache_path, compress=3)
            logger.info("Successfully cached CLIP features")
        except Exception as e:
            logger.error(f"Error saving CLIP cache: {str(e)}")

        return features

    def calculate_dinov2_similarity(self, query_feature):
        if self.dinov2_features.shape[0] == 0:
            raise ValueError("DINOv2 features are empty.")

        query_tensor = torch.tensor(query_feature).to(self.device)
        query_tensor = query_tensor / torch.norm(query_tensor)

        similarities = []
        for feature in self.dinov2_features:
            sim = self.cos(query_tensor, feature).item()
            similarities.append(sim)

        return np.array(similarities)

    def image_to_image_retrieve(self, query_image):
        query_image = self.preprocessor.preprocess(query_image)
        query_feature = self.Dinov2(query_image)

        logger.info(f"Query feature shape: {query_feature.shape}")
        logger.info(f"Database features shape: {self.dinov2_features.shape}")

        similarities = self.calculate_dinov2_similarity(query_feature)
        sorted_indices = np.argsort(similarities)[::-1]

        return [
            (str(self.database_paths[i]), float(similarities[i]))
            for i in sorted_indices
        ]

    def text_to_image_retrieve(self, query_text):
        with torch.no_grad():
            query_feature = torch.tensor(
                self.clip_loader.extract_text_features(query_text)
            ).to(self.device)
            query_feature = query_feature / torch.norm(query_feature)
            similarities = []
            for feature in self.clip_features:
                sim = self.cos(query_feature, feature).item()
                sim = (sim + 1) / 2
                similarities.append(sim)

            similarities = np.array(similarities)
            sorted_indices = np.argsort(similarities)[::-1]

            return [
                (str(self.database_paths[i]), float(similarities[i]))
                for i in sorted_indices
            ]

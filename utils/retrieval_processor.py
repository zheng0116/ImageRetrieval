import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalProcessor:
    def __init__(self, model_loader, preprocessor, database_folder):
        self.model_loader = model_loader
        self.preprocessor = preprocessor
        self.database_folder = Path(database_folder)
        self.cache_path = self.database_folder / "features_cache.pkl"
        self.database_features, self.database_paths = self.load_or_extract_features()

    def glob_images(self):
        image_paths = (
            list(self.database_folder.glob("*.jpg"))
            + list(self.database_folder.glob("*.jpeg"))
            + list(self.database_folder.glob("*.png"))
        )
        logger.info(f"Found {len(image_paths)} images in the database folder")

        # Log all files in the database folder
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

    def extract_features(self, img_paths):
        features = []
        for img_path in tqdm(img_paths, desc="Extracting features"):
            try:
                img = self.preprocessor.preprocess(img_path)
                feature = self.model_loader(img)
                features.append(feature)
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {str(e)}")
        return np.array(features)

    def load_or_extract_features(self):
        if self.cache_path.exists():
            logger.info("Loading features from cache")
            with open(self.cache_path, "rb") as f:
                features, paths = pickle.load(f)
            logger.info(f"Loaded {len(features)} features from cache")
        else:
            logger.info("Cache not found. Extracting features from images")
            img_paths = self.glob_images()
            if len(img_paths) == 0:
                raise ValueError(
                    f"No images found in the database folder: {self.database_folder}"
                )
            features = self.extract_features(img_paths)
            paths = img_paths
            logger.info(f"Extracted features from {len(features)} images")
            with open(self.cache_path, "wb") as f:
                pickle.dump((features, paths), f)

        if len(features) == 0:
            raise ValueError(
                "No features extracted from the database. Please check the database folder and ensure it contains valid image files."
            )

        return features, paths

    def calculate_similarity(self, query_feature):
        if self.database_features.shape[0] == 0:
            raise ValueError(
                "Database features are empty. Please check the database folder and feature extraction process."
            )
        return np.dot(self.database_features, query_feature)

    def retrieve(self, query_image):
        query_image = self.preprocessor.preprocess(query_image)
        query_feature = self.model_loader(query_image)

        logger.info(f"Query feature shape: {query_feature.shape}")
        logger.info(f"Database features shape: {self.database_features.shape}")

        similarities = self.calculate_similarity(query_feature)
        sorted_indices = np.argsort(similarities)[::-1]

        return [
            (str(self.database_paths[i]), float(similarities[i]))
            for i in sorted_indices
        ]

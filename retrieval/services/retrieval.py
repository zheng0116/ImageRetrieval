import os
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from ..config.logger import set_logger

logger = set_logger("retrieval", "INFO")


class RetrievalProcessor:
    def __init__(
        self,
        Dinov2_loader,
        clip_loader,
        preprocessor,
        database_folder,
        overall_model,
        top_k,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Dinov2 = Dinov2_loader
        self.clip_loader = clip_loader
        self.preprocessor = preprocessor
        self.database_folder = Path(database_folder)
        self.overall_model = overall_model
        self.top_k = top_k
        self.chroma_db_path = self.database_folder / "chroma_db"
        self.chroma_db_path.mkdir(exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_db_path), settings=Settings(anonymized_telemetry=False)
        )

        self.dinov2_collection = self.chroma_client.get_or_create_collection(
            name="dinov2_features", metadata={"hnsw:space": "cosine"}
        )

        self.clip_collection = self.chroma_client.get_or_create_collection(
            name="clip_features", metadata={"hnsw:space": "cosine"}
        )

        self.database_paths = self.get_image_files()
        self.check_features()

        self.cos = nn.CosineSimilarity(dim=0)

    def get_image_files(self):
        support_formats = ("*.jpg", "*.jpeg", "*.png")
        image_paths = [
            path
            for pattern in support_formats
            for path in self.database_folder.glob(pattern)
        ]
        return image_paths

    def check_features(self):
        dinov2_count = self.dinov2_collection.count()  # Count DINOv2 features
        clip_count = self.clip_collection.count()  # Count CLIP features
        expected_count = len(self.database_paths)  # database size

        if (
            dinov2_count == 0 or abs(dinov2_count - expected_count) > 0
        ):  # todo revise logic
            logger.info("Computing and storing DINOv2 features...")
            self.store_dinov2_features()
        else:
            logger.info("DINOv2 features already exist in database")

        if clip_count == 0 or abs(clip_count - expected_count) > 0:
            logger.info("Computing and storing CLIP features...")
            self.store_clip_features()
        else:
            logger.info("CLIP features already exist in database")

    def store_dinov2_features(self):
        if len(self.database_paths) == 0:
            raise ValueError(
                f"No images found in the database folder: {self.database_folder}"
            )
        try:
            self.chroma_client.delete_collection("dinov2_features")
        except Exception as e:
            logger.info(
                f"Collection dinov2_features doesn't exist or couldn't be deleted: {e}"
            )
        self.dinov2_collection = self.chroma_client.get_or_create_collection(
            name="dinov2_features", metadata={"hnsw:space": "cosine"}
        )
        embeddings = []
        documents = []
        ids = []
        metadatas = []

        for idx, img_path in enumerate(
            tqdm(self.database_paths, desc="Computing DINOv2 features")
        ):
            try:
                img = self.preprocessor.preprocess(img_path)
                with torch.no_grad():
                    feature = torch.tensor(self.Dinov2(img)).to(self.device)
                    feature = feature / torch.norm(feature)

                embeddings.append(feature.cpu().numpy().tolist())
                documents.append(str(img_path))
                ids.append(f"dinov2_{idx}")
                metadatas.append({"image_path": str(img_path), "type": "dinov2"})

            except Exception as e:
                logger.error(f"Error processing image {img_path}: {str(e)}")

        if embeddings:
            self.dinov2_collection.add(
                embeddings=embeddings, documents=documents, ids=ids, metadatas=metadatas
            )
            logger.info(f"Successfully stored {len(embeddings)} DINOv2 features")

    def store_clip_features(self):
        try:
            self.chroma_client.delete_collection("clip_features")
        except Exception as e:
            logger.info(
                f"Collection clip_features doesn't exist or couldn't be deleted: {e}"
            )

        self.clip_collection = self.chroma_client.get_or_create_collection(
            name="clip_features", metadata={"hnsw:space": "cosine"}
        )

        embeddings = []
        documents = []
        ids = []
        metadatas = []

        for idx, img_path in enumerate(
            tqdm(self.database_paths, desc="Computing CLIP features")
        ):
            try:
                img = self.preprocessor.preprocess(img_path)
                with torch.no_grad():
                    feature = torch.tensor(
                        self.clip_loader.extract_image_features(img)
                    ).to(self.device)
                    feature = feature / torch.norm(feature)

                embeddings.append(feature.cpu().numpy().tolist())
                documents.append(str(img_path))
                ids.append(f"clip_{idx}")
                metadatas.append({"image_path": str(img_path), "type": "clip"})

            except Exception as e:
                logger.error(f"Error processing image {img_path}: {str(e)}")

        if embeddings:
            self.clip_collection.add(
                embeddings=embeddings, documents=documents, ids=ids, metadatas=metadatas
            )
            logger.info(f"Successfully stored {len(embeddings)} CLIP features")

    def image_to_image_retrieve(self, query_image):
        """Retrieve similar images using ChromaDB"""
        query_image = self.preprocessor.preprocess(query_image)

        if self.overall_model == "True":
            # Use CLIP for retrieval
            query_feature = self.clip_loader.extract_image_features(query_image)
            logger.info(f"Query feature shape: {query_feature.shape}")

            results = self.clip_collection.query(
                query_embeddings=[query_feature.tolist()], n_results=self.top_k
            )
        else:
            # Use DINOv2 for retrieval
            query_feature = self.Dinov2(query_image)
            logger.info(f"Query feature shape: {query_feature.shape}")

            results = self.dinov2_collection.query(
                query_embeddings=[query_feature.tolist()], n_results=self.top_k
            )

        retrieved_results = []
        if results["documents"] and results["distances"]:
            for doc, distance in zip(results["documents"][0], results["distances"][0]):
                similarity = 1.0 - distance
                retrieved_results.append((doc, float(similarity)))

        return retrieved_results

    def text_to_image_retrieve(self, query_text):
        with torch.no_grad():
            query_feature = self.clip_loader.extract_text_features(query_text)

            results = self.clip_collection.query(
                query_embeddings=[query_feature.tolist()], n_results=self.top_k
            )

        retrieved_results = []
        if results["documents"] and results["distances"]:
            for doc, distance in zip(results["documents"][0], results["distances"][0]):
                similarity = 1.0 - distance
                similarity = (similarity + 1) / 2
                retrieved_results.append((doc, float(similarity)))

        return retrieved_results

    def add_images(self, new_image_paths):
        if not isinstance(new_image_paths, list):
            new_image_paths = [new_image_paths]
        self.database_paths.extend(new_image_paths)
        dinov2_embeddings = []
        dinov2_documents = []
        dinov2_ids = []
        dinov2_metadatas = []

        clip_embeddings = []
        clip_documents = []
        clip_ids = []
        clip_metadatas = []

        start_idx = self.dinov2_collection.count()

        for idx, img_path in enumerate(tqdm(new_image_paths, desc="Adding new images")):
            try:
                img = self.preprocessor.preprocess(img_path)

                with torch.no_grad():
                    dinov2_feature = torch.tensor(self.Dinov2(img)).to(self.device)
                    dinov2_feature = dinov2_feature / torch.norm(dinov2_feature)

                dinov2_embeddings.append(dinov2_feature.cpu().numpy().tolist())
                dinov2_documents.append(str(img_path))
                dinov2_ids.append(f"dinov2_{start_idx + idx}")
                dinov2_metadatas.append({"image_path": str(img_path), "type": "dinov2"})

                with torch.no_grad():
                    clip_feature = torch.tensor(
                        self.clip_loader.extract_image_features(img)
                    ).to(self.device)
                    clip_feature = clip_feature / torch.norm(clip_feature)

                clip_embeddings.append(clip_feature.cpu().numpy().tolist())
                clip_documents.append(str(img_path))
                clip_ids.append(f"clip_{start_idx + idx}")
                clip_metadatas.append({"image_path": str(img_path), "type": "clip"})

            except Exception as e:
                logger.error(f"Error processing new image {img_path}: {str(e)}")

        if dinov2_embeddings:
            self.dinov2_collection.add(
                embeddings=dinov2_embeddings,
                documents=dinov2_documents,
                ids=dinov2_ids,
                metadatas=dinov2_metadatas,
            )

        if clip_embeddings:
            self.clip_collection.add(
                embeddings=clip_embeddings,
                documents=clip_documents,
                ids=clip_ids,
                metadatas=clip_metadatas,
            )

        logger.info(f"Successfully added {len(new_image_paths)} new images to database")

    def delete_image(self, image_path):
        image_path_str = str(image_path)

        try:
            dinov2_results = self.dinov2_collection.get(
                where={"image_path": image_path_str}
            )
            if dinov2_results["ids"]:
                self.dinov2_collection.delete(ids=dinov2_results["ids"])
                logger.info(
                    f"Deleted {len(dinov2_results['ids'])} DINOv2 features for {image_path}"
                )
            clip_results = self.clip_collection.get(
                where={"image_path": image_path_str}
            )
            if clip_results["ids"]:
                self.clip_collection.delete(ids=clip_results["ids"])
                logger.info(
                    f"Deleted {len(clip_results['ids'])} CLIP features for {image_path}"
                )
            if Path(image_path) in self.database_paths:
                self.database_paths.remove(Path(image_path))

            logger.info(f"Successfully deleted image {image_path} from database")

        except Exception as e:
            logger.error(f"Error deleting image {image_path}: {str(e)}")
            raise

    def get_database_stats(self):
        return {
            "total_images": len(self.database_paths),
            "dinov2_features": self.dinov2_collection.count(),
            "clip_features": self.clip_collection.count(),
            "database_folder": str(self.database_folder),
            "chroma_db_path": str(self.chroma_db_path),
        }

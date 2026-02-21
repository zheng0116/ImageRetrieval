from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chromadb
import torch
from chromadb.config import Settings
from tqdm import tqdm

from ..config.logger import set_logger

MAX_WORKERS = min(4, os.cpu_count() or 1)
SUPPORTED_EMBED_MODELS = {"dinov2", "clip", "siglip2_base", "mobileclip2_s0"}

logger = set_logger("retrieval", "INFO")


class RetrievalProcessor:
    def __init__(
        self,
        dinov2_loader: Any,
        multimodal_loader: Any,
        preprocessor: Any,
        database_folder: str,
        embed_model: str,
        multimodal_model: str,
        top_k: int,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dinov2 = dinov2_loader
        self.multimodal_loader = multimodal_loader
        self.preprocessor = preprocessor
        self.database_folder = Path(database_folder)
        self.embed_model = embed_model.strip().lower()
        self.multimodal_model = multimodal_model.strip().lower()
        self.top_k = top_k

        if self.embed_model not in SUPPORTED_EMBED_MODELS:
            raise ValueError(
                f"Unsupported embed_model '{self.embed_model}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EMBED_MODELS))}"
            )

        self.chroma_db_path = self.database_folder / "chroma_db"
        self.chroma_db_path.mkdir(exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_db_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.dinov2_collection = self.chroma_client.get_or_create_collection(
            name="dinov2_features",
            metadata={"hnsw:space": "cosine"},
        )
        self.multimodal_collection_name = f"{self.multimodal_model}_features"
        self.multimodal_collection = self.chroma_client.get_or_create_collection(
            name=self.multimodal_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.database_paths = self.get_image_files()
        self.check_features()

    def get_image_files(self) -> list[Path]:
        supported_formats = ("*.jpg", "*.jpeg", "*.png")
        return [
            path
            for pattern in supported_formats
            for path in self.database_folder.glob(pattern)
        ]

    def check_features(self) -> None:
        self.database_paths = self.get_image_files()
        expected_count = len(self.database_paths)
        dinov2_count = self.dinov2_collection.count()
        multimodal_count = self.multimodal_collection.count()

        if dinov2_count != expected_count:
            logger.info("Computing and storing DINOv2 features...")
            self.store_dinov2_features()
        else:
            logger.info("DINOv2 features already exist in database")

        if multimodal_count != expected_count:
            logger.info(f"Computing and storing {self.multimodal_model} features...")
            self.store_multimodal_features()
        else:
            logger.info(f"{self.multimodal_model} features already exist in database")

    def _extract_dinov2_feature(self, img_path: Path) -> tuple[list[float] | None, str]:
        try:
            image = self.preprocessor.preprocess(img_path)
            with torch.no_grad():
                feature = torch.tensor(self.dinov2(image), device=self.device)
                feature = feature / torch.norm(feature)
            return feature.cpu().numpy().tolist(), str(img_path)
        except Exception as exc:
            logger.error(f"Error processing image {img_path}: {exc}")
            return None, str(img_path)

    def store_dinov2_features(self) -> None:
        if not self.database_paths:
            raise ValueError(f"No images found in database folder: {self.database_folder}")

        try:
            self.chroma_client.delete_collection("dinov2_features")
        except Exception as exc:
            logger.info(f"Collection dinov2_features not deleted: {exc}")

        self.dinov2_collection = self.chroma_client.get_or_create_collection(
            name="dinov2_features",
            metadata={"hnsw:space": "cosine"},
        )

        embeddings: list[list[float]] = []
        documents: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, str]] = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._extract_dinov2_feature, img_path): idx
                for idx, img_path in enumerate(self.database_paths)
            }
            for future in tqdm(
                as_completed(futures),
                total=len(self.database_paths),
                desc="Computing DINOv2 features",
            ):
                idx = futures[future]
                embedding, path = future.result()
                if embedding is not None:
                    embeddings.append(embedding)
                    documents.append(path)
                    ids.append(f"dinov2_{idx}")
                    metadatas.append({"image_path": path, "type": "dinov2"})

        if embeddings:
            self.dinov2_collection.add(
                embeddings=embeddings,
                documents=documents,
                ids=ids,
                metadatas=metadatas,
            )
            logger.info(f"Stored {len(embeddings)} DINOv2 features")

    def _extract_multimodal_feature(
        self, img_path: Path
    ) -> tuple[list[float] | None, str]:
        try:
            image = self.preprocessor.preprocess(img_path)
            with torch.no_grad():
                feature = torch.tensor(
                    self.multimodal_loader.extract_image_features(image),
                    device=self.device,
                )
                feature = feature / torch.norm(feature)
            return feature.cpu().numpy().tolist(), str(img_path)
        except Exception as exc:
            logger.error(f"Error processing image {img_path}: {exc}")
            return None, str(img_path)

    def store_multimodal_features(self) -> None:
        try:
            self.chroma_client.delete_collection(self.multimodal_collection_name)
        except Exception as exc:
            logger.info(f"Collection {self.multimodal_collection_name} not deleted: {exc}")

        self.multimodal_collection = self.chroma_client.get_or_create_collection(
            name=self.multimodal_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        embeddings: list[list[float]] = []
        documents: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, str]] = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._extract_multimodal_feature, img_path): idx
                for idx, img_path in enumerate(self.database_paths)
            }
            for future in tqdm(
                as_completed(futures),
                total=len(self.database_paths),
                desc=f"Computing {self.multimodal_model} features",
            ):
                idx = futures[future]
                embedding, path = future.result()
                if embedding is not None:
                    embeddings.append(embedding)
                    documents.append(path)
                    ids.append(f"{self.multimodal_model}_{idx}")
                    metadatas.append({"image_path": path, "type": self.multimodal_model})

        if embeddings:
            self.multimodal_collection.add(
                embeddings=embeddings,
                documents=documents,
                ids=ids,
                metadatas=metadatas,
            )
            logger.info(f"Stored {len(embeddings)} {self.multimodal_model} features")

    @staticmethod
    def _build_results(
        results: dict[str, list[list[Any]]], scale_to_unit: bool
    ) -> list[tuple[str, float]]:
        retrieved: list[tuple[str, float]] = []
        if results["documents"] and results["distances"]:
            for doc, distance in zip(results["documents"][0], results["distances"][0]):
                similarity = 1.0 - distance
                if scale_to_unit:
                    similarity = (similarity + 1) / 2
                retrieved.append((doc, float(similarity)))
        return retrieved

    def image_to_image_retrieve(self, query_image: Path) -> list[tuple[str, float]]:
        query_image = self.preprocessor.preprocess(query_image)

        if self.embed_model == "dinov2":
            query_feature = self.dinov2(query_image)
            logger.info(f"Query feature shape: {query_feature.shape} (dinov2)")
            results = self.dinov2_collection.query(
                query_embeddings=[query_feature.tolist()],
                n_results=self.top_k,
            )
        else:
            query_feature = self.multimodal_loader.extract_image_features(query_image)
            logger.info(
                f"Query feature shape: {query_feature.shape} ({self.multimodal_model})"
            )
            results = self.multimodal_collection.query(
                query_embeddings=[query_feature.tolist()],
                n_results=self.top_k,
            )

        return self._build_results(results, scale_to_unit=False)

    def text_to_image_retrieve(self, query_text: str) -> list[tuple[str, float]]:
        with torch.no_grad():
            query_feature = self.multimodal_loader.extract_text_features(query_text)
            results = self.multimodal_collection.query(
                query_embeddings=[query_feature.tolist()],
                n_results=self.top_k,
            )
        return self._build_results(results, scale_to_unit=True)

    def _extract_dinov2_single(
        self, img_path: Path, start_idx: int
    ) -> dict[str, Any] | None:
        try:
            image = self.preprocessor.preprocess(img_path)
            with torch.no_grad():
                feature = torch.tensor(self.dinov2(image), device=self.device)
                feature = feature / torch.norm(feature)
            return {
                "embedding": feature.cpu().numpy().tolist(),
                "path": str(img_path),
                "idx": start_idx,
            }
        except Exception as exc:
            logger.error(f"Error processing image {img_path}: {exc}")
            return None

    def _extract_multimodal_single(
        self, img_path: Path, start_idx: int
    ) -> dict[str, Any] | None:
        try:
            image = self.preprocessor.preprocess(img_path)
            with torch.no_grad():
                feature = torch.tensor(
                    self.multimodal_loader.extract_image_features(image),
                    device=self.device,
                )
                feature = feature / torch.norm(feature)
            return {
                "embedding": feature.cpu().numpy().tolist(),
                "path": str(img_path),
                "idx": start_idx,
            }
        except Exception as exc:
            logger.error(f"Error processing image {img_path}: {exc}")
            return None

    def add_images(self, new_image_paths: list[Path] | Path) -> None:
        if not isinstance(new_image_paths, list):
            new_image_paths = [new_image_paths]

        dinov2_embeddings: list[list[float]] = []
        dinov2_documents: list[str] = []
        dinov2_ids: list[str] = []
        dinov2_metadatas: list[dict[str, str]] = []

        multimodal_embeddings: list[list[float]] = []
        multimodal_documents: list[str] = []
        multimodal_ids: list[str] = []
        multimodal_metadatas: list[dict[str, str]] = []

        start_idx = self.dinov2_collection.count()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(self._extract_dinov2_single, img_path, start_idx + idx)
                for idx, img_path in enumerate(new_image_paths)
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(new_image_paths),
                desc="Extracting DINOv2 features",
            ):
                result = future.result()
                if result:
                    dinov2_embeddings.append(result["embedding"])
                    dinov2_documents.append(result["path"])
                    dinov2_ids.append(f"dinov2_{result['idx']}")
                    dinov2_metadatas.append({"image_path": result["path"], "type": "dinov2"})

        multimodal_start_idx = self.multimodal_collection.count()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(
                    self._extract_multimodal_single, img_path, multimodal_start_idx + idx
                )
                for idx, img_path in enumerate(new_image_paths)
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(new_image_paths),
                desc=f"Extracting {self.multimodal_model} features",
            ):
                result = future.result()
                if result:
                    multimodal_embeddings.append(result["embedding"])
                    multimodal_documents.append(result["path"])
                    multimodal_ids.append(f"{self.multimodal_model}_{result['idx']}")
                    multimodal_metadatas.append(
                        {"image_path": result["path"], "type": self.multimodal_model}
                    )

        self.database_paths.extend(new_image_paths)

        if dinov2_embeddings:
            self.dinov2_collection.add(
                embeddings=dinov2_embeddings,
                documents=dinov2_documents,
                ids=dinov2_ids,
                metadatas=dinov2_metadatas,
            )

        if multimodal_embeddings:
            self.multimodal_collection.add(
                embeddings=multimodal_embeddings,
                documents=multimodal_documents,
                ids=multimodal_ids,
                metadatas=multimodal_metadatas,
            )

        logger.info(f"Successfully added {len(new_image_paths)} new images to database")

    def delete_image(self, image_path: Path) -> None:
        image_path_str = str(image_path)
        try:
            dinov2_results = self.dinov2_collection.get(where={"image_path": image_path_str})
            if dinov2_results["ids"]:
                self.dinov2_collection.delete(ids=dinov2_results["ids"])
                logger.info(
                    f"Deleted {len(dinov2_results['ids'])} DINOv2 features for {image_path}"
                )

            multimodal_results = self.multimodal_collection.get(
                where={"image_path": image_path_str}
            )
            if multimodal_results["ids"]:
                self.multimodal_collection.delete(ids=multimodal_results["ids"])
                logger.info(
                    f"Deleted {len(multimodal_results['ids'])} {self.multimodal_model} features for {image_path}"
                )

            if image_path in self.database_paths:
                self.database_paths.remove(image_path)

            logger.info(f"Successfully deleted image {image_path} from database")
        except Exception as exc:
            logger.error(f"Error deleting image {image_path}: {exc}")
            raise

    def get_database_stats(self) -> dict[str, Any]:
        return {
            "total_images": len(self.database_paths),
            "dinov2_features": self.dinov2_collection.count(),
            f"{self.multimodal_model}_features": self.multimodal_collection.count(),
            "embed_model": self.embed_model,
            "multimodal_model": self.multimodal_model,
            "database_folder": str(self.database_folder),
            "chroma_db_path": str(self.chroma_db_path),
        }

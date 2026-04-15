import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.schemas import SimilarItem
from src.settings.settings import settings


class QdrantService:
    def __init__(self):
        try:
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                api_key=settings.qdrant_api_key,
                https=False,
            )
            self.collection_name = "predictions"
            self._init_collection()
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Qdrant client: {exc}") from exc

    def _init_collection(self):
        try:
            exists = self.client.collection_exists(self.collection_name)

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=784,
                        distance=Distance.COSINE,
                    ),
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize collection: {exc}") from exc

    def save_prediction(self, vector, prediction: dict):
        try:
            vector = vector.reshape(-1).astype(float)

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector.tolist(),
                        payload=prediction,
                    )
                ],
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to save prediction: {exc}") from exc

    def search_similar(self, vector, limit: int = 5):
        try:
            vector = vector.reshape(-1).astype(float)

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector.tolist(),
                limit=limit,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to search similar vectors: {exc}") from exc

        return [
            SimilarItem(
                id=str(point.id),
                score=point.score,
                payload=point.payload,
            )
            for point in results
        ]
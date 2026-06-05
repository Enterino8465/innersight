"""Qdrant-backed suspect discovery via embedding similarity (Phase 5).

Stores per-user behavioural embeddings in a Qdrant vector collection so that,
given one suspicious user, the nearest users in embedding space can be surfaced
as related suspects — optionally across CERT versions. All Qdrant access is
defensive: a missing server (or missing ``qdrant-client``) logs a warning and
degrades gracefully rather than crashing the app.
"""

from __future__ import annotations

import logging
import uuid

import numpy as np

logger = logging.getLogger(__name__)


class SuspectFinder:
    """Sync user embeddings to Qdrant and find similar users.

    Args:
        qdrant_url: Base URL of the Qdrant server.
        collection_name: Collection that stores the embeddings.
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "innersight_embeddings",
    ) -> None:
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self._client = None

    def _get_client(self):
        """Lazily build (and cache) the Qdrant client; imports the optional dep."""
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self.qdrant_url, timeout=5.0)
        return self._client

    @staticmethod
    def _point_id(version: str, user_id: str) -> str:
        """Deterministic point id = UUID5 hash of (version, user_id)."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{version}:{user_id}"))

    # ── Server health ─────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Return True if Qdrant is reachable, False otherwise (never raises)."""
        try:
            self._get_client().get_collections()
            return True
        except Exception as exc:  # connection refused, missing dep, timeout, …
            logger.warning("SuspectFinder | Qdrant not reachable at %s: %s", self.qdrant_url, exc)
            return False

    # ── Collection / upsert ─────────────────────────────────────────────────────

    def ensure_collection(self, vector_dim: int = 128) -> None:
        """Create the collection (Cosine) and payload indexes if it is missing."""
        try:
            from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

            client = self._get_client()
            existing = {c.name for c in client.get_collections().collections}
            if self.collection_name in existing:
                return
            client.create_collection(
                self.collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )
            for field, schema in (
                ("user_id", PayloadSchemaType.KEYWORD),
                ("version", PayloadSchemaType.KEYWORD),
                ("department", PayloadSchemaType.KEYWORD),
                ("scenario", PayloadSchemaType.INTEGER),
                ("score", PayloadSchemaType.FLOAT),
            ):
                client.create_payload_index(self.collection_name, field_name=field, field_schema=schema)
            logger.info("SuspectFinder | created collection %r (dim=%d, cosine)",
                        self.collection_name, vector_dim)
        except Exception as exc:
            logger.warning("SuspectFinder | ensure_collection failed: %s", exc)

    def sync_embeddings(self, embeddings, user_ids, metadata_list, version: str) -> int:
        """Upsert per-user embeddings with payloads. Returns the number upserted.

        Args:
            embeddings: ``(N, dim)`` array of user embeddings.
            user_ids: Length-``N`` user ids aligned with ``embeddings``.
            metadata_list: Length-``N`` dicts with optional ``score`` / ``department``
                / ``scenario``.
            version: CERT version tag stored on every point.
        """
        try:
            from qdrant_client.models import PointStruct

            emb = np.asarray(embeddings, dtype=float)
            if emb.ndim != 2 or emb.shape[0] == 0:
                logger.warning("SuspectFinder | nothing to sync (embeddings shape %s).", emb.shape)
                return 0
            self.ensure_collection(emb.shape[1])
            client = self._get_client()
            points = []
            for i, uid in enumerate(user_ids):
                md = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
                payload = {
                    "user_id": str(uid),
                    "version": str(version),
                    "score": float(md.get("score", 0.0)),
                    "department": str(md.get("department", "")),
                    "scenario": int(md.get("scenario", 0)),
                }
                points.append(PointStruct(id=self._point_id(version, uid),
                                          vector=emb[i].tolist(), payload=payload))
            client.upsert(self.collection_name, points=points)
            logger.info("SuspectFinder | upserted %d embeddings for version %s.", len(points), version)
            return len(points)
        except Exception as exc:
            logger.warning("SuspectFinder | sync_embeddings failed: %s", exc)
            return 0

    # ── Similarity query ────────────────────────────────────────────────────────

    def find_similar(self, user_id: str, k: int = 10, version: str | None = None) -> list[dict]:
        """Return the ``k`` users most similar to ``user_id`` (excluding itself).

        Args:
            user_id: The user to query around.
            k: Number of neighbours to return.
            version: Restrict to this version; ``None`` searches across all
                versions (cross-version discovery).

        Returns:
            List of ``{user_id, score, similarity, department, version}`` dicts,
            or ``[]`` on any error / if the user has no stored embedding.
        """
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            client = self._get_client()

            # Resolve the query vector and the query point's id (to exclude self).
            if version is not None:
                query_id = self._point_id(version, user_id)
                records = client.retrieve(self.collection_name, ids=[query_id], with_vectors=True)
            else:
                query_id = None
                records, _ = client.scroll(
                    self.collection_name,
                    scroll_filter=Filter(must=[FieldCondition(
                        key="user_id", match=MatchValue(value=str(user_id)))]),
                    limit=1, with_vectors=True,
                )
            if not records:
                logger.info("SuspectFinder | no embedding for user %s (version=%s).", user_id, version)
                return []
            query_vector = records[0].vector
            if query_id is None:
                query_id = records[0].id

            query_filter = None
            if version is not None:
                query_filter = Filter(must=[FieldCondition(
                    key="version", match=MatchValue(value=str(version)))])

            response = client.query_points(
                self.collection_name, query=query_vector, limit=k + 1,
                query_filter=query_filter, with_payload=True,
            )
            results = []
            for hit in response.points:
                if hit.id == query_id:
                    continue  # skip the query user itself
                payload = hit.payload or {}
                results.append({
                    "user_id": payload.get("user_id"),
                    "score": payload.get("score"),
                    "similarity": hit.score,
                    "department": payload.get("department"),
                    "version": payload.get("version"),
                })
                if len(results) >= k:
                    break
            return results
        except Exception as exc:
            logger.warning("SuspectFinder | find_similar failed: %s", exc)
            return []

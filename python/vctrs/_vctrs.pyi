"""Type stubs for vctrs — a fast vector database powered by HNSW.

Provides IDE autocompletion, type checking, and inline documentation
for the native Rust-backed ``Database`` and ``SearchResult`` classes.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt


class SearchResult:
    """A single search result returned by :meth:`Database.search`.

    Attributes:
        id: The unique string identifier of the matched vector.
        distance: The distance between the query and this vector.
            Lower is more similar for cosine and euclidean metrics.
        metadata: The metadata dict attached to this vector, or ``None``.

    Supports indexing (``result[0]`` → id, ``result[1]`` → distance,
    ``result[2]`` → metadata) for tuple-style destructuring.
    """

    id: str
    distance: float
    metadata: Optional[dict[str, Any]]

    def __repr__(self) -> str: ...
    def __getitem__(self, idx: int) -> Any: ...


class Database:
    """A persistent vector database backed by an HNSW index.

    Stores vectors with string IDs and optional JSON metadata.
    Supports cosine, euclidean (L2), and dot-product distance metrics.
    Uses memory-mapped I/O for fast loading and a write-ahead log for
    crash safety.

    Args:
        path: Directory path for the database files. Created if it
            doesn't exist.
        dim: Vector dimensionality (e.g. 384 for MiniLM). Required when
            creating a new database, omit when opening an existing one.
        metric: Distance metric — ``"cosine"`` (default), ``"euclidean"``
            / ``"l2"``, or ``"dot"`` / ``"dot_product"``.
        m: HNSW ``M`` parameter — max edges per node (default 16).
            Higher values improve recall at the cost of memory.
        ef_construction: HNSW build-time search width (default 200).
            Higher values improve index quality at the cost of build time.
        quantize: If ``True``, enable scalar quantization (SQ8) for
            faster search with slightly lower precision.

    Raises:
        ValueError: If opening an existing database without ``dim`` and
            the database doesn't exist, or if ``metric`` is invalid.

    Example::

        with Database("my_db", dim=384) as db:
            db.add("doc1", embedding, {"title": "Hello"})
            results = db.search(query_vector, k=5)
            for r in results:
                print(r.id, r.distance)

    Example (opening existing)::

        db = Database("my_db")  # auto-detects dim and metric
    """

    def __init__(
        self,
        path: str,
        dim: Optional[int] = None,
        metric: Optional[str] = None,
        m: Optional[int] = None,
        ef_construction: Optional[int] = None,
        quantize: bool = False,
    ) -> None: ...

    def add(
        self,
        id: str,
        vector: Union[list[float], npt.NDArray[np.float32]],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add a vector with a unique string ID.

        Args:
            id: Unique identifier. Raises if the ID already exists
                (use :meth:`upsert` to insert-or-update).
            vector: The embedding vector. Must match the database's
                dimensionality. Accepts a Python list or a 1-D numpy
                array of float32.
            metadata: Optional JSON-serializable dict of metadata
                (e.g. ``{"category": "science", "year": 2024}``).

        Raises:
            ValueError: If the ID already exists or the vector
                dimension doesn't match.
        """
        ...

    def upsert(
        self,
        id: str,
        vector: Union[list[float], npt.NDArray[np.float32]],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Insert a vector, or update it if the ID already exists.

        Same as :meth:`add` but overwrites existing entries instead
        of raising an error.

        Args:
            id: Unique identifier.
            vector: The embedding vector.
            metadata: Optional metadata dict.

        Raises:
            ValueError: If the vector dimension doesn't match.
        """
        ...

    def add_many(
        self,
        ids: list[str],
        vectors: Union[list[list[float]], npt.NDArray[np.float32]],
        metadatas: Optional[list[Optional[dict[str, Any]]]] = None,
    ) -> None:
        """Batch insert multiple vectors.

        All three lists must have the same length. Vectors can be
        passed as a list of lists or a 2-D numpy array of shape
        ``(n, dim)``.

        Args:
            ids: List of unique identifiers.
            vectors: Batch of embedding vectors.
            metadatas: Optional list of metadata dicts (or ``None``
                per entry).

        Raises:
            ValueError: If any ID already exists, lengths mismatch,
                or dimensions don't match.
        """
        ...

    def upsert_many(
        self,
        ids: list[str],
        vectors: Union[list[list[float]], npt.NDArray[np.float32]],
        metadatas: Optional[list[Optional[dict[str, Any]]]] = None,
    ) -> None:
        """Batch upsert — inserts new vectors, updates existing ones.

        More efficient than calling :meth:`upsert` in a loop because
        new vectors are batch-inserted together.

        Args:
            ids: List of identifiers.
            vectors: Batch of embedding vectors.
            metadatas: Optional list of metadata dicts.

        Raises:
            ValueError: If lengths mismatch or dimensions don't match.
        """
        ...

    def search(
        self,
        vector: Union[list[float], npt.NDArray[np.float32]],
        k: int = 10,
        ef_search: Optional[int] = None,
        where_filter: Optional[dict[str, Any]] = None,
        max_distance: Optional[float] = None,
    ) -> list[SearchResult]:
        """Find the k nearest neighbors to a query vector.

        Args:
            vector: The query embedding.
            k: Number of results to return (default 10).
            ef_search: HNSW search-time width. Higher values improve
                recall at the cost of latency. Defaults to ``max(k, 10)``.
            where_filter: Optional metadata filter. Supports:

                - Equality: ``{"field": "value"}``
                - Not-equal: ``{"field": {"$ne": "value"}}``
                - In-set: ``{"field": {"$in": ["a", "b"]}}``
                - Numeric ranges: ``{"field": {"$gt": 10, "$lte": 20}}``
                - Compound (AND): ``{"f1": "v1", "f2": {"$gt": 5}}``

            max_distance: Optional distance threshold. Results with
                distance greater than this value are discarded. Useful
                for finding only "close enough" matches.

        Returns:
            List of :class:`SearchResult` objects sorted by distance
            (ascending).

        Raises:
            ValueError: If the vector dimension doesn't match.
        """
        ...

    def search_many(
        self,
        vectors: Union[list[list[float]], npt.NDArray[np.float32]],
        k: int = 10,
        ef_search: Optional[int] = None,
        where_filter: Optional[dict[str, Any]] = None,
        max_distance: Optional[float] = None,
    ) -> list[list[SearchResult]]:
        """Search multiple queries in parallel using Rayon.

        Significantly faster than calling :meth:`search` in a loop
        for multiple queries.

        Args:
            vectors: Batch of query embeddings (list of lists or 2-D
                numpy array).
            k: Number of results per query (default 10).
            ef_search: HNSW search-time width.
            where_filter: Optional metadata filter (same syntax as
                :meth:`search`). Applied to all queries.
            max_distance: Optional distance threshold applied to all
                queries.

        Returns:
            List of result lists, one per query.

        Raises:
            ValueError: If any vector dimension doesn't match.
        """
        ...

    def get(self, id: str) -> tuple[list[float], Optional[dict[str, Any]]]:
        """Retrieve a vector and its metadata by ID.

        Args:
            id: The vector's unique identifier.

        Returns:
            A tuple of ``(vector, metadata)`` where metadata may be
            ``None``.

        Raises:
            ValueError: If the ID is not found.
        """
        ...

    def delete(self, id: str) -> bool:
        """Delete a vector by ID.

        The slot is marked as deleted but not reclaimed until
        :meth:`compact` is called.

        Args:
            id: The vector's unique identifier.

        Returns:
            ``True`` if the vector was found and deleted, ``False``
            if the ID was not found.

        Raises:
            ValueError: On internal errors (e.g. WAL write failure).
        """
        ...

    def delete_many(self, ids: list[str]) -> int:
        """Delete multiple vectors by ID.

        More efficient than calling :meth:`delete` in a loop because
        locks are held once for the entire batch.

        Args:
            ids: List of vector IDs to delete.

        Returns:
            Number of vectors actually deleted (IDs not found are skipped).

        Raises:
            ValueError: On internal errors (e.g. WAL write failure).
        """
        ...

    def update(
        self,
        id: str,
        vector: Optional[Union[list[float], npt.NDArray[np.float32]]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update a vector's embedding and/or metadata in-place.

        At least one of ``vector`` or ``metadata`` must be provided.

        Args:
            id: The vector's unique identifier.
            vector: New embedding vector (or ``None`` to keep existing).
            metadata: New metadata dict (or ``None`` to keep existing).

        Raises:
            ValueError: If the ID is not found or the vector dimension
                doesn't match.
        """
        ...

    def count(self, where_filter: Optional[dict[str, Any]] = None) -> int:
        """Count vectors matching a filter, or all vectors if no filter.

        Uses the inverted metadata index for fast counting with
        equality and ``$in`` filters.

        Args:
            where_filter: Optional metadata filter (same syntax as
                :meth:`search`). If ``None``, returns total count.

        Returns:
            Number of matching vectors.
        """
        ...

    def __contains__(self, id: str) -> bool:
        """Check if a vector ID exists in the database.

        Example::

            if "doc1" in db:
                print("exists")
        """
        ...

    def ids(self) -> list[str]:
        """Return a list of all vector IDs in the database."""
        ...

    def save(self) -> None:
        """Persist the database to disk.

        Writes the HNSW graph, metadata, and vectors to the database
        directory and truncates the write-ahead log.

        Raises:
            ValueError: On I/O errors (e.g. disk full, permission denied).
        """
        ...

    def compact(self) -> None:
        """Rebuild the index with only live vectors.

        Reclaims slots from deleted vectors, reducing memory usage
        and on-disk size. Call after many deletions.

        Raises:
            ValueError: On internal errors.
        """
        ...

    def enable_quantized_search(self) -> None:
        """Enable SQ8 quantized search for faster HNSW traversal.

        Quantized vectors use 4x less memory for distance comparisons
        during graph traversal, with full-precision re-ranking of
        final candidates.
        """
        ...

    def disable_quantized_search(self) -> None:
        """Disable quantized search and use full-precision vectors."""
        ...

    def stats(self) -> dict[str, Any]:
        """Get graph-level statistics for diagnostics.

        Returns:
            Dict with keys: ``num_vectors``, ``num_deleted``,
            ``num_layers``, ``avg_degree_layer0``, ``max_degree_layer0``,
            ``min_degree_layer0``, ``memory_vectors_bytes``,
            ``memory_graph_bytes``, ``memory_quantized_bytes``,
            ``uses_brute_force``, ``uses_quantized_search``.
        """
        ...

    @property
    def quantized_search(self) -> bool:
        """Whether quantized search is currently enabled."""
        ...

    @property
    def deleted_count(self) -> int:
        """Number of deleted slots not yet reclaimed by :meth:`compact`."""
        ...

    @property
    def total_slots(self) -> int:
        """Total allocated slots (active + deleted)."""
        ...

    @property
    def dim(self) -> int:
        """The vector dimensionality of this database."""
        ...

    @property
    def metric(self) -> str:
        """The distance metric: ``"cosine"``, ``"euclidean"``, or ``"dot_product"``."""
        ...

    def __len__(self) -> int:
        """Number of active vectors in the database."""
        ...

    def __enter__(self) -> Database:
        """Context manager entry — returns self."""
        ...

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        """Context manager exit — auto-saves the database."""
        ...

    def export_json(self, path: str, pretty: bool = False) -> None:
        """Export all vectors and metadata to a JSON file.

        Args:
            path: File path to write the JSON export.
            pretty: If ``True``, pretty-print the JSON output.
        """
        ...

    def import_json(self, path: str) -> None:
        """Import vectors from a JSON file (upsert semantics).

        Updates existing IDs and inserts new ones.

        Args:
            path: File path to read the JSON export from.

        Raises:
            ValueError: If the JSON dimension doesn't match.
        """
        ...


class Client:
    """Multi-collection client for managing named vector databases.

    Each collection is stored in its own subdirectory under the root path.

    Args:
        path: Root directory for all collections.

    Example::

        client = Client("/data/vectors")
        movies = client.create_collection("movies", dim=384)
        docs = client.get_or_create_collection("docs", dim=768)
        print(client.list_collections())  # ["docs", "movies"]
    """

    def __init__(self, path: str) -> None: ...

    def create_collection(
        self,
        name: str,
        dim: int,
        metric: Optional[str] = None,
        m: Optional[int] = None,
        ef_construction: Optional[int] = None,
        quantize: bool = False,
    ) -> Database:
        """Create a new collection.

        Args:
            name: Collection name (alphanumeric, hyphens, underscores).
            dim: Vector dimensionality.
            metric: Distance metric (default ``"cosine"``).
            m: HNSW M parameter (default 16).
            ef_construction: HNSW build-time width (default 200).
            quantize: Enable SQ8 quantization.

        Raises:
            ValueError: If the collection already exists or the name
                is invalid.
        """
        ...

    def get_collection(self, name: str) -> Database:
        """Open an existing collection.

        Args:
            name: Collection name.

        Raises:
            ValueError: If the collection doesn't exist.
        """
        ...

    def get_or_create_collection(
        self,
        name: str,
        dim: int,
        metric: Optional[str] = None,
    ) -> Database:
        """Get or create a collection.

        If the collection exists, opens it (dim/metric are ignored).
        Otherwise creates a new one.

        Args:
            name: Collection name.
            dim: Vector dimensionality (used only for creation).
            metric: Distance metric (used only for creation).
        """
        ...

    def delete_collection(self, name: str) -> bool:
        """Delete a collection and all its data.

        Args:
            name: Collection name.

        Returns:
            ``True`` if the collection existed and was deleted.
        """
        ...

    def list_collections(self) -> list[str]:
        """List all collection names (sorted alphabetically)."""
        ...

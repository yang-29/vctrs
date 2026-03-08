"""Type stubs for the vctrs package.

Exports:
    Database: Synchronous vector database.
    AsyncDatabase: Async wrapper for use with asyncio.
    SearchResult: A single search result with id, distance, metadata.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

from vctrs._vctrs import Database as Database
from vctrs._vctrs import SearchResult as SearchResult

__version__: str
__all__: list[str]


class AsyncDatabase:
    """Async wrapper around :class:`Database`.

    All mutation and search methods are non-blocking — they run the
    underlying Rust operations in a thread pool via
    ``asyncio.to_thread()``.

    Properties (``dim``, ``metric``, etc.) are synchronous since they
    are instant lookups with no I/O.

    Usage::

        async with AsyncDatabase("my_db", dim=384) as db:
            await db.add("doc1", embedding, {"title": "Hello"})
            results = await db.search(query_vector, k=5)

    Or with the async factory method::

        db = await AsyncDatabase.open("my_db", dim=384)
    """

    _db: Database

    def __init__(
        self,
        path: str,
        dim: Optional[int] = None,
        metric: Optional[str] = None,
        m: Optional[int] = None,
        ef_construction: Optional[int] = None,
        quantize: bool = False,
    ) -> None: ...

    @classmethod
    async def open(
        cls,
        path: str,
        dim: Optional[int] = None,
        metric: Optional[str] = None,
        m: Optional[int] = None,
        ef_construction: Optional[int] = None,
        quantize: bool = False,
    ) -> AsyncDatabase:
        """Async factory method to open or create a database.

        Equivalent to ``AsyncDatabase(...)`` but runs the constructor
        in a thread pool so it doesn't block the event loop if the
        database needs to replay a WAL on startup.
        """
        ...

    async def add(
        self,
        id: str,
        vector: Union[list[float], npt.NDArray[np.float32]],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add a vector with a unique string ID.

        Raises:
            ValueError: If the ID already exists or dimension mismatch.
        """
        ...

    async def upsert(
        self,
        id: str,
        vector: Union[list[float], npt.NDArray[np.float32]],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Insert a vector, or update it if the ID already exists."""
        ...

    async def add_many(
        self,
        ids: list[str],
        vectors: Union[list[list[float]], npt.NDArray[np.float32]],
        metadatas: Optional[list[Optional[dict[str, Any]]]] = None,
    ) -> None:
        """Batch insert multiple vectors."""
        ...

    async def upsert_many(
        self,
        ids: list[str],
        vectors: Union[list[list[float]], npt.NDArray[np.float32]],
        metadatas: Optional[list[Optional[dict[str, Any]]]] = None,
    ) -> None:
        """Batch upsert — inserts new vectors, updates existing ones."""
        ...

    async def search(
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
            ef_search: HNSW search-time width.
            where_filter: Optional metadata filter.
            max_distance: Optional distance threshold.

        Returns:
            List of SearchResult objects sorted by distance.
        """
        ...

    async def search_many(
        self,
        vectors: Union[list[list[float]], npt.NDArray[np.float32]],
        k: int = 10,
        ef_search: Optional[int] = None,
        where_filter: Optional[dict[str, Any]] = None,
        max_distance: Optional[float] = None,
    ) -> list[list[SearchResult]]:
        """Search multiple queries in parallel.

        Returns:
            List of result lists, one per query.
        """
        ...

    async def get(self, id: str) -> tuple[list[float], Optional[dict[str, Any]]]:
        """Retrieve a vector and its metadata by ID.

        Raises:
            ValueError: If the ID is not found.
        """
        ...

    async def delete(self, id: str) -> bool:
        """Delete a vector by ID.

        Returns:
            True if found and deleted, False if not found.
        """
        ...

    async def delete_many(self, ids: list[str]) -> int:
        """Delete multiple vectors by ID.

        Returns:
            Number of vectors actually deleted.
        """
        ...

    async def update(
        self,
        id: str,
        vector: Optional[Union[list[float], npt.NDArray[np.float32]]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update a vector's embedding and/or metadata.

        Raises:
            ValueError: If the ID is not found.
        """
        ...

    async def save(self) -> None:
        """Persist the database to disk."""
        ...

    async def compact(self) -> None:
        """Rebuild the index with only live vectors."""
        ...

    def count(self, where_filter: Optional[dict[str, Any]] = None) -> int:
        """Count vectors matching a filter, or all vectors if no filter."""
        ...

    def __contains__(self, id: str) -> bool: ...
    def __len__(self) -> int: ...

    @property
    def dim(self) -> int:
        """The vector dimensionality of this database."""
        ...

    @property
    def metric(self) -> str:
        """The distance metric."""
        ...

    @property
    def quantized_search(self) -> bool:
        """Whether quantized search is currently enabled."""
        ...

    @property
    def deleted_count(self) -> int:
        """Number of deleted slots not yet reclaimed."""
        ...

    @property
    def total_slots(self) -> int:
        """Total allocated slots (active + deleted)."""
        ...

    def ids(self) -> list[str]:
        """Return a list of all vector IDs."""
        ...

    def stats(self) -> dict[str, Any]:
        """Get graph-level statistics for diagnostics."""
        ...

    def enable_quantized_search(self) -> None:
        """Enable SQ8 quantized search."""
        ...

    def disable_quantized_search(self) -> None:
        """Disable quantized search."""
        ...

    async def __aenter__(self) -> AsyncDatabase: ...
    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool: ...

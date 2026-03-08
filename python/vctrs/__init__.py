from vctrs._vctrs import Database, SearchResult

import asyncio
from functools import partial


class AsyncDatabase:
    """Async wrapper around Database. All methods are non-blocking and run the
    underlying Rust operations in a thread pool via asyncio.to_thread().

    Usage::

        async with AsyncDatabase("my_db", dim=384) as db:
            await db.add("id1", vector, {"key": "value"})
            results = await db.search(query_vector, k=10)
    """

    def __init__(self, path, dim=None, metric=None, m=None, ef_construction=None, quantize=False):
        self._db = Database(path, dim=dim, metric=metric, m=m,
                            ef_construction=ef_construction, quantize=quantize)

    @classmethod
    async def open(cls, path, dim=None, metric=None, m=None, ef_construction=None, quantize=False):
        """Async factory method to open or create a database."""
        return await asyncio.to_thread(
            cls, path, dim=dim, metric=metric, m=m,
            ef_construction=ef_construction, quantize=quantize
        )

    async def add(self, id, vector, metadata=None):
        return await asyncio.to_thread(self._db.add, id, vector, metadata)

    async def upsert(self, id, vector, metadata=None):
        return await asyncio.to_thread(self._db.upsert, id, vector, metadata)

    async def add_many(self, ids, vectors, metadatas=None):
        return await asyncio.to_thread(self._db.add_many, ids, vectors, metadatas)

    async def search(self, vector, k=10, ef_search=None, where_filter=None):
        return await asyncio.to_thread(
            partial(self._db.search, vector, k=k, ef_search=ef_search, where_filter=where_filter)
        )

    async def search_many(self, vectors, k=10, ef_search=None):
        return await asyncio.to_thread(
            partial(self._db.search_many, vectors, k=k, ef_search=ef_search)
        )

    async def get(self, id):
        return await asyncio.to_thread(self._db.get, id)

    async def delete(self, id):
        return await asyncio.to_thread(self._db.delete, id)

    async def update(self, id, vector=None, metadata=None):
        return await asyncio.to_thread(
            partial(self._db.update, id, vector=vector, metadata=metadata)
        )

    async def save(self):
        return await asyncio.to_thread(self._db.save)

    async def compact(self):
        return await asyncio.to_thread(self._db.compact)

    # Sync properties (fast, no I/O).
    def __contains__(self, id):
        return id in self._db

    def __len__(self):
        return len(self._db)

    @property
    def dim(self):
        return self._db.dim

    @property
    def metric(self):
        return self._db.metric

    @property
    def quantized_search(self):
        return self._db.quantized_search

    @property
    def deleted_count(self):
        return self._db.deleted_count

    @property
    def total_slots(self):
        return self._db.total_slots

    def ids(self):
        return self._db.ids()

    def stats(self):
        return self._db.stats()

    def enable_quantized_search(self):
        self._db.enable_quantized_search()

    def disable_quantized_search(self):
        self._db.disable_quantized_search()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.save()
        return False


__all__ = ["Database", "AsyncDatabase", "SearchResult"]
__version__ = "0.2.0"

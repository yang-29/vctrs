"""Tests for vctrs."""
import tempfile
import os
import numpy as np
from vctrs import Database


def test_add_and_search():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "testdb"), dim=3, metric="cosine")

        db.add("doc1", [1.0, 0.0, 0.0], {"title": "first"})
        db.add("doc2", [0.0, 1.0, 0.0], {"title": "second"})
        db.add("doc3", [0.0, 0.0, 1.0])

        results = db.search([0.9, 0.1, 0.0], k=2)
        assert len(results) == 2
        assert results[0].id == "doc1"
        assert results[0].distance < 0.1
        assert results[0].metadata["title"] == "first"
        # Tuple-style access still works.
        assert results[0][0] == "doc1"
        print(f"  search: {results[0]}")


def test_auto_detect():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "testdb")

        db = Database(path, dim=3, metric="euclidean")
        db.add("a", [1.0, 2.0, 3.0])
        db.save()
        del db

        # Reopen without dim/metric.
        db2 = Database(path)
        assert db2.dim == 3
        assert db2.metric == "euclidean"
        assert len(db2) == 1
        print(f"  auto-detect: dim={db2.dim}, metric={db2.metric}")


def test_context_manager():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "testdb")

        with Database(path, dim=2) as db:
            db.add("x", [1.0, 0.0])
        # auto-saved on exit

        db2 = Database(path)
        assert len(db2) == 1
        print(f"  context manager: auto-saved")


def test_upsert():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "testdb"), dim=2)

        db.upsert("a", [1.0, 0.0], {"v": 1})
        assert len(db) == 1

        db.upsert("a", [0.0, 1.0], {"v": 2})
        assert len(db) == 1  # still 1

        vec, meta = db.get("a")
        assert vec == [0.0, 1.0]
        assert meta["v"] == 2
        print(f"  upsert: updated in-place")


def test_filtered_search():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "testdb"), dim=2, metric="euclidean")

        db.add("a", [1.0, 0.0], {"category": "science"})
        db.add("b", [0.9, 0.1], {"category": "art"})
        db.add("c", [0.8, 0.2], {"category": "science"})

        # No filter.
        results = db.search([1.0, 0.0], k=1)
        assert results[0].id == "a"

        # Filter: category == science.
        results = db.search([1.0, 0.0], k=2, where_filter={"category": "science"})
        assert len(results) == 2
        assert all(r.metadata["category"] == "science" for r in results)

        # Filter: category != science.
        results = db.search([1.0, 0.0], k=10, where_filter={"category": {"$ne": "science"}})
        assert len(results) == 1
        assert results[0].id == "b"

        # Filter: $in.
        results = db.search([1.0, 0.0], k=10, where_filter={"category": {"$in": ["art", "music"]}})
        assert len(results) == 1
        assert results[0].id == "b"
        print(f"  filtered search: all operators work")


def test_ids():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "testdb"), dim=2)
        db.add("x", [1.0, 0.0])
        db.add("y", [0.0, 1.0])
        ids = sorted(db.ids())
        assert ids == ["x", "y"]
        print(f"  ids: {ids}")


def test_persistence():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "testdb")

        db = Database(path, dim=2, metric="euclidean")
        db.add("a", [1.0, 2.0], {"n": 42})
        db.add("b", [3.0, 4.0])
        db.save()
        del db

        db2 = Database(path)
        assert len(db2) == 2
        vec, meta = db2.get("a")
        assert vec == [1.0, 2.0]
        assert meta["n"] == 42
        print(f"  persistence OK")


def test_numpy_input():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "testdb"), dim=4, metric="cosine")

        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        db.add("np1", vec, {"source": "numpy"})

        query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        results = db.search(query, k=1)
        assert results[0].id == "np1"
        print(f"  numpy OK")


def test_batch_insert():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "testdb"), dim=4, metric="euclidean")

        ids = ["a", "b", "c", "d"]
        vectors = np.random.rand(4, 4).astype(np.float32)
        metadatas = [{"i": 0}, {"i": 1}, None, {"i": 3}]
        db.add_many(ids, vectors, metadatas)

        assert len(db) == 4
        print(f"  batch insert OK")


def test_delete():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "testdb"), dim=3, metric="cosine")

        db.add("a", [1.0, 0.0, 0.0])
        db.add("b", [0.0, 1.0, 0.0])
        db.add("c", [0.0, 0.0, 1.0])

        assert db.delete("b")
        assert len(db) == 2
        assert "b" not in db
        print(f"  delete OK")


def test_update():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "testdb"), dim=2, metric="euclidean")

        db.add("a", [1.0, 0.0], {"v": 1})
        db.update("a", vector=[0.0, 1.0])
        vec, meta = db.get("a")
        assert vec == [0.0, 1.0]
        assert meta["v"] == 1
        print(f"  update OK")


def test_large_batch():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "testdb"), dim=128, metric="cosine")

        n = 10000
        ids = [f"vec_{i}" for i in range(n)]
        vectors = np.random.rand(n, 128).astype(np.float32)
        db.add_many(ids, vectors)

        results = db.search(vectors[42], k=1)
        assert results[0].id == "vec_42"
        print(f"  large batch OK ({n} vectors)")


if __name__ == "__main__":
    tests = [
        ("add_and_search", test_add_and_search),
        ("auto_detect", test_auto_detect),
        ("context_manager", test_context_manager),
        ("upsert", test_upsert),
        ("filtered_search", test_filtered_search),
        ("ids", test_ids),
        ("persistence", test_persistence),
        ("numpy_input", test_numpy_input),
        ("batch_insert", test_batch_insert),
        ("delete", test_delete),
        ("update", test_update),
        ("large_batch", test_large_batch),
    ]
    for name, fn in tests:
        print(f"test_{name}...")
        fn()
        print(f"  PASSED")
    print(f"\nAll {len(tests)} tests passed!")

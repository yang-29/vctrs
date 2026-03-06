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
        assert results[0][0] == "doc1"
        print(f"  search results: {results}")


def test_persistence():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "testdb")

        db = Database(path, dim=2, metric="euclidean")
        db.add("a", [1.0, 2.0], {"n": 42})
        db.add("b", [3.0, 4.0])
        db.save()
        del db

        db2 = Database(path, dim=2, metric="euclidean")
        assert len(db2) == 2
        vec, meta = db2.get("a")
        assert vec == [1.0, 2.0]
        assert meta["n"] == 42
        print(f"  persistence OK")


def test_numpy_input():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "testdb"), dim=4, metric="cosine")

        # Add with numpy array
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        db.add("np1", vec, {"source": "numpy"})

        # Search with numpy array
        query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        results = db.search(query, k=1)
        assert results[0][0] == "np1"
        print(f"  numpy single OK")


def test_batch_insert():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "testdb"), dim=4, metric="euclidean")

        # Batch insert with numpy 2D array
        ids = ["a", "b", "c", "d"]
        vectors = np.random.rand(4, 4).astype(np.float32)
        metadatas = [{"i": 0}, {"i": 1}, None, {"i": 3}]
        db.add_many(ids, vectors, metadatas)

        assert len(db) == 4
        vec, meta = db.get("a")
        assert np.allclose(vec, vectors[0])
        assert meta["i"] == 0
        print(f"  batch insert OK ({len(db)} vectors)")


def test_batch_insert_lists():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "testdb"), dim=2, metric="euclidean")

        ids = ["x", "y"]
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        db.add_many(ids, vectors)
        assert len(db) == 2
        print(f"  batch insert with lists OK")


def test_delete():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "testdb"), dim=3, metric="cosine")

        db.add("a", [1.0, 0.0, 0.0])
        db.add("b", [0.0, 1.0, 0.0])
        db.add("c", [0.0, 0.0, 1.0])

        assert len(db) == 3
        assert db.delete("b")
        assert len(db) == 2
        assert "b" not in db

        results = db.search([0.0, 1.0, 0.0], k=3)
        assert all(r[0] != "b" for r in results)
        print(f"  delete OK, remaining: {[r[0] for r in results]}")


def test_update():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "testdb"), dim=2, metric="euclidean")

        db.add("a", [1.0, 0.0], {"v": 1})

        # Update vector
        db.update("a", vector=[0.0, 1.0])
        vec, meta = db.get("a")
        assert vec == [0.0, 1.0]
        assert meta["v"] == 1  # metadata unchanged

        # Update metadata
        db.update("a", metadata={"v": 2, "new": True})
        _, meta = db.get("a")
        assert meta["v"] == 2
        assert meta["new"] is True

        # Update with numpy
        db.update("a", vector=np.array([0.5, 0.5], dtype=np.float32))
        vec, _ = db.get("a")
        assert abs(vec[0] - 0.5) < 1e-6
        print(f"  update OK")


def test_contains():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "testdb"), dim=2)
        db.add("x", [1.0, 0.0])
        assert "x" in db
        assert "y" not in db
        print(f"  contains OK")


def test_large_batch():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "testdb"), dim=128, metric="cosine")

        n = 10000
        ids = [f"vec_{i}" for i in range(n)]
        vectors = np.random.rand(n, 128).astype(np.float32)
        db.add_many(ids, vectors)

        assert len(db) == n

        # Search should find similar vectors
        results = db.search(vectors[42], k=1)
        assert results[0][0] == "vec_42"
        print(f"  large batch OK ({n} vectors)")


if __name__ == "__main__":
    tests = [
        ("add_and_search", test_add_and_search),
        ("persistence", test_persistence),
        ("numpy_input", test_numpy_input),
        ("batch_insert", test_batch_insert),
        ("batch_insert_lists", test_batch_insert_lists),
        ("delete", test_delete),
        ("update", test_update),
        ("contains", test_contains),
        ("large_batch", test_large_batch),
    ]
    for name, fn in tests:
        print(f"test_{name}...")
        fn()
        print(f"  PASSED")
    print(f"\nAll {len(tests)} tests passed!")

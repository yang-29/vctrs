"""Basic smoke tests for vctrs."""
import tempfile
import os
from vctrs import Database


def test_add_and_search():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "testdb"), dim=3, metric="cosine")

        db.add("doc1", [1.0, 0.0, 0.0], {"title": "first"})
        db.add("doc2", [0.0, 1.0, 0.0], {"title": "second"})
        db.add("doc3", [0.0, 0.0, 1.0])

        results = db.search([0.9, 0.1, 0.0], k=2)
        assert len(results) == 2
        assert results[0][0] == "doc1"  # closest
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
        print(f"  persistence OK: vec={vec}, meta={meta}")


def test_get():
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(os.path.join(tmp, "testdb"), dim=2)
        db.add("x", [1.0, 0.0], {"color": "red"})
        vec, meta = db.get("x")
        assert vec == [1.0, 0.0]
        assert meta["color"] == "red"
        print(f"  get OK: vec={vec}, meta={meta}")


if __name__ == "__main__":
    for name, fn in [
        ("add_and_search", test_add_and_search),
        ("persistence", test_persistence),
        ("get", test_get),
    ]:
        print(f"test_{name}...")
        fn()
        print(f"  PASSED")
    print("\nAll tests passed!")

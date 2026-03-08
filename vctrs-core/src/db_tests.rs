    use super::*;

    #[test]
    fn test_add_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 3, Metric::Cosine).unwrap();

        db.add("a", vec![1.0, 0.0, 0.0], Some(serde_json::json!({"label": "x-axis"}))).unwrap();
        db.add("b", vec![0.0, 1.0, 0.0], None).unwrap();
        db.add("c", vec![0.0, 0.0, 1.0], None).unwrap();

        let results = db.search(&[0.9, 0.1, 0.0], 2, None, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_auto_detect_on_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open_or_create(path.to_str().unwrap(), 3, Metric::Euclidean).unwrap();
            db.add("x", vec![1.0, 2.0, 3.0], None).unwrap();
            db.save().unwrap();
        }

        // Reopen with just the path — should auto-detect dim and metric.
        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            assert_eq!(db.dim(), 3);
            assert_eq!(db.metric(), Metric::Euclidean);
            let (vec, _) = db.get("x").unwrap();
            assert_eq!(vec, vec![1.0, 2.0, 3.0]);
        }
    }

    #[test]
    fn test_upsert() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        // Insert via upsert.
        db.upsert("a", vec![1.0, 0.0], Some(serde_json::json!({"v": 1}))).unwrap();
        assert_eq!(db.len(), 1);
        let (vec, meta) = db.get("a").unwrap();
        assert_eq!(vec, vec![1.0, 0.0]);
        assert_eq!(meta.unwrap()["v"], 1);

        // Update via upsert.
        db.upsert("a", vec![0.0, 1.0], Some(serde_json::json!({"v": 2}))).unwrap();
        assert_eq!(db.len(), 1);
        let (vec, meta) = db.get("a").unwrap();
        assert_eq!(vec, vec![0.0, 1.0]);
        assert_eq!(meta.unwrap()["v"], 2);
    }

    #[test]
    fn test_filtered_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "sci"}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"cat": "art"}))).unwrap();
        db.add("c", vec![0.8, 0.2], Some(serde_json::json!({"cat": "sci"}))).unwrap();

        // Unfiltered: a is closest.
        let results = db.search(&[1.0, 0.0], 1, None, None).unwrap();
        assert_eq!(results[0].id, "a");

        // Filtered to cat=art: b is the only match.
        let filter = Filter::Eq("cat".to_string(), serde_json::json!("art"));
        let results = db.search(&[1.0, 0.0], 1, None, Some(&filter)).unwrap();
        assert_eq!(results[0].id, "b");

        // Filtered to cat=sci: a is closest.
        let filter = Filter::Eq("cat".to_string(), serde_json::json!("sci"));
        let results = db.search(&[1.0, 0.0], 2, None, Some(&filter)).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
        assert_eq!(results[1].id, "c");
    }

    #[test]
    fn test_ids() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        db.add("x", vec![1.0, 0.0], None).unwrap();
        db.add("y", vec![0.0, 1.0], None).unwrap();

        let mut ids = db.ids();
        ids.sort();
        assert_eq!(ids, vec!["x", "y"]);
    }

    #[test]
    fn test_persistence_with_graph() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();
            db.add("x", vec![1.0, 2.0], Some(serde_json::json!({"n": 1}))).unwrap();
            db.add("y", vec![3.0, 4.0], None).unwrap();
            db.save().unwrap();
        }

        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            assert_eq!(db.len(), 2);
            let (vec, meta) = db.get("x").unwrap();
            assert_eq!(vec, vec![1.0, 2.0]);
            assert!(meta.is_some());

            let results = db.search(&[1.0, 2.0], 1, None, None).unwrap();
            assert_eq!(results[0].id, "x");
        }
    }

    #[test]
    fn test_duplicate_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        db.add("a", vec![1.0, 0.0], None).unwrap();
        let result = db.add("a", vec![0.0, 1.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_delete() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 3, Metric::Cosine).unwrap();

        db.add("a", vec![1.0, 0.0, 0.0], None).unwrap();
        db.add("b", vec![0.0, 1.0, 0.0], None).unwrap();
        db.add("c", vec![0.0, 0.0, 1.0], None).unwrap();

        assert_eq!(db.len(), 3);
        assert!(db.delete("b").unwrap());
        assert_eq!(db.len(), 2);
        assert!(!db.contains("b"));

        let results = db.search(&[0.0, 1.0, 0.0], 3, None, None).unwrap();
        assert!(results.iter().all(|r| r.id != "b"));
    }

    #[test]
    fn test_update() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"v": 1}))).unwrap();

        db.update("a", Some(vec![0.0, 1.0]), None).unwrap();
        let (vec, meta) = db.get("a").unwrap();
        assert_eq!(vec, vec![0.0, 1.0]);
        assert_eq!(meta.unwrap()["v"], 1);

        db.update("a", None, Some(Some(serde_json::json!({"v": 2})))).unwrap();
        let (_, meta) = db.get("a").unwrap();
        assert_eq!(meta.unwrap()["v"], 2);
    }

    #[test]
    fn test_batch_insert() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        let items = vec![
            ("a".to_string(), vec![1.0, 0.0], None),
            ("b".to_string(), vec![0.0, 1.0], Some(serde_json::json!({"x": 1}))),
            ("c".to_string(), vec![1.0, 1.0], None),
        ];
        db.add_many(items).unwrap();
        assert_eq!(db.len(), 3);

        let results = db.search(&[1.0, 0.0], 1, None, None).unwrap();
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_compact_reclaims_slots() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 3, Metric::Cosine).unwrap();

        db.add("a", vec![1.0, 0.0, 0.0], Some(serde_json::json!({"v": 1}))).unwrap();
        db.add("b", vec![0.0, 1.0, 0.0], Some(serde_json::json!({"v": 2}))).unwrap();
        db.add("c", vec![0.0, 0.0, 1.0], Some(serde_json::json!({"v": 3}))).unwrap();
        db.add("d", vec![0.5, 0.5, 0.0], None).unwrap();

        // Delete two vectors.
        db.delete("b").unwrap();
        db.delete("d").unwrap();
        assert_eq!(db.len(), 2);
        assert_eq!(db.total_slots(), 4);
        assert_eq!(db.deleted_count(), 2);

        // Compact.
        db.compact().unwrap();
        assert_eq!(db.len(), 2);
        assert_eq!(db.total_slots(), 2); // Slots reclaimed.
        assert_eq!(db.deleted_count(), 0);

        // Search still works.
        let results = db.search(&[1.0, 0.0, 0.0], 1, None, None).unwrap();
        assert_eq!(results[0].id, "a");

        // Metadata preserved.
        let (_, meta) = db.get("a").unwrap();
        assert_eq!(meta.unwrap()["v"], 1);
        let (_, meta) = db.get("c").unwrap();
        assert_eq!(meta.unwrap()["v"], 3);

        // Deleted ids are gone.
        assert!(!db.contains("b"));
        assert!(!db.contains("d"));
    }

    #[test]
    fn test_compact_then_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();
            db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"k": "a"}))).unwrap();
            db.add("b", vec![0.0, 1.0], None).unwrap();
            db.add("c", vec![1.0, 1.0], Some(serde_json::json!({"k": "c"}))).unwrap();
            db.delete("b").unwrap();
            db.compact().unwrap();
            db.save().unwrap();
        }

        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            assert_eq!(db.len(), 2);
            assert_eq!(db.total_slots(), 2);
            assert!(db.contains("a"));
            assert!(!db.contains("b"));
            assert!(db.contains("c"));

            let results = db.search(&[1.0, 0.0], 1, None, None).unwrap();
            assert_eq!(results[0].id, "a");

            let (_, meta) = db.get("c").unwrap();
            assert_eq!(meta.unwrap()["k"], "c");
        }
    }

    #[test]
    fn test_compact_noop_when_no_deletes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        db.add("a", vec![1.0, 0.0], None).unwrap();
        db.add("b", vec![0.0, 1.0], None).unwrap();

        db.compact().unwrap(); // Should be a no-op.
        assert_eq!(db.len(), 2);
        assert_eq!(db.total_slots(), 2);
    }

    #[test]
    fn test_compact_all_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        db.add("a", vec![1.0, 0.0], None).unwrap();
        db.add("b", vec![0.0, 1.0], None).unwrap();
        db.delete("a").unwrap();
        db.delete("b").unwrap();

        db.compact().unwrap();
        assert_eq!(db.len(), 0);
        assert_eq!(db.total_slots(), 0);
        assert_eq!(db.deleted_count(), 0);
    }

    #[test]
    fn test_compact_then_insert() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], None).unwrap();
        db.add("b", vec![0.0, 1.0], None).unwrap();
        db.delete("a").unwrap();
        db.compact().unwrap();

        // Insert after compact.
        db.add("c", vec![0.5, 0.5], Some(serde_json::json!({"new": true}))).unwrap();
        assert_eq!(db.len(), 2);
        assert_eq!(db.total_slots(), 2);

        let results = db.search(&[0.5, 0.5], 1, None, None).unwrap();
        assert_eq!(results[0].id, "c");
    }

    #[test]
    fn test_delete_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();
            db.add("a", vec![1.0, 0.0], None).unwrap();
            db.add("b", vec![0.0, 1.0], None).unwrap();
            db.delete("a").unwrap();
            db.save().unwrap();
        }

        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            assert_eq!(db.len(), 1);
            assert!(!db.contains("a"));
            assert!(db.contains("b"));
        }
    }

    // -- Edge case tests for Database layer -----------------------------------

    #[test]
    fn test_add_wrong_dimension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 3, Metric::Cosine).unwrap();

        let result = db.add("a", vec![1.0, 0.0], None);
        assert!(matches!(result.unwrap_err(), VctrsError::DimensionMismatch { expected: 3, got: 2 }));
    }

    #[test]
    fn test_search_wrong_dimension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 3, Metric::Cosine).unwrap();
        db.add("a", vec![1.0, 0.0, 0.0], None).unwrap();

        let result = db.search(&[1.0, 0.0], 1, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        let result = db.get("missing");
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        assert!(!db.delete("missing").unwrap());
    }

    #[test]
    fn test_update_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        let result = db.update("missing", Some(vec![1.0, 0.0]), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_update_wrong_dimension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();
        db.add("a", vec![1.0, 0.0], None).unwrap();

        let result = db.update("a", Some(vec![1.0, 0.0, 0.0]), None);
        assert!(matches!(result.unwrap_err(), VctrsError::DimensionMismatch { expected: 2, got: 3 }));
    }

    #[test]
    fn test_upsert_wrong_dimension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        let result = db.upsert("a", vec![1.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_many_duplicate_in_batch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        // Existing id.
        db.add("a", vec![1.0, 0.0], None).unwrap();
        let items = vec![
            ("b".to_string(), vec![0.0, 1.0], None),
            ("a".to_string(), vec![0.5, 0.5], None), // duplicate
        ];
        let result = db.add_many(items);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_many_wrong_dimension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        let items = vec![
            ("a".to_string(), vec![1.0, 0.0], None),
            ("b".to_string(), vec![1.0], None), // wrong dim
        ];
        let result = db.add_many(items);
        assert!(result.is_err());
    }

    #[test]
    fn test_open_nonexistent() {
        let result = Database::open("/tmp/vctrs_definitely_not_here_12345");
        assert!(result.is_err());
    }

    #[test]
    fn test_search_empty_db() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        let results = db.search(&[1.0, 0.0], 10, None, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_filtered_search_no_matches() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "x"}))).unwrap();
        db.add("b", vec![0.0, 1.0], Some(serde_json::json!({"cat": "y"}))).unwrap();

        let filter = Filter::Eq("cat".to_string(), serde_json::json!("z"));
        let results = db.search(&[1.0, 0.0], 10, None, Some(&filter)).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_filtered_search_with_ne() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "x"}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"cat": "y"}))).unwrap();
        db.add("c", vec![0.0, 1.0], Some(serde_json::json!({"cat": "x"}))).unwrap();

        let filter = Filter::Ne("cat".to_string(), serde_json::json!("x"));
        let results = db.search(&[1.0, 0.0], 10, None, Some(&filter)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }

    #[test]
    fn test_filtered_search_with_in() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "x"}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"cat": "y"}))).unwrap();
        db.add("c", vec![0.0, 1.0], Some(serde_json::json!({"cat": "z"}))).unwrap();

        let filter = Filter::In("cat".to_string(), vec![serde_json::json!("x"), serde_json::json!("z")]);
        let results = db.search(&[1.0, 0.0], 10, None, Some(&filter)).unwrap();
        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"c"));
    }

    #[test]
    fn test_filtered_search_and_combinator() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "x", "val": 1}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"cat": "x", "val": 2}))).unwrap();
        db.add("c", vec![0.0, 1.0], Some(serde_json::json!({"cat": "y", "val": 1}))).unwrap();

        let filter = Filter::And(vec![
            Filter::Eq("cat".to_string(), serde_json::json!("x")),
            Filter::Eq("val".to_string(), serde_json::json!(2)),
        ]);
        let results = db.search(&[1.0, 0.0], 10, None, Some(&filter)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }

    #[test]
    fn test_filtered_search_or_combinator() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "x"}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"cat": "y"}))).unwrap();
        db.add("c", vec![0.0, 1.0], Some(serde_json::json!({"cat": "z"}))).unwrap();

        let filter = Filter::Or(vec![
            Filter::Eq("cat".to_string(), serde_json::json!("x")),
            Filter::Eq("cat".to_string(), serde_json::json!("z")),
        ]);
        let results = db.search(&[1.0, 0.0], 10, None, Some(&filter)).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_filter_no_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        // Vectors with no metadata should not match any Eq filter.
        db.add("a", vec![1.0, 0.0], None).unwrap();
        db.add("b", vec![0.0, 1.0], Some(serde_json::json!({"cat": "x"}))).unwrap();

        let filter = Filter::Eq("cat".to_string(), serde_json::json!("x"));
        let results = db.search(&[1.0, 0.0], 10, None, Some(&filter)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }

    #[test]
    fn test_compact_with_metadata_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "x"}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"cat": "y"}))).unwrap();
        db.add("c", vec![0.0, 1.0], Some(serde_json::json!({"cat": "x"}))).unwrap();
        db.delete("b").unwrap();
        db.compact().unwrap();

        // Filtered search should work after compact.
        let filter = Filter::Eq("cat".to_string(), serde_json::json!("x"));
        let results = db.search(&[1.0, 0.0], 10, None, Some(&filter)).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_upsert_then_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.upsert("a", vec![1.0, 0.0], None).unwrap();
        db.upsert("b", vec![0.0, 1.0], None).unwrap();

        // Move "a" to be near "b".
        db.upsert("a", vec![0.0, 1.0], None).unwrap();

        let results = db.search(&[0.0, 1.0], 2, None, None).unwrap();
        // Both should be found at [0,1].
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }

    #[test]
    fn test_search_many_filtered() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "x"}))).unwrap();
        db.add("b", vec![0.0, 1.0], Some(serde_json::json!({"cat": "y"}))).unwrap();
        db.add("c", vec![0.5, 0.5], Some(serde_json::json!({"cat": "x"}))).unwrap();

        let filter = Filter::Eq("cat".to_string(), serde_json::json!("x"));
        let results = db.search_many(
            &[&[1.0, 0.0], &[0.0, 1.0]],
            10, None, Some(&filter),
        ).unwrap();

        assert_eq!(results.len(), 2);
        // Both query results should only contain cat=x items.
        for batch in &results {
            for r in batch {
                assert!(r.id == "a" || r.id == "c");
            }
        }
    }

    #[test]
    fn test_save_load_with_quantized_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open_or_create_with_config(
                path.to_str().unwrap(), 2, Metric::Euclidean,
                HnswConfig { m: 16, ef_construction: 200, quantize: true },
            ).unwrap();
            db.add("a", vec![1.0, 0.0], None).unwrap();
            db.add("b", vec![0.0, 1.0], None).unwrap();
            db.save().unwrap();
        }

        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            assert!(db.has_quantized_search());
            let results = db.search(&[1.0, 0.0], 1, None, None).unwrap();
            assert_eq!(results[0].id, "a");
        }
    }

    #[test]
    fn test_compact_then_save_load_search() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open_or_create(path.to_str().unwrap(), 3, Metric::Cosine).unwrap();
            db.add("a", vec![1.0, 0.0, 0.0], Some(serde_json::json!({"k": 1}))).unwrap();
            db.add("b", vec![0.0, 1.0, 0.0], None).unwrap();
            db.add("c", vec![0.0, 0.0, 1.0], Some(serde_json::json!({"k": 3}))).unwrap();
            db.add("d", vec![0.5, 0.5, 0.0], None).unwrap();
            db.delete("b").unwrap();
            db.delete("d").unwrap();
            db.compact().unwrap();
            db.save().unwrap();
        }

        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            assert_eq!(db.len(), 2);
            assert_eq!(db.total_slots(), 2);
            assert!(db.contains("a"));
            assert!(db.contains("c"));
            assert!(!db.contains("b"));
            assert!(!db.contains("d"));

            let (_, meta) = db.get("a").unwrap();
            assert_eq!(meta.unwrap()["k"], 1);

            let results = db.search(&[1.0, 0.0, 0.0], 1, None, None).unwrap();
            assert_eq!(results[0].id, "a");

            // Can insert new items after load.
            db.add("e", vec![0.0, 0.0, 1.0], None).unwrap();
            assert_eq!(db.len(), 3);
        }
    }

    #[test]
    fn test_ids_after_operations() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Cosine).unwrap();

        db.add("x", vec![1.0, 0.0], None).unwrap();
        db.add("y", vec![0.0, 1.0], None).unwrap();
        db.add("z", vec![0.5, 0.5], None).unwrap();

        db.delete("y").unwrap();
        let mut ids = db.ids();
        ids.sort();
        assert_eq!(ids, vec!["x", "z"]);

        db.compact().unwrap();
        let mut ids = db.ids();
        ids.sort();
        assert_eq!(ids, vec!["x", "z"]);
    }

    #[test]
    fn test_len_consistency() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        assert_eq!(db.len(), 0);
        assert!(db.is_empty());

        db.add("a", vec![1.0, 0.0], None).unwrap();
        assert_eq!(db.len(), 1);

        db.add("b", vec![0.0, 1.0], None).unwrap();
        assert_eq!(db.len(), 2);

        db.delete("a").unwrap();
        assert_eq!(db.len(), 1);
        assert_eq!(db.deleted_count(), 1);
        assert_eq!(db.total_slots(), 2);

        db.compact().unwrap();
        assert_eq!(db.len(), 1);
        assert_eq!(db.deleted_count(), 0);
        assert_eq!(db.total_slots(), 1);
    }

    #[test]
    fn test_stats() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 3, Metric::Cosine).unwrap();

        // Empty database stats.
        let s = db.stats();
        assert_eq!(s.num_vectors, 0);
        assert_eq!(s.num_deleted, 0);
        assert!(s.uses_brute_force);
        assert!(!s.uses_quantized_search);

        // Add some vectors.
        db.add("a", vec![1.0, 0.0, 0.0], None).unwrap();
        db.add("b", vec![0.0, 1.0, 0.0], None).unwrap();
        db.add("c", vec![0.0, 0.0, 1.0], None).unwrap();

        let s = db.stats();
        assert_eq!(s.num_vectors, 3);
        assert_eq!(s.num_deleted, 0);
        assert!(s.memory_vectors_bytes > 0);
        assert!(s.num_layers >= 1);

        // Delete one.
        db.delete("b").unwrap();
        let s = db.stats();
        assert_eq!(s.num_vectors, 2);
        assert_eq!(s.num_deleted, 1);

        // Enable quantized search.
        db.enable_quantized_search();
        let s = db.stats();
        assert!(s.uses_quantized_search);
        assert!(s.memory_quantized_bytes > 0);
    }

    #[test]
    fn test_filter_gt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"score": 10}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"score": 20}))).unwrap();
        db.add("c", vec![0.0, 1.0], Some(serde_json::json!({"score": 30}))).unwrap();

        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Gt("score".into(), 15.0))).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.id != "a"));
    }

    #[test]
    fn test_filter_gte() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"score": 10}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"score": 20}))).unwrap();

        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Gte("score".into(), 20.0))).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }

    #[test]
    fn test_filter_lt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"score": 10}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"score": 20}))).unwrap();
        db.add("c", vec![0.0, 1.0], Some(serde_json::json!({"score": 30}))).unwrap();

        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Lt("score".into(), 25.0))).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.id != "c"));
    }

    #[test]
    fn test_filter_lte() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"score": 10}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"score": 20}))).unwrap();

        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Lte("score".into(), 10.0))).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_filter_range_combined() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        for i in 0..10 {
            db.add(
                &format!("v{}", i),
                vec![i as f32, 0.0],
                Some(serde_json::json!({"val": i})),
            ).unwrap();
        }

        // 3 <= val < 7
        let filter = Filter::And(vec![
            Filter::Gte("val".into(), 3.0),
            Filter::Lt("val".into(), 7.0),
        ]);
        let results = db.search(&[5.0, 0.0], 10, None, Some(&filter)).unwrap();
        assert_eq!(results.len(), 4); // 3, 4, 5, 6
        for r in &results {
            let val = r.metadata.as_ref().unwrap()["val"].as_i64().unwrap();
            assert!(val >= 3 && val < 7, "val {} not in [3, 7)", val);
        }
    }

    #[test]
    fn test_filter_gt_non_numeric_field_excluded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"name": "alice"}))).unwrap();
        db.add("b", vec![0.0, 1.0], Some(serde_json::json!({"score": 10}))).unwrap();

        // $gt on a string field should return no match for that doc.
        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Gt("name".into(), 5.0))).unwrap();
        assert!(results.iter().all(|r| r.id != "a"), "string field matched numeric $gt");
    }

    #[test]
    fn test_filter_gt_missing_field_excluded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], None).unwrap();
        db.add("b", vec![0.0, 1.0], Some(serde_json::json!({"score": 10}))).unwrap();

        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Gt("score".into(), 5.0))).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }

    // -- Metadata index tests --

    #[test]
    fn test_meta_index_survives_delete() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "x"}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"cat": "x"}))).unwrap();
        db.add("c", vec![0.0, 1.0], Some(serde_json::json!({"cat": "y"}))).unwrap();

        // Both a and b match cat=x.
        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Eq("cat".into(), serde_json::json!("x")))).unwrap();
        assert_eq!(results.len(), 2);

        // Delete a — meta index should remove it.
        db.delete("a").unwrap();
        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Eq("cat".into(), serde_json::json!("x")))).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "b");
    }

    #[test]
    fn test_meta_index_survives_update() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "x"}))).unwrap();
        db.add("b", vec![0.0, 1.0], Some(serde_json::json!({"cat": "y"}))).unwrap();

        // a has cat=x.
        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Eq("cat".into(), serde_json::json!("x")))).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");

        // Update a's metadata to cat=y.
        db.update("a", None, Some(Some(serde_json::json!({"cat": "y"})))).unwrap();

        // Now cat=x should return nothing.
        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Eq("cat".into(), serde_json::json!("x")))).unwrap();
        assert_eq!(results.len(), 0);

        // And cat=y should return both.
        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Eq("cat".into(), serde_json::json!("y")))).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_meta_index_survives_upsert() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "x"}))).unwrap();

        // Upsert changes metadata.
        db.upsert("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "z"}))).unwrap();

        // Old value should be gone.
        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Eq("cat".into(), serde_json::json!("x")))).unwrap();
        assert_eq!(results.len(), 0);

        // New value should be present.
        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Eq("cat".into(), serde_json::json!("z")))).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_meta_index_survives_compact() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "x"}))).unwrap();
        db.add("b", vec![0.9, 0.1], Some(serde_json::json!({"cat": "y"}))).unwrap();
        db.add("c", vec![0.0, 1.0], Some(serde_json::json!({"cat": "x"}))).unwrap();

        db.delete("b").unwrap();
        db.compact().unwrap();

        // After compact, filtered search should still work with the inverted index.
        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Eq("cat".into(), serde_json::json!("x")))).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.id == "a" || r.id == "c"));

        // y was deleted.
        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Eq("cat".into(), serde_json::json!("y")))).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_meta_index_rebuild_on_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();
            db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"cat": "x"}))).unwrap();
            db.add("b", vec![0.0, 1.0], Some(serde_json::json!({"cat": "y"}))).unwrap();
            db.save().unwrap();
        }

        // Reopen — meta_index should be rebuilt from loaded metadata.
        let db = Database::open(path.to_str().unwrap()).unwrap();
        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Eq("cat".into(), serde_json::json!("x")))).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_meta_index_add_many() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();

        db.add_many(vec![
            ("a".into(), vec![1.0, 0.0], Some(serde_json::json!({"cat": "x"}))),
            ("b".into(), vec![0.9, 0.1], Some(serde_json::json!({"cat": "y"}))),
            ("c".into(), vec![0.0, 1.0], Some(serde_json::json!({"cat": "x"}))),
        ]).unwrap();

        let results = db.search(&[1.0, 0.0], 10, None, Some(&Filter::Eq("cat".into(), serde_json::json!("x")))).unwrap();
        assert_eq!(results.len(), 2);
    }

    // -- WAL integration tests --

    #[test]
    fn test_wal_recovery_after_crash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        // Create db, save initial state.
        {
            let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();
            db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"v": 1}))).unwrap();
            db.save().unwrap(); // Snapshot written, WAL truncated.
        }

        // Add more data WITHOUT saving (simulates crash).
        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            db.add("b", vec![0.0, 1.0], Some(serde_json::json!({"v": 2}))).unwrap();
            db.add("c", vec![0.5, 0.5], None).unwrap();
            // No save() — "crash" here. WAL has entries for b and c.
        }

        // Reopen — WAL should be replayed, recovering b and c.
        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            assert_eq!(db.len(), 3);
            assert!(db.contains("a"));
            assert!(db.contains("b"));
            assert!(db.contains("c"));

            let (vec, meta) = db.get("b").unwrap();
            assert_eq!(vec, vec![0.0, 1.0]);
            assert_eq!(meta.unwrap()["v"], 2);

            // Search should find all three.
            let results = db.search(&[1.0, 0.0], 3, None, None).unwrap();
            assert_eq!(results.len(), 3);
        }
    }

    #[test]
    fn test_wal_recovery_delete() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();
            db.add("a", vec![1.0, 0.0], None).unwrap();
            db.add("b", vec![0.0, 1.0], None).unwrap();
            db.save().unwrap();
        }

        // Delete without saving.
        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            db.delete("a").unwrap();
            // No save().
        }

        // Reopen — WAL replay should delete "a".
        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            assert_eq!(db.len(), 1);
            assert!(!db.contains("a"));
            assert!(db.contains("b"));
        }
    }

    #[test]
    fn test_wal_recovery_update() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();
            db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"v": 1}))).unwrap();
            db.save().unwrap();
        }

        // Update without saving.
        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            db.update("a", Some(vec![0.0, 1.0]), Some(Some(serde_json::json!({"v": 99})))).unwrap();
            // No save().
        }

        // Reopen — WAL replay should apply the update.
        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            let (vec, meta) = db.get("a").unwrap();
            assert_eq!(vec, vec![0.0, 1.0]);
            assert_eq!(meta.unwrap()["v"], 99);
        }
    }

    #[test]
    fn test_wal_cleared_after_save() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();
            db.add("a", vec![1.0, 0.0], None).unwrap();
            db.add("b", vec![0.0, 1.0], None).unwrap();
            db.save().unwrap();
        }

        // WAL should be empty after save.
        let wal = Wal::new(&dir.path().join("testdb"));
        assert!(!wal.has_entries());
    }

    #[test]
    fn test_wal_recovery_mixed_operations() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();
            db.add("a", vec![1.0, 0.0], Some(serde_json::json!({"v": 1}))).unwrap();
            db.add("b", vec![0.0, 1.0], Some(serde_json::json!({"v": 2}))).unwrap();
            db.save().unwrap();
        }

        // Perform a mix of unsaved operations.
        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            db.add("c", vec![0.5, 0.5], Some(serde_json::json!({"v": 3}))).unwrap();
            db.delete("a").unwrap();
            db.update("b", None, Some(Some(serde_json::json!({"v": 20})))).unwrap();
            // No save().
        }

        // Reopen — should have b (updated) and c, but not a.
        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            assert_eq!(db.len(), 2);
            assert!(!db.contains("a"));
            assert!(db.contains("b"));
            assert!(db.contains("c"));

            let (_, meta) = db.get("b").unwrap();
            assert_eq!(meta.unwrap()["v"], 20);

            let (_, meta) = db.get("c").unwrap();
            assert_eq!(meta.unwrap()["v"], 3);
        }
    }

    // -- Concurrency tests --

    #[test]
    fn test_concurrent_reads_during_writes() {
        use std::sync::Arc;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Arc::new(Database::open_or_create(path.to_str().unwrap(), 4, Metric::Euclidean).unwrap());

        // Pre-populate with some data.
        for i in 0..50 {
            db.add(
                &format!("init_{}", i),
                vec![i as f32, 0.0, 0.0, 0.0],
                Some(serde_json::json!({"idx": i})),
            ).unwrap();
        }

        let mut handles = Vec::new();

        // Spawn writer threads.
        for t in 0..4 {
            let db = Arc::clone(&db);
            handles.push(thread::spawn(move || {
                for i in 0..25 {
                    let id = format!("w{}_{}", t, i);
                    db.add(&id, vec![t as f32, i as f32, 0.0, 0.0], None).unwrap();
                }
            }));
        }

        // Spawn reader threads.
        for _ in 0..4 {
            let db = Arc::clone(&db);
            handles.push(thread::spawn(move || {
                for _ in 0..50 {
                    let _ = db.search(&[1.0, 0.0, 0.0, 0.0], 5, None, None).unwrap();
                    let _ = db.len();
                    let _ = db.ids();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // All writes should have succeeded.
        assert_eq!(db.len(), 50 + 4 * 25);
    }

    #[test]
    fn test_concurrent_search_many_filtered() {
        use std::sync::Arc;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Arc::new(Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap());

        for i in 0..20 {
            db.add(
                &format!("v{}", i),
                vec![i as f32, (20 - i) as f32],
                Some(serde_json::json!({"cat": if i % 2 == 0 { "even" } else { "odd" }})),
            ).unwrap();
        }

        let mut handles = Vec::new();

        // Multiple threads doing filtered search_many concurrently.
        for _ in 0..4 {
            let db = Arc::clone(&db);
            handles.push(thread::spawn(move || {
                let filter = Filter::Eq("cat".into(), serde_json::json!("even"));
                for _ in 0..20 {
                    let results = db.search_many(
                        &[&[0.0, 20.0], &[19.0, 1.0]],
                        5, None, Some(&filter),
                    ).unwrap();
                    assert_eq!(results.len(), 2);
                    for batch in &results {
                        for r in batch {
                            let cat = r.metadata.as_ref().unwrap()["cat"].as_str().unwrap();
                            assert_eq!(cat, "even");
                        }
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_concurrent_upsert() {
        use std::sync::Arc;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");
        let db = Arc::new(Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap());

        // Multiple threads upserting the same keys.
        let mut handles = Vec::new();
        for t in 0..4 {
            let db = Arc::clone(&db);
            handles.push(thread::spawn(move || {
                for i in 0..20 {
                    let id = format!("shared_{}", i);
                    db.upsert(&id, vec![t as f32, i as f32], Some(serde_json::json!({"t": t}))).unwrap();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Should have exactly 20 unique ids.
        assert_eq!(db.len(), 20);

        // Each key should exist with valid data.
        for i in 0..20 {
            let id = format!("shared_{}", i);
            assert!(db.contains(&id));
            let (vec, meta) = db.get(&id).unwrap();
            assert_eq!(vec.len(), 2);
            assert!(meta.is_some());
        }
    }

    #[test]
    fn test_wal_idempotent_replay() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("testdb");

        {
            let db = Database::open_or_create(path.to_str().unwrap(), 2, Metric::Euclidean).unwrap();
            db.add("a", vec![1.0, 0.0], None).unwrap();
            db.save().unwrap();
        }

        // Add without saving.
        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            db.add("b", vec![0.0, 1.0], None).unwrap();
        }

        // Open twice without saving — WAL should replay idempotently.
        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            assert_eq!(db.len(), 2);
        }
        {
            let db = Database::open(path.to_str().unwrap()).unwrap();
            assert_eq!(db.len(), 2);
        }
    }

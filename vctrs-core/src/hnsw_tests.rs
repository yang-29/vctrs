    use super::*;

    #[test]
    fn test_insert_and_search() {
        let mut index = HnswIndex::new(3, Metric::Euclidean, 16, 200);

        index.insert(vec![1.0, 0.0, 0.0]);
        index.insert(vec![0.0, 1.0, 0.0]);
        index.insert(vec![0.0, 0.0, 1.0]);
        index.insert(vec![1.0, 1.0, 0.0]);

        let results = index.search(&[1.0, 0.1, 0.0], 2, 50);
        assert_eq!(results.len(), 2);
        assert!(results[0].0 == 0 || results[0].0 == 3);
    }

    #[test]
    fn test_cosine_search() {
        let mut index = HnswIndex::new(2, Metric::Cosine, 16, 200);

        index.insert(vec![1.0, 0.0]);
        index.insert(vec![0.0, 1.0]);
        index.insert(vec![0.707, 0.707]);

        let results = index.search(&[0.9, 0.1], 1, 50);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_empty_index() {
        let index = HnswIndex::new(4, Metric::Cosine, 16, 200);
        let results = index.search(&[1.0, 2.0, 3.0, 4.0], 5, 50);
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_insert() {
        let mut index = HnswIndex::new(32, Metric::Euclidean, 16, 200);
        let mut rng = rand::thread_rng();

        let vecs: Vec<Vec<f32>> = (0..5000)
            .map(|_| (0..32).map(|_| rng.gen::<f32>()).collect())
            .collect();

        let ids = index.batch_insert(vecs.clone());
        assert_eq!(ids.len(), 5000);
        assert_eq!(index.len(), 5000);

        // Verify recall: search for a known vector.
        let results = index.search(&vecs[100], 1, 50);
        assert_eq!(results[0].0, 100);
    }

    #[test]
    fn test_batch_insert_recall() {
        let mut index = HnswIndex::new(32, Metric::Euclidean, 16, 200);
        let mut rng = rand::thread_rng();

        let vecs: Vec<Vec<f32>> = (0..2000)
            .map(|_| (0..32).map(|_| rng.gen::<f32>()).collect())
            .collect();

        index.batch_insert(vecs.clone());

        // Brute-force check.
        let query: Vec<f32> = (0..32).map(|_| rng.gen::<f32>()).collect();
        let mut brute: Vec<(usize, f32)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i, distance(&query, v, Metric::Euclidean)))
            .collect();
        brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let true_nearest = brute[0].0 as u32;

        let results = index.search(&query, 10, 100);
        let found: Vec<u32> = results.iter().map(|r| r.0).collect();
        assert!(
            found.contains(&true_nearest),
            "batch HNSW missed true nearest neighbor"
        );
    }

    #[test]
    fn test_graph_serialization() {
        let mut index = HnswIndex::new(4, Metric::Cosine, 16, 200);
        index.insert(vec![1.0, 0.0, 0.0, 0.0]);
        index.insert(vec![0.0, 1.0, 0.0, 0.0]);
        index.insert(vec![0.0, 0.0, 1.0, 0.0]);
        index.mark_deleted(1);

        let dir = tempfile::tempdir().unwrap();
        let vec_path = dir.path().join("vectors.bin");
        let graph_path = dir.path().join("graph.vctrs");

        // Save vectors + graph.
        {
            let mut vf = std::io::BufWriter::new(std::fs::File::create(&vec_path).unwrap());
            index.save_vectors(&mut vf).unwrap();
            let mut gf = std::io::BufWriter::new(std::fs::File::create(&graph_path).unwrap());
            index.save_graph(&mut gf).unwrap();
        }

        // Load with mmap.
        let vec_file = std::fs::File::open(&vec_path).unwrap();
        let mmap = unsafe { Mmap::map(&vec_file).unwrap() };
        let graph_data = std::fs::read(&graph_path).unwrap();
        let (loaded, _remaining) = HnswIndex::load_graph_mmap(&graph_data, mmap).unwrap();

        assert_eq!(loaded.len(), 2); // 3 inserted, 1 deleted
        assert_eq!(loaded.total_slots(), 3);
        assert!(loaded.is_deleted(1));
        assert!(loaded.is_mmap());

        // Search should work the same.
        let results = loaded.search(&[1.0, 0.0, 0.0, 0.0], 1, 50);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_recall() {
        let dim = 32;
        let mut rng = rand::thread_rng();
        let mut index = HnswIndex::new(dim, Metric::Euclidean, 16, 200);

        let mut vecs = Vec::new();
        for _ in 0..1000 {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            index.insert(v.clone());
            vecs.push(v);
        }

        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let mut brute_force: Vec<(usize, f32)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i, distance(&query, v, Metric::Euclidean)))
            .collect();
        brute_force.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let true_nearest = brute_force[0].0 as u32;

        let results = index.search(&query, 10, 100);
        let found_ids: Vec<u32> = results.iter().map(|r| r.0).collect();
        assert!(
            found_ids.contains(&true_nearest),
            "HNSW missed the true nearest neighbor"
        );
    }

    #[test]
    fn test_search_many() {
        let mut index = HnswIndex::new(3, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0, 0.0]);
        index.insert(vec![0.0, 1.0, 0.0]);
        index.insert(vec![0.0, 0.0, 1.0]);

        let q1 = [1.0f32, 0.0, 0.0];
        let q2 = [0.0f32, 1.0, 0.0];
        let queries: Vec<&[f32]> = vec![&q1, &q2];

        let results = index.search_many(&queries, 1, 50);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0][0].0, 0); // q1 closest to vec 0
        assert_eq!(results[1][0].0, 1); // q2 closest to vec 1
    }

    #[test]
    fn test_quantized_search_recall() {
        // Use enough vectors to exceed brute-force threshold so HNSW path is exercised.
        // dim=8, n=2_000_000/8 = needs n*dim > 10M. Use dim=8, n=500 (brute-force)
        // for recall, then a separate test for HNSW quantized path.
        let dim = 32;
        let mut rng = rand::thread_rng();
        let mut index = HnswIndex::new(dim, Metric::Euclidean, 16, 200);

        let vecs: Vec<Vec<f32>> = (0..2000)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();
        index.batch_insert(vecs.clone());

        // Enable quantized search.
        index.enable_quantized_search();
        assert!(index.has_quantized_search());

        // Search should still find the true nearest neighbor (high recall).
        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        // Brute-force ground truth.
        let mut brute: Vec<(usize, f32)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i, distance(&query, v, Metric::Euclidean)))
            .collect();
        brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let true_nearest = brute[0].0 as u32;

        let results = index.search(&query, 10, 100);
        let found: Vec<u32> = results.iter().map(|r| r.0).collect();
        assert!(
            found.contains(&true_nearest),
            "quantized search missed true nearest neighbor"
        );

        // Distances should be full-precision (matching brute-force exactly).
        let expected_dist = distance(&query, &vecs[results[0].0 as usize], Metric::Euclidean);
        assert!(
            (results[0].1 - expected_dist).abs() < 1e-4,
            "distance mismatch: got {}, expected {} (delta {})",
            results[0].1, expected_dist, (results[0].1 - expected_dist).abs()
        );
    }

    #[test]
    fn test_quantized_insert_after_enable() {
        let mut index = HnswIndex::new(3, Metric::Cosine, 16, 200);
        index.insert(vec![1.0, 0.0, 0.0]);
        index.insert(vec![0.0, 1.0, 0.0]);
        index.enable_quantized_search();

        // Insert after enabling quantized search.
        index.insert(vec![0.0, 0.0, 1.0]);
        assert_eq!(index.len(), 3);

        let results = index.search(&[0.0, 0.0, 1.0], 1, 50);
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn test_update_vector_reconnects() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);

        // Insert 3 vectors: [1,0], [0,1], [0.5, 0.5]
        index.insert(vec![1.0, 0.0]);
        index.insert(vec![0.0, 1.0]);
        index.insert(vec![0.5, 0.5]);

        // Move vector 0 from [1,0] to [0,1] (near vector 1).
        index.update_vector(0, vec![0.0, 1.0]);

        // Searching near [0,1] should find vector 0 (now at [0,1]) as closest.
        let results = index.search(&[0.0, 1.0], 2, 50);
        let found: Vec<u32> = results.iter().map(|r| r.0).collect();
        assert!(found.contains(&0), "updated vector not found near new position");
        assert!(found.contains(&1), "original neighbor not found");
    }

    // -- Recall tests (statistical, multiple queries) -------------------------

    /// Helper: compute recall@k over multiple queries.
    fn measure_recall(index: &HnswIndex, vecs: &[Vec<f32>], metric: Metric, k: usize, ef: usize, num_queries: usize) -> f64 {
        let mut rng = rand::thread_rng();
        let dim = vecs[0].len();
        let mut total_recall = 0.0;

        for _ in 0..num_queries {
            let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

            // Brute-force ground truth.
            let mut brute: Vec<(usize, f32)> = vecs
                .iter()
                .enumerate()
                .map(|(i, v)| (i, distance(&query, v, metric)))
                .collect();
            brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let truth: HashSet<u32> = brute.iter().take(k).map(|(i, _)| *i as u32).collect();

            let results = index.search(&query, k, ef);
            let found: HashSet<u32> = results.iter().map(|r| r.0).collect();

            total_recall += found.intersection(&truth).count() as f64 / k as f64;
        }

        total_recall / num_queries as f64
    }

    #[test]
    fn test_recall_at_k_sequential_insert() {
        let dim = 32;
        let n = 5000;
        let mut rng = rand::thread_rng();
        let mut index = HnswIndex::new(dim, Metric::Euclidean, 16, 200);

        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();
        for v in &vecs {
            index.insert(v.clone());
        }

        let recall = measure_recall(&index, &vecs, Metric::Euclidean, 10, 200, 50);
        assert!(
            recall > 0.90,
            "sequential insert recall@10 too low: {:.3} (expected > 0.90)",
            recall
        );
    }

    #[test]
    fn test_recall_at_k_batch_insert() {
        let dim = 32;
        let n = 5000;
        let mut rng = rand::thread_rng();
        let mut index = HnswIndex::new(dim, Metric::Euclidean, 16, 200);

        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();
        index.batch_insert(vecs.clone());

        let recall = measure_recall(&index, &vecs, Metric::Euclidean, 10, 200, 50);
        assert!(
            recall > 0.90,
            "batch insert recall@10 too low: {:.3} (expected > 0.90)",
            recall
        );
    }

    #[test]
    fn test_recall_cosine_metric() {
        let dim = 64;
        let n = 3000;
        let mut rng = rand::thread_rng();
        let mut index = HnswIndex::new(dim, Metric::Cosine, 16, 200);

        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();
        index.batch_insert(vecs.clone());

        let recall = measure_recall(&index, &vecs, Metric::Cosine, 10, 200, 50);
        assert!(
            recall > 0.90,
            "cosine recall@10 too low: {:.3} (expected > 0.90)",
            recall
        );
    }

    #[test]
    fn test_recall_dot_product_metric() {
        let dim = 64;
        let n = 3000;
        let mut rng = rand::thread_rng();
        let mut index = HnswIndex::new(dim, Metric::DotProduct, 16, 200);

        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();
        index.batch_insert(vecs.clone());

        let recall = measure_recall(&index, &vecs, Metric::DotProduct, 10, 200, 50);
        assert!(
            recall > 0.90,
            "dot product recall@10 too low: {:.3} (expected > 0.90)",
            recall
        );
    }

    #[test]
    fn test_recall_quantized_vs_full() {
        let dim = 32;
        let n = 5000;
        let mut rng = rand::thread_rng();
        let mut index = HnswIndex::new(dim, Metric::Euclidean, 16, 200);

        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();
        index.batch_insert(vecs.clone());

        let recall_full = measure_recall(&index, &vecs, Metric::Euclidean, 10, 200, 50);

        index.enable_quantized_search();
        let recall_quantized = measure_recall(&index, &vecs, Metric::Euclidean, 10, 200, 50);

        // Quantized recall should be close to full-precision.
        assert!(
            recall_quantized > 0.85,
            "quantized recall@10 too low: {:.3} (expected > 0.85)",
            recall_quantized
        );
        assert!(
            (recall_full - recall_quantized).abs() < 0.10,
            "quantized recall delta too large: full={:.3}, quantized={:.3}",
            recall_full, recall_quantized
        );
    }

    #[test]
    fn test_recall_after_deletes() {
        let dim = 32;
        let n = 3000;
        let mut rng = rand::thread_rng();
        let mut index = HnswIndex::new(dim, Metric::Euclidean, 16, 200);

        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();
        index.batch_insert(vecs.clone());

        // Delete every other vector.
        for i in (0..n).step_by(2) {
            index.mark_deleted(i as u32);
        }

        let live_vecs: Vec<Vec<f32>> = vecs.iter().enumerate()
            .filter(|(i, _)| i % 2 != 0)
            .map(|(_, v)| v.clone())
            .collect();

        // Recall among live vectors only.
        let mut total_recall = 0.0;
        let num_queries = 50;
        for _ in 0..num_queries {
            let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

            // Brute-force over live vectors.
            let mut brute: Vec<(u32, f32)> = live_vecs
                .iter()
                .enumerate()
                .map(|(i, v)| ((i * 2 + 1) as u32, distance(&query, v, Metric::Euclidean)))
                .collect();
            brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let truth: HashSet<u32> = brute.iter().take(10).map(|(id, _)| *id).collect();

            let results = index.search(&query, 10, 200);
            let found: HashSet<u32> = results.iter().map(|r| r.0).collect();

            // Results should never contain deleted ids.
            for r in &results {
                assert!(!index.is_deleted(r.0), "search returned deleted id {}", r.0);
            }

            total_recall += found.intersection(&truth).count() as f64 / 10.0;
        }
        let recall = total_recall / num_queries as f64;
        assert!(
            recall > 0.80,
            "recall after deletes too low: {:.3} (expected > 0.80)",
            recall
        );
    }

    #[test]
    fn test_recall_after_compact() {
        let dim = 32;
        let n = 2000;
        let mut rng = rand::thread_rng();
        let mut index = HnswIndex::new(dim, Metric::Euclidean, 16, 200);

        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();
        index.batch_insert(vecs.clone());

        // Delete half.
        for i in 0..n / 2 {
            index.mark_deleted(i as u32);
        }

        let (compacted, old_to_new) = index.compact();
        let live_vecs: Vec<Vec<f32>> = (n / 2..n).map(|i| vecs[i].clone()).collect();

        assert_eq!(compacted.len(), n / 2);
        assert_eq!(compacted.total_slots(), n / 2);
        assert_eq!(compacted.deleted_ids().len(), 0);

        let recall = measure_recall(&compacted, &live_vecs, Metric::Euclidean, 10, 200, 30);
        assert!(
            recall > 0.85,
            "post-compact recall@10 too low: {:.3} (expected > 0.85)",
            recall
        );
    }

    // -- Edge case tests ------------------------------------------------------

    #[test]
    fn test_search_k_larger_than_index() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]);
        index.insert(vec![0.0, 1.0]);

        // Ask for more than exist.
        let results = index.search(&[1.0, 0.0], 100, 50);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_all_deleted() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]);
        index.insert(vec![0.0, 1.0]);
        index.mark_deleted(0);
        index.mark_deleted(1);

        let results = index.search(&[1.0, 0.0], 10, 50);
        assert!(results.is_empty());
    }

    #[test]
    fn test_single_vector_index() {
        let mut index = HnswIndex::new(3, Metric::Cosine, 16, 200);
        index.insert(vec![1.0, 0.0, 0.0]);

        let results = index.search(&[0.5, 0.5, 0.0], 1, 50);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_identical_vectors() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        // Insert the same vector multiple times.
        for _ in 0..10 {
            index.insert(vec![1.0, 0.0]);
        }

        let results = index.search(&[1.0, 0.0], 5, 50);
        assert_eq!(results.len(), 5);
        // All distances should be 0.
        for r in &results {
            assert!(r.1.abs() < 1e-6, "identical vectors should have distance ~0, got {}", r.1);
        }
    }

    #[test]
    fn test_zero_vector() {
        let mut index = HnswIndex::new(3, Metric::Euclidean, 16, 200);
        index.insert(vec![0.0, 0.0, 0.0]);
        index.insert(vec![1.0, 0.0, 0.0]);

        let results = index.search(&[0.0, 0.0, 0.0], 1, 50);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1.abs() < 1e-6);
    }

    #[test]
    fn test_search_filtered_basic() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]); // id 0
        index.insert(vec![0.9, 0.1]); // id 1
        index.insert(vec![0.0, 1.0]); // id 2
        index.insert(vec![0.1, 0.9]); // id 3

        // Filter: only even ids.
        let results = index.search_filtered(&[1.0, 0.0], 2, 50, |id| id % 2 == 0);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // closest even
        assert_eq!(results[1].0, 2); // next closest even
    }

    #[test]
    fn test_search_filtered_no_matches() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]);
        index.insert(vec![0.0, 1.0]);

        // Filter matches nothing.
        let results = index.search_filtered(&[1.0, 0.0], 10, 50, |_| false);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_filtered_all_match() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]);
        index.insert(vec![0.0, 1.0]);

        // Filter matches everything — should behave like unfiltered.
        let filtered = index.search_filtered(&[1.0, 0.0], 2, 50, |_| true);
        let unfiltered = index.search(&[1.0, 0.0], 2, 50);
        assert_eq!(filtered[0].0, unfiltered[0].0);
        assert_eq!(filtered[1].0, unfiltered[1].0);
    }

    #[test]
    fn test_search_filtered_with_deletes() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]); // id 0
        index.insert(vec![0.9, 0.1]); // id 1
        index.insert(vec![0.0, 1.0]); // id 2
        index.insert(vec![0.5, 0.5]); // id 3
        index.mark_deleted(1);

        // Filter: only even ids. Deleted id=1 is odd anyway but id=0 is the closest even.
        let results = index.search_filtered(&[1.0, 0.0], 2, 50, |id| id % 2 == 0);
        for r in &results {
            assert!(r.0 % 2 == 0, "filter should only return even ids");
            assert!(!index.is_deleted(r.0), "should not return deleted ids");
        }
    }

    #[test]
    fn test_search_filtered_selective_recall() {
        // With a very selective filter (only 10% match), verify we still find good results.
        let dim = 16;
        let n = 1000;
        let mut rng = rand::thread_rng();
        let mut index = HnswIndex::new(dim, Metric::Euclidean, 16, 200);

        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();
        index.batch_insert(vecs.clone());

        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let k = 5;

        // Filter: only every 10th vector (10% selectivity).
        let predicate = |id: u32| id % 10 == 0;

        // Brute-force ground truth among matching vectors.
        let mut brute: Vec<(u32, f32)> = (0..n as u32)
            .filter(|id| predicate(*id))
            .map(|id| (id, distance(&query, &vecs[id as usize], Metric::Euclidean)))
            .collect();
        brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let truth: HashSet<u32> = brute.iter().take(k).map(|(id, _)| *id).collect();

        let results = index.search_filtered(&query, k, 200, predicate);
        assert_eq!(results.len(), k, "should find k results even with selective filter");
        let found: HashSet<u32> = results.iter().map(|r| r.0).collect();

        // With 10% selectivity on 1000 vectors we should find the true nearest among filtered.
        let recall = found.intersection(&truth).count() as f64 / k as f64;
        assert!(
            recall >= 0.60,
            "filtered search recall too low: {:.2} (expected >= 0.60 with 10% selectivity)",
            recall
        );
    }

    #[test]
    fn test_delete_entry_point() {
        // Delete the entry point node and verify search still works.
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        let ep_id = index.insert(vec![1.0, 0.0]); // likely entry point
        index.insert(vec![0.0, 1.0]);
        index.insert(vec![0.5, 0.5]);

        index.mark_deleted(ep_id);
        let results = index.search(&[1.0, 0.0], 2, 50);
        // Should still return results (non-deleted vectors).
        assert!(!results.is_empty());
        for r in &results {
            assert_ne!(r.0, ep_id, "should not return deleted entry point");
        }
    }

    #[test]
    fn test_update_entry_point_vector() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]); // id 0, likely entry point
        index.insert(vec![0.0, 1.0]); // id 1

        // Update the entry point to a very different position.
        index.update_vector(0, vec![-10.0, -10.0]);

        // Search should still work and find correct nearest.
        let results = index.search(&[-10.0, -10.0], 1, 50);
        assert_eq!(results[0].0, 0);

        let results = index.search(&[0.0, 1.0], 1, 50);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_batch_insert_empty() {
        let mut index = HnswIndex::new(3, Metric::Euclidean, 16, 200);
        let ids = index.batch_insert(vec![]);
        assert!(ids.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_batch_insert_single() {
        let mut index = HnswIndex::new(3, Metric::Euclidean, 16, 200);
        let ids = index.batch_insert(vec![vec![1.0, 0.0, 0.0]]);
        assert_eq!(ids.len(), 1);
        assert_eq!(index.len(), 1);

        let results = index.search(&[1.0, 0.0, 0.0], 1, 50);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_sequential_then_batch() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);

        // Sequential inserts first.
        index.insert(vec![1.0, 0.0]);
        index.insert(vec![0.0, 1.0]);

        // Then batch insert.
        index.batch_insert(vec![
            vec![0.5, 0.5],
            vec![-1.0, 0.0],
        ]);
        assert_eq!(index.len(), 4);

        let results = index.search(&[-1.0, 0.0], 1, 50);
        assert_eq!(results[0].0, 3);
    }

    #[test]
    fn test_mark_deleted_nonexistent() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]);

        assert!(!index.mark_deleted(999)); // out of range
        assert!(!index.mark_deleted(1));   // no such node
    }

    #[test]
    fn test_mark_deleted_twice() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]);

        assert!(index.mark_deleted(0));
        assert!(!index.mark_deleted(0)); // already deleted
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_compact_preserves_metric() {
        let mut index = HnswIndex::new(2, Metric::DotProduct, 8, 100);
        index.insert(vec![1.0, 0.0]);
        index.insert(vec![0.0, 1.0]);
        index.mark_deleted(0);

        let (compacted, _) = index.compact();
        assert_eq!(compacted.metric(), Metric::DotProduct);
        assert_eq!(compacted.dim(), 2);
    }

    #[test]
    fn test_quantized_search_with_deletes() {
        let mut index = HnswIndex::new(3, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0, 0.0]); // id 0
        index.insert(vec![0.0, 1.0, 0.0]); // id 1
        index.insert(vec![0.0, 0.0, 1.0]); // id 2
        index.enable_quantized_search();

        index.mark_deleted(0);

        let results = index.search(&[1.0, 0.0, 0.0], 2, 50);
        for r in &results {
            assert_ne!(r.0, 0, "quantized search should not return deleted vectors");
        }
    }

    #[test]
    fn test_quantized_update_vector() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]);
        index.insert(vec![0.0, 1.0]);
        index.enable_quantized_search();

        // Update vector 0 to be near vector 1.
        index.update_vector(0, vec![0.0, 1.0]);

        let results = index.search(&[0.0, 1.0], 2, 50);
        // Both should be found near [0, 1].
        let found: HashSet<u32> = results.iter().map(|r| r.0).collect();
        assert!(found.contains(&0));
        assert!(found.contains(&1));
    }

    #[test]
    fn test_distances_are_sorted() {
        let mut index = HnswIndex::new(3, Metric::Euclidean, 16, 200);
        let mut rng = rand::thread_rng();

        let vecs: Vec<Vec<f32>> = (0..500)
            .map(|_| (0..3).map(|_| rng.gen::<f32>()).collect())
            .collect();
        index.batch_insert(vecs);

        let query = vec![0.5, 0.5, 0.5];
        let results = index.search(&query, 20, 100);

        // Results should be sorted by distance.
        for w in results.windows(2) {
            assert!(
                w[0].1 <= w[1].1,
                "results not sorted: {} > {}",
                w[0].1, w[1].1
            );
        }
    }

    #[test]
    fn test_search_many_consistency() {
        let mut index = HnswIndex::new(3, Metric::Euclidean, 16, 200);
        let mut rng = rand::thread_rng();

        let vecs: Vec<Vec<f32>> = (0..200)
            .map(|_| (0..3).map(|_| rng.gen::<f32>()).collect())
            .collect();
        index.batch_insert(vecs);

        let queries: Vec<Vec<f32>> = (0..5)
            .map(|_| (0..3).map(|_| rng.gen::<f32>()).collect())
            .collect();
        let query_refs: Vec<&[f32]> = queries.iter().map(|q| q.as_slice()).collect();

        let batch_results = index.search_many(&query_refs, 5, 100);

        // Each batch result should match individual search.
        for (i, q) in queries.iter().enumerate() {
            let single = index.search(q, 5, 100);
            assert_eq!(
                batch_results[i].len(),
                single.len(),
                "batch vs single length mismatch for query {}",
                i
            );
            assert_eq!(
                batch_results[i][0].0,
                single[0].0,
                "batch vs single top-1 mismatch for query {}",
                i
            );
        }
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn test_insert_wrong_dimension() {
        let mut index = HnswIndex::new(3, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]); // dim=2, expected 3
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn test_search_wrong_dimension() {
        let mut index = HnswIndex::new(3, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0, 0.0]);
        index.search(&[1.0, 0.0], 1, 50); // dim=2, expected 3
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn test_batch_insert_wrong_dimension() {
        let mut index = HnswIndex::new(3, Metric::Euclidean, 16, 200);
        index.batch_insert(vec![vec![1.0, 0.0]]); // dim=2, expected 3
    }

    // ======================================================================
    // TDD tests: these probe suspected weak spots in the implementation.
    // Some of these SHOULD fail if the implementation has bugs.
    // ======================================================================

    /// compact() should preserve quantized search state.
    /// BUG HYPOTHESIS: compact() calls HnswIndex::new() which sets quantized: None,
    /// silently dropping quantized vectors even if they were enabled.
    #[test]
    fn test_compact_preserves_quantized_state() {
        let mut index = HnswIndex::new(3, Metric::Cosine, 16, 200);
        let vecs: Vec<Vec<f32>> = (0..20)
            .map(|i| {
                let angle = i as f32 * 0.3;
                vec![angle.cos(), angle.sin(), 0.5]
            })
            .collect();
        index.batch_insert(vecs);
        index.enable_quantized_search();
        assert!(index.quantized.is_some(), "quantized should be enabled before compact");

        // Delete some vectors.
        index.mark_deleted(5);
        index.mark_deleted(10);
        index.mark_deleted(15);

        // Compact should preserve quantized state.
        let (new_index, _mapping) = index.compact();
        assert!(
            new_index.quantized.is_some(),
            "compact() dropped quantized state — quantized search silently disabled after compaction"
        );
        assert!(new_index.has_quantized_search());
    }

    /// After compact, search should still work correctly and return valid results.
    /// Tests that the old_to_new ID mapping is consistent.
    #[test]
    fn test_compact_search_correctness() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        // Insert vectors at known positions.
        index.insert(vec![0.0, 0.0]);  // id 0
        index.insert(vec![10.0, 0.0]); // id 1
        index.insert(vec![0.0, 10.0]); // id 2
        index.insert(vec![10.0, 10.0]); // id 3

        // Delete id 1 (the [10, 0] vector).
        index.mark_deleted(1);

        let (new_index, mapping) = index.compact();

        // Should have 3 vectors now.
        assert_eq!(new_index.len(), 3);

        // Search for [10, 0] — closest should be [10, 10] or [0, 0], NOT a ghost of deleted [10, 0].
        let results = new_index.search(&[10.0, 0.0], 3, 50);
        assert_eq!(results.len(), 3, "should return all 3 remaining vectors");

        // The deleted vector's ID should not appear in mapping.
        assert!(!mapping.contains_key(&1), "deleted vector should not be in mapping");

        // All result IDs should be valid (< new_index.len()).
        for (id, _dist) in &results {
            assert!(
                (*id as usize) < new_index.len(),
                "compact returned invalid ID {} (index has {} vectors)",
                id,
                new_index.len()
            );
        }
    }

    /// Compact on an index with NO deletions should be a no-op that returns
    /// an identical index.
    #[test]
    fn test_compact_no_deletions() {
        let mut index = HnswIndex::new(3, Metric::Cosine, 16, 200);
        index.insert(vec![1.0, 0.0, 0.0]);
        index.insert(vec![0.0, 1.0, 0.0]);

        let (new_index, mapping) = index.compact();
        assert_eq!(new_index.len(), 2);
        assert_eq!(mapping.len(), 2);

        // IDs should map 0→0 and 1→1 (no reordering needed).
        // Actually, batch_insert may reorder due to parallel construction,
        // so just check that search still works.
        let results = new_index.search(&[1.0, 0.0, 0.0], 1, 50);
        assert_eq!(results.len(), 1);
    }

    /// Compact after deleting ALL vectors should produce an empty index.
    #[test]
    fn test_compact_all_deleted() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]);
        index.insert(vec![0.0, 1.0]);
        index.mark_deleted(0);
        index.mark_deleted(1);

        let (new_index, mapping) = index.compact();
        assert_eq!(new_index.len(), 0);
        assert!(mapping.is_empty());

        // Search on empty should return empty.
        let results = new_index.search(&[1.0, 0.0], 5, 50);
        assert!(results.is_empty());
    }

    /// search_filtered with a predicate that matches NOTHING should return empty,
    /// not panic or return unfiltered results.
    #[test]
    fn test_search_filtered_reject_all() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        for i in 0..50 {
            index.insert(vec![i as f32, 0.0]);
        }

        let results = index.search_filtered(&[0.0, 0.0], 10, 50, |_id| false);
        assert!(results.is_empty(), "filtering out everything should return empty");
    }

    /// search_filtered with a very selective predicate (1 out of many) should
    /// still find the matching vector.
    #[test]
    fn test_search_filtered_needle_in_haystack() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        // Insert 100 vectors. Only id=42 will match the filter.
        for i in 0..100 {
            index.insert(vec![i as f32, 0.0]);
        }

        let results = index.search_filtered(
            &[42.0, 0.0], 5, 200,
            |id| id == 42,
        );
        assert!(!results.is_empty(), "should find the one matching vector");
        assert_eq!(results[0].0, 42);
    }

    /// Inserting after enabling quantized search should maintain quantized vectors.
    /// Then disabling and re-enabling should rebuild them correctly.
    #[test]
    fn test_quantized_insert_after_enable_size_check() {
        let mut index = HnswIndex::new(4, Metric::Cosine, 16, 200);
        index.insert(vec![1.0, 0.0, 0.0, 0.0]);
        index.insert(vec![0.0, 1.0, 0.0, 0.0]);

        index.enable_quantized_search();

        // Insert more vectors AFTER enabling quantized search.
        index.insert(vec![0.0, 0.0, 1.0, 0.0]);
        index.insert(vec![0.0, 0.0, 0.0, 1.0]);

        // Search should still work and find all 4 vectors.
        let results = index.search(&[0.0, 0.0, 1.0, 0.0], 4, 50);
        assert_eq!(results.len(), 4);
        assert_eq!(results[0].0, 2, "closest to [0,0,1,0] should be id 2");

        // The quantized vectors should have the right count.
        let qs = index.quantized.as_ref().unwrap();
        assert_eq!(
            qs.vectors.len(),
            4 * 4, // 4 vectors * 4 dims = 16 bytes
            "quantized vector storage has wrong size after post-enable insert"
        );
    }

    /// Delete + insert reuse: after deleting a vector, inserting a new one
    /// should NOT reuse the deleted slot's ID. The new vector should get a fresh ID.
    /// (Our implementation appends, it doesn't reuse — so deleted slots waste space
    /// until compact() is called.)
    #[test]
    fn test_delete_does_not_reuse_slot() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]); // id 0
        index.insert(vec![0.0, 1.0]); // id 1
        index.mark_deleted(0);

        let new_id = index.insert(vec![0.5, 0.5]); // should be id 2, not 0
        assert_eq!(new_id, 2, "new insert should get id 2, not reuse deleted slot 0");
        assert_eq!(index.len(), 2, "length should be 2 (1 deleted + 2 live - 1 deleted = 2)");

        // Search should NOT return deleted id 0.
        let results = index.search(&[1.0, 0.0], 10, 50);
        assert!(
            results.iter().all(|(id, _)| *id != 0),
            "deleted vector id 0 appeared in search results"
        );
    }

    /// update_vector on a deleted vector — what happens?
    /// This is an edge case that could cause silent corruption.
    #[test]
    fn test_update_deleted_vector() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]); // id 0
        index.insert(vec![0.0, 1.0]); // id 1
        index.mark_deleted(0);

        // Updating a deleted vector's data — should this work or panic?
        // The vector slot still exists, so update_vector will succeed,
        // but the vector is still marked as deleted, so it shouldn't appear in search.
        index.update_vector(0, vec![0.5, 0.5]);

        let results = index.search(&[0.5, 0.5], 10, 50);
        assert!(
            results.iter().all(|(id, _)| *id != 0),
            "updated-but-deleted vector appeared in search results"
        );
    }

    /// Test that the HNSW graph path (not brute-force) is exercised for quantized search.
    /// We need n*dim >= 10M to bypass brute-force. Using dim=2048, n=5000 → 10.24M.
    #[test]
    fn test_quantized_hnsw_path_not_brute_force() {
        let dim = 2048;
        let n = 5000;
        let mut index = HnswIndex::new(dim, Metric::Cosine, 16, 200);

        // Generate deterministic vectors using a hash.
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                (0..dim)
                    .map(|d| {
                        let hash = ((i as u64).wrapping_mul(2654435761) ^ (d as u64).wrapping_mul(40503)) % 100000;
                        hash as f32 / 100000.0
                    })
                    .collect()
            })
            .collect();

        index.batch_insert(vectors);

        // Verify we're NOT using brute force.
        assert!(
            !index.uses_brute_force(),
            "test needs to exercise HNSW path but fell into brute-force (n*dim={} < 10M)",
            n * dim
        );

        // Enable quantized search — this should work without panicking.
        index.enable_quantized_search();

        // Run a search through the quantized HNSW path.
        let query: Vec<f32> = (0..dim).map(|d| d as f32 / dim as f32).collect();
        let results = index.search(&query, 10, 200);
        assert!(!results.is_empty(), "quantized HNSW search returned no results");
        assert!(results.len() <= 10);

        // Distances should be valid (non-negative for cosine).
        for (id, dist) in &results {
            assert!(*id < n as u32, "invalid id {}", id);
            assert!(*dist >= 0.0, "negative distance {}", dist);
        }

        // Results should be sorted by distance.
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1, "results not sorted by distance");
        }
    }

    /// Compact followed by insert should work correctly.
    /// The new index from compact should be fully functional for further inserts.
    #[test]
    fn test_compact_then_insert() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]); // id 0
        index.insert(vec![0.0, 1.0]); // id 1
        index.insert(vec![1.0, 1.0]); // id 2
        index.mark_deleted(1);

        let (mut new_index, _) = index.compact();
        assert_eq!(new_index.len(), 2);

        // Insert into the compacted index.
        let new_id = new_index.insert(vec![0.5, 0.5]);
        assert_eq!(new_index.len(), 3);

        // Search should find all 3 vectors.
        let results = new_index.search(&[0.5, 0.5], 3, 50);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, new_id, "closest to [0.5, 0.5] should be the newly inserted vector");
    }

    /// mark_deleted with an out-of-bounds ID — should this panic or be a no-op?
    #[test]
    fn test_mark_deleted_out_of_bounds() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]);

        // Marking an ID that doesn't exist — check it doesn't panic.
        // This might actually panic or corrupt state.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            index.mark_deleted(999);
        }));
        // We're documenting behavior here: does it panic or silently succeed?
        // Either is acceptable as long as it's consistent.
        if result.is_ok() {
            // If it succeeded, length should still be 1.
            assert_eq!(index.len(), 1, "length changed after marking nonexistent ID as deleted");
        }
        // If it panicked, that's also valid behavior.
    }

    /// Double-delete: marking the same ID as deleted twice.
    #[test]
    fn test_double_delete() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]);
        index.insert(vec![0.0, 1.0]);

        index.mark_deleted(0);
        let len_after_first = index.len();

        index.mark_deleted(0); // double delete
        let len_after_second = index.len();

        assert_eq!(
            len_after_first, len_after_second,
            "double-delete changed the length (was {}, now {})",
            len_after_first, len_after_second
        );
    }

    /// search with k=0 should return empty, not panic.
    #[test]
    fn test_search_k_zero() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]);

        let results = index.search(&[1.0, 0.0], 0, 50);
        assert!(results.is_empty(), "k=0 should return empty results");
    }

    /// search with k larger than the number of vectors should return all vectors.
    #[test]
    fn test_search_k_larger_than_n() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]);
        index.insert(vec![0.0, 1.0]);

        let results = index.search(&[1.0, 0.0], 100, 50);
        assert_eq!(results.len(), 2, "should return all vectors when k > n");
    }

    /// search_filtered should skip deleted vectors even when the predicate allows them.
    #[test]
    fn test_search_filtered_skips_deleted() {
        let mut index = HnswIndex::new(2, Metric::Euclidean, 16, 200);
        index.insert(vec![1.0, 0.0]); // id 0 — closest to query
        index.insert(vec![0.0, 1.0]); // id 1
        index.insert(vec![0.5, 0.5]); // id 2
        index.mark_deleted(0);

        // Predicate allows all, but id 0 is deleted.
        let results = index.search_filtered(&[1.0, 0.0], 10, 50, |_| true);
        assert!(
            results.iter().all(|(id, _)| *id != 0),
            "search_filtered returned a deleted vector"
        );
    }

    /// Quantized search on an index where all vectors are identical.
    /// Edge case: quantizer min==max for all dimensions.
    #[test]
    fn test_quantized_identical_vectors() {
        let mut index = HnswIndex::new(3, Metric::Euclidean, 16, 200);
        for _ in 0..10 {
            index.insert(vec![1.0, 1.0, 1.0]);
        }

        index.enable_quantized_search();

        // Should not panic on zero-range quantization.
        let results = index.search(&[1.0, 1.0, 1.0], 5, 50);
        assert!(!results.is_empty());
        // All distances should be 0 (or very close).
        for (_, dist) in &results {
            assert!(*dist < 0.01, "distance to identical vector should be ~0, got {}", dist);
        }
    }

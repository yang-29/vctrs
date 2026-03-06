/// HNSW (Hierarchical Navigable Small World) index.
///
/// This is the algorithm that makes vector search fast — O(log n) instead of O(n).
/// It builds a multi-layer graph where each layer is progressively sparser.
/// Search starts at the top (sparse) layer and greedily descends to find neighbors.

use crate::distance::{distance, Metric};
use rand::Rng;
use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;

/// A neighbor candidate with its distance (used in priority queues).
#[derive(Clone, Debug)]
struct Candidate {
    id: u32,
    dist: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Candidate {}

// Min-heap ordering (smallest distance first).
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Reverse ordering for max-heap behavior.
#[derive(Clone, Debug)]
struct RevCandidate {
    id: u32,
    dist: f32,
}

impl PartialEq for RevCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RevCandidate {}

impl Ord for RevCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for RevCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A single node in the HNSW graph.
struct Node {
    /// Neighbors at each layer. neighbors[layer] = vec of neighbor node ids.
    neighbors: Vec<Vec<u32>>,
}

/// The HNSW index.
pub struct HnswIndex {
    /// All vectors stored in the index, indexed by internal id.
    vectors: Vec<Vec<f32>>,
    /// Graph nodes with neighbor lists per layer.
    nodes: Vec<Node>,
    /// Entry point node id (top of the graph).
    entry_point: Option<u32>,
    /// Maximum layer of the entry point.
    max_layer: usize,
    /// Distance metric to use.
    metric: Metric,
    /// Max neighbors per node per layer.
    m: usize,
    /// Max neighbors for layer 0 (typically 2*m).
    m0: usize,
    /// Size of the dynamic candidate list during construction.
    ef_construction: usize,
    /// Normalization factor for level generation: 1/ln(m).
    ml: f64,
    /// Vector dimensionality.
    dim: usize,
}

impl HnswIndex {
    pub fn new(dim: usize, metric: Metric, m: usize, ef_construction: usize) -> Self {
        let ml = 1.0 / (m as f64).ln();
        HnswIndex {
            vectors: Vec::new(),
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
            metric,
            m,
            m0: m * 2,
            ef_construction,
            ml,
            dim,
        }
    }

    /// Randomly assign a layer for a new node.
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        (-r.ln() * self.ml).floor() as usize
    }

    /// Insert a vector into the index. Returns the internal id.
    pub fn insert(&mut self, vector: Vec<f32>) -> u32 {
        assert_eq!(vector.len(), self.dim, "vector dimension mismatch");

        let id = self.vectors.len() as u32;
        let level = self.random_level();

        // Create node with empty neighbor lists for each layer.
        let node = Node {
            neighbors: vec![Vec::new(); level + 1],
        };
        self.vectors.push(vector);
        self.nodes.push(node);

        // First node — just set as entry point.
        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.max_layer = level;
            return id;
        }

        let ep = self.entry_point.unwrap();
        let mut current_ep = ep;

        // Phase 1: Greedily traverse from top layer down to the node's layer + 1.
        // At these upper layers we just find the single closest node.
        if self.max_layer > level {
            for l in ((level + 1)..=self.max_layer).rev() {
                current_ep = self.search_layer_single(&self.vectors[id as usize], current_ep, l);
            }
        }

        // Phase 2: From the node's layer down to layer 0, find ef_construction
        // nearest neighbors and connect them.
        for l in (0..=level.min(self.max_layer)).rev() {
            let max_neighbors = if l == 0 { self.m0 } else { self.m };
            let neighbors = self.search_layer(
                &self.vectors[id as usize],
                current_ep,
                self.ef_construction,
                l,
            );

            // Select the closest `max_neighbors` from the candidates.
            let selected: Vec<u32> = neighbors
                .iter()
                .take(max_neighbors)
                .map(|c| c.id)
                .collect();

            // Set this node's neighbors at layer l.
            self.nodes[id as usize].neighbors[l] = selected.clone();

            // Add bidirectional connections.
            for &neighbor_id in &selected {
                let neighbor = &mut self.nodes[neighbor_id as usize];
                if l < neighbor.neighbors.len() {
                    neighbor.neighbors[l].push(id);
                    // Prune if over capacity.
                    if neighbor.neighbors[l].len() > max_neighbors {
                        let nv = &self.vectors[neighbor_id as usize];
                        let mut scored: Vec<(f32, u32)> = neighbor.neighbors[l]
                            .iter()
                            .map(|&nid| {
                                (distance(nv, &self.vectors[nid as usize], self.metric), nid)
                            })
                            .collect();
                        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
                        neighbor.neighbors[l] =
                            scored.into_iter().take(max_neighbors).map(|(_, nid)| nid).collect();
                    }
                }
            }

            // Update entry point for next layer down.
            if let Some(closest) = neighbors.first() {
                current_ep = closest.id;
            }
        }

        // If this node's level is higher than the current max, update entry point.
        if level > self.max_layer {
            self.entry_point = Some(id);
            self.max_layer = level;
        }

        id
    }

    /// Greedy search for single nearest node at a given layer.
    fn search_layer_single(&self, query: &[f32], entry: u32, layer: usize) -> u32 {
        let mut current = entry;
        let mut current_dist = distance(query, &self.vectors[current as usize], self.metric);

        loop {
            let mut changed = false;
            let node = &self.nodes[current as usize];
            if layer < node.neighbors.len() {
                for &neighbor_id in &node.neighbors[layer] {
                    let d = distance(query, &self.vectors[neighbor_id as usize], self.metric);
                    if d < current_dist {
                        current = neighbor_id;
                        current_dist = d;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    /// Search a layer for the ef nearest neighbors to query, starting from entry.
    /// Returns candidates sorted by distance (closest first).
    fn search_layer(&self, query: &[f32], entry: u32, ef: usize, layer: usize) -> Vec<Candidate> {
        let entry_dist = distance(query, &self.vectors[entry as usize], self.metric);

        // Min-heap of candidates to explore.
        let mut candidates = BinaryHeap::new();
        candidates.push(Candidate {
            id: entry,
            dist: entry_dist,
        });

        // Max-heap of current best results.
        let mut results = BinaryHeap::new();
        results.push(RevCandidate {
            id: entry,
            dist: entry_dist,
        });

        let mut visited = HashSet::new();
        visited.insert(entry);

        while let Some(current) = candidates.pop() {
            // If the closest candidate is farther than the farthest result, stop.
            let farthest_dist = results.peek().map(|r| r.dist).unwrap_or(f32::MAX);
            if current.dist > farthest_dist {
                break;
            }

            let node = &self.nodes[current.id as usize];
            if layer >= node.neighbors.len() {
                continue;
            }

            for &neighbor_id in &node.neighbors[layer] {
                if visited.contains(&neighbor_id) {
                    continue;
                }
                visited.insert(neighbor_id);

                let d = distance(query, &self.vectors[neighbor_id as usize], self.metric);
                let farthest_dist = results.peek().map(|r| r.dist).unwrap_or(f32::MAX);

                if d < farthest_dist || results.len() < ef {
                    candidates.push(Candidate {
                        id: neighbor_id,
                        dist: d,
                    });
                    results.push(RevCandidate {
                        id: neighbor_id,
                        dist: d,
                    });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        // Drain the max-heap and sort by distance.
        let mut result_vec: Vec<Candidate> = results
            .into_iter()
            .map(|r| Candidate {
                id: r.id,
                dist: r.dist,
            })
            .collect();
        result_vec.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        result_vec
    }

    /// Search for the k nearest neighbors to the query vector.
    /// Returns (internal_id, distance) pairs sorted by distance.
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(u32, f32)> {
        assert_eq!(query.len(), self.dim, "query dimension mismatch");

        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let ef = ef_search.max(k);

        // Traverse upper layers greedily.
        let mut current_ep = ep;
        if self.max_layer > 0 {
            for l in (1..=self.max_layer).rev() {
                current_ep = self.search_layer_single(query, current_ep, l);
            }
        }

        // Search layer 0 with ef candidates.
        let candidates = self.search_layer(query, current_ep, ef, 0);

        candidates
            .into_iter()
            .take(k)
            .map(|c| (c.id, c.dist))
            .collect()
    }

    /// Get a vector by internal id.
    pub fn get_vector(&self, id: u32) -> Option<&[f32]> {
        self.vectors.get(id as usize).map(|v| v.as_slice())
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
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
        // Closest should be [1,0,0] (id=0) or [1,1,0] (id=3).
        assert!(results[0].0 == 0 || results[0].0 == 3);
    }

    #[test]
    fn test_cosine_search() {
        let mut index = HnswIndex::new(2, Metric::Cosine, 16, 200);

        index.insert(vec![1.0, 0.0]);
        index.insert(vec![0.0, 1.0]);
        index.insert(vec![0.707, 0.707]);

        // Query close to [1, 0] direction.
        let results = index.search(&[0.9, 0.1], 1, 50);
        assert_eq!(results[0].0, 0); // [1, 0] is closest in cosine.
    }

    #[test]
    fn test_empty_index() {
        let index = HnswIndex::new(4, Metric::Cosine, 16, 200);
        let results = index.search(&[1.0, 2.0, 3.0, 4.0], 5, 50);
        assert!(results.is_empty());
    }

    #[test]
    fn test_recall() {
        // Insert 1000 random vectors, verify we find the actual nearest neighbor.
        let dim = 32;
        let mut rng = rand::thread_rng();
        let mut index = HnswIndex::new(dim, Metric::Euclidean, 16, 200);

        let mut vecs = Vec::new();
        for _ in 0..1000 {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            index.insert(v.clone());
            vecs.push(v);
        }

        // Brute-force find the actual nearest neighbor to a query.
        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let mut brute_force: Vec<(usize, f32)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i, distance(&query, v, Metric::Euclidean)))
            .collect();
        brute_force.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let true_nearest = brute_force[0].0 as u32;

        // HNSW should find it (or very close).
        let results = index.search(&query, 10, 100);
        let found_ids: Vec<u32> = results.iter().map(|r| r.0).collect();
        assert!(
            found_ids.contains(&true_nearest),
            "HNSW missed the true nearest neighbor"
        );
    }
}

/// HNSW (Hierarchical Navigable Small World) index.
///
/// This is the algorithm that makes vector search fast — O(log n) instead of O(n).
/// It builds a multi-layer graph where each layer is progressively sparser.
/// Search starts at the top (sparse) layer and greedily descends to find neighbors.
///
/// Supports concurrent batch inserts via per-node Mutex locking (hnswlib-style).

use crate::distance::{distance, Metric};
use parking_lot::Mutex;
use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::io::{self, Read as IoRead, Write as IoWrite};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering as AtomicOrdering};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

// -- Priority queue helpers --------------------------------------------------

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

// -- Node with per-layer Mutex -----------------------------------------------

struct Node {
    /// Neighbors at each layer, each protected by a Mutex for concurrent access.
    neighbors: Vec<Mutex<Vec<u32>>>,
}

impl Node {
    fn new(levels: usize) -> Self {
        Node {
            neighbors: (0..levels).map(|_| Mutex::new(Vec::new())).collect(),
        }
    }

    fn num_layers(&self) -> usize {
        self.neighbors.len()
    }
}

// -- HNSW Index --------------------------------------------------------------

pub struct HnswIndex {
    vectors: Vec<Vec<f32>>,
    nodes: Vec<Node>,
    deleted: HashSet<u32>,
    entry_point: AtomicU32,
    max_layer: AtomicUsize,
    metric: Metric,
    m: usize,
    m0: usize,
    ef_construction: usize,
    ml: f64,
    dim: usize,
    active_count: AtomicUsize,
}

// Entry point sentinel: no entry point set yet.
const NO_ENTRY: u32 = u32::MAX;

impl HnswIndex {
    pub fn new(dim: usize, metric: Metric, m: usize, ef_construction: usize) -> Self {
        let ml = 1.0 / (m as f64).ln();
        HnswIndex {
            vectors: Vec::new(),
            nodes: Vec::new(),
            deleted: HashSet::new(),
            entry_point: AtomicU32::new(NO_ENTRY),
            max_layer: AtomicUsize::new(0),
            metric,
            m,
            m0: m * 2,
            ef_construction,
            ml,
            dim,
            active_count: AtomicUsize::new(0),
        }
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        (-r.ln() * self.ml).floor() as usize
    }

    /// Insert a single vector (sequential). Returns the internal id.
    pub fn insert(&mut self, vector: Vec<f32>) -> u32 {
        assert_eq!(vector.len(), self.dim, "vector dimension mismatch");

        let id = self.vectors.len() as u32;
        let level = self.random_level();

        self.vectors.push(vector);
        self.nodes.push(Node::new(level + 1));
        self.active_count.fetch_add(1, AtomicOrdering::Relaxed);

        let ep = self.entry_point.load(AtomicOrdering::Relaxed);
        if ep == NO_ENTRY {
            self.entry_point.store(id, AtomicOrdering::Relaxed);
            self.max_layer.store(level, AtomicOrdering::Relaxed);
            return id;
        }

        self.connect_node(id, level, ep);

        if level > self.max_layer.load(AtomicOrdering::Relaxed) {
            self.entry_point.store(id, AtomicOrdering::Relaxed);
            self.max_layer.store(level, AtomicOrdering::Relaxed);
        }

        id
    }

    /// Batch insert with parallel graph construction using rayon.
    /// Pre-allocates all slots, then connects nodes concurrently.
    pub fn batch_insert(&mut self, vectors: Vec<Vec<f32>>) -> Vec<u32> {
        if vectors.is_empty() {
            return Vec::new();
        }
        for v in &vectors {
            assert_eq!(v.len(), self.dim, "vector dimension mismatch");
        }

        let start_id = self.vectors.len() as u32;
        let n = vectors.len();

        // Pre-compute levels.
        let levels: Vec<usize> = (0..n).map(|_| self.random_level()).collect();

        // Pre-allocate all slots so concurrent reads are safe.
        for (i, vec) in vectors.into_iter().enumerate() {
            self.vectors.push(vec);
            self.nodes.push(Node::new(levels[i] + 1));
        }

        let ids: Vec<u32> = (start_id..start_id + n as u32).collect();

        // If this is the first batch, seed the entry point with the first vector.
        let ep = self.entry_point.load(AtomicOrdering::Relaxed);
        if ep == NO_ENTRY {
            self.entry_point.store(start_id, AtomicOrdering::Relaxed);
            self.max_layer.store(levels[0], AtomicOrdering::Relaxed);
        }

        // Insert a small seed sequentially first (first 128 or all if small batch)
        // so the graph has structure for parallel threads to navigate.
        let seed_count = n.min(128);
        let seed_start = if ep == NO_ENTRY { 1 } else { 0 }; // skip first if it's the entry point
        for i in seed_start..seed_count {
            let id = start_id + i as u32;
            let current_ep = self.entry_point.load(AtomicOrdering::Relaxed);
            self.connect_node(id, levels[i], current_ep);
            if levels[i] > self.max_layer.load(AtomicOrdering::Relaxed) {
                self.entry_point.store(id, AtomicOrdering::Relaxed);
                self.max_layer.store(levels[i], AtomicOrdering::Relaxed);
            }
        }

        // Insert remaining nodes in parallel.
        if seed_count < n {
            let remaining: Vec<(u32, usize)> = (seed_count..n)
                .map(|i| (start_id + i as u32, levels[i]))
                .collect();

            // Safety: vectors and nodes are pre-allocated and won't resize.
            // Each node's neighbors are Mutex-protected. entry_point/max_layer are atomic.
            // We reborrow as &Self which is Sync.
            let this: &Self = unsafe { &*(self as *const Self) };
            remaining.par_iter().for_each(|&(id, level)| {
                let current_ep = this.entry_point.load(AtomicOrdering::Relaxed);
                this.connect_node_concurrent(id, level, current_ep);
            });

            // Update entry point to highest-level node from the parallel batch.
            for i in seed_count..n {
                if levels[i] > self.max_layer.load(AtomicOrdering::Relaxed) {
                    self.entry_point
                        .store(start_id + i as u32, AtomicOrdering::Relaxed);
                    self.max_layer.store(levels[i], AtomicOrdering::Relaxed);
                }
            }
        }

        self.active_count.fetch_add(n, AtomicOrdering::Relaxed);
        ids
    }

    /// Connect a node to its neighbors at all layers (sequential version).
    fn connect_node(&self, id: u32, level: usize, ep: u32) {
        let max_layer = self.max_layer.load(AtomicOrdering::Relaxed);
        let mut current_ep = ep;

        // Phase 1: Greedy descent through upper layers.
        if max_layer > level {
            for l in ((level + 1)..=max_layer).rev() {
                current_ep = self.search_layer_single(&self.vectors[id as usize], current_ep, l);
            }
        }

        // Phase 2: Search and connect at each layer.
        for l in (0..=level.min(max_layer)).rev() {
            let max_neighbors = if l == 0 { self.m0 } else { self.m };
            let neighbors =
                self.search_layer(&self.vectors[id as usize], current_ep, self.ef_construction, l);

            let selected: Vec<u32> = neighbors.iter().take(max_neighbors).map(|c| c.id).collect();

            // Set this node's neighbors.
            *self.nodes[id as usize].neighbors[l].lock() = selected.clone();

            // Add bidirectional connections.
            for &neighbor_id in &selected {
                let neighbor = &self.nodes[neighbor_id as usize];
                if l < neighbor.num_layers() {
                    let mut nb = neighbor.neighbors[l].lock();
                    nb.push(id);
                    if nb.len() > max_neighbors {
                        self.prune_neighbors(&mut nb, neighbor_id, max_neighbors);
                    }
                }
            }

            if let Some(closest) = neighbors.first() {
                current_ep = closest.id;
            }
        }
    }

    /// Connect a node to its neighbors (concurrent-safe version, uses only & self).
    fn connect_node_concurrent(&self, id: u32, level: usize, ep: u32) {
        // Same logic as connect_node — works because neighbors are Mutex-protected.
        self.connect_node(id, level, ep);
    }

    /// Prune a neighbor list to keep only the closest `max_neighbors`.
    fn prune_neighbors(&self, neighbors: &mut Vec<u32>, node_id: u32, max_neighbors: usize) {
        let nv = &self.vectors[node_id as usize];
        let mut scored: Vec<(f32, u32)> = neighbors
            .iter()
            .map(|&nid| (distance(nv, &self.vectors[nid as usize], self.metric), nid))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        *neighbors = scored
            .into_iter()
            .take(max_neighbors)
            .map(|(_, nid)| nid)
            .collect();
    }

    pub fn mark_deleted(&mut self, id: u32) -> bool {
        if (id as usize) < self.vectors.len() && !self.deleted.contains(&id) {
            self.deleted.insert(id);
            self.active_count.fetch_sub(1, AtomicOrdering::Relaxed);
            true
        } else {
            false
        }
    }

    pub fn is_deleted(&self, id: u32) -> bool {
        self.deleted.contains(&id)
    }

    pub fn update_vector(&mut self, id: u32, vector: Vec<f32>) {
        assert_eq!(vector.len(), self.dim);
        self.vectors[id as usize] = vector;
    }

    // -- Search (read-only, lock-free reads via Mutex::lock snapshots) --------

    fn search_layer_single(&self, query: &[f32], entry: u32, layer: usize) -> u32 {
        let mut current = entry;
        let mut current_dist = distance(query, &self.vectors[current as usize], self.metric);

        loop {
            let mut changed = false;
            let node = &self.nodes[current as usize];
            if layer < node.num_layers() {
                let nb = node.neighbors[layer].lock();
                for &neighbor_id in nb.iter() {
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

    fn search_layer(&self, query: &[f32], entry: u32, ef: usize, layer: usize) -> Vec<Candidate> {
        let total = self.vectors.len();
        let entry_dist = distance(query, &self.vectors[entry as usize], self.metric);

        let mut candidates = BinaryHeap::new();
        candidates.push(Candidate {
            id: entry,
            dist: entry_dist,
        });

        let mut results = BinaryHeap::new();
        results.push(RevCandidate {
            id: entry,
            dist: entry_dist,
        });

        // Bit vector for visited nodes — much faster than HashSet for sequential u32 ids.
        let mut visited = vec![false; total];
        visited[entry as usize] = true;

        while let Some(current) = candidates.pop() {
            let farthest_dist = results.peek().map(|r| r.dist).unwrap_or(f32::MAX);
            if current.dist > farthest_dist {
                break;
            }

            let node = &self.nodes[current.id as usize];
            if layer >= node.num_layers() {
                continue;
            }

            // Process neighbors under lock — no clone needed, just read u32s.
            let nb = node.neighbors[layer].lock();
            for &neighbor_id in nb.iter() {
                let nid = neighbor_id as usize;
                if visited[nid] {
                    continue;
                }
                visited[nid] = true;

                let d = distance(query, &self.vectors[nid], self.metric);
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

    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(u32, f32)> {
        assert_eq!(query.len(), self.dim, "query dimension mismatch");

        let ep = self.entry_point.load(AtomicOrdering::Relaxed);
        if ep == NO_ENTRY {
            return Vec::new();
        }

        let ef = ef_search.max(k);
        let max_layer = self.max_layer.load(AtomicOrdering::Relaxed);

        let mut current_ep = ep;
        if max_layer > 0 {
            for l in (1..=max_layer).rev() {
                current_ep = self.search_layer_single(query, current_ep, l);
            }
        }

        let candidates = self.search_layer(query, current_ep, ef, 0);

        candidates
            .into_iter()
            .filter(|c| !self.deleted.contains(&c.id))
            .take(k)
            .map(|c| (c.id, c.dist))
            .collect()
    }

    // -- Accessors ------------------------------------------------------------

    pub fn get_vector(&self, id: u32) -> Option<&[f32]> {
        self.vectors.get(id as usize).map(|v| v.as_slice())
    }

    pub fn len(&self) -> usize {
        self.active_count.load(AtomicOrdering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn total_slots(&self) -> usize {
        self.vectors.len()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    // -- Graph serialization --------------------------------------------------
    //
    // Format:
    //   magic: u32 = 0x48535747 ("HSWG")
    //   version: u32 = 1
    //   dim: u32
    //   metric: u8
    //   m: u32
    //   ef_construction: u32
    //   entry_point: u32
    //   max_layer: u32
    //   num_nodes: u32
    //   num_deleted: u32
    //   deleted_ids: [u32; num_deleted]
    //   for each node:
    //     vector: [f32; dim]
    //     num_layers: u32
    //     for each layer:
    //       num_neighbors: u32
    //       neighbors: [u32; num_neighbors]

    const GRAPH_MAGIC: u32 = 0x48535747;
    const GRAPH_VERSION: u32 = 1;

    pub fn save_graph<W: IoWrite>(&self, w: &mut W) -> io::Result<()> {
        let mut bw = io::BufWriter::new(w);

        bw.write_u32::<LittleEndian>(Self::GRAPH_MAGIC)?;
        bw.write_u32::<LittleEndian>(Self::GRAPH_VERSION)?;
        bw.write_u32::<LittleEndian>(self.dim as u32)?;
        bw.write_u8(match self.metric {
            Metric::Cosine => 0,
            Metric::Euclidean => 1,
            Metric::DotProduct => 2,
        })?;
        bw.write_u32::<LittleEndian>(self.m as u32)?;
        bw.write_u32::<LittleEndian>(self.ef_construction as u32)?;
        bw.write_u32::<LittleEndian>(self.entry_point.load(AtomicOrdering::Relaxed))?;
        bw.write_u32::<LittleEndian>(self.max_layer.load(AtomicOrdering::Relaxed) as u32)?;
        bw.write_u32::<LittleEndian>(self.vectors.len() as u32)?;

        // Deleted set.
        bw.write_u32::<LittleEndian>(self.deleted.len() as u32)?;
        for &id in &self.deleted {
            bw.write_u32::<LittleEndian>(id)?;
        }

        // Nodes: vector + graph structure.
        for i in 0..self.vectors.len() {
            // Vector data.
            for &val in &self.vectors[i] {
                bw.write_f32::<LittleEndian>(val)?;
            }

            // Graph structure.
            let node = &self.nodes[i];
            bw.write_u32::<LittleEndian>(node.num_layers() as u32)?;
            for l in 0..node.num_layers() {
                let nb = node.neighbors[l].lock();
                bw.write_u32::<LittleEndian>(nb.len() as u32)?;
                for &neighbor_id in nb.iter() {
                    bw.write_u32::<LittleEndian>(neighbor_id)?;
                }
            }
        }

        bw.flush()?;
        Ok(())
    }

    pub fn load_graph<R: IoRead>(r: &mut R) -> io::Result<Self> {
        let mut br = io::BufReader::new(r);

        let magic = br.read_u32::<LittleEndian>()?;
        if magic != Self::GRAPH_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad graph magic"));
        }
        let version = br.read_u32::<LittleEndian>()?;
        if version != Self::GRAPH_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported graph version: {}", version),
            ));
        }

        let dim = br.read_u32::<LittleEndian>()? as usize;
        let metric = match br.read_u8()? {
            0 => Metric::Cosine,
            1 => Metric::Euclidean,
            2 => Metric::DotProduct,
            v => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown metric: {}", v),
                ))
            }
        };
        let m = br.read_u32::<LittleEndian>()? as usize;
        let ef_construction = br.read_u32::<LittleEndian>()? as usize;
        let entry_point = br.read_u32::<LittleEndian>()?;
        let max_layer = br.read_u32::<LittleEndian>()? as usize;
        let num_nodes = br.read_u32::<LittleEndian>()? as usize;

        // Deleted set.
        let num_deleted = br.read_u32::<LittleEndian>()? as usize;
        let mut deleted = HashSet::with_capacity(num_deleted);
        for _ in 0..num_deleted {
            deleted.insert(br.read_u32::<LittleEndian>()?);
        }

        let mut vectors = Vec::with_capacity(num_nodes);
        let mut nodes = Vec::with_capacity(num_nodes);

        for _ in 0..num_nodes {
            // Vector.
            let mut vec = Vec::with_capacity(dim);
            for _ in 0..dim {
                vec.push(br.read_f32::<LittleEndian>()?);
            }
            vectors.push(vec);

            // Graph structure.
            let num_layers = br.read_u32::<LittleEndian>()? as usize;
            let mut neighbors = Vec::with_capacity(num_layers);
            for _ in 0..num_layers {
                let num_nb = br.read_u32::<LittleEndian>()? as usize;
                let mut nb = Vec::with_capacity(num_nb);
                for _ in 0..num_nb {
                    nb.push(br.read_u32::<LittleEndian>()?);
                }
                neighbors.push(Mutex::new(nb));
            }
            nodes.push(Node { neighbors });
        }

        let active_count = num_nodes - deleted.len();
        let ml = 1.0 / (m as f64).ln();

        Ok(HnswIndex {
            vectors,
            nodes,
            deleted,
            entry_point: AtomicU32::new(entry_point),
            max_layer: AtomicUsize::new(max_layer),
            metric,
            m,
            m0: m * 2,
            ef_construction,
            ml,
            dim,
            active_count: AtomicUsize::new(active_count),
        })
    }
}

// Safety: HnswIndex is Send+Sync because all mutable state is behind Mutex/Atomic.
// The raw pointer in batch_insert is safe because we pre-allocate all slots
// and only mutate through Mutex-protected neighbors.
unsafe impl Sync for HnswIndex {}

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

        // Save.
        let mut buf = Vec::new();
        index.save_graph(&mut buf).unwrap();

        // Load.
        let loaded = HnswIndex::load_graph(&mut &buf[..]).unwrap();

        assert_eq!(loaded.len(), 2); // 3 inserted, 1 deleted
        assert_eq!(loaded.total_slots(), 3);
        assert!(loaded.is_deleted(1));

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
}

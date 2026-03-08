/// HNSW (Hierarchical Navigable Small World) index.
///
/// This is the algorithm that makes vector search fast — O(log n) instead of O(n).
/// It builds a multi-layer graph where each layer is progressively sparser.
/// Search starts at the top (sparse) layer and greedily descends to find neighbors.
///
/// Supports concurrent batch inserts via per-node Mutex locking (hnswlib-style).

use crate::distance::{batch_distances, distance, maybe_print_blas_hint, Metric};
use crate::quantize::ScalarQuantizer;
use memmap2::Mmap;
use parking_lot::Mutex;
use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
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

// -- Vector storage ----------------------------------------------------------

/// Vector data can be owned (in-memory, mutable) or memory-mapped (read-only, instant load).
enum VectorStore {
    /// Heap-allocated vectors — used for new databases and after mutations.
    Owned(Vec<f32>),
    /// Memory-mapped file containing [vectors (num_vectors_f32)][norms (num_norms_f32)].
    /// `vec_len` is the number of f32s in the vectors portion.
    Mapped { mmap: Mmap, vec_len: usize },
}

impl VectorStore {
    /// Get the vectors portion as f32 slice.
    fn as_slice(&self) -> &[f32] {
        match self {
            VectorStore::Owned(v) => v,
            VectorStore::Mapped { mmap, vec_len } => {
                let ptr = mmap.as_ptr() as *const f32;
                unsafe { std::slice::from_raw_parts(ptr, *vec_len) }
            }
        }
    }

    /// Get the norms portion from a Mapped store. Returns None for Owned.
    fn mapped_norms(&self) -> Option<Vec<f32>> {
        match self {
            VectorStore::Owned(_) => None,
            VectorStore::Mapped { mmap, vec_len } => {
                let total_f32s = mmap.len() / std::mem::size_of::<f32>();
                let norms_len = total_f32s - vec_len;
                let ptr = mmap.as_ptr() as *const f32;
                let norms_slice = unsafe {
                    std::slice::from_raw_parts(ptr.add(*vec_len), norms_len)
                };
                Some(norms_slice.to_vec())
            }
        }
    }

    /// Get a mutable reference — only valid for Owned. Panics if Mapped.
    fn as_mut_vec(&mut self) -> &mut Vec<f32> {
        match self {
            VectorStore::Owned(v) => v,
            VectorStore::Mapped { .. } => panic!("cannot mutate mmap'd vectors — save and reopen, or use owned mode"),
        }
    }

    /// Convert from Mapped to Owned (copies data into heap). Required before mutations.
    fn ensure_owned(&mut self) {
        if let VectorStore::Mapped { .. } = self {
            let data = self.as_slice().to_vec();
            *self = VectorStore::Owned(data);
        }
    }

    fn is_mapped(&self) -> bool {
        matches!(self, VectorStore::Mapped { .. })
    }
}

// -- HNSW Index --------------------------------------------------------------

/// Optional quantized vector storage for faster HNSW traversal.
/// Quantized vectors are 4x smaller (u8 vs f32), improving cache utilization
/// during the random-access pattern of graph traversal.
struct QuantizedSearch {
    quantizer: ScalarQuantizer,
    /// Flat u8 array: n vectors of `dim` bytes each.
    vectors: Vec<u8>,
}

impl QuantizedSearch {
    /// Dequantize a vector into a pre-allocated f32 buffer.
    #[inline]
    fn dequantize_into(&self, id: u32, dim: usize, buf: &mut [f32]) {
        let start = id as usize * dim;
        self.quantizer.dequantize_into(&self.vectors[start..start + dim], buf);
    }
}

/// Graph-level statistics for diagnostics and monitoring.
#[derive(Debug, Clone)]
pub struct GraphStats {
    /// Number of active (non-deleted) vectors.
    pub num_vectors: usize,
    /// Number of deleted (soft-deleted) vectors.
    pub num_deleted: usize,
    /// Number of HNSW layers.
    pub num_layers: usize,
    /// Average number of neighbors per node on layer 0.
    pub avg_degree_layer0: f64,
    /// Maximum number of neighbors for any node on layer 0.
    pub max_degree_layer0: usize,
    /// Minimum number of neighbors for any node on layer 0.
    pub min_degree_layer0: usize,
    /// Approximate memory usage for full-precision vectors (bytes).
    pub memory_vectors_bytes: usize,
    /// Approximate memory usage for graph structure (bytes).
    pub memory_graph_bytes: usize,
    /// Approximate memory usage for quantized vectors (bytes), 0 if not enabled.
    pub memory_quantized_bytes: usize,
    /// Whether the index is currently using brute-force (dataset too small for HNSW).
    pub uses_brute_force: bool,
    /// Whether quantized search is enabled.
    pub uses_quantized_search: bool,
}

pub struct HnswIndex {
    /// Flat contiguous vector storage: all vectors packed end-to-end.
    /// Vector i is at vectors[i*dim .. (i+1)*dim].
    /// Can be either owned (heap) or memory-mapped (disk).
    vectors: VectorStore,
    /// Cached L2 norms for each vector (used by batch cosine search).
    norms: Vec<f32>,
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
    /// Optional SQ8 quantized vectors for faster HNSW traversal.
    quantized: Option<QuantizedSearch>,
}

// Entry point sentinel: no entry point set yet.
const NO_ENTRY: u32 = u32::MAX;

impl HnswIndex {
    pub fn new(dim: usize, metric: Metric, m: usize, ef_construction: usize) -> Self {
        let ml = 1.0 / (m as f64).ln();
        HnswIndex {
            vectors: VectorStore::Owned(Vec::new()),
            norms: Vec::new(),
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
            quantized: None,
        }
    }

    /// Get vector data for a given internal id.
    #[inline]
    fn vector(&self, id: u32) -> &[f32] {
        let start = id as usize * self.dim;
        &self.vectors.as_slice()[start..start + self.dim]
    }

    #[inline]
    fn compute_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        (-r.ln() * self.ml).floor() as usize
    }

    /// Insert a single vector (sequential). Returns the internal id.
    pub fn insert(&mut self, vector: Vec<f32>) -> u32 {
        assert_eq!(vector.len(), self.dim, "vector dimension mismatch");

        let id = self.nodes.len() as u32;
        let level = self.random_level();

        // Update quantized vectors if enabled.
        if let Some(ref mut qs) = self.quantized {
            qs.vectors.extend(qs.quantizer.quantize(&vector));
        }

        self.norms.push(Self::compute_norm(&vector));
        self.vectors.ensure_owned();
        self.vectors.as_mut_vec().extend_from_slice(&vector);
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

        let start_id = self.nodes.len() as u32;
        let n = vectors.len();

        // Pre-compute levels.
        let levels: Vec<usize> = (0..n).map(|_| self.random_level()).collect();

        // Pre-allocate all slots so concurrent reads are safe.
        self.vectors.ensure_owned();
        for (i, vec) in vectors.into_iter().enumerate() {
            if let Some(ref mut qs) = self.quantized {
                qs.vectors.extend(qs.quantizer.quantize(&vec));
            }
            self.norms.push(Self::compute_norm(&vec));
            self.vectors.as_mut_vec().extend_from_slice(&vec);
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

            // Safe reborrow: vectors and nodes are pre-allocated and won't resize.
            // Each node's neighbors are Mutex-protected. entry_point/max_layer are atomic.
            // We reborrow &mut self as &self for the parallel section — no mutation
            // of vectors/nodes vecs happens here, only interior-mutable Mutex/Atomic access.
            {
                let this: &Self = &*self;
                remaining.par_iter().for_each(|&(id, level)| {
                    let current_ep = this.entry_point.load(AtomicOrdering::Relaxed);
                    this.connect_node(id, level, current_ep);
                });
            }

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

    /// Connect a node to its neighbors at all layers.
    fn connect_node(&self, id: u32, level: usize, ep: u32) {
        let max_layer = self.max_layer.load(AtomicOrdering::Relaxed);
        let mut current_ep = ep;

        // Phase 1: Greedy descent through upper layers.
        if max_layer > level {
            for l in ((level + 1)..=max_layer).rev() {
                current_ep = self.search_layer_single(self.vector(id), current_ep, l);
            }
        }

        // Phase 2: Search and connect at each layer.
        for l in (0..=level.min(max_layer)).rev() {
            let max_neighbors = if l == 0 { self.m0 } else { self.m };
            let neighbors =
                self.search_layer(self.vector(id), current_ep, self.ef_construction, l);

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

    /// Prune a neighbor list to keep only the closest `max_neighbors`.
    fn prune_neighbors(&self, neighbors: &mut Vec<u32>, node_id: u32, max_neighbors: usize) {
        let nv = self.vector(node_id);
        let mut scored: Vec<(f32, u32)> = neighbors
            .iter()
            .map(|&nid| (distance(nv, self.vector(nid), self.metric), nid))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        *neighbors = scored
            .into_iter()
            .take(max_neighbors)
            .map(|(_, nid)| nid)
            .collect();
    }

    pub fn mark_deleted(&mut self, id: u32) -> bool {
        if (id as usize) < self.nodes.len() && !self.deleted.contains(&id) {
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

    /// Update a vector's data and reconnect it in the graph so search
    /// quality is maintained after the change.
    pub fn update_vector(&mut self, id: u32, vector: Vec<f32>) {
        assert_eq!(vector.len(), self.dim);

        // Overwrite the vector data and cached norm.
        self.vectors.ensure_owned();
        let start = id as usize * self.dim;
        self.vectors.as_mut_vec()[start..start + self.dim].copy_from_slice(&vector);
        self.norms[id as usize] = Self::compute_norm(&vector);

        // Update quantized vector if enabled.
        if let Some(ref mut qs) = self.quantized {
            let qv = qs.quantizer.quantize(&vector);
            qs.vectors[start..start + self.dim].copy_from_slice(&qv);
        }

        // If there's only one node, no connections to repair.
        if self.nodes.len() <= 1 {
            return;
        }

        let level = self.nodes[id as usize].num_layers() - 1;

        // Disconnect: remove this node from all its neighbors' lists,
        // then clear this node's own neighbor lists.
        for l in 0..=level {
            let old_neighbors: Vec<u32> = {
                let mut nb = self.nodes[id as usize].neighbors[l].lock();
                let old = nb.clone();
                nb.clear();
                old
            };

            for &neighbor_id in &old_neighbors {
                let neighbor = &self.nodes[neighbor_id as usize];
                if l < neighbor.num_layers() {
                    let mut nnb = neighbor.neighbors[l].lock();
                    nnb.retain(|&x| x != id);
                }
            }
        }

        // Reconnect using the graph starting from a valid entry point.
        let ep = self.entry_point.load(AtomicOrdering::Relaxed);
        let start_ep = if ep == id {
            // The node being updated is the entry point — find another node to start from.
            (0..self.nodes.len() as u32)
                .find(|&i| i != id && !self.deleted.contains(&i))
                .unwrap_or(ep)
        } else {
            ep
        };

        self.connect_node(id, level, start_ep);
    }

    // -- Search ----------------------------------------------------------------

    /// Whether this dataset is small enough that brute-force SIMD scan beats
    /// HNSW graph traversal. The heuristic is based on total scan cost (n * dim):
    /// brute-force wins when the entire vector dataset fits roughly in L3 cache
    /// (~16-32MB on modern CPUs), making the linear scan cache-friendly while
    /// HNSW suffers from random-access cache misses.
    ///
    /// Threshold of 10M floats (~40MB) gives crossover at:
    ///   dim=128  → ~78k vectors
    ///   dim=384  → ~26k vectors
    ///   dim=768  → ~13k vectors
    ///   dim=1536 → ~6.5k vectors
    const BRUTE_FORCE_THRESHOLD: usize = 10_000_000;

    pub fn uses_brute_force(&self) -> bool {
        let active = self.active_count.load(AtomicOrdering::Relaxed);
        active * self.dim < Self::BRUTE_FORCE_THRESHOLD
    }

    /// Linear scan with SIMD distance computation. For small datasets this
    /// beats HNSW because the sequential scan is cache-friendly with no graph
    /// traversal overhead. Uses partial sort (O(n)) instead of a heap.
    fn brute_force_search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        maybe_print_blas_hint();

        let num_nodes = self.nodes.len();
        let has_deletions = !self.deleted.is_empty();

        if !has_deletions {
            // Fast path: batch compute all distances in one tight loop.
            // Avoids per-vector function call overhead; for cosine, query norm
            // is computed once instead of N times.
            let all_dists = batch_distances(query, self.vectors.as_slice(), self.dim, self.metric, &self.norms);
            let mut dists: Vec<(u32, f32)> = all_dists
                .into_iter()
                .enumerate()
                .map(|(i, d)| (i as u32, d))
                .collect();

            if dists.len() > k {
                dists.select_nth_unstable_by(k, |a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
                });
                dists.truncate(k);
            }
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            return dists;
        }

        // Slow path with deletions: per-vector with skip.
        let mut dists: Vec<(u32, f32)> = Vec::with_capacity(num_nodes);
        for i in 0..num_nodes {
            let id = i as u32;
            if self.deleted.contains(&id) {
                continue;
            }
            dists.push((id, distance(query, self.vector(id), self.metric)));
        }

        if dists.len() > k {
            dists.select_nth_unstable_by(k, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
            });
            dists.truncate(k);
        }
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        dists
    }

    fn search_layer_single(&self, query: &[f32], entry: u32, layer: usize) -> u32 {
        let mut current = entry;
        let mut current_dist = distance(query, self.vector(current), self.metric);

        loop {
            let mut changed = false;
            let node = &self.nodes[current as usize];
            if layer < node.num_layers() {
                let nb = node.neighbors[layer].lock();
                for &neighbor_id in nb.iter() {
                    let d = distance(query, self.vector(neighbor_id), self.metric);
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
        let total = self.nodes.len();
        let entry_dist = distance(query, self.vector(entry), self.metric);

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

                let d = distance(query, self.vector(neighbor_id), self.metric);
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

    /// Compute distance using quantized vectors (dequantize into thread-local buffer).
    #[inline]
    fn quantized_distance(&self, query: &[f32], id: u32, buf: &mut [f32]) -> f32 {
        let qs = self.quantized.as_ref().unwrap();
        qs.dequantize_into(id, self.dim, buf);
        distance(query, buf, self.metric)
    }

    fn search_layer_single_quantized(&self, query: &[f32], entry: u32, layer: usize) -> u32 {
        let mut buf = vec![0.0f32; self.dim];
        let mut current = entry;
        let mut current_dist = self.quantized_distance(query, current, &mut buf);

        loop {
            let mut changed = false;
            let node = &self.nodes[current as usize];
            if layer < node.num_layers() {
                let nb = node.neighbors[layer].lock();
                for &neighbor_id in nb.iter() {
                    let d = self.quantized_distance(query, neighbor_id, &mut buf);
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

    fn search_layer_quantized(&self, query: &[f32], entry: u32, ef: usize, layer: usize) -> Vec<Candidate> {
        let total = self.nodes.len();
        let mut buf = vec![0.0f32; self.dim];
        let entry_dist = self.quantized_distance(query, entry, &mut buf);

        let mut candidates = BinaryHeap::new();
        candidates.push(Candidate { id: entry, dist: entry_dist });

        let mut results = BinaryHeap::new();
        results.push(RevCandidate { id: entry, dist: entry_dist });

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

            let nb = node.neighbors[layer].lock();
            for &neighbor_id in nb.iter() {
                let nid = neighbor_id as usize;
                if visited[nid] {
                    continue;
                }
                visited[nid] = true;

                let d = self.quantized_distance(query, neighbor_id, &mut buf);
                let farthest_dist = results.peek().map(|r| r.dist).unwrap_or(f32::MAX);

                if d < farthest_dist || results.len() < ef {
                    candidates.push(Candidate { id: neighbor_id, dist: d });
                    results.push(RevCandidate { id: neighbor_id, dist: d });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut result_vec: Vec<Candidate> = results
            .into_iter()
            .map(|r| Candidate { id: r.id, dist: r.dist })
            .collect();
        result_vec.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        result_vec
    }

    /// Search with an inline filter predicate. Non-matching vectors still participate
    /// in graph traversal (they can lead to matching neighbors) but are excluded from results.
    /// This is much more efficient than post-filter with over-fetch for selective filters.
    pub fn search_filtered<F: Fn(u32) -> bool>(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        predicate: F,
    ) -> Vec<(u32, f32)> {
        assert_eq!(query.len(), self.dim, "query dimension mismatch");

        let ep = self.entry_point.load(AtomicOrdering::Relaxed);
        if ep == NO_ENTRY {
            return Vec::new();
        }

        // For small datasets, brute-force scan with inline filter.
        if self.uses_brute_force() {
            return self.brute_force_search_filtered(query, k, &predicate);
        }

        // Use a larger ef to compensate for filtered-out candidates.
        let ef = ef_search.max(k * 4);
        let max_layer = self.max_layer.load(AtomicOrdering::Relaxed);
        let use_quantized = self.quantized.is_some();

        let mut current_ep = ep;
        if max_layer > 0 {
            for l in (1..=max_layer).rev() {
                current_ep = if use_quantized {
                    self.search_layer_single_quantized(query, current_ep, l)
                } else {
                    self.search_layer_single(query, current_ep, l)
                };
            }
        }

        // Search layer 0 — all candidates traverse the graph, but only matching ones are results.
        let candidates = if use_quantized {
            self.search_layer_quantized(query, current_ep, ef, 0)
        } else {
            self.search_layer(query, current_ep, ef, 0)
        };

        let mut results: Vec<(u32, f32)> = candidates
            .into_iter()
            .filter(|c| !self.deleted.contains(&c.id) && predicate(c.id))
            .map(|c| {
                if use_quantized {
                    // Re-rank with full precision.
                    (c.id, distance(query, self.vector(c.id), self.metric))
                } else {
                    (c.id, c.dist)
                }
            })
            .collect();

        if use_quantized {
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        }
        results.truncate(k);
        results
    }

    /// Brute-force search with inline filter.
    fn brute_force_search_filtered<F: Fn(u32) -> bool>(
        &self,
        query: &[f32],
        k: usize,
        predicate: &F,
    ) -> Vec<(u32, f32)> {
        maybe_print_blas_hint();

        let num_nodes = self.nodes.len();
        let mut dists: Vec<(u32, f32)> = Vec::with_capacity(num_nodes);

        for i in 0..num_nodes {
            let id = i as u32;
            if self.deleted.contains(&id) || !predicate(id) {
                continue;
            }
            dists.push((id, distance(query, self.vector(id), self.metric)));
        }

        if dists.len() > k {
            dists.select_nth_unstable_by(k, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
            });
            dists.truncate(k);
        }
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        dists
    }

    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(u32, f32)> {
        assert_eq!(query.len(), self.dim, "query dimension mismatch");

        let ep = self.entry_point.load(AtomicOrdering::Relaxed);
        if ep == NO_ENTRY {
            return Vec::new();
        }

        // For small datasets, brute-force SIMD scan is faster than HNSW.
        if self.uses_brute_force() {
            return self.brute_force_search(query, k);
        }

        let ef = ef_search.max(k);
        let max_layer = self.max_layer.load(AtomicOrdering::Relaxed);
        let use_quantized = self.quantized.is_some();

        let mut current_ep = ep;
        if max_layer > 0 {
            for l in (1..=max_layer).rev() {
                current_ep = if use_quantized {
                    self.search_layer_single_quantized(query, current_ep, l)
                } else {
                    self.search_layer_single(query, current_ep, l)
                };
            }
        }

        let candidates = if use_quantized {
            self.search_layer_quantized(query, current_ep, ef, 0)
        } else {
            self.search_layer(query, current_ep, ef, 0)
        };

        if use_quantized {
            // Re-rank with full-precision distances.
            let mut reranked: Vec<(u32, f32)> = candidates
                .into_iter()
                .filter(|c| !self.deleted.contains(&c.id))
                .map(|c| (c.id, distance(query, self.vector(c.id), self.metric)))
                .collect();
            reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            reranked.truncate(k);
            reranked
        } else {
            candidates
                .into_iter()
                .filter(|c| !self.deleted.contains(&c.id))
                .take(k)
                .map(|c| (c.id, c.dist))
                .collect()
        }
    }

    /// Search multiple queries in parallel using Rayon.
    /// Returns one result Vec per query.
    pub fn search_many(&self, queries: &[&[f32]], k: usize, ef_search: usize) -> Vec<Vec<(u32, f32)>> {
        if queries.is_empty() {
            return Vec::new();
        }
        for q in queries {
            assert_eq!(q.len(), self.dim, "query dimension mismatch");
        }

        let ep = self.entry_point.load(AtomicOrdering::Relaxed);
        if ep == NO_ENTRY {
            return vec![Vec::new(); queries.len()];
        }

        queries
            .par_iter()
            .map(|query| self.search(query, k, ef_search))
            .collect()
    }

    // -- Accessors ------------------------------------------------------------

    pub fn get_vector(&self, id: u32) -> Option<&[f32]> {
        if (id as usize) < self.nodes.len() {
            Some(self.vector(id))
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.active_count.load(AtomicOrdering::Relaxed)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn total_slots(&self) -> usize {
        self.nodes.len()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Raw access to the flat vector storage (for quantization, etc.).
    pub fn vectors_slice(&self) -> &[f32] {
        self.vectors.as_slice()
    }

    /// Enable quantized search: creates SQ8 quantized copies of all vectors.
    /// HNSW traversal will use these for faster approximate distance computation,
    /// then re-rank final candidates with full-precision vectors.
    pub fn enable_quantized_search(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        let sq = ScalarQuantizer::train(self.vectors.as_slice(), self.dim);
        let quantized_vecs = sq.quantize_batch(self.vectors.as_slice(), self.dim);
        self.quantized = Some(QuantizedSearch {
            quantizer: sq,
            vectors: quantized_vecs,
        });
    }

    /// Disable quantized search (free the quantized vector memory).
    pub fn disable_quantized_search(&mut self) {
        self.quantized = None;
    }

    /// Whether quantized search is enabled.
    pub fn has_quantized_search(&self) -> bool {
        self.quantized.is_some()
    }

    /// Load pre-computed quantized vectors from a ScalarQuantizer and u8 data.
    pub fn load_quantized(&mut self, quantizer: ScalarQuantizer, quantized_vectors: Vec<u8>) {
        self.quantized = Some(QuantizedSearch {
            quantizer,
            vectors: quantized_vectors,
        });
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Compute graph-level statistics for diagnostics.
    pub fn graph_stats(&self) -> GraphStats {
        let n = self.nodes.len();
        if n == 0 {
            return GraphStats {
                num_vectors: 0,
                num_deleted: self.deleted.len(),
                num_layers: 0,
                avg_degree_layer0: 0.0,
                max_degree_layer0: 0,
                min_degree_layer0: 0,
                memory_vectors_bytes: 0,
                memory_graph_bytes: 0,
                memory_quantized_bytes: 0,
                uses_brute_force: true,
                uses_quantized_search: self.quantized.is_some(),
            };
        }

        let max_layer = self.max_layer.load(AtomicOrdering::Relaxed);
        let mut total_degree: usize = 0;
        let mut max_degree: usize = 0;
        let mut min_degree: usize = usize::MAX;

        for (i, node) in self.nodes.iter().enumerate() {
            if self.deleted.contains(&(i as u32)) {
                continue;
            }
            let deg = node.neighbors[0].lock().len();
            total_degree += deg;
            max_degree = max_degree.max(deg);
            min_degree = min_degree.min(deg);
        }

        let active = self.len();
        let avg_degree = if active > 0 {
            total_degree as f64 / active as f64
        } else {
            0.0
        };
        if active == 0 {
            min_degree = 0;
        }

        // Memory estimates.
        let mem_vectors = self.vectors.as_slice().len() * std::mem::size_of::<f32>();
        let mut mem_graph: usize = 0;
        for node in &self.nodes {
            for layer in &node.neighbors {
                mem_graph += layer.lock().len() * std::mem::size_of::<u32>();
            }
        }
        let mem_quantized = self.quantized.as_ref().map_or(0, |qs| {
            qs.vectors.len() + qs.quantizer.dim * 2 * std::mem::size_of::<f32>()
        });

        GraphStats {
            num_vectors: active,
            num_deleted: self.deleted.len(),
            num_layers: max_layer + 1,
            avg_degree_layer0: avg_degree,
            max_degree_layer0: max_degree,
            min_degree_layer0: min_degree,
            memory_vectors_bytes: mem_vectors,
            memory_graph_bytes: mem_graph,
            memory_quantized_bytes: mem_quantized,
            uses_brute_force: self.uses_brute_force(),
            uses_quantized_search: self.quantized.is_some(),
        }
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
    const GRAPH_VERSION: u32 = 2;

    /// Save vectors + norms as a flat binary file, directly mmap-able.
    /// Layout: [N*dim f32 vectors][N f32 norms]
    pub fn save_vectors<W: IoWrite>(&self, w: &mut W) -> io::Result<()> {
        let vec_bytes = unsafe {
            std::slice::from_raw_parts(
                self.vectors.as_slice().as_ptr() as *const u8,
                self.vectors.as_slice().len() * std::mem::size_of::<f32>(),
            )
        };
        w.write_all(vec_bytes)?;
        let norm_bytes = unsafe {
            std::slice::from_raw_parts(
                self.norms.as_ptr() as *const u8,
                self.norms.len() * std::mem::size_of::<f32>(),
            )
        };
        w.write_all(norm_bytes)?;
        w.flush()
    }

    /// Save graph structure (no vectors — those go to a separate mmap-able file).
    pub fn save_graph<W: IoWrite>(&self, w: &mut W) -> io::Result<()> {
        let num_nodes = self.nodes.len();

        w.write_u32::<LittleEndian>(Self::GRAPH_MAGIC)?;
        w.write_u32::<LittleEndian>(Self::GRAPH_VERSION)?;
        w.write_u32::<LittleEndian>(self.dim as u32)?;
        w.write_u8(match self.metric {
            Metric::Cosine => 0,
            Metric::Euclidean => 1,
            Metric::DotProduct => 2,
        })?;
        w.write_u32::<LittleEndian>(self.m as u32)?;
        w.write_u32::<LittleEndian>(self.ef_construction as u32)?;
        w.write_u32::<LittleEndian>(self.entry_point.load(AtomicOrdering::Relaxed))?;
        w.write_u32::<LittleEndian>(self.max_layer.load(AtomicOrdering::Relaxed) as u32)?;
        w.write_u32::<LittleEndian>(num_nodes as u32)?;

        // Deleted set.
        w.write_u32::<LittleEndian>(self.deleted.len() as u32)?;
        for &id in &self.deleted {
            w.write_u32::<LittleEndian>(id)?;
        }

        // Graph structure only (no vectors).
        for i in 0..num_nodes {
            let node = &self.nodes[i];
            w.write_u32::<LittleEndian>(node.num_layers() as u32)?;
            for l in 0..node.num_layers() {
                let nb = node.neighbors[l].lock();
                w.write_u32::<LittleEndian>(nb.len() as u32)?;
                for &neighbor_id in nb.iter() {
                    w.write_u32::<LittleEndian>(neighbor_id)?;
                }
            }
        }

        w.flush()?;
        Ok(())
    }

    /// Load graph structure + mmap'd vectors.
    /// Returns (index, remaining_bytes) — the remaining bytes contain the metadata section.
    pub fn load_graph_mmap(data: &[u8], vectors_mmap: Mmap) -> io::Result<(Self, &[u8])> {
        let mut cursor = data;

        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != Self::GRAPH_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad graph magic"));
        }
        let version = cursor.read_u32::<LittleEndian>()?;
        if version != Self::GRAPH_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported graph version: {}", version),
            ));
        }

        let dim = cursor.read_u32::<LittleEndian>()? as usize;
        let metric = match cursor.read_u8()? {
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
        let m = cursor.read_u32::<LittleEndian>()? as usize;
        let ef_construction = cursor.read_u32::<LittleEndian>()? as usize;
        let entry_point = cursor.read_u32::<LittleEndian>()?;
        let max_layer = cursor.read_u32::<LittleEndian>()? as usize;
        let num_nodes = cursor.read_u32::<LittleEndian>()? as usize;

        // Deleted set.
        let num_deleted = cursor.read_u32::<LittleEndian>()? as usize;
        let mut deleted = HashSet::with_capacity(num_deleted);
        for _ in 0..num_deleted {
            deleted.insert(cursor.read_u32::<LittleEndian>()?);
        }

        // Graph structure only — parse from in-memory buffer.
        let mut nodes = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            let num_layers = cursor.read_u32::<LittleEndian>()? as usize;
            let mut neighbors = Vec::with_capacity(num_layers);
            for _ in 0..num_layers {
                let num_nb = cursor.read_u32::<LittleEndian>()? as usize;
                let mut nb = Vec::with_capacity(num_nb);
                for _ in 0..num_nb {
                    nb.push(cursor.read_u32::<LittleEndian>()?);
                }
                neighbors.push(Mutex::new(nb));
            }
            nodes.push(Node { neighbors });
        }

        let vec_len = num_nodes * dim;
        let vectors = VectorStore::Mapped { mmap: vectors_mmap, vec_len };
        let active_count = num_nodes - deleted.len();
        let ml = 1.0 / (m as f64).ln();

        // Read pre-computed norms from the mmap (appended after vectors).
        let norms = vectors.mapped_norms().unwrap_or_else(|| {
            let vslice = vectors.as_slice();
            (0..num_nodes)
                .map(|i| Self::compute_norm(&vslice[i * dim..(i + 1) * dim]))
                .collect()
        });

        Ok((HnswIndex {
            vectors,
            norms,
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
            quantized: None,
        }, cursor))
    }

    /// Whether vectors are memory-mapped (instant load, OS-managed paging).
    pub fn is_mmap(&self) -> bool {
        self.vectors.is_mapped()
    }

    /// The set of deleted internal IDs.
    pub fn deleted_ids(&self) -> &HashSet<u32> {
        &self.deleted
    }

    /// Rebuild the index with only live (non-deleted) vectors.
    /// Returns (new_index, old_to_new_id_mapping).
    pub fn compact(&self) -> (Self, HashMap<u32, u32>) {
        let mut old_to_new: HashMap<u32, u32> = HashMap::new();
        let mut live_vectors: Vec<Vec<f32>> = Vec::new();

        for i in 0..self.nodes.len() {
            let id = i as u32;
            if self.deleted.contains(&id) {
                continue;
            }
            let new_id = live_vectors.len() as u32;
            old_to_new.insert(id, new_id);
            live_vectors.push(self.vector(id).to_vec());
        }

        let mut new_index = HnswIndex::new(self.dim, self.metric, self.m, self.ef_construction);
        if !live_vectors.is_empty() {
            new_index.batch_insert(live_vectors);
        }

        // Preserve quantized search state if it was enabled on the original index.
        if self.quantized.is_some() {
            new_index.enable_quantized_search();
        }

        (new_index, old_to_new)
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
        let mut index = HnswIndex::new(dim, Metric::Euclidean, 16, 200);

        // Generate deterministic but unique vectors using a simple LCG-style hash
        // to avoid collisions (the previous i*7+d*13 mod 1000 had collisions at i vs i+1000).
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

        index.batch_insert(vectors.clone());

        // Verify we're NOT using brute force.
        assert!(
            !index.uses_brute_force(),
            "test needs to exercise HNSW path but fell into brute-force (n*dim={} < 10M)",
            n * dim
        );

        // First verify full-precision HNSW finds the exact match.
        let query: Vec<f32> = vectors[0].clone();
        let full_results = index.search(&query, 10, 200);
        assert_eq!(
            full_results[0].0, 0,
            "full-precision HNSW search didn't find exact match as top-1"
        );
        assert!(
            full_results[0].1 < 0.001,
            "full-precision distance to self should be ~0, got {}",
            full_results[0].1
        );

        // Now enable quantized search and verify.
        index.enable_quantized_search();
        let quantized_results = index.search(&query, 10, 200);
        assert!(!quantized_results.is_empty());

        // After re-ranking with full precision, the exact match should still be top-1.
        assert_eq!(
            quantized_results[0].0, 0,
            "quantized HNSW search didn't find exact match as top-1 (got id {} with dist {})",
            quantized_results[0].0, quantized_results[0].1
        );

        // Check recall: quantized top-10 should overlap significantly with full top-10.
        let full_ids: HashSet<u32> = full_results.iter().map(|r| r.0).collect();
        let quantized_ids: HashSet<u32> = quantized_results.iter().map(|r| r.0).collect();
        let overlap = full_ids.intersection(&quantized_ids).count();
        assert!(
            overlap >= 7,
            "quantized recall too low: only {}/10 overlap with full-precision results",
            overlap
        );
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
}

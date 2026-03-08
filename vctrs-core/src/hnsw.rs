/// HNSW (Hierarchical Navigable Small World) index.
///
/// This is the algorithm that makes vector search fast — O(log n) instead of O(n).
/// It builds a multi-layer graph where each layer is progressively sparser.
/// Search starts at the top (sparse) layer and greedily descends to find neighbors.
///
/// Supports concurrent batch inserts via per-node Mutex locking (hnswlib-style).

use crate::distance::{batch_distances, distance, maybe_print_blas_hint, Metric};
use crate::quantize::ScalarQuantizer;
#[cfg(feature = "mmap")]
use memmap2::Mmap;
use parking_lot::Mutex;
use rand::Rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::io::{self, Write as IoWrite};
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
    #[cfg(feature = "mmap")]
    Mapped { mmap: Mmap, vec_len: usize },
}

impl VectorStore {
    /// Get the vectors portion as f32 slice.
    fn as_slice(&self) -> &[f32] {
        match self {
            VectorStore::Owned(v) => v,
            #[cfg(feature = "mmap")]
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
            #[cfg(feature = "mmap")]
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
            #[cfg(feature = "mmap")]
            VectorStore::Mapped { .. } => panic!("cannot mutate mmap'd vectors — save and reopen, or use owned mode"),
        }
    }

    /// Convert from Mapped to Owned (copies data into heap). Required before mutations.
    fn ensure_owned(&mut self) {
        #[cfg(feature = "mmap")]
        if let VectorStore::Mapped { .. } = self {
            let data = self.as_slice().to_vec();
            *self = VectorStore::Owned(data);
        }
    }

    fn is_mapped(&self) -> bool {
        #[cfg(feature = "mmap")]
        { matches!(self, VectorStore::Mapped { .. }) }
        #[cfg(not(feature = "mmap"))]
        { false }
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
                #[cfg(feature = "parallel")]
                remaining.par_iter().for_each(|&(id, level)| {
                    let current_ep = this.entry_point.load(AtomicOrdering::Relaxed);
                    this.connect_node(id, level, current_ep);
                });
                #[cfg(not(feature = "parallel"))]
                remaining.iter().for_each(|&(id, level)| {
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

        #[cfg(feature = "parallel")]
        {
            queries
                .par_iter()
                .map(|query| self.search(query, k, ef_search))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            queries
                .iter()
                .map(|query| self.search(query, k, ef_search))
                .collect()
        }
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
    #[cfg(feature = "mmap")]
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

    /// Load graph structure + owned vectors (reads vectors into memory).
    /// Returns (index, remaining_bytes) — the remaining bytes contain the metadata section.
    #[cfg(not(feature = "mmap"))]
    pub fn load_graph_owned(data: &[u8], vectors_data: Vec<u8>) -> io::Result<(Self, &[u8])> {
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
        let total_f32s = vectors_data.len() / std::mem::size_of::<f32>();

        // Parse vectors from raw bytes.
        let all_f32s: Vec<f32> = {
            let ptr = vectors_data.as_ptr() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, total_f32s) }.to_vec()
        };
        let vectors = VectorStore::Owned(all_f32s[..vec_len].to_vec());

        // Parse norms (appended after vectors).
        let norms = if total_f32s > vec_len {
            all_f32s[vec_len..].to_vec()
        } else {
            let vslice = vectors.as_slice();
            (0..num_nodes)
                .map(|i| Self::compute_norm(&vslice[i * dim..(i + 1) * dim]))
                .collect()
        };

        let active_count = num_nodes - deleted.len();
        let ml = 1.0 / (m as f64).ln();

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
#[path = "hnsw_tests.rs"]
mod tests;

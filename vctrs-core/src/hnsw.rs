/// HNSW (Hierarchical Navigable Small World) index.
///
/// This is the algorithm that makes vector search fast — O(log n) instead of O(n).
/// It builds a multi-layer graph where each layer is progressively sparser.
/// Search starts at the top (sparse) layer and greedily descends to find neighbors.
///
/// Supports concurrent batch inserts via per-node Mutex locking (hnswlib-style).

use crate::distance::{batch_distances, distance, maybe_print_blas_hint, Metric};
use memmap2::Mmap;
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
            .map(|query| {
                if self.uses_brute_force() {
                    return self.brute_force_search(query, k);
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
            })
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
        }, cursor))
    }

    /// Whether vectors are memory-mapped (instant load, OS-managed paging).
    pub fn is_mmap(&self) -> bool {
        self.vectors.is_mapped()
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
}

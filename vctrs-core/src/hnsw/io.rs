use super::*;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Write as IoWrite};

impl HnswIndex {
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
}

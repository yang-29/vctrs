/// On-disk storage for vectors, metadata, and HNSW graph.
///
/// Two files:
///   data.vctrs  — string ids + metadata (small, human-relevant)
///   graph.vctrs — vectors + full HNSW graph structure (fast to load)

use crate::hnsw::HnswIndex;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read as IoRead, Write as IoWrite};
use std::path::{Path, PathBuf};

const META_MAGIC: u32 = 0x56435452;
const META_VERSION: u32 = 2; // v2: id mapping + metadata only (graph is separate)

/// A stored record: string id -> metadata (vectors live in the graph file).
#[derive(Clone, Debug)]
pub struct Record {
    pub string_id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
}

/// Metadata-only record (for the new format where graph stores vectors).
#[derive(Clone, Debug)]
pub struct MetaRecord {
    pub internal_id: u32,
    pub string_id: String,
    pub metadata: Option<serde_json::Value>,
}

pub struct Storage {
    data_path: PathBuf,
    graph_path: PathBuf,
}

impl Storage {
    pub fn new(dir: &Path, _dim: usize) -> Self {
        Storage {
            data_path: dir.join("meta.vctrs"),
            graph_path: dir.join("graph.vctrs"),
        }
    }

    /// Check if a saved database exists at this path.
    pub fn exists(&self) -> bool {
        self.graph_path.exists()
    }

    /// Save the HNSW graph + metadata to disk.
    pub fn save_full(
        &self,
        index: &HnswIndex,
        meta_records: &[MetaRecord],
    ) -> io::Result<()> {
        // Ensure parent directory exists.
        if let Some(parent) = self.graph_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Save graph (vectors + HNSW structure).
        let graph_tmp = self.graph_path.with_extension("tmp");
        {
            let mut f = File::create(&graph_tmp)?;
            index.save_graph(&mut f)?;
        }
        fs::rename(&graph_tmp, &self.graph_path)?;

        // Save metadata (string ids + JSON metadata).
        let meta_tmp = self.data_path.with_extension("tmp");
        {
            let f = File::create(&meta_tmp)?;
            let mut w = BufWriter::new(f);

            w.write_u32::<LittleEndian>(META_MAGIC)?;
            w.write_u32::<LittleEndian>(META_VERSION)?;
            w.write_u32::<LittleEndian>(meta_records.len() as u32)?;

            for rec in meta_records {
                w.write_u32::<LittleEndian>(rec.internal_id)?;
                let id_bytes = rec.string_id.as_bytes();
                w.write_u32::<LittleEndian>(id_bytes.len() as u32)?;
                w.write_all(id_bytes)?;

                let meta_bytes = match &rec.metadata {
                    Some(m) => serde_json::to_vec(m).unwrap_or_default(),
                    None => Vec::new(),
                };
                w.write_u32::<LittleEndian>(meta_bytes.len() as u32)?;
                w.write_all(&meta_bytes)?;
            }
            w.flush()?;
        }
        fs::rename(&meta_tmp, &self.data_path)?;

        Ok(())
    }

    /// Load the HNSW graph from disk.
    pub fn load_graph(&self) -> io::Result<HnswIndex> {
        let mut f = File::open(&self.graph_path)?;
        HnswIndex::load_graph(&mut f)
    }

    /// Load metadata records from disk.
    pub fn load_meta(&self) -> io::Result<Vec<MetaRecord>> {
        if !self.data_path.exists() {
            return Ok(Vec::new());
        }

        let f = File::open(&self.data_path)?;
        let mut r = BufReader::new(f);

        let magic = r.read_u32::<LittleEndian>()?;
        if magic != META_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad meta magic"));
        }
        let version = r.read_u32::<LittleEndian>()?;
        if version != META_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported meta version: {}", version),
            ));
        }
        let count = r.read_u32::<LittleEndian>()? as usize;

        let mut records = Vec::with_capacity(count);
        for _ in 0..count {
            let internal_id = r.read_u32::<LittleEndian>()?;
            let id_len = r.read_u32::<LittleEndian>()? as usize;
            let mut id_bytes = vec![0u8; id_len];
            r.read_exact(&mut id_bytes)?;
            let string_id = String::from_utf8(id_bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            let meta_len = r.read_u32::<LittleEndian>()? as usize;
            let metadata = if meta_len > 0 {
                let mut meta_bytes = vec![0u8; meta_len];
                r.read_exact(&mut meta_bytes)?;
                serde_json::from_slice(&meta_bytes).ok()
            } else {
                None
            };

            records.push(MetaRecord {
                internal_id,
                string_id,
                metadata,
            });
        }

        Ok(records)
    }

    // -- Legacy v1 support (for migrating old databases) ----------------------

    /// Try loading legacy v1 format (vectors + metadata in single file).
    pub fn load_legacy(&self, legacy_path: &Path) -> io::Result<Vec<Record>> {
        if !legacy_path.exists() {
            return Ok(Vec::new());
        }

        let f = File::open(legacy_path)?;
        let mut r = BufReader::new(f);

        let magic = r.read_u32::<LittleEndian>()?;
        if magic != META_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        let version = r.read_u32::<LittleEndian>()?;
        if version != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a v1 file",
            ));
        }
        let dim = r.read_u32::<LittleEndian>()? as usize;
        let count = r.read_u32::<LittleEndian>()? as usize;

        let mut records = Vec::with_capacity(count);
        for _ in 0..count {
            let id_len = r.read_u32::<LittleEndian>()? as usize;
            let mut id_bytes = vec![0u8; id_len];
            r.read_exact(&mut id_bytes)?;
            let string_id = String::from_utf8(id_bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            let mut vector = Vec::with_capacity(dim);
            for _ in 0..dim {
                vector.push(r.read_f32::<LittleEndian>()?);
            }

            let meta_len = r.read_u32::<LittleEndian>()? as usize;
            let metadata = if meta_len > 0 {
                let mut meta_bytes = vec![0u8; meta_len];
                r.read_exact(&mut meta_bytes)?;
                serde_json::from_slice(&meta_bytes).ok()
            } else {
                None
            };

            records.push(Record {
                string_id,
                vector,
                metadata,
            });
        }

        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;

    #[test]
    fn test_graph_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::new(dir.path(), 3);

        let mut index = HnswIndex::new(3, Metric::Cosine, 16, 200);
        index.insert(vec![1.0, 0.0, 0.0]);
        index.insert(vec![0.0, 1.0, 0.0]);

        let meta_records = vec![
            MetaRecord {
                internal_id: 0,
                string_id: "a".to_string(),
                metadata: Some(serde_json::json!({"x": 1})),
            },
            MetaRecord {
                internal_id: 1,
                string_id: "b".to_string(),
                metadata: None,
            },
        ];

        storage.save_full(&index, &meta_records).unwrap();

        // Load back.
        let loaded_index = storage.load_graph().unwrap();
        let loaded_meta = storage.load_meta().unwrap();

        assert_eq!(loaded_index.len(), 2);
        assert_eq!(loaded_meta.len(), 2);
        assert_eq!(loaded_meta[0].string_id, "a");

        // Search should work.
        let results = loaded_index.search(&[1.0, 0.0, 0.0], 1, 50);
        assert_eq!(results[0].0, 0);
    }
}

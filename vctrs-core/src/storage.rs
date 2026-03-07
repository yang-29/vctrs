/// On-disk storage for the database.
///
/// Two files:
///   vectors.bin — flat f32 array, memory-mapped on load (instant, zero-copy)
///   graph.vctrs — HNSW graph structure + metadata
///
/// Metadata section (appended to graph.vctrs):
///   meta_magic: u32 = 0x4D455441 ("META")
///   num_records: u32
///   for each record:
///     internal_id: u32
///     id_len: u32
///     id: [u8; id_len]
///     meta_len: u32
///     metadata: [u8; meta_len]

use crate::hnsw::HnswIndex;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use memmap2::Mmap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read as IoRead, Write as IoWrite};
use std::path::{Path, PathBuf};

const META_SECTION_MAGIC: u32 = 0x4D455441;

#[derive(Clone, Debug)]
pub struct MetaRecord {
    pub internal_id: u32,
    pub string_id: String,
    pub metadata: Option<serde_json::Value>,
}

pub struct Storage {
    graph_path: PathBuf,
    vectors_path: PathBuf,
}

impl Storage {
    pub fn new(dir: &Path, _dim: usize) -> Self {
        Storage {
            graph_path: dir.join("graph.vctrs"),
            vectors_path: dir.join("vectors.bin"),
        }
    }

    pub fn exists(&self) -> bool {
        self.graph_path.exists() && self.vectors_path.exists()
    }

    /// Save graph + vectors + metadata to disk.
    pub fn save(
        &self,
        index: &HnswIndex,
        meta_records: &[MetaRecord],
    ) -> io::Result<()> {
        if let Some(parent) = self.graph_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Write vectors.bin (flat f32 array — mmap-friendly).
        let vec_tmp = self.vectors_path.with_extension("tmp");
        {
            let f = File::create(&vec_tmp)?;
            let mut w = BufWriter::new(f);
            index.save_vectors(&mut w)?;
        }
        fs::rename(&vec_tmp, &self.vectors_path)?;

        // Write graph.vctrs (structure + metadata).
        let graph_tmp = self.graph_path.with_extension("tmp");
        {
            let f = File::create(&graph_tmp)?;
            let mut w = BufWriter::new(f);

            index.save_graph(&mut w)?;

            // Append metadata section.
            w.write_u32::<LittleEndian>(META_SECTION_MAGIC)?;
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
        fs::rename(&graph_tmp, &self.graph_path)?;

        Ok(())
    }

    /// Load graph + metadata with mmap'd vectors (instant vector access, zero-copy).
    pub fn load(&self) -> io::Result<(HnswIndex, Vec<MetaRecord>)> {
        // Memory-map vectors.
        let vec_file = File::open(&self.vectors_path)?;
        let vectors_mmap = unsafe { Mmap::map(&vec_file)? };

        // Read entire graph file at once for fast parsing.
        let graph_data = fs::read(&self.graph_path)?;
        let (index, remaining) = HnswIndex::load_graph_mmap(&graph_data, vectors_mmap)?;

        // Parse metadata section from remaining bytes.
        let mut cursor = remaining;
        let meta_magic = cursor.read_u32::<LittleEndian>()?;
        if meta_magic != META_SECTION_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad metadata section magic",
            ));
        }
        let count = cursor.read_u32::<LittleEndian>()? as usize;

        let mut records = Vec::with_capacity(count);
        for _ in 0..count {
            let internal_id = cursor.read_u32::<LittleEndian>()?;
            let id_len = cursor.read_u32::<LittleEndian>()? as usize;
            let mut id_bytes = vec![0u8; id_len];
            cursor.read_exact(&mut id_bytes)?;
            let string_id = String::from_utf8(id_bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            let meta_len = cursor.read_u32::<LittleEndian>()? as usize;
            let metadata = if meta_len > 0 {
                let mut meta_bytes = vec![0u8; meta_len];
                cursor.read_exact(&mut meta_bytes)?;
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

        Ok((index, records))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;

    #[test]
    fn test_single_file_roundtrip() {
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

        storage.save(&index, &meta_records).unwrap();

        assert!(dir.path().join("graph.vctrs").exists());
        assert!(dir.path().join("vectors.bin").exists());

        // Load back — vectors are mmap'd.
        let (loaded_index, loaded_meta) = storage.load().unwrap();

        assert_eq!(loaded_index.len(), 2);
        assert!(loaded_index.is_mmap());
        assert_eq!(loaded_meta.len(), 2);
        assert_eq!(loaded_meta[0].string_id, "a");

        let results = loaded_index.search(&[1.0, 0.0, 0.0], 1, 50);
        assert_eq!(results[0].0, 0);
    }
}

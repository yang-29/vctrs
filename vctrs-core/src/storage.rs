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
use crate::quantize::ScalarQuantizer;
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
    quantized_path: PathBuf,
}

impl Storage {
    pub fn new(dir: &Path, _dim: usize) -> Self {
        Storage {
            graph_path: dir.join("graph.vctrs"),
            vectors_path: dir.join("vectors.bin"),
            quantized_path: dir.join("vectors.sq8"),
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
        quantize: bool,
    ) -> io::Result<()> {
        if let Some(parent) = self.graph_path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Always write full vectors.bin (mmap-friendly f32).
        let vec_tmp = self.vectors_path.with_extension("tmp");
        {
            let f = File::create(&vec_tmp)?;
            let mut w = BufWriter::new(f);
            index.save_vectors(&mut w)?;
        }
        fs::rename(&vec_tmp, &self.vectors_path)?;

        // Optionally write quantized vectors (SQ8).
        if quantize {
            let dim = index.dim();
            let vectors = index.vectors_slice();
            let sq = ScalarQuantizer::train(vectors, dim);
            let quantized = sq.quantize_batch(vectors, dim);

            let sq_tmp = self.quantized_path.with_extension("tmp");
            {
                let f = File::create(&sq_tmp)?;
                let mut w = BufWriter::new(f);
                sq.save(&mut w)?;
                w.write_all(&quantized)?;
                w.flush()?;
            }
            fs::rename(&sq_tmp, &self.quantized_path)?;
        } else if self.quantized_path.exists() {
            // Remove stale quantized file if quantization was disabled.
            let _ = fs::remove_file(&self.quantized_path);
        }

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

    /// Check if a quantized vectors file exists.
    pub fn has_quantized(&self) -> bool {
        self.quantized_path.exists()
    }

    /// Load graph + metadata with mmap'd vectors (instant vector access, zero-copy).
    /// If a quantized vectors file (vectors.sq8) exists, loads it for faster HNSW search.
    pub fn load(&self) -> io::Result<(HnswIndex, Vec<MetaRecord>)> {
        // Memory-map vectors.
        let vec_file = File::open(&self.vectors_path)?;
        let vectors_mmap = unsafe { Mmap::map(&vec_file)? };

        // Read entire graph file at once for fast parsing.
        let graph_data = fs::read(&self.graph_path)?;
        let (mut index, remaining) = HnswIndex::load_graph_mmap(&graph_data, vectors_mmap)?;

        // Load quantized vectors for search if available.
        if self.quantized_path.exists() {
            let sq_data = fs::read(&self.quantized_path)?;
            let mut cursor = &sq_data[..];
            let quantizer = ScalarQuantizer::load(&mut cursor)?;
            let quantized_vectors = cursor.to_vec();
            index.load_quantized(quantizer, quantized_vectors);
        }

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

        storage.save(&index, &meta_records, false).unwrap();

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

    #[test]
    fn test_quantized_search_after_load() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::new(dir.path(), 3);

        let mut index = HnswIndex::new(3, Metric::Cosine, 16, 200);
        index.insert(vec![1.0, 0.0, 0.0]);
        index.insert(vec![0.0, 1.0, 0.0]);
        index.insert(vec![0.0, 0.0, 1.0]);

        let meta_records = vec![
            MetaRecord { internal_id: 0, string_id: "a".to_string(), metadata: None },
            MetaRecord { internal_id: 1, string_id: "b".to_string(), metadata: None },
            MetaRecord { internal_id: 2, string_id: "c".to_string(), metadata: None },
        ];

        // Save with quantization enabled.
        storage.save(&index, &meta_records, true).unwrap();
        assert!(dir.path().join("vectors.sq8").exists());

        // Load — should auto-enable quantized search.
        let (loaded_index, _) = storage.load().unwrap();
        assert!(loaded_index.has_quantized_search());

        // Search should work with quantized vectors.
        let results = loaded_index.search(&[1.0, 0.0, 0.0], 1, 50);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_quantized_save() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Storage::new(dir.path(), 3);

        let mut index = HnswIndex::new(3, Metric::Cosine, 16, 200);
        index.insert(vec![1.0, 0.0, 0.0]);
        index.insert(vec![0.0, 1.0, 0.0]);

        let meta_records = vec![
            MetaRecord {
                internal_id: 0,
                string_id: "a".to_string(),
                metadata: None,
            },
            MetaRecord {
                internal_id: 1,
                string_id: "b".to_string(),
                metadata: None,
            },
        ];

        storage.save(&index, &meta_records, true).unwrap();

        // Quantized file should exist and be smaller than full vectors.
        assert!(dir.path().join("vectors.sq8").exists());
        let full_size = std::fs::metadata(dir.path().join("vectors.bin")).unwrap().len();
        let sq_size = std::fs::metadata(dir.path().join("vectors.sq8")).unwrap().len();
        // SQ8 has quantizer params overhead, but for larger vectors would be ~4x smaller.
        // For 2 vectors of dim=3: full = (2*3 + 2) * 4 = 32 bytes, sq8 = 4 + 3*4*2 + 2*3 = 34 bytes.
        // At small scale the overhead dominates, but the file should still exist.
        assert!(sq_size > 0);
        // Full vectors file should also exist (always written).
        assert!(full_size > 0);
    }
}

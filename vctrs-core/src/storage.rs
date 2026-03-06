/// On-disk storage for the entire database in a single file: db.vctrs
///
/// Format: [HNSW graph (vectors + structure)] [metadata section]
///
/// The metadata section is appended after the graph data:
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
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read as IoRead, Write as IoWrite};
use std::path::{Path, PathBuf};

const META_SECTION_MAGIC: u32 = 0x4D455441;

/// Legacy v1 format magic.
const LEGACY_MAGIC: u32 = 0x56435452;

#[derive(Clone, Debug)]
pub struct Record {
    pub string_id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Clone, Debug)]
pub struct MetaRecord {
    pub internal_id: u32,
    pub string_id: String,
    pub metadata: Option<serde_json::Value>,
}

pub struct Storage {
    path: PathBuf,
}

impl Storage {
    pub fn new(dir: &Path, _dim: usize) -> Self {
        Storage {
            path: dir.join("db.vctrs"),
        }
    }

    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    /// Save everything to a single file: graph + metadata.
    pub fn save(
        &self,
        index: &HnswIndex,
        meta_records: &[MetaRecord],
    ) -> io::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }

        let tmp_path = self.path.with_extension("tmp");
        {
            let f = File::create(&tmp_path)?;
            let mut w = BufWriter::new(f);

            // Write HNSW graph (vectors + structure).
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

        fs::rename(&tmp_path, &self.path)?;
        Ok(())
    }

    /// Load everything from the single file.
    pub fn load(&self) -> io::Result<(HnswIndex, Vec<MetaRecord>)> {
        let f = File::open(&self.path)?;
        let mut r = BufReader::new(f);

        // Read HNSW graph.
        let index = HnswIndex::load_graph(&mut r)?;

        // Read metadata section.
        let meta_magic = r.read_u32::<LittleEndian>()?;
        if meta_magic != META_SECTION_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad metadata section magic",
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

        Ok((index, records))
    }

    /// Check for and migrate legacy formats (v1 data.vctrs or v2 split files).
    pub fn legacy_path(&self) -> Option<PathBuf> {
        // v2 split format
        let graph = self.path.parent()?.join("graph.vctrs");
        if graph.exists() {
            return Some(graph);
        }
        // v1 format
        let data = self.path.parent()?.join("data.vctrs");
        if data.exists() {
            return Some(data);
        }
        None
    }

    pub fn load_legacy_v1(&self, path: &Path) -> io::Result<Vec<Record>> {
        let f = File::open(path)?;
        let mut r = BufReader::new(f);

        let magic = r.read_u32::<LittleEndian>()?;
        if magic != LEGACY_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        let version = r.read_u32::<LittleEndian>()?;
        if version != 1 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a v1 file"));
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

            records.push(Record { string_id, vector, metadata });
        }

        Ok(records)
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

        // Should be a single file.
        assert!(dir.path().join("db.vctrs").exists());
        assert!(!dir.path().join("graph.vctrs").exists());
        assert!(!dir.path().join("meta.vctrs").exists());

        // Load back.
        let (loaded_index, loaded_meta) = storage.load().unwrap();

        assert_eq!(loaded_index.len(), 2);
        assert_eq!(loaded_meta.len(), 2);
        assert_eq!(loaded_meta[0].string_id, "a");

        let results = loaded_index.search(&[1.0, 0.0, 0.0], 1, 50);
        assert_eq!(results[0].0, 0);
    }
}

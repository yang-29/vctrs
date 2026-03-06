/// On-disk storage for vectors and metadata.
///
/// File format:
///   [header: 16 bytes]
///   [record]*
///
/// Header:
///   magic: u32 = 0x56435452 ("VCTR")
///   version: u32 = 1
///   dim: u32
///   count: u32
///
/// Record:
///   id_len: u32
///   id: [u8; id_len]        (the string id)
///   vector: [f32; dim]
///   meta_len: u32
///   metadata: [u8; meta_len] (JSON bytes)

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read as IoRead, Write as IoWrite};
use std::path::{Path, PathBuf};

const MAGIC: u32 = 0x56435452;
const VERSION: u32 = 1;

/// A stored record: string id -> vector + metadata.
#[derive(Clone, Debug)]
pub struct Record {
    pub string_id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
}

/// Manages reading/writing the database file.
pub struct Storage {
    path: PathBuf,
    dim: usize,
}

impl Storage {
    pub fn new(path: &Path, dim: usize) -> Self {
        Storage {
            path: path.to_path_buf(),
            dim,
        }
    }

    /// Save all records to disk (full rewrite).
    pub fn save(&self, records: &[Record]) -> io::Result<()> {
        let tmp_path = self.path.with_extension("tmp");
        {
            let file = File::create(&tmp_path)?;
            let mut w = BufWriter::new(file);

            // Header.
            w.write_u32::<LittleEndian>(MAGIC)?;
            w.write_u32::<LittleEndian>(VERSION)?;
            w.write_u32::<LittleEndian>(self.dim as u32)?;
            w.write_u32::<LittleEndian>(records.len() as u32)?;

            // Records.
            for record in records {
                let id_bytes = record.string_id.as_bytes();
                w.write_u32::<LittleEndian>(id_bytes.len() as u32)?;
                w.write_all(id_bytes)?;

                for &val in &record.vector {
                    w.write_f32::<LittleEndian>(val)?;
                }

                let meta_bytes = match &record.metadata {
                    Some(m) => serde_json::to_vec(m).unwrap_or_default(),
                    None => Vec::new(),
                };
                w.write_u32::<LittleEndian>(meta_bytes.len() as u32)?;
                w.write_all(&meta_bytes)?;
            }

            w.flush()?;
        }

        // Atomic rename for crash safety.
        fs::rename(&tmp_path, &self.path)?;
        Ok(())
    }

    /// Load all records from disk.
    pub fn load(&self) -> io::Result<Vec<Record>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&self.path)?;
        let mut r = BufReader::new(file);

        // Header.
        let magic = r.read_u32::<LittleEndian>()?;
        if magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        let version = r.read_u32::<LittleEndian>()?;
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported version: {}", version),
            ));
        }
        let dim = r.read_u32::<LittleEndian>()? as usize;
        if dim != self.dim {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("dimension mismatch: file={}, expected={}", dim, self.dim),
            ));
        }
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

    #[test]
    fn test_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.vctrs");
        let storage = Storage::new(&path, 3);

        let records = vec![
            Record {
                string_id: "doc1".to_string(),
                vector: vec![1.0, 2.0, 3.0],
                metadata: Some(serde_json::json!({"title": "hello"})),
            },
            Record {
                string_id: "doc2".to_string(),
                vector: vec![4.0, 5.0, 6.0],
                metadata: None,
            },
        ];

        storage.save(&records).unwrap();
        let loaded = storage.load().unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].string_id, "doc1");
        assert_eq!(loaded[0].vector, vec![1.0, 2.0, 3.0]);
        assert_eq!(loaded[1].string_id, "doc2");
        assert!(loaded[1].metadata.is_none());
    }

    #[test]
    fn test_load_nonexistent() {
        let storage = Storage::new(Path::new("/tmp/nonexistent_vctrs_test.vctrs"), 3);
        let records = storage.load().unwrap();
        assert!(records.is_empty());
    }
}

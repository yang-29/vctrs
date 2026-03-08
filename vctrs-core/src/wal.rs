/// Write-ahead log for crash recovery.
///
/// Mutations (add, delete, update) are appended to a WAL file before being applied.
/// On startup, the WAL is replayed on top of the last snapshot.
/// `save()` writes a full snapshot and truncates the WAL.
///
/// Format:
///   Each entry: [op_type: u8][payload_len: u32][payload: bytes][checksum: u32]
///   Op types: 1=Add, 2=Delete, 3=Update
///   Checksum: CRC32 of [op_type + payload]

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub enum WalEntry {
    Add {
        id: String,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    },
    Delete {
        id: String,
    },
    Update {
        id: String,
        vector: Option<Vec<f32>>,
        metadata: Option<Option<serde_json::Value>>,
    },
}

const OP_ADD: u8 = 1;
const OP_DELETE: u8 = 2;
const OP_UPDATE: u8 = 3;

/// Simple CRC32 (IEEE) checksum.
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

pub struct Wal {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
}

impl Wal {
    pub fn new(dir: &Path) -> Self {
        Wal {
            path: dir.join("wal.vctrs"),
            writer: None,
        }
    }

    /// Open the WAL file for appending. Creates it if it doesn't exist.
    fn ensure_writer(&mut self) -> io::Result<&mut BufWriter<File>> {
        if self.writer.is_none() {
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)?;
            self.writer = Some(BufWriter::new(file));
        }
        Ok(self.writer.as_mut().unwrap())
    }

    /// Append an entry to the WAL.
    pub fn append(&mut self, entry: &WalEntry) -> io::Result<()> {
        let (op_type, payload) = serialize_entry(entry);

        // Checksum covers op_type + payload.
        let mut checksum_data = vec![op_type];
        checksum_data.extend_from_slice(&payload);
        let checksum = crc32(&checksum_data);

        let w = self.ensure_writer()?;
        w.write_u8(op_type)?;
        w.write_u32::<LittleEndian>(payload.len() as u32)?;
        w.write_all(&payload)?;
        w.write_u32::<LittleEndian>(checksum)?;
        w.flush()?;

        Ok(())
    }

    /// Read all entries from the WAL file. Skips corrupt entries at the end
    /// (partial writes from crashes).
    pub fn read_entries(&self) -> io::Result<Vec<WalEntry>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&self.path)?;
        let file_len = file.metadata()?.len();
        if file_len == 0 {
            return Ok(Vec::new());
        }

        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();

        loop {
            // Try to read op_type.
            let op_type = match reader.read_u8() {
                Ok(v) => v,
                Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            };

            // Read payload length.
            let payload_len = match reader.read_u32::<LittleEndian>() {
                Ok(v) => v as usize,
                Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break, // partial write
                Err(e) => return Err(e),
            };

            // Read payload.
            let mut payload = vec![0u8; payload_len];
            match reader.read_exact(&mut payload) {
                Ok(()) => {}
                Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }

            // Read checksum.
            let stored_checksum = match reader.read_u32::<LittleEndian>() {
                Ok(v) => v,
                Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            };

            // Verify checksum.
            let mut checksum_data = vec![op_type];
            checksum_data.extend_from_slice(&payload);
            let computed_checksum = crc32(&checksum_data);
            if stored_checksum != computed_checksum {
                // Corrupt entry — stop here (remaining entries are unreliable).
                break;
            }

            // Deserialize.
            if let Some(entry) = deserialize_entry(op_type, &payload) {
                entries.push(entry);
            }
        }

        Ok(entries)
    }

    /// Truncate the WAL (called after a successful snapshot/save).
    pub fn truncate(&mut self) -> io::Result<()> {
        // Close the writer first.
        self.writer = None;
        if self.path.exists() {
            fs::remove_file(&self.path)?;
        }
        Ok(())
    }

    /// Whether the WAL file exists and has entries.
    pub fn has_entries(&self) -> bool {
        self.path.exists() && fs::metadata(&self.path).map_or(false, |m| m.len() > 0)
    }
}

// -- Serialization -----------------------------------------------------------

fn serialize_entry(entry: &WalEntry) -> (u8, Vec<u8>) {
    let mut buf = Vec::new();

    match entry {
        WalEntry::Add { id, vector, metadata } => {
            // id_len: u32, id: bytes, vec_len: u32, vector: f32s, meta_len: u32, meta: bytes
            write_string(&mut buf, id);
            write_vector(&mut buf, vector);
            write_metadata(&mut buf, metadata);
            (OP_ADD, buf)
        }
        WalEntry::Delete { id } => {
            write_string(&mut buf, id);
            (OP_DELETE, buf)
        }
        WalEntry::Update { id, vector, metadata } => {
            write_string(&mut buf, id);
            // has_vector: u8 (0 or 1)
            if let Some(vec) = vector {
                buf.push(1);
                write_vector(&mut buf, vec);
            } else {
                buf.push(0);
            }
            // has_metadata: u8 (0 or 1)
            if let Some(meta) = metadata {
                buf.push(1);
                write_metadata(&mut buf, meta);
            } else {
                buf.push(0);
            }
            (OP_UPDATE, buf)
        }
    }
}

fn deserialize_entry(op_type: u8, data: &[u8]) -> Option<WalEntry> {
    let mut cursor = &data[..];

    match op_type {
        OP_ADD => {
            let id = read_string(&mut cursor)?;
            let vector = read_vector(&mut cursor)?;
            let metadata = read_metadata(&mut cursor)?;
            Some(WalEntry::Add { id, vector, metadata })
        }
        OP_DELETE => {
            let id = read_string(&mut cursor)?;
            Some(WalEntry::Delete { id })
        }
        OP_UPDATE => {
            let id = read_string(&mut cursor)?;
            let has_vector = read_u8(&mut cursor)?;
            let vector = if has_vector == 1 {
                Some(read_vector(&mut cursor)?)
            } else {
                None
            };
            let has_metadata = read_u8(&mut cursor)?;
            let metadata = if has_metadata == 1 {
                Some(read_metadata(&mut cursor)?)
            } else {
                None
            };
            Some(WalEntry::Update { id, vector, metadata })
        }
        _ => None,
    }
}

// -- IO helpers --------------------------------------------------------------

fn write_string(buf: &mut Vec<u8>, s: &str) {
    buf.write_u32::<LittleEndian>(s.len() as u32).unwrap();
    buf.extend_from_slice(s.as_bytes());
}

fn write_vector(buf: &mut Vec<u8>, vec: &[f32]) {
    buf.write_u32::<LittleEndian>(vec.len() as u32).unwrap();
    for &v in vec {
        buf.write_f32::<LittleEndian>(v).unwrap();
    }
}

fn write_metadata(buf: &mut Vec<u8>, meta: &Option<serde_json::Value>) {
    match meta {
        Some(v) => {
            let json = serde_json::to_vec(v).unwrap();
            buf.write_u32::<LittleEndian>(json.len() as u32).unwrap();
            buf.extend_from_slice(&json);
        }
        None => {
            buf.write_u32::<LittleEndian>(0).unwrap();
        }
    }
}

fn read_u8(cursor: &mut &[u8]) -> Option<u8> {
    cursor.read_u8().ok()
}

fn read_string(cursor: &mut &[u8]) -> Option<String> {
    let len = cursor.read_u32::<LittleEndian>().ok()? as usize;
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

fn read_vector(cursor: &mut &[u8]) -> Option<Vec<f32>> {
    let len = cursor.read_u32::<LittleEndian>().ok()? as usize;
    let mut vec = Vec::with_capacity(len);
    for _ in 0..len {
        vec.push(cursor.read_f32::<LittleEndian>().ok()?);
    }
    Some(vec)
}

fn read_metadata(cursor: &mut &[u8]) -> Option<Option<serde_json::Value>> {
    let len = cursor.read_u32::<LittleEndian>().ok()? as usize;
    if len == 0 {
        return Some(None);
    }
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf).ok()?;
    let val = serde_json::from_slice(&buf).ok()?;
    Some(Some(val))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut wal = Wal::new(dir.path());

        // Append several entries.
        wal.append(&WalEntry::Add {
            id: "a".into(),
            vector: vec![1.0, 2.0, 3.0],
            metadata: Some(serde_json::json!({"k": "v"})),
        }).unwrap();

        wal.append(&WalEntry::Delete { id: "b".into() }).unwrap();

        wal.append(&WalEntry::Update {
            id: "a".into(),
            vector: Some(vec![4.0, 5.0, 6.0]),
            metadata: None,
        }).unwrap();

        wal.append(&WalEntry::Update {
            id: "a".into(),
            vector: None,
            metadata: Some(Some(serde_json::json!({"new": true}))),
        }).unwrap();

        // Read them back.
        let entries = wal.read_entries().unwrap();
        assert_eq!(entries.len(), 4);

        match &entries[0] {
            WalEntry::Add { id, vector, metadata } => {
                assert_eq!(id, "a");
                assert_eq!(vector, &[1.0, 2.0, 3.0]);
                assert_eq!(metadata.as_ref().unwrap()["k"], "v");
            }
            _ => panic!("expected Add"),
        }

        match &entries[1] {
            WalEntry::Delete { id } => assert_eq!(id, "b"),
            _ => panic!("expected Delete"),
        }

        match &entries[2] {
            WalEntry::Update { id, vector, metadata } => {
                assert_eq!(id, "a");
                assert_eq!(vector.as_ref().unwrap(), &[4.0, 5.0, 6.0]);
                assert!(metadata.is_none());
            }
            _ => panic!("expected Update"),
        }

        match &entries[3] {
            WalEntry::Update { id, vector, metadata } => {
                assert_eq!(id, "a");
                assert!(vector.is_none());
                assert_eq!(metadata.as_ref().unwrap().as_ref().unwrap()["new"], true);
            }
            _ => panic!("expected Update"),
        }
    }

    #[test]
    fn test_wal_truncate() {
        let dir = tempfile::tempdir().unwrap();
        let mut wal = Wal::new(dir.path());

        wal.append(&WalEntry::Add {
            id: "a".into(),
            vector: vec![1.0],
            metadata: None,
        }).unwrap();

        assert!(wal.has_entries());

        wal.truncate().unwrap();
        assert!(!wal.has_entries());

        let entries = wal.read_entries().unwrap();
        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn test_wal_empty() {
        let dir = tempfile::tempdir().unwrap();
        let wal = Wal::new(dir.path());

        assert!(!wal.has_entries());
        let entries = wal.read_entries().unwrap();
        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn test_wal_corrupt_entry_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let mut wal = Wal::new(dir.path());

        // Write a valid entry.
        wal.append(&WalEntry::Add {
            id: "a".into(),
            vector: vec![1.0, 2.0],
            metadata: None,
        }).unwrap();

        // Append garbage bytes to simulate a partial write / crash.
        wal.writer = None; // close the buffered writer
        let mut file = OpenOptions::new().append(true).open(&wal.path).unwrap();
        file.write_all(&[0xFF, 0xFF, 0xFF]).unwrap();

        // Reading should return the valid entry and skip the corrupt tail.
        let entries = wal.read_entries().unwrap();
        assert_eq!(entries.len(), 1);
        match &entries[0] {
            WalEntry::Add { id, .. } => assert_eq!(id, "a"),
            _ => panic!("expected Add"),
        }
    }

    #[test]
    fn test_wal_checksum_detects_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let mut wal = Wal::new(dir.path());

        wal.append(&WalEntry::Add {
            id: "a".into(),
            vector: vec![1.0],
            metadata: None,
        }).unwrap();
        wal.append(&WalEntry::Add {
            id: "b".into(),
            vector: vec![2.0],
            metadata: None,
        }).unwrap();
        wal.writer = None;

        // Corrupt one byte in the middle of the file (entry 2's payload).
        let mut data = fs::read(&wal.path).unwrap();
        let midpoint = data.len() / 2 + 5; // somewhere in the second entry
        data[midpoint] ^= 0xFF;
        fs::write(&wal.path, &data).unwrap();

        // Should recover at least the first entry.
        let entries = wal.read_entries().unwrap();
        assert!(entries.len() >= 1);
        match &entries[0] {
            WalEntry::Add { id, .. } => assert_eq!(id, "a"),
            _ => panic!("expected Add"),
        }
    }

    #[test]
    fn test_crc32() {
        // Known test vectors.
        assert_eq!(crc32(b""), 0x00000000);
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }
}

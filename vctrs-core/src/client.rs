/// Multi-collection client: manages named collections in a root directory.
///
/// Each collection is a `Database` stored in its own subdirectory:
///   {root}/{name}/graph.vctrs
///   {root}/{name}/vectors.bin
///   ...

use crate::db::{Database, HnswConfig};
use crate::distance::Metric;
use crate::error::{Result, VctrsError};
use std::fs;
use std::path::PathBuf;

pub struct Client {
    root: PathBuf,
}

impl Client {
    /// Open or create a client rooted at the given directory.
    pub fn new(path: &str) -> Result<Self> {
        let root = PathBuf::from(path);
        fs::create_dir_all(&root)?;
        Ok(Client { root })
    }

    /// Create a new collection. Fails if it already exists.
    pub fn create_collection(
        &self,
        name: &str,
        dim: usize,
        metric: Metric,
    ) -> Result<Database> {
        self.create_collection_with_config(name, dim, metric, HnswConfig::default())
    }

    /// Create a new collection with custom HNSW parameters.
    pub fn create_collection_with_config(
        &self,
        name: &str,
        dim: usize,
        metric: Metric,
        config: HnswConfig,
    ) -> Result<Database> {
        validate_name(name)?;
        let col_path = self.root.join(name);
        if col_path.exists() {
            return Err(VctrsError::CollectionExists(name.to_string()));
        }
        let path_str = col_path.to_string_lossy().to_string();
        Database::open_or_create_with_config(&path_str, dim, metric, config)
    }

    /// Get an existing collection by name.
    pub fn get_collection(&self, name: &str) -> Result<Database> {
        validate_name(name)?;
        let col_path = self.root.join(name);
        let path_str = col_path.to_string_lossy().to_string();
        Database::open(&path_str)
    }

    /// Get or create a collection. If it exists, opens it (dim/metric ignored).
    /// If it doesn't exist, creates it with the given dim and metric.
    pub fn get_or_create_collection(
        &self,
        name: &str,
        dim: usize,
        metric: Metric,
    ) -> Result<Database> {
        validate_name(name)?;
        let col_path = self.root.join(name);
        let path_str = col_path.to_string_lossy().to_string();
        Database::open_or_create(&path_str, dim, metric)
    }

    /// Delete a collection and all its data. Returns true if it existed.
    pub fn delete_collection(&self, name: &str) -> Result<bool> {
        validate_name(name)?;
        let col_path = self.root.join(name);
        if !col_path.exists() {
            return Ok(false);
        }
        fs::remove_dir_all(&col_path)?;
        Ok(true)
    }

    /// List all collection names.
    pub fn list_collections(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    // Only include directories that look like vctrs collections.
                    let graph = entry.path().join("graph.vctrs");
                    if graph.exists() {
                        names.push(name.to_string());
                    }
                }
            }
        }
        names.sort();
        Ok(names)
    }
}

/// Validate collection name: must be non-empty, alphanumeric + hyphens/underscores, no path traversal.
fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(VctrsError::InvalidCollectionName("name cannot be empty".to_string()));
    }
    if name.contains('/') || name.contains('\\') || name == "." || name == ".." {
        return Err(VctrsError::InvalidCollectionName(
            "name cannot contain path separators or be '.' or '..'".to_string(),
        ));
    }
    if !name.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
        return Err(VctrsError::InvalidCollectionName(
            "name must be alphanumeric, hyphens, or underscores".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_list() {
        let dir = tempfile::tempdir().unwrap();
        let client = Client::new(dir.path().to_str().unwrap()).unwrap();

        let db = client.create_collection("movies", 3, Metric::Cosine).unwrap();
        db.add("m1", vec![1.0, 0.0, 0.0], None).unwrap();
        db.save().unwrap();

        let names = client.list_collections().unwrap();
        assert_eq!(names, vec!["movies"]);
    }

    #[test]
    fn test_get_collection() {
        let dir = tempfile::tempdir().unwrap();
        let client = Client::new(dir.path().to_str().unwrap()).unwrap();

        let db = client.create_collection("docs", 4, Metric::Cosine).unwrap();
        db.add("d1", vec![1.0, 0.0, 0.0, 0.0], None).unwrap();
        db.save().unwrap();
        drop(db);

        let db2 = client.get_collection("docs").unwrap();
        assert_eq!(db2.len(), 1);
    }

    #[test]
    fn test_get_or_create() {
        let dir = tempfile::tempdir().unwrap();
        let client = Client::new(dir.path().to_str().unwrap()).unwrap();

        let db = client.get_or_create_collection("items", 3, Metric::Cosine).unwrap();
        db.add("i1", vec![1.0, 0.0, 0.0], None).unwrap();
        db.save().unwrap();
        drop(db);

        // Opening again should work and return existing data.
        let db2 = client.get_or_create_collection("items", 3, Metric::Cosine).unwrap();
        assert_eq!(db2.len(), 1);
    }

    #[test]
    fn test_delete_collection() {
        let dir = tempfile::tempdir().unwrap();
        let client = Client::new(dir.path().to_str().unwrap()).unwrap();

        let db = client.create_collection("temp", 2, Metric::Cosine).unwrap();
        db.add("t1", vec![1.0, 0.0], None).unwrap();
        db.save().unwrap();
        drop(db);

        assert!(client.delete_collection("temp").unwrap());
        assert!(!client.delete_collection("temp").unwrap());
        assert!(client.list_collections().unwrap().is_empty());
    }

    #[test]
    fn test_duplicate_collection() {
        let dir = tempfile::tempdir().unwrap();
        let client = Client::new(dir.path().to_str().unwrap()).unwrap();

        let db = client.create_collection("x", 2, Metric::Cosine).unwrap();
        db.save().unwrap();

        let result = client.create_collection("x", 2, Metric::Cosine);
        assert!(matches!(result, Err(VctrsError::CollectionExists(_))));
    }

    #[test]
    fn test_invalid_names() {
        let dir = tempfile::tempdir().unwrap();
        let client = Client::new(dir.path().to_str().unwrap()).unwrap();

        assert!(client.create_collection("", 2, Metric::Cosine).is_err());
        assert!(client.create_collection("../evil", 2, Metric::Cosine).is_err());
        assert!(client.create_collection("a/b", 2, Metric::Cosine).is_err());
        assert!(client.create_collection("..", 2, Metric::Cosine).is_err());
        assert!(client.create_collection("has space", 2, Metric::Cosine).is_err());
    }

    #[test]
    fn test_multiple_collections() {
        let dir = tempfile::tempdir().unwrap();
        let client = Client::new(dir.path().to_str().unwrap()).unwrap();

        let movies = client.create_collection("movies", 3, Metric::Cosine).unwrap();
        movies.add("m1", vec![1.0, 0.0, 0.0], None).unwrap();
        movies.save().unwrap();

        let docs = client.create_collection("docs", 768, Metric::DotProduct).unwrap();
        docs.add("d1", vec![0.1; 768], None).unwrap();
        docs.save().unwrap();

        let names = client.list_collections().unwrap();
        assert_eq!(names, vec!["docs", "movies"]);

        // Each collection has independent dim/metric.
        assert_eq!(movies.dim(), 3);
        assert_eq!(docs.dim(), 768);
    }
}

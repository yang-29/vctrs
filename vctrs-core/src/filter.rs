/// Filter types and metadata indexing for filtered vector search.

use crate::error::{VctrsError, Result};
use std::collections::{HashMap, HashSet};

/// Filter predicate for metadata-filtered search.
#[derive(Debug, Clone)]
pub enum Filter {
    /// Field equals value.
    Eq(String, serde_json::Value),
    /// Field not equals value.
    Ne(String, serde_json::Value),
    /// Field is in list of values.
    In(String, Vec<serde_json::Value>),
    /// Field greater than value (numeric).
    Gt(String, f64),
    /// Field greater than or equal to value (numeric).
    Gte(String, f64),
    /// Field less than value (numeric).
    Lt(String, f64),
    /// Field less than or equal to value (numeric).
    Lte(String, f64),
    /// All sub-filters must match.
    And(Vec<Filter>),
    /// Any sub-filter must match.
    Or(Vec<Filter>),
}

/// Extract a JSON value as f64 for numeric comparison.
fn as_f64(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

impl Filter {
    /// Try to resolve this filter to a set of matching internal IDs using the inverted index.
    /// Returns Some(set) for $eq and $in filters. Returns None for operators that
    /// require per-candidate evaluation ($ne, $gt, $lt, etc.).
    pub(crate) fn resolve_from_index(&self, meta_index: &MetadataIndex) -> Option<HashSet<u32>> {
        match self {
            Filter::Eq(key, val) => {
                Some(meta_index.get_eq(key, val).cloned().unwrap_or_default())
            }
            Filter::In(key, vals) => {
                Some(meta_index.get_in(key, vals).unwrap_or_default())
            }
            Filter::And(filters) => {
                // If ALL sub-filters can be resolved, intersect their results.
                // If any can't, return None (fall back to per-candidate).
                let mut result: Option<HashSet<u32>> = None;
                let mut has_unresolvable = false;
                for f in filters {
                    if let Some(ids) = f.resolve_from_index(meta_index) {
                        result = Some(match result {
                            Some(existing) => existing.intersection(&ids).copied().collect(),
                            None => ids,
                        });
                    } else {
                        has_unresolvable = true;
                    }
                }
                // If we resolved at least one sub-filter, return that (even if some
                // sub-filters are unresolvable — we'll still scan those but over a
                // smaller candidate set).
                if result.is_some() {
                    result
                } else if has_unresolvable {
                    None
                } else {
                    Some(HashSet::new())
                }
            }
            // $ne, $gt, $gte, $lt, $lte, $or cannot be efficiently resolved from an eq-index.
            _ => None,
        }
    }

    pub fn matches(&self, metadata: &Option<serde_json::Value>) -> bool {
        let obj = match metadata {
            Some(serde_json::Value::Object(m)) => m,
            _ => return false,
        };

        match self {
            Filter::Eq(key, val) => obj.get(key).map_or(false, |v| v == val),
            Filter::Ne(key, val) => obj.get(key).map_or(true, |v| v != val),
            Filter::In(key, vals) => obj.get(key).map_or(false, |v| vals.contains(v)),
            Filter::Gt(key, threshold) => {
                obj.get(key).and_then(as_f64).map_or(false, |v| v > *threshold)
            }
            Filter::Gte(key, threshold) => {
                obj.get(key).and_then(as_f64).map_or(false, |v| v >= *threshold)
            }
            Filter::Lt(key, threshold) => {
                obj.get(key).and_then(as_f64).map_or(false, |v| v < *threshold)
            }
            Filter::Lte(key, threshold) => {
                obj.get(key).and_then(as_f64).map_or(false, |v| v <= *threshold)
            }
            Filter::And(filters) => filters.iter().all(|f| f.matches(metadata)),
            Filter::Or(filters) => filters.iter().any(|f| f.matches(metadata)),
        }
    }
}

/// Inverted index over metadata fields for fast filtered search.
/// Maps field_name → value → set of internal IDs that have that field=value.
/// Only indexes top-level scalar fields (strings, numbers, bools).
pub(crate) struct MetadataIndex {
    /// field_name → value → set of internal IDs.
    fields: HashMap<String, HashMap<serde_json::Value, HashSet<u32>>>,
}

impl MetadataIndex {
    pub(crate) fn new() -> Self {
        MetadataIndex { fields: HashMap::new() }
    }

    /// Index all scalar fields from a metadata object for a given internal ID.
    pub(crate) fn index(&mut self, id: u32, metadata: &Option<serde_json::Value>) {
        let obj = match metadata {
            Some(serde_json::Value::Object(m)) => m,
            _ => return,
        };
        for (key, val) in obj {
            // Only index scalar values (string, number, bool). Skip null, arrays, objects.
            if val.is_null() || val.is_array() || val.is_object() {
                continue;
            }
            self.fields
                .entry(key.clone())
                .or_default()
                .entry(val.clone())
                .or_default()
                .insert(id);
        }
    }

    /// Remove all entries for a given internal ID.
    pub(crate) fn remove(&mut self, id: u32, metadata: &Option<serde_json::Value>) {
        let obj = match metadata {
            Some(serde_json::Value::Object(m)) => m,
            _ => return,
        };
        for (key, val) in obj {
            if val.is_null() || val.is_array() || val.is_object() {
                continue;
            }
            if let Some(value_map) = self.fields.get_mut(key) {
                if let Some(id_set) = value_map.get_mut(val) {
                    id_set.remove(&id);
                    if id_set.is_empty() {
                        value_map.remove(val);
                    }
                }
                if value_map.is_empty() {
                    self.fields.remove(key);
                }
            }
        }
    }

    /// Get the set of IDs matching an $eq filter. Returns None if the field isn't indexed.
    pub(crate) fn get_eq(&self, field: &str, value: &serde_json::Value) -> Option<&HashSet<u32>> {
        self.fields.get(field)?.get(value)
    }

    /// Get the union of ID sets matching an $in filter.
    pub(crate) fn get_in(&self, field: &str, values: &[serde_json::Value]) -> Option<HashSet<u32>> {
        let value_map = self.fields.get(field)?;
        let mut result = HashSet::new();
        for val in values {
            if let Some(ids) = value_map.get(val) {
                result.extend(ids);
            }
        }
        Some(result)
    }

    /// Rebuild the entire index from scratch.
    pub(crate) fn rebuild(&mut self, metadata: &[Option<serde_json::Value>], deleted: &HashSet<u32>) {
        self.fields.clear();
        for (i, meta) in metadata.iter().enumerate() {
            let id = i as u32;
            if deleted.contains(&id) {
                continue;
            }
            self.index(id, meta);
        }
    }
}

/// Parse a JSON filter object into a Filter.
/// Supports: {"field": "value"}, {"field": {"$ne": "value"}}, {"field": {"$in": [...]}}
/// Multiple keys = AND.
pub fn parse_json_filter(value: &serde_json::Value) -> Result<Filter> {
    let obj = value.as_object()
        .ok_or_else(|| VctrsError::InvalidFilter("filter must be an object".to_string()))?;

    let mut filters = Vec::new();

    for (key, val) in obj {
        if let Some(op_obj) = val.as_object() {
            for (op, op_val) in op_obj {
                match op.as_str() {
                    "$eq" => filters.push(Filter::Eq(key.clone(), op_val.clone())),
                    "$ne" => filters.push(Filter::Ne(key.clone(), op_val.clone())),
                    "$in" => {
                        let arr = op_val.as_array()
                            .ok_or_else(|| VctrsError::InvalidFilter("$in value must be an array".to_string()))?;
                        filters.push(Filter::In(key.clone(), arr.clone()));
                    }
                    "$gt" => {
                        let n = op_val.as_f64()
                            .ok_or_else(|| VctrsError::InvalidFilter("$gt value must be a number".to_string()))?;
                        filters.push(Filter::Gt(key.clone(), n));
                    }
                    "$gte" => {
                        let n = op_val.as_f64()
                            .ok_or_else(|| VctrsError::InvalidFilter("$gte value must be a number".to_string()))?;
                        filters.push(Filter::Gte(key.clone(), n));
                    }
                    "$lt" => {
                        let n = op_val.as_f64()
                            .ok_or_else(|| VctrsError::InvalidFilter("$lt value must be a number".to_string()))?;
                        filters.push(Filter::Lt(key.clone(), n));
                    }
                    "$lte" => {
                        let n = op_val.as_f64()
                            .ok_or_else(|| VctrsError::InvalidFilter("$lte value must be a number".to_string()))?;
                        filters.push(Filter::Lte(key.clone(), n));
                    }
                    _ => return Err(VctrsError::InvalidFilter(format!("unknown operator: {}", op))),
                }
            }
        } else {
            filters.push(Filter::Eq(key.clone(), val.clone()));
        }
    }

    if filters.len() == 1 {
        Ok(filters.into_iter().next().unwrap())
    } else {
        Ok(Filter::And(filters))
    }
}

use crate::quantize::ScalarQuantizer;
#[cfg(feature = "mmap")]
use memmap2::Mmap;

// -- Vector storage ----------------------------------------------------------

/// Vector data can be owned (in-memory, mutable) or memory-mapped (read-only, instant load).
pub(crate) enum VectorStore {
    /// Heap-allocated vectors — used for new databases and after mutations.
    Owned(Vec<f32>),
    /// Memory-mapped file containing [vectors (num_vectors_f32)][norms (num_norms_f32)].
    /// `vec_len` is the number of f32s in the vectors portion.
    #[cfg(feature = "mmap")]
    Mapped { mmap: Mmap, vec_len: usize },
}

impl VectorStore {
    /// Get the vectors portion as f32 slice.
    pub fn as_slice(&self) -> &[f32] {
        match self {
            VectorStore::Owned(v) => v,
            #[cfg(feature = "mmap")]
            VectorStore::Mapped { mmap, vec_len } => {
                let ptr = mmap.as_ptr() as *const f32;
                unsafe { std::slice::from_raw_parts(ptr, *vec_len) }
            }
        }
    }

    /// Get the norms portion from a Mapped store. Returns None for Owned.
    pub fn mapped_norms(&self) -> Option<Vec<f32>> {
        match self {
            VectorStore::Owned(_) => None,
            #[cfg(feature = "mmap")]
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
    pub fn as_mut_vec(&mut self) -> &mut Vec<f32> {
        match self {
            VectorStore::Owned(v) => v,
            #[cfg(feature = "mmap")]
            VectorStore::Mapped { .. } => panic!("cannot mutate mmap'd vectors — save and reopen, or use owned mode"),
        }
    }

    /// Convert from Mapped to Owned (copies data into heap). Required before mutations.
    pub fn ensure_owned(&mut self) {
        #[cfg(feature = "mmap")]
        if let VectorStore::Mapped { .. } = self {
            let data = self.as_slice().to_vec();
            *self = VectorStore::Owned(data);
        }
    }

    pub fn is_mapped(&self) -> bool {
        #[cfg(feature = "mmap")]
        { matches!(self, VectorStore::Mapped { .. }) }
        #[cfg(not(feature = "mmap"))]
        { false }
    }
}

// -- HNSW Index --------------------------------------------------------------

/// Optional quantized vector storage for faster HNSW traversal.
/// Quantized vectors are 4x smaller (u8 vs f32), improving cache utilization
/// during the random-access pattern of graph traversal.
pub(crate) struct QuantizedSearch {
    pub quantizer: ScalarQuantizer,
    /// Flat u8 array: n vectors of `dim` bytes each.
    pub vectors: Vec<u8>,
}

impl QuantizedSearch {
    /// Dequantize a vector into a pre-allocated f32 buffer.
    #[inline]
    pub fn dequantize_into(&self, id: u32, dim: usize, buf: &mut [f32]) {
        let start = id as usize * dim;
        self.quantizer.dequantize_into(&self.vectors[start..start + dim], buf);
    }
}

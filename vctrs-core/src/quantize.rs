/// Scalar quantization (SQ8): compress f32 vectors to u8 with per-dimension min/max scaling.
/// 4x memory reduction with minimal recall loss (~99%+ recall on typical workloads).
///
/// Each f32 value is linearly mapped to [0, 255] using per-dimension min/max:
///   quantized = round((value - min) / (max - min) * 255)
///   reconstructed = quantized / 255 * (max - min) + min

/// Quantization parameters: min and max for each dimension.
#[derive(Clone, Debug)]
pub struct ScalarQuantizer {
    pub dim: usize,
    /// Per-dimension minimum values.
    pub mins: Vec<f32>,
    /// Per-dimension scale factors: (max - min) / 255.
    pub scales: Vec<f32>,
}

impl ScalarQuantizer {
    /// Train quantizer from a set of vectors (flat layout: vectors[i*dim..(i+1)*dim]).
    pub fn train(vectors: &[f32], dim: usize) -> Self {
        let n = vectors.len() / dim;
        let mut mins = vec![f32::MAX; dim];
        let mut maxs = vec![f32::MIN; dim];

        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            for (d, &val) in v.iter().enumerate() {
                if val < mins[d] {
                    mins[d] = val;
                }
                if val > maxs[d] {
                    maxs[d] = val;
                }
            }
        }

        let scales: Vec<f32> = mins
            .iter()
            .zip(maxs.iter())
            .map(|(&mn, &mx)| {
                let range = mx - mn;
                if range == 0.0 { 1.0 } else { range / 255.0 }
            })
            .collect();

        ScalarQuantizer { dim, mins, scales }
    }

    /// Quantize a single f32 vector to u8.
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        vector
            .iter()
            .enumerate()
            .map(|(d, &val)| {
                let normalized = (val - self.mins[d]) / self.scales[d];
                normalized.clamp(0.0, 255.0).round() as u8
            })
            .collect()
    }

    /// Quantize many vectors (flat f32 layout) to flat u8 layout.
    pub fn quantize_batch(&self, vectors: &[f32], dim: usize) -> Vec<u8> {
        let n = vectors.len() / dim;
        let mut out = Vec::with_capacity(n * dim);
        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            out.extend(self.quantize(v));
        }
        out
    }

    /// Dequantize a u8 vector back to f32.
    pub fn dequantize(&self, quantized: &[u8]) -> Vec<f32> {
        quantized
            .iter()
            .enumerate()
            .map(|(d, &val)| val as f32 * self.scales[d] + self.mins[d])
            .collect()
    }

    /// Dequantize a single vector in-place into a pre-allocated buffer.
    #[inline]
    pub fn dequantize_into(&self, quantized: &[u8], out: &mut [f32]) {
        for (d, &val) in quantized.iter().enumerate() {
            out[d] = val as f32 * self.scales[d] + self.mins[d];
        }
    }

    /// Serialized size in bytes: 4 (dim) + dim*4 (mins) + dim*4 (scales).
    pub fn serialized_size(&self) -> usize {
        4 + self.dim * 4 * 2
    }

    /// Write quantizer parameters.
    pub fn save<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        use byteorder::{LittleEndian, WriteBytesExt};
        w.write_u32::<LittleEndian>(self.dim as u32)?;
        for &m in &self.mins {
            w.write_f32::<LittleEndian>(m)?;
        }
        for &s in &self.scales {
            w.write_f32::<LittleEndian>(s)?;
        }
        Ok(())
    }

    /// Read quantizer parameters.
    pub fn load<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        use byteorder::{LittleEndian, ReadBytesExt};
        let dim = r.read_u32::<LittleEndian>()? as usize;
        let mut mins = Vec::with_capacity(dim);
        for _ in 0..dim {
            mins.push(r.read_f32::<LittleEndian>()?);
        }
        let mut scales = Vec::with_capacity(dim);
        for _ in 0..dim {
            scales.push(r.read_f32::<LittleEndian>()?);
        }
        Ok(ScalarQuantizer { dim, mins, scales })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let vectors = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            0.0, 0.0, 0.0,
        ];
        let sq = ScalarQuantizer::train(&vectors, 3);

        let q = sq.quantize(&[2.0, 3.5, 4.5]);
        let reconstructed = sq.dequantize(&q);

        for i in 0..3 {
            assert!(
                (reconstructed[i] - [2.0, 3.5, 4.5][i]).abs() < 0.05,
                "dim {}: expected ~{}, got {}",
                i, [2.0, 3.5, 4.5][i], reconstructed[i]
            );
        }
    }

    #[test]
    fn test_boundary_values() {
        // Two 2D vectors: [0, 1] and [10, 100].
        let vectors = vec![0.0, 1.0, 10.0, 100.0];
        let sq = ScalarQuantizer::train(&vectors, 2);

        // Min values [0.0, 1.0] should quantize to [0, 0].
        let q = sq.quantize(&[0.0, 1.0]);
        assert_eq!(q[0], 0);
        assert_eq!(q[1], 0);

        // Max values [10.0, 100.0] should quantize to [255, 255].
        let q = sq.quantize(&[10.0, 100.0]);
        assert_eq!(q[0], 255);
        assert_eq!(q[1], 255);
    }

    #[test]
    fn test_save_load() {
        let vectors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let sq = ScalarQuantizer::train(&vectors, 3);

        let mut buf = Vec::new();
        sq.save(&mut buf).unwrap();

        let loaded = ScalarQuantizer::load(&mut &buf[..]).unwrap();
        assert_eq!(loaded.dim, 3);
        assert_eq!(loaded.mins, sq.mins);
        assert_eq!(loaded.scales, sq.scales);
    }
}

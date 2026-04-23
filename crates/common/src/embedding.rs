// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/common/src/embedding.rs
//
// Dense vector embedding produced by the neural embedder (BGE-large-en-v1.5).

/// A 1024-dimensional dense embedding vector.
///
/// Produced by `NeuralEngine::embed()` and consumed by `SemanticIndex`.
/// The dimensionality matches BAAI/bge-large-en-v1.5 (feature-extraction).
#[derive(Clone, Debug)]
pub struct Embedding {
    /// Raw f32 components. Length must be 1024 for BGE-large-en-v1.5.
    pub values: Vec<f32>,
}

impl Embedding {
    /// The output dimensionality of BAAI/bge-large-en-v1.5.
    pub const DIM: usize = 1024;

    /// Wraps a pre-computed vector of values.
    pub fn new(values: Vec<f32>) -> Self {
        Self { values }
    }

    /// Returns the number of dimensions.
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// Computes the cosine similarity between `self` and `other`.
    ///
    /// Returns 0.0 if either vector has zero magnitude.
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot: f32 = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum();
        let mag_a: f32 = self.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mag_b: f32 = other.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }
        dot / (mag_a * mag_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_similarity_one() {
        let v = Embedding::new(vec![1.0, 0.0, 0.0]);
        assert!((v.cosine_similarity(&v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_vectors_have_similarity_zero() {
        let a = Embedding::new(vec![1.0, 0.0]);
        let b = Embedding::new(vec![0.0, 1.0]);
        assert!(a.cosine_similarity(&b).abs() < 1e-6);
    }
}

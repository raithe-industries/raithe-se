// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/common/src/simhash.rs
//
// 64-bit SimHash fingerprint with Hamming-distance near-duplicate detection.

use serde::{Deserialize, Serialize};

/// A 64-bit SimHash fingerprint of a document's content.
///
/// Two documents are considered near-duplicates when their Hamming distance
/// is at or below the detection threshold (typically 3 bits).
#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    PartialEq,
    Deserialize,
    Serialize,
)]
pub struct SimHash(u64);

impl SimHash {
    /// Wraps a pre-computed `fingerprint`.
    pub fn new(fingerprint: u64) -> Self {
        Self(fingerprint)
    }

    /// Returns the raw 64-bit fingerprint.
    pub fn get(self) -> u64 {
        self.0
    }

    /// Returns the Hamming distance between `self` and `other`.
    ///
    /// Counts the number of bit positions that differ. A distance of zero
    /// means the fingerprints are identical.
    pub fn hamming_distance(self, other: Self) -> u32 {
        (self.0 ^ other.0).count_ones()
    }

    /// Returns `true` when `self` and `other` are near-duplicates.
    ///
    /// Uses the given `threshold` — documents with a Hamming distance at or
    /// below the threshold are treated as duplicates.
    pub fn is_near_duplicate(self, other: Self, threshold: u32) -> bool {
        self.hamming_distance(other) <= threshold
    }

    /// Computes a SimHash fingerprint from the given token iterator.
    ///
    /// Each token is hashed with a simple 64-bit mix; its bits are used to
    /// update per-bit counters. The sign of each counter determines the
    /// corresponding output bit.
    pub fn from_tokens<'tok>(tokens: impl Iterator<Item = &'tok str>) -> Self {
        let mut counts = [0i32; 64];
        for token in tokens {
            let hash = hash_token(token);
            for bit in 0..64u32 {
                let vote = if (hash >> bit) & 1 == 1 { 1i32 } else { -1i32 };
                counts[bit as usize] += vote;
            }
        }
        let fingerprint = counts
            .iter()
            .enumerate()
            .fold(0u64, |acc, (bit, &count)| {
                if count > 0 {
                    acc | (1u64 << bit)
                } else {
                    acc
                }
            });
        Self(fingerprint)
    }
}

/// 64-bit mix hash for a token string.
fn hash_token(token: &str) -> u64 {
    use std::hash::Hash;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    token.hash(&mut hasher);
    std::hash::Hasher::finish(&hasher)
}

impl std::fmt::Display for SimHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Lowercase hex per RBP §Hex values.
        write!(f, "{:016x}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_fingerprints_have_zero_distance() {
        let a = SimHash::new(0xdeadbeef_cafebabe);
        assert_eq!(a.hamming_distance(a), 0);
    }

    #[test]
    fn known_hamming_distance() {
        let a = SimHash::new(0b1010);
        let b = SimHash::new(0b0101);
        assert_eq!(a.hamming_distance(b), 4);
    }

    #[test]
    fn near_duplicate_threshold() {
        let a = SimHash::new(0b1111);
        let b = SimHash::new(0b1110);
        assert!(a.is_near_duplicate(b, 1));
        assert!(!a.is_near_duplicate(b, 0));
    }

    #[test]
    fn display_is_lowercase_hex() {
        let s = SimHash::new(0xab5c4d320974a3bc);
        assert_eq!(s.to_string(), "ab5c4d320974a3bc");
    }

    #[test]
    fn from_tokens_is_deterministic() {
        let tokens = ["the", "quick", "brown", "fox"];
        let a = SimHash::from_tokens(tokens.iter().copied());
        let b = SimHash::from_tokens(tokens.iter().copied());
        assert_eq!(a, b);
    }
}

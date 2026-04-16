// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/semantic/src/lib.rs
//
// HNSW approximate nearest-neighbour index for dense embedding retrieval.
// Implemented from scratch per §5.12 — no external HNSW crate dependency.

use std::collections::{BinaryHeap, HashMap, HashSet};

use raithe_common::{DocumentId, Embedding};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("document {id} already present in the index")]
    Duplicate { id: DocumentId },
    #[error("index is empty — cannot search")]
    Empty,
}

pub type Result<T> = std::result::Result<T, Error>;

/// HNSW construction and search tuning parameters.
#[derive(Clone, Debug)]
pub struct HnswConfig {
    /// Maximum number of neighbours retained per node per layer (M).
    pub m:            usize,
    /// Candidate list size during construction (ef_construction).
    pub ef_construct: usize,
    /// Candidate list size during search (ef_search).
    pub ef_search:    usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m:            16,
            ef_construct: 200,
            ef_search:    50,
        }
    }
}

/// A stored embedding entry.
#[derive(Clone, Debug)]
pub struct EmbeddingRecord {
    pub id:     DocumentId,
    pub vector: Vec<f32>,
}

// ── Internal node ────────────────────────────────────────────────────────────

struct Node {
    record:    EmbeddingRecord,
    /// Neighbours per layer. `neighbours[0]` is the base layer.
    neighbours: Vec<Vec<usize>>,
}

// ── Ordered-float wrapper for BinaryHeap ─────────────────────────────────────

/// `(distance, node_index)` ordered by distance ascending (min-heap via negation).
#[derive(Clone, Copy, PartialEq)]
struct Candidate {
    dist:  f32,
    index: usize,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse so BinaryHeap becomes a min-heap on distance.
        other.dist.partial_cmp(&self.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ── SemanticIndex ─────────────────────────────────────────────────────────────

/// HNSW approximate nearest-neighbour index over 1024-dim BGE embeddings.
///
/// Thread-safety: the index is not `Sync` — wrap in `Mutex` or `RwLock` if
/// shared across threads. `insert` takes `&mut self`; `search` takes `&self`.
pub struct SemanticIndex {
    nodes:       Vec<Node>,
    id_to_index: HashMap<DocumentId, usize>,
    entry_point: Option<usize>,
    max_layer:   usize,
    config:      HnswConfig,
}

impl SemanticIndex {
    /// Creates an empty index with the given configuration.
    pub fn new(config: HnswConfig) -> Self {
        Self {
            nodes:       Vec::new(),
            id_to_index: HashMap::new(),
            entry_point: None,
            max_layer:   0,
            config,
        }
    }

    /// Returns the number of documents in the index.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` when no documents have been inserted.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Inserts `embedding` for the given `id`.
    ///
    /// Returns `Error::Duplicate` if `id` is already present.
    pub fn insert(&mut self, id: DocumentId, embedding: &Embedding) -> Result<()> {
        if self.id_to_index.contains_key(&id) {
            return Err(Error::Duplicate { id });
        }

        let node_index = self.nodes.len();
        let level      = self.random_level();
        let max_layers = level + 1;

        let record = EmbeddingRecord {
            id,
            vector: embedding.values.clone(),
        };
        let neighbours = vec![Vec::new(); max_layers];

        self.nodes.push(Node { record, neighbours });
        self.id_to_index.insert(id, node_index);

        if self.entry_point.is_none() {
            self.entry_point = Some(node_index);
            self.max_layer   = level;
            return Ok(());
        }

        let entry = self.entry_point.unwrap();

        // Greedy descent from max_layer down to level+1 (single candidate).
        let mut ep = entry;
        for lc in (level + 1..=self.max_layer).rev() {
            ep = self.greedy_search_layer(&embedding.values, ep, lc);
        }

        // For each layer from min(level, max_layer) down to 0, find
        // ef_construct candidates and wire bidirectional edges.
        for lc in (0..=level.min(self.max_layer)).rev() {
            let candidates =
                self.search_layer(&embedding.values, ep, self.config.ef_construct, lc);

            let neighbours: Vec<usize> = candidates
                .iter()
                .take(self.config.m)
                .map(|c| c.index)
                .collect();

            self.nodes[node_index].neighbours[lc] = neighbours.clone();

            for &nb in &neighbours {
                self.nodes[nb].neighbours[lc].push(node_index);
                if self.nodes[nb].neighbours[lc].len() > self.config.m * 2 {
                    self.prune_neighbours(nb, lc);
                }
            }

            if let Some(&top) = candidates.first() {
                ep = top.index;
            }
        }

        if level > self.max_layer {
            self.max_layer   = level;
            self.entry_point = Some(node_index);
        }

        Ok(())
    }

    /// Returns the `k` nearest neighbours of `query_embedding` by cosine distance.
    ///
    /// Results are ordered by similarity descending (most similar first).
    /// Returns `Error::Empty` when the index contains no documents.
    pub fn search(
        &self,
        query_embedding: &Embedding,
        k: usize,
    ) -> Result<Vec<(DocumentId, f32)>> {
        let entry = self.entry_point.ok_or(Error::Empty)?;

        let query = &query_embedding.values;
        let mut ep = entry;

        for lc in (1..=self.max_layer).rev() {
            ep = self.greedy_search_layer(query, ep, lc);
        }

        let candidates = self.search_layer(query, ep, self.config.ef_search.max(k), 0);

        let results = candidates
            .into_iter()
            .take(k)
            .map(|c| {
                let id         = self.nodes[c.index].record.id;
                let similarity = 1.0 - c.dist;
                (id, similarity)
            })
            .collect();

        Ok(results)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Greedy single-path descent — returns the closest node index at `layer`.
    fn greedy_search_layer(&self, query: &[f32], entry: usize, layer: usize) -> usize {
        let mut current      = entry;
        let mut current_dist = cosine_distance(query, &self.nodes[current].record.vector);

        loop {
            let mut improved = false;
            let neighbours   = self.neighbours_at(current, layer);

            for &nb in neighbours {
                let d = cosine_distance(query, &self.nodes[nb].record.vector);
                if d < current_dist {
                    current      = nb;
                    current_dist = d;
                    improved     = true;
                }
            }

            if !improved {
                break;
            }
        }

        current
    }

    /// Beam search returning up to `ef` candidates at `layer`, ordered by
    /// distance ascending.
    fn search_layer(
        &self,
        query:  &[f32],
        entry:  usize,
        ef:     usize,
        layer:  usize,
    ) -> Vec<Candidate> {
        let entry_dist = cosine_distance(query, &self.nodes[entry].record.vector);

        let mut candidates:  BinaryHeap<Candidate> = BinaryHeap::new();
        let mut results:     BinaryHeap<std::cmp::Reverse<Candidate>> = BinaryHeap::new();
        let mut visited:     HashSet<usize> = HashSet::new();

        candidates.push(Candidate { dist: entry_dist, index: entry });
        results.push(std::cmp::Reverse(Candidate { dist: entry_dist, index: entry }));
        visited.insert(entry);

        while let Some(current) = candidates.pop() {
            let worst_result = results.peek().map(|r| r.0.dist).unwrap_or(f32::MAX);
            if current.dist > worst_result && results.len() >= ef {
                break;
            }

            for &nb in self.neighbours_at(current.index, layer) {
                if visited.contains(&nb) {
                    continue;
                }
                visited.insert(nb);

                let d = cosine_distance(query, &self.nodes[nb].record.vector);
                let worst = results.peek().map(|r| r.0.dist).unwrap_or(f32::MAX);

                if d < worst || results.len() < ef {
                    candidates.push(Candidate { dist: d, index: nb });
                    results.push(std::cmp::Reverse(Candidate { dist: d, index: nb }));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out: Vec<Candidate> = results.into_iter().map(|r| r.0).collect();
        out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    fn neighbours_at(&self, index: usize, layer: usize) -> &[usize] {
        let node = &self.nodes[index];
        if layer < node.neighbours.len() {
            &node.neighbours[layer]
        } else {
            &[]
        }
    }

    fn prune_neighbours(&mut self, index: usize, layer: usize) {
        let query: Vec<f32> = self.nodes[index].record.vector.clone();
        let max             = self.config.m * 2;

        let mut scored: Vec<(f32, usize)> = self.nodes[index].neighbours[layer]
            .iter()
            .map(|&nb| (cosine_distance(&query, &self.nodes[nb].record.vector), nb))
            .collect();

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(max);

        self.nodes[index].neighbours[layer] = scored.into_iter().map(|(_, nb)| nb).collect();
    }

    /// Samples a random insertion level using the standard HNSW formula.
    fn random_level(&self) -> usize {
        let m_l = 1.0 / (self.config.m as f64).ln();
        let r: f64 = rand_f64();
        (-r.ln() * m_l).floor() as usize
    }
}

// ── Distance ─────────────────────────────────────────────────────────────────

/// Cosine distance in [0, 2] (0 = identical, 2 = opposite).
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot:   f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 1.0;
    }
    1.0 - (dot / (mag_a * mag_b))
}

/// Portable xorshift64 pseudo-random float in (0, 1).
/// Seeded from a stack address for cheap per-call entropy without dependencies.
fn rand_f64() -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    let addr: usize = &hasher as *const _ as usize;
    addr.hash(&mut hasher);
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos()
        .hash(&mut hasher);
    let bits = hasher.finish();
    // Map to (0, 1) — never exactly 0 or 1.
    (bits as f64 / u64::MAX as f64).clamp(1e-10, 1.0 - 1e-10)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(values: Vec<f32>) -> Embedding {
        Embedding::new(values)
    }

    fn unit(dim: usize, hot: usize) -> Embedding {
        let mut v = vec![0.0f32; dim];
        v[hot] = 1.0;
        make_embedding(v)
    }

    #[test]
    fn insert_and_len() {
        let mut idx = SemanticIndex::new(HnswConfig::default());
        assert!(idx.is_empty());
        idx.insert(DocumentId::new(1), &unit(16, 0)).unwrap();
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn duplicate_insert_fails() {
        let mut idx = SemanticIndex::new(HnswConfig::default());
        idx.insert(DocumentId::new(1), &unit(16, 0)).unwrap();
        assert!(matches!(
            idx.insert(DocumentId::new(1), &unit(16, 0)),
            Err(Error::Duplicate { .. })
        ));
    }

    #[test]
    fn search_empty_fails() {
        let idx = SemanticIndex::new(HnswConfig::default());
        assert!(matches!(idx.search(&unit(16, 0), 5), Err(Error::Empty)));
    }

    #[test]
    fn nearest_neighbour_correct() {
        let mut idx = SemanticIndex::new(HnswConfig {
            m:            4,
            ef_construct: 20,
            ef_search:    10,
        });

        // Insert 8 orthogonal unit vectors.
        for i in 0..8u64 {
            idx.insert(DocumentId::new(i), &unit(8, i as usize)).unwrap();
        }

        // Query with unit vector 3 — nearest neighbour must be doc 3.
        let results = idx.search(&unit(8, 3), 1).unwrap();
        assert_eq!(results[0].0, DocumentId::new(3));
    }

    #[test]
    fn search_returns_k_results() {
        let mut idx = SemanticIndex::new(HnswConfig::default());
        for i in 0..10u64 {
            idx.insert(DocumentId::new(i), &unit(16, i as usize % 16)).unwrap();
        }
        let results = idx.search(&unit(16, 0), 5).unwrap();
        assert!(results.len() <= 5);
        assert!(!results.is_empty());
    }
}

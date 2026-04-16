// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/linkgraph/src/lib.rs
//
// Compressed Sparse Row (CSR) link graph and iterative PageRank.

use std::collections::HashMap;

use raithe_common::DocumentId;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("PageRank did not converge after {iterations} iterations")]
    DidNotConverge { iterations: usize },
}

pub type Result<T> = std::result::Result<T, Error>;

/// PageRank computation parameters.
#[derive(Clone, Debug)]
pub struct PageRankConfig {
    /// Damping factor (standard default 0.85).
    pub damping:   f64,
    /// Maximum iterations before returning the best estimate.
    pub max_iter:  usize,
    /// L1 convergence tolerance — stops when sum of absolute score changes
    /// falls below this threshold.
    pub tolerance: f64,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping:   0.85,
            max_iter:  100,
            tolerance: 1e-6,
        }
    }
}

/// Per-document PageRank scores.
pub type PageRankScores = HashMap<DocumentId, f32>;

/// CSR-backed directed link graph.
///
/// Edges are added incrementally via `add_edge`. `build` compresses the
/// adjacency list into CSR format for efficient PageRank iteration.
/// After `build`, further `add_edge` calls require another `build`.
pub struct LinkGraph {
    /// Raw edge list accumulated before compression.
    edges:      Vec<(DocumentId, DocumentId)>,
    /// All unique nodes seen via `add_edge`.
    nodes:      Vec<DocumentId>,
    node_index: HashMap<DocumentId, usize>,

    // CSR arrays populated by `build`.
    offsets: Vec<usize>,
    targets: Vec<usize>,
    built:   bool,
}

impl LinkGraph {
    /// Creates an empty graph.
    pub fn new() -> Self {
        Self {
            edges:      Vec::new(),
            nodes:      Vec::new(),
            node_index: HashMap::new(),
            offsets:    Vec::new(),
            targets:    Vec::new(),
            built:      false,
        }
    }

    /// Returns the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Records a directed edge from `src` to `dst`.
    ///
    /// Both nodes are registered if not already present. Marks the CSR
    /// representation as stale — call `build` before `compute_pagerank`.
    pub fn add_edge(&mut self, src: DocumentId, dst: DocumentId) {
        self.register(src);
        self.register(dst);
        self.edges.push((src, dst));
        self.built = false;
    }

    /// Compresses the edge list into CSR format.
    ///
    /// Must be called after all `add_edge` calls and before
    /// `compute_pagerank`. Idempotent if no edges have been added since the
    /// last `build`.
    pub fn build(&mut self) {
        if self.built {
            return;
        }

        let n = self.nodes.len();
        let mut out_edges: Vec<Vec<usize>> = vec![Vec::new(); n];

        for &(src, dst) in &self.edges {
            let s = self.node_index[&src];
            let d = self.node_index[&dst];
            out_edges[s].push(d);
        }

        let mut offsets = Vec::with_capacity(n + 1);
        let mut targets = Vec::with_capacity(self.edges.len());
        offsets.push(0);

        for neighbours in &out_edges {
            targets.extend_from_slice(neighbours);
            offsets.push(targets.len());
        }

        self.offsets = offsets;
        self.targets = targets;
        self.built   = true;
    }

    /// Runs iterative PageRank and returns a score per document.
    ///
    /// Calls `build` internally if the CSR representation is stale. Scores
    /// are normalised so they sum to ~1.0. Runs up to `config.max_iter`
    /// iterations, stopping early when L1 delta falls below `config.tolerance`.
    pub fn compute_pagerank(&mut self, config: &PageRankConfig) -> PageRankScores {
        self.build();

        let n = self.nodes.len();
        if n == 0 {
            return PageRankScores::new();
        }

        let init       = 1.0 / n as f64;
        let mut scores = vec![init; n];
        let dangling_d = (1.0 - config.damping) / n as f64;

        for _ in 0..config.max_iter {
            let mut next = vec![dangling_d; n];

            for src in 0..n {
                let start      = self.offsets[src];
                let end        = self.offsets[src + 1];
                let out_degree = end - start;

                if out_degree == 0 {
                    // Dangling node — distribute score evenly to all nodes.
                    let share = config.damping * scores[src] / n as f64;
                    for v in next.iter_mut() {
                        *v += share;
                    }
                } else {
                    let share = config.damping * scores[src] / out_degree as f64;
                    for &dst in &self.targets[start..end] {
                        next[dst] += share;
                    }
                }
            }

            let delta: f64 = scores
                .iter()
                .zip(next.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            scores = next;

            if delta < config.tolerance {
                break;
            }
        }

        self.nodes
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, scores[i] as f32))
            .collect()
    }

    fn register(&mut self, id: DocumentId) {
        if !self.node_index.contains_key(&id) {
            let index = self.nodes.len();
            self.nodes.push(id);
            self.node_index.insert(id, index);
        }
    }
}

impl Default for LinkGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u64) -> DocumentId {
        DocumentId::new(n)
    }

    #[test]
    fn add_edge_registers_nodes() {
        let mut g = LinkGraph::new();
        g.add_edge(id(1), id(2));
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn single_node_pagerank() {
        let mut g = LinkGraph::new();
        g.add_edge(id(1), id(1));
        let scores = g.compute_pagerank(&PageRankConfig::default());
        assert!(scores.contains_key(&id(1)));
        let s = scores[&id(1)];
        assert!((s - 1.0).abs() < 0.01);
    }

    #[test]
    fn higher_indegree_higher_score() {
        let mut g = LinkGraph::new();
        // Closed graph — no dangling nodes, no sinks.
        // Nodes 1, 2, 3 each point to node 4.
        // Node 4 points back to nodes 1, 2, 3 (round-robin return).
        // Node 4 receives 3 concentrated in-links from distinct nodes;
        // each of 1/2/3 receives only 1/3 of node 4's out-score.
        // By symmetry of 1/2/3 and concentration into 4, node 4 scores
        // strictly higher than any of nodes 1/2/3.
        g.add_edge(id(1), id(4));
        g.add_edge(id(2), id(4));
        g.add_edge(id(3), id(4));
        g.add_edge(id(4), id(1));
        g.add_edge(id(4), id(2));
        g.add_edge(id(4), id(3));

        let scores = g.compute_pagerank(&PageRankConfig::default());
        assert!(
            scores[&id(4)] > scores[&id(1)],
            "node 4 score={} node 1 score={}",
            scores[&id(4)],
            scores[&id(1)],
        );
    }

    #[test]
    fn scores_sum_to_approximately_one() {
        let mut g = LinkGraph::new();
        for i in 0..5u64 {
            g.add_edge(id(i), id((i + 1) % 5));
        }
        let scores = g.compute_pagerank(&PageRankConfig::default());
        let total: f32 = scores.values().sum();
        assert!((total - 1.0).abs() < 0.01);
    }
}

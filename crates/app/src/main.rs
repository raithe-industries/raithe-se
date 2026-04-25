// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/app/src/main.rs
//
// Single binary entry point. Phase 1: BM25-only — no neural / semantic /
// link-graph / freshness paths. Phase 2/3 brings them back behind
// engine.phase1_only=false.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};

use raithe_app::Phase1Engine;
use raithe_common::Url;
use raithe_config::{watch as watch_config, Config};

struct Cli {
    config_path: PathBuf,
    json_logs:   bool,
    seeds_path:  Option<PathBuf>,
}

impl Cli {
    fn parse() -> Self {
        let mut args        = std::env::args().skip(1).peekable();
        let mut config_path = PathBuf::from("data/config/engine.toml");
        let mut json_logs   = false;
        let mut seeds_path  = None;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--json-logs" => json_logs = true,
                "--seeds"     => { seeds_path = args.next().map(PathBuf::from); }
                "--config"    => {
                    if let Some(p) = args.next() { config_path = PathBuf::from(p); }
                }
                _ => {}
            }
        }
        Self { config_path, json_logs, seeds_path }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli    = Cli::parse();
    let config = Config::load(&cli.config_path)
        .with_context(|| format!("loading config from {}", cli.config_path.display()))?;

    raithe_metrics::init_tracing(cli.json_logs).context("tracing init")?;

    let seeds = load_seeds(cli.seeds_path.as_deref())?;
    if seeds.len() < config.crawler.min_seeds {
        anyhow::bail!(
            "insufficient seeds: need at least {}, got {} — \
             provide a seed file via --seeds <path>",
            config.crawler.min_seeds, seeds.len(),
        );
    }

    if !config.engine.phase1_only {
        // Phase 2/3 path — not implemented in this branch yet. Fail loudly so
        // it's obvious nothing heavy is wired in.
        anyhow::bail!(
            "engine.phase1_only=false but the heavy path (neural / semantic / \
             link-graph / freshness) is currently disabled — re-enable in code \
             before flipping this flag"
        );
    }

    let data_dir = PathBuf::from("data");
    let engine   = Phase1Engine::build(config.clone(), seeds, &data_dir).context("engine build")?;

    {
        let path = cli.config_path.clone();
        match watch_config(path) {
            Ok(_rx)  => {}
            Err(err) => tracing::warn!("config watcher failed to start: {err}"),
        }
    }

    engine.spawn_pipeline();

    let server   = engine.build_server()?;
    let bind_str = config.serving.bind.clone();
    let _state   = Arc::clone(&engine.state); // keep alive for server's lifetime

    tracing::info!(bind = %bind_str, "raithe-se ready (phase 1)");
    server.run().await.context("HTTP server error")
}

fn load_seeds(path: Option<&std::path::Path>) -> Result<Vec<Url>> {
    let path = match path {
        Some(p) => p.to_path_buf(),
        None    => PathBuf::from("data/seeds.txt"),
    };

    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::warn!(path = %path.display(), "seed file not found — empty frontier");
            return Ok(Vec::new());
        }
        Err(e) => return Err(e).with_context(|| format!("reading seeds from {}", path.display())),
    };

    let seeds: Vec<Url> = text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| match Url::parse(l) {
            Ok(u)  => Some(u),
            Err(_) => { tracing::warn!(line = l, "invalid seed URL — skipping"); None }
        })
        .collect();

    tracing::info!(count = seeds.len(), "seeds loaded");
    Ok(seeds)
}
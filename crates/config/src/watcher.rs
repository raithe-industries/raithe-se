// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/config/src/watcher.rs
//
// Hot-reload watcher. Spawns a tokio task that watches engine.toml via notify.
// On valid change, loads and validates the new config, then sends it on the
// watch channel. Invalid configs are logged and discarded — the running config
// is never replaced with an invalid one.
//
// NOTE: config is not permitted to depend on tracing (§5.2). Diagnostic
// output from the watcher uses eprintln! so the binary crate can observe
// watcher health without a tracing dependency in this crate.

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::watch;

use crate::{Config, Error, Result};

/// Spawns a background task that watches `path` for changes.
///
/// Returns a `watch::Receiver<Config>` seeded with the initial load. When the
/// file changes and the new content is valid, the receiver is updated.
/// Invalid configs are discarded — the receiver retains the last valid config.
pub fn watch(path: PathBuf) -> Result<watch::Receiver<Config>> {
    let initial = Config::load(&path)?;
    initial.validate()?;

    let (tx, rx) = watch::channel(initial);

    tokio::spawn(async move {
        if let Err(err) = run_watcher(path, tx).await {
            eprintln!("raithe-config: watcher stopped: {err}");
        }
    });

    Ok(rx)
}

async fn run_watcher(path: PathBuf, tx: watch::Sender<Config>) -> Result<()> {
    // std::sync::mpsc so the notify callback (non-async thread) can send
    // without needing a runtime handle.
    let (event_tx, event_rx) = mpsc::channel::<()>();

    let mut watcher = RecommendedWatcher::new(
        move |_event: notify::Result<notify::Event>| {
            // Best-effort; ignore send errors (receiver may have dropped).
            let _ = event_tx.send(());
        },
        notify::Config::default().with_poll_interval(Duration::from_secs(2)),
    )
    .map_err(|source| Error::Watch { source })?;

    watcher
        .watch(&path, RecursiveMode::NonRecursive)
        .map_err(|source| Error::Watch { source })?;

    let mut event_rx = event_rx;

    loop {
        let (still_open, returned_rx) = tokio::task::spawn_blocking(move || {
            let still_open = event_rx.recv().is_ok();
            (still_open, event_rx)
        })
        .await
        .unwrap_or((false, mpsc::channel::<()>().1));

        event_rx = returned_rx;

        if !still_open {
            break;
        }

        // Coalesce rapid successive saves (e.g. editor write-then-fsync).
        tokio::time::sleep(Duration::from_millis(50)).await;

        match Config::load(&path).and_then(|cfg| cfg.validate().map(|()| cfg)) {
            Ok(cfg) => {
                let _ = tx.send(cfg);
                eprintln!("raithe-config: reloaded from {}", path.display());
            }
            Err(err) => {
                eprintln!("raithe-config: reload rejected, running config unchanged: {err}");
            }
        }
    }

    Ok(())
}

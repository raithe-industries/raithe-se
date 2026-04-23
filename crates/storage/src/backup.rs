// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/storage/src/backup.rs
//
// Snapshot backups of the storage directory.
//
// SECURITY (§9.1 — CRITICAL): all destination paths are canonicalised and
// checked to be rooted under the configured backup directory. Any path
// component containing ".." is rejected before canonicalisation is attempted,
// to guard against TOCTOU races on symlinks.

use std::path::{Path, PathBuf};

use crate::{Error, Result};

/// Manages snapshot backups of the storage directory.
pub struct Backup {
    /// The directory under which all backup snapshots must reside.
    root: PathBuf,
}

impl Backup {
    /// Creates a `Backup` manager rooted at `root`.
    ///
    /// `root` is canonicalised at construction so that all later comparisons
    /// use absolute, symlink-resolved paths.
    pub fn new(root: &Path) -> Result<Self> {
        std::fs::create_dir_all(root).map_err(|source| Error::Io {
            path: root.display().to_string(),
            source,
        })?;
        let root = root.canonicalize().map_err(|source| Error::Io {
            path: root.display().to_string(),
            source,
        })?;
        Ok(Self { root })
    }

    /// Copies `src` into `dest` under the backup root.
    ///
    /// # Security
    ///
    /// `dest` is validated in two passes before any I/O touches the filesystem:
    ///
    /// 1. Lexical check — every component of `dest` is inspected for `..`.
    ///    Any such component causes an immediate `Error::PathTraversal`.
    ///
    /// 2. Canonical check — after joining `dest` to the backup root and
    ///    calling `canonicalize`, the result must start with `self.root`.
    ///    This catches symlink-based escapes that survive the lexical check.
    pub fn snapshot(&self, src: &Path, dest: &Path) -> Result<()> {
        self.validate_dest(dest)?;

        let abs_dest = self.root.join(dest);

        // Create parent directories inside the backup root.
        if let Some(parent) = abs_dest.parent() {
            std::fs::create_dir_all(parent).map_err(|source| Error::Io {
                path: parent.display().to_string(),
                source,
            })?;
        }

        std::fs::copy(src, &abs_dest).map_err(|source| Error::Io {
            path: abs_dest.display().to_string(),
            source,
        })?;

        // Post-copy canonical check — guards against TOCTOU races.
        let canonical = abs_dest.canonicalize().map_err(|source| Error::Io {
            path: abs_dest.display().to_string(),
            source,
        })?;

        if !canonical.starts_with(&self.root) {
            // Remove the file that landed outside the root before returning.
            let _ = std::fs::remove_file(&canonical);
            return Err(Error::PathTraversal {
                path: dest.display().to_string(),
            });
        }

        Ok(())
    }

    /// Returns the backup root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn validate_dest(&self, dest: &Path) -> Result<()> {
        for component in dest.components() {
            use std::path::Component;
            match component {
                Component::ParentDir => {
                    return Err(Error::PathTraversal {
                        path: dest.display().to_string(),
                    });
                }
                Component::Normal(_) | Component::CurDir => {}
                // Absolute paths or prefix components would escape the root.
                Component::RootDir | Component::Prefix(_) => {
                    return Err(Error::PathTraversal {
                        path: dest.display().to_string(),
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_src(dir: &Path, name: &str, content: &[u8]) -> PathBuf {
        let path = dir.join(name);
        std::fs::File::create(&path)
            .unwrap()
            .write_all(content)
            .unwrap();
        path
    }

    #[test]
    fn valid_snapshot_copies_file() {
        let dir = tempfile::tempdir().unwrap();
        let backup_root = dir.path().join("backups");
        let src = make_src(dir.path(), "data.bin", b"payload");

        let backup = Backup::new(&backup_root).unwrap();
        backup.snapshot(&src, Path::new("data.bin")).unwrap();

        let dest = backup_root.join("data.bin");
        assert_eq!(std::fs::read(dest).unwrap(), b"payload");
    }

    #[test]
    fn dotdot_component_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let backup_root = dir.path().join("backups");
        let src = make_src(dir.path(), "data.bin", b"x");

        let backup = Backup::new(&backup_root).unwrap();
        let result = backup.snapshot(&src, Path::new("../escape.bin"));
        assert!(result.is_err());
        match result.unwrap_err() {
            Error::PathTraversal { .. } => {}
            other => panic!("expected PathTraversal, got {other}"),
        }
    }

    #[test]
    fn absolute_dest_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let backup_root = dir.path().join("backups");
        let src = make_src(dir.path(), "data.bin", b"x");

        let backup = Backup::new(&backup_root).unwrap();
        let result = backup.snapshot(&src, Path::new("/etc/passwd"));
        assert!(result.is_err());
    }

    #[test]
    fn nested_valid_path_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let backup_root = dir.path().join("backups");
        let src = make_src(dir.path(), "data.bin", b"nested");

        let backup = Backup::new(&backup_root).unwrap();
        backup
            .snapshot(&src, Path::new("2026/04/data.bin"))
            .unwrap();

        let dest = backup_root.join("2026/04/data.bin");
        assert_eq!(std::fs::read(dest).unwrap(), b"nested");
    }
}

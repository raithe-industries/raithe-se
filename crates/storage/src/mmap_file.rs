// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/storage/src/mmap_file.rs
//
// Read-only memory-mapped file access via memmap2.

use std::ops::Deref;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::{Error, Result};

/// A read-only memory-mapped view of a file on disk.
///
/// The mapping is established at construction and remains valid for the
/// lifetime of the `MmapFile`. The underlying file must not be truncated
/// or deleted while the mapping is live.
pub struct MmapFile {
    path: PathBuf,
    mmap: Mmap,
}

impl MmapFile {
    /// Opens `path` and maps it into memory read-only.
    pub fn open(path: &Path) -> Result<Self> {
        let path = path.to_path_buf();
        let file = std::fs::File::open(&path).map_err(|source| Error::Io {
            path: path.display().to_string(),
            source,
        })?;
        // SAFETY: The caller is responsible for ensuring the file is not
        // concurrently truncated or deleted while this mapping is live.
        // MmapFile is used for read-only access to immutable index segments.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|source| Error::Io {
            path: path.display().to_string(),
            source,
        })?;
        Ok(Self { path, mmap })
    }

    /// Returns the length of the mapped region in bytes.
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Returns `true` when the mapped file is empty.
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Returns the path this file was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Deref for MmapFile {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.mmap
    }
}

impl AsRef<[u8]> for MmapFile {
    fn as_ref(&self) -> &[u8] {
        &self.mmap
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn maps_file_contents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seg.bin");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(b"raithe")
            .unwrap();

        let mmap = MmapFile::open(&path).unwrap();
        assert_eq!(&*mmap, b"raithe");
        assert_eq!(mmap.len(), 6);
    }

    #[test]
    fn empty_file_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.bin");
        std::fs::File::create(&path).unwrap();

        let mmap = MmapFile::open(&path).unwrap();
        assert!(mmap.is_empty());
    }
}

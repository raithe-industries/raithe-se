// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/common/src/timestamp.rs
//
// Newtype wrapper over i64 Unix milliseconds for point-in-time values.

use serde::{Deserialize, Serialize};

/// A point in time expressed as milliseconds since the Unix epoch.
///
/// Negative values represent instants before 1970-01-01T00:00:00Z.
#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Deserialize,
    Serialize,
)]
pub struct Timestamp(i64);

impl Timestamp {
    /// The Unix epoch: 1970-01-01T00:00:00.000Z.
    pub const EPOCH: Self = Self(0);

    /// Wraps the given `millis` value.
    pub fn from_millis(millis: i64) -> Self {
        Self(millis)
    }

    /// Returns the raw millisecond value.
    pub fn as_millis(self) -> i64 {
        self.0
    }

    /// Returns the number of whole seconds since the Unix epoch.
    pub fn as_secs(self) -> i64 {
        self.0 / 1_000
    }

    /// Returns the elapsed milliseconds between `self` and `later`.
    ///
    /// Returns `None` if `later` precedes `self` (i.e. the result would be
    /// negative), preserving the invariant that elapsed time is non-negative.
    pub fn elapsed_millis_until(self, later: Self) -> Option<i64> {
        later.0.checked_sub(self.0).filter(|&d| d >= 0)
    }
}

impl std::fmt::Display for Timestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<i64> for Timestamp {
    fn from(millis: i64) -> Self {
        Self(millis)
    }
}

impl From<Timestamp> for i64 {
    fn from(ts: Timestamp) -> Self {
        ts.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_is_zero() {
        assert_eq!(Timestamp::EPOCH.as_millis(), 0);
    }

    #[test]
    fn as_secs_truncates() {
        let ts = Timestamp::from_millis(1_500);
        assert_eq!(ts.as_secs(), 1);
    }

    #[test]
    fn elapsed_forward() {
        let a = Timestamp::from_millis(1_000);
        let b = Timestamp::from_millis(3_000);
        assert_eq!(a.elapsed_millis_until(b), Some(2_000));
    }

    #[test]
    fn elapsed_backward_is_none() {
        let a = Timestamp::from_millis(3_000);
        let b = Timestamp::from_millis(1_000);
        assert_eq!(a.elapsed_millis_until(b), None);
    }

    #[test]
    fn ordering() {
        let earlier = Timestamp::from_millis(100);
        let later = Timestamp::from_millis(200);
        assert!(earlier < later);
    }
}

// © RAiTHE INDUSTRIES INCORPORATED 2026
// crates/instant/src/lib.rs
//
// Instant answer resolution — arithmetic expressions, unit conversions,
// currency (offline rates), and time zone queries.

use raithe_common::ParsedQuery;

// ── Public types ─────────────────────────────────────────────────────────────

/// The category of instant answer produced.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AnswerKind {
    Calculation,
    UnitConversion,
    Currency,
    TimeZone,
}

/// A resolved instant answer ready for display in the search results panel.
#[derive(Clone, Debug)]
pub struct InstantAnswer {
    pub kind:    AnswerKind,
    /// The human-readable answer string shown to the user.
    pub display: String,
    /// The input expression or query that produced this answer.
    pub input:   String,
}

// ── Engine ────────────────────────────────────────────────────────────────────

/// Resolves instant answers from a `ParsedQuery`.
///
/// Tries each resolver in priority order and returns the first match.
/// Returns `None` when no resolver fires — the query proceeds to normal search.
pub struct InstantEngine;

impl InstantEngine {
    /// Creates a new `InstantEngine`.
    pub fn new() -> Self {
        Self
    }

    /// Attempts to resolve `query` as an instant answer.
    ///
    /// Returns `Some(InstantAnswer)` when the original query matches a known
    /// pattern, or `None` to indicate normal search should proceed.
    pub fn resolve(&self, query: &ParsedQuery) -> Option<InstantAnswer> {
        let raw = query.original.trim();

        try_calculation(raw)
            .or_else(|| try_unit_conversion(raw))
            .or_else(|| try_currency(raw))
            .or_else(|| try_time_zone(raw))
    }
}

impl Default for InstantEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Resolvers ─────────────────────────────────────────────────────────────────

/// Evaluates arithmetic and mathematical expressions via `evalexpr`.
///
/// Accepts expressions like `2 + 2`, `sqrt(16)`, `sin(pi/2)`.
fn try_calculation(raw: &str) -> Option<InstantAnswer> {
    // Only attempt evaluation if the input looks like a math expression.
    // Reject pure words to avoid false positives on search queries.
    if !has_math_chars(raw) {
        return None;
    }

    let result = evalexpr::eval(raw).ok()?;
    let display = format!("{raw} = {result}");
    let input   = raw.to_owned();

    Some(InstantAnswer {
        kind: AnswerKind::Calculation,
        display,
        input,
    })
}

/// Resolves simple unit conversion queries.
///
/// Handles patterns like `5 km to miles`, `100 kg in lbs`, `32 f to c`.
fn try_unit_conversion(raw: &str) -> Option<InstantAnswer> {
    let lower = raw.to_lowercase();

    // Pattern: <number> <from_unit> to|in <to_unit>
    let (connector, idx) = [" to ", " in "]
        .iter()
        .find_map(|&sep| lower.find(sep).map(|i| (sep, i)))?;

    let lhs = lower[..idx].trim();
    let rhs = lower[idx + connector.len()..].trim();

    let (value, from_unit) = split_number_unit(lhs)?;
    let to_unit            = rhs;

    let converted = convert_units(value, from_unit, to_unit)?;
    let display   = format!("{value} {from_unit} = {converted:.4} {to_unit}");
    let input     = raw.to_owned();

    Some(InstantAnswer {
        kind: AnswerKind::UnitConversion,
        display,
        input,
    })
}

/// Resolves offline currency conversion queries.
///
/// Uses hard-coded approximate rates relative to USD. Handles patterns like
/// `100 USD to EUR`, `50 cad in usd`.
/// TODO(impl) — replace with periodically updated offline rate table.
fn try_currency(raw: &str) -> Option<InstantAnswer> {
    let lower = raw.to_lowercase();

    let (connector, idx) = [" to ", " in "]
        .iter()
        .find_map(|&sep| lower.find(sep).map(|i| (sep, i)))?;

    let lhs = lower[..idx].trim();
    let rhs = lower[idx + connector.len()..].trim().to_uppercase();

    let (value, from_code) = split_number_unit(lhs)?;
    let from_code          = from_code.to_uppercase();

    let from_rate = usd_rate(&from_code)?;
    let to_rate   = usd_rate(&rhs)?;
    let converted = value / from_rate * to_rate;
    let display   = format!("{value} {from_code} ≈ {converted:.2} {rhs}");
    let input     = raw.to_owned();

    Some(InstantAnswer {
        kind: AnswerKind::Currency,
        display,
        input,
    })
}

/// Resolves time zone queries.
///
/// Handles patterns like `time in Tokyo`, `current time London`.
/// TODO(impl) — wire to a proper tz database lookup.
fn try_time_zone(raw: &str) -> Option<InstantAnswer> {
    let lower = raw.to_lowercase();
    if !lower.starts_with("time in ") && !lower.starts_with("current time ") {
        return None;
    }

    // Placeholder — real implementation queries system tz database.
    // TODO(impl) — resolve city/region to UTC offset and format local time.
    let location = lower
        .trim_start_matches("time in ")
        .trim_start_matches("current time ")
        .trim();

    if location.is_empty() {
        return None;
    }

    let display = format!("Time in {location}: TODO(impl)");
    let input   = raw.to_owned();

    Some(InstantAnswer {
        kind: AnswerKind::TimeZone,
        display,
        input,
    })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Returns `true` when `s` contains characters that appear in math expressions.
fn has_math_chars(s: &str) -> bool {
    s.chars().any(|c| "+-*/^()".contains(c))
        || s.split_whitespace().any(|t| t.parse::<f64>().is_ok())
}

/// Splits a string like `"5.2 km"` into `(5.2, "km")`.
fn split_number_unit(s: &str) -> Option<(f64, &str)> {
    let s       = s.trim();
    let split   = s.find(|c: char| c.is_alphabetic())?;
    let number  = s[..split].trim().parse::<f64>().ok()?;
    let unit    = s[split..].trim();
    Some((number, unit))
}

/// Converts `value` from `from` unit to `to` unit.
/// Covers length, mass, temperature, and speed.
fn convert_units(value: f64, from: &str, to: &str) -> Option<f64> {
    // Normalise to SI base unit then convert out.
    let si = to_si(value, from)?;
    from_si(si, to)
}

fn to_si(value: f64, unit: &str) -> Option<f64> {
    match unit {
        // Length → metres
        "m" | "metre" | "metres"     => Some(value),
        "km" | "kilometres"          => Some(value * 1_000.0),
        "cm"                         => Some(value / 100.0),
        "mm"                         => Some(value / 1_000.0),
        "mi" | "miles" | "mile"      => Some(value * 1_609.344),
        "ft" | "feet" | "foot"       => Some(value * 0.3048),
        "in" | "inches" | "inch"     => Some(value * 0.0254),
        "yd" | "yards" | "yard"      => Some(value * 0.9144),
        // Mass → kilograms
        "kg" | "kilograms"           => Some(value),
        "g"  | "grams"               => Some(value / 1_000.0),
        "lb" | "lbs" | "pounds"      => Some(value * 0.453_592),
        "oz" | "ounces"              => Some(value * 0.028_349_5),
        "t"  | "tonnes"              => Some(value * 1_000.0),
        // Temperature → Celsius (as SI proxy for conversions)
        "c"  | "celsius"             => Some(value),
        "f"  | "fahrenheit"          => Some((value - 32.0) * 5.0 / 9.0),
        "k"  | "kelvin"              => Some(value - 273.15),
        // Speed → m/s
        "ms" | "m/s"                 => Some(value),
        "kmh" | "km/h"               => Some(value / 3.6),
        "mph"                        => Some(value * 0.44704),
        _                            => None,
    }
}

fn from_si(si: f64, unit: &str) -> Option<f64> {
    match unit {
        "m" | "metre" | "metres"     => Some(si),
        "km" | "kilometres"          => Some(si / 1_000.0),
        "cm"                         => Some(si * 100.0),
        "mm"                         => Some(si * 1_000.0),
        "mi" | "miles" | "mile"      => Some(si / 1_609.344),
        "ft" | "feet" | "foot"       => Some(si / 0.3048),
        "in" | "inches" | "inch"     => Some(si / 0.0254),
        "yd" | "yards" | "yard"      => Some(si / 0.9144),
        "kg" | "kilograms"           => Some(si),
        "g"  | "grams"               => Some(si * 1_000.0),
        "lb" | "lbs" | "pounds"      => Some(si / 0.453_592),
        "oz" | "ounces"              => Some(si / 0.028_349_5),
        "t"  | "tonnes"              => Some(si / 1_000.0),
        "c"  | "celsius"             => Some(si),
        "f"  | "fahrenheit"          => Some(si * 9.0 / 5.0 + 32.0),
        "k"  | "kelvin"              => Some(si + 273.15),
        "ms" | "m/s"                 => Some(si),
        "kmh" | "km/h"               => Some(si * 3.6),
        "mph"                        => Some(si / 0.44704),
        _                            => None,
    }
}

/// Returns approximate USD exchange rate for common currency codes.
/// TODO(impl) — replace with offline rate table updated periodically.
fn usd_rate(code: &str) -> Option<f64> {
    match code {
        "USD" => Some(1.0),
        "EUR" => Some(0.92),
        "GBP" => Some(0.79),
        "CAD" => Some(1.36),
        "AUD" => Some(1.53),
        "JPY" => Some(149.5),
        "CHF" => Some(0.90),
        "CNY" => Some(7.24),
        "INR" => Some(83.1),
        "MXN" => Some(17.1),
        _     => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(raw: &str) -> ParsedQuery {
        ParsedQuery::raw(raw)
    }

    #[test]
    fn arithmetic_resolved() {
        let engine = InstantEngine::new();
        let answer = engine.resolve(&query("2 + 2")).unwrap();
        assert_eq!(answer.kind, AnswerKind::Calculation);
        assert!(answer.display.contains('4'));
    }

    #[test]
    fn plain_text_not_resolved_as_calc() {
        let engine = InstantEngine::new();
        assert!(engine.resolve(&query("how does rust work")).is_none());
    }

    #[test]
    fn km_to_miles() {
        let engine = InstantEngine::new();
        let answer = engine.resolve(&query("5 km to miles")).unwrap();
        assert_eq!(answer.kind, AnswerKind::UnitConversion);
        assert!(answer.display.contains("3.10"));
    }

    #[test]
    fn celsius_to_fahrenheit() {
        let engine = InstantEngine::new();
        let answer = engine.resolve(&query("100 c to f")).unwrap();
        assert_eq!(answer.kind, AnswerKind::UnitConversion);
        assert!(answer.display.contains("212"));
    }

    #[test]
    fn currency_conversion() {
        let engine = InstantEngine::new();
        let answer = engine.resolve(&query("100 usd to eur")).unwrap();
        assert_eq!(answer.kind, AnswerKind::Currency);
        assert!(answer.display.contains("92"));
    }

    #[test]
    fn time_zone_placeholder() {
        let engine = InstantEngine::new();
        let answer = engine.resolve(&query("time in Tokyo")).unwrap();
        assert_eq!(answer.kind, AnswerKind::TimeZone);
    }
}

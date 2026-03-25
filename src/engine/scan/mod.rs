// Core scanning engine.
//
// Builds Aho-Corasick automata from spelling and case rules, then scans input
// text for violations:
//
//   1. Build excluded ranges (URLs, paths, @mentions, code fences).
//   2. Detect Chinese type (Traditional vs Simplified).
//   3. Aho-Corasick scan for spelling rules — skip excluded positions,
//      skip variant rules when text is Simplified.
//   4. Aho-Corasick scan for case rules — check word boundaries and
//      compare matched text against valid forms (term + alternatives).
//   5. Punctuation, spacing, ellipsis, quote checks.
//   6. Overlap resolution (longest match wins).
//   7. Grammar checks (interlingual transfer, A-not-A + 嗎 clash) —
//      run after overlap resolution to avoid suppressing narrower issues.

mod case_rule;
mod ellipsis;
mod grammar;
mod overlap;
mod punctuation;
mod quotes;
mod spacing;
mod spelling;

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use daachorse::{CharwiseDoubleArrayAhoCorasickBuilder, MatchKind as DaacMatchKind};

use super::excluded::{build_excluded_ranges, merge_ranges_pub, ByteRange};
use super::lineindex::{ColumnEncoding, LineIndex};
use super::markdown::{
    build_markdown_excluded_ranges, build_markdown_excluded_ranges_no_code,
    build_yaml_excluded_ranges,
};
use super::normalize::{map_offset, normalize_nfc, Normalized};
use super::segment::Segmenter;
use super::suppression::build_suppression_ranges;
use serde::{Deserialize, Serialize};

use super::zhtype::{detect_chinese_type, ChineseType};
use crate::rules::ruleset::{
    CaseRule, Issue, IssueType, Profile, ProfileConfig, Severity, SpellingRule,
};

use self::ellipsis::scan_ellipsis;
use self::overlap::resolve_overlaps;
use self::quotes::{fix_quote_pairing, validate_quote_hierarchy};

// Public types

/// Output of a scan operation: detected issues plus the Chinese script type
/// detected during scanning.  Returning detected_script here eliminates the
/// need for callers to run a second O(n) detect_chinese_type pass over the
/// same text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanOutput {
    pub issues: Vec<Issue>,
    pub detected_script: ChineseType,
    /// AI writing signature report.  Present only when AI scoring is
    /// requested (editorial profile or explicit detect_ai/ai_score).
    #[serde(default)]
    pub ai_signature: Option<crate::engine::ai_score::AiSignatureReport>,
}

/// Content type for determining exclusion strategy.
///
/// Shared between CLI and MCP pipelines (20.4 deduplication).  Lives in the
/// engine so both consumers can use the same scan_for_content_type method
/// without duplicating the dispatch logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    Plain,
    Markdown,
    /// Like Markdown, but code blocks (fenced/indented) are NOT excluded from
    /// scanning.  Inline code and HTML blocks are still excluded.  Useful when
    /// code blocks contain prose (e.g. translated output, comments) that should
    /// be linted.
    MarkdownScanCode,
    Yaml,
}

// Constants

/// Number of characters around a match to examine for context clues.
/// Shared with fixer.rs which re-exports it.
pub(crate) const CONTEXT_WINDOW_CHARS: usize = 40;

/// Minimum context clue matches for the scanner to emit a context-dependent
/// issue.  One nearby clue word is enough to confirm the right domain.
/// The fixer uses a stricter threshold (2) before auto-applying corrections.
const MIN_SCAN_CLUE_MATCHES: usize = 1;

/// Number of characters for positional clue windows (before:/after:).
/// Narrower than the general context window (40) because positional clues
/// express proximity, not just co-occurrence.
const POSITIONAL_WINDOW_CHARS: usize = 20;

/// A parsed positional condition for disambiguation.
///
/// Positional clues constrain WHERE a context term must appear relative to
/// the AC match, unlike flat context_clues which check presence anywhere
/// in the +-40-char window.
#[derive(Debug, Clone)]
pub(crate) enum PositionalClue {
    /// TERM must appear within POSITIONAL_WINDOW_CHARS chars AFTER the match.
    Before(String),
    /// TERM must appear within POSITIONAL_WINDOW_CHARS chars BEFORE the match.
    After(String),
    /// TERM must be immediately adjacent to the match (no gap, either side).
    Adjacent(String),
    /// TERM must NOT appear within POSITIONAL_WINDOW_CHARS chars AFTER the match.
    NotBefore(String),
    /// TERM must NOT appear within POSITIONAL_WINDOW_CHARS chars BEFORE the match.
    NotAfter(String),
}

impl PositionalClue {
    /// Parse a positional clue string (e.g. "before:函式", "not_after:的").
    /// Returns None if the syntax is unrecognized.
    fn parse(s: &str) -> Option<Self> {
        if let Some(term) = s.strip_prefix("before:") {
            if !term.is_empty() {
                return Some(PositionalClue::Before(term.to_string()));
            }
        }
        if let Some(term) = s.strip_prefix("after:") {
            if !term.is_empty() {
                return Some(PositionalClue::After(term.to_string()));
            }
        }
        if let Some(term) = s.strip_prefix("adjacent:") {
            if !term.is_empty() {
                return Some(PositionalClue::Adjacent(term.to_string()));
            }
        }
        if let Some(term) = s.strip_prefix("not_before:") {
            if !term.is_empty() {
                return Some(PositionalClue::NotBefore(term.to_string()));
            }
        }
        if let Some(term) = s.strip_prefix("not_after:") {
            if !term.is_empty() {
                return Some(PositionalClue::NotAfter(term.to_string()));
            }
        }
        None
    }
}

// Shared helper functions

/// Returns true if the text between `prev_end` and `offset` contains a
/// paragraph break (\n\n or \r\n\r\n).
fn has_paragraph_break(text: &str, prev_end: usize, offset: usize) -> bool {
    text.get(prev_end..offset)
        .is_some_and(|s| s.contains("\n\n") || s.contains("\r\n\r\n"))
}

/// Split text into paragraph blocks at double-newline boundaries.
///
/// Returns (byte_offset, paragraph_slice) pairs. Handles both \n\n (LF)
/// and \r\n\r\n (CRLF) paragraph separators.
fn split_paragraphs(text: &str) -> Vec<(usize, &str)> {
    let mut result = Vec::new();
    let mut prev = 0;
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        if i + 3 < len
            && bytes[i] == b'\r'
            && bytes[i + 1] == b'\n'
            && bytes[i + 2] == b'\r'
            && bytes[i + 3] == b'\n'
        {
            result.push((prev, &text[prev..i]));
            prev = i + 4;
            i = prev;
            continue;
        }
        if i + 1 < len && bytes[i] == b'\n' && bytes[i + 1] == b'\n' {
            result.push((prev, &text[prev..i]));
            prev = i + 2;
            i = prev;
            continue;
        }
        i += 1;
    }
    if prev < text.len() {
        result.push((prev, &text[prev..]));
    }
    result
}

/// Extract a surrounding text window (in chars) around a byte range.
///
/// Returns the substring spanning CONTEXT_WINDOW_CHARS characters before
/// the match start and after the match end, including the match itself.
pub(crate) fn surrounding_window(text: &str, start: usize, end: usize) -> &str {
    if text.is_empty() {
        return &text[0..0];
    }

    // Walk backward CONTEXT_WINDOW_CHARS characters from start.
    let mut byte_start = start;
    for _ in 0..CONTEXT_WINDOW_CHARS {
        if byte_start == 0 {
            break;
        }
        byte_start = text.floor_char_boundary(byte_start - 1);
    }

    // Walk forward CONTEXT_WINDOW_CHARS characters from end.
    let mut byte_end = end;
    for _ in 0..CONTEXT_WINDOW_CHARS {
        if byte_end >= text.len() {
            break;
        }
        byte_end = text.ceil_char_boundary(byte_end + 1);
    }

    &text[byte_start..byte_end]
}

/// Clamp a byte-offset window at excluded-range boundaries.
///
/// Given an unclamped window [win_start, win_end) around a match at
/// [match_start, match_end), narrows the window so it does not extend
/// past adjacent excluded ranges.  Snaps results to valid UTF-8 char
/// boundaries.  Returns (clamped_start, clamped_end).
fn clamp_at_excluded(
    text: &str,
    win_start: usize,
    win_end: usize,
    match_start: usize,
    match_end: usize,
    excluded: &[ByteRange],
) -> (usize, usize) {
    let mut clamped_start = win_start;
    let mut clamped_end = win_end;

    // Clamp left edge: excluded ranges ending before match_start.
    let left_idx = excluded.partition_point(|r| r.start < match_start);
    for excl in excluded[..left_idx].iter().rev() {
        if excl.end <= clamped_start {
            break;
        }
        if excl.end <= match_start && excl.end > clamped_start {
            clamped_start = excl.end;
        }
    }

    // Clamp right edge: excluded ranges starting after match_end.
    let right_idx = excluded.partition_point(|r| r.start < match_end);
    for excl in &excluded[right_idx..] {
        if excl.start >= clamped_end {
            break;
        }
        if excl.start >= match_end && excl.start < clamped_end {
            clamped_end = excl.start;
        }
    }

    // Snap inward to valid UTF-8 char boundaries.
    let clamped_start = text.ceil_char_boundary(clamped_start);
    let clamped_end = text.floor_char_boundary(clamped_end.min(text.len()));

    if clamped_start > clamped_end {
        (clamped_start, clamped_start)
    } else {
        (clamped_start, clamped_end)
    }
}

/// Like surrounding_window but clamps the window at excluded-range
/// boundaries so that context clues inside a code block (or other excluded
/// region) cannot influence rules that fire outside it.
pub(crate) fn surrounding_window_bounded<'a>(
    text: &'a str,
    start: usize,
    end: usize,
    excluded: &[ByteRange],
) -> &'a str {
    let window = surrounding_window(text, start, end);
    if excluded.is_empty() {
        return window;
    }

    let win_start = window.as_ptr() as usize - text.as_ptr() as usize;
    let win_end = win_start + window.len();
    let (cs, ce) = clamp_at_excluded(text, win_start, win_end, start, end, excluded);
    &text[cs..ce]
}

/// Remap issue offsets from NFC-normalized text back to original positions.
/// Updates offset, length, found text, and recomputes line/col.
fn remap_issues_to_original(issues: &mut [Issue], original: &str, norm: &Normalized) {
    let line_index = LineIndex::new(original);
    for issue in issues.iter_mut() {
        let orig_offset = map_offset(&norm.offset_map, issue.offset);
        let orig_end = map_offset(&norm.offset_map, issue.offset + issue.length);
        issue.offset = orig_offset;
        issue.length = orig_end.saturating_sub(orig_offset);
        let end = (orig_offset + issue.length).min(original.len());
        if let Some(found) = original.get(orig_offset..end) {
            issue.found = found.to_string();
        }
        let (line, col) = line_index.line_col(issue.offset, ColumnEncoding::Utf16);
        issue.line = line;
        issue.col = col;
    }
}

/// Build suggestion list from a rule's `to` and `english` fields.
///
/// Filters empty strings from `to`. If no suggestions remain, falls back to
/// the `english` field (used when no Chinese translation exists).
///
/// AiFiller deletion rules (`to: [""]`) are special: the empty string is
/// the intended suggestion (delete the filler phrase), so it is preserved
/// as-is instead of being filtered away.
fn effective_suggestions(rule: &SpellingRule) -> Vec<String> {
    // AiFiller deletion: to == [""] means 'delete this phrase'.
    // Preserve the empty-string suggestion so the fixer can apply it.
    if rule.is_deletion_rule() {
        return rule.to.clone();
    }
    let to = &rule.to;
    // Fast path: most rules have no empty strings in to.
    if !to.is_empty() && to.iter().all(|s| !s.is_empty()) {
        return to.clone();
    }
    let filtered: Vec<String> = to.iter().filter(|s| !s.is_empty()).cloned().collect();
    if !filtered.is_empty() {
        return filtered;
    }
    match rule.english.as_deref() {
        Some(e) if !e.is_empty() => vec![e.to_string()],
        _ => Vec::new(),
    }
}

/// Returns true if the text around the match already contains one of the
/// rule's correct forms as a superstring of the matched wrong term.
/// E.g., skip "算法" match when surrounding text reads "演算法".
///
/// Checks all positions where the wrong term appears within each correct
/// form (not just the first), handling cases like wrong="A", correct="ABA".
fn already_correct_form(text: &str, match_start: usize, rule: &SpellingRule) -> bool {
    for correct in &rule.to {
        for (wrong_pos, _) in correct.match_indices(&rule.from) {
            if let Some(correct_start) = match_start.checked_sub(wrong_pos) {
                let correct_end = correct_start + correct.len();
                // Use get() to avoid panic on non-char-boundary indices.
                if text.get(correct_start..correct_end) == Some(correct.as_str()) {
                    return true;
                }
            }
        }
    }
    false
}

/// Returns true if ch is a CJK ideograph (unified, extensions A-I,
/// compatibility, or bopomofo).  Excludes CJK Symbols/Punctuation
/// (U+3000..U+303F) to avoid false positives when full-width marks sit
/// next to half-width punctuation.
pub(crate) fn is_cjk_ideograph(ch: char) -> bool {
    matches!(ch,
        '\u{3100}'..='\u{312F}' |   // Bopomofo
        '\u{3400}'..='\u{4DBF}' |   // CJK Extension A
        '\u{4E00}'..='\u{9FFF}' |   // CJK Unified Ideographs
        '\u{F900}'..='\u{FAFF}' |   // CJK Compatibility Ideographs
        '\u{20000}'..='\u{2A6DF}' | // CJK Extension B
        '\u{2A700}'..='\u{2B73F}' | // CJK Extension C
        '\u{2B740}'..='\u{2B81F}' | // CJK Extension D
        '\u{2B820}'..='\u{2CEAF}' | // CJK Extension E
        '\u{2CEB0}'..='\u{2EBEF}' | // CJK Extension F
        '\u{2EBF0}'..='\u{2EE5F}' | // CJK Extension I
        '\u{30000}'..='\u{3134F}' | // CJK Extension G
        '\u{31350}'..='\u{323AF}'   // CJK Extension H
    )
}

/// Returns true if ch is a CJK context character — either a CJK ideograph
/// or a CJK punctuation/bracket mark.  Used by adjacent_cjk so that
/// text like 他說「你好」. correctly recognises 」 as CJK context.
pub(crate) fn is_cjk_context(ch: char) -> bool {
    is_cjk_ideograph(ch)
        || matches!(ch,
            // CJK Symbols and Punctuation (U+3001..U+303F, skip U+3000 = ideographic space)
            '\u{3001}'..='\u{303F}' |
            // Fullwidth Forms — fullwidth punctuation and letters (U+FF01..U+FF60)
            '\u{FF01}'..='\u{FF60}' |
            // Halfwidth CJK punctuation (U+FF61..U+FF65)
            '\u{FF61}'..='\u{FF65}' |
            // CJK Compatibility Forms (U+FE30..U+FE4F)
            '\u{FE30}'..='\u{FE4F}'
        )
}

/// Scan backward (before=true) or forward (before=false) from byte_pos,
/// skipping all Unicode whitespace (including ideographic space U+3000),
/// and return true if the first non-whitespace character is a CJK context
/// character (ideograph or CJK punctuation).
fn adjacent_cjk(text: &str, byte_pos: usize, before: bool) -> bool {
    adjacent_cjk_inner(text, byte_pos, before, usize::MAX)
}

/// Check whether the immediately adjacent character (no whitespace skip) is CJK.
fn immediate_cjk(text: &str, byte_pos: usize, before: bool) -> bool {
    adjacent_cjk_inner(text, byte_pos, before, 0)
}

/// Check whether the nearest non-whitespace character in the given direction
/// is a CJK context character.  `max_ws` limits how many whitespace chars
/// to skip (0 = immediate adjacency, usize::MAX = unlimited).
fn adjacent_cjk_inner(text: &str, byte_pos: usize, before: bool, max_ws: usize) -> bool {
    let mut ws = 0usize;
    if before {
        let mut pos = byte_pos;
        loop {
            if pos == 0 {
                return false;
            }
            pos = text.floor_char_boundary(pos - 1);
            let ch = match text[pos..].chars().next() {
                Some(c) => c,
                None => return false,
            };
            if ch.is_whitespace() {
                ws += 1;
                if ws > max_ws {
                    return false;
                }
                continue;
            }
            return is_cjk_context(ch);
        }
    } else {
        for ch in text[byte_pos..].chars() {
            if ch.is_whitespace() {
                ws += 1;
                if ws > max_ws {
                    return false;
                }
                continue;
            }
            return is_cjk_context(ch);
        }
        false
    }
}

/// Construct a punctuation Issue at the given byte offset with explicit severity.
fn punct_issue_sev(
    offset: usize,
    found: &str,
    suggestion: &str,
    context: &str,
    severity: Severity,
) -> Issue {
    Issue::new(
        offset,
        found.len(),
        found,
        vec![suggestion.into()],
        IssueType::Punctuation,
        severity,
    )
    .with_context(context)
}

/// Construct a Warning-severity punctuation Issue at the given byte offset.
fn punct_issue(offset: usize, found: &str, suggestion: &str, context: &str) -> Issue {
    punct_issue_sev(offset, found, suggestion, context, Severity::Warning)
}

/// Build exclusion ranges for text based on content type.
///
/// Combines content-pattern exclusions (URLs, paths, mentions) with
/// structural exclusions appropriate to the content type and inline
/// suppression markers.  Shared between CLI and MCP pipelines.
pub fn build_exclusions_for_content_type(text: &str, content_type: ContentType) -> Vec<ByteRange> {
    let mut excluded = build_excluded_ranges(text);
    match content_type {
        ContentType::Markdown => excluded.extend(build_markdown_excluded_ranges(text)),
        ContentType::MarkdownScanCode => {
            excluded.extend(build_markdown_excluded_ranges_no_code(text))
        }
        ContentType::Yaml => excluded.extend(build_yaml_excluded_ranges(text)),
        ContentType::Plain => {}
    }
    excluded.extend(build_suppression_ranges(text));
    merge_ranges_pub(excluded)
}

// Scanner struct and public API

/// Compiled scanner, reusable across multiple scan calls.
pub struct Scanner {
    /// Charwise double-array AC for spelling (primary; ~3x fewer states for CJK).
    spelling_ac_charwise: Option<daachorse::CharwiseDoubleArrayAhoCorasick<usize>>,
    /// Bytewise AC fallback (built only when charwise is unavailable).
    spelling_ac_bytewise: Option<AhoCorasick>,
    spelling_rules: Vec<SpellingRule>,
    /// Precomputed suggestions per rule (avoids per-match allocation).
    spelling_suggestions: Vec<Vec<String>>,

    case_ac: Option<AhoCorasick>,
    case_rules: Vec<CaseRule>,

    /// MMSEG segmenter for fixer context-clue checks and public accessor.
    segmenter: Segmenter,

    /// Bytewise AC over all unique clue strings (positive + negative).
    clue_ac: Option<AhoCorasick>,
    /// Per-rule positive clue IDs into the clue AC pattern list.
    rule_pos_clue_ids: Vec<Option<Vec<u16>>>,
    /// Per-rule negative clue IDs into the clue AC pattern list.
    rule_neg_clue_ids: Vec<Option<Vec<u16>>>,
    /// Per-rule parsed positional clues (checked after context-clue gate).
    rule_positional_clues: Vec<Option<Vec<PositionalClue>>>,
    /// Absorption patterns retained for force_bytewise() rebuild.
    #[cfg_attr(not(test), allow(dead_code))]
    absorber_strings: Vec<String>,
    /// Per-rule bitflags gating optional filter stages (spelling::FILTER_*).
    rule_filter_flags: Vec<u8>,
    /// Per-rule dispatch class for monomorphic fast paths (spelling::CLASS_*).
    /// Computed from filter flags at build time; determines which
    /// process_match_dispatch<CLASS> variant handles each AC hit.
    rule_classes: Vec<u8>,
}

impl Scanner {
    /// Read-only access to the spelling rules held by this scanner.
    pub fn spelling_rules(&self) -> &[SpellingRule] {
        &self.spelling_rules
    }

    /// Build a scanner from loaded rules.
    ///
    /// The spelling automaton matches literally (Chinese terms don't need
    /// case folding). The case automaton is ASCII-case-insensitive so it
    /// catches e.g. "javascript" when the canonical form is "JavaScript".
    pub fn new(spelling_rules: Vec<SpellingRule>, case_rules: Vec<CaseRule>) -> Self {
        let mut spelling_rules: Vec<SpellingRule> =
            spelling_rules.into_iter().filter(|r| !r.disabled).collect();
        let case_rules: Vec<CaseRule> = case_rules.into_iter().filter(|r| !r.disabled).collect();

        // Deduplicate by `from` key (last wins; overrides come after embedded).
        {
            let mut seen = std::collections::HashSet::new();
            let mut i = spelling_rules.len();
            while i > 0 {
                i -= 1;
                if !seen.insert(spelling_rules[i].from.clone()) {
                    spelling_rules.remove(i);
                }
            }
        }

        // Deduplicate context clues within each rule.
        for rule in &mut spelling_rules {
            if let Some(ref mut clues) = rule.context_clues {
                let mut seen = std::collections::HashSet::new();
                clues.retain(|c| seen.insert(c.clone()));
            }
            if let Some(ref mut clues) = rule.negative_context_clues {
                let mut seen = std::collections::HashSet::new();
                clues.retain(|c| seen.insert(c.clone()));
            }
        }

        let spelling_suggestions: Vec<Vec<String>> =
            spelling_rules.iter().map(effective_suggestions).collect();

        let segmenter = Segmenter::from_rules(&spelling_rules);

        // Build clue AC: intern all unique clue strings, map per-rule clue
        // lists to indices, build a bytewise AC for windowed lookups.
        let (clue_ac, rule_pos_clue_ids, rule_neg_clue_ids) = {
            let mut clue_map: std::collections::HashMap<String, u16> =
                std::collections::HashMap::new();
            let mut clue_vec: Vec<String> = Vec::new();

            let mut intern_clue = |s: &String| -> u16 {
                if let Some(&idx) = clue_map.get(s) {
                    idx
                } else {
                    let idx = u16::try_from(clue_vec.len()).expect("clue index overflow");
                    clue_map.insert(s.clone(), idx);
                    clue_vec.push(s.clone());
                    idx
                }
            };

            let mut pos_ids: Vec<Option<Vec<u16>>> = Vec::with_capacity(spelling_rules.len());
            let mut neg_ids: Vec<Option<Vec<u16>>> = Vec::with_capacity(spelling_rules.len());

            for rule in &spelling_rules {
                let pos = rule.context_clues.as_ref().and_then(|clues| {
                    if clues.is_empty() {
                        None
                    } else {
                        Some(clues.iter().map(&mut intern_clue).collect())
                    }
                });
                let neg = rule.negative_context_clues.as_ref().and_then(|clues| {
                    if clues.is_empty() {
                        None
                    } else {
                        Some(clues.iter().map(&mut intern_clue).collect())
                    }
                });
                pos_ids.push(pos);
                neg_ids.push(neg);
            }

            let ac = if clue_vec.is_empty() {
                None
            } else {
                match AhoCorasickBuilder::new()
                    .match_kind(MatchKind::Standard)
                    .build(&clue_vec)
                {
                    Ok(ac) => Some(ac),
                    Err(e) => {
                        eprintln!("[zhtw-mcp] clue AC build failed: {e}");
                        None
                    }
                }
            };

            (ac, pos_ids, neg_ids)
        };

        // Validate clue-ID counts fit the fixed bitset (capacity 32).
        for (i, pos) in rule_pos_clue_ids.iter().enumerate() {
            if let Some(ids) = pos {
                assert!(
                    ids.len() <= 32,
                    "rule '{}' has {} positive clues, exceeds bitset capacity 32",
                    spelling_rules[i].from,
                    ids.len(),
                );
            }
        }
        for (i, neg) in rule_neg_clue_ids.iter().enumerate() {
            if let Some(ids) = neg {
                assert!(
                    ids.len() <= 32,
                    "rule '{}' has {} negative clues, exceeds bitset capacity 32",
                    spelling_rules[i].from,
                    ids.len(),
                );
            }
        }

        let rule_positional_clues: Vec<Option<Vec<PositionalClue>>> = spelling_rules
            .iter()
            .map(|rule| {
                rule.positional_clues.as_ref().and_then(|raw| {
                    let parsed: Vec<PositionalClue> = raw
                        .iter()
                        .filter_map(|s| {
                            let clue = PositionalClue::parse(s);
                            if clue.is_none() {
                                eprintln!(
                                    "[zhtw-mcp] rule '{}': unrecognized positional clue '{}'",
                                    rule.from, s
                                );
                            }
                            clue
                        })
                        .collect();
                    if parsed.is_empty() {
                        None
                    } else {
                        Some(parsed)
                    }
                })
            })
            .collect();

        let spelling_patterns: Vec<&str> = spelling_rules.iter().map(|r| r.from.as_str()).collect();

        // Absorption patterns: exception phrases and superstring `to` forms
        // injected into the AC so LeftmostLongest suppresses shorter `from`
        // matches.  Indices >= spelling_rules.len() act as sentinels.
        let absorber_strings: Vec<String> = {
            let from_set: std::collections::HashSet<&str> =
                spelling_patterns.iter().copied().collect();
            let mut candidates: Vec<(String, &str)> = Vec::new();
            let mut dedup = std::collections::HashSet::new();
            for rule in &spelling_rules {
                if let Some(ref exceptions) = rule.exceptions {
                    for exc in exceptions {
                        if exc.contains(&rule.from)
                            && !from_set.contains(exc.as_str())
                            && dedup.insert(exc.clone())
                        {
                            candidates.push((exc.clone(), rule.from.as_str()));
                        }
                    }
                }
                for to in &rule.to {
                    if to.contains(&rule.from)
                        && to != &rule.from
                        && !from_set.contains(to.as_str())
                        && dedup.insert(to.clone())
                    {
                        candidates.push((to.clone(), rule.from.as_str()));
                    }
                }
            }
            // Reject absorbers that would shadow a different rule's `from`.
            candidates
                .into_iter()
                .filter(|(absorber, orig_from)| {
                    !spelling_patterns.iter().any(|&f| {
                        if f == *orig_from {
                            return false;
                        }
                        if absorber.contains(f) {
                            return true;
                        }
                        // Right-boundary overlap: proper suffix of absorber
                        // is a prefix of f.
                        let mut chars = absorber.char_indices();
                        chars.next(); // skip position 0 (full string)
                        for (byte_idx, _) in chars {
                            let suffix = &absorber[byte_idx..];
                            if f.starts_with(suffix) {
                                return true;
                            }
                        }
                        false
                    })
                })
                .map(|(s, _)| s)
                .collect()
        };
        let all_patterns: Vec<&str> = spelling_patterns
            .iter()
            .copied()
            .chain(absorber_strings.iter().map(|s| s.as_str()))
            .collect();

        let spelling_ac_charwise = {
            let patvals: Vec<(&str, usize)> = all_patterns
                .iter()
                .enumerate()
                .map(|(i, &p)| (p, i))
                .collect();
            match CharwiseDoubleArrayAhoCorasickBuilder::new()
                .match_kind(DaacMatchKind::LeftmostLongest)
                .build_with_values(patvals)
            {
                Ok(ac) => Some(ac),
                Err(e) => {
                    eprintln!("[zhtw-mcp] charwise AC build failed, using bytewise fallback: {e}");
                    None
                }
            }
        };

        let spelling_ac_bytewise = if spelling_ac_charwise.is_none() {
            match AhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostLongest)
                .build(&all_patterns)
            {
                Ok(ac) => Some(ac),
                Err(e) => {
                    eprintln!("[zhtw-mcp] bytewise spelling AC build failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        let case_patterns: Vec<String> = case_rules.iter().map(|r| r.term.to_lowercase()).collect();

        let case_ac = match AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .ascii_case_insensitive(true)
            .build(&case_patterns)
        {
            Ok(ac) => Some(ac),
            Err(e) => {
                eprintln!("[zhtw-mcp] case AC build failed: {e}");
                None
            }
        };

        let rule_filter_flags: Vec<u8> = spelling_rules
            .iter()
            .enumerate()
            .map(|(i, rule)| {
                let mut f: u8 = 0;
                if rule.to.iter().any(|t| t.contains(&rule.from)) {
                    f |= spelling::FILTER_HAS_SUPERSTRING;
                }
                if rule.exceptions.as_ref().is_some_and(|v| !v.is_empty()) {
                    f |= spelling::FILTER_HAS_EXCEPTIONS;
                }
                if rule_pos_clue_ids[i].is_some() {
                    f |= spelling::FILTER_HAS_POS_CLUES;
                }
                if rule_neg_clue_ids[i].is_some() {
                    f |= spelling::FILTER_HAS_NEG_CLUES;
                }
                if rule_positional_clues[i].is_some() {
                    f |= spelling::FILTER_HAS_POSITIONAL;
                }
                if rule.is_deletion_rule() {
                    f |= spelling::FILTER_IS_DELETION;
                }
                f
            })
            .collect();

        // Dispatch class per rule: determines which monomorphic fast path
        // handles each AC hit.  Positional implies FULL; context clues
        // without positional implies CLUED; everything else is SIMPLE.
        let rule_classes: Vec<u8> = rule_filter_flags
            .iter()
            .map(|&f| {
                if f & spelling::FILTER_HAS_POSITIONAL != 0 {
                    spelling::CLASS_FULL
                } else if f & (spelling::FILTER_HAS_POS_CLUES | spelling::FILTER_HAS_NEG_CLUES) != 0
                {
                    spelling::CLASS_CLUED
                } else {
                    spelling::CLASS_SIMPLE
                }
            })
            .collect();

        Self {
            spelling_ac_charwise,
            spelling_ac_bytewise,
            spelling_rules,
            spelling_suggestions,
            case_ac,
            case_rules,
            segmenter,
            clue_ac,
            rule_pos_clue_ids,
            rule_neg_clue_ids,
            rule_positional_clues,
            absorber_strings,
            rule_filter_flags,
            rule_classes,
        }
    }

    /// Access the internal segmenter for context-clue analysis.
    pub fn segmenter(&self) -> &Segmenter {
        &self.segmenter
    }

    /// Force the scanner to use the bytewise AC fallback path for testing.
    /// Disables charwise and builds bytewise if not already present.
    #[cfg(test)]
    fn force_bytewise(&mut self) {
        if self.spelling_ac_bytewise.is_none() {
            let patterns: Vec<&str> = self
                .spelling_rules
                .iter()
                .map(|r| r.from.as_str())
                .chain(self.absorber_strings.iter().map(|s| s.as_str()))
                .collect();
            self.spelling_ac_bytewise = Some(
                AhoCorasickBuilder::new()
                    .match_kind(MatchKind::LeftmostLongest)
                    .build(&patterns)
                    .expect("build bytewise AC for test"),
            );
        }
        self.spelling_ac_charwise = None;
    }

    /// Scan text with Profile::Default and return all issues found.
    ///
    /// Applies NFC normalization, builds excluded ranges (including inline
    /// suppression markers), then scans and maps offsets back to the
    /// original text. Use scan_profiled for non-default profiles.
    pub fn scan(&self, text: &str) -> ScanOutput {
        self.scan_profiled(text, Profile::Default)
    }

    /// Scan text with the given profile and return all issues found.
    ///
    /// Uses pulldown-cmark for code block / inline code exclusion (handles
    /// both plain text and Markdown gracefully), plus regex-based exclusion
    /// for URLs, file paths, and @mentions.
    pub fn scan_profiled(&self, text: &str, profile: Profile) -> ScanOutput {
        self.scan_profiled_md(text, profile, true)
    }

    /// Scan with explicit control over Markdown structure exclusion.
    ///
    /// When use_markdown is true, pulldown-cmark detects code blocks (fenced
    /// and indented), inline code, and HTML -- matching Markdown input.
    /// When false, only content-pattern exclusions (URLs, paths, @mentions) and
    /// inline suppression markers are applied. Use false for plain text to
    /// avoid 4-space-indented paragraphs being falsely excluded as code.
    pub fn scan_profiled_md(&self, text: &str, profile: Profile, use_markdown: bool) -> ScanOutput {
        let content_type = if use_markdown {
            ContentType::Markdown
        } else {
            ContentType::Plain
        };
        self.scan_nfc_with_content_type(text, None, profile.config(), content_type)
    }

    /// Scan YAML text with key-token exclusion.
    ///
    /// Excludes YAML key tokens (key name + colon) so that bare ASCII colons
    /// in key-value separators do not trigger false-positive colon warnings.
    /// YAML values after the colon are scanned normally as prose.
    pub fn scan_profiled_yaml(&self, text: &str, profile: Profile) -> ScanOutput {
        self.scan_nfc_with_content_type(text, None, profile.config(), ContentType::Yaml)
    }

    /// Scan with NFC normalization, reusing pre-built excluded ranges.
    ///
    /// When the input text is already NFC (common case), the provided
    /// excluded ranges are used directly, avoiding a redundant
    /// recomputation of exclusion zones. When NFC normalization changes
    /// byte offsets, exclusions are rebuilt on the normalized text.
    ///
    /// content_type controls which structural exclusion pass is applied
    /// during the NFC-rebuild slow path (Markdown, YAML, or plain text).
    pub fn scan_with_prebuilt_excluded(
        &self,
        text: &str,
        excluded: &[ByteRange],
        profile: Profile,
        content_type: ContentType,
    ) -> ScanOutput {
        self.scan_nfc_with_content_type(text, Some(excluded), profile.config(), content_type)
    }

    /// Like scan_with_prebuilt_excluded but with explicit ProfileConfig.
    pub fn scan_with_prebuilt_excluded_config(
        &self,
        text: &str,
        excluded: &[ByteRange],
        cfg: ProfileConfig,
        content_type: ContentType,
    ) -> ScanOutput {
        self.scan_nfc_with_content_type(text, Some(excluded), cfg, content_type)
    }

    /// Scan text using the content-type-aware exclusion strategy.
    ///
    /// Shared entry point for CLI and MCP pipelines (20.4 deduplication).
    /// Dispatches to the appropriate scan method based on content type.
    pub fn scan_for_content_type(
        &self,
        text: &str,
        content_type: ContentType,
        profile: Profile,
    ) -> ScanOutput {
        self.scan_nfc_with_content_type(text, None, profile.config(), content_type)
    }

    /// Scan with content-type-aware exclusions and explicit ProfileConfig.
    /// Use this when the caller needs to override individual config flags
    /// (e.g. detect_ai enabling density detection on a non-editorial profile).
    pub fn scan_for_content_type_with_config(
        &self,
        text: &str,
        content_type: ContentType,
        cfg: ProfileConfig,
    ) -> ScanOutput {
        self.scan_nfc_with_content_type(text, None, cfg, content_type)
    }

    /// Core NFC-normalize → build exclusions → scan → remap pipeline.
    fn scan_nfc_with_content_type(
        &self,
        text: &str,
        prebuilt_excluded: Option<&[ByteRange]>,
        cfg: ProfileConfig,
        content_type: ContentType,
    ) -> ScanOutput {
        let norm = normalize_nfc(text);
        let scan_text = &norm.text;
        let nfc_changed = !norm.offset_map.is_empty();

        let mut output = match prebuilt_excluded {
            Some(excl) if !nfc_changed => self.scan_with_config(scan_text, excl, cfg),
            _ => {
                let excl = build_exclusions_for_content_type(scan_text, content_type);
                self.scan_with_config(scan_text, &excl, cfg)
            }
        };

        if nfc_changed {
            remap_issues_to_original(&mut output.issues, text, &norm);
        }

        output
    }

    /// Scan text using pre-built excluded ranges and a profile.
    ///
    /// Use this when the caller also needs the excluded ranges for a
    /// subsequent apply_fixes call, avoiding a redundant recomputation.
    ///
    /// excluded must be sorted by start position and non-overlapping
    /// (as returned by build_excluded_ranges). The is_excluded check
    /// uses binary search for large lists and will produce wrong results
    /// if ranges are unsorted.
    pub fn scan_with_excluded(
        &self,
        text: &str,
        excluded: &[ByteRange],
        profile: Profile,
    ) -> ScanOutput {
        self.scan_with_config(text, excluded, profile.config())
    }

    /// Scan with a fully-specified ProfileConfig (allows stance overrides).
    pub fn scan_with_config(
        &self,
        text: &str,
        excluded: &[ByteRange],
        cfg: ProfileConfig,
    ) -> ScanOutput {
        if text.is_empty() {
            return ScanOutput {
                issues: Vec::new(),
                detected_script: ChineseType::Unknown,
                ai_signature: None,
            };
        }

        let zh_type = detect_chinese_type(text);

        // Pre-allocate with a capacity estimate based on text length.
        // Typical CJK prose yields ~1 issue per 2KB, so this avoids
        // incremental reallocation for most documents.
        let estimated_issues = (text.len() / 2048).max(8);
        let mut issues = Vec::with_capacity(estimated_issues);
        if cfg.spelling {
            self.scan_spelling(text, excluded, zh_type, &mut issues, &cfg);
        }
        if cfg.casing {
            self.scan_case(text, excluded, &mut issues);
        }
        if cfg.basic_punctuation {
            self.scan_punctuation(text, excluded, &mut issues, &cfg);
        }
        if cfg.dunhao_detection {
            self.scan_dunhao(text, excluded, &mut issues);
        }
        if cfg.range_normalization {
            self.scan_range_indicators(text, excluded, &mut issues, &cfg);
        }
        if cfg.ellipsis_normalization {
            scan_ellipsis(text, excluded, &mut issues);
        }
        if cfg.basic_punctuation {
            self.scan_cn_curly_quotes(text, excluded, &mut issues);
            self.scan_spacing(text, excluded, &mut issues);
        }
        // Sort by offset, then by length (longer match first for same offset).
        issues.sort_by(|a, b| a.offset.cmp(&b.offset).then(b.length.cmp(&a.length)));

        // Remove overlapping issues: longer match wins; on tie, higher severity
        // wins. Handles both same-offset and cross-offset overlaps.
        resolve_overlaps(&mut issues);

        // Grammar checks run AFTER overlap resolution so broad grammar spans
        // (e.g. 是不是…嗎) do not suppress narrower spelling/case issues
        // that happen to fall inside the grammar match range.
        if cfg.grammar_checks {
            grammar::scan_grammar(text, excluded, &mut issues);
        }

        // AI writing detection grammar checks: semantic safety words,
        // copula avoidance, passive voice overuse.  Separate from base grammar
        // checks — gated by ai_semantic_safety profile flag.
        if cfg.ai_semantic_safety {
            grammar::scan_ai_grammar(text, excluded, &mut issues);
        }

        // Structural AI pattern detection: binary contrast density,
        // paragraph endings, dash overuse, formulaic headings, list density.
        if cfg.ai_structural_patterns {
            grammar::scan_ai_structural(text, excluded, &mut issues, cfg.ai_threshold_multiplier);
        }

        // Density-based AI phrase detection: post-scan frequency pass counts
        // tracked phrases across the full document and flags when density
        // exceeds per-phrase thresholds.  Distinct from per-occurrence filler
        // detection — catches the statistical signature of AI writing.
        if cfg.ai_density_detection {
            grammar::scan_ai_density(text, excluded, &mut issues, cfg.ai_threshold_multiplier);
        }

        // Fix CN quotation mark pairing with depth-based nesting:
        // well-formed quotes use character-based depth tracking; misordered
        // or all-same-char quotes fall back to positional alternation.
        // Paragraph breaks reset nesting depth.
        fix_quote_pairing(text, &mut issues);

        // Validate structural nesting of existing TW bracket quotes:
        // checks for mismatched, interleaved, and unclosed quotes per paragraph.
        validate_quote_hierarchy(text, excluded, &mut issues);

        // Compute AI signature score when any AI detection flag is active.
        let ai_signature = if cfg.ai_filler_detection
            || cfg.ai_semantic_safety
            || cfg.ai_density_detection
            || cfg.ai_structural_patterns
        {
            crate::engine::ai_score::compute_ai_score(
                text,
                &issues,
                excluded,
                cfg.ai_threshold_multiplier,
            )
        } else {
            None
        };

        // Skip O(n) line index construction when no issues found (common case).
        if issues.is_empty() {
            return ScanOutput {
                issues,
                detected_script: zh_type,
                ai_signature,
            };
        }

        // Fill line/col coordinates from the pre-computed line index.
        let line_index = LineIndex::new(text);
        for issue in &mut issues {
            let (line, col) = line_index.line_col(issue.offset, ColumnEncoding::Utf16);
            issue.line = line;
            issue.col = col;
        }

        // Deterministic output contract: issues are sorted by byte offset
        // ascending, then severity descending, then rule_type discriminant for
        // stable, diffable output.
        issues.sort_by(|a, b| {
            a.offset
                .cmp(&b.offset)
                .then(b.severity.cmp(&a.severity))
                .then(a.rule_type.sort_order().cmp(&b.rule_type.sort_order()))
        });

        ScanOutput {
            issues,
            detected_script: zh_type,
            ai_signature,
        }
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::ruleset::RuleType;

    fn sample_spelling_rules() -> Vec<SpellingRule> {
        vec![
            SpellingRule::new("軟件", vec!["軟體".into()], RuleType::CrossStrait),
            SpellingRule::new("內存", vec!["記憶體".into()], RuleType::CrossStrait),
            SpellingRule::new("服務器", vec!["伺服器".into()], RuleType::CrossStrait),
        ]
    }

    fn sample_case_rules() -> Vec<CaseRule> {
        vec![
            CaseRule {
                term: "JavaScript".into(),
                alternatives: Some(vec!["javascript".into(), "JAVASCRIPT".into()]),
                disabled: false,
            },
            CaseRule {
                term: "TypeScript".into(),
                alternatives: None,
                disabled: false,
            },
            CaseRule {
                term: "API".into(),
                alternatives: Some(vec!["Api".into(), "api".into(), "APIs".into()]),
                disabled: false,
            },
        ]
    }

    #[test]
    fn basic_spelling_detection() {
        let scanner = Scanner::new(sample_spelling_rules(), vec![]);
        let issues = scanner.scan("這個軟件很好用").issues;
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].found, "軟件");
        assert_eq!(issues[0].suggestions, vec!["軟體"]);
        assert_eq!(issues[0].rule_type, IssueType::CrossStrait);
    }

    #[test]
    fn multiple_spelling_issues() {
        let scanner = Scanner::new(sample_spelling_rules(), vec![]);
        let issues = scanner.scan("這個軟件的服務器內存不夠").issues;
        assert_eq!(issues.len(), 3);
        assert_eq!(issues[0].found, "軟件");
        assert_eq!(issues[1].found, "服務器");
        assert_eq!(issues[2].found, "內存");
    }

    #[test]
    fn spelling_in_code_fence_excluded() {
        let scanner = Scanner::new(sample_spelling_rules(), vec![]);
        let issues = scanner.scan("請看 `軟件` 的說明").issues;
        assert_eq!(issues.len(), 0);
    }

    #[test]
    fn spelling_in_url_excluded() {
        let scanner = Scanner::new(sample_spelling_rules(), vec![]);
        let issues = scanner
            .scan("https://example.com/軟件/download 是連結")
            .issues;
        assert_eq!(
            issues.len(),
            0,
            "CJK inside URL path should be excluded: {issues:?}"
        );
    }

    #[test]
    fn case_rule_basic() {
        let scanner = Scanner::new(vec![], sample_case_rules());
        let issues = scanner.scan("I use Javascript for work").issues;
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].found, "Javascript");
        assert_eq!(issues[0].suggestions, vec!["JavaScript"]);
        assert_eq!(issues[0].rule_type, IssueType::Case);
    }

    #[test]
    fn case_rule_correct_form_no_issue() {
        let scanner = Scanner::new(vec![], sample_case_rules());
        let issues = scanner.scan("I use JavaScript for work").issues;
        assert_eq!(issues.len(), 0);
    }

    #[test]
    fn case_rule_alternative_no_issue() {
        let scanner = Scanner::new(vec![], sample_case_rules());
        let issues = scanner.scan("I use javascript for work").issues;
        assert_eq!(issues.len(), 0);
    }

    #[test]
    fn case_rule_word_boundary() {
        let scanner = Scanner::new(vec![], sample_case_rules());
        let issues = scanner.scan("This is Unreactive").issues;
        assert_eq!(issues.len(), 0);
    }

    #[test]
    fn case_rule_in_code_excluded() {
        let scanner = Scanner::new(vec![], sample_case_rules());
        let issues = scanner.scan("Use `typescript` in your code").issues;
        assert_eq!(issues.len(), 0);
    }

    #[test]
    fn mixed_spelling_and_case() {
        let scanner = Scanner::new(sample_spelling_rules(), sample_case_rules());
        let issues = scanner.scan("這個軟件用 typescript 寫的").issues;
        assert_eq!(issues.len(), 2);
        assert_eq!(issues[0].found, "軟件");
        assert_eq!(issues[1].found, "typescript");
    }

    #[test]
    fn empty_text() {
        let scanner = Scanner::new(sample_spelling_rules(), sample_case_rules());
        let issues = scanner.scan("").issues;
        assert!(issues.is_empty());
    }

    #[test]
    fn clean_text_no_issues() {
        let scanner = Scanner::new(sample_spelling_rules(), sample_case_rules());
        let issues = scanner.scan("這個軟體用 TypeScript 寫的").issues;
        assert!(issues.is_empty());
    }

    #[test]
    fn api_case_wrong() {
        let scanner = Scanner::new(vec![], sample_case_rules());
        let issues = scanner.scan("This aPi is slow").issues;
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].found, "aPi");
        assert_eq!(issues[0].suggestions, vec!["API"]);
    }

    #[test]
    fn api_case_correct_alternatives() {
        let scanner = Scanner::new(vec![], sample_case_rules());
        assert!(scanner.scan("The API is fast").issues.is_empty());
        assert!(scanner.scan("The Api is fast").issues.is_empty());
        assert!(scanner.scan("The api is fast").issues.is_empty());
    }

    // Spelling AC (charwise / bytewise) tests

    #[test]
    fn charwise_ac_is_built_for_cjk_patterns() {
        let scanner = Scanner::new(sample_spelling_rules(), vec![]);
        assert!(
            scanner.spelling_ac_charwise.is_some(),
            "charwise AC should be built for CJK-only patterns"
        );
    }

    #[test]
    fn charwise_and_bytewise_produce_identical_results() {
        let rules = sample_spelling_rules();
        let text = "這個軟件的服務器內存不夠，需要升級軟件的記憶體";
        let scanner = Scanner::new(rules.clone(), vec![]);

        // Run with charwise (default path).
        let charwise_issues = scanner.scan(text).issues;

        // Force bytewise path for comparison.
        let mut bytewise_scanner = Scanner::new(rules, vec![]);
        bytewise_scanner.force_bytewise();
        let bytewise_issues = bytewise_scanner.scan(text).issues;

        assert_eq!(
            charwise_issues.len(),
            bytewise_issues.len(),
            "charwise and bytewise should find the same number of issues"
        );
        for (cw, bw) in charwise_issues.iter().zip(bytewise_issues.iter()) {
            assert_eq!(cw.offset, bw.offset, "offsets must match");
            assert_eq!(cw.length, bw.length, "lengths must match");
            assert_eq!(cw.found, bw.found, "found text must match");
            assert_eq!(cw.suggestions, bw.suggestions, "suggestions must match");
        }
    }

    #[test]
    fn charwise_leftmost_longest_on_overlapping_patterns() {
        // "數據" and "數據庫" overlap — leftmost-longest must pick "數據庫".
        let rules = vec![
            SpellingRule::new("數據", vec!["資料".into()], RuleType::CrossStrait),
            SpellingRule::new("數據庫", vec!["資料庫".into()], RuleType::CrossStrait),
        ];
        let scanner = Scanner::new(rules, vec![]);
        assert!(scanner.spelling_ac_charwise.is_some());

        let issues = scanner.scan("這個數據庫很大").issues;
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].found, "數據庫");
        assert_eq!(issues[0].suggestions, vec!["資料庫"]);
    }

    #[test]
    fn charwise_single_char_cjk_pattern() {
        // Single CJK character pattern: shortest possible charwise match.
        let rules = vec![SpellingRule::new(
            "裏",
            vec!["裡".into()],
            RuleType::Variant,
        )];
        let scanner = Scanner::new(rules, vec![]);
        assert!(scanner.spelling_ac_charwise.is_some());

        let issues = scanner
            .scan_profiled("裏面有東西", Profile::StrictMoe)
            .issues;
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].found, "裏");
        assert_eq!(issues[0].suggestions, vec!["裡"]);
    }

    #[test]
    fn charwise_mixed_cjk_ascii_patterns() {
        // Patterns with both CJK and ASCII characters.
        let rules = vec![
            SpellingRule::new("IP地址", vec!["IP 位址".into()], RuleType::CrossStrait),
            SpellingRule::new(
                "CPU使用率",
                vec!["CPU 使用率".into()],
                RuleType::CrossStrait,
            ),
        ];
        let scanner = Scanner::new(rules, vec![]);
        assert!(scanner.spelling_ac_charwise.is_some());

        let issues = scanner.scan("查看IP地址和CPU使用率").issues;
        let spelling: Vec<_> = issues
            .iter()
            .filter(|i| i.rule_type == IssueType::from(RuleType::CrossStrait))
            .collect();
        assert_eq!(spelling.len(), 2);
        assert_eq!(spelling[0].found, "IP地址");
        assert_eq!(spelling[1].found, "CPU使用率");
    }

    #[test]
    fn charwise_exception_phrase_respected() {
        // Exception phrases must work identically on both AC paths.
        let rules = vec![SpellingRule {
            from: "著".into(),
            to: vec!["著".into()],
            rule_type: RuleType::Variant,
            disabled: false,
            context: None,
            english: None,
            context_clues: None,
            negative_context_clues: None,
            exceptions: Some(vec!["下著".into()]),
            positional_clues: None,
            tags: None,
        }];
        let scanner = Scanner::new(rules, vec![]);
        assert!(scanner.spelling_ac_charwise.is_some());

        // "下著" is an exception — should not fire.
        let issues = scanner.scan_profiled("下著棋", Profile::StrictMoe).issues;
        assert!(
            issues.is_empty(),
            "exception phrase '下著' should suppress the match: {issues:?}"
        );
    }

    #[test]
    fn charwise_context_clues_gate() {
        // Context clues must gate correctly on the charwise path.
        let rules = vec![SpellingRule {
            from: "支持".into(),
            to: vec!["支援".into()],
            rule_type: RuleType::CrossStrait,
            disabled: false,
            context: None,
            english: None,
            context_clues: Some(vec!["程式".into(), "軟體".into()]),
            negative_context_clues: None,
            exceptions: None,
            positional_clues: None,
            tags: None,
        }];
        let scanner = Scanner::new(rules, vec![]);
        assert!(scanner.spelling_ac_charwise.is_some());

        // No context clue present — should NOT fire.
        let issues = scanner.scan("我支持你的決定").issues;
        assert!(
            issues.is_empty(),
            "should not fire without context clues: {issues:?}"
        );

        // Context clue present — SHOULD fire.
        let issues = scanner.scan("這個程式支持多種格式").issues;
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].found, "支持");
    }

    #[test]
    fn charwise_negative_clues_veto() {
        // Negative context clues must veto correctly on the charwise path.
        let rules = vec![SpellingRule {
            from: "卸載".into(),
            to: vec!["解除安裝".into()],
            rule_type: RuleType::CrossStrait,
            disabled: false,
            context: None,
            english: None,
            context_clues: None,
            negative_context_clues: Some(vec!["掛載".into(), "mount".into()]),
            exceptions: None,
            positional_clues: None,
            tags: None,
        }];
        let scanner = Scanner::new(rules, vec![]);
        assert!(scanner.spelling_ac_charwise.is_some());

        // No negative clue — should fire.
        let issues = scanner.scan("請卸載這個應用程式").issues;
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].found, "卸載");

        // Negative clue present — should NOT fire.
        let issues = scanner.scan("掛載和卸載檔案系統").issues;
        assert!(
            issues.is_empty(),
            "negative clue '掛載' should veto: {issues:?}"
        );
    }

    #[test]
    fn bytewise_fallback_when_charwise_unavailable() {
        // Force bytewise path, verify results still correct.
        let rules = sample_spelling_rules();
        let mut scanner = Scanner::new(rules, vec![]);
        scanner.force_bytewise();

        let issues = scanner.scan("這個軟件的服務器內存不夠").issues;
        assert_eq!(issues.len(), 3);
        assert_eq!(issues[0].found, "軟件");
        assert_eq!(issues[1].found, "服務器");
        assert_eq!(issues[2].found, "內存");
    }

    #[test]
    fn charwise_many_patterns_same_prefix() {
        // Stress the double-array trie with patterns sharing a common prefix.
        let rules = vec![
            {
                let mut r = SpellingRule::new("數", vec!["數".into()], RuleType::CrossStrait);
                r.context_clues = Some(vec!["不存在的線索".into()]);
                r
            },
            SpellingRule::new("數據", vec!["資料".into()], RuleType::CrossStrait),
            SpellingRule::new("數據庫", vec!["資料庫".into()], RuleType::CrossStrait),
            SpellingRule::new("數據結構", vec!["資料結構".into()], RuleType::CrossStrait),
        ];
        let scanner = Scanner::new(rules, vec![]);
        assert!(scanner.spelling_ac_charwise.is_some());

        // Leftmost-longest: "數據結構" beats "數據" beats "數".
        let issues = scanner.scan("學習數據結構").issues;
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].found, "數據結構");
        assert_eq!(issues[0].suggestions, vec!["資料結構"]);

        // When only "數據" present, the shorter match wins.
        let issues = scanner.scan("處理數據").issues;
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].found, "數據");

        // "數" alone has context_clues that won't match, so it stays quiet.
        let issues = scanner.scan("數字很大").issues;
        assert!(issues.is_empty());
    }

    #[test]
    fn charwise_adjacent_non_overlapping_matches() {
        // Two patterns that appear back-to-back without overlap.
        let rules = vec![
            SpellingRule::new("軟件", vec!["軟體".into()], RuleType::CrossStrait),
            SpellingRule::new("開發", vec!["研發".into()], RuleType::CrossStrait),
        ];
        let scanner = Scanner::new(rules, vec![]);
        assert!(scanner.spelling_ac_charwise.is_some());

        // "軟件開發" — both patterns match adjacently.
        let issues = scanner.scan("軟件開發很重要").issues;
        assert_eq!(issues.len(), 2);
        assert_eq!(issues[0].found, "軟件");
        assert_eq!(issues[1].found, "開發");
    }

    #[test]
    fn charwise_full_ruleset_builds() {
        // Verify the embedded ruleset (776+ patterns) builds charwise successfully.
        let ruleset = crate::rules::loader::load_embedded_ruleset().unwrap();
        let scanner = Scanner::new(ruleset.spelling_rules, ruleset.case_rules);
        assert!(
            scanner.spelling_ac_charwise.is_some(),
            "charwise AC should build for the full embedded ruleset"
        );
    }

    // --- positional_clues tests ---

    #[test]
    fn positional_before_fires_when_term_follows() {
        // before:函式 means 函式 must appear within 20 chars AFTER the match.
        let rules = vec![SpellingRule {
            from: "調用".into(),
            to: vec!["呼叫".into()],
            rule_type: RuleType::CrossStrait,
            disabled: false,
            context: None,
            english: None,
            context_clues: None,
            negative_context_clues: None,
            exceptions: None,
            positional_clues: Some(vec!["before:函式".into()]),
            tags: None,
        }];
        let scanner = Scanner::new(rules, vec![]);

        // 函式 follows 調用 — should fire.
        let issues = scanner.scan("請調用函式來處理").issues;
        assert_eq!(issues.len(), 1, "should fire when 函式 follows: {issues:?}");
        assert_eq!(issues[0].found, "調用");

        // 函式 absent — should NOT fire.
        let issues = scanner.scan("請調用這個方法").issues;
        assert!(
            issues.is_empty(),
            "should not fire without 函式 after match: {issues:?}"
        );
    }

    #[test]
    fn positional_after_fires_when_term_precedes() {
        // after:請 means 請 must appear within 20 chars BEFORE the match.
        let rules = vec![SpellingRule {
            from: "調用".into(),
            to: vec!["呼叫".into()],
            rule_type: RuleType::CrossStrait,
            disabled: false,
            context: None,
            english: None,
            context_clues: None,
            negative_context_clues: None,
            exceptions: None,
            positional_clues: Some(vec!["after:請".into()]),
            tags: None,
        }];
        let scanner = Scanner::new(rules, vec![]);

        // 請 precedes 調用 — should fire.
        let issues = scanner.scan("請調用函式").issues;
        assert_eq!(issues.len(), 1, "should fire when 請 precedes: {issues:?}");

        // 請 absent — should NOT fire.
        let issues = scanner.scan("直接調用函式").issues;
        assert!(
            issues.is_empty(),
            "should not fire without 請 before match: {issues:?}"
        );
    }

    #[test]
    fn positional_adjacent_fires_when_immediately_next() {
        // adjacent:函式 means 函式 must be immediately adjacent (no gap).
        let rules = vec![SpellingRule {
            from: "調用".into(),
            to: vec!["呼叫".into()],
            rule_type: RuleType::CrossStrait,
            disabled: false,
            context: None,
            english: None,
            context_clues: None,
            negative_context_clues: None,
            exceptions: None,
            positional_clues: Some(vec!["adjacent:函式".into()]),
            tags: None,
        }];
        let scanner = Scanner::new(rules, vec![]);

        // 函式 immediately after 調用 — should fire.
        let issues = scanner.scan("調用函式").issues;
        assert_eq!(
            issues.len(),
            1,
            "should fire when 函式 is adjacent: {issues:?}"
        );

        // Gap between them — should NOT fire.
        let issues = scanner.scan("調用某個函式").issues;
        assert!(
            issues.is_empty(),
            "should not fire with gap between match and term: {issues:?}"
        );

        // 函式 immediately before 調用 — should also fire (adjacent = either side).
        let issues = scanner.scan("函式調用方式").issues;
        assert_eq!(
            issues.len(),
            1,
            "should fire when 函式 is adjacent before: {issues:?}"
        );
    }

    #[test]
    fn positional_not_before_vetoes() {
        // not_before:的 means 的 must NOT appear within 20 chars after.
        let rules = vec![SpellingRule {
            from: "項目".into(),
            to: vec!["專案".into()],
            rule_type: RuleType::CrossStrait,
            disabled: false,
            context: None,
            english: None,
            context_clues: None,
            negative_context_clues: None,
            exceptions: None,
            positional_clues: Some(vec!["not_before:的".into()]),
            tags: None,
        }];
        let scanner = Scanner::new(rules, vec![]);

        // No 的 after — should fire.
        let issues = scanner.scan("這個項目進度超前").issues;
        assert_eq!(issues.len(), 1, "should fire without veto term: {issues:?}");

        // 的 follows — should NOT fire.
        let issues = scanner.scan("項目的名稱").issues;
        assert!(
            issues.is_empty(),
            "should be vetoed by 的 after match: {issues:?}"
        );
    }

    #[test]
    fn positional_not_after_vetoes() {
        // not_after:清單 means 清單 must NOT appear within 20 chars before.
        let rules = vec![SpellingRule {
            from: "項目".into(),
            to: vec!["專案".into()],
            rule_type: RuleType::CrossStrait,
            disabled: false,
            context: None,
            english: None,
            context_clues: None,
            negative_context_clues: None,
            exceptions: None,
            positional_clues: Some(vec!["not_after:清單".into()]),
            tags: None,
        }];
        let scanner = Scanner::new(rules, vec![]);

        // 清單 absent — should fire.
        let issues = scanner.scan("這個項目進度超前").issues;
        assert_eq!(issues.len(), 1, "should fire without veto term: {issues:?}");

        // 清單 precedes — should NOT fire.
        let issues = scanner.scan("清單項目需要確認").issues;
        assert!(
            issues.is_empty(),
            "should be vetoed by 清單 before match: {issues:?}"
        );
    }

    #[test]
    fn positional_and_context_clues_both_required() {
        // Rule has both context_clues and positional_clues.  Both must pass.
        let rules = vec![SpellingRule {
            from: "調用".into(),
            to: vec!["呼叫".into()],
            rule_type: RuleType::CrossStrait,
            disabled: false,
            context: None,
            english: None,
            context_clues: Some(vec!["程式".into()]),
            negative_context_clues: None,
            exceptions: None,
            positional_clues: Some(vec!["before:函式".into()]),
            tags: None,
        }];
        let scanner = Scanner::new(rules, vec![]);

        // Both satisfied: 程式 in window AND 函式 after — should fire.
        let issues = scanner.scan("這個程式調用函式").issues;
        assert_eq!(
            issues.len(),
            1,
            "should fire when both context and positional match: {issues:?}"
        );

        // context_clues satisfied but positional NOT — should not fire.
        let issues = scanner.scan("這個程式調用方法").issues;
        assert!(
            issues.is_empty(),
            "positional fails, should not fire: {issues:?}"
        );

        // positional satisfied but context_clues NOT — should not fire.
        let issues = scanner.scan("直接調用函式").issues;
        assert!(
            issues.is_empty(),
            "context_clues fails, should not fire: {issues:?}"
        );
    }

    #[test]
    fn positional_multiple_conditions_all_must_pass() {
        // Multiple positional conditions: all must pass (AND).
        let rules = vec![SpellingRule {
            from: "調用".into(),
            to: vec!["呼叫".into()],
            rule_type: RuleType::CrossStrait,
            disabled: false,
            context: None,
            english: None,
            context_clues: None,
            negative_context_clues: None,
            exceptions: None,
            positional_clues: Some(vec!["after:請".into(), "before:函式".into()]),
            tags: None,
        }];
        let scanner = Scanner::new(rules, vec![]);

        // Both conditions met.
        let issues = scanner.scan("請調用函式").issues;
        assert_eq!(
            issues.len(),
            1,
            "both positional conditions met: {issues:?}"
        );

        // Only one condition met.
        let issues = scanner.scan("請調用方法").issues;
        assert!(
            issues.is_empty(),
            "only after: met, before: not — should not fire: {issues:?}"
        );

        let issues = scanner.scan("直接調用函式").issues;
        assert!(
            issues.is_empty(),
            "only before: met, after: not — should not fire: {issues:?}"
        );
    }

    #[test]
    fn positional_no_regression_without_positional_clues() {
        // Rules without positional_clues should behave exactly as before.
        let rules = vec![SpellingRule {
            from: "軟件".into(),
            to: vec!["軟體".into()],
            rule_type: RuleType::CrossStrait,
            disabled: false,
            context: None,
            english: None,
            context_clues: None,
            negative_context_clues: None,
            exceptions: None,
            positional_clues: None,
            tags: None,
        }];
        let scanner = Scanner::new(rules, vec![]);
        let issues = scanner.scan("這個軟件很好用").issues;
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].found, "軟件");
    }

    #[test]
    fn positional_before_stops_at_paragraph_break() {
        // before:函式 should NOT match across a paragraph boundary.
        let rules = vec![SpellingRule {
            from: "調用".into(),
            to: vec!["呼叫".into()],
            rule_type: RuleType::CrossStrait,
            disabled: false,
            context: None,
            english: None,
            context_clues: None,
            negative_context_clues: None,
            exceptions: None,
            positional_clues: Some(vec!["before:函式".into()]),
            tags: None,
        }];
        let scanner = Scanner::new(rules, vec![]);

        // 函式 is in the next paragraph — should NOT fire.
        let issues = scanner.scan("請調用方法\n\n函式定義在此").issues;
        assert!(
            issues.is_empty(),
            "before: must not match across paragraph break: {issues:?}"
        );

        // 函式 is in the same paragraph — should fire.
        let issues = scanner.scan("請調用函式").issues;
        assert_eq!(issues.len(), 1);
    }

    #[test]
    fn positional_after_stops_at_paragraph_break() {
        // after:請 should NOT match across a paragraph boundary.
        let rules = vec![SpellingRule {
            from: "調用".into(),
            to: vec!["呼叫".into()],
            rule_type: RuleType::CrossStrait,
            disabled: false,
            context: None,
            english: None,
            context_clues: None,
            negative_context_clues: None,
            exceptions: None,
            positional_clues: Some(vec!["after:請".into()]),
            tags: None,
        }];
        let scanner = Scanner::new(rules, vec![]);

        // 請 is in the previous paragraph — should NOT fire.
        let issues = scanner.scan("請看這裡\n\n調用方法").issues;
        assert!(
            issues.is_empty(),
            "after: must not match across paragraph break: {issues:?}"
        );
    }

    #[test]
    fn positional_before_stops_at_code_span() {
        // In Markdown, before:函式 should NOT match text inside a code span.
        let rules = vec![SpellingRule {
            from: "調用".into(),
            to: vec!["呼叫".into()],
            rule_type: RuleType::CrossStrait,
            disabled: false,
            context: None,
            english: None,
            context_clues: None,
            negative_context_clues: None,
            exceptions: None,
            positional_clues: Some(vec!["before:函式".into()]),
            tags: None,
        }];
        let scanner = Scanner::new(rules, vec![]);

        // 函式 is inside a code span — positional window should stop
        // at the excluded range boundary, so the clue is invisible.
        let md_text = "調用`函式`來處理";
        let issues = scanner
            .scan_for_content_type(md_text, ContentType::Markdown, Profile::Default)
            .issues;
        assert!(
            issues.is_empty(),
            "before: must not see text inside code spans: {issues:?}"
        );

        // Same text without code span — should fire.
        let plain_text = "調用函式來處理";
        let issues = scanner
            .scan_for_content_type(plain_text, ContentType::Markdown, Profile::Default)
            .issues;
        assert_eq!(
            issues.len(),
            1,
            "should fire when 函式 is not in code span: {issues:?}"
        );
    }

    #[test]
    fn positional_adjacent_excluded_region() {
        // adjacent:函式 should NOT match if 函式 is inside an excluded region.
        let rules = vec![SpellingRule {
            from: "調用".into(),
            to: vec!["呼叫".into()],
            rule_type: RuleType::CrossStrait,
            disabled: false,
            context: None,
            english: None,
            context_clues: None,
            negative_context_clues: None,
            exceptions: None,
            positional_clues: Some(vec!["adjacent:函式".into()]),
            tags: None,
        }];
        let scanner = Scanner::new(rules, vec![]);

        // 函式 inside a code span (Markdown) — adjacent should not match.
        let md_text = "調用`函式`";
        let issues = scanner
            .scan_for_content_type(md_text, ContentType::Markdown, Profile::Default)
            .issues;
        assert!(
            issues.is_empty(),
            "adjacent: must not match term inside excluded region: {issues:?}"
        );
    }

    // Remaining tests are included from the original scan.rs via include.
    // Rather than duplicating 2000+ lines inline, the tests are appended
    // by extracting from the original monolithic file.
    include!("tests_generated.rs");
}

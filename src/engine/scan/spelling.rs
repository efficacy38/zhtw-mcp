// Spelling rule scan using Aho-Corasick (charwise daachorse primary,
// bytewise fallback).  Context-clue checking uses a windowed AC scan
// over bounded slices rather than full-document pre-scan.

use std::collections::HashMap;

use crate::engine::excluded::{is_excluded, ByteRange};
use crate::engine::zhtype::ChineseType;
use crate::rules::ruleset::{Issue, IssueType, ProfileConfig, RuleType};

use super::{
    already_correct_form, clamp_at_excluded, PositionalClue, Scanner, CONTEXT_WINDOW_CHARS,
    MIN_SCAN_CLUE_MATCHES, POSITIONAL_WINDOW_CHARS,
};

// Per-rule bitflags gating optional filter stages in process_spelling_match.
// Most rules have flags == 0 (no optional stages), skipping all guarded
// blocks at near-zero cost.
pub(crate) const FILTER_HAS_SUPERSTRING: u8 = 1 << 0;
pub(crate) const FILTER_HAS_EXCEPTIONS: u8 = 1 << 1;
pub(crate) const FILTER_HAS_POS_CLUES: u8 = 1 << 2;
pub(crate) const FILTER_HAS_NEG_CLUES: u8 = 1 << 3;
pub(crate) const FILTER_HAS_POSITIONAL: u8 = 1 << 4;
pub(crate) const FILTER_IS_DELETION: u8 = 1 << 5;

const FILTER_HAS_ANY_CLUE: u8 = FILTER_HAS_POS_CLUES | FILTER_HAS_NEG_CLUES;

// Rule dispatch classes: monomorphic fast paths for common rule shapes.
// Computed once at AC build time, dispatched per-match to eliminate dead
// branches in the filter cascade (45.2 step 2).
pub(crate) const CLASS_SIMPLE: u8 = 0; // no context clues, no positional
pub(crate) const CLASS_CLUED: u8 = 1; // context clues only (pos/neg)
pub(crate) const CLASS_FULL: u8 = 2; // has positional clues (± context)

impl Scanner {
    /// Run spelling rules over `text`, appending matches to `issues`.
    ///
    /// Dispatches each AC hit to one of three monomorphic fast paths based
    /// on the precomputed rule class (45.2 step 2).  The compiler generates
    /// three separate copies of `process_match_dispatch`, each with dead
    /// clue/positional branches eliminated at compile time.
    pub(crate) fn scan_spelling(
        &self,
        text: &str,
        excluded: &[ByteRange],
        zh_type: ChineseType,
        issues: &mut Vec<Issue>,
        cfg: &ProfileConfig,
    ) {
        let mut excl_cursor: usize = 0;
        let mut boundary_cache: HashMap<usize, bool> = HashMap::new();

        // Dispatch macro: routes each AC hit to the monomorphic variant
        // matching the rule's precomputed class.  Absorption sentinels
        // (index >= rule count) are filtered before dispatch.
        macro_rules! dispatch {
            ($start:expr, $end:expr, $idx:expr) => {
                match self.rule_classes[$idx] {
                    CLASS_CLUED => self.process_match_dispatch::<CLASS_CLUED>(
                        text,
                        excluded,
                        &mut excl_cursor,
                        zh_type,
                        issues,
                        cfg,
                        &mut boundary_cache,
                        $start,
                        $end,
                        $idx,
                    ),
                    CLASS_FULL => self.process_match_dispatch::<CLASS_FULL>(
                        text,
                        excluded,
                        &mut excl_cursor,
                        zh_type,
                        issues,
                        cfg,
                        &mut boundary_cache,
                        $start,
                        $end,
                        $idx,
                    ),
                    _ => self.process_match_dispatch::<CLASS_SIMPLE>(
                        text,
                        excluded,
                        &mut excl_cursor,
                        zh_type,
                        issues,
                        cfg,
                        &mut boundary_cache,
                        $start,
                        $end,
                        $idx,
                    ),
                }
            };
        }

        if let Some(ref cw_ac) = self.spelling_ac_charwise {
            for mat in cw_ac.leftmost_find_iter(text) {
                let idx = mat.value();
                if idx >= self.spelling_rules.len() {
                    continue;
                }
                dispatch!(mat.start(), mat.end(), idx);
            }
        } else if let Some(ref bw_ac) = self.spelling_ac_bytewise {
            for mat in bw_ac.find_iter(text) {
                let idx = mat.pattern().as_usize();
                if idx >= self.spelling_rules.len() {
                    continue;
                }
                dispatch!(mat.start(), mat.end(), idx);
            }
        }
    }

    /// Monomorphic fast path for a single spelling AC match.
    ///
    /// CLASS is a compile-time constant (CLASS_SIMPLE / CLASS_CLUED / CLASS_FULL).
    /// The compiler monomorphizes three copies, dead-code-eliminating the
    /// context-clue and positional-clue blocks for simpler rule classes:
    ///   - CLASS_SIMPLE: skips context-clue window + AC scan + positional check
    ///   - CLASS_CLUED:  runs context-clue gate, skips positional check
    ///   - CLASS_FULL:   runs both gates
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn process_match_dispatch<const CLASS: u8>(
        &self,
        text: &str,
        excluded: &[ByteRange],
        excl_cursor: &mut usize,
        zh_type: ChineseType,
        issues: &mut Vec<Issue>,
        cfg: &ProfileConfig,
        boundary_cache: &mut HashMap<usize, bool>,
        start: usize,
        end: usize,
        rule_idx: usize,
    ) {
        let rule = &self.spelling_rules[rule_idx];
        let flags = self.rule_filter_flags[rule_idx];

        // Variant rules only apply in Traditional Chinese context.
        if rule.rule_type == RuleType::Variant
            && (!cfg.variant_normalization || zh_type == ChineseType::Simplified)
        {
            return;
        }

        // AI filler rules are profile-gated.
        if rule.rule_type == RuleType::AiFiller && !cfg.ai_filler_detection {
            return;
        }

        if rule.rule_type == RuleType::PoliticalColoring
            && !cfg.political_stance.allows_rule(&rule.from)
        {
            return;
        }

        // Advancing-cursor exclusion check (amortized O(1)).
        while *excl_cursor < excluded.len() && excluded[*excl_cursor].end <= start {
            *excl_cursor += 1;
        }
        if *excl_cursor < excluded.len()
            && excluded[*excl_cursor].start < end
            && start < excluded[*excl_cursor].end
        {
            return;
        }

        // Skip if surrounding text already contains a correct superstring form.
        if flags & FILTER_HAS_SUPERSTRING != 0 && already_correct_form(text, start, rule) {
            return;
        }

        // Word-boundary check: skip if a dictionary word straddles the
        // match edge (e.g. "積分" inside "累積分佈").  Memoized per offset.
        let straddles_start = *boundary_cache
            .entry(start)
            .or_insert_with(|| self.segmenter.word_straddles_boundary(text, start));
        let straddles_end = *boundary_cache
            .entry(end)
            .or_insert_with(|| self.segmenter.word_straddles_boundary(text, end));
        if straddles_start || straddles_end {
            return;
        }

        // Exception check: skip if match falls inside an exception phrase.
        if flags & FILTER_HAS_EXCEPTIONS != 0 {
            if let Some(ref exceptions) = rule.exceptions {
                let in_exception = exceptions.iter().any(|exc| {
                    for (pos, _) in exc.match_indices(&rule.from) {
                        if let Some(exc_start) = start.checked_sub(pos) {
                            let exc_end = exc_start + exc.len();
                            if text.get(exc_start..exc_end) == Some(exc.as_str()) {
                                return true;
                            }
                        }
                    }
                    false
                });
                if in_exception {
                    return;
                }
            }
        }

        // Context-clue gate (dead-code-eliminated for CLASS_SIMPLE).
        // Outer: const-evaluated (DCE for CLASS_SIMPLE).  Inner: runtime flag.
        #[allow(clippy::collapsible_if)]
        if CLASS >= CLASS_CLUED {
            if flags & FILTER_HAS_ANY_CLUE != 0 {
                let (win_start, win_end) = context_byte_window(text, start, end, excluded);
                let (pos_matches, any_neg) = scan_clues_in_window(
                    self.clue_ac.as_ref(),
                    &text[win_start..win_end],
                    self.rule_pos_clue_ids[rule_idx].as_deref(),
                    self.rule_neg_clue_ids[rule_idx].as_deref(),
                );

                if flags & FILTER_HAS_POS_CLUES != 0 && pos_matches < MIN_SCAN_CLUE_MATCHES {
                    return;
                }

                if flags & FILTER_HAS_NEG_CLUES != 0 && any_neg {
                    return;
                }
            }
        }

        // Positional clue gate (dead-code-eliminated for CLASS_SIMPLE/CLASS_CLUED).
        // Outer: const-evaluated (DCE for SIMPLE/CLUED).  Inner: runtime flag.
        #[allow(clippy::collapsible_if)]
        if CLASS >= CLASS_FULL {
            if flags & FILTER_HAS_POSITIONAL != 0 {
                if let Some(ref pos_clues) = self.rule_positional_clues[rule_idx] {
                    if !check_positional_clues(text, start, end, excluded, pos_clues) {
                        return;
                    }
                }
            }
        }

        // Deletion rules: extend span to consume trailing fullwidth punctuation.
        let end = if flags & FILTER_IS_DELETION != 0 {
            match text[end..].chars().next() {
                Some(c @ ('\u{FF0C}' | '\u{FF1A}'))
                    if !is_excluded(end, end + c.len_utf8(), excluded) =>
                {
                    end + c.len_utf8()
                }
                _ => end,
            }
        } else {
            end
        };

        let mut issue = Issue::new(
            start,
            end - start,
            &text[start..end],
            self.spelling_suggestions[rule_idx].clone(),
            IssueType::from(rule.rule_type),
            rule.rule_type.default_severity(),
        );
        issue.context.clone_from(&rule.context);
        issue.english.clone_from(&rule.english);
        issue.context_clues.clone_from(&rule.context_clues);
        issues.push(issue);
    }
}

/// Compute the byte-offset window for context-clue proximity checks,
/// clamped at paragraph breaks and excluded-range boundaries.
fn context_byte_window(
    text: &str,
    match_start: usize,
    match_end: usize,
    excluded: &[ByteRange],
) -> (usize, usize) {
    let bytes = text.as_bytes();
    let max_search = CONTEXT_WINDOW_CHARS * 4;
    let para_start = {
        let search_start = match_start.saturating_sub(max_search);
        let search = &bytes[search_start..match_start];
        find_last_paragraph_break(search).map_or(0, |pos| search_start + pos + 1)
    };
    let para_end = {
        let search_end = (match_end + max_search).min(text.len());
        let search = &bytes[match_end..search_end];
        find_first_paragraph_break(search).map_or(text.len(), |pos| match_end + pos)
    };

    let mut byte_start = match_start;
    for _ in 0..CONTEXT_WINDOW_CHARS {
        if byte_start <= para_start {
            byte_start = para_start;
            break;
        }
        byte_start = text.floor_char_boundary(byte_start - 1);
    }
    byte_start = byte_start.max(para_start);

    let mut byte_end = match_end;
    for _ in 0..CONTEXT_WINDOW_CHARS {
        if byte_end >= para_end {
            byte_end = para_end;
            break;
        }
        byte_end = text.ceil_char_boundary(byte_end + 1);
    }
    byte_end = byte_end.min(para_end);

    if excluded.is_empty() {
        return (byte_start, byte_end);
    }

    clamp_at_excluded(text, byte_start, byte_end, match_start, match_end, excluded)
}

/// Last `\n\n` (or `\r\n\r\n`) offset in `bytes`, pointing at the second `\n`.
fn find_last_paragraph_break(bytes: &[u8]) -> Option<usize> {
    // Scan backward for \n\n.
    let len = bytes.len();
    if len < 2 {
        return None;
    }
    let mut i = len - 1;
    while i > 0 {
        if bytes[i] == b'\n' && bytes[i - 1] == b'\n' {
            return Some(i);
        }
        // Handle \r\n\r\n: bytes[i]=\n, bytes[i-1]=\r, bytes[i-2]=\n
        if i >= 2 && bytes[i] == b'\n' && bytes[i - 1] == b'\r' && bytes[i - 2] == b'\n' {
            return Some(i);
        }
        i -= 1;
    }
    None
}

/// First `\n\n` (or `\n\r\n`) offset in `bytes`, pointing at the first `\n`.
fn find_first_paragraph_break(bytes: &[u8]) -> Option<usize> {
    let len = bytes.len();
    if len < 2 {
        return None;
    }
    for i in 0..len - 1 {
        if bytes[i] == b'\n' && bytes[i + 1] == b'\n' {
            return Some(i);
        }
        // \n\r\n also counts.
        if i + 2 < len && bytes[i] == b'\n' && bytes[i + 1] == b'\r' && bytes[i + 2] == b'\n' {
            return Some(i);
        }
    }
    None
}

/// Scan a bounded text window for positive and negative clue IDs in one pass.
/// Returns `(positive_matches, any_negative_match)`.
fn scan_clues_in_window(
    clue_ac: Option<&aho_corasick::AhoCorasick>,
    window: &str,
    pos_ids: Option<&[u16]>,
    neg_ids: Option<&[u16]>,
) -> (usize, bool) {
    let Some(clue_ac) = clue_ac else {
        return (0, false);
    };
    if window.is_empty() || (pos_ids.is_none() && neg_ids.is_none()) {
        return (0, false);
    }

    let mut pos_seen = [false; 32];
    let mut pos_found = 0usize;

    for mat in clue_ac.find_overlapping_iter(window) {
        let clue_id = mat.pattern().as_usize() as u16;

        if let Some(pos_ids) = pos_ids {
            if let Some(pos) = pos_ids.iter().position(|&id| id == clue_id) {
                if !pos_seen[pos] {
                    pos_seen[pos] = true;
                    pos_found += 1;
                }
            }
        }

        if let Some(neg_ids) = neg_ids {
            if neg_ids.contains(&clue_id) {
                return (pos_found, true);
            }
        }

        if pos_ids.is_none_or(|ids| pos_found >= ids.len()) && neg_ids.is_none() {
            break;
        }
    }

    (pos_found, false)
}

/// Check all positional clues for a match at [start, end).
/// Positive clues use AND semantics; any negative clue vetoes.
fn check_positional_clues(
    text: &str,
    start: usize,
    end: usize,
    excluded: &[ByteRange],
    clues: &[PositionalClue],
) -> bool {
    let mut after_win: Option<(usize, usize)> = None;
    let mut before_win: Option<(usize, usize)> = None;

    for clue in clues {
        match clue {
            PositionalClue::Before(term) => {
                let (ws, we) =
                    *after_win.get_or_insert_with(|| positional_bounds_after(text, end, excluded));
                if !text[ws..we].contains(term.as_str()) {
                    return false;
                }
            }
            PositionalClue::After(term) => {
                let (ws, we) = *before_win
                    .get_or_insert_with(|| positional_bounds_before(text, start, excluded));
                if !text[ws..we].contains(term.as_str()) {
                    return false;
                }
            }
            PositionalClue::Adjacent(term) => {
                // Immediately before: term ends right at match start.
                let before_ok = start >= term.len()
                    && text.get(start - term.len()..start) == Some(term.as_str())
                    && !is_excluded(start - term.len(), start, excluded);
                // Immediately after: term starts right at match end.
                let after_ok = text.get(end..end + term.len()) == Some(term.as_str())
                    && !is_excluded(end, end + term.len(), excluded);
                if !before_ok && !after_ok {
                    return false;
                }
            }
            PositionalClue::NotBefore(term) => {
                let (ws, we) =
                    *after_win.get_or_insert_with(|| positional_bounds_after(text, end, excluded));
                if text[ws..we].contains(term.as_str()) {
                    return false;
                }
            }
            PositionalClue::NotAfter(term) => {
                let (ws, we) = *before_win
                    .get_or_insert_with(|| positional_bounds_before(text, start, excluded));
                if text[ws..we].contains(term.as_str()) {
                    return false;
                }
            }
        }
    }
    true
}

/// Positional window AFTER the match, clamped at paragraph/excluded boundaries.
fn positional_bounds_after(text: &str, match_end: usize, excluded: &[ByteRange]) -> (usize, usize) {
    if match_end >= text.len() {
        return (text.len(), text.len());
    }
    let bytes = text.as_bytes();
    let max_search = POSITIONAL_WINDOW_CHARS * 4;
    let search_end = (match_end + max_search).min(text.len());
    let para_end = {
        let search = &bytes[match_end..search_end];
        find_first_paragraph_break(search).map_or(text.len(), |pos| match_end + pos)
    };

    let mut byte_end = match_end;
    for _ in 0..POSITIONAL_WINDOW_CHARS {
        if byte_end >= para_end {
            byte_end = para_end;
            break;
        }
        byte_end = text.ceil_char_boundary(byte_end + 1);
    }
    byte_end = byte_end.min(para_end);

    if !excluded.is_empty() {
        let right_idx = excluded.partition_point(|r| r.start < match_end);
        for excl in &excluded[right_idx..] {
            if excl.start >= byte_end {
                break;
            }
            if excl.start >= match_end && excl.start < byte_end {
                byte_end = excl.start;
            }
        }
    }

    let byte_end = text.floor_char_boundary(byte_end.min(text.len()));
    if match_end > byte_end {
        return (match_end, match_end);
    }
    (match_end, byte_end)
}

/// Positional window BEFORE the match, clamped at paragraph/excluded boundaries.
fn positional_bounds_before(
    text: &str,
    match_start: usize,
    excluded: &[ByteRange],
) -> (usize, usize) {
    if match_start == 0 {
        return (0, 0);
    }
    let bytes = text.as_bytes();
    let max_search = POSITIONAL_WINDOW_CHARS * 4;
    let search_start = match_start.saturating_sub(max_search);
    let para_start = {
        let search = &bytes[search_start..match_start];
        find_last_paragraph_break(search).map_or(0, |pos| search_start + pos + 1)
    };

    let mut byte_start = match_start;
    for _ in 0..POSITIONAL_WINDOW_CHARS {
        if byte_start <= para_start {
            byte_start = para_start;
            break;
        }
        byte_start = text.floor_char_boundary(byte_start - 1);
    }
    byte_start = byte_start.max(para_start);

    if !excluded.is_empty() {
        let left_idx = excluded.partition_point(|r| r.start < match_start);
        for excl in excluded[..left_idx].iter().rev() {
            if excl.end <= byte_start {
                break;
            }
            if excl.end <= match_start && excl.end > byte_start {
                byte_start = excl.end;
            }
        }
    }

    let byte_start = text.ceil_char_boundary(byte_start);
    if byte_start > match_start {
        return (match_start, match_start);
    }
    (byte_start, match_start)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scanner() -> Scanner {
        use crate::rules::loader::load_embedded_ruleset;
        let rs = load_embedded_ruleset().expect("load embedded ruleset");
        Scanner::new(rs.spelling_rules, rs.case_rules)
    }

    #[test]
    fn filter_flags_match_rule_properties() {
        // Derive expected flags from raw rule fields only, NOT from scanner
        // caches (which share the same normalization pipeline as the flags).
        let scanner = make_scanner();
        for (i, rule) in scanner.spelling_rules.iter().enumerate() {
            let f = scanner.rule_filter_flags[i];
            assert_eq!(
                f & FILTER_HAS_SUPERSTRING != 0,
                rule.to.iter().any(|t| t.contains(&rule.from)),
                "rule '{}': SUPERSTRING mismatch",
                rule.from
            );
            assert_eq!(
                f & FILTER_HAS_EXCEPTIONS != 0,
                rule.exceptions.as_ref().is_some_and(|v| !v.is_empty()),
                "rule '{}': EXCEPTIONS mismatch",
                rule.from
            );
            assert_eq!(
                f & FILTER_HAS_POS_CLUES != 0,
                rule.context_clues.as_ref().is_some_and(|v| !v.is_empty()),
                "rule '{}': POS_CLUES mismatch",
                rule.from
            );
            assert_eq!(
                f & FILTER_HAS_NEG_CLUES != 0,
                rule.negative_context_clues
                    .as_ref()
                    .is_some_and(|v| !v.is_empty()),
                "rule '{}': NEG_CLUES mismatch",
                rule.from
            );
            assert_eq!(
                f & FILTER_HAS_POSITIONAL != 0,
                rule.positional_clues
                    .as_ref()
                    .is_some_and(|v| v.iter().any(|s| PositionalClue::parse(s).is_some())),
                "rule '{}': POSITIONAL mismatch",
                rule.from
            );
            assert_eq!(
                f & FILTER_IS_DELETION != 0,
                rule.is_deletion_rule(),
                "rule '{}': IS_DELETION mismatch",
                rule.from
            );
        }
    }

    #[test]
    fn filter_vecs_aligned() {
        let scanner = make_scanner();
        let n = scanner.spelling_rules.len();
        assert_eq!(scanner.rule_filter_flags.len(), n);
        assert_eq!(scanner.rule_classes.len(), n);
        assert_eq!(scanner.rule_pos_clue_ids.len(), n);
        assert_eq!(scanner.rule_neg_clue_ids.len(), n);
        assert_eq!(scanner.rule_positional_clues.len(), n);
        assert_eq!(scanner.spelling_suggestions.len(), n);
    }

    #[test]
    fn rule_classes_match_filter_flags() {
        let scanner = make_scanner();
        for (i, &f) in scanner.rule_filter_flags.iter().enumerate() {
            let has_clues = f & (FILTER_HAS_POS_CLUES | FILTER_HAS_NEG_CLUES) != 0;
            let has_positional = f & FILTER_HAS_POSITIONAL != 0;
            let expected = if has_positional {
                CLASS_FULL
            } else if has_clues {
                CLASS_CLUED
            } else {
                CLASS_SIMPLE
            };
            assert_eq!(
                scanner.rule_classes[i], expected,
                "rule '{}': class mismatch (flags=0x{:02x})",
                scanner.spelling_rules[i].from, f
            );
        }
    }

    #[test]
    fn rule_class_distribution() {
        // Sanity check: majority of rules should be CLASS_SIMPLE (the 79%
        // from PR #49 analysis).  At least 60% to guard against drift.
        let scanner = make_scanner();
        let total = scanner.rule_classes.len();
        let simple = scanner
            .rule_classes
            .iter()
            .filter(|&&c| c == CLASS_SIMPLE)
            .count();
        let clued = scanner
            .rule_classes
            .iter()
            .filter(|&&c| c == CLASS_CLUED)
            .count();
        let full = scanner
            .rule_classes
            .iter()
            .filter(|&&c| c == CLASS_FULL)
            .count();
        assert_eq!(simple + clued + full, total);
        assert!(
            simple * 100 / total >= 60,
            "expected >= 60% simple rules, got {simple}/{total} ({:.0}%)",
            simple as f64 / total as f64 * 100.0
        );
        eprintln!(
            "rule class distribution: simple={simple} ({:.0}%), clued={clued} ({:.0}%), full={full} ({:.0}%)",
            simple as f64 / total as f64 * 100.0,
            clued as f64 / total as f64 * 100.0,
            full as f64 / total as f64 * 100.0,
        );
    }
}

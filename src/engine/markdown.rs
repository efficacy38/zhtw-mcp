// Markdown-aware text extraction for zhtw-mcp linting.
//
// Uses pulldown-cmark to parse Markdown and identify regions that should be
// excluded from linting: code blocks, inline code, HTML blocks, and YAML
// frontmatter. Returns byte ranges to exclude before scanning.

use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

use super::excluded::{merge_ranges_pub, ByteRange};

/// Build excluded byte ranges from Markdown structure.
///
/// Excludes: fenced/indented code blocks, inline code, HTML blocks/tags,
/// and YAML frontmatter (leading --- fences).
///
/// The returned ranges are sorted by start position and non-overlapping.
pub fn build_markdown_excluded_ranges(text: &str) -> Vec<ByteRange> {
    let mut ranges = Vec::new();

    // Pre-pass: detect YAML frontmatter (leading --- fence).  Exclude only
    // the structural tokens (--- fences, key+colon spans), leaving value
    // prose scannable so that linting catches issues in title/description.
    if let Some(fm_end) = detect_frontmatter(text) {
        collect_frontmatter_structural_ranges(text, fm_end, &mut ranges);
    }

    collect_container_fence_ranges(text, &mut ranges);

    let opts = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH;
    let parser = Parser::new_ext(text, opts);
    let mut in_code_block = false;
    let mut code_block_start = 0usize;

    for (event, range) in parser.into_offset_iter() {
        match event {
            // Fenced or indented code blocks: exclude entire block.
            Event::Start(Tag::CodeBlock(_)) => {
                in_code_block = true;
                code_block_start = range.start;
            }
            Event::End(TagEnd::CodeBlock) if in_code_block => {
                ranges.push(ByteRange {
                    start: code_block_start,
                    end: range.end,
                });
                in_code_block = false;
            }

            // Inline code: exclude the span including backticks.
            Event::Code(_) | Event::Html(_) | Event::InlineHtml(_) => {
                ranges.push(ByteRange {
                    start: range.start,
                    end: range.end,
                });
            }

            _ => {}
        }
    }

    // Sort and merge (frontmatter + parser ranges may overlap).
    merge_ranges_pub(ranges)
}

/// Like [build_markdown_excluded_ranges], but fenced/indented code blocks are
/// NOT excluded.  Only inline code (`backtick`), HTML, and YAML frontmatter
/// are excluded.  This allows linting Chinese prose inside code blocks
/// (comments, translated output, etc.) while still protecting inline code
/// and HTML from false positives.
pub fn build_markdown_excluded_ranges_no_code(text: &str) -> Vec<ByteRange> {
    let mut ranges = Vec::new();

    // Pre-pass: detect YAML frontmatter — exclude structural tokens only.
    if let Some(fm_end) = detect_frontmatter(text) {
        collect_frontmatter_structural_ranges(text, fm_end, &mut ranges);
    }

    collect_container_fence_ranges(text, &mut ranges);

    let opts = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH;
    let parser = Parser::new_ext(text, opts);

    for (event, range) in parser.into_offset_iter() {
        match event {
            // Skip code blocks entirely — let them be scanned.
            Event::Start(Tag::CodeBlock(_)) | Event::End(TagEnd::CodeBlock) => {}
            // Inline code and HTML: still exclude.
            Event::Code(_) | Event::Html(_) | Event::InlineHtml(_) => {
                ranges.push(ByteRange {
                    start: range.start,
                    end: range.end,
                });
            }
            _ => {}
        }
    }

    merge_ranges_pub(ranges)
}

/// Build excluded byte ranges for YAML structural tokens.
///
/// Excludes YAML key tokens (the key name + colon) so that bare ASCII colons
/// in key-value separators do not trigger false-positive colon warnings.
/// Only the key portion is excluded; values after the colon are prose and
/// remain scannable.
///
/// Pattern matched: /^\s*\w[\w-]*\s*:/ on each line.
/// Examples excluded: title:, key-name:, summary  :.
/// Not excluded: list items (- value), comments (# text), values.
pub fn build_yaml_excluded_ranges(text: &str) -> Vec<ByteRange> {
    let mut ranges = Vec::new();
    let mut pos = 0usize;

    for raw_line in text.split('\n') {
        let line_len = raw_line.len();

        if let Some(colon_pos) = yaml_key_colon_pos(raw_line) {
            // Exclude from the start of the line through the ':' (inclusive).
            ranges.push(ByteRange {
                start: pos,
                end: pos + colon_pos + 1,
            });
        }

        pos += line_len + 1; // +1 for the '\n'
    }

    ranges
}

/// Find the byte offset of the YAML key-separator colon in a line, if present.
///
/// Per the YAML spec, a block-mapping key separator is a : followed by a
/// space, tab, or end-of-line.  This handles all common block-mapping forms:
///
/// - key: value           — simple key
/// - key-name: value      — hyphenated key
/// -   indented: value    — indented key
/// - - key: value         — key inside a list item
/// - "quoted-key": value  — quoted key (quote skipped; no escape handling)
///
/// Known limitations (acceptable for prose documentation YAML):
/// - Flow mappings without whitespace after : (e.g. {key:"val"}) are not
///   detected; those are rare in documentation YAML and equivalent to JSON.
/// - Only the first key-colon per line is excluded; additional key-colons in
///   flow sequences ({a: 1, b: 2}) on the same line are not excluded.
/// - Escaped quotes inside quoted keys (e.g. "key\"name": v) may confuse
///   the quote-tracking state.
///
/// Returns the byte offset of the : within the line, or None.
/// Colons inside single- or double-quoted strings are skipped.
fn yaml_key_colon_pos(line: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut in_quote = false;
    let mut quote_char = b'\0';

    while i < len {
        let b = bytes[i];
        if in_quote {
            if b == quote_char {
                in_quote = false;
            }
        } else if b == b'"' || b == b'\'' {
            in_quote = true;
            quote_char = b;
        } else if b == b':' {
            // YAML key separator: colon must be followed by whitespace or be at EOL.
            let next = bytes.get(i + 1).copied().unwrap_or(b' ');
            if next == b' ' || next == b'\t' || next == b'\r' {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// A Markdown table cell span with `(row, column)` coordinates.
#[derive(Debug, Clone, Copy)]
pub struct TableCellSpan {
    pub start: usize,
    pub end: usize,
    pub row: usize,
    pub col: usize,
}

/// Extract byte ranges of all Markdown table cells with their (row, column).
///
/// Row 0 is the header row (or row 1 if no separator).  Column index is the
/// 0-based position within the row.  Returns empty if the document has no
/// tables.
pub fn extract_table_cell_spans(text: &str) -> Vec<TableCellSpan> {
    let opts = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH;
    let parser = Parser::new_ext(text, opts);
    let mut spans = Vec::new();
    let mut in_table = false;
    let mut current_row = 0usize;
    let mut current_col = 0usize;
    let mut cell_active = false;
    let mut cell_start = 0usize;

    for (event, range) in parser.into_offset_iter() {
        match event {
            Event::Start(Tag::Table(_)) => {
                in_table = true;
                current_row = 0;
            }
            Event::End(TagEnd::Table) => {
                in_table = false;
            }
            Event::Start(Tag::TableHead) | Event::Start(Tag::TableRow) if in_table => {
                current_col = 0;
            }
            Event::End(TagEnd::TableHead) | Event::End(TagEnd::TableRow) if in_table => {
                current_row += 1;
            }
            Event::Start(Tag::TableCell) if in_table => {
                cell_active = true;
                cell_start = range.start;
            }
            Event::End(TagEnd::TableCell) if in_table && cell_active => {
                spans.push(TableCellSpan {
                    start: cell_start,
                    end: range.end,
                    row: current_row,
                    col: current_col,
                });
                current_col += 1;
                cell_active = false;
            }
            _ => {}
        }
    }
    spans
}

/// Extract byte ranges of heading text inline content in Markdown.
///
/// Returns ranges covering only the inline text inside heading containers
/// (H1-H6), excluding the leading `#` markers and trailing whitespace.
/// Used by the scan pipeline to apply severity boosting on heading text.
///
/// Issues outside the inline text (e.g. inside ATX `#` markers) are not
/// boosted, since they are unlikely to represent real heading prose hits.
///
/// YAML frontmatter is excluded: pulldown-cmark would otherwise interpret
/// the closing `---` as a setext H2 underline, falsely treating frontmatter
/// values as heading text.
pub fn extract_heading_ranges(text: &str) -> Vec<ByteRange> {
    let frontmatter_end = detect_frontmatter(text).unwrap_or(0);

    let opts = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH;
    let parser = Parser::new_ext(text, opts);
    let mut ranges = Vec::new();
    let mut in_heading = false;
    let mut current_inline_start: Option<usize> = None;
    let mut current_inline_end: Option<usize> = None;

    for (event, range) in parser.into_offset_iter() {
        match event {
            Event::Start(Tag::Heading { .. }) => {
                in_heading = true;
                current_inline_start = None;
                current_inline_end = None;
            }
            Event::End(TagEnd::Heading(_)) if in_heading => {
                if let (Some(start), Some(end)) = (current_inline_start, current_inline_end) {
                    // Skip false-positive headings synthesised from
                    // frontmatter content + closing `---`.
                    if start >= frontmatter_end {
                        ranges.push(ByteRange { start, end });
                    }
                }
                in_heading = false;
            }
            Event::Text(_) | Event::Code(_) | Event::Html(_) | Event::InlineHtml(_)
                if in_heading =>
            {
                if current_inline_start.is_none() {
                    current_inline_start = Some(range.start);
                }
                current_inline_end = Some(range.end);
            }
            _ => {}
        }
    }
    ranges
}

/// Collect YAML frontmatter structural ranges: opening `---` line, closing
/// `---` line, and per-line key+colon spans.  Values remain scannable.
fn collect_frontmatter_structural_ranges(text: &str, fm_end: usize, ranges: &mut Vec<ByteRange>) {
    let fm = &text[..fm_end];
    let mut pos = 0usize;

    for raw_line in fm.split('\n') {
        let line_len = raw_line.len();
        let trimmed = raw_line.trim_end_matches('\r');

        if trimmed == "---" {
            // Exclude the entire fence line (and the trailing \n if present).
            let end = (pos + line_len + 1).min(fm_end);
            ranges.push(ByteRange { start: pos, end });
        } else if let Some(colon_pos) = yaml_key_colon_pos(raw_line) {
            // Exclude the key+colon prefix; value text remains scannable.
            ranges.push(ByteRange {
                start: pos,
                end: pos + colon_pos + 1,
            });
        }

        pos += line_len + 1; // +1 for the '\n'
        if pos >= fm_end {
            break;
        }
    }
}

/// Collect container-block fence lines (:::keyword / :::) as excluded ranges.
/// Used by HackMD and Docusaurus for admonitions.  Only the fence lines
/// themselves are excluded; the prose content between them is still scanned.
fn collect_container_fence_ranges(text: &str, ranges: &mut Vec<ByteRange>) {
    let mut pos = 0usize;
    for raw_line in text.split('\n') {
        let line_len = raw_line.len();
        let trimmed = raw_line.trim_start_matches([' ', '\t']);
        if trimmed.starts_with(":::") {
            ranges.push(ByteRange {
                start: pos,
                end: pos + line_len,
            });
        }
        pos += line_len + 1; // +1 for the '\n'
    }
}

/// Detect YAML frontmatter delimited by --- at the start of the document.
/// Returns the byte offset just past the closing ---\n (or end of closing ---).
fn detect_frontmatter(text: &str) -> Option<usize> {
    // Must start at the very beginning with --- followed by a newline.
    if !text.starts_with("---") {
        return None;
    }

    let after_open = if text.starts_with("---\n") {
        4
    } else if text.starts_with("---\r\n") {
        5
    } else {
        return None;
    };

    // Find the closing --- on its own line.
    let rest = &text[after_open..];
    for (line_start, line) in rest.split('\n').scan(0usize, |pos, line| {
        let start = *pos;
        *pos += line.len() + 1; // +1 for the \n
        Some((start, line))
    }) {
        let trimmed = line.trim_end_matches('\r');
        if trimmed == "---" {
            // End position is after the closing ---\n.
            let abs_end = after_open + line_start + line.len() + 1;
            return Some(abs_end.min(text.len()));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fenced_code_block_excluded() {
        let md = "前言\n```rust\nlet x = 1;\n```\n後語\n";
        let ranges = build_markdown_excluded_ranges(md);
        assert!(!ranges.is_empty());
        let excluded_text: String = ranges
            .iter()
            .map(|r| &md[r.start..r.end])
            .collect::<Vec<_>>()
            .join("");
        assert!(excluded_text.contains("let x = 1;"));
        assert!(!excluded_text.contains("前言"));
        assert!(!excluded_text.contains("後語"));
    }

    #[test]
    fn inline_code_excluded() {
        let md = "使用 `println!` 來輸出\n";
        let ranges = build_markdown_excluded_ranges(md);
        assert!(!ranges.is_empty());
        let any_covers_println = ranges
            .iter()
            .any(|r| md[r.start..r.end].contains("println"));
        assert!(any_covers_println);
    }

    #[test]
    fn yaml_frontmatter_keys_excluded_values_scannable() {
        let md = "---\ntitle: 測試\ndate: 2024-01-01\n---\n正文開始\n";
        let ranges = build_markdown_excluded_ranges(md);
        assert!(!ranges.is_empty());
        // Key+colon is excluded.
        let any_covers_title_key = ranges.iter().any(|r| md[r.start..r.end].contains("title:"));
        assert!(any_covers_title_key, "title: key should be excluded");
        // Value text is NOT excluded — it's prose to be scanned.
        let value_excluded = ranges.iter().any(|r| md[r.start..r.end].contains("測試"));
        assert!(
            !value_excluded,
            "frontmatter value 測試 should be scannable"
        );
        // Body is not excluded.
        let body_excluded = ranges
            .iter()
            .any(|r| md[r.start..r.end].contains("正文開始"));
        assert!(!body_excluded);
        // Closing --- fence is excluded.
        let close_excluded = ranges.iter().any(|r| md[r.start..r.end].trim() == "---");
        assert!(close_excluded, "closing --- fence should be excluded");
    }

    #[test]
    fn extract_heading_ranges_skips_frontmatter_setext() {
        // pulldown-cmark synthesises a setext H2 from frontmatter content +
        // closing `---`.  We must skip that false-positive heading so the
        // severity boost does not apply to frontmatter values.
        let md = "---\ntitle: 軟件測試指南\ndate: 2026\n---\n# 真標題\n正文。\n";
        let ranges = extract_heading_ranges(md);
        // Should contain the real heading "真標題" but NOT the synthesised
        // frontmatter heading.
        for r in &ranges {
            let span = &md[r.start..r.end];
            assert!(
                !span.contains("title:"),
                "frontmatter content leaked into heading ranges: {span:?}"
            );
        }
        let real_heading_present = ranges.iter().any(|r| md[r.start..r.end].contains("真標題"));
        assert!(
            real_heading_present,
            "real heading '真標題' should still be detected: {ranges:?}"
        );
    }

    #[test]
    fn yaml_frontmatter_not_in_middle() {
        let md = "前言\n---\ntitle: 測試\n---\n後語\n";
        let ranges = build_markdown_excluded_ranges(md);
        // --- in the middle is not frontmatter (it's a thematic break).
        let any_covers_title = ranges.iter().any(|r| md[r.start..r.end].contains("title:"));
        assert!(!any_covers_title);
    }

    #[test]
    fn html_block_excluded() {
        let md = "前言\n<div>some html</div>\n後語\n";
        let ranges = build_markdown_excluded_ranges(md);
        let any_covers_html = ranges.iter().any(|r| md[r.start..r.end].contains("<div>"));
        assert!(any_covers_html);
    }

    #[test]
    fn nested_list_with_code() {
        let md = "- 項目一\n  - `code` 子項目\n- 項目二\n";
        let ranges = build_markdown_excluded_ranges(md);
        let any_covers_code = ranges.iter().any(|r| md[r.start..r.end].contains("code"));
        assert!(any_covers_code);
        // List text should not be excluded.
        let any_covers_item = ranges.iter().any(|r| md[r.start..r.end].contains("項目一"));
        assert!(!any_covers_item);
    }

    #[test]
    fn blockquote_with_code() {
        let md = "> 引用文字 `inline` 繼續\n";
        let ranges = build_markdown_excluded_ranges(md);
        let any_covers_inline = ranges.iter().any(|r| md[r.start..r.end].contains("inline"));
        assert!(any_covers_inline);
        let any_covers_quote = ranges
            .iter()
            .any(|r| md[r.start..r.end].contains("引用文字"));
        assert!(!any_covers_quote);
    }

    #[test]
    fn empty_input() {
        let ranges = build_markdown_excluded_ranges("");
        assert!(ranges.is_empty());
    }

    #[test]
    fn plain_text_no_exclusions() {
        let md = "這是純文字，沒有任何 Markdown 語法。\n";
        let ranges = build_markdown_excluded_ranges(md);
        assert!(ranges.is_empty());
    }

    #[test]
    fn code_block_with_language_tag() {
        let md = "```python\nprint('hello')\n```\n";
        let ranges = build_markdown_excluded_ranges(md);
        assert!(!ranges.is_empty());
        let excluded: String = ranges
            .iter()
            .map(|r| &md[r.start..r.end])
            .collect::<Vec<_>>()
            .join("");
        assert!(excluded.contains("print('hello')"));
    }

    #[test]
    fn multiple_code_blocks() {
        let md = "文字\n```\nblock1\n```\n中間\n```\nblock2\n```\n結尾\n";
        let ranges = build_markdown_excluded_ranges(md);
        assert!(ranges.len() >= 2);
    }

    #[test]
    fn container_fence_lines_excluded() {
        // The :::warning and ::: fence lines must be excluded.
        // The prose content between them must NOT be excluded.
        let md = "前言\n:::warning\n這是警告內容，請注意：細節。\n:::\n後語\n";
        let ranges = build_markdown_excluded_ranges(md);
        let any_covers_open_fence = ranges
            .iter()
            .any(|r| md[r.start..r.end].contains(":::warning"));
        assert!(
            any_covers_open_fence,
            "opening fence line should be excluded"
        );
        let any_covers_close_fence = ranges.iter().any(|r| {
            let s = &md[r.start..r.end];
            s.trim() == ":::"
        });
        assert!(
            any_covers_close_fence,
            "closing fence line should be excluded"
        );
        // Prose content between fences must remain scannable.
        let prose_excluded = ranges
            .iter()
            .any(|r| md[r.start..r.end].contains("警告內容"));
        assert!(
            !prose_excluded,
            "prose content between fences must not be excluded"
        );
    }

    #[test]
    fn container_fence_four_colons() {
        // :::: (4-colon) fences must also be excluded.
        let md = "文字\n::::note\n注意事項\n::::\n後語\n";
        let ranges = build_markdown_excluded_ranges(md);
        let any_covers_open = ranges
            .iter()
            .any(|r| md[r.start..r.end].contains("::::note"));
        assert!(any_covers_open, "4-colon opening fence should be excluded");
    }

    // --- Tests for build_yaml_excluded_ranges ---

    #[test]
    fn yaml_key_colon_excluded() {
        let yaml = "title: 繁體中文文件\nsummary: 說明文字\n";
        let ranges = build_yaml_excluded_ranges(yaml);
        // "title:" should be excluded.
        let any_covers_title_colon = ranges
            .iter()
            .any(|r| yaml[r.start..r.end].contains("title:"));
        assert!(any_covers_title_colon, "YAML key colon must be excluded");
        // The value "繁體中文文件" must NOT be excluded.
        let value_excluded = ranges
            .iter()
            .any(|r| yaml[r.start..r.end].contains("繁體中文文件"));
        assert!(!value_excluded, "YAML value must remain scannable");
    }

    #[test]
    fn yaml_key_with_spaces_before_colon() {
        let yaml = "title  : 文字\n";
        let ranges = build_yaml_excluded_ranges(yaml);
        let covers_key = ranges
            .iter()
            .any(|r| yaml[r.start..r.end].contains("title  :"));
        assert!(covers_key, "key with spaces before colon must be excluded");
    }

    #[test]
    fn yaml_pure_list_items_not_excluded() {
        // Pure list items with no key-value colon are not excluded.
        let yaml = "- 項目一\n- 項目二\n";
        let ranges = build_yaml_excluded_ranges(yaml);
        assert!(
            ranges.is_empty(),
            "pure list items (no colon) must not be excluded"
        );
    }

    #[test]
    fn yaml_list_mapping_key_excluded() {
        // - key: value — the key colon inside a list item must be excluded.
        let yaml = "- name: 測試\n- label: 標籤\n";
        let ranges = build_yaml_excluded_ranges(yaml);
        // Each list mapping line should have one excluded range covering "- name:" / "- label:".
        let covers_name = ranges
            .iter()
            .any(|r| yaml[r.start..r.end].contains("name:"));
        assert!(covers_name, "key colon inside list item must be excluded");
        // Values must remain scannable.
        let value_excluded = ranges.iter().any(|r| yaml[r.start..r.end].contains("測試"));
        assert!(!value_excluded, "list mapping value must remain scannable");
    }

    #[test]
    fn yaml_hyphenated_key_excluded() {
        let yaml = "key-name: 值\n";
        let ranges = build_yaml_excluded_ranges(yaml);
        let covers = ranges
            .iter()
            .any(|r| yaml[r.start..r.end].contains("key-name:"));
        assert!(covers, "hyphenated key must be excluded");
    }

    #[test]
    fn yaml_indented_key_excluded() {
        let yaml = "outer:\n  inner: 值\n";
        let ranges = build_yaml_excluded_ranges(yaml);
        // "outer:" at col 0 and "  inner:" (indented) both excluded.
        let covers_outer = ranges
            .iter()
            .any(|r| yaml[r.start..r.end].contains("outer:"));
        let covers_inner = ranges
            .iter()
            .any(|r| yaml[r.start..r.end].contains("inner:"));
        assert!(covers_outer, "top-level key must be excluded");
        assert!(covers_inner, "indented key must be excluded");
    }
}

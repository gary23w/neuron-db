//! Streaming primitives for piping an app's output into a neuron-db scope. The transports
//! (`capture` from stdin, `run` from a child's stdout, `follow` from a growing logfile) all differ
//! only in their byte source; the shared piece is splitting that byte stream into lines without
//! losing or mangling a partial line that straddles two reads.

/// Splits a raw byte stream into lines. Buffers a partial line across `feed` calls and emits it on
/// the next newline or at `flush` (EOF). A line longer than `max` bytes is force-emitted so a binary
/// stream with no newlines can't grow the buffer without bound. A trailing `\r` (CRLF) is dropped.
pub struct LineSplitter {
    buf: Vec<u8>,
    max: usize,
}

impl LineSplitter {
    pub fn new(max: usize) -> Self { LineSplitter { buf: Vec::new(), max: max.max(1) } }

    /// Feed a chunk of bytes; `emit` is called once per complete line, without the trailing newline.
    pub fn feed(&mut self, bytes: &[u8], emit: &mut dyn FnMut(&[u8])) {
        for &b in bytes {
            if b == b'\n' {
                let end = if self.buf.last() == Some(&b'\r') { self.buf.len() - 1 } else { self.buf.len() };
                emit(&self.buf[..end]);
                self.buf.clear();
            } else {
                self.buf.push(b);
                if self.buf.len() >= self.max { emit(&self.buf); self.buf.clear(); }
            }
        }
    }

    /// Emit any buffered partial line. Call once at EOF so a final unterminated line isn't lost.
    pub fn flush(&mut self, emit: &mut dyn FnMut(&[u8])) {
        if !self.buf.is_empty() { emit(&self.buf); self.buf.clear(); }
    }
}

#[cfg(test)]
mod tests {
    use super::LineSplitter;
    fn collect(chunks: &[&[u8]], max: usize, flush: bool) -> Vec<String> {
        let mut s = LineSplitter::new(max);
        let mut out = Vec::new();
        for c in chunks { s.feed(c, &mut |l| out.push(String::from_utf8_lossy(l).into_owned())); }
        if flush { s.flush(&mut |l| out.push(String::from_utf8_lossy(l).into_owned())); }
        out
    }

    #[test]
    fn splits_on_newline_and_strips_cr() {
        assert_eq!(collect(&[b"a\r\nbb\nccc\n"], 4096, true), vec!["a", "bb", "ccc"]);
    }
    #[test]
    fn buffers_partial_line_across_feeds() {
        // a line split across two reads must be reassembled, not emitted twice
        assert_eq!(collect(&[b"hel", b"lo\nwor", b"ld\n"], 4096, true), vec!["hello", "world"]);
    }
    #[test]
    fn flush_emits_unterminated_tail() {
        assert_eq!(collect(&[b"no newline here"], 4096, true), vec!["no newline here"]);
        assert_eq!(collect(&[b"dropped without flush"], 4096, false), Vec::<String>::new());
    }
    #[test]
    fn force_emits_overlong_line() {
        // 10 bytes, no newline, cap 4 -> emitted in <=4-byte pieces, nothing buffered forever
        let out = collect(&[b"0123456789"], 4, true);
        assert!(out.join("").contains("0123456789"));
        assert!(out.iter().all(|p| p.len() <= 4));
    }
}

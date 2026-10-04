// Stop-string handling for streamed text: finds the earliest stop string and
// withholds just enough trailing text that a stop string split across chunks
// can't leak out before it completes.

pub struct StopFilter {
    stops: Vec<String>,
    buf: String,
}

impl StopFilter {
    pub fn new(stops: Vec<String>) -> Self {
        StopFilter {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            buf: String::new(),
        }
    }

    // Feeds new text; returns (text that is safe to emit, whether a stop
    // string was hit). On a hit the stop string and everything after it are
    // discarded.
    pub fn push(&mut self, text: &str) -> (String, bool) {
        self.buf.push_str(text);
        if let Some(pos) = self
            .stops
            .iter()
            .filter_map(|s| self.buf.find(s.as_str()))
            .min()
        {
            let out = self.buf[..pos].to_string();
            self.buf.clear();
            return (out, true);
        }
        // Keep the longest buffer suffix that could still grow into a stop.
        let mut keep = 0;
        for stop in &self.stops {
            for k in (1..stop.len()).rev() {
                if k > keep && stop.is_char_boundary(k) && self.buf.ends_with(&stop[..k]) {
                    keep = k;
                    break;
                }
            }
        }
        let emit_to = self.buf.len() - keep;
        let out = self.buf[..emit_to].to_string();
        self.buf.drain(..emit_to);
        (out, false)
    }

    // End of generation without a stop hit: release the withheld tail.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], chunks: &[&str]) -> (String, bool) {
        let mut f = StopFilter::new(stops.iter().map(|s| s.to_string()).collect());
        let mut out = String::new();
        for c in chunks {
            let (t, hit) = f.push(c);
            out.push_str(&t);
            if hit {
                return (out, true);
            }
        }
        out.push_str(&f.flush());
        (out, false)
    }

    #[test]
    fn passes_through_without_stops() {
        assert_eq!(run(&[], &["ab", "cd"]), ("abcd".into(), false));
    }

    #[test]
    fn stops_inside_one_chunk() {
        assert_eq!(run(&["\n\n"], &["hello\n\nworld"]), ("hello".into(), true));
    }

    #[test]
    fn stop_split_across_chunks() {
        assert_eq!(run(&["STOP"], &["abST", "OPxyz"]), ("ab".into(), true));
    }

    #[test]
    fn partial_match_that_fails_is_released() {
        assert_eq!(run(&["STOP"], &["abST", "ART"]), ("abSTART".into(), false));
    }

    #[test]
    fn earliest_of_several_stops_wins() {
        assert_eq!(run(&["cc", "b"], &["abcc"]), ("a".into(), true));
    }

    #[test]
    fn multibyte_stop() {
        assert_eq!(run(&["日本"], &["x日", "本y"]), ("x".into(), true));
    }
}

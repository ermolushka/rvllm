use tokenizers::Tokenizer;

pub struct SmollLM230MTokenizer {
    inner: Tokenizer,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
}

impl SmollLM230MTokenizer {
    pub fn from_file(
        path: &str,
        eos_token_id: u32,
        bos_token_id: u32,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let inner = tokenizers::Tokenizer::from_file(path)?;
        Ok(Self {
            inner,
            eos_token_id,
            bos_token_id,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
        let encoding = self.inner.encode(text, false)?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        Ok(self.inner.decode(ids, true)?)
    }
}

impl SmollLM230MTokenizer {
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

// Turns a stream of token ids into text deltas. Decoding one id at a time
// would mangle multi-byte characters and leading spaces, so this decodes a
// small trailing window and only releases text once it no longer ends in an
// incomplete character (U+FFFD).
#[derive(Default)]
pub struct StreamDecoder {
    ids: Vec<u32>,
    // ids[prefix_offset..read_offset] is already-emitted context kept so the
    // next delta is decoded with the right spacing.
    prefix_offset: usize,
    read_offset: usize,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(
        &mut self,
        tok: &SmollLM230MTokenizer,
        id: u32,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        self.ids.push(id);
        let prefix = tok.decode(&self.ids[self.prefix_offset..self.read_offset])?;
        let text = tok.decode(&self.ids[self.prefix_offset..])?;
        if text.len() > prefix.len() && !text.ends_with('\u{FFFD}') {
            let delta = text.get(prefix.len()..).unwrap_or_default().to_string();
            self.prefix_offset = self.read_offset;
            self.read_offset = self.ids.len();
            Ok(delta)
        } else {
            Ok(String::new())
        }
    }

    // Whatever is still held back (an incomplete trailing character).
    pub fn flush(
        &mut self,
        tok: &SmollLM230MTokenizer,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let prefix = tok.decode(&self.ids[self.prefix_offset..self.read_offset])?;
        let text = tok.decode(&self.ids[self.prefix_offset..])?;
        self.prefix_offset = self.ids.len();
        self.read_offset = self.ids.len();
        Ok(text.get(prefix.len()..).unwrap_or_default().to_string())
    }
}

#[test]
fn stream_decoder_matches_full_decode() {
    let tok = SmollLM230MTokenizer::from_file("tokenizer.json", 0, 0).unwrap();
    // Emoji and accented text span several byte-level tokens.
    let text = "Héllo wörld 🙂 — streaming, ok?";
    let ids = tok.encode(text).unwrap();
    let mut dec = StreamDecoder::new();
    let mut out = String::new();
    for &id in &ids {
        out.push_str(&dec.push(&tok, id).unwrap());
    }
    out.push_str(&dec.flush(&tok).unwrap());
    assert_eq!(out, tok.decode(&ids).unwrap());
    assert!(!out.contains('\u{FFFD}'));
}

#[test]
fn round_trip() {
    let tok = SmollLM230MTokenizer::from_file("tokenizer.json", 0, 0).unwrap();
    let ids = tok.encode("Hello, world!").unwrap();
    let text = tok.decode(&ids).unwrap();
    assert_eq!(text.trim(), "Hello, world!");
}

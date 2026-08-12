use candle_nn::encoding;
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

#[test]
fn round_trip() {
    let tok = SmollLM230MTokenizer::from_file("../tokenizer.json", 0, 0).unwrap();
    let ids = tok.encode("Hello, world!").unwrap();
    let text = tok.decode(&ids).unwrap();
    assert_eq!(text.trim(), "Hello, world!");
}

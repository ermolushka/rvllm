use std::path::{Path, PathBuf};

use crate::model::Config;

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
    // Whether `encode` starts prompts with BOS, as the model was trained.
    add_bos: bool,
}

#[deprecated(note = "renamed to `Tokenizer`")]
pub type SmollLM230MTokenizer = Tokenizer;

impl Tokenizer {
    pub fn from_file(
        path: &str,
        eos_token_id: u32,
        bos_token_id: u32,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("can't load tokenizer {path}: {e}"))?;
        Ok(Self {
            inner,
            eos_token_id,
            bos_token_id,
            add_bos: false,
        })
    }

    // The tokenizer for `config`'s model. Fails when `path` holds another
    // model's tokenizer: one that gives the model's BOS or EOS token another id.
    pub fn for_model(
        path: &str,
        config: &Config,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut tokenizer = Self::from_file(path, config.eos_token_id, config.bos_token_id)?;
        let special = [
            (&config.bos_token, config.bos_token_id),
            (&config.eos_token, config.eos_token_id),
        ];
        for (token, id) in special {
            let Some(token) = token else { continue };
            let got = tokenizer.token_to_id(token);
            if got != Some(id) {
                let got = got.map_or("no such token".to_string(), |got| format!("token {got}"));
                return Err(format!(
                    "{path} is not this model's tokenizer: the model's {token} is token {id}, but this tokenizer has {got}"
                )
                .into());
            }
        }
        tokenizer.add_bos = config.add_bos;
        Ok(tokenizer)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
        let encoding = self.inner.encode(text, false)?;
        let mut ids = encoding.get_ids().to_vec();
        // An empty prompt stays empty, and one that already starts with BOS
        // doesn't get a second.
        if self.add_bos && ids.first().is_some_and(|&first| first != self.bos_token_id) {
            ids.insert(0, self.bos_token_id);
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        Ok(self.inner.decode(ids, true)?)
    }
}

impl Tokenizer {
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

// The tokenizer for a model file: the `tokenizer.json` next to it if there is
// one, else `fallback`.
pub fn path_for(model: &Path, fallback: &Path) -> PathBuf {
    let sibling = model.with_file_name("tokenizer.json");
    if sibling.exists() {
        sibling
    } else {
        fallback.to_path_buf()
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
        tok: &Tokenizer,
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
        tok: &Tokenizer,
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
    let tok = Tokenizer::from_file("tokenizer.json", 0, 0).unwrap();
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
    let tok = Tokenizer::from_file("tokenizer.json", 0, 0).unwrap();
    let ids = tok.encode("Hello, world!").unwrap();
    let text = tok.decode(&ids).unwrap();
    assert_eq!(text.trim(), "Hello, world!");
}

// SmolLM2-Instruct's special tokens, as in the repo's tokenizer.json.
#[cfg(test)]
fn smollm2_instruct_config() -> Config {
    Config {
        bos_token_id: 1,
        eos_token_id: 2,
        bos_token: Some("<|im_start|>".into()),
        eos_token: Some("<|im_end|>".into()),
        ..Config::for_tests()
    }
}

#[test]
fn bos_is_prepended_once_when_the_model_wants_it() {
    let config = Config {
        add_bos: true,
        ..smollm2_instruct_config()
    };
    let tok = Tokenizer::for_model("tokenizer.json", &config).unwrap();
    let plain = Tokenizer::from_file("tokenizer.json", 2, 1).unwrap();
    let with_bos = [vec![1], plain.encode("Hello").unwrap()].concat();
    assert_eq!(tok.encode("Hello").unwrap(), with_bos);
    assert_eq!(tok.encode("<|im_start|>Hello").unwrap(), with_bos);
    assert!(tok.encode("").unwrap().is_empty());
}

#[test]
fn load_error_names_the_file() {
    let Err(err) = Tokenizer::from_file("no/such/tokenizer.json", 0, 0) else {
        panic!("loaded a missing tokenizer");
    };
    assert!(err.to_string().contains("no/such/tokenizer.json"), "{err}");
}

#[test]
fn path_for_prefers_the_tokenizer_next_to_the_model() {
    let dir = std::env::temp_dir().join(format!("rvllm-path-for-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("model.gguf");
    let fallback = Path::new("tokenizer.json");
    assert_eq!(path_for(&model, fallback), fallback);
    std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    assert_eq!(path_for(&model, fallback), dir.join("tokenizer.json"));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn rejects_another_models_tokenizer() {
    assert!(Tokenizer::for_model("tokenizer.json", &smollm2_instruct_config()).is_ok());
    // Llama 3's EOS, which the SmolLM2 vocabulary doesn't have.
    let config = Config {
        eos_token_id: 128009,
        eos_token: Some("<|eot_id|>".into()),
        ..smollm2_instruct_config()
    };
    let Err(err) = Tokenizer::for_model("tokenizer.json", &config) else {
        panic!("accepted another model's tokenizer");
    };
    assert!(
        err.to_string().contains("<|eot_id|> is token 128009"),
        "{err}"
    );
}

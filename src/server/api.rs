// OpenAI-compatible request/response shapes. Unknown request fields are
// ignored, so clients sending options we don't implement still work.

use serde::{Deserialize, Serialize};

// `"stop": "x"` or `"stop": ["x", "y"]`.
#[derive(Deserialize)]
#[serde(untagged)]
pub enum StopField {
    One(String),
    Many(Vec<String>),
}

impl StopField {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopField::One(s) => vec![s],
            StopField::Many(v) => v,
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum PromptField {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Deserialize)]
pub struct CompletionRequest {
    pub prompt: PromptField,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub stop: Option<StopField>,
    pub n: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
}

#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    // Plain string, or an array of `{"type": "text", "text": ...}` parts.
    pub content: Option<ChatContent>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Deserialize)]
pub struct ContentPart {
    pub text: Option<String>,
}

impl ChatContent {
    pub fn into_text(self) -> String {
        match self {
            ChatContent::Text(t) => t,
            ChatContent::Parts(parts) => parts.into_iter().filter_map(|p| p.text).collect(),
        }
    }
}

#[derive(Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub stop: Option<StopField>,
    pub n: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
}

#[derive(Serialize, Clone, Copy)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelInfo>,
}

#[derive(Serialize)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub param: Option<String>,
    pub code: Option<&'static str>,
}

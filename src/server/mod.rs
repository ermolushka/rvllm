// OpenAI-compatible HTTP server: /v1/completions, /v1/chat/completions (both
// streaming and not), /v1/models and /health.
//
// Concurrency model: any number of connections are served by tokio, but all
// inference happens on one engine thread (`worker`) that continuously batches
// whatever is in flight. Two clients hitting the endpoint at once simply
// share decode steps. Beyond `max_running` concurrent sequences, requests wait
// in the engine's FIFO queue; beyond `max_running + max_queue` in flight, new
// requests get HTTP 429 instead of piling up unboundedly.

mod api;
mod launch;
mod stop;
pub mod worker;

pub use launch::{ServeOptions, run};

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::extract::{Json, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures_util::StreamExt;
use rand::Rng;
use serde_json::{Value, json};
use tokio::sync::Semaphore;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::chat::{self, END_OF_TURN};
use crate::engine::FinishReason;
use crate::sampler::SamplingParams;
use crate::tokenizer::{SmollLM230MTokenizer, StreamDecoder};

use api::*;
use stop::StopFilter;
use worker::{Job, WorkerEvent, WorkerStatus};

pub struct ServerConfig {
    pub model_name: String,
    // Concurrency the engine will batch together; informational here (the
    // engine enforces it), reported by /health.
    pub max_running: usize,
    // Requests allowed to wait beyond `max_running` before 429s start.
    pub max_queue: usize,
    pub default_max_tokens: usize,
    pub default_temperature: f32,
    pub default_top_p: f32,
    // Facts about the engine's limits, for rejecting requests up front.
    pub context_length: usize,
    pub max_sequence_tokens: usize,
}

pub struct AppState {
    cfg: ServerConfig,
    tokenizer: SmollLM230MTokenizer,
    jobs: UnboundedSender<Job>,
    slots: Arc<Semaphore>,
    status: Arc<WorkerStatus>,
    stop_tokens: Vec<u32>,
}

impl AppState {
    pub fn new(
        cfg: ServerConfig,
        tokenizer: SmollLM230MTokenizer,
        jobs: UnboundedSender<Job>,
        status: Arc<WorkerStatus>,
    ) -> Arc<Self> {
        let slots = Arc::new(Semaphore::new(cfg.max_running + cfg.max_queue));
        let mut stop_tokens = vec![tokenizer.eos_token_id];
        if let Some(id) = tokenizer.token_to_id(END_OF_TURN) {
            stop_tokens.push(id);
        }
        Arc::new(AppState {
            cfg,
            tokenizer,
            jobs,
            slots,
            status,
            stop_tokens,
        })
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .with_state(state)
}

pub async fn serve(listener: tokio::net::TcpListener, state: Arc<AppState>) -> std::io::Result<()> {
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
}

// ---- errors ---------------------------------------------------------------

pub struct ApiError {
    status: StatusCode,
    kind: &'static str,
    code: Option<&'static str>,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            code: None,
            message: message.into(),
        }
    }

    fn overloaded() -> Self {
        ApiError {
            status: StatusCode::TOO_MANY_REQUESTS,
            kind: "rate_limit_error",
            code: Some("queue_full"),
            message: "the server is at capacity (all slots and the queue are full); retry shortly"
                .into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: "server_error",
            code: None,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ErrorBody {
            error: ErrorDetail {
                message: self.message,
                kind: self.kind,
                param: None,
                code: self.code,
            },
        };
        let mut resp = (self.status, Json(body)).into_response();
        if self.status == StatusCode::TOO_MANY_REQUESTS {
            resp.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
        }
        resp
    }
}

// ---- generation -----------------------------------------------------------

struct GenRequest {
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    stop: Vec<String>,
    n: Option<usize>,
}

enum Chunk {
    Text(String),
    Done {
        finish_reason: &'static str,
        usage: Usage,
    },
    Error(String),
}

fn finish_str(f: FinishReason) -> &'static str {
    match f {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn new_id(prefix: &str) -> String {
    format!("{prefix}-{:024x}", rand::rng().random::<u128>() >> 8)
}

// Validates, takes a queue slot, hands the job to the engine thread and
// returns a channel of decoded text chunks. A task translates the engine's
// token events into text (incremental detokenization + stop strings); when
// the HTTP client disconnects, the chunk receiver drops, the task exits, the
// engine's event receiver drops with it, and the worker aborts the request.
fn start_generation(
    state: &Arc<AppState>,
    req: GenRequest,
) -> Result<UnboundedReceiver<Chunk>, ApiError> {
    if req.n.is_some_and(|n| n != 1) {
        return Err(ApiError::bad_request("only n=1 is supported"));
    }
    let temperature = req.temperature.unwrap_or(state.cfg.default_temperature);
    let top_p = req.top_p.unwrap_or(state.cfg.default_top_p);
    if !(0.0..=2.0).contains(&temperature) {
        return Err(ApiError::bad_request("temperature must be between 0 and 2"));
    }
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(ApiError::bad_request("top_p must be in (0, 1]"));
    }
    let max_tokens = req.max_tokens.unwrap_or(state.cfg.default_max_tokens);
    if max_tokens == 0 {
        return Err(ApiError::bad_request("max_tokens must be at least 1"));
    }

    let tokens = state
        .tokenizer
        .encode(&req.prompt)
        .map_err(|e| ApiError::internal(format!("tokenization failed: {e}")))?;
    if tokens.is_empty() {
        return Err(ApiError::bad_request("prompt is empty"));
    }
    let prompt_tokens = tokens.len();
    if prompt_tokens >= state.cfg.context_length {
        return Err(ApiError::bad_request(format!(
            "prompt is {prompt_tokens} tokens, which does not fit the model's {}-token context",
            state.cfg.context_length
        )));
    }
    if prompt_tokens >= state.cfg.max_sequence_tokens {
        return Err(ApiError::bad_request(format!(
            "prompt is {prompt_tokens} tokens but the server's KV cache holds at most {} per request",
            state.cfg.max_sequence_tokens
        )));
    }
    // Generation is capped by whatever room the context and KV pool leave.
    let max_tokens = max_tokens.min(state.cfg.max_sequence_tokens - prompt_tokens);

    let permit = state
        .slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| ApiError::overloaded())?;

    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    state
        .jobs
        .send(Job {
            tokens,
            max_tokens,
            stop_tokens: state.stop_tokens.clone(),
            sampling: SamplingParams {
                temperature,
                top_p,
                seed: req.seed.unwrap_or_else(|| rand::rng().random()),
            },
            events: event_tx,
            permit,
        })
        .map_err(|_| ApiError::internal("inference worker is not running"))?;

    let (chunk_tx, chunk_rx) = mpsc::unbounded_channel();
    let state = state.clone();
    let mut stop_filter = StopFilter::new(req.stop);
    tokio::spawn(async move {
        let mut decoder = StreamDecoder::new();
        let mut completion_tokens = 0usize;
        let done = |finish_reason, completion_tokens| Chunk::Done {
            finish_reason,
            usage: Usage::new(prompt_tokens, completion_tokens),
        };
        while let Some(ev) = event_rx.recv().await {
            let (token, finish) = match ev {
                WorkerEvent::Token { token, finish } => (token, finish),
                WorkerEvent::Error(message) => {
                    let _ = chunk_tx.send(Chunk::Error(message));
                    return;
                }
            };
            completion_tokens += 1;
            // The stop token ends the turn; it isn't part of the text.
            let hit_stop_token = finish == Some(FinishReason::Stop);
            let text = if hit_stop_token {
                decoder.flush(&state.tokenizer).unwrap_or_default()
            } else {
                match decoder.push(&state.tokenizer, token) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = chunk_tx.send(Chunk::Error(format!("detokenization failed: {e}")));
                        return;
                    }
                }
            };
            let (mut out, hit_stop_string) = stop_filter.push(&text);
            if finish.is_some() && !hit_stop_string {
                out.push_str(&stop_filter.flush());
            }
            if !out.is_empty() && chunk_tx.send(Chunk::Text(out)).is_err() {
                return;
            }
            if hit_stop_string {
                let _ = chunk_tx.send(done("stop", completion_tokens));
                return; // drops event_rx -> worker aborts the request
            }
            if let Some(f) = finish {
                let _ = chunk_tx.send(done(finish_str(f), completion_tokens));
                return;
            }
        }
        let _ = chunk_tx.send(Chunk::Error("engine stopped before finishing".into()));
    });
    Ok(chunk_rx)
}

// Collects a whole completion for the non-streaming endpoints.
async fn collect(
    mut rx: UnboundedReceiver<Chunk>,
) -> Result<(String, &'static str, Usage), ApiError> {
    let mut text = String::new();
    while let Some(chunk) = rx.recv().await {
        match chunk {
            Chunk::Text(t) => text.push_str(&t),
            Chunk::Done {
                finish_reason,
                usage,
            } => return Ok((text, finish_reason, usage)),
            Chunk::Error(m) => return Err(ApiError::internal(m)),
        }
    }
    Err(ApiError::internal("generation ended unexpectedly"))
}

fn sse(
    rx: UnboundedReceiver<Chunk>,
    include_usage: bool,
    mut render: impl FnMut(Option<&str>, Option<&str>, Option<Usage>) -> Value + Send + 'static,
    first: Option<Value>,
) -> Response {
    let mut first = first;
    let stream = UnboundedReceiverStream::new(rx).flat_map(move |chunk| {
        let mut events: Vec<Result<Event, Infallible>> = Vec::new();
        if let Some(f) = first.take() {
            events.push(Ok(Event::default().data(f.to_string())));
        }
        match chunk {
            Chunk::Text(t) => events.push(Ok(
                Event::default().data(render(Some(&t), None, None).to_string())
            )),
            Chunk::Done {
                finish_reason,
                usage,
            } => {
                events.push(Ok(
                    Event::default().data(render(None, Some(finish_reason), None).to_string())
                ));
                if include_usage {
                    events.push(Ok(
                        Event::default().data(render(None, None, Some(usage)).to_string())
                    ));
                }
                events.push(Ok(Event::default().data("[DONE]")));
            }
            Chunk::Error(message) => {
                let body = json!({"error": {"message": message, "type": "server_error"}});
                events.push(Ok(Event::default().data(body.to_string())));
                events.push(Ok(Event::default().data("[DONE]")));
            }
        }
        tokio_stream::iter(events)
    });
    Sse::new(stream).into_response()
}

// ---- handlers -------------------------------------------------------------

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    let prompt = match body.prompt {
        PromptField::One(p) => p,
        PromptField::Many(mut v) if v.len() == 1 => v.pop().unwrap(),
        PromptField::Many(_) => {
            return Err(ApiError::bad_request(
                "batched prompts are not supported; send one request per prompt",
            ));
        }
    };
    let rx = start_generation(
        &state,
        GenRequest {
            prompt,
            max_tokens: body.max_tokens,
            temperature: body.temperature,
            top_p: body.top_p,
            seed: body.seed,
            stop: body.stop.map(StopField::into_vec).unwrap_or_default(),
            n: body.n,
        },
    )?;
    let id = new_id("cmpl");
    let created = now();
    let model = state.cfg.model_name.clone();

    if body.stream {
        let include_usage = body.stream_options.is_some_and(|o| o.include_usage);
        return Ok(sse(
            rx,
            include_usage,
            move |text, finish, usage| {
                let mut v = json!({
                    "id": id, "object": "text_completion", "created": created, "model": model,
                    "choices": if usage.is_some() { json!([]) } else {
                        json!([{"index": 0, "text": text.unwrap_or(""), "logprobs": null, "finish_reason": finish}])
                    },
                });
                if let Some(u) = usage {
                    v["usage"] = serde_json::to_value(u).unwrap();
                }
                v
            },
            None,
        ));
    }

    let (text, finish_reason, usage) = collect(rx).await?;
    Ok(Json(json!({
        "id": id, "object": "text_completion", "created": created, "model": model,
        "choices": [{"index": 0, "text": text, "logprobs": null, "finish_reason": finish_reason}],
        "usage": usage,
    }))
    .into_response())
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    if body.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    let messages: Vec<chat::ChatMessage> = body
        .messages
        .into_iter()
        .map(|m| chat::ChatMessage {
            role: m.role,
            content: m.content.map(ChatContent::into_text).unwrap_or_default(),
        })
        .collect();
    let rx = start_generation(
        &state,
        GenRequest {
            prompt: chat::chatml(&messages),
            max_tokens: body.max_completion_tokens.or(body.max_tokens),
            temperature: body.temperature,
            top_p: body.top_p,
            seed: body.seed,
            stop: body.stop.map(StopField::into_vec).unwrap_or_default(),
            n: body.n,
        },
    )?;
    let id = new_id("chatcmpl");
    let created = now();
    let model = state.cfg.model_name.clone();

    if body.stream {
        let include_usage = body.stream_options.is_some_and(|o| o.include_usage);
        let chunk_of = {
            let (id, model) = (id.clone(), model.clone());
            move |delta: Value, finish: Option<&str>, usage: Option<Usage>| {
                let mut v = json!({
                    "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": if usage.is_some() { json!([]) } else {
                        json!([{"index": 0, "delta": delta, "finish_reason": finish}])
                    },
                });
                if let Some(u) = usage {
                    v["usage"] = serde_json::to_value(u).unwrap();
                }
                v
            }
        };
        let first = chunk_of(json!({"role": "assistant", "content": ""}), None, None);
        return Ok(sse(
            rx,
            include_usage,
            move |text, finish, usage| match text {
                Some(t) => chunk_of(json!({"content": t}), None, None),
                None => chunk_of(json!({}), finish, usage),
            },
            Some(first),
        ));
    }

    let (text, finish_reason, usage) = collect(rx).await?;
    Ok(Json(json!({
        "id": id, "object": "chat.completion", "created": created, "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }))
    .into_response())
}

async fn models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelInfo {
            id: state.cfg.model_name.clone(),
            object: "model",
            created: now(),
            owned_by: "rvllm",
        }],
    })
}

async fn health(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "running": state.status.running.load(Ordering::Relaxed),
        "waiting": state.status.waiting.load(Ordering::Relaxed),
        "max_running": state.cfg.max_running,
        "max_queue": state.cfg.max_queue,
    }))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use axum::body::Body;
    use axum::http::Request as HttpRequest;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;
    use crate::engine::testing::CountingBackend;
    use crate::engine::{Backend, BoxError, Engine, EngineConfig};
    use crate::model::{BatchItem, Config};
    use crate::sampler::Sampler;

    // CountingBackend with a per-forward delay so requests stay in flight
    // long enough to observe queueing and cancellation.
    struct SlowBackend {
        inner: CountingBackend,
        delay: Duration,
    }

    impl Backend for SlowBackend {
        fn config(&self) -> &Config {
            self.inner.config()
        }

        fn forward_sample(
            &mut self,
            items: &[BatchItem],
            last_only: bool,
            samplers: &mut [&mut Sampler],
        ) -> Result<Vec<u32>, BoxError> {
            std::thread::sleep(self.delay);
            self.inner.forward_sample(items, last_only, samplers)
        }
    }

    struct Harness {
        app: Router,
        status: Arc<WorkerStatus>,
    }

    fn harness(max_running: usize, max_queue: usize, delay_ms: u64) -> Harness {
        let backend = SlowBackend {
            inner: CountingBackend::new(512),
            delay: Duration::from_millis(delay_ms),
        };
        let engine = Engine::new(
            backend,
            EngineConfig {
                block_size: 4,
                num_blocks: 256,
                max_running,
            },
        );
        let max_sequence_tokens = engine.max_sequence_tokens();
        let (job_tx, job_rx) = mpsc::unbounded_channel();
        let status = Arc::new(WorkerStatus::default());
        let worker_status = status.clone();
        std::thread::spawn(move || worker::run_worker(engine, job_rx, worker_status));
        let tokenizer = SmollLM230MTokenizer::from_file("tokenizer.json", 999, 0).unwrap();
        let state = AppState::new(
            ServerConfig {
                model_name: "test-model".into(),
                max_running,
                max_queue,
                default_max_tokens: 16,
                default_temperature: 0.0,
                default_top_p: 1.0,
                context_length: 512,
                max_sequence_tokens,
            },
            tokenizer,
            job_tx,
            status.clone(),
        );
        Harness {
            app: router(state),
            status,
        }
    }

    fn post(path: &str, body: Value) -> HttpRequest<Body> {
        HttpRequest::post(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    async fn json_of(resp: Response) -> (StatusCode, Value) {
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn completion_returns_text_and_usage() {
        let h = harness(4, 4, 0);
        let resp = h
            .app
            .oneshot(post(
                "/v1/completions",
                json!({"prompt": "Hello", "max_tokens": 5}),
            ))
            .await
            .unwrap();
        let (status, body) = json_of(resp).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "text_completion");
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(body["usage"]["completion_tokens"], 5);
        assert!(body["usage"]["prompt_tokens"].as_u64().unwrap() >= 1);
        assert!(!body["choices"][0]["text"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn chat_completion_non_streaming() {
        let h = harness(4, 4, 0);
        let resp = h
            .app
            .oneshot(post(
                "/v1/chat/completions",
                json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}),
            ))
            .await
            .unwrap();
        let (status, body) = json_of(resp).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        assert_eq!(body["usage"]["completion_tokens"], 4);
    }

    #[tokio::test]
    async fn chat_streaming_sends_sse_chunks_then_done() {
        let h = harness(4, 4, 0);
        let resp = h
            .app
            .oneshot(post(
                "/v1/chat/completions",
                json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
                       "stream": true, "stream_options": {"include_usage": true}}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let events: Vec<&str> = text
            .lines()
            .filter_map(|l| l.strip_prefix("data:"))
            .map(str::trim)
            .collect();
        assert_eq!(*events.last().unwrap(), "[DONE]");
        let chunks: Vec<Value> = events[..events.len() - 1]
            .iter()
            .map(|e| serde_json::from_str(e).unwrap())
            .collect();
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert!(
            chunks
                .iter()
                .all(|c| c["object"] == "chat.completion.chunk")
        );
        let finish: Vec<_> = chunks
            .iter()
            .filter_map(|c| c["choices"][0]["finish_reason"].as_str())
            .collect();
        assert_eq!(finish, vec!["length"]);
        let usage_chunk = chunks.last().unwrap();
        assert_eq!(usage_chunk["usage"]["completion_tokens"], 4);
    }

    #[tokio::test]
    async fn stop_string_truncates_and_reports_stop() {
        let h = harness(4, 4, 0);
        // Greedy fake output is a fixed token sequence; find some of its text
        // first, then ask for a stop string taken from the middle of it.
        let full = h
            .app
            .clone()
            .oneshot(post(
                "/v1/completions",
                json!({"prompt": "Hello", "max_tokens": 12}),
            ))
            .await
            .unwrap();
        let (_, full) = json_of(full).await;
        let full_text = full["choices"][0]["text"].as_str().unwrap().to_string();
        let chars: Vec<char> = full_text.chars().collect();
        assert!(chars.len() > 8, "fake output too short: {full_text:?}");
        let stop: String = chars[4..6].iter().collect();
        let expected_prefix = full_text.split(&stop).next().unwrap().to_string();

        let resp = h
            .app
            .oneshot(post(
                "/v1/completions",
                json!({"prompt": "Hello", "max_tokens": 12, "stop": [stop]}),
            ))
            .await
            .unwrap();
        let (_, body) = json_of(resp).await;
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["choices"][0]["text"], expected_prefix);
    }

    #[tokio::test]
    async fn rejects_bad_requests() {
        let h = harness(4, 4, 0);
        for body in [
            json!({"prompt": "x", "max_tokens": 0}),
            json!({"prompt": "x", "n": 2}),
            json!({"prompt": ["a", "b"]}),
            json!({"prompt": "x", "temperature": 5.0}),
            json!({"prompt": ""}),
        ] {
            let resp = h
                .app
                .clone()
                .oneshot(post("/v1/completions", body.clone()))
                .await
                .unwrap();
            let (status, err) = json_of(resp).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(err["error"]["type"], "invalid_request_error");
        }
    }

    #[tokio::test]
    async fn full_queue_gets_429_and_slot_frees_up_afterwards() {
        // One running slot, no queue: while the first request is generating,
        // the second has nowhere to go.
        let h = harness(1, 0, 20);
        // Non-streaming `oneshot` resolves only when done, so run it in the
        // background and probe while it is in flight.
        let app = h.app.clone();
        let slow = tokio::spawn(async move {
            app.oneshot(post(
                "/v1/completions",
                json!({"prompt": "Hello", "max_tokens": 30}),
            ))
            .await
            .unwrap()
        });
        tokio::time::sleep(Duration::from_millis(150)).await;
        let rejected = h
            .app
            .clone()
            .oneshot(post(
                "/v1/completions",
                json!({"prompt": "Hi", "max_tokens": 2}),
            ))
            .await
            .unwrap();
        assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(rejected.headers().contains_key(header::RETRY_AFTER));
        let (_, err) = json_of(rejected).await;
        assert_eq!(err["error"]["code"], "queue_full");

        let (status, _) = json_of(slow.await.unwrap()).await;
        assert_eq!(status, StatusCode::OK);
        let again = h
            .app
            .oneshot(post(
                "/v1/completions",
                json!({"prompt": "Hi", "max_tokens": 2}),
            ))
            .await
            .unwrap();
        assert_eq!(again.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn concurrent_requests_queue_fifo_and_all_complete() {
        // 2 slots + 4 queue places, 6 simultaneous requests: all succeed.
        let h = harness(2, 4, 5);
        let mut tasks = Vec::new();
        for i in 0..6 {
            let app = h.app.clone();
            tasks.push(tokio::spawn(async move {
                let resp = app
                    .oneshot(post(
                        "/v1/completions",
                        json!({"prompt": format!("Request number {i}"), "max_tokens": 6}),
                    ))
                    .await
                    .unwrap();
                json_of(resp).await
            }));
        }
        for t in tasks {
            let (status, body) = t.await.unwrap();
            assert_eq!(status, StatusCode::OK);
            assert_eq!(body["usage"]["completion_tokens"], 6);
        }
    }

    #[tokio::test]
    async fn dropped_client_frees_its_slot() {
        let h = harness(1, 0, 10);
        let resp = h
            .app
            .clone()
            .oneshot(post(
                "/v1/completions",
                json!({"prompt": "Hello", "max_tokens": 400, "stream": true}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(h.status.running.load(Ordering::Relaxed), 1);
        drop(resp); // client disconnects mid-stream

        // The worker notices, aborts the sequence and releases the slot.
        let mut freed = false;
        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(20)).await;
            if h.status.running.load(Ordering::Relaxed) == 0 {
                freed = true;
                break;
            }
        }
        assert!(freed, "engine kept generating for a disconnected client");
        let ok = h
            .app
            .oneshot(post(
                "/v1/completions",
                json!({"prompt": "Hi", "max_tokens": 2}),
            ))
            .await
            .unwrap();
        assert_eq!(ok.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn models_and_health() {
        let h = harness(3, 5, 0);
        let resp = h
            .app
            .clone()
            .oneshot(HttpRequest::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let (_, models) = json_of(resp).await;
        assert_eq!(models["data"][0]["id"], "test-model");
        let resp = h
            .app
            .oneshot(HttpRequest::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let (_, health) = json_of(resp).await;
        assert_eq!(health["status"], "ok");
        assert_eq!(health["max_running"], 3);
        assert_eq!(health["max_queue"], 5);
    }
}

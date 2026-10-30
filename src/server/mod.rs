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
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::answer::{self, After, Answer, Ended};
use crate::chat::{self, ChatFormat};
use crate::engine::{BoxError, FinishReason};
use crate::history::{Compaction, History, Strategy};
use crate::sampler::SamplingParams;
use crate::tokenizer::{StreamDecoder, Tokenizer};

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
    // How a chat that outgrows the KV pool is compacted.
    pub compact: Strategy,
}

pub struct AppState {
    cfg: ServerConfig,
    tokenizer: Tokenizer,
    chat_format: Option<ChatFormat>,
    jobs: UnboundedSender<Job>,
    slots: Arc<Semaphore>,
    status: Arc<WorkerStatus>,
    stop_tokens: Vec<u32>,
}

impl AppState {
    pub fn new(
        cfg: ServerConfig,
        tokenizer: Tokenizer,
        chat_format: Option<ChatFormat>,
        jobs: UnboundedSender<Job>,
        status: Arc<WorkerStatus>,
    ) -> Arc<Self> {
        let slots = Arc::new(Semaphore::new(cfg.max_running + cfg.max_queue));
        let stop_tokens = chat::stop_tokens(&tokenizer, chat_format);
        Arc::new(AppState {
            cfg,
            tokenizer,
            chat_format,
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

// What a request asks the model to continue.
enum Prompt {
    // Raw text, from /v1/completions. There is no history to compact, so the
    // answer is capped at the room the pool leaves beside the prompt.
    Text(String),
    // A conversation, from /v1/chat/completions. It is compacted when it
    // outgrows the room the pool leaves, and an answer that fills the pool is
    // continued after compaction.
    Chat {
        messages: Vec<chat::ChatMessage>,
        format: ChatFormat,
    },
}

struct GenRequest {
    prompt: Prompt,
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

// The API's finish reason for how an answer ended.
fn ended_str(ended: Ended) -> &'static str {
    match ended {
        Ended::Stop => "stop",
        Ended::Length | Ended::PoolFull | Ended::Cancelled => "length",
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

// Validates a request, takes a queue slot, and starts its first segment on the
// engine thread. Returns the channel of text chunks; a task then forwards the
// rest of the answer, including any continuation after compaction. A client
// that disconnects drops the channel, which ends the task and aborts the
// request on the engine.
async fn start_generation(
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
    let pool = state.cfg.max_sequence_tokens;

    // A text prompt is checked first, so a bad one is refused as such even when
    // the queue is full.
    let text_tokens = match &req.prompt {
        Prompt::Text(text) => {
            let tokens = encode(state, text)?;
            check_prompt_fits(state, tokens.len())?;
            Some(tokens)
        }
        Prompt::Chat { .. } => None,
    };
    // The slot is taken before any more work, so a full queue is refused at once.
    let permit = state
        .slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| ApiError::overloaded())?;

    let (tokens, chat) = match (req.prompt, text_tokens) {
        (_, Some(tokens)) => (tokens, None),
        (Prompt::Text(_), None) => unreachable!("text prompts are encoded above"),
        (Prompt::Chat { messages, format }, None) => {
            let mut history = history_of(&messages);
            let mut tokens = encode(state, &format.render(&messages))?;
            // A conversation that fits as sent is used as it is. Otherwise it is
            // compacted to leave the answer its room.
            let reserve = answer::reserve_for(max_tokens, pool);
            if tokens.len() > pool.saturating_sub(reserve) {
                let (compacted, _) =
                    compact(state, format, history, reserve, 0)
                        .await
                        .map_err(|e| {
                            ApiError::internal(format!("compacting the history failed: {e}"))
                        })?;
                history = compacted;
                tokens = encode(state, &format.render(&history.messages()))?;
            }
            check_prompt_fits(state, tokens.len())?;
            (tokens, Some((history, format)))
        }
    };

    let continuable = chat.is_some();
    let prompt_tokens = tokens.len();
    let answer = Answer::new(pool, max_tokens, tokens, continuable);
    let segment = answer
        .next_segment()
        .ok_or_else(|| ApiError::internal("the prompt leaves no room for the answer"))?;
    let sampling = SamplingParams {
        temperature,
        top_p,
        seed: req.seed.unwrap_or_else(|| rand::rng().random()),
    };
    let (events, event_rx) = mpsc::unbounded_channel();
    send_segment(
        state,
        &answer,
        segment,
        sampling,
        continuable,
        events,
        Some(permit),
    )?;

    let (chunks, chunk_rx) = mpsc::unbounded_channel();
    let stream = Stream {
        state: state.clone(),
        answer,
        chat,
        sampling,
        prompt_tokens,
        chunks,
        events: event_rx,
        decoder: StreamDecoder::new(),
        stop_filter: StopFilter::new(req.stop),
        completion_tokens: 0,
    };
    tokio::spawn(stream.run());
    Ok(chunk_rx)
}

fn encode(state: &AppState, text: &str) -> Result<Vec<u32>, ApiError> {
    let tokens = state
        .tokenizer
        .encode(text)
        .map_err(|e| ApiError::internal(format!("tokenization failed: {e}")))?;
    if tokens.is_empty() {
        return Err(ApiError::bad_request("prompt is empty"));
    }
    Ok(tokens)
}

fn check_prompt_fits(state: &AppState, prompt_tokens: usize) -> Result<(), ApiError> {
    if prompt_tokens >= state.cfg.context_length {
        return Err(ApiError::bad_request(format!(
            "prompt is {prompt_tokens} tokens, which does not fit the model's {}-token context",
            state.cfg.context_length
        )));
    }
    if prompt_tokens >= state.cfg.max_sequence_tokens {
        return Err(ApiError::bad_request(format!(
            "prompt is {prompt_tokens} tokens, which does not fit the server's KV cache of {} per request",
            state.cfg.max_sequence_tokens
        )));
    }
    Ok(())
}

// The conversation as a `History`. Leading system messages become its system
// prompt, and the rest are its turns.
fn history_of(messages: &[chat::ChatMessage]) -> History {
    let leading = messages.iter().take_while(|m| m.role == "system").count();
    let system = (leading > 0).then(|| {
        messages[..leading]
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    });
    let mut history = History::new(system);
    for message in &messages[leading..] {
        history.push(&message.role, message.content.clone());
    }
    history
}

// Compacts `history` to fit `reserve` beside the prompt, with `pending` tokens
// of answer so far. Summaries are written on the engine, so this runs on a
// blocking thread that waits for them; the async side is never held up.
async fn compact(
    state: &Arc<AppState>,
    format: ChatFormat,
    mut history: History,
    reserve: usize,
    pending: usize,
) -> Result<(History, Compaction), BoxError> {
    let state = state.clone();
    let strategy = state.cfg.compact;
    let pool = state.cfg.max_sequence_tokens;
    let handle = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        let report = history.fit(
            strategy,
            pool,
            reserve,
            pending,
            |messages| Ok(state.tokenizer.encode(&format.render(messages))?.len()),
            |messages, cap| handle.block_on(summarize(&state, format, messages, cap)),
        )?;
        Ok((history, report))
    })
    .await?
}

// Writes the model's greedy summary of `messages`. It runs as a request of its
// own, with no queue slot: the request it serves already holds one.
async fn summarize(
    state: &AppState,
    format: ChatFormat,
    messages: &[chat::ChatMessage],
    max_tokens: usize,
) -> Result<String, BoxError> {
    let tokens = state.tokenizer.encode(&format.render(messages))?;
    let (events, mut rx) = mpsc::unbounded_channel();
    state
        .jobs
        .send(Job {
            tokens,
            max_tokens,
            stop_tokens: chat::stop_tokens(&state.tokenizer, Some(format)),
            sampling: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                seed: 0,
            },
            continuable: false,
            events,
            permit: None,
        })
        .map_err(|_| "inference worker is not running")?;
    let mut ids = Vec::new();
    let mut finished = false;
    while let Some(event) = rx.recv().await {
        match event {
            WorkerEvent::Token { token, finish } => {
                if finish != Some(FinishReason::Stop) {
                    ids.push(token);
                }
                if finish.is_some() {
                    finished = true;
                    break;
                }
            }
            WorkerEvent::Error(message) => return Err(message.into()),
        }
    }
    if !finished {
        return Err("inference worker stopped before the summary finished".into());
    }
    state.tokenizer.decode(&ids)
}

// Hands one segment of `answer` to the engine thread.
fn send_segment(
    state: &AppState,
    answer: &Answer,
    max_tokens: usize,
    sampling: SamplingParams,
    continuable: bool,
    events: UnboundedSender<WorkerEvent>,
    permit: Option<OwnedSemaphorePermit>,
) -> Result<(), ApiError> {
    state
        .jobs
        .send(Job {
            tokens: answer.tokens(),
            max_tokens,
            stop_tokens: state.stop_tokens.clone(),
            sampling,
            continuable,
            events,
            permit,
        })
        .map_err(|_| ApiError::internal("inference worker is not running"))
}

// One answer on its way to the client: its segments, the text they produce,
// and any compaction between them.
struct Stream {
    state: Arc<AppState>,
    answer: Answer,
    // The conversation, for a chat: compacted between segments.
    chat: Option<(History, ChatFormat)>,
    sampling: SamplingParams,
    prompt_tokens: usize,
    chunks: UnboundedSender<Chunk>,
    events: UnboundedReceiver<WorkerEvent>,
    decoder: StreamDecoder,
    stop_filter: StopFilter,
    completion_tokens: usize,
}

impl Stream {
    async fn run(mut self) {
        loop {
            let (ids, ended) = match self.segment().await {
                Ok(Some(end)) => end,
                // Finished already: a stop string, or the client has gone.
                Ok(None) => return,
                Err(message) => {
                    let _ = self.chunks.send(Chunk::Error(message));
                    return;
                }
            };
            match self.answer.record(&ids, ended) {
                After::Finished(ended) => return self.finish(ended_str(ended)),
                After::Limit => return self.finish("length"),
                After::Compact => {
                    if !self.compact_and_resume().await {
                        return self.finish("length");
                    }
                }
            }
        }
    }

    // Forwards the text of one segment. Returns its tokens and why it ended, or
    // None when the response was finished here.
    async fn segment(&mut self) -> Result<Option<(Vec<u32>, Ended)>, String> {
        let mut ids = Vec::new();
        loop {
            let Some(event) = self.events.recv().await else {
                return Err("engine stopped before finishing".into());
            };
            let (token, finish) = match event {
                WorkerEvent::Token { token, finish } => (token, finish),
                WorkerEvent::Error(message) => return Err(message),
            };
            ids.push(token);
            self.completion_tokens += 1;
            // The stop token ends the turn; it isn't part of the text.
            let text = if finish == Some(FinishReason::Stop) {
                self.decoder.flush(&self.state.tokenizer)
            } else {
                self.decoder.push(&self.state.tokenizer, token)
            }
            .map_err(|e| format!("detokenization failed: {e}"))?;
            let (out, hit_stop_string) = self.stop_filter.push(&text);
            if !out.is_empty() && self.chunks.send(Chunk::Text(out)).is_err() {
                return Ok(None);
            }
            if hit_stop_string {
                self.done("stop");
                return Ok(None);
            }
            if let Some(finish) = finish {
                return Ok(Some((ids, finish.into())));
            }
        }
    }

    // Compacts the conversation to make room beside the answer so far, and
    // starts the next segment. False if the answer can't go on.
    async fn compact_and_resume(&mut self) -> bool {
        let Some((history, format)) = self.chat.take() else {
            return false;
        };
        let reserve = self.answer.reserve();
        let pending = self.answer.ids().len();
        let history = match compact(&self.state, format, history, reserve, pending).await {
            Ok((history, _)) => history,
            Err(e) => {
                eprintln!("compacting the history failed: {e}");
                return false;
            }
        };
        let Ok(prompt) = self
            .state
            .tokenizer
            .encode(&format.render(&history.messages()))
        else {
            return false;
        };
        self.chat = Some((history, format));
        if !self.answer.rebase(prompt) || self.chunks.is_closed() {
            return false;
        }
        // Waits for a slot like any request; the segment before this one has
        // given its slot back when it ended.
        let Ok(permit) = self.state.slots.clone().acquire_owned().await else {
            return false;
        };
        let Some(segment) = self.answer.next_segment() else {
            return false;
        };
        let (events, event_rx) = mpsc::unbounded_channel();
        if send_segment(
            &self.state,
            &self.answer,
            segment,
            self.sampling,
            true,
            events,
            Some(permit),
        )
        .is_err()
        {
            return false;
        }
        self.events = event_rx;
        true
    }

    // Ends the answer: flushes what the decoder and stop filter still hold.
    fn finish(&mut self, reason: &'static str) {
        let rest = self
            .decoder
            .flush(&self.state.tokenizer)
            .unwrap_or_default();
        let (mut out, hit_stop_string) = self.stop_filter.push(&rest);
        if !hit_stop_string {
            out.push_str(&self.stop_filter.flush());
        }
        if !out.is_empty() {
            let _ = self.chunks.send(Chunk::Text(out));
        }
        self.done(if hit_stop_string { "stop" } else { reason });
    }

    fn done(&self, finish_reason: &'static str) {
        let _ = self.chunks.send(Chunk::Done {
            finish_reason,
            usage: Usage::new(self.prompt_tokens, self.completion_tokens),
        });
    }
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
            prompt: Prompt::Text(prompt),
            max_tokens: body.max_tokens,
            temperature: body.temperature,
            top_p: body.top_p,
            seed: body.seed,
            stop: body.stop.map(StopField::into_vec).unwrap_or_default(),
            n: body.n,
        },
    )
    .await?;
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
    let Some(format) = state.chat_format else {
        return Err(ApiError::bad_request(
            "this model's chat format is unknown; use /v1/completions",
        ));
    };
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
            prompt: Prompt::Chat { messages, format },
            max_tokens: body.max_completion_tokens.or(body.max_tokens),
            temperature: body.temperature,
            top_p: body.top_p,
            seed: body.seed,
            stop: body.stop.map(StopField::into_vec).unwrap_or_default(),
            n: body.n,
        },
    )
    .await?;
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
    use crate::answer::Ended;
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
        harness_with_format(max_running, max_queue, delay_ms, Some(ChatFormat::SMOLLM2))
    }

    fn harness_with_format(
        max_running: usize,
        max_queue: usize,
        delay_ms: u64,
        chat_format: Option<ChatFormat>,
    ) -> Harness {
        harness_full(
            max_running,
            max_queue,
            delay_ms,
            chat_format,
            512,
            256,
            Strategy::Summarize,
        )
    }

    // The pool is `num_blocks` x 4 tokens, capped at the context length.
    fn harness_full(
        max_running: usize,
        max_queue: usize,
        delay_ms: u64,
        chat_format: Option<ChatFormat>,
        context: u32,
        num_blocks: usize,
        compact: Strategy,
    ) -> Harness {
        let backend = SlowBackend {
            inner: CountingBackend::new(context),
            delay: Duration::from_millis(delay_ms),
        };
        let engine = Engine::new(
            backend,
            EngineConfig {
                block_size: 4,
                num_blocks,
                max_running,
            },
        );
        let max_sequence_tokens = engine.max_sequence_tokens();
        let (job_tx, job_rx) = mpsc::unbounded_channel();
        let status = Arc::new(WorkerStatus::default());
        let worker_status = status.clone();
        std::thread::spawn(move || worker::run_worker(engine, job_rx, worker_status));
        let tokenizer = Tokenizer::from_file("tokenizer.json", 999, 0).unwrap();
        let state = AppState::new(
            ServerConfig {
                model_name: "test-model".into(),
                max_running,
                max_queue,
                default_max_tokens: 16,
                default_temperature: 0.0,
                default_top_p: 1.0,
                context_length: context as usize,
                max_sequence_tokens,
                compact,
            },
            tokenizer,
            chat_format,
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
    async fn chat_needs_a_known_chat_format() {
        let h = harness_with_format(4, 4, 0, None);
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});
        let resp = h
            .app
            .clone()
            .oneshot(post("/v1/chat/completions", body))
            .await
            .unwrap();
        let (status, err) = json_of(resp).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let message = err["error"]["message"].as_str().unwrap();
        assert!(message.contains("chat format is unknown"), "{message}");
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

    fn tokens_of(text: &str) -> usize {
        Tokenizer::from_file("tokenizer.json", 999, 0)
            .unwrap()
            .encode(text)
            .unwrap()
            .len()
    }

    // `n` tokens' worth of ordinary text.
    fn words(n: usize) -> String {
        "cat ".repeat(n)
    }

    // A chat body from (role, words) pairs.
    fn chat(turns: &[(&str, usize)], max_tokens: usize, stream: bool) -> Value {
        let messages: Vec<Value> = turns
            .iter()
            .enumerate()
            .map(|(i, (role, n))| json!({"role": role, "content": format!("{} {i}", words(*n))}))
            .collect();
        json!({"messages": messages, "max_tokens": max_tokens, "stream": stream})
    }

    // The text of a non-streaming chat answer.
    async fn chat_text(app: &Router, body: Value) -> (Value, String) {
        let resp = app
            .clone()
            .oneshot(post("/v1/chat/completions", body))
            .await
            .unwrap();
        let (status, json) = json_of(resp).await;
        assert_eq!(status, StatusCode::OK, "{json}");
        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_string();
        (json, text)
    }

    #[test]
    fn ended_strings_map_the_engine_reasons_to_the_api() {
        assert_eq!(ended_str(Ended::Stop), "stop");
        assert_eq!(ended_str(Ended::Length), "length");
        assert_eq!(ended_str(Ended::PoolFull), "length");
        assert_eq!(ended_str(Ended::Cancelled), "length");
    }

    #[test]
    fn leading_system_messages_become_the_system_prompt() {
        let m = |role: &str, content: &str| chat::ChatMessage {
            role: role.into(),
            content: content.into(),
        };
        let history = history_of(&[
            m("system", "be terse"),
            m("system", "in French"),
            m("user", "hi"),
        ]);
        let messages = history.messages();
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[0].content, "be terse\n\nin French");
        assert_eq!(messages[1].content, "hi");
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn a_conversation_without_a_system_message_has_none() {
        let m = |role: &str, content: &str| chat::ChatMessage {
            role: role.into(),
            content: content.into(),
        };
        let history = history_of(&[m("user", "hi"), m("assistant", "hello"), m("user", "again")]);
        let roles: Vec<String> = history.messages().into_iter().map(|m| m.role).collect();
        assert_eq!(roles, vec!["user", "assistant", "user"]);
    }

    #[tokio::test]
    async fn a_chat_that_fits_is_sent_as_it_is() {
        let h = harness(4, 4, 0);
        let turns = [("user", 20), ("assistant", 20), ("user", 20)];
        let body = chat(&turns, 4, false);
        let (json, _) = chat_text(&h.app, body).await;
        let rendered: String = {
            let messages: Vec<chat::ChatMessage> = turns
                .iter()
                .enumerate()
                .map(|(i, (role, n))| chat::ChatMessage {
                    role: role.to_string(),
                    content: format!("{} {i}", words(*n)),
                })
                .collect();
            ChatFormat::SMOLLM2.render(&messages)
        };
        assert_eq!(
            json["usage"]["prompt_tokens"].as_u64().unwrap() as usize,
            tokens_of(&rendered),
            "a conversation that fits must not be changed"
        );
        assert_eq!(json["usage"]["completion_tokens"], 4);
    }

    #[tokio::test]
    async fn a_chat_over_the_budget_is_compacted_and_answered() {
        // Five turns of about 110 tokens each: more than the pool's 512 less
        // the answer's reserve, so older turns must go.
        let h = harness(4, 4, 0);
        let turns = [
            ("user", 110),
            ("assistant", 110),
            ("user", 110),
            ("assistant", 110),
            ("user", 110),
        ];
        let body = chat(&turns, 16, false);
        let (json, _) = chat_text(&h.app, body).await;
        let prompt = json["usage"]["prompt_tokens"].as_u64().unwrap() as usize;
        assert!(
            prompt < 512 - 16,
            "prompt of {prompt} tokens is over the budget"
        );
        assert_eq!(json["usage"]["completion_tokens"], 16);
        assert_eq!(json["choices"][0]["finish_reason"], "length");
    }

    #[tokio::test]
    async fn a_chat_over_the_budget_is_compacted_by_dropping_too() {
        let backend = SlowBackend {
            inner: CountingBackend::new(512),
            delay: Duration::from_millis(0),
        };
        let engine = Engine::new(
            backend,
            EngineConfig {
                block_size: 4,
                num_blocks: 256,
                max_running: 4,
            },
        );
        let (job_tx, job_rx) = mpsc::unbounded_channel();
        let status = Arc::new(WorkerStatus::default());
        let worker_status = status.clone();
        std::thread::spawn(move || worker::run_worker(engine, job_rx, worker_status));
        let state = AppState::new(
            ServerConfig {
                model_name: "test-model".into(),
                max_running: 4,
                max_queue: 4,
                default_max_tokens: 16,
                default_temperature: 0.0,
                default_top_p: 1.0,
                context_length: 512,
                max_sequence_tokens: 512,
                compact: Strategy::Drop,
            },
            Tokenizer::from_file("tokenizer.json", 999, 0).unwrap(),
            Some(ChatFormat::SMOLLM2),
            job_tx,
            status,
        );
        let app = router(state);
        let body = chat(
            &[
                ("user", 110),
                ("assistant", 110),
                ("user", 110),
                ("assistant", 110),
                ("user", 110),
            ],
            16,
            false,
        );
        let (json, _) = chat_text(&app, body).await;
        assert_eq!(json["usage"]["completion_tokens"], 16);
    }

    #[tokio::test]
    async fn a_chat_answer_longer_than_the_room_is_continued() {
        // About 225 tokens of prompt leave 287 of the 512 for the answer, which
        // asks for 400. The answer must be continued after compaction.
        let h = harness(4, 4, 0);
        let body = chat(&[("user", 80), ("assistant", 80), ("user", 20)], 400, false);
        let (json, _) = chat_text(&h.app, body).await;
        assert_eq!(json["usage"]["completion_tokens"], 400, "{json}");
        assert_eq!(json["choices"][0]["finish_reason"], "length");
    }

    #[tokio::test]
    async fn a_continued_answer_reads_as_one_uninterrupted_answer() {
        // The same chat in a pool big enough for the whole answer. The output
        // depends only on the last prompt token, which is the same in both.
        let turns = [("user", 80), ("assistant", 80), ("user", 20)];
        let small = harness(4, 4, 0);
        let (_, short) = chat_text(&small.app, chat(&turns, 400, false)).await;
        let big = harness_full(
            4,
            4,
            0,
            Some(ChatFormat::SMOLLM2),
            4096,
            1024,
            Strategy::Drop,
        );
        let (_, long) = chat_text(&big.app, chat(&turns, 400, false)).await;
        assert_eq!(short, long);
    }

    #[tokio::test]
    async fn a_continued_answer_streams_the_same_text_as_it_collects() {
        let turns = [("user", 80), ("assistant", 80), ("user", 20)];
        let h = harness(4, 4, 0);
        let (_, collected) = chat_text(&h.app, chat(&turns, 400, false)).await;
        let resp = h
            .app
            .clone()
            .oneshot(post("/v1/chat/completions", chat(&turns, 400, true)))
            .await
            .unwrap();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let mut streamed = String::new();
        let mut finish = Vec::new();
        for event in text
            .lines()
            .filter_map(|l| l.strip_prefix("data:"))
            .map(str::trim)
        {
            if event == "[DONE]" {
                continue;
            }
            let chunk: Value = serde_json::from_str(event).unwrap();
            if let Some(piece) = chunk["choices"][0]["delta"]["content"].as_str() {
                streamed.push_str(piece);
            }
            if let Some(reason) = chunk["choices"][0]["finish_reason"].as_str() {
                finish.push(reason.to_string());
            }
        }
        assert_eq!(streamed, collected);
        assert_eq!(finish, vec!["length"], "exactly one finish, at the end");
    }

    #[tokio::test]
    async fn a_stop_string_after_the_continuation_still_ends_the_answer() {
        let turns = [("user", 80), ("assistant", 80), ("user", 20)];
        let h = harness(4, 4, 0);
        let (_, full) = chat_text(&h.app, chat(&turns, 400, false)).await;
        let chars: Vec<char> = full.chars().collect();
        assert!(chars.len() > 40, "answer too short: {full:?}");
        // Near the end, which is past the point where the first segment stops.
        let stop: String = chars[chars.len() - 6..chars.len() - 4].iter().collect();
        let expected = full.split(&stop).next().unwrap().to_string();
        let mut body = chat(&turns, 400, false);
        body["stop"] = json!([stop]);
        let (json, text) = chat_text(&h.app, body).await;
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(text, expected);
    }

    #[tokio::test]
    async fn a_completion_is_capped_at_the_room_and_never_continued() {
        let h = harness(4, 4, 0);
        let resp = h
            .app
            .clone()
            .oneshot(post(
                "/v1/completions",
                json!({"prompt": words(200), "max_tokens": 400}),
            ))
            .await
            .unwrap();
        let (status, json) = json_of(resp).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["choices"][0]["finish_reason"], "length");
        let prompt = json["usage"]["prompt_tokens"].as_u64().unwrap() as usize;
        let completion = json["usage"]["completion_tokens"].as_u64().unwrap() as usize;
        assert_eq!(
            prompt + completion,
            512,
            "the answer fills the pool and stops"
        );
    }

    #[tokio::test]
    async fn a_chat_message_too_big_for_the_pool_is_refused() {
        let h = harness(4, 4, 0);
        let body = chat(&[("user", 600)], 8, false);
        let resp = h
            .app
            .clone()
            .oneshot(post("/v1/chat/completions", body))
            .await
            .unwrap();
        let (status, err) = json_of(resp).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let message = err["error"]["message"].as_str().unwrap();
        assert!(message.contains("does not fit"), "{message}");
    }

    #[tokio::test]
    async fn a_bad_text_prompt_is_refused_even_when_the_queue_is_full() {
        // One slot, no queue, and a slow request holding the slot.
        let h = harness(1, 0, 20);
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
        let resp = h
            .app
            .clone()
            .oneshot(post(
                "/v1/completions",
                json!({"prompt": words(600), "max_tokens": 2}),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let _ = json_of(slow.await.unwrap()).await;
    }

    #[tokio::test]
    async fn a_client_leaving_during_a_chat_answer_frees_its_slot() {
        let h = harness(1, 0, 10);
        let resp = h
            .app
            .clone()
            .oneshot(post(
                "/v1/chat/completions",
                chat(&[("user", 80), ("assistant", 80), ("user", 20)], 400, true),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(resp);
        let mut freed = false;
        for _ in 0..100 {
            tokio::time::sleep(Duration::from_millis(20)).await;
            if h.status.running.load(Ordering::Relaxed) == 0 {
                freed = true;
                break;
            }
        }
        assert!(freed, "the engine kept generating for a client that left");
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

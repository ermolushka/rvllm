// Full-screen terminal UI for `rvllm cli --tui`: a model picker plus a chat
// view with streamed output.
//
// The engine (and the model/runtime it borrows) lives on a worker thread so
// the UI never blocks on a weight load or a decode step, and device state
// (CUDA context) never crosses threads. The two sides talk over channels:
// `Cmd` goes to the worker, `Event` comes back. Cancelling a generation uses a
// shared flag the worker checks between decode steps.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use kv_cache_scheduler::sequence::SequenceId;
use ratatui::crossterm::event::{self, Event as TermEvent, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::{DefaultTerminal, Frame};

use crate::chat::{self, ChatFormat, ChatMessage};
use crate::engine::{self, BoxError, Engine, EngineConfig, Request};
use crate::history::{self, History, Strategy};
use crate::model::Model;
use crate::runtime::Runtime;
use crate::sampler::{Sampler, SamplingParams};
use crate::tokenizer::{self, StreamDecoder, Tokenizer};

pub struct TuiOptions {
    pub device: String,
    pub block_size: usize,
    pub num_blocks: Option<usize>,
    pub max_tokens: usize,
    pub chat: bool,
    // System prompt for chat mode, in place of the model's default one.
    pub system: Option<String>,
    pub sampling: SamplingParams,
    // Used for models with no `tokenizer.json` next to them.
    pub fallback_tokenizer: PathBuf,
    // Where the picker looks for `.gguf` files.
    pub models_dir: PathBuf,
    // Load this model right away instead of opening the picker.
    pub initial_model: Option<PathBuf>,
    // What to do with old turns once a chat outgrows the KV budget.
    pub compact: Strategy,
}

#[derive(Clone)]
struct ModelEntry {
    path: PathBuf,
    tokenizer: PathBuf,
    size: u64,
}

impl ModelEntry {
    fn name(&self) -> String {
        self.path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    }
}

// `.gguf` files in `dir` and one level of subdirectories.
fn discover(dir: &Path, fallback: &Path) -> Vec<ModelEntry> {
    fn scan(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && depth > 0 {
                scan(&path, depth - 1, out);
            } else if path.extension().is_some_and(|e| e == "gguf") {
                out.push(path);
            }
        }
    }
    let mut paths = Vec::new();
    scan(dir, 1, &mut paths);
    paths.sort();
    paths
        .into_iter()
        .map(|path| ModelEntry {
            tokenizer: tokenizer::path_for(&path, fallback),
            size: std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
            path,
        })
        .collect()
}

fn human_size(bytes: u64) -> String {
    const MB: f64 = 1024.0 * 1024.0;
    let mb = bytes as f64 / MB;
    if mb >= 1024.0 {
        format!("{:.1} GB", mb / 1024.0)
    } else {
        format!("{mb:.0} MB")
    }
}

// Per-request knobs, adjustable live from the settings panel.
#[derive(Clone)]
struct Settings {
    sampling: SamplingParams,
    max_tokens: usize,
    chat: bool,
}

enum Cmd {
    Load {
        model: PathBuf,
        tokenizer: PathBuf,
        device: String,
    },
    Send(u64, String, Settings),
    NewChat,
}

// One message on its way to the model. `id` is how the UI learns that the
// worker took it into the chat.
struct Turn {
    id: u64,
    line: String,
}

enum Event {
    Loading(String),
    Ready {
        name: String,
        context: usize,
    },
    Token(String),
    Finished {
        tokens: usize,
        secs: f64,
        hit_limit: bool,
    },
    // A line for the transcript, such as what compaction did.
    Notice(String),
    // The worker took this message into the chat, so it is no longer unsent.
    Accepted(u64),
    // The send was stopped before it got a reply, so no reply entry is kept.
    Stopped,
    Error(String),
}

struct WorkerCtx {
    opts: Arc<TuiOptions>,
    tx: Sender<Event>,
    cancel: Arc<AtomicBool>,
}

fn run_worker(ctx: WorkerCtx, rx: Receiver<Cmd>) {
    let mut next = rx.recv().ok();
    while let Some(cmd) = next.take() {
        next = match cmd {
            Cmd::Load {
                model,
                tokenizer,
                device,
            } => match session(&ctx, &rx, &model, &tokenizer, &device) {
                Ok(next) => next,
                Err(e) => {
                    let _ = ctx.tx.send(Event::Error(e.to_string()));
                    rx.recv().ok()
                }
            },
            _ => {
                let _ = ctx.tx.send(Event::Error("no model loaded".into()));
                rx.recv().ok()
            }
        };
    }
}

// Serves one loaded model until the UI asks for another (returned) or hangs up.
fn session(
    ctx: &WorkerCtx,
    rx: &Receiver<Cmd>,
    model_path: &Path,
    tokenizer_path: &Path,
    device: &str,
) -> Result<Option<Cmd>, BoxError> {
    let opts = &ctx.opts;
    let name = model_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let _ = ctx.tx.send(Event::Loading(name.clone()));

    let model = Model::load(&model_path.to_string_lossy())?;
    let tokenizer = Tokenizer::for_model(&tokenizer_path.to_string_lossy(), &model.config)?;
    let chat_format = ChatFormat::for_model(&model.config);
    let runtime = Runtime::new(&model, device)?;
    let context = model.config.context_length as usize;
    let num_blocks = opts
        .num_blocks
        .unwrap_or_else(|| context.min(4096).div_ceil(opts.block_size).max(1));
    let mut engine = Engine::new(
        runtime.backend(opts.block_size, num_blocks)?,
        EngineConfig {
            block_size: opts.block_size,
            num_blocks,
            max_running: 1,
        },
    );
    let _ = ctx.tx.send(Event::Ready { name, context });

    // Every chat, including one started with Ctrl-N, opens with the system
    // prompt, if one was given.
    let mut history = History::new(chat::system_prompt(chat_format, opts.system.as_deref()));
    loop {
        match rx.recv() {
            Ok(Cmd::NewChat) => history.clear(),
            Ok(load @ Cmd::Load { .. }) => return Ok(Some(load)),
            Ok(Cmd::Send(id, line, settings)) => {
                if let Err(e) = generate(
                    ctx,
                    &mut engine,
                    &tokenizer,
                    chat_format,
                    &mut history,
                    Turn { id, line },
                    &settings,
                ) {
                    let _ = ctx.tx.send(Event::Error(e.to_string()));
                }
            }
            Err(_) => return Ok(None),
        }
    }
}

fn generate<B: engine::Backend>(
    ctx: &WorkerCtx,
    engine: &mut Engine<B>,
    tokenizer: &Tokenizer,
    chat_format: Option<ChatFormat>,
    history: &mut History,
    turn: Turn,
    settings: &Settings,
) -> Result<(), BoxError> {
    let chat = settings.chat;
    // Cleared before compacting, so an Esc during a summary stops the send.
    ctx.cancel.store(false, Ordering::Relaxed);
    if chat {
        if chat_format.is_none() {
            return Err("this model's chat format is unknown; turn chat off".into());
        }
        history.push("user", turn.line.clone());
    }
    let reply = match send(
        ctx,
        engine,
        tokenizer,
        chat_format,
        history,
        &turn,
        settings,
    ) {
        Ok(reply) => reply,
        Err(e) => {
            // The turn never got an answer, so it doesn't stay in the history.
            if chat {
                history.pop();
            }
            if e.is::<history::Cancelled>() {
                // Esc while summarizing stops the send: no reply, and no error.
                let _ = ctx.tx.send(Event::Notice(
                    "stopped while summarizing; the message was not sent".into(),
                ));
                let _ = ctx.tx.send(Event::Stopped);
                return Ok(());
            }
            return Err(e);
        }
    };
    if chat {
        history.push("assistant", reply.text);
    }
    let _ = ctx.tx.send(Event::Finished {
        tokens: reply.tokens,
        secs: reply.secs,
        hit_limit: reply.hit_limit,
    });
    Ok(())
}

// The room kept free for an answer: its length, but at most a quarter of the
// pool, so a long answer can't squeeze the history out. An answer that outgrows
// that room is continued after compaction.
fn answer_reserve(max_tokens: usize, pool: usize) -> usize {
    max_tokens.min(pool / 4).max(1)
}

// How many times one answer may be continued after compaction.
const MAX_CONTINUATIONS: usize = 64;

// Builds the prompt for `turn`, then streams the reply to the transcript. In
// chat, an answer that fills the pool is continued after the history is
// compacted, so the pool never cuts it short.
fn send<B: engine::Backend>(
    ctx: &WorkerCtx,
    engine: &mut Engine<B>,
    tokenizer: &Tokenizer,
    chat_format: Option<ChatFormat>,
    history: &mut History,
    turn: &Turn,
    settings: &Settings,
) -> Result<Reply, BoxError> {
    let pool = engine.max_sequence_tokens();
    let format = chat_format.filter(|_| settings.chat);
    let mut base = match format {
        Some(format) => {
            let reserve = answer_reserve(settings.max_tokens, pool);
            compact(ctx, engine, tokenizer, format, history, reserve, 0)?;
            tokenizer.encode(&format.render(&history.messages()))?
        }
        None => tokenizer.encode(&turn.line)?,
    };
    let stop_tokens = chat::stop_tokens(tokenizer, chat_format);
    let _ = ctx.tx.send(Event::Accepted(turn.id));
    let mut reply = Reply::default();
    let mut continuations = 0;
    loop {
        // Each segment resumes from the prompt plus the answer so far. It grows
        // until the pool is full; then the history is compacted and it carries
        // on, if the pool has room for `reserve` tokens after that.
        let remaining = settings.max_tokens - reply.ids.len();
        let mut tokens = base.clone();
        tokens.extend_from_slice(&reply.ids);
        let room = match format {
            Some(_) => pool.saturating_sub(tokens.len()),
            None => remaining,
        };
        let segment = remaining.min(room);
        if segment == 0 {
            let _ = ctx.tx.send(Event::Notice(
                "the answer stopped: the history can't make room to continue it".into(),
            ));
            break;
        }
        let request = Request {
            tokens,
            max_tokens: segment,
            stop_tokens: stop_tokens.clone(),
            sampler: Sampler::from_params(settings.sampling),
        };
        let part = run_request(
            engine,
            tokenizer,
            &ctx.cancel,
            request,
            format.is_some(),
            |text| ctx.tx.send(Event::Token(text.to_string())).is_ok(),
        )?;
        reply.text.push_str(&part.text);
        reply.ids.extend_from_slice(&part.ids);
        reply.tokens += part.tokens;
        reply.secs += part.secs;
        reply.hit_limit = part.hit_limit && segment == remaining;
        reply.cancelled = part.cancelled;
        // The segment ended for want of room, not because the answer is done.
        let continues = part.pool_full || (part.hit_limit && segment < remaining);
        if !continues || part.cancelled {
            break;
        }
        continuations += 1;
        if continuations > MAX_CONTINUATIONS {
            let _ = ctx.tx.send(Event::Notice(format!(
                "the answer stopped after {MAX_CONTINUATIONS} continuations"
            )));
            break;
        }
        let Some(chat_format) = format else { break };
        // Compact the older turns to make room beside the answer so far, then
        // carry on from there. The budget is the pool less the reserve, as for
        // the first prompt.
        let reserve = answer_reserve(settings.max_tokens - reply.ids.len(), pool);
        match compact(
            ctx,
            engine,
            tokenizer,
            chat_format,
            history,
            reserve,
            reply.ids.len(),
        ) {
            Ok(_) => {}
            // Esc during the summary keeps the answer so far.
            Err(e) if e.is::<history::Cancelled>() => {
                reply.cancelled = true;
                break;
            }
            Err(e) => return Err(e),
        }
        base = tokenizer.encode(&chat_format.render(&history.messages()))?;
        if pool.saturating_sub(base.len() + reply.ids.len()) < reserve {
            let _ = ctx.tx.send(Event::Notice(
                "the answer stopped: the history can't make room to continue it".into(),
            ));
            break;
        }
        let _ = ctx.tx.send(Event::Notice(
            "continuing the answer after compacting the history".into(),
        ));
    }
    Ok(reply)
}

// What a request produced. `secs` is the time from its first step on, so it
// leaves out any compaction that ran before the request.
#[derive(Default)]
struct Reply {
    text: String,
    tokens: usize,
    // The generated token ids, in order, so a continuation resumes exactly.
    ids: Vec<u32>,
    hit_limit: bool,
    // The pool filled while this was the only request, so it can be continued.
    pool_full: bool,
    cancelled: bool,
    secs: f64,
}

// Runs `request` to the end, handing each piece of its text to `on_text`,
// which returns false to stop early. Setting `cancel` stops it too. If it
// fails part way, the request is aborted, so it doesn't keep running in the
// engine. A `continuable` request may stop with `pool_full`.
fn run_request<B: engine::Backend>(
    engine: &mut Engine<B>,
    tokenizer: &Tokenizer,
    cancel: &AtomicBool,
    request: Request,
    continuable: bool,
    mut on_text: impl FnMut(&str) -> bool,
) -> Result<Reply, BoxError> {
    let id = if continuable {
        engine.add_continuable_request(request)?
    } else {
        engine.add_request(request)?
    };
    let mut reply = Reply::default();
    let started = Instant::now();
    if let Err(e) = decode_request(engine, tokenizer, cancel, id, &mut reply, &mut on_text) {
        engine.abort(id);
        return Err(e);
    }
    reply.secs = started.elapsed().as_secs_f64();
    Ok(reply)
}

// The decode loop behind `run_request`. Only events for `id` count, so tokens
// from any other sequence never reach the reply.
fn decode_request<B: engine::Backend>(
    engine: &mut Engine<B>,
    tokenizer: &Tokenizer,
    cancel: &AtomicBool,
    id: SequenceId,
    reply: &mut Reply,
    on_text: &mut impl FnMut(&str) -> bool,
) -> Result<(), BoxError> {
    let mut decoder = StreamDecoder::new();
    'generate: while engine.has_work() {
        if cancel.load(Ordering::Relaxed) {
            engine.abort(id);
            reply.cancelled = true;
            break;
        }
        for ev in engine.step()? {
            if ev.id != id {
                continue;
            }
            reply.tokens += 1;
            reply.ids.push(ev.token);
            // A stop token is not part of the text. The last token of any other
            // finish is followed by whatever the decoder still holds.
            let text = match ev.finish {
                Some(engine::FinishReason::Stop) => decoder.flush(tokenizer)?,
                Some(_) => {
                    let mut text = decoder.push(tokenizer, ev.token)?;
                    text.push_str(&decoder.flush(tokenizer)?);
                    text
                }
                None => decoder.push(tokenizer, ev.token)?,
            };
            reply.text.push_str(&text);
            if !on_text(&text) {
                engine.abort(id);
                break 'generate;
            }
            reply.hit_limit = ev.finish == Some(engine::FinishReason::Length);
            reply.pool_full = ev.finish == Some(engine::FinishReason::PoolFull);
            if ev.finish.is_some() {
                break 'generate;
            }
        }
    }
    Ok(())
}

// Fits the chat history under the prompt budget: the pool less `reserve`
// tokens for the answer. The `extra` tokens of the answer so far count towards
// the prompt too. Says in the transcript what that cost.
fn compact<B: engine::Backend>(
    ctx: &WorkerCtx,
    engine: &mut Engine<B>,
    tokenizer: &Tokenizer,
    format: ChatFormat,
    history: &mut History,
    reserve: usize,
    extra: usize,
) -> Result<history::Compaction, BoxError> {
    let report = history.fit(
        ctx.opts.compact,
        engine.max_sequence_tokens(),
        reserve,
        extra,
        |messages| Ok(tokenizer.encode(&format.render(messages))?.len()),
        |messages, cap| {
            // Shown before the model starts writing, since the summary isn't streamed.
            let _ = ctx.tx.send(Event::Notice(
                "summarizing the earlier conversation...".into(),
            ));
            summarize(ctx, engine, tokenizer, format, messages, cap)
        },
    )?;
    if !report.is_empty() {
        let _ = ctx.tx.send(Event::Notice(compaction_note(&report)));
    }
    Ok(report)
}

// Writes the model's summary of `messages`, with the same template, stop
// tokens and engine as a reply. The summary is greedy and not streamed.
fn summarize<B: engine::Backend>(
    ctx: &WorkerCtx,
    engine: &mut Engine<B>,
    tokenizer: &Tokenizer,
    format: ChatFormat,
    messages: &[ChatMessage],
    max_tokens: usize,
) -> Result<String, BoxError> {
    let request = Request {
        tokens: tokenizer.encode(&format.render(messages))?,
        max_tokens,
        stop_tokens: chat::stop_tokens(tokenizer, Some(format)),
        sampler: Sampler::from_params(SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            seed: 0,
        }),
    };
    let reply = run_request(engine, tokenizer, &ctx.cancel, request, false, |_| true)?;
    if reply.cancelled {
        return Err(history::Cancelled.into());
    }
    Ok(reply.text)
}

fn compaction_note(report: &history::Compaction) -> String {
    let mut parts = Vec::new();
    if report.summarized > 0 {
        parts.push(format!("summarized {} earlier messages", report.summarized));
    }
    if report.dropped > 0 {
        parts.push(format!("dropped {} earlier messages", report.dropped));
    }
    if report.summary_dropped {
        parts.push("dropped the summary".to_string());
    }
    if let Some(reason) = &report.summary_error {
        parts.push(format!("couldn't summarize: {reason}"));
    }
    format!("history over the prompt budget: {}", parts.join(", "))
}

// Messages typed but not yet accepted by the worker, oldest first. A message
// leaves the outbox when the worker takes it into the chat, so until then it
// is shown as unsent.
#[derive(Default)]
struct Outbox {
    next_id: u64,
    items: VecDeque<(u64, String)>,
    // A message was handed to the worker and has not been accepted, stopped or
    // refused yet.
    in_flight: bool,
    // A send was stopped or refused, so nothing goes out until Enter.
    paused: bool,
}

impl Outbox {
    fn push(&mut self, text: String) {
        self.items.push_back((self.next_id, text));
        self.next_id += 1;
    }

    // The oldest unsent message, handed out once: nothing else goes out while
    // one is in flight or while paused.
    fn next_to_send(&mut self) -> Option<(u64, String)> {
        if self.in_flight || self.paused {
            return None;
        }
        let (id, text) = self.items.front()?.clone();
        self.in_flight = true;
        Some((id, text))
    }

    // The worker took message `id` into the chat: it leaves the outbox.
    fn accepted(&mut self, id: u64) -> Option<String> {
        if !self.in_flight || self.items.front()?.0 != id {
            return None;
        }
        self.in_flight = false;
        self.items.pop_front().map(|(_, text)| text)
    }

    // The in-flight message was stopped before the worker took it: it stays
    // unsent, and nothing goes out until Enter.
    fn stopped(&mut self) {
        self.in_flight = false;
        self.paused = true;
    }

    // The worker refused the in-flight message: it is dropped, and nothing else
    // goes out until Enter. Returns whether a message was in flight.
    fn refused(&mut self) -> bool {
        if !self.in_flight {
            return false;
        }
        self.in_flight = false;
        self.items.pop_front();
        self.paused = true;
        true
    }

    fn resume(&mut self) {
        self.paused = false;
    }

    fn clear(&mut self) {
        self.items.clear();
        self.in_flight = false;
        self.paused = false;
    }

    fn is_paused(&self) -> bool {
        self.paused
    }

    fn texts(&self) -> impl Iterator<Item = &str> {
        self.items.iter().map(|(_, text)| text.as_str())
    }
}

#[derive(PartialEq)]
enum Status {
    NoModel,
    Loading(String),
    Idle,
    Generating,
}

#[derive(Clone, Copy, PartialEq)]
enum Role {
    User,
    Assistant,
    Notice,
}

struct App {
    opts: Arc<TuiOptions>,
    cmd_tx: Sender<Cmd>,
    events: Receiver<Event>,
    cancel: Arc<AtomicBool>,

    status: Status,
    settings: Settings,
    // Device new loads use; `good_device` is the last one that loaded fine.
    device: String,
    good_device: String,
    current: Option<ModelEntry>,
    model_name: Option<String>,
    context: usize,
    last_speed: Option<String>,

    transcript: Vec<(Role, String)>,
    input: String,
    // Lines scrolled up from the bottom; 0 follows the stream.
    scroll_up: u16,

    picker: Option<Picker>,
    quit: bool,
    // Messages typed but not yet accepted by the worker.
    outbox: Outbox,
    // The last transcript entry is the reply being written.
    answering: bool,
}

#[derive(PartialEq)]
enum Focus {
    Models,
    Settings,
}

struct Picker {
    entries: Vec<ModelEntry>,
    state: ListState,
    focus: Focus,
    field: usize,
    // Digits typed into the selected settings field, applied on Enter.
    edit: Option<String>,
}

const FIELDS: usize = 5;

// Current value of a settings field as editable text.
fn field_text(settings: &Settings, field: usize) -> String {
    match field {
        0 => format!("{:.1}", settings.sampling.temperature),
        1 => format!("{:.2}", settings.sampling.top_p),
        2 => settings.max_tokens.to_string(),
        _ => settings.sampling.seed.to_string(),
    }
}

// Applies typed text to a numeric field; unparsable input is ignored.
fn apply_text(settings: &mut Settings, field: usize, text: &str) {
    match field {
        0 => {
            if let Ok(t) = text.parse::<f32>() {
                settings.sampling.temperature = t.clamp(0.0, 2.0);
            }
        }
        1 => {
            if let Ok(p) = text.parse::<f32>() {
                settings.sampling.top_p = p.clamp(0.01, 1.0);
            }
        }
        2 => {
            if let Ok(n) = text.parse::<usize>() {
                settings.max_tokens = n.max(1);
            }
        }
        3 => {
            if let Ok(n) = text.parse::<u64>() {
                settings.sampling.seed = n;
            }
        }
        _ => {}
    }
}

// Nudges one settings field (`coarse` for PgUp/PgDn-sized steps); returns true
// if the chat template toggled.
fn adjust(settings: &mut Settings, field: usize, dir: i32, coarse: bool) -> bool {
    let d = dir as f32;
    match field {
        0 => {
            let t = settings.sampling.temperature + 0.1 * d;
            settings.sampling.temperature = (t * 10.0).round().clamp(0.0, 20.0) / 10.0;
        }
        1 => {
            let p = settings.sampling.top_p + 0.05 * d;
            settings.sampling.top_p = (p * 20.0).round().clamp(1.0, 20.0) / 20.0;
        }
        2 => {
            let step = match (coarse, settings.max_tokens >= 256) {
                (false, _) => 1,
                (true, true) => 64,
                (true, false) => 16,
            };
            let n = settings.max_tokens as i64 + step * dir as i64;
            settings.max_tokens = n.max(1) as usize;
        }
        3 => {
            settings.sampling.seed = settings
                .sampling
                .seed
                .saturating_add_signed(dir as i64 * if coarse { 10 } else { 1 });
        }
        _ => {
            settings.chat = !settings.chat;
            return true;
        }
    }
    false
}

impl App {
    fn open_picker(&mut self) {
        let entries = discover(&self.opts.models_dir, &self.opts.fallback_tokenizer);
        let mut state = ListState::default();
        if !entries.is_empty() {
            state.select(Some(0));
        }
        self.picker = Some(Picker {
            entries,
            state,
            focus: Focus::Models,
            field: 0,
            edit: None,
        });
    }

    fn load(&mut self, entry: &ModelEntry) {
        self.status = Status::Loading(entry.name());
        self.current = Some(entry.clone());
        let _ = self.cmd_tx.send(Cmd::Load {
            model: entry.path.clone(),
            tokenizer: entry.tokenizer.clone(),
            device: self.device.clone(),
        });
    }

    fn drain_events(&mut self) {
        while let Ok(ev) = self.events.try_recv() {
            match ev {
                Event::Loading(name) => self.status = Status::Loading(name),
                Event::Ready { name, context } => {
                    self.outbox.clear();
                    self.transcript.clear();
                    self.transcript
                        .push((Role::Notice, format!("loaded {name}")));
                    self.good_device = self.device.clone();
                    self.model_name = Some(name);
                    self.context = context;
                    self.last_speed = None;
                    self.status = Status::Idle;
                }
                Event::Token(text) => {
                    if let Some((Role::Assistant, answer)) = self.transcript.last_mut() {
                        answer.push_str(&text);
                    }
                }
                Event::Notice(text) => {
                    // While a reply is being written, its entry is the last one
                    // and tokens still go to it, so the notice goes before it.
                    if self.answering {
                        let at = self.transcript.len() - 1;
                        self.transcript.insert(at, (Role::Notice, text));
                    } else {
                        self.transcript.push((Role::Notice, text));
                    }
                }
                Event::Accepted(id) => {
                    if let Some(text) = self.outbox.accepted(id) {
                        self.transcript.push((Role::User, text));
                        self.transcript.push((Role::Assistant, String::new()));
                        self.answering = true;
                        self.scroll_up = 0;
                    }
                }
                Event::Stopped => {
                    // Nothing was accepted, so the message stays unsent.
                    self.outbox.stopped();
                    if self.status == Status::Generating {
                        self.status = Status::Idle;
                    }
                }
                Event::Finished {
                    tokens,
                    secs,
                    hit_limit,
                } => {
                    self.answering = false;
                    // A cancelled generation can finish after a model switch
                    // was already requested.
                    if self.status == Status::Generating {
                        self.status = Status::Idle;
                    }
                    self.last_speed = Some(format!("{:.1} tok/s", tokens as f64 / secs.max(1e-9)));
                    if hit_limit {
                        self.transcript.push((
                            Role::Notice,
                            format!("stopped at the {}-token limit", self.settings.max_tokens),
                        ));
                    }
                    self.dispatch();
                }
                Event::Error(e) => {
                    self.answering = false;
                    self.transcript.push((Role::Notice, format!("error: {e}")));
                    // A failed load leaves the previous model unusable (the
                    // worker already dropped it), so go back to the picker.
                    if matches!(self.status, Status::Loading(_)) {
                        self.status = Status::NoModel;
                        self.model_name = None;
                        self.current = None;
                        self.device = self.good_device.clone();
                        self.open_picker();
                    } else {
                        // A refused send leaves the outbox, and nothing else goes
                        // out until Enter.
                        self.outbox.refused();
                        self.status = Status::Idle;
                    }
                }
            }
        }
    }

    fn submit(&mut self) {
        let line = self.input.trim().to_string();
        self.input.clear();
        if matches!(line.as_str(), "exit" | "quit") {
            self.quit = true;
            return;
        }
        if !line.is_empty() {
            self.outbox.push(line);
        }
        // Enter also resumes sends that a stop left unsent.
        self.outbox.resume();
        self.dispatch();
    }

    // Hands the oldest unsent message to the worker, once the worker is idle.
    fn dispatch(&mut self) {
        if self.status != Status::Idle {
            return;
        }
        if let Some((id, text)) = self.outbox.next_to_send() {
            self.status = Status::Generating;
            self.scroll_up = 0;
            let _ = self.cmd_tx.send(Cmd::Send(id, text, self.settings.clone()));
        }
    }

    fn handle_key(&mut self, code: KeyCode, mods: KeyModifiers) {
        let ctrl = mods.contains(KeyModifiers::CONTROL);
        if ctrl && code == KeyCode::Char('c') {
            self.cancel.store(true, Ordering::Relaxed);
            self.quit = true;
            return;
        }
        if self.picker.is_some() {
            self.handle_picker_key(code);
            return;
        }
        match code {
            KeyCode::Char('o') if ctrl => self.open_picker(),
            KeyCode::Char('s') if ctrl => {
                self.open_picker();
                if let Some(picker) = &mut self.picker {
                    picker.focus = Focus::Settings;
                }
            }
            KeyCode::Char('d') if ctrl => self.switch_device(),
            KeyCode::Char('n') if ctrl && self.status == Status::Idle => {
                let _ = self.cmd_tx.send(Cmd::NewChat);
                self.outbox.clear();
                self.transcript.clear();
                self.transcript
                    .push((Role::Notice, "new conversation".into()));
            }
            KeyCode::Esc => {
                if self.status == Status::Generating {
                    self.cancel.store(true, Ordering::Relaxed);
                }
            }
            KeyCode::PageUp => self.scroll_up = self.scroll_up.saturating_add(8),
            KeyCode::PageDown => self.scroll_up = self.scroll_up.saturating_sub(8),
            KeyCode::Enter => self.submit(),
            KeyCode::Backspace => {
                self.input.pop();
            }
            KeyCode::Char(c) if !ctrl => self.input.push(c),
            _ => {}
        }
    }

    // Cycles through the devices this build supports and reloads the current
    // model on the new one.
    fn switch_device(&mut self) {
        let devices: &[&str] = if cfg!(feature = "cuda") {
            &["cpu", "cuda"]
        } else {
            &["cpu"]
        };
        let at = devices.iter().position(|d| *d == self.device).unwrap_or(0);
        let next = devices[(at + 1) % devices.len()];
        if next == self.device {
            self.transcript.push((
                Role::Notice,
                "only cpu is available (build with --features cuda for cuda)".into(),
            ));
            return;
        }
        self.device = next.to_string();
        if let Some(entry) = self.current.clone() {
            // Generation holds the worker; stop it so the load runs.
            self.cancel.store(true, Ordering::Relaxed);
            self.load(&entry);
        }
    }

    fn handle_picker_key(&mut self, code: KeyCode) {
        let Some(picker) = &mut self.picker else {
            return;
        };
        if code == KeyCode::Tab {
            picker.focus = match picker.focus {
                Focus::Models => Focus::Settings,
                Focus::Settings => Focus::Models,
            };
            return;
        }
        if picker.focus == Focus::Settings {
            let mut chat_toggled = false;
            let field = picker.field;
            match code {
                KeyCode::Up | KeyCode::Char('k') => {
                    picker.edit = None;
                    picker.field = (field + FIELDS - 1) % FIELDS;
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    picker.edit = None;
                    picker.field = (field + 1) % FIELDS;
                }
                KeyCode::Left | KeyCode::Char('h') | KeyCode::Char('-') => {
                    picker.edit = None;
                    chat_toggled = adjust(&mut self.settings, field, -1, false);
                }
                KeyCode::Right | KeyCode::Char('l') | KeyCode::Char('+') | KeyCode::Char('=') => {
                    picker.edit = None;
                    chat_toggled = adjust(&mut self.settings, field, 1, false);
                }
                KeyCode::PageDown => {
                    picker.edit = None;
                    chat_toggled = adjust(&mut self.settings, field, -1, true);
                }
                KeyCode::PageUp => {
                    picker.edit = None;
                    chat_toggled = adjust(&mut self.settings, field, 1, true);
                }
                // Typing into a numeric field (not the chat toggle).
                KeyCode::Char(c)
                    if field < 4 && (c.is_ascii_digit() || (c == '.' && field < 2)) =>
                {
                    picker.edit.get_or_insert_with(String::new).push(c);
                }
                KeyCode::Backspace if field < 4 => {
                    let text = picker
                        .edit
                        .take()
                        .unwrap_or_else(|| field_text(&self.settings, field));
                    picker.edit = Some(text[..text.len().saturating_sub(1)].to_string());
                }
                KeyCode::Enter if picker.edit.is_some() => {
                    if let Some(text) = picker.edit.take() {
                        apply_text(&mut self.settings, field, &text);
                    }
                }
                KeyCode::Esc if picker.edit.is_some() => picker.edit = None,
                KeyCode::Enter | KeyCode::Esc | KeyCode::Char('q') => {
                    // Opened with nothing loaded: go back to the model list.
                    if self.status == Status::NoModel {
                        picker.focus = Focus::Models;
                    } else {
                        self.picker = None;
                    }
                }
                _ => {}
            }
            if chat_toggled {
                // Old turns were built under the other mode.
                let _ = self.cmd_tx.send(Cmd::NewChat);
            }
            return;
        }
        let len = picker.entries.len();
        match code {
            KeyCode::Up | KeyCode::Char('k') if len > 0 => {
                let i = picker.state.selected().unwrap_or(0);
                picker.state.select(Some((i + len - 1) % len));
            }
            KeyCode::Down | KeyCode::Char('j') if len > 0 => {
                let i = picker.state.selected().unwrap_or(0);
                picker.state.select(Some((i + 1) % len));
            }
            KeyCode::Enter => {
                let chosen = picker
                    .state
                    .selected()
                    .and_then(|i| picker.entries.get(i).cloned());
                if let Some(entry) = chosen {
                    // Generation holds the worker; stop it so the load runs.
                    self.cancel.store(true, Ordering::Relaxed);
                    self.picker = None;
                    self.load(&entry);
                }
            }
            KeyCode::Esc | KeyCode::Char('q') => {
                self.picker = None;
                // Nothing loaded and nothing to go back to.
                if self.status == Status::NoModel {
                    self.quit = true;
                }
            }
            _ => {}
        }
    }

    fn draw(&mut self, frame: &mut Frame) {
        let unsent = self.outbox.texts().count().min(3) as u16;
        let queue_rows = if unsent == 0 { 0 } else { unsent + 2 };
        let [main, queue, input, status] = Layout::vertical([
            Constraint::Min(3),
            Constraint::Length(queue_rows),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        self.draw_transcript(frame, main);
        if queue_rows > 0 {
            self.draw_queue(frame, queue);
        }
        self.draw_input(frame, input);
        self.draw_status(frame, status);
        if let Some(picker) = &mut self.picker {
            draw_picker(frame, picker, &self.opts.models_dir, &self.settings);
        }
    }

    fn draw_transcript(&mut self, frame: &mut Frame, area: Rect) {
        let mut lines: Vec<Line> = Vec::new();
        for (role, text) in &self.transcript {
            let (label, color) = match role {
                Role::User => ("you", Color::Cyan),
                Role::Assistant => ("model", Color::Green),
                Role::Notice => ("", Color::DarkGray),
            };
            if *role == Role::Notice {
                lines.push(Line::styled(text.clone(), Style::new().fg(color)));
            } else {
                lines.push(Line::styled(
                    label,
                    Style::new().fg(color).add_modifier(Modifier::BOLD),
                ));
                lines.extend(text.split('\n').map(|l| Line::raw(l.to_string())));
            }
            lines.push(Line::raw(""));
        }
        if self.transcript.is_empty() && self.picker.is_none() && self.status == Status::NoModel {
            lines.push(Line::styled(
                "no model loaded - press Ctrl-O to pick one",
                Style::new().fg(Color::DarkGray),
            ));
        }

        let block = Block::default().borders(Borders::ALL).title(" rvllm ");
        let inner_width = area.width.saturating_sub(2);
        let inner_height = area.height.saturating_sub(2);
        let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
        let total = paragraph.line_count(inner_width) as u16;
        let max_scroll = total.saturating_sub(inner_height);
        self.scroll_up = self.scroll_up.min(max_scroll);
        frame.render_widget(
            paragraph
                .block(block)
                .scroll((max_scroll - self.scroll_up, 0)),
            area,
        );
    }

    // The messages still unsent, above the input: at most three lines.
    fn draw_queue(&self, frame: &mut Frame, area: Rect) {
        let title = if self.outbox.is_paused() {
            " unsent (Enter sends) "
        } else {
            " unsent "
        };
        let texts: Vec<&str> = self.outbox.texts().collect();
        let mut lines: Vec<Line> = texts
            .iter()
            .take(3)
            .map(|text| Line::styled(format!("you: {text}"), Style::new().fg(Color::Yellow)))
            .collect();
        if texts.len() > 3 {
            lines.truncate(2);
            lines.push(Line::styled(
                format!("... {} more", texts.len() - 2),
                Style::new().fg(Color::Yellow),
            ));
        }
        let block = Block::default().borders(Borders::ALL).title(title);
        frame.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn draw_input(&self, frame: &mut Frame, area: Rect) {
        let ready = self.status == Status::Idle;
        let title = if ready {
            " message "
        } else {
            " message (Enter queues it) "
        };
        let style = if ready {
            Style::new()
        } else {
            Style::new().fg(Color::DarkGray)
        };
        let block = Block::default().borders(Borders::ALL).title(title);
        // Keep the end of long input visible.
        let width = area.width.saturating_sub(2) as usize;
        let chars = self.input.chars().count();
        let shown: String = self
            .input
            .chars()
            .skip(chars.saturating_sub(width.saturating_sub(1)))
            .collect();
        frame.render_widget(
            Paragraph::new(shown.clone()).style(style).block(block),
            area,
        );
        if self.picker.is_none() {
            frame.set_cursor_position((area.x + 1 + shown.chars().count() as u16, area.y + 1));
        }
    }

    fn draw_status(&self, frame: &mut Frame, area: Rect) {
        let state = match &self.status {
            Status::NoModel => "no model".to_string(),
            Status::Loading(name) => format!("loading {name}..."),
            Status::Idle => "ready".into(),
            Status::Generating => "generating".into(),
        };
        let mut spans = vec![
            Span::styled(
                format!(" {state} "),
                Style::new().add_modifier(Modifier::REVERSED),
            ),
            Span::raw(format!(" {} ", self.device)),
        ];
        if let Some(name) = &self.model_name {
            spans.push(Span::raw(format!("| {name} (ctx {}) ", self.context)));
        }
        if let Some(speed) = &self.last_speed {
            spans.push(Span::raw(format!("| {speed} ")));
        }
        spans.push(Span::styled(
            "| Enter send  Esc stop  Ctrl-O models  Ctrl-S settings  Ctrl-D device  Ctrl-N new chat  PgUp/PgDn scroll  Ctrl-C quit",
            Style::new().fg(Color::DarkGray),
        ));
        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }
}

fn draw_picker(frame: &mut Frame, picker: &mut Picker, dir: &Path, settings: &Settings) {
    let area = frame.area();
    let width = area.width.saturating_sub(8).clamp(20, 90);
    let list_rows = (picker.entries.len() as u16).max(1);
    let height = (list_rows + FIELDS as u16 + 6).min(area.height);
    let rect = Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width: width.min(area.width),
        height,
    };
    frame.render_widget(Clear, rect);
    let outer = Block::default()
        .borders(Borders::ALL)
        .title(" Tab switch panel, Enter load/close, Esc close ");
    let inner = outer.inner(rect);
    frame.render_widget(outer, rect);
    let [models_area, settings_area] =
        Layout::vertical([Constraint::Min(2), Constraint::Length(FIELDS as u16 + 2)]).areas(inner);

    let focused = |on: bool| {
        if on {
            Style::new().fg(Color::Cyan)
        } else {
            Style::new()
        }
    };
    let models_block = Block::default()
        .borders(Borders::ALL)
        .border_style(focused(picker.focus == Focus::Models))
        .title(" models ");
    if picker.entries.is_empty() {
        let msg = format!("no .gguf files under {}", dir.display());
        frame.render_widget(
            Paragraph::new(msg)
                .wrap(Wrap { trim: true })
                .block(models_block),
            models_area,
        );
    } else {
        let items: Vec<ListItem> = picker
            .entries
            .iter()
            .map(|e| {
                let shown = e.path.strip_prefix(dir).unwrap_or(&e.path).display();
                ListItem::new(format!("{shown}  ({})", human_size(e.size)))
            })
            .collect();
        let list = List::new(items)
            .block(models_block)
            .highlight_style(Style::new().add_modifier(Modifier::REVERSED))
            .highlight_symbol("> ");
        frame.render_stateful_widget(list, models_area, &mut picker.state);
    }

    let temp = settings.sampling.temperature;
    let mut rows = [
        format!(
            "temperature  {temp:.1}{}",
            if temp == 0.0 { "  (greedy)" } else { "" }
        ),
        format!("top_p        {:.2}", settings.sampling.top_p),
        format!("max tokens   {}", settings.max_tokens),
        format!("seed         {}", settings.sampling.seed),
        format!("chat format  {}", if settings.chat { "on" } else { "off" }),
    ];
    if let Some(text) = &picker.edit {
        let label = [
            "temperature  ",
            "top_p        ",
            "max tokens   ",
            "seed         ",
        ];
        if let Some(l) = label.get(picker.field) {
            rows[picker.field] = format!("{l}{text}_  (Enter apply, Esc cancel)");
        }
    }
    let lines: Vec<Line> = rows
        .into_iter()
        .enumerate()
        .map(|(i, row)| {
            if picker.focus == Focus::Settings && i == picker.field {
                Line::styled(
                    format!("> {row}"),
                    Style::new().add_modifier(Modifier::REVERSED),
                )
            } else {
                Line::raw(format!("  {row}"))
            }
        })
        .collect();
    let settings_block = Block::default()
        .borders(Borders::ALL)
        .border_style(focused(picker.focus == Focus::Settings))
        .title(" settings (Up/Down field, Left/Right +-1, PgUp/PgDn big step, or type) ");
    frame.render_widget(Paragraph::new(lines).block(settings_block), settings_area);
}

pub fn run(opts: TuiOptions) -> Result<(), BoxError> {
    let opts = Arc::new(opts);
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (event_tx, events) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));

    let mut app = App {
        opts: opts.clone(),
        cmd_tx,
        events,
        cancel: cancel.clone(),
        status: Status::NoModel,
        settings: Settings {
            sampling: opts.sampling,
            max_tokens: opts.max_tokens,
            chat: opts.chat,
        },
        device: opts.device.clone(),
        good_device: opts.device.clone(),
        current: None,
        model_name: None,
        context: 0,
        last_speed: None,
        transcript: Vec::new(),
        input: String::new(),
        scroll_up: 0,
        picker: None,
        quit: false,
        outbox: Outbox::default(),
        answering: false,
    };
    match &opts.initial_model {
        Some(path) => app.load(&ModelEntry {
            tokenizer: tokenizer::path_for(path, &opts.fallback_tokenizer),
            path: path.clone(),
            size: 0,
        }),
        None => app.open_picker(),
    }

    let worker = std::thread::spawn(move || {
        run_worker(
            WorkerCtx {
                opts,
                tx: event_tx,
                cancel,
            },
            cmd_rx,
        )
    });

    let mut terminal = ratatui::init();
    let result = event_loop(&mut terminal, &mut app);
    ratatui::restore();

    // Hanging up the command channel ends the worker; the cancel flag cuts
    // short a generation in flight.
    app.cancel.store(true, Ordering::Relaxed);
    drop(app);
    let _ = worker.join();
    result
}

fn event_loop(terminal: &mut DefaultTerminal, app: &mut App) -> Result<(), BoxError> {
    while !app.quit {
        app.drain_events();
        terminal.draw(|f| app.draw(f))?;
        if event::poll(Duration::from_millis(50))? {
            if let TermEvent::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    app.handle_key(key.code, key.modifiers);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::engine::testing::CountingBackend;
    use crate::model::{BatchItem, Config};

    // A backend whose forward pass fails from the first step on, and that
    // records which sequences were released.
    struct FailingBackend {
        config: Config,
        released: Arc<Mutex<Vec<SequenceId>>>,
    }

    impl engine::Backend for FailingBackend {
        fn config(&self) -> &Config {
            &self.config
        }

        fn forward_sample(
            &mut self,
            _items: &[BatchItem],
            _last_only: bool,
            _samplers: &mut [&mut Sampler],
        ) -> Result<Vec<u32>, BoxError> {
            Err("device lost".into())
        }

        fn release(&mut self, seq_id: SequenceId) {
            self.released.lock().unwrap().push(seq_id);
        }
    }

    fn engine_config() -> EngineConfig {
        EngineConfig {
            block_size: 16,
            num_blocks: 64,
            max_running: 4,
        }
    }

    fn request(tokens: Vec<u32>, max_tokens: usize) -> Request {
        Request {
            tokens,
            max_tokens,
            stop_tokens: Vec::new(),
            sampler: Sampler::from_params(SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                seed: 0,
            }),
        }
    }

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_file("tokenizer.json", 0, 0).unwrap()
    }

    fn empty_reply() -> Reply {
        Reply::default()
    }

    fn options() -> TuiOptions {
        TuiOptions {
            device: "cpu".into(),
            block_size: 16,
            num_blocks: Some(64),
            max_tokens: 8,
            chat: true,
            system: None,
            sampling: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                seed: 0,
            },
            fallback_tokenizer: "tokenizer.json".into(),
            models_dir: ".".into(),
            initial_model: None,
            compact: Strategy::Summarize,
        }
    }

    #[test]
    fn a_failed_decode_aborts_the_request() {
        let released = Arc::new(Mutex::new(Vec::new()));
        let backend = FailingBackend {
            config: Config {
                context_length: 8192,
                eos_token_id: 999,
                vocab_size: 1000,
                ..Config::for_tests()
            },
            released: released.clone(),
        };
        let mut engine = Engine::new(backend, engine_config());
        let cancel = AtomicBool::new(false);
        let result = run_request(
            &mut engine,
            &tokenizer(),
            &cancel,
            request(vec![1, 2, 3], 3),
            false,
            |_| true,
        );
        assert!(result.is_err());
        assert!(!engine.has_work());
        assert_eq!(released.lock().unwrap().len(), 1);
    }

    #[test]
    fn decode_counts_only_its_own_sequence() {
        let mut engine = Engine::new(CountingBackend::new(8192), engine_config());
        let own = engine.add_request(request(vec![1, 2, 3], 3)).unwrap();
        engine.add_request(request(vec![5, 6], 10)).unwrap();
        let cancel = AtomicBool::new(false);
        let mut reply = empty_reply();
        let mut pieces = 0;
        decode_request(
            &mut engine,
            &tokenizer(),
            &cancel,
            own,
            &mut reply,
            &mut |_| {
                pieces += 1;
                true
            },
        )
        .unwrap();
        assert_eq!(reply.tokens, 3);
        assert!(reply.hit_limit);
        assert_eq!(pieces, 3);
    }

    #[test]
    fn a_cancelled_request_reports_no_reply() {
        let mut engine = Engine::new(CountingBackend::new(8192), engine_config());
        let id = engine.add_request(request(vec![1, 2], 5)).unwrap();
        let cancel = AtomicBool::new(true);
        let mut reply = empty_reply();
        decode_request(
            &mut engine,
            &tokenizer(),
            &cancel,
            id,
            &mut reply,
            &mut |_| true,
        )
        .unwrap();
        assert!(reply.cancelled);
        assert_eq!(reply.tokens, 0);
        assert!(!engine.has_work());
    }

    #[test]
    fn a_cancelled_summary_is_reported_as_cancelled() {
        let (tx, _rx) = mpsc::channel::<Event>();
        let ctx = WorkerCtx {
            opts: Arc::new(options()),
            tx,
            cancel: Arc::new(AtomicBool::new(true)),
        };
        let mut engine = Engine::new(CountingBackend::new(8192), engine_config());
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: "hi".into(),
        }];
        let error = summarize(
            &ctx,
            &mut engine,
            &tokenizer(),
            ChatFormat::SMOLLM2,
            &messages,
            8,
        )
        .unwrap_err();
        assert!(error.is::<history::Cancelled>());
    }

    #[test]
    fn outbox_hands_out_one_message_at_a_time() {
        let mut outbox = Outbox::default();
        outbox.push("a".into());
        outbox.push("b".into());
        let (id_a, text) = outbox.next_to_send().unwrap();
        assert_eq!(text, "a");
        assert!(outbox.next_to_send().is_none());
        assert_eq!(outbox.accepted(id_a).as_deref(), Some("a"));
        let next = outbox.next_to_send().map(|(_, text)| text);
        assert_eq!(next.as_deref(), Some("b"));
    }

    #[test]
    fn outbox_ignores_acceptance_of_another_message() {
        let mut outbox = Outbox::default();
        outbox.push("a".into());
        let (id, _) = outbox.next_to_send().unwrap();
        assert!(outbox.accepted(id + 1).is_none());
        assert!(outbox.next_to_send().is_none());
    }

    #[test]
    fn outbox_stays_unsent_when_stopped_until_resumed() {
        let mut outbox = Outbox::default();
        outbox.push("a".into());
        outbox.next_to_send().unwrap();
        outbox.stopped();
        assert!(outbox.next_to_send().is_none());
        assert_eq!(outbox.texts().collect::<Vec<_>>(), vec!["a"]);
        outbox.resume();
        assert!(outbox.next_to_send().is_some());
    }

    #[test]
    fn outbox_drops_a_refused_message_and_pauses() {
        let mut outbox = Outbox::default();
        outbox.push("a".into());
        outbox.push("b".into());
        outbox.next_to_send().unwrap();
        assert!(outbox.refused());
        assert!(!outbox.refused());
        assert!(outbox.next_to_send().is_none());
        assert_eq!(outbox.texts().collect::<Vec<_>>(), vec!["b"]);
    }

    #[test]
    fn answer_reserve_is_at_most_a_quarter_of_the_pool() {
        assert_eq!(answer_reserve(256, 4096), 256);
        assert_eq!(answer_reserve(2048, 4096), 1024);
        assert_eq!(answer_reserve(300, 256), 64);
        assert_eq!(answer_reserve(0, 256), 1);
    }

    #[test]
    fn a_full_pool_stops_the_segment_with_its_tokens() {
        // 4 blocks of 4 slots: from a 3-token prompt, 14 tokens fill the pool.
        let mut engine = Engine::new(
            CountingBackend::new(8192),
            EngineConfig {
                block_size: 4,
                num_blocks: 4,
                max_running: 4,
            },
        );
        let id = engine
            .add_continuable_request(request(vec![1, 2, 3], 40))
            .unwrap();
        let cancel = AtomicBool::new(false);
        let mut reply = empty_reply();
        decode_request(
            &mut engine,
            &tokenizer(),
            &cancel,
            id,
            &mut reply,
            &mut |_| true,
        )
        .unwrap();
        assert!(reply.pool_full);
        assert!(!reply.hit_limit);
        assert_eq!(reply.ids.len(), 14);
        assert_eq!(reply.tokens, 14);
    }

    #[test]
    fn compaction_note_names_each_outcome() {
        let summarized = history::Compaction {
            summarized: 6,
            ..Default::default()
        };
        assert_eq!(
            compaction_note(&summarized),
            "history over the prompt budget: summarized 6 earlier messages"
        );
        let failed = history::Compaction {
            dropped: 2,
            summary_dropped: true,
            summary_error: Some("the summary request doesn't fit the pool".into()),
            ..Default::default()
        };
        assert_eq!(
            compaction_note(&failed),
            "history over the prompt budget: dropped 2 earlier messages, dropped the summary, couldn't summarize: the summary request doesn't fit the pool"
        );
    }
}

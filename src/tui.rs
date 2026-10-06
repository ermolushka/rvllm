// Full-screen terminal UI for `rvllm cli --tui`: a model picker plus a chat
// view with streamed output.
//
// The engine (and the model/runtime it borrows) lives on a worker thread so
// the UI never blocks on a weight load or a decode step, and device state
// (CUDA context) never crosses threads. The two sides talk over channels:
// `Cmd` goes to the worker, `Event` comes back. Cancelling a generation uses a
// shared flag the worker checks between decode steps.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use ratatui::crossterm::event::{self, Event as TermEvent, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::{DefaultTerminal, Frame};

use crate::chat::{self, ChatMessage};
use crate::engine::{self, BoxError, Engine, EngineConfig, Request};
use crate::model::Model;
use crate::runtime::Runtime;
use crate::sampler::{Sampler, SamplingParams};
use crate::tokenizer::{SmollLM230MTokenizer, StreamDecoder};

pub struct TuiOptions {
    pub device: String,
    pub block_size: usize,
    pub num_blocks: Option<usize>,
    pub max_tokens: usize,
    pub chat: bool,
    pub sampling: SamplingParams,
    // Used for models with no `tokenizer.json` next to them.
    pub fallback_tokenizer: PathBuf,
    // Where the picker looks for `.gguf` files.
    pub models_dir: PathBuf,
    // Load this model right away instead of opening the picker.
    pub initial_model: Option<PathBuf>,
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

fn tokenizer_for(model: &Path, fallback: &Path) -> PathBuf {
    let sibling = model.with_file_name("tokenizer.json");
    if sibling.exists() {
        sibling
    } else {
        fallback.to_path_buf()
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
            tokenizer: tokenizer_for(&path, fallback),
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
    Send(String, Settings),
    NewChat,
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
    let tokenizer = SmollLM230MTokenizer::from_file(
        &tokenizer_path.to_string_lossy(),
        model.config.eos_token_id,
        model.config.bos_token_id,
    )?;
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
    let mut stops = vec![tokenizer.eos_token_id];
    stops.extend(tokenizer.token_to_id(chat::END_OF_TURN));
    let _ = ctx.tx.send(Event::Ready { name, context });

    let mut history: Vec<ChatMessage> = Vec::new();
    loop {
        match rx.recv() {
            Ok(Cmd::NewChat) => history.clear(),
            Ok(load @ Cmd::Load { .. }) => return Ok(Some(load)),
            Ok(Cmd::Send(line, settings)) => {
                if let Err(e) = generate(
                    ctx,
                    &mut engine,
                    &tokenizer,
                    &stops,
                    &mut history,
                    line,
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
    tokenizer: &SmollLM230MTokenizer,
    stops: &[u32],
    history: &mut Vec<ChatMessage>,
    line: String,
    settings: &Settings,
) -> Result<(), BoxError> {
    let chat = settings.chat;
    let prompt = if chat {
        history.push(ChatMessage {
            role: "user".into(),
            content: line,
        });
        chat::chatml(history)
    } else {
        line
    };
    let request = Request {
        tokens: tokenizer.encode(&prompt)?,
        max_tokens: settings.max_tokens,
        stop_tokens: stops.to_vec(),
        sampler: Sampler::from_params(settings.sampling),
    };
    let id = match engine.add_request(request) {
        Ok(id) => id,
        Err(e) => {
            if chat {
                history.pop();
            }
            return Err(e.into());
        }
    };

    ctx.cancel.store(false, Ordering::Relaxed);
    let started = Instant::now();
    let mut decoder = StreamDecoder::new();
    let mut answer = String::new();
    let mut tokens = 0;
    let mut hit_limit = false;
    'generate: while engine.has_work() {
        if ctx.cancel.load(Ordering::Relaxed) {
            engine.abort(id);
            break;
        }
        for ev in engine.step()? {
            tokens += 1;
            let stopped = ev.finish == Some(engine::FinishReason::Stop);
            let text = if stopped {
                decoder.flush(tokenizer)?
            } else {
                decoder.push(tokenizer, ev.token)?
            };
            answer.push_str(&text);
            if ctx.tx.send(Event::Token(text)).is_err() {
                engine.abort(id);
                break 'generate;
            }
            hit_limit = ev.finish == Some(engine::FinishReason::Length);
            if ev.finish.is_some() {
                break 'generate;
            }
        }
    }
    if chat {
        history.push(ChatMessage {
            role: "assistant".into(),
            content: answer,
        });
    }
    let _ = ctx.tx.send(Event::Finished {
        tokens,
        secs: started.elapsed().as_secs_f64(),
        hit_limit,
    });
    Ok(())
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
                Event::Finished {
                    tokens,
                    secs,
                    hit_limit,
                } => {
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
                }
                Event::Error(e) => {
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
                        self.status = Status::Idle;
                    }
                }
            }
        }
    }

    fn submit(&mut self) {
        let line = self.input.trim().to_string();
        self.input.clear();
        if line.is_empty() {
            return;
        }
        if matches!(line.as_str(), "exit" | "quit") {
            self.quit = true;
            return;
        }
        self.transcript.push((Role::User, line.clone()));
        self.transcript.push((Role::Assistant, String::new()));
        self.scroll_up = 0;
        self.status = Status::Generating;
        let _ = self.cmd_tx.send(Cmd::Send(line, self.settings.clone()));
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
            KeyCode::Enter if self.status == Status::Idle => self.submit(),
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
        let [main, input, status] = Layout::vertical([
            Constraint::Min(3),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .areas(frame.area());

        self.draw_transcript(frame, main);
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

    fn draw_input(&self, frame: &mut Frame, area: Rect) {
        let ready = self.status == Status::Idle;
        let title = if ready {
            " message "
        } else {
            " message (busy) "
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
        if ready && self.picker.is_none() {
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
    };
    match &opts.initial_model {
        Some(path) => app.load(&ModelEntry {
            tokenizer: tokenizer_for(path, &opts.fallback_tokenizer),
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

// What the HTTP routes talk to. Holds the current model's `AppState` and, when
// the server was started with a models directory, can swap it for another
// model at runtime (the chat UI's model picker). Also serves the UI itself.
//
// A swap is sequential on purpose: the old engine is drained and its thread
// joined *before* the new model is loaded, so two models never sit in GPU
// memory together. Requests that arrive meanwhile get a 503.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::thread::JoinHandle;

use axum::extract::{Json, State};
use axum::http::{HeaderValue, header};
use axum::response::{Html, IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{ApiError, AppState};

const INDEX_HTML: &str = include_str!("ui.html");

// A running model: its request-facing state plus the engine thread behind it.
pub struct Active {
    pub state: Arc<AppState>,
    pub thread: Option<JoinHandle<()>>,
    pub path: PathBuf,
}

// Loads a model and starts its engine thread. Blocking (reads the weights).
pub type Loader = Box<dyn Fn(&Path) -> Result<Active, String> + Send + Sync>;

pub struct Switcher {
    pub loader: Loader,
    pub models_dir: PathBuf,
    pub device: String,
    // Held for the whole swap, so two can't overlap.
    lock: tokio::sync::Mutex<()>,
}

impl Switcher {
    pub fn new(loader: Loader, models_dir: PathBuf, device: String) -> Self {
        Switcher {
            loader,
            models_dir,
            device,
            lock: tokio::sync::Mutex::new(()),
        }
    }
}

pub struct Hub {
    current: RwLock<Option<Active>>,
    // True while a model is being loaded or unloaded. With no model and this
    // false, the server is deliberately unloaded.
    switching: AtomicBool,
    switcher: Option<Switcher>,
}

impl Hub {
    // A hub that serves one model for good (tests, library users).
    pub fn fixed(state: Arc<AppState>) -> Arc<Self> {
        Arc::new(Hub {
            current: RwLock::new(Some(Active {
                state,
                thread: None,
                path: PathBuf::new(),
            })),
            switching: AtomicBool::new(false),
            switcher: None,
        })
    }

    pub fn switchable(active: Active, switcher: Switcher) -> Arc<Self> {
        Arc::new(Hub {
            current: RwLock::new(Some(active)),
            switching: AtomicBool::new(false),
            switcher: Some(switcher),
        })
    }

    // The model to serve a request with.
    pub fn state(&self) -> Result<Arc<AppState>, ApiError> {
        self.current
            .read()
            .unwrap()
            .as_ref()
            .map(|a| a.state.clone())
            .ok_or_else(|| {
                if self.switching.load(Ordering::Relaxed) {
                    ApiError::loading()
                } else {
                    ApiError::unloaded()
                }
            })
    }

    // Stops the engine and waits for it. Called once the server has shut down.
    pub fn shutdown(&self) {
        let active = self.current.write().unwrap().take();
        join(active);
    }

    fn is_switching(&self) -> bool {
        self.switching.load(Ordering::Relaxed)
    }
}

// Drops the request-facing state (closing the job channel) and waits for the
// engine thread, which first finishes whatever is still in flight.
fn join(active: Option<Active>) {
    if let Some(Active { state, thread, .. }) = active {
        drop(state);
        if let Some(t) = thread {
            let _ = t.join();
        }
    }
}

// `.gguf` files in `dir` and one level of subdirectories, sorted by path.
pub fn discover(dir: &Path) -> Vec<PathBuf> {
    fn scan(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && depth > 0 {
                scan(&path, depth - 1, out);
            } else if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    scan(dir, 1, &mut out);
    out.sort();
    out
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

fn display(p: &Path) -> String {
    p.canonicalize().unwrap_or_else(|_| p.to_path_buf()).to_string_lossy().into_owned()
}

fn info(hub: &Hub) -> Value {
    let current = hub.current.read().unwrap();
    let model = current.as_ref().map(|a| {
        let cfg = &a.state.cfg;
        json!({
            "name": cfg.model_name,
            "path": display(&a.path),
            "context_length": cfg.context_length,
            "max_sequence_tokens": cfg.max_sequence_tokens,
            "chat": a.state.chat_format.is_some(),
            "defaults": {
                "temperature": cfg.default_temperature,
                "top_p": cfg.default_top_p,
                "max_tokens": cfg.default_max_tokens,
            },
        })
    });
    let available: Vec<Value> = hub
        .switcher
        .as_ref()
        .map(|s| {
            discover(&s.models_dir)
                .into_iter()
                .map(|p| {
                    let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                    json!({
                        "name": p.file_stem().map(|s| s.to_string_lossy().into_owned()),
                        "path": display(&p),
                        "size_bytes": size,
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    json!({
        "loading": current.is_none() && hub.is_switching(),
        "unloaded": current.is_none() && !hub.is_switching(),
        "can_switch": hub.switcher.is_some(),
        "device": hub.switcher.as_ref().map(|s| s.device.clone()),
        "model": model,
        "available": available,
    })
}

pub async fn index() -> Response {
    let mut resp = Html(INDEX_HTML).into_response();
    resp.headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    resp
}

pub async fn models_info(State(hub): State<Arc<Hub>>) -> Json<Value> {
    Json(info(&hub))
}

#[derive(Deserialize)]
pub struct LoadRequest {
    path: String,
}

pub async fn load_model(
    State(hub): State<Arc<Hub>>,
    Json(req): Json<LoadRequest>,
) -> Result<Json<Value>, ApiError> {
    let Some(switcher) = &hub.switcher else {
        return Err(ApiError::bad_request(
            "this server was not started with a models directory (--models-dir)",
        ));
    };
    // Only files the picker lists: the endpoint must not load arbitrary paths.
    let requested = PathBuf::from(&req.path);
    let Some(path) = discover(&switcher.models_dir)
        .into_iter()
        .find(|p| same_file(p, &requested))
    else {
        return Err(ApiError::bad_request("not a model in the models directory"));
    };
    let Ok(_guard) = switcher.lock.try_lock() else {
        return Err(ApiError::conflict("a model is already being loaded"));
    };
    if hub
        .current
        .read()
        .unwrap()
        .as_ref()
        .is_some_and(|a| same_file(&a.path, &path))
    {
        return Ok(Json(info(&hub)));
    }

    hub.switching.store(true, Ordering::Relaxed);
    let previous = hub.current.write().unwrap().take();
    let previous_path = previous.as_ref().map(|a| a.path.clone());

    let hub2 = hub.clone();
    let target = path.clone();
    let result = tokio::task::spawn_blocking(move || {
        let switcher = hub2.switcher.as_ref().unwrap();
        join(previous);
        match (switcher.loader)(&target) {
            Ok(active) => Ok(active),
            Err(e) => {
                // Put the old model back so a bad file doesn't leave the
                // server empty.
                if let Some(old) = previous_path
                    && let Ok(active) = (switcher.loader)(&old)
                {
                    *hub2.current.write().unwrap() = Some(active);
                }
                Err(e)
            }
        }
    })
    .await
    .map_err(|e| {
        hub.switching.store(false, Ordering::Relaxed);
        ApiError::internal(format!("model switch panicked: {e}"))
    })?;

    if result.is_err() {
        hub.switching.store(false, Ordering::Relaxed);
    }
    match result {
        Ok(active) => {
            *hub.current.write().unwrap() = Some(active);
            hub.switching.store(false, Ordering::Relaxed);
            Ok(Json(info(&hub)))
        }
        Err(e) => Err(ApiError::bad_request(format!("could not load model: {e}"))),
    }
}

pub fn health_fields(hub: &Hub) -> Value {
    if hub.is_switching() {
        json!({"status": "loading"})
    } else {
        json!({"status": "unloaded"})
    }
}

// Frees the model (engine thread, weights, KV pool) without loading another.
pub async fn unload_model(State(hub): State<Arc<Hub>>) -> Result<Json<Value>, ApiError> {
    let Some(switcher) = &hub.switcher else {
        return Err(ApiError::bad_request(
            "this server was not started with a models directory (--models-dir)",
        ));
    };
    let Ok(_guard) = switcher.lock.try_lock() else {
        return Err(ApiError::conflict("a model is already being loaded"));
    };
    hub.switching.store(true, Ordering::Relaxed);
    let previous = hub.current.write().unwrap().take();
    let joined = tokio::task::spawn_blocking(move || join(previous)).await;
    hub.switching.store(false, Ordering::Relaxed);
    joined.map_err(|e| ApiError::internal(format!("unload panicked: {e}")))?;
    Ok(Json(info(&hub)))
}

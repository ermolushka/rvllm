// The engine thread. HTTP handlers (async, many) hand it `Job`s over a channel
// and get `WorkerEvent`s back; this thread is the only one that touches the
// model, so the engine needs no locking. While requests are in flight it
// steps continuously (that is the continuous batching: jobs that arrive
// mid-flight join the very next step); when idle it sleeps on the channel.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use kv_cache_scheduler::sequence::SequenceId;
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::engine::{Backend, Engine, FinishReason, Request};
use crate::sampler::{Sampler, SamplingParams};

pub struct Job {
    pub tokens: Vec<u32>,
    pub max_tokens: usize,
    pub stop_tokens: Vec<u32>,
    pub sampling: SamplingParams,
    // A chat segment: when the pool fills it ends with `FinishReason::PoolFull`,
    // so the caller can compact and continue it.
    pub continuable: bool,
    pub events: UnboundedSender<WorkerEvent>,
    // Held while the job is queued or running, so that a full queue is refused
    // at once. Releasing it frees a slot for the next request. Summaries, which
    // belong to a request that already holds its slot, carry none.
    pub permit: Option<OwnedSemaphorePermit>,
}

pub enum WorkerEvent {
    Token {
        token: u32,
        finish: Option<FinishReason>,
    },
    Error(String),
}

// Live engine occupancy, for /health.
#[derive(Default)]
pub struct WorkerStatus {
    pub running: AtomicUsize,
    pub waiting: AtomicUsize,
}

struct Live {
    events: UnboundedSender<WorkerEvent>,
    permit: Option<OwnedSemaphorePermit>,
}

// Runs until every `Job` sender is dropped and all in-flight work is done.
pub fn run_worker<B: Backend>(
    mut engine: Engine<B>,
    mut jobs: UnboundedReceiver<Job>,
    status: Arc<WorkerStatus>,
) {
    let mut live: HashMap<SequenceId, Live> = HashMap::new();

    let admit = |engine: &mut Engine<B>, live: &mut HashMap<SequenceId, Live>, job: Job| {
        let request = Request {
            tokens: job.tokens,
            max_tokens: job.max_tokens,
            stop_tokens: job.stop_tokens,
            sampler: Sampler::from_params(job.sampling),
        };
        let added = if job.continuable {
            engine.add_continuable_request(request)
        } else {
            engine.add_request(request)
        };
        match added {
            Ok(id) => {
                live.insert(
                    id,
                    Live {
                        events: job.events,
                        permit: job.permit,
                    },
                );
            }
            // The HTTP layer pre-validates, so this is a safety net.
            Err(e) => {
                let _ = job.events.send(WorkerEvent::Error(e.to_string()));
            }
        }
    };

    loop {
        if !engine.has_work() {
            match jobs.blocking_recv() {
                Some(job) => admit(&mut engine, &mut live, job),
                None => break,
            }
        }
        while let Ok(job) = jobs.try_recv() {
            admit(&mut engine, &mut live, job);
        }

        // Clients that went away (dropped their receiver) stop costing
        // anything: free their KV blocks and their queue slot now.
        let gone: Vec<SequenceId> = live
            .iter()
            .filter(|(_, l)| l.events.is_closed())
            .map(|(&id, _)| id)
            .collect();
        for id in gone {
            engine.abort(id);
            live.remove(&id);
        }

        match engine.step() {
            Ok(events) => {
                for ev in events {
                    if ev.finish.is_some() {
                        // Release the queue slot *before* the final event goes
                        // out: once the client has its response, an immediate
                        // follow-up request must not see the slot still taken.
                        let Some(Live { events, permit }) = live.remove(&ev.id) else {
                            continue;
                        };
                        drop(permit);
                        let _ = events.send(WorkerEvent::Token {
                            token: ev.token,
                            finish: ev.finish,
                        });
                        continue;
                    }
                    let Some(l) = live.get(&ev.id) else { continue };
                    let delivered = l
                        .events
                        .send(WorkerEvent::Token {
                            token: ev.token,
                            finish: None,
                        })
                        .is_ok();
                    if !delivered {
                        engine.abort(ev.id);
                        live.remove(&ev.id);
                    }
                }
            }
            Err(e) => {
                // A failed forward pass leaves the KV pool in an unknown state,
                // so fail everything in flight rather than limp on.
                eprintln!("engine step failed: {e}");
                for (id, l) in live.drain() {
                    let _ = l
                        .events
                        .send(WorkerEvent::Error(format!("engine error: {e}")));
                    engine.abort(id);
                }
            }
        }
        status
            .running
            .store(engine.num_running(), Ordering::Relaxed);
        status
            .waiting
            .store(engine.num_waiting(), Ordering::Relaxed);
    }
}

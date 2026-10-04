// Continuous-batching engine, shared by the CLI (`main.rs`), the HTTP server
// (`server/`) and the benchmark harness (`bench.rs`).
//
// `Engine` is steppable: callers `add_request` whenever a request shows up,
// call `step` in a loop, and get back the tokens produced by that step. Each
// step admits waiting requests (one prefill forward each, FIFO, up to
// `max_running` concurrent sequences and as far as the KV block pool allows),
// then runs one batched decode forward over every running sequence.
// `run` below wraps that loop for the fixed-request-set callers (bench, CLI
// one-shot).
//
// Everything device-specific lives behind `Backend`: one forward pass plus
// sampling. `CpuBackend` is here; the CUDA one is in `cuda/engine.rs`.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use kv_cache_scheduler::block_pool::BlockID;
use kv_cache_scheduler::sequence::{Scheduler, SequenceId, TokenId};

use crate::kv_storage::KvStorage;
use crate::model::{BatchItem, Config, Model};
use crate::sampler::{Sampler, SamplingParams};

pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

// One forward pass over a batch, plus picking each row's next token.
pub trait Backend {
    fn config(&self) -> &Config;

    // Returns one sampled token per item, using that item's own sampler.
    // `last_only` is true for prefill (a single item, many tokens), where only
    // the final position's logits are needed; false for a decode step (one
    // token per item).
    fn forward_sample(
        &mut self,
        items: &[BatchItem],
        last_only: bool,
        samplers: &mut [&mut Sampler],
    ) -> Result<Vec<u32>, BoxError>;
}

impl<B: Backend + ?Sized> Backend for Box<B> {
    fn config(&self) -> &Config {
        (**self).config()
    }

    fn forward_sample(
        &mut self,
        items: &[BatchItem],
        last_only: bool,
        samplers: &mut [&mut Sampler],
    ) -> Result<Vec<u32>, BoxError> {
        (**self).forward_sample(items, last_only, samplers)
    }
}

pub struct CpuBackend<'a> {
    model: &'a Model,
    kv_storage: KvStorage,
}

impl<'a> CpuBackend<'a> {
    pub fn new(model: &'a Model, block_size: usize, num_blocks: usize) -> Result<Self, BoxError> {
        let config = &model.config;
        let kv_storage = KvStorage::new(
            config.n_layers as usize,
            num_blocks,
            block_size,
            config.n_kv_heads as usize,
            config.head_dim(),
            model.device(),
        )?;
        Ok(CpuBackend { model, kv_storage })
    }
}

impl Backend for CpuBackend<'_> {
    fn config(&self) -> &Config {
        &self.model.config
    }

    fn forward_sample(
        &mut self,
        items: &[BatchItem],
        last_only: bool,
        samplers: &mut [&mut Sampler],
    ) -> Result<Vec<u32>, BoxError> {
        let logits = if last_only {
            self.model.forward_last(items, &mut self.kv_storage)?
        } else {
            self.model.forward(items, &mut self.kv_storage)?
        }; // [batch, 1, vocab_size]
        let mut next = Vec::with_capacity(items.len());
        for (i, sampler) in samplers.iter_mut().enumerate() {
            next.push(sampler.sample(&logits.get(i)?.get(0)?)?);
        }
        Ok(next)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    pub block_size: usize,
    pub num_blocks: usize,
    // Cap on concurrently running (decoding) sequences; the rest wait in the
    // FIFO queue. `usize::MAX` = limited only by the block pool.
    pub max_running: usize,
}

pub struct Request {
    pub tokens: Vec<u32>,
    pub max_tokens: usize,
    // Generation ends after emitting any of these (the stop token itself is
    // still reported in the final event).
    pub stop_tokens: Vec<u32>,
    pub sampler: Sampler,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

pub struct TokenEvent {
    pub id: SequenceId,
    pub token: u32,
    pub finish: Option<FinishReason>,
    // Set on a request's first token: how long its prefill forward took.
    pub ttft: Option<Duration>,
}

#[derive(Debug)]
pub enum RequestError {
    EmptyPrompt,
    ZeroMaxTokens,
    // Prompt alone fills the model's context window.
    PromptTooLong { prompt: usize, context: usize },
    // Prompt + generation would need more KV blocks than the whole pool has,
    // so it could never run even alone.
    ExceedsKvCapacity { tokens: usize, capacity: usize },
}

impl fmt::Display for RequestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RequestError::EmptyPrompt => write!(f, "prompt is empty"),
            RequestError::ZeroMaxTokens => write!(f, "max_tokens must be at least 1"),
            RequestError::PromptTooLong { prompt, context } => write!(
                f,
                "prompt is {prompt} tokens, which does not fit the model's {context}-token context"
            ),
            RequestError::ExceedsKvCapacity { tokens, capacity } => write!(
                f,
                "request needs {tokens} tokens of KV cache but the pool only holds {capacity}"
            ),
        }
    }
}

impl std::error::Error for RequestError {}

#[derive(Default, Clone)]
pub struct EngineStats {
    // Wall time spent inside prefill forward calls / batched decode steps
    // (incl. sampling), and how many tokens each processed.
    pub prefill_time: Duration,
    pub prefill_tokens: usize,
    pub decode_time: Duration,
    pub decode_steps: usize,
}

// Everything the engine tracks per sequence, keyed by the scheduler's id.
struct SeqState {
    // Length of the original prompt, before any generated tokens were appended.
    prompt_len: usize,
    max_tokens: usize,
    stop_tokens: Vec<u32>,
    num_generated: usize,
    // Most recently sampled token: the input of the next decode step (it is
    // emitted as soon as it's sampled, but not in the KV cache until then).
    last_token: u32,
    sampler: Sampler,
}

// One running sequence's contribution to a decode step. Owned here so the
// `BatchItem`s can borrow from it; the block table is borrowed straight from
// the scheduler instead of copied.
struct DecodeRow {
    seq_id: SequenceId,
    token: [u32; 1],
    write: [(BlockID, usize); 1],
    position_offset: usize,
}

pub struct Engine<B: Backend> {
    backend: B,
    cfg: EngineConfig,
    scheduler: Scheduler,
    seqs: HashMap<SequenceId, SeqState>,
    pub stats: EngineStats,
}

impl<B: Backend> Engine<B> {
    pub fn new(backend: B, cfg: EngineConfig) -> Self {
        Engine {
            backend,
            cfg,
            scheduler: Scheduler::new(cfg.num_blocks as u32, cfg.block_size),
            seqs: HashMap::new(),
            stats: EngineStats::default(),
        }
    }

    pub fn config(&self) -> &Config {
        self.backend.config()
    }

    pub fn engine_config(&self) -> &EngineConfig {
        &self.cfg
    }

    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    // Longest sequence (prompt + generated) a single request may reach.
    pub fn max_sequence_tokens(&self) -> usize {
        (self.cfg.num_blocks * self.cfg.block_size).min(self.config().context_length as usize)
    }

    pub fn num_running(&self) -> usize {
        self.scheduler.running.len()
    }

    pub fn num_waiting(&self) -> usize {
        self.scheduler.waiting.len()
    }

    pub fn has_work(&self) -> bool {
        !self.scheduler.running.is_empty() || !self.scheduler.waiting.is_empty()
    }

    // Queues a request; it starts running on a later `step` once there is
    // room. `max_tokens` is clamped to what fits in the context window.
    pub fn add_request(&mut self, req: Request) -> Result<SequenceId, RequestError> {
        if req.tokens.is_empty() {
            return Err(RequestError::EmptyPrompt);
        }
        if req.max_tokens == 0 {
            return Err(RequestError::ZeroMaxTokens);
        }
        let context = self.config().context_length as usize;
        let prompt_len = req.tokens.len();
        if prompt_len >= context {
            return Err(RequestError::PromptTooLong {
                prompt: prompt_len,
                context,
            });
        }
        let max_tokens = req.max_tokens.min(context - prompt_len);
        let capacity = self.cfg.num_blocks * self.cfg.block_size;
        if prompt_len + max_tokens > capacity {
            return Err(RequestError::ExceedsKvCapacity {
                tokens: prompt_len + max_tokens,
                capacity,
            });
        }

        let seq_id = self
            .scheduler
            .add_request(req.tokens.iter().map(|&t| TokenId(t as i32)).collect());
        self.seqs.insert(
            seq_id,
            SeqState {
                prompt_len,
                max_tokens,
                stop_tokens: req.stop_tokens,
                num_generated: 0,
                last_token: 0,
                sampler: req.sampler,
            },
        );
        Ok(seq_id)
    }

    // Drops a request (e.g. the client went away), freeing its KV blocks.
    pub fn abort(&mut self, id: SequenceId) {
        if self.seqs.remove(&id).is_none() {
            return;
        }
        self.scheduler.waiting.retain(|&w| w != id);
        if self.scheduler.running.contains(&id) {
            self.scheduler.finish_sequence(id);
        }
        self.scheduler.sequences.remove(&id);
    }

    // One engine iteration: admit + prefill what fits, then one batched
    // decode step. Returns every token produced, in order.
    pub fn step(&mut self) -> Result<Vec<TokenEvent>, BoxError> {
        let mut events = Vec::new();
        self.admit_waiting(&mut events)?;
        self.decode(&mut events)?;
        Ok(events)
    }

    fn admit_waiting(&mut self, events: &mut Vec<TokenEvent>) -> Result<(), BoxError> {
        let block_size = self.cfg.block_size;
        while self.scheduler.running.len() < self.cfg.max_running {
            let Some(&seq_id) = self.scheduler.waiting.front() else {
                break;
            };
            let needed_blocks = self.scheduler.sequences[&seq_id]
                .token_ids
                .len()
                .div_ceil(block_size);
            if self.scheduler.pool.available() < needed_blocks {
                break;
            }

            let token_ids = self.scheduler.sequences[&seq_id].token_ids.clone();
            let (_matched_tokens, matched_blocks) =
                self.scheduler.prefix_cache.match_prefix(&token_ids);
            let skip_tokens = (matched_blocks.len() * block_size).min(token_ids.len() - 1);

            let prefill_started = Instant::now();
            self.scheduler.prefill(seq_id);

            let seq = &self.scheduler.sequences[&seq_id];
            let tokens: Vec<u32> = seq.token_ids.iter().map(|t| t.0 as u32).collect();
            let write_positions: Vec<(BlockID, usize)> = (skip_tokens..tokens.len())
                .map(|i| seq.block_table.logical_to_physical(i))
                .collect();

            // Only the tokens past the cached prefix run through the model,
            // but attention reads the sequence's whole block table.
            let item = BatchItem {
                tokens: &tokens[skip_tokens..],
                write_positions: &write_positions,
                read_blocks: seq.block_table.blocks(),
                read_num_tokens: seq.block_table.num_tokens(),
                position_offset: skip_tokens,
            };
            let state = self.seqs.get_mut(&seq_id).unwrap();
            let next = self
                .backend
                .forward_sample(&[item], true, &mut [&mut state.sampler])?[0];
            let prefill_elapsed = prefill_started.elapsed();
            self.stats.prefill_time += prefill_elapsed;
            self.stats.prefill_tokens += tokens.len() - skip_tokens;

            let first = state.num_generated == 0;
            self.emit(seq_id, next, first.then_some(prefill_elapsed), events);
        }
        Ok(())
    }

    // Batched decode: every running sequence contributes exactly one token
    // (the last one sampled) to a single forward call.
    fn decode(&mut self, events: &mut Vec<TokenEvent>) -> Result<(), BoxError> {
        if self.scheduler.running.is_empty() {
            return Ok(());
        }
        let decode_started = Instant::now();
        let running = self.scheduler.running.clone();
        let position_offsets: HashMap<SequenceId, usize> = running
            .iter()
            .map(|&id| (id, self.scheduler.sequences[&id].block_table.num_tokens()))
            .collect();

        self.scheduler.step();

        // A sequence that dropped out of `running` was preempted (its blocks
        // evicted). Rebuild its token list as prompt + every token sampled so
        // far, so it re-prefills cleanly - and samples its next token - when
        // readmitted. The scheduler may or may not have appended its
        // placeholder token before the eviction, and the latest sampled
        // token was never written back into the list, so fix up the tail.
        for &seq_id in &running {
            if !self.scheduler.running.contains(&seq_id) {
                let state = &self.seqs[&seq_id];
                let real_len = state.prompt_len + state.num_generated;
                let token_ids = &mut self.scheduler.sequences.get_mut(&seq_id).unwrap().token_ids;
                token_ids.truncate(real_len - 1);
                token_ids.push(TokenId(state.last_token as i32));
            }
        }

        let mut rows: Vec<DecodeRow> = Vec::with_capacity(running.len());
        for &seq_id in &running {
            if !self.scheduler.running.contains(&seq_id) {
                continue;
            }
            let last_token = self.seqs[&seq_id].last_token;
            let seq = self.scheduler.sequences.get_mut(&seq_id).unwrap();
            if let Some(t) = seq.token_ids.last_mut() {
                *t = TokenId(last_token as i32);
            }
            let position_offset = position_offsets[&seq_id];
            rows.push(DecodeRow {
                seq_id,
                token: [last_token],
                write: [seq.block_table.logical_to_physical(position_offset)],
                position_offset,
            });
        }
        if rows.is_empty() {
            return Ok(());
        }

        let next_ids = {
            let items: Vec<BatchItem> = rows
                .iter()
                .map(|row| {
                    let table = &self.scheduler.sequences[&row.seq_id].block_table;
                    BatchItem {
                        tokens: &row.token,
                        write_positions: &row.write,
                        read_blocks: table.blocks(),
                        read_num_tokens: table.num_tokens(),
                        position_offset: row.position_offset,
                    }
                })
                .collect();
            let mut by_id: HashMap<SequenceId, &mut Sampler> = self
                .seqs
                .iter_mut()
                .map(|(&id, state)| (id, &mut state.sampler))
                .collect();
            let mut samplers: Vec<&mut Sampler> = rows
                .iter()
                .map(|row| by_id.remove(&row.seq_id).unwrap())
                .collect();
            self.backend.forward_sample(&items, false, &mut samplers)?
        };

        for (row, &next) in rows.iter().zip(&next_ids) {
            self.emit(row.seq_id, next, None, events);
        }
        self.stats.decode_time += decode_started.elapsed();
        self.stats.decode_steps += 1;
        Ok(())
    }

    // Records a freshly sampled token, retiring the sequence if it's done.
    fn emit(
        &mut self,
        seq_id: SequenceId,
        token: u32,
        ttft: Option<Duration>,
        events: &mut Vec<TokenEvent>,
    ) {
        let state = self.seqs.get_mut(&seq_id).unwrap();
        state.num_generated += 1;
        state.last_token = token;
        let finish = if state.stop_tokens.contains(&token) {
            Some(FinishReason::Stop)
        } else if state.num_generated >= state.max_tokens {
            Some(FinishReason::Length)
        } else {
            None
        };
        if finish.is_some() {
            self.scheduler.finish_sequence(seq_id);
            // Finished sequences would otherwise pile up forever in a
            // long-running server.
            self.scheduler.sequences.remove(&seq_id);
            self.seqs.remove(&seq_id);
        }
        events.push(TokenEvent {
            id: seq_id,
            token,
            finish,
            ttft,
        });
    }
}

pub struct RequestSpec {
    pub tokens: Vec<u32>,
    pub max_tokens: usize,
    // Decode step at which this request becomes eligible for admission.
    pub arrival_step: usize,
}

#[derive(Default)]
pub struct RunStats {
    pub steps: usize,
    pub elapsed: Duration,
    pub output_tokens: usize,
    pub prefix_hits: usize,
    pub prefix_total: usize,
    pub total_block_allocs: usize,
    pub prefill_time: Duration,
    pub prefill_tokens: usize,
    pub decode_time: Duration,
    pub decode_steps: usize,
    // Per-request time-to-first-token: prefill start -> first sampled token.
    // Same order/length as the input `requests` slice.
    pub ttft: Vec<Duration>,
}

pub struct RunResult {
    // Same order/length as the input `requests` slice. A request that ended on
    // a stop token includes it as its last entry.
    pub generated: Vec<Vec<u32>>,
    pub stats: RunStats,
}

// Runs a fixed set of requests to completion. Each request samples with its
// own `Sampler` built from `sampling`.
pub fn run<B: Backend>(
    backend: B,
    requests: &[RequestSpec],
    sampling: SamplingParams,
    cfg: EngineConfig,
) -> Result<RunResult, BoxError> {
    let mut engine = Engine::new(backend, cfg);
    let eos_token_id = engine.config().eos_token_id;

    let mut order: Vec<usize> = (0..requests.len()).collect();
    order.sort_by_key(|&i| requests[i].arrival_step);
    let mut next_pending = 0;

    let mut req_of: HashMap<SequenceId, usize> = HashMap::new();
    let mut generated: Vec<Vec<u32>> = vec![Vec::new(); requests.len()];
    let mut ttft = vec![Duration::ZERO; requests.len()];

    let started = Instant::now();
    let mut step_idx = 0usize;
    loop {
        while next_pending < order.len() && requests[order[next_pending]].arrival_step <= step_idx {
            let i = order[next_pending];
            next_pending += 1;
            let id = engine.add_request(Request {
                tokens: requests[i].tokens.clone(),
                max_tokens: requests[i].max_tokens,
                stop_tokens: vec![eos_token_id],
                sampler: Sampler::from_params(sampling),
            })?;
            req_of.insert(id, i);
        }

        if !engine.has_work() && next_pending == order.len() {
            break;
        }

        for ev in engine.step()? {
            let i = req_of[&ev.id];
            generated[i].push(ev.token);
            if let Some(t) = ev.ttft {
                ttft[i] = t;
            }
        }
        step_idx += 1;
    }

    let scheduler = engine.scheduler();
    let stats = RunStats {
        steps: step_idx,
        elapsed: started.elapsed(),
        output_tokens: generated.iter().map(Vec::len).sum(),
        prefix_hits: scheduler.metrics.prefix_hits,
        prefix_total: scheduler.metrics.prefix_total,
        total_block_allocs: scheduler.pool.total_allocs,
        prefill_time: engine.stats.prefill_time,
        prefill_tokens: engine.stats.prefill_tokens,
        decode_time: engine.stats.decode_time,
        decode_steps: engine.stats.decode_steps,
        ttft,
    };
    Ok(RunResult { generated, stats })
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    // Deterministic stand-in for a model: the next token is always the last
    // input token + 1 (mod vocab). No KV involved, so scheduling behaviour -
    // queueing, batching, preemption, aborts - can be tested without weights.
    pub struct CountingBackend {
        pub config: Config,
        pub max_batch_seen: usize,
    }

    impl CountingBackend {
        pub fn new(context_length: u32) -> Self {
            CountingBackend {
                config: Config {
                    n_layers: 1,
                    n_heads: 1,
                    n_kv_heads: 1,
                    hidden_dim: 1,
                    ffn_dim: 1,
                    rope_theta: 10000.0,
                    rms_eps: 1e-5,
                    vocab_size: 1000,
                    context_length,
                    eos_token_id: 999,
                    bos_token_id: 0,
                },
                max_batch_seen: 0,
            }
        }
    }

    impl Backend for CountingBackend {
        fn config(&self) -> &Config {
            &self.config
        }

        fn forward_sample(
            &mut self,
            items: &[BatchItem],
            _last_only: bool,
            samplers: &mut [&mut Sampler],
        ) -> Result<Vec<u32>, BoxError> {
            assert_eq!(items.len(), samplers.len());
            self.max_batch_seen = self.max_batch_seen.max(items.len());
            Ok(items
                .iter()
                .map(|item| (item.tokens.last().unwrap() + 1) % 1000)
                .collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::CountingBackend;
    use super::*;

    fn cfg(num_blocks: usize, max_running: usize) -> EngineConfig {
        EngineConfig {
            block_size: 4,
            num_blocks,
            max_running,
        }
    }

    fn req(tokens: Vec<u32>, max_tokens: usize) -> Request {
        Request {
            tokens,
            max_tokens,
            stop_tokens: vec![],
            sampler: Sampler::from_params(SamplingParams::GREEDY),
        }
    }

    fn drain(engine: &mut Engine<CountingBackend>) -> Vec<TokenEvent> {
        let mut all = Vec::new();
        while engine.has_work() {
            all.extend(engine.step().unwrap());
        }
        all
    }

    fn tokens_of(events: &[TokenEvent], id: SequenceId) -> Vec<u32> {
        events
            .iter()
            .filter(|e| e.id == id)
            .map(|e| e.token)
            .collect()
    }

    #[test]
    fn first_sampled_token_is_emitted() {
        // The token sampled by prefill is part of the output (it used to be
        // dropped, so completions were missing their first token).
        let mut engine = Engine::new(CountingBackend::new(64), cfg(16, 8));
        let id = engine.add_request(req(vec![10, 11, 12], 3)).unwrap();
        let events = drain(&mut engine);
        assert_eq!(tokens_of(&events, id), vec![13, 14, 15]);
        assert_eq!(events.last().unwrap().finish, Some(FinishReason::Length));
        assert!(events[0].ttft.is_some());
        assert!(events[1].ttft.is_none());
    }

    #[test]
    fn stop_token_ends_generation() {
        let mut engine = Engine::new(CountingBackend::new(64), cfg(16, 8));
        let mut r = req(vec![10], 50);
        r.stop_tokens = vec![13];
        let id = engine.add_request(r).unwrap();
        let events = drain(&mut engine);
        assert_eq!(tokens_of(&events, id), vec![11, 12, 13]);
        assert_eq!(events.last().unwrap().finish, Some(FinishReason::Stop));
    }

    #[test]
    fn max_tokens_one_finishes_at_prefill() {
        let mut engine = Engine::new(CountingBackend::new(64), cfg(16, 8));
        let id = engine.add_request(req(vec![5, 6], 1)).unwrap();
        let events = engine.step().unwrap();
        assert_eq!(tokens_of(&events, id), vec![7]);
        assert_eq!(events[0].finish, Some(FinishReason::Length));
        assert!(!engine.has_work());
    }

    #[test]
    fn max_running_queues_the_rest_fifo() {
        let mut engine = Engine::new(CountingBackend::new(64), cfg(64, 2));
        let ids: Vec<_> = (0..5)
            .map(|i| engine.add_request(req(vec![100 * (i + 1)], 4)).unwrap())
            .collect();
        let mut events = engine.step().unwrap();
        assert_eq!(engine.num_running(), 2);
        assert_eq!(engine.num_waiting(), 3);

        let mut finish_order = Vec::new();
        while engine.has_work() {
            for ev in engine.step().unwrap() {
                if ev.finish.is_some() {
                    finish_order.push(ev.id);
                }
                events.push(ev);
            }
        }
        // Equal-length requests started in submission order finish in it.
        assert_eq!(finish_order, ids);
        assert!(engine.backend.max_batch_seen <= 2);
        for (i, id) in ids.iter().enumerate() {
            let p = 100 * (i as u32 + 1);
            assert_eq!(tokens_of(&events, *id), (p + 1..p + 5).collect::<Vec<_>>());
        }
    }

    #[test]
    fn outputs_do_not_depend_on_batching() {
        let run_alone = |prompt: u32| {
            let mut engine = Engine::new(CountingBackend::new(64), cfg(16, 8));
            let id = engine.add_request(req(vec![prompt], 5)).unwrap();
            tokens_of(&drain(&mut engine), id)
        };
        let mut engine = Engine::new(CountingBackend::new(64), cfg(16, 8));
        let a = engine.add_request(req(vec![10], 5)).unwrap();
        let b = engine.add_request(req(vec![200], 5)).unwrap();
        let events = drain(&mut engine);
        assert_eq!(tokens_of(&events, a), run_alone(10));
        assert_eq!(tokens_of(&events, b), run_alone(200));
        assert_eq!(engine.backend.max_batch_seen, 2);
    }

    #[test]
    fn preemption_still_completes_every_request() {
        // 6 blocks of 4 slots can't hold three 10-token sequences at once, so
        // some get evicted and re-prefilled; all must still finish with the
        // right tokens.
        let mut engine = Engine::new(CountingBackend::new(64), cfg(6, 8));
        let ids: Vec<_> = (0..3)
            .map(|i| engine.add_request(req(vec![10 * (i + 1)], 9)).unwrap())
            .collect();
        let events = drain(&mut engine);
        for (i, id) in ids.iter().enumerate() {
            let start = 10 * (i as u32 + 1) + 1;
            assert_eq!(
                tokens_of(&events, *id),
                (start..start + 9).collect::<Vec<_>>()
            );
        }
        assert_eq!(engine.scheduler().pool.available(), 6);
    }

    #[test]
    fn abort_frees_running_and_waiting() {
        let mut engine = Engine::new(CountingBackend::new(64), cfg(16, 1));
        let a = engine.add_request(req(vec![1, 2, 3], 50)).unwrap();
        let b = engine.add_request(req(vec![4, 5, 6], 50)).unwrap();
        engine.step().unwrap();
        assert_eq!((engine.num_running(), engine.num_waiting()), (1, 1));
        engine.abort(a);
        engine.abort(b);
        assert!(!engine.has_work());
        assert_eq!(engine.scheduler().pool.available(), 16);
        assert!(engine.scheduler().sequences.is_empty());
        engine.abort(a); // idempotent
    }

    #[test]
    fn rejects_requests_that_can_never_run() {
        let mut engine = Engine::new(CountingBackend::new(32), cfg(4, 8)); // 16 slots
        assert!(matches!(
            engine.add_request(req(vec![], 5)),
            Err(RequestError::EmptyPrompt)
        ));
        assert!(matches!(
            engine.add_request(req(vec![1], 0)),
            Err(RequestError::ZeroMaxTokens)
        ));
        assert!(matches!(
            engine.add_request(req(vec![1; 40], 5)),
            Err(RequestError::PromptTooLong { .. })
        ));
        assert!(matches!(
            engine.add_request(req(vec![1; 10], 20)),
            Err(RequestError::ExceedsKvCapacity { .. })
        ));
        // max_tokens beyond the context window is clamped, not rejected.
        let mut big = Engine::new(CountingBackend::new(8), cfg(16, 8));
        let id = big.add_request(req(vec![1, 2], 100)).unwrap();
        assert_eq!(tokens_of(&drain(&mut big), id).len(), 6);
    }

    #[test]
    fn run_wrapper_honours_arrival_steps() {
        let requests = vec![
            RequestSpec {
                tokens: vec![1],
                max_tokens: 3,
                arrival_step: 0,
            },
            RequestSpec {
                tokens: vec![50],
                max_tokens: 3,
                arrival_step: 2,
            },
        ];
        let result = run(
            CountingBackend::new(64),
            &requests,
            SamplingParams::GREEDY,
            cfg(16, 8),
        )
        .unwrap();
        assert_eq!(result.generated, vec![vec![2, 3, 4], vec![51, 52, 53]]);
        assert_eq!(result.stats.output_tokens, 6);
    }
}

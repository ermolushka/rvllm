// Answers on an engine the caller owns, for the interactive frontends: the
// TUI's worker thread and the terminal chat. The rules for segments and
// continuation live in `answer`; this module runs them.
//
// A chat keeps a `History`. Before a message is sent, the history is compacted
// so the prompt leaves room for the answer. If the answer then fills the pool,
// the history is compacted again, and the answer carries on from there.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::answer::{self, After, Answer, Ended};
use crate::chat::{self, ChatFormat, ChatMessage};
use crate::engine::{Backend, BoxError, Engine, FinishReason, Request};
use crate::history::{self, Compaction, History, Strategy};
use crate::sampler::{Sampler, SamplingParams};
use crate::tokenizer::{StreamDecoder, Tokenizer};

// Where an answer's text and notices go.
pub trait Sink {
    // A piece of the answer. Returns false to stop generating.
    fn text(&mut self, piece: &str) -> bool;
    // A line for the transcript, such as what compaction did.
    fn notice(&mut self, line: String);
}

// What every answer in a session shares.
#[derive(Clone)]
pub struct Options {
    pub max_tokens: usize,
    pub stop_tokens: Vec<u32>,
    pub sampling: SamplingParams,
}

// What an answer produced.
#[derive(Debug)]
pub struct Reply {
    pub text: String,
    // Tokens generated, over every segment.
    pub tokens: usize,
    pub ended: Ended,
    // Whether the answer reached its `max_tokens`.
    pub hit_limit: bool,
    pub secs: f64,
}

// Greedy decoding, for summaries.
const GREEDY: SamplingParams = SamplingParams {
    temperature: 0.0,
    top_p: 1.0,
    seed: 0,
};

const NOTICE_STOPPED_NO_ROOM: &str =
    "the answer stopped: the history can't make room to continue it";
const NOTICE_CONTINUING: &str = "continuing the answer after compacting the history";
const NOTICE_SUMMARIZING: &str = "summarizing the earlier conversation...";

// A chat: its history, and how to compact it.
pub struct Chat {
    format: ChatFormat,
    history: History,
    strategy: Strategy,
}

impl Chat {
    pub fn new(format: ChatFormat, system: Option<String>, strategy: Strategy) -> Self {
        Chat {
            format,
            history: History::new(system),
            strategy,
        }
    }

    // Forgets the conversation; the system prompt stays.
    pub fn clear(&mut self) {
        self.history.clear();
    }

    // Adds `user` to the history, compacts it so the prompt leaves room for
    // `max_tokens`, and returns the prompt to pass to `finish_turn`. If that
    // fails, the message is removed again, so the history is as it was.
    pub fn begin_turn<B: Backend>(
        &mut self,
        engine: &mut Engine<B>,
        tokenizer: &Tokenizer,
        user: String,
        max_tokens: usize,
        cancel: &AtomicBool,
        sink: &mut dyn Sink,
    ) -> Result<Vec<u32>, BoxError> {
        self.history.push("user", user);
        let pool = engine.max_sequence_tokens();
        let reserve = answer::reserve_for(max_tokens, pool);
        match self.compact_prompt(engine, tokenizer, reserve, 0, cancel, sink) {
            Ok(prompt) if prompt.len() < pool => Ok(prompt),
            Ok(prompt) => {
                self.history.pop();
                Err(format!(
                    "the conversation is {} tokens, which doesn't fit the {pool}-token KV budget even after compacting it",
                    prompt.len()
                )
                .into())
            }
            Err(e) => {
                self.history.pop();
                Err(e)
            }
        }
    }

    // Runs the answer to the last message, streaming it to `sink`, and adds it
    // to the history. `prompt` is what `begin_turn` returned. On error the
    // message is removed again. A cancelled answer keeps what it had.
    pub fn finish_turn<B: Backend>(
        &mut self,
        engine: &mut Engine<B>,
        tokenizer: &Tokenizer,
        prompt: Vec<u32>,
        opts: &Options,
        cancel: &AtomicBool,
        sink: &mut dyn Sink,
    ) -> Result<Reply, BoxError> {
        let mut answer = Answer::new(engine.max_sequence_tokens(), opts.max_tokens, prompt, true);
        match drive(
            engine,
            tokenizer,
            &mut answer,
            Some(self),
            opts,
            cancel,
            sink,
        ) {
            Ok(reply) => {
                self.history.push("assistant", reply.text.clone());
                Ok(reply)
            }
            Err(e) => {
                self.history.pop();
                Err(e)
            }
        }
    }

    // Compacts the history to fit `reserve` beside the prompt, with `pending`
    // tokens of answer so far counted towards it, and returns the prompt.
    fn compact_prompt<B: Backend>(
        &mut self,
        engine: &mut Engine<B>,
        tokenizer: &Tokenizer,
        reserve: usize,
        pending: usize,
        cancel: &AtomicBool,
        sink: &mut dyn Sink,
    ) -> Result<Vec<u32>, BoxError> {
        let format = self.format;
        let pool = engine.max_sequence_tokens();
        let report = self.history.fit(
            self.strategy,
            pool,
            reserve,
            pending,
            |messages| Ok(tokenizer.encode(&format.render(messages))?.len()),
            |messages, cap| {
                // Shown before the model starts writing, since a summary isn't streamed.
                sink.notice(NOTICE_SUMMARIZING.into());
                summarize(engine, tokenizer, format, messages, cap, cancel)
            },
        )?;
        if !report.is_empty() {
            sink.notice(compaction_note(&report));
        }
        tokenizer.encode(&format.render(&self.history.messages()))
    }
}

// An answer with no chat history. It is capped at the room the pool leaves
// beside the prompt.
pub fn plain<B: Backend>(
    engine: &mut Engine<B>,
    tokenizer: &Tokenizer,
    prompt: Vec<u32>,
    opts: &Options,
    cancel: &AtomicBool,
    sink: &mut dyn Sink,
) -> Result<Reply, BoxError> {
    let mut answer = Answer::new(engine.max_sequence_tokens(), opts.max_tokens, prompt, false);
    drive(engine, tokenizer, &mut answer, None, opts, cancel, sink)
}

// Runs `answer` segment by segment until it ends. One decoder carries across
// the segments, so a character split by a segment boundary comes out whole.
// `chat` is compacted between segments; without one, the answer can't be
// continued.
fn drive<B: Backend>(
    engine: &mut Engine<B>,
    tokenizer: &Tokenizer,
    answer: &mut Answer,
    mut chat: Option<&mut Chat>,
    opts: &Options,
    cancel: &AtomicBool,
    sink: &mut dyn Sink,
) -> Result<Reply, BoxError> {
    let started = Instant::now();
    let mut decoder = StreamDecoder::new();
    let mut text = String::new();
    let ended = loop {
        if answer.remaining() == 0 {
            break Ended::Length;
        }
        let Some(max) = answer.next_segment() else {
            let line = if chat.is_some() {
                NOTICE_STOPPED_NO_ROOM
            } else {
                "the answer stopped: the prompt fills the KV pool"
            };
            sink.notice(line.into());
            break Ended::Length;
        };
        let segment = run_segment(
            engine,
            answer.tokens(),
            max,
            opts,
            cancel,
            &mut |token, finish| {
                // A stop token is not part of the text.
                let piece = if finish == Some(FinishReason::Stop) {
                    decoder.flush(tokenizer)?
                } else {
                    decoder.push(tokenizer, token)?
                };
                if piece.is_empty() {
                    return Ok(true);
                }
                text.push_str(&piece);
                Ok(sink.text(&piece))
            },
        )?;
        match answer.record(&segment.ids, segment.ended) {
            After::Finished(ended) => {
                if ended == Ended::Length && answer.remaining() > 0 {
                    sink.notice("the answer stopped: the KV pool is full".into());
                }
                break ended;
            }
            After::Limit => {
                sink.notice(format!(
                    "the answer stopped after {} continuations",
                    answer::MAX_CONTINUATIONS
                ));
                break Ended::Length;
            }
            After::Compact => {
                let Some(chat) = chat.as_deref_mut() else {
                    break Ended::Length;
                };
                let prompt = match chat.compact_prompt(
                    engine,
                    tokenizer,
                    answer.reserve(),
                    answer.ids().len(),
                    cancel,
                    sink,
                ) {
                    Ok(prompt) => prompt,
                    // Esc during the summary keeps the answer so far.
                    Err(e) if e.is::<history::Cancelled>() => break Ended::Cancelled,
                    Err(e) => return Err(e),
                };
                if !answer.rebase(prompt) {
                    sink.notice(NOTICE_STOPPED_NO_ROOM.into());
                    break Ended::Length;
                }
                sink.notice(NOTICE_CONTINUING.into());
            }
        }
    };
    let tail = decoder.flush(tokenizer)?;
    if !tail.is_empty() {
        text.push_str(&tail);
        sink.text(&tail);
    }
    Ok(Reply {
        text,
        tokens: answer.ids().len(),
        ended,
        hit_limit: ended == Ended::Length && answer.remaining() == 0,
        secs: started.elapsed().as_secs_f64(),
    })
}

// What one request produced before it ended.
struct Segment {
    ids: Vec<u32>,
    ended: Ended,
}

// Runs one request to its end on the engine. `on_token` gets each token with
// the finish reason it came with, and returns false to stop. `cancel` stops it
// too. A failure aborts the request, so it doesn't keep running.
fn run_segment<B: Backend>(
    engine: &mut Engine<B>,
    tokens: Vec<u32>,
    max_tokens: usize,
    opts: &Options,
    cancel: &AtomicBool,
    on_token: &mut dyn FnMut(u32, Option<FinishReason>) -> Result<bool, BoxError>,
) -> Result<Segment, BoxError> {
    let id = engine.add_request(Request {
        tokens,
        max_tokens,
        stop_tokens: opts.stop_tokens.clone(),
        sampler: Sampler::from_params(opts.sampling),
    })?;
    let mut ids = Vec::new();
    while engine.has_work() {
        if cancel.load(Ordering::Relaxed) {
            engine.abort(id);
            return Ok(Segment {
                ids,
                ended: Ended::Cancelled,
            });
        }
        let events = match engine.step() {
            Ok(events) => events,
            Err(e) => {
                engine.abort(id);
                return Err(e);
            }
        };
        for ev in events.into_iter().filter(|ev| ev.id == id) {
            ids.push(ev.token);
            let keep_going = match on_token(ev.token, ev.finish) {
                Ok(keep_going) => keep_going,
                Err(e) => {
                    engine.abort(id);
                    return Err(e);
                }
            };
            if let Some(finish) = ev.finish {
                return Ok(Segment {
                    ids,
                    ended: finish.into(),
                });
            }
            if !keep_going {
                engine.abort(id);
                return Ok(Segment {
                    ids,
                    ended: Ended::Cancelled,
                });
            }
        }
    }
    Err("the engine finished without ending the request".into())
}

// Writes the model's summary of `messages`, with the chat template's stop
// tokens. The summary is greedy and not streamed.
fn summarize<B: Backend>(
    engine: &mut Engine<B>,
    tokenizer: &Tokenizer,
    format: ChatFormat,
    messages: &[ChatMessage],
    max_tokens: usize,
    cancel: &AtomicBool,
) -> Result<String, BoxError> {
    let opts = Options {
        max_tokens,
        stop_tokens: chat::stop_tokens(tokenizer, Some(format)),
        sampling: GREEDY,
    };
    let tokens = tokenizer.encode(&format.render(messages))?;
    let mut ids = Vec::new();
    let segment = run_segment(
        engine,
        tokens,
        max_tokens,
        &opts,
        cancel,
        &mut |token, finish| {
            if finish != Some(FinishReason::Stop) {
                ids.push(token);
            }
            Ok(true)
        },
    )?;
    if segment.ended == Ended::Cancelled {
        return Err(history::Cancelled.into());
    }
    tokenizer.decode(&ids)
}

fn compaction_note(report: &Compaction) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::engine::testing::CountingBackend;

    // Collects what an answer streams and says.
    #[derive(Default)]
    struct Collect {
        text: String,
        notices: Vec<String>,
        // Stop after this many pieces, if set.
        limit: Option<usize>,
        pieces: usize,
    }

    impl Sink for Collect {
        fn text(&mut self, piece: &str) -> bool {
            self.text.push_str(piece);
            self.pieces += 1;
            self.limit.is_none_or(|n| self.pieces < n)
        }

        fn notice(&mut self, line: String) {
            self.notices.push(line);
        }
    }

    // CountingBackend emits last token + 1 each step, so the answer is a run of
    // consecutive ids, and a continuation continues it exactly.
    fn engine(block_size: usize, num_blocks: usize, context: u32) -> Engine<CountingBackend> {
        Engine::new(
            CountingBackend::new(context),
            EngineConfig {
                block_size,
                num_blocks,
                max_running: 1,
            },
        )
    }

    fn tokenizer() -> Tokenizer {
        Tokenizer::from_file("tokenizer.json", 999, 0).unwrap()
    }

    fn options(max_tokens: usize) -> Options {
        Options {
            max_tokens,
            stop_tokens: Vec::new(),
            sampling: GREEDY,
        }
    }

    fn no_cancel() -> AtomicBool {
        AtomicBool::new(false)
    }

    fn turn_text(words: usize, n: usize) -> String {
        // Roughly `words` tokens of ordinary text, so budgets are easy to reason about.
        std::iter::repeat_n("cat", words)
            .collect::<Vec<_>>()
            .join(" ")
            + &format!(" {n}")
    }

    #[test]
    fn a_plain_answer_is_capped_at_the_room_the_pool_leaves() {
        // 16 blocks x 4 = 64 tokens. The prompt takes most of it.
        let mut engine = engine(4, 16, 4096);
        let tok = tokenizer();
        let prompt = tok.encode(&turn_text(40, 1)).unwrap();
        let room = 64 - prompt.len();
        let mut sink = Collect::default();
        let reply = plain(
            &mut engine,
            &tok,
            prompt,
            &options(500),
            &no_cancel(),
            &mut sink,
        )
        .unwrap();
        assert_eq!(reply.tokens, room);
        assert_eq!(reply.ended, Ended::Length);
        assert!(!reply.hit_limit);
        assert!(sink.notices.iter().any(|n| n.contains("KV pool is full")));
    }

    #[test]
    fn a_plain_answer_reaches_its_length_when_it_fits() {
        let mut engine = engine(4, 64, 4096);
        let tok = tokenizer();
        let prompt = tok.encode("hello there").unwrap();
        let mut sink = Collect::default();
        let reply = plain(
            &mut engine,
            &tok,
            prompt,
            &options(7),
            &no_cancel(),
            &mut sink,
        )
        .unwrap();
        assert_eq!(reply.tokens, 7);
        assert!(reply.hit_limit);
        assert_eq!(reply.ended, Ended::Length);
        assert!(sink.notices.is_empty());
    }

    #[test]
    fn a_stop_token_ends_the_answer_and_is_not_in_the_text() {
        let mut engine = engine(4, 64, 4096);
        let tok = tokenizer();
        let prompt = tok.encode("hello there").unwrap();
        let last = *prompt.last().unwrap();
        // The backend emits last + 1, last + 2, ...: stop on the third.
        let opts = Options {
            stop_tokens: vec![last + 3],
            ..options(20)
        };
        let mut sink = Collect::default();
        let reply = plain(&mut engine, &tok, prompt, &opts, &no_cancel(), &mut sink).unwrap();
        assert_eq!(reply.ended, Ended::Stop);
        assert!(!reply.hit_limit);
        assert_eq!(reply.tokens, 3);
        let without_stop = tok.decode(&[last + 1, last + 2]).unwrap();
        assert_eq!(sink.text, without_stop);
    }

    #[test]
    fn the_sink_can_stop_the_answer() {
        let mut engine = engine(4, 64, 4096);
        let tok = tokenizer();
        let prompt = tok.encode("hello there").unwrap();
        let mut sink = Collect {
            limit: Some(2),
            ..Default::default()
        };
        let reply = plain(
            &mut engine,
            &tok,
            prompt,
            &options(20),
            &no_cancel(),
            &mut sink,
        )
        .unwrap();
        assert_eq!(reply.ended, Ended::Cancelled);
        assert!(!reply.hit_limit);
        assert!(reply.tokens < 20);
        assert!(!engine.has_work());
    }

    #[test]
    fn a_cancel_flag_set_beforehand_generates_nothing() {
        let mut engine = engine(4, 64, 4096);
        let tok = tokenizer();
        let prompt = tok.encode("hello there").unwrap();
        let cancel = AtomicBool::new(true);
        let mut sink = Collect::default();
        let reply = plain(&mut engine, &tok, prompt, &options(20), &cancel, &mut sink).unwrap();
        assert_eq!(reply.ended, Ended::Cancelled);
        assert_eq!(reply.tokens, 0);
        assert!(sink.text.is_empty());
    }

    #[test]
    fn an_empty_prompt_is_an_error_and_leaves_no_request_running() {
        let mut engine = engine(4, 64, 4096);
        let tok = tokenizer();
        let mut sink = Collect::default();
        assert!(
            plain(
                &mut engine,
                &tok,
                Vec::new(),
                &options(5),
                &no_cancel(),
                &mut sink
            )
            .is_err()
        );
        assert!(!engine.has_work());
    }

    // Two short earlier turns, then a long question, then an answer longer than
    // the room left beside its prompt. `blocks` x 4 tokens is the pool.
    fn long_answer(blocks: usize) -> (Reply, Collect) {
        let mut engine = engine(4, blocks, 4096);
        let tok = tokenizer();
        let mut chat = Chat::new(ChatFormat::SMOLLM2, None, Strategy::Drop);
        let mut sink = Collect::default();
        for n in 0..2 {
            let prompt = chat
                .begin_turn(
                    &mut engine,
                    &tok,
                    turn_text(60, n),
                    60,
                    &no_cancel(),
                    &mut sink,
                )
                .unwrap();
            chat.finish_turn(
                &mut engine,
                &tok,
                prompt,
                &options(60),
                &no_cancel(),
                &mut sink,
            )
            .unwrap();
        }
        sink.text.clear();
        let prompt = chat
            .begin_turn(
                &mut engine,
                &tok,
                turn_text(100, 9),
                300,
                &no_cancel(),
                &mut sink,
            )
            .unwrap();
        let reply = chat
            .finish_turn(
                &mut engine,
                &tok,
                prompt,
                &options(300),
                &no_cancel(),
                &mut sink,
            )
            .unwrap();
        (reply, sink)
    }

    #[test]
    fn a_chat_answer_continues_across_compaction_to_its_full_length() {
        // 512 tokens of pool: the 300-token answer can't fit beside the prompt,
        // so it has to be continued after the old turns are dropped.
        let (reply, sink) = long_answer(128);
        assert_eq!(reply.tokens, 300, "cut short: {:?}", sink.notices);
        assert!(reply.hit_limit);
        assert_eq!(reply.ended, Ended::Length);
        assert!(
            sink.notices.iter().any(|n| n == NOTICE_CONTINUING),
            "{:?}",
            sink.notices
        );
    }

    #[test]
    fn a_continued_answer_reads_as_one_uninterrupted_answer() {
        // The same chat in a pool big enough for the whole answer. The backend's
        // output depends only on the last token of the prompt, which is the same
        // in both, so the texts must match exactly.
        let (small, _) = long_answer(128);
        let (big, big_sink) = long_answer(256);
        assert!(!big_sink.notices.iter().any(|n| n == NOTICE_CONTINUING));
        assert_eq!(small.tokens, big.tokens);
        assert_eq!(small.text, big.text);
    }

    #[test]
    fn an_answer_stops_when_compaction_cannot_make_room() {
        // One huge turn that can't be dropped (the newest message is kept), so
        // the pool stays full after compaction and the answer must stop.
        let mut engine = engine(4, 16, 4096);
        let tok = tokenizer();
        let mut chat = Chat::new(ChatFormat::SMOLLM2, None, Strategy::Drop);
        let mut sink = Collect::default();
        let prompt = chat
            .begin_turn(
                &mut engine,
                &tok,
                turn_text(30, 1),
                40,
                &no_cancel(),
                &mut sink,
            )
            .unwrap();
        let reply = chat
            .finish_turn(
                &mut engine,
                &tok,
                prompt,
                &options(40),
                &no_cancel(),
                &mut sink,
            )
            .unwrap();
        assert!(!reply.hit_limit);
        assert!(reply.tokens < 40);
        assert!(
            sink.notices.iter().any(|n| n == NOTICE_STOPPED_NO_ROOM),
            "{:?}",
            sink.notices
        );
    }

    #[test]
    fn a_message_too_big_for_the_pool_is_refused_and_not_kept() {
        let mut engine = engine(4, 16, 4096);
        let tok = tokenizer();
        let mut chat = Chat::new(ChatFormat::SMOLLM2, None, Strategy::Drop);
        let mut sink = Collect::default();
        let error = chat
            .begin_turn(
                &mut engine,
                &tok,
                turn_text(200, 1),
                8,
                &no_cancel(),
                &mut sink,
            )
            .unwrap_err();
        assert!(error.to_string().contains("doesn't fit"), "{error}");
        assert!(chat.history.messages().iter().all(|m| m.role != "user"));
    }

    #[test]
    fn a_failed_answer_removes_its_message_from_the_history() {
        let mut engine = engine(4, 64, 4096);
        let tok = tokenizer();
        let mut chat = Chat::new(ChatFormat::SMOLLM2, None, Strategy::Drop);
        let mut sink = Collect::default();
        chat.begin_turn(&mut engine, &tok, "hi".into(), 8, &no_cancel(), &mut sink)
            .unwrap();
        // An empty prompt cannot run, so the answer fails.
        let error = chat
            .finish_turn(
                &mut engine,
                &tok,
                Vec::new(),
                &options(8),
                &no_cancel(),
                &mut sink,
            )
            .unwrap_err();
        assert!(error.to_string().contains("empty"), "{error}");
        assert!(chat.history.messages().iter().all(|m| m.role != "user"));
    }

    #[test]
    fn esc_during_the_summary_leaves_the_history_as_it_was() {
        let mut engine = engine(4, 32, 4096);
        let tok = tokenizer();
        let mut chat = Chat::new(ChatFormat::SMOLLM2, None, Strategy::Summarize);
        let mut sink = Collect::default();
        let opts = options(8);
        for n in 0..3 {
            let prompt = chat
                .begin_turn(
                    &mut engine,
                    &tok,
                    turn_text(12, n),
                    8,
                    &no_cancel(),
                    &mut sink,
                )
                .unwrap();
            chat.finish_turn(&mut engine, &tok, prompt, &opts, &no_cancel(), &mut sink)
                .unwrap();
        }
        let before: Vec<String> = chat
            .history
            .messages()
            .into_iter()
            .map(|m| m.content)
            .collect();
        // Cancel, so the summary request is stopped before it generates.
        let cancel = AtomicBool::new(true);
        let error = chat
            .begin_turn(&mut engine, &tok, turn_text(40, 9), 8, &cancel, &mut sink)
            .unwrap_err();
        assert!(error.is::<history::Cancelled>(), "{error}");
        let after: Vec<String> = chat
            .history
            .messages()
            .into_iter()
            .map(|m| m.content)
            .collect();
        assert_eq!(after, before);
        assert!(!engine.has_work());
    }

    #[test]
    fn compaction_notes_name_what_happened() {
        let summarized = Compaction {
            summarized: 6,
            ..Default::default()
        };
        assert_eq!(
            compaction_note(&summarized),
            "history over the prompt budget: summarized 6 earlier messages"
        );
        let failed = Compaction {
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

    #[test]
    fn a_chat_without_compaction_keeps_the_whole_history() {
        let mut engine = engine(4, 64, 4096);
        let tok = tokenizer();
        let mut chat = Chat::new(
            ChatFormat::SMOLLM2,
            Some("be brief".into()),
            Strategy::Summarize,
        );
        let mut sink = Collect::default();
        let opts = options(4);
        for n in 0..2 {
            let prompt = chat
                .begin_turn(
                    &mut engine,
                    &tok,
                    format!("question {n}"),
                    4,
                    &no_cancel(),
                    &mut sink,
                )
                .unwrap();
            chat.finish_turn(&mut engine, &tok, prompt, &opts, &no_cancel(), &mut sink)
                .unwrap();
        }
        assert!(sink.notices.is_empty());
        let roles: Vec<String> = chat
            .history
            .messages()
            .into_iter()
            .map(|m| m.role)
            .collect();
        assert_eq!(
            roles,
            vec!["system", "user", "assistant", "user", "assistant"]
        );
    }
}

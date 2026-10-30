// Keeps a chat's prompt inside the KV budget. When a conversation outgrows it,
// the oldest turns are either dropped, or folded into a rolling summary that
// rides along in the system message.
//
// Counting tokens and writing the summary need a model, so `History::fit`
// takes them as closures. The TUI passes the loaded model's tokenizer and
// engine; a server or `Generator` can pass its own.

use crate::chat::ChatMessage;
use crate::engine::BoxError;

// The longest summary the model may write, in tokens.
pub const SUMMARY_MAX_TOKENS: usize = 256;

const SUMMARY_INSTRUCTIONS: &str = "Write a summary of the conversation so far, updating any \
    summary given with the new messages. Keep the names, facts, numbers and decisions the user \
    may refer back to. Reply with the summary only.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    // Remove the oldest user/assistant pairs.
    Drop,
    // Fold the oldest turns into a summary the model writes. Falls back to
    // dropping them if the summary can't be written.
    Summarize,
}

// What `History::fit` did to the history.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Compaction {
    // Turns folded into the summary.
    pub summarized: usize,
    // Turns removed outright, including any dropped to fit the summary request.
    pub dropped: usize,
    // The old summary was removed to fit the limit.
    pub summary_dropped: bool,
    // Why a summary was wanted but couldn't be written, so turns were dropped
    // instead.
    pub summary_error: Option<String>,
}

impl Compaction {
    pub fn is_empty(&self) -> bool {
        self.summarized == 0
            && self.dropped == 0
            && !self.summary_dropped
            && self.summary_error.is_none()
    }

    pub fn summary_failed(&self) -> bool {
        self.summary_error.is_some()
    }
}

// What `summarize` returns when the caller stopped it, such as on Esc. `fit`
// passes it on and leaves the history as it was, rather than falling back to
// dropping turns.
#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cancelled")
    }
}

impl std::error::Error for Cancelled {}

// Token limits for one `fit`, in the prompt's own tokens.
#[derive(Clone, Copy)]
struct Budget {
    // Most tokens one request can hold: the KV pool or the context, whichever
    // is smaller.
    max_sequence: usize,
    // What a prompt may use: `max_sequence` less the answer's `max_tokens`.
    limit: usize,
    // Where compaction stops. Below `limit`, so a long chat isn't compacted
    // again on every turn.
    target: usize,
    // Tokens of the answer so far. They count towards the prompt, but not
    // towards a summary request, which is a request of its own.
    pending: usize,
}

// One chat: the system prompt, the summary of turns folded away so far, and
// the turns still kept, oldest first.
#[derive(Clone)]
pub struct History {
    system: Option<String>,
    summary: Option<String>,
    turns: Vec<ChatMessage>,
}

impl History {
    pub fn new(system: Option<String>) -> Self {
        History {
            system,
            summary: None,
            turns: Vec::new(),
        }
    }

    // Forgets the turns and the summary; the system prompt stays.
    pub fn clear(&mut self) {
        self.summary = None;
        self.turns.clear();
    }

    pub fn push(&mut self, role: &str, content: String) {
        self.turns.push(ChatMessage {
            role: role.into(),
            content,
        });
    }

    // Removes the newest turn, for when its request is refused.
    pub fn pop(&mut self) {
        self.turns.pop();
    }

    // The messages to render: the system prompt with the summary after it,
    // then the turns.
    pub fn messages(&self) -> Vec<ChatMessage> {
        let mut system: Vec<String> = self.system.iter().cloned().collect();
        if let Some(summary) = &self.summary {
            system.push(format!("Summary of the earlier conversation:\n{summary}"));
        }
        let mut messages = Vec::with_capacity(self.turns.len() + 1);
        if !system.is_empty() {
            messages.push(ChatMessage {
                role: "system".into(),
                content: system.join("\n\n"),
            });
        }
        messages.extend(self.turns.iter().cloned());
        messages
    }

    // Compacts the history so the prompt fits: the most one request can hold
    // (`max_sequence_tokens`) less the `max_tokens` the answer may take.
    // `measure` counts the tokens of rendered messages. `pending` is how many
    // tokens of the answer so far count towards the prompt too. `summarize`
    // writes a summary of the messages it's given, at most the number of tokens
    // it's also given.
    //
    // Does nothing if the prompt already fits. If the system prompt and the
    // newest message don't fit on their own, nothing changes either, so the
    // engine's refusal is what the user sees.
    pub fn fit<M, S>(
        &mut self,
        strategy: Strategy,
        max_sequence_tokens: usize,
        max_tokens: usize,
        pending: usize,
        measure: M,
        mut summarize: S,
    ) -> Result<Compaction, BoxError>
    where
        M: Fn(&[ChatMessage]) -> Result<usize, BoxError>,
        S: FnMut(&[ChatMessage], usize) -> Result<String, BoxError>,
    {
        // The answer keeps its full length, so the prompt gets whatever the pool
        // holds beside it. If that is nothing, nothing can be compacted to fit.
        let limit = max_sequence_tokens.saturating_sub(max_tokens);
        let mut report = Compaction::default();
        if measure(&self.messages())? + pending <= limit {
            return Ok(report);
        }
        let newest_alone = History {
            system: self.system.clone(),
            summary: None,
            turns: self.turns.last().cloned().into_iter().collect(),
        };
        if measure(&newest_alone.messages())? + pending > limit {
            return Ok(report);
        }

        let budget = Budget {
            max_sequence: max_sequence_tokens,
            limit,
            target: limit * 3 / 4,
            pending,
        };
        if strategy == Strategy::Summarize && self.turns.len() > 1 {
            match self.summarize_oldest(budget, &measure, &mut summarize)? {
                Ok((summarized, dropped)) => {
                    report.summarized = summarized;
                    report.dropped = dropped;
                }
                Err(reason) => report.summary_error = Some(reason),
            }
        }
        let (dropped, summary_dropped) = self.drop_until(budget, &measure)?;
        report.dropped += dropped;
        report.summary_dropped = summary_dropped;
        Ok(report)
    }

    // Folds the oldest turns into the summary: at least one pair, and as many
    // more as it takes for what remains, plus room for the summary, to fit the
    // target. The result is how many turns were summarized and how many were
    // dropped to fit the summary request, or the reason no summary could be
    // written. In that case the history is left as it was. A `Cancelled` from
    // `summarize` is returned as an error, with the history left as it was too.
    //
    // The inner `Result` is the summary's outcome, and the outer one carries
    // errors from `measure` and `Cancelled`.
    fn summarize_oldest<M, S>(
        &mut self,
        budget: Budget,
        measure: &M,
        summarize: &mut S,
    ) -> Result<Result<(usize, usize), String>, BoxError>
    where
        M: Fn(&[ChatMessage]) -> Result<usize, BoxError>,
        S: FnMut(&[ChatMessage], usize) -> Result<String, BoxError>,
    {
        let before = measure(&self.messages())? + budget.pending;
        let cap = summary_cap(budget.limit);
        // While folding, the summary counts as a placeholder of `cap` tokens.
        // The old summary is replaced by the new one, and a summary stands in
        // for the model's default system prompt, so neither is counted twice.
        let mut rest = self.clone();
        rest.summary = Some(String::new());
        let mut folded = Vec::new();
        // At least one pair is always folded, so an old summary is rolled
        // forward rather than silently dropped to make room.
        while rest.turns.len() > 1
            && (folded.is_empty()
                || measure(&rest.messages())? + budget.pending + cap > budget.target)
        {
            let n = 2.min(rest.turns.len() - 1);
            folded.extend(rest.turns.drain(..n));
        }

        // The summary request is itself a prompt, and it has to leave room in
        // the pool for the summary it writes. The oldest folded turns that
        // don't fit are dropped instead of summarized.
        let mut skip = 0;
        let request = loop {
            if skip == folded.len() {
                return Ok(Err("the summary request doesn't fit the pool".into()));
            }
            let request = summary_request(self.summary.as_deref(), &folded[skip..]);
            if measure(&request)? + cap <= budget.max_sequence {
                break request;
            }
            skip += 2.min(folded.len() - skip);
        };
        let summary = match summarize(&request, cap) {
            Ok(summary) => summary,
            Err(e) if e.is::<Cancelled>() => return Err(e),
            Err(e) => return Ok(Err(format!("the model failed: {e}"))),
        };
        let summary = summary.trim();
        if summary.is_empty() {
            return Ok(Err("the model wrote no summary".into()));
        }
        rest.summary = Some(summary.to_string());
        if measure(&rest.messages())? + budget.pending >= before {
            return Ok(Err("the summary doesn't shrink the prompt".into()));
        }
        *self = rest;
        Ok(Ok((folded.len() - skip, skip)))
    }

    // Removes the oldest pairs until the prompt is within the target, always
    // keeping the newest turn. If the prompt is still over the limit, the
    // summary goes too. Returns how many turns were removed, and whether the
    // summary was.
    fn drop_until<M>(&mut self, budget: Budget, measure: &M) -> Result<(usize, bool), BoxError>
    where
        M: Fn(&[ChatMessage]) -> Result<usize, BoxError>,
    {
        let mut dropped = 0;
        while self.turns.len() > 1 && measure(&self.messages())? + budget.pending > budget.target {
            let n = 2.min(self.turns.len() - 1);
            self.turns.drain(..n);
            dropped += n;
        }
        let mut summary_dropped = false;
        if self.summary.is_some() && measure(&self.messages())? + budget.pending > budget.limit {
            self.summary = None;
            summary_dropped = true;
        }
        Ok((dropped, summary_dropped))
    }
}

// Room for a summary, in tokens: the most it may take, or a quarter of the
// limit when that is smaller.
fn summary_cap(limit: usize) -> usize {
    SUMMARY_MAX_TOKENS.min(limit / 4).max(1)
}

// The messages that ask the model for a summary: the summary so far, if
// there is one, and the turns to fold into it.
fn summary_request(previous: Option<&str>, turns: &[ChatMessage]) -> Vec<ChatMessage> {
    let mut text = String::new();
    if let Some(previous) = previous {
        text.push_str(&format!("Summary so far:\n{previous}\n\n"));
    }
    text.push_str("New messages:\n");
    for turn in turns {
        let speaker = if turn.role == "assistant" {
            "Assistant"
        } else {
            "User"
        };
        text.push_str(&format!("{speaker}: {}\n\n", turn.content.trim()));
    }
    vec![
        ChatMessage {
            role: "system".into(),
            content: SUMMARY_INSTRUCTIONS.into(),
        },
        ChatMessage {
            role: "user".into(),
            content: text,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    // Each character is one token, so a turn's size is its length.
    fn chars(messages: &[ChatMessage]) -> Result<usize, BoxError> {
        Ok(messages.iter().map(|m| m.content.len()).sum())
    }

    fn turn(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
        }
    }

    // `pairs` user/assistant exchanges of 100 characters each, then a newest
    // user message of 100 characters.
    fn chat(system: Option<&str>, pairs: usize) -> History {
        let mut history = History::new(system.map(Into::into));
        for i in 0..pairs {
            history.push("user", format!("{i:0>100}"));
            history.push("assistant", format!("{i:1>100}"));
        }
        history.push("user", format!("{:x>100}", "newest"));
        history
    }

    // Budget of 900 tokens for the prompt: 2000 for the pool, 1100 for the
    // answer.
    const MAX_SEQUENCE: usize = 2000;
    const MAX_TOKENS: usize = 1100;

    #[test]
    fn leaves_a_prompt_that_fits_alone() {
        let mut history = chat(None, 1);
        let before = history.messages().len();
        let report = history
            .fit(
                Strategy::Summarize,
                MAX_SEQUENCE,
                MAX_TOKENS,
                0,
                chars,
                |_, _| panic!("nothing to summarize"),
            )
            .unwrap();
        assert!(report.is_empty());
        assert_eq!(history.messages().len(), before);
    }

    #[test]
    fn drop_removes_the_oldest_pairs_and_keeps_the_newest() {
        let mut history = chat(Some("sys"), 5);
        let report = history
            .fit(
                Strategy::Drop,
                MAX_SEQUENCE,
                MAX_TOKENS,
                0,
                chars,
                |_, _| panic!("drop never summarizes"),
            )
            .unwrap();
        assert!(report.dropped > 0 && report.dropped.is_multiple_of(2));
        assert_eq!(report.summarized, 0);
        assert!(!report.summary_failed());

        let messages = history.messages();
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[0].content, "sys");
        assert_eq!(messages[1].role, "user");
        assert_eq!(
            messages.last().unwrap().content,
            format!("{:x>100}", "newest")
        );
        assert!(chars(&messages).unwrap() <= 900 * 3 / 4);
    }

    #[test]
    fn summarize_folds_the_oldest_turns_into_the_system_prompt() {
        let mut history = chat(Some("sys"), 5);
        let mut asked = Vec::new();
        let report = history
            .fit(
                Strategy::Summarize,
                MAX_SEQUENCE,
                MAX_TOKENS,
                0,
                chars,
                |messages, cap| {
                    asked.push((messages.to_vec(), cap));
                    Ok("the summary".into())
                },
            )
            .unwrap();

        assert!(report.summarized > 0);
        assert_eq!(report.dropped, 0);
        assert!(!report.summary_failed());
        assert_eq!(asked.len(), 1);
        let (request, cap) = &asked[0];
        assert_eq!(*cap, 225);
        assert!(request[1].content.contains("New messages:"));
        assert!(!request[1].content.contains("Summary so far"));

        let messages = history.messages();
        assert_eq!(messages[0].role, "system");
        assert_eq!(
            messages[0].content,
            "sys\n\nSummary of the earlier conversation:\nthe summary"
        );
        assert_eq!(
            messages.last().unwrap().content,
            format!("{:x>100}", "newest")
        );
    }

    #[test]
    fn rolling_summary_goes_back_to_the_model() {
        let mut history = chat(None, 5);
        history
            .fit(
                Strategy::Summarize,
                MAX_SEQUENCE,
                MAX_TOKENS,
                0,
                chars,
                |_, _| Ok("first".into()),
            )
            .unwrap();
        for _ in 0..5 {
            history.push("user", format!("{:y>100}", "more"));
            history.push("assistant", format!("{:z>100}", "reply"));
        }
        let mut request_text = String::new();
        history
            .fit(
                Strategy::Summarize,
                MAX_SEQUENCE,
                MAX_TOKENS,
                0,
                chars,
                |messages, _| {
                    request_text = messages[1].content.clone();
                    Ok("second".into())
                },
            )
            .unwrap();
        assert!(request_text.starts_with("Summary so far:\nfirst\n\n"));
        assert!(history.messages()[0].content.ends_with("\nsecond"));
    }

    #[test]
    fn falls_back_to_dropping_when_the_summary_fails() {
        let mut history = chat(Some("sys"), 5);
        let report = history
            .fit(
                Strategy::Summarize,
                MAX_SEQUENCE,
                MAX_TOKENS,
                0,
                chars,
                |_, _| Err("model refused".into()),
            )
            .unwrap();
        assert!(report.summary_failed());
        assert_eq!(report.summarized, 0);
        assert!(report.dropped > 0);
        assert_eq!(history.messages()[0].content, "sys");
        assert!(chars(&history.messages()).unwrap() <= 900 * 3 / 4);
    }

    #[test]
    fn falls_back_to_dropping_when_the_summary_does_not_shrink_the_prompt() {
        let mut history = chat(None, 5);
        let report = history
            .fit(
                Strategy::Summarize,
                MAX_SEQUENCE,
                MAX_TOKENS,
                0,
                chars,
                |_, _| Ok("S".repeat(5000)),
            )
            .unwrap();
        assert!(report.summary_failed());
        assert!(report.dropped > 0);
        assert!(
            history
                .messages()
                .iter()
                .all(|m| !m.content.contains("SSSS"))
        );
    }

    #[test]
    fn leaves_history_alone_when_the_newest_message_alone_is_too_big() {
        let mut history = chat(Some("sys"), 1);
        history.push("user", "y".repeat(2000));
        let before = history.messages();
        let report = history
            .fit(
                Strategy::Summarize,
                MAX_SEQUENCE,
                MAX_TOKENS,
                0,
                chars,
                |_, _| panic!("nothing should be summarized"),
            )
            .unwrap();
        assert!(report.is_empty());
        assert_eq!(history.messages().len(), before.len());
    }

    #[test]
    fn clear_keeps_the_system_prompt_and_forgets_the_rest() {
        let mut history = chat(Some("sys"), 5);
        history
            .fit(
                Strategy::Summarize,
                MAX_SEQUENCE,
                MAX_TOKENS,
                0,
                chars,
                |_, _| Ok("summary".into()),
            )
            .unwrap();
        history.clear();
        let messages = history.messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "sys");
    }

    // An old summary too big to keep, and too big to fold into a summary
    // request, is removed and reported, not dropped silently.
    #[test]
    fn summary_that_no_longer_fits_is_reported_as_dropped() {
        let mut history = History::new(None);
        history.push("user", tagged("A1"));
        history.push("assistant", tagged("B1"));
        history.push("user", tagged("N"));
        history.summary = Some("S".repeat(850));
        let report = history
            .fit(Strategy::Summarize, 1000, 100, 0, chars, |_, _| {
                panic!("the request doesn't fit, so the model is not asked")
            })
            .unwrap();
        assert!(report.summary_failed());
        assert!(report.summary_dropped);
        assert_eq!(report.dropped, 2);
        let kept: String = history
            .messages()
            .iter()
            .map(|m| m.content.clone())
            .collect();
        assert!(!kept.contains("SSSS"));
        assert!(kept.contains("[N]"));
    }

    // A `max_tokens` at least as big as the pool leaves no room for a prompt, so
    // nothing is compacted. The answer is never shortened to make room.
    #[test]
    fn large_max_tokens_leaves_no_room_so_nothing_is_compacted() {
        let mut history = chat(None, 5);
        let before: Vec<String> = history.messages().into_iter().map(|m| m.content).collect();
        let report = history
            .fit(Strategy::Drop, 1000, 5000, 0, chars, |_, _| {
                panic!("drop never summarizes")
            })
            .unwrap();
        assert!(report.is_empty());
        let after: Vec<String> = history.messages().into_iter().map(|m| m.content).collect();
        assert_eq!(after, before);
    }

    // Esc during the summary is not a failed summary: the error comes back, and
    // no turn is dropped.
    #[test]
    fn answer_so_far_counts_towards_the_prompt_but_not_the_summary_request() {
        // 1000 tokens in the pool and 100 for the answer leave 900 for the
        // prompt. 500 tokens of answer so far push the prompt over that, but
        // the summary request is a request of its own, so it still fits.
        let mut history = chat(None, 4);
        let mut asked = 0;
        let report = history
            .fit(Strategy::Summarize, 1000, 100, 500, chars, |_, _| {
                asked += 1;
                Ok("summary".into())
            })
            .unwrap();
        assert_eq!(asked, 1);
        assert!(!report.summary_failed());
        assert!(report.summarized > 0);
        assert!(chars(&history.messages()).unwrap() + 500 <= 900);
    }

    #[test]
    fn cancelled_summary_leaves_history_unchanged() {
        let mut history = chat(Some("sys"), 5);
        let before: Vec<String> = history.messages().into_iter().map(|m| m.content).collect();
        let error = history
            .fit(
                Strategy::Summarize,
                MAX_SEQUENCE,
                MAX_TOKENS,
                0,
                chars,
                |_, _| Err(Cancelled.into()),
            )
            .unwrap_err();
        assert!(error.is::<Cancelled>());
        let after: Vec<String> = history.messages().into_iter().map(|m| m.content).collect();
        assert_eq!(after, before);
    }

    // A 100-character turn that names itself, so a summary request can be
    // checked for which turns it holds.
    fn tagged(tag: &str) -> String {
        format!("{:-<100}", format!("[{tag}]"))
    }

    // A second summary, after the first, must still fit a tight pool: the old
    // summary counts once, and the request stays within the pool.
    #[test]
    fn rolling_summary_still_fits_at_a_tight_budget() {
        let mut history = History::new(None);
        for i in 0..5 {
            history.push("user", tagged(&format!("A{i}")));
            history.push("assistant", tagged(&format!("B{i}")));
        }
        history.push("user", tagged("N"));
        history
            .fit(Strategy::Summarize, 1000, 100, 0, chars, |_, _| {
                Ok("first".repeat(45))
            })
            .unwrap();
        for i in 5..11 {
            history.push("user", tagged(&format!("A{i}")));
            history.push("assistant", tagged(&format!("B{i}")));
        }

        let mut request_sizes = Vec::new();
        let report = history
            .fit(Strategy::Summarize, 1000, 100, 0, chars, |messages, cap| {
                request_sizes.push(chars(messages).unwrap() + cap);
                Ok("second".into())
            })
            .unwrap();
        assert!(!report.summary_failed());
        assert!(report.summarized > 0);
        assert!(request_sizes.iter().all(|&size| size <= 1000));
    }

    // When the folded turns don't fit the summary request, the oldest of them
    // are dropped and reported, and the newest are summarized.
    #[test]
    fn oversized_summary_request_drops_the_oldest_folded_turns() {
        let mut history = History::new(None);
        for i in 1..=4 {
            history.push("user", tagged(&format!("A{i}")));
            history.push("assistant", tagged(&format!("B{i}")));
        }
        history.push("user", tagged("N"));

        let mut request = String::new();
        let mut request_size = 0;
        let report = history
            .fit(Strategy::Summarize, 700, 100, 0, chars, |messages, cap| {
                request = messages[1].content.clone();
                request_size = chars(messages).unwrap() + cap;
                Ok("summary".into())
            })
            .unwrap();
        assert!(!report.summary_failed());
        assert!(report.summarized > 0);
        assert!(report.dropped > 0);
        assert!(!request.contains("[A1]"));
        assert!(request.contains("[A4]"));
        assert!(request_size <= 700);
        let kept: String = history
            .messages()
            .iter()
            .map(|m| m.content.clone())
            .collect();
        assert!(!kept.contains("[A1]"));
        assert!(kept.contains("[N]"));
    }

    #[test]
    fn summary_request_labels_and_trims_each_turn() {
        let request = summary_request(None, &[turn("user", " hi "), turn("assistant", "yo")]);
        assert_eq!(request[0].role, "system");
        assert_eq!(
            request[1].content,
            "New messages:\nUser: hi\n\nAssistant: yo\n\n"
        );
    }
}

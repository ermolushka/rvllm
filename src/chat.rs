// Chat prompt formatting: one renderer per chat template family, each
// producing what the model's own template does for plain-text chat.

use crate::model::{Arch, Config};
use crate::tokenizer::Tokenizer;

pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

// SmolLM2-Instruct's chat template adds this system message to a conversation
// that doesn't start with one.
const SMOLLM2_SYSTEM: &str = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

// Likewise for Qwen2.5-Instruct.
const QWEN2_5_SYSTEM: &str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

// Terminates each ChatML turn.
pub const END_OF_TURN: &str = "<|im_end|>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatFormat {
    // `<|im_start|>role\n...<|im_end|>\n` turns. `default_system` is used when
    // the conversation doesn't start with a system message. `skip_thinking`
    // opens the reply with an empty thinking block, which is how Qwen3's
    // template turns thinking off.
    ChatMl {
        default_system: Option<&'static str>,
        skip_thinking: bool,
    },
    // Llama 3's `<|start_header_id|>role<|end_header_id|>` turns.
    Llama3,
    // Gemma's `<start_of_turn>role\n...<end_of_turn>\n` turns, with the
    // assistant as `model`.
    Gemma,
}

impl ChatFormat {
    pub const SMOLLM2: ChatFormat = ChatFormat::ChatMl {
        default_system: Some(SMOLLM2_SYSTEM),
        skip_thinking: false,
    };
    pub const QWEN2_5: ChatFormat = ChatFormat::ChatMl {
        default_system: Some(QWEN2_5_SYSTEM),
        skip_thinking: false,
    };
    pub const QWEN3: ChatFormat = ChatFormat::ChatMl {
        default_system: None,
        skip_thinking: true,
    };
    // Unlike LFM2's template, this keeps an empty system message.
    pub const LFM2: ChatFormat = ChatFormat::ChatMl {
        default_system: None,
        skip_thinking: false,
    };

    // The chat format of `config`'s model, if rvllm knows its family. Within
    // an architecture, the GGUF's tokenizer family tells models apart.
    pub fn for_model(config: &Config) -> Option<ChatFormat> {
        match (config.arch, config.tokenizer_pre.as_deref()) {
            (Arch::Llama, Some("smollm")) => Some(ChatFormat::SMOLLM2),
            (Arch::Llama, Some("llama-bpe")) => Some(ChatFormat::Llama3),
            (Arch::Qwen2, Some("qwen2")) => Some(ChatFormat::QWEN2_5),
            (Arch::Qwen3, Some("qwen2")) => Some(ChatFormat::QWEN3),
            (Arch::Gemma3, _) => Some(ChatFormat::Gemma),
            (Arch::Lfm2, Some("lfm2")) => Some(ChatFormat::LFM2),
            _ => None,
        }
    }

    // The prompt for `messages`, ending where the assistant's reply starts. A
    // BOS token the template starts with is left to the tokenizer.
    pub fn render(&self, messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        match *self {
            ChatFormat::ChatMl {
                default_system,
                skip_thinking,
            } => {
                if let Some(system) = default_system
                    && messages.first().is_none_or(|m| m.role != "system")
                {
                    out.push_str(&format!("<|im_start|>system\n{system}{END_OF_TURN}\n"));
                }
                for m in messages {
                    out.push_str(&format!(
                        "<|im_start|>{}\n{}{END_OF_TURN}\n",
                        m.role, m.content
                    ));
                }
                out.push_str("<|im_start|>assistant\n");
                if skip_thinking {
                    out.push_str("<think>\n\n</think>\n\n");
                }
            }
            ChatFormat::Llama3 => {
                let (system, turns) = match messages.split_first() {
                    Some((first, rest)) if first.role == "system" => (first.content.trim(), rest),
                    _ => ("", messages),
                };
                // The template also writes "Today Date: <date>" here, and falls
                // back to a fixed 26 Jul 2024 without a clock. A wrong date is
                // worse than none, and today's date would make the same
                // conversation give a different prompt each day, so the line
                // is left out, as Ollama's Llama 3.2 template does.
                out.push_str(&format!(
                    "<|start_header_id|>system<|end_header_id|>\n\n\
                     Cutting Knowledge Date: December 2023\n\n{system}<|eot_id|>"
                ));
                for m in turns {
                    out.push_str(&format!(
                        "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                        m.role,
                        m.content.trim()
                    ));
                }
                out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
            ChatFormat::Gemma => {
                // Gemma has no system role: its template puts a leading system
                // message, untrimmed, in front of the first turn. Turns that
                // don't alternate between user and assistant are rendered as
                // given (the template rejects them, llama.cpp doesn't).
                let (system, turns) = match messages.split_first() {
                    Some((first, rest)) if first.role == "system" => (Some(&first.content), rest),
                    _ => (None, messages),
                };
                for (i, m) in turns.iter().enumerate() {
                    let role = if m.role == "assistant" {
                        "model"
                    } else {
                        &m.role
                    };
                    out.push_str(&format!("<start_of_turn>{role}\n"));
                    if i == 0
                        && let Some(system) = system
                    {
                        out.push_str(&format!("{system}\n\n"));
                    }
                    out.push_str(&format!("{}<end_of_turn>\n", m.content.trim()));
                }
                out.push_str("<start_of_turn>model\n");
            }
        }
        out
    }

    // Tokens besides EOS that end a reply in this format.
    fn end_of_turn(&self) -> &'static [&'static str] {
        match self {
            ChatFormat::ChatMl { .. } => &[END_OF_TURN],
            ChatFormat::Llama3 => &["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"],
            ChatFormat::Gemma => &["<end_of_turn>"],
        }
    }
}

// Tokens that end a generation: the model's EOS, plus its chat format's
// end-of-turn markers (ChatML's when the format is unknown) that the
// vocabulary has.
pub fn stop_tokens(tokenizer: &Tokenizer, format: Option<ChatFormat>) -> Vec<u32> {
    let markers = format.map_or(&[END_OF_TURN][..], |f| f.end_of_turn());
    let mut stops = vec![tokenizer.eos_token_id];
    for id in markers.iter().filter_map(|m| tokenizer.token_to_id(m)) {
        if !stops.contains(&id) {
            stops.push(id);
        }
    }
    stops
}

// ChatML with SmolLM2's default system prompt.
pub fn chatml(messages: &[ChatMessage]) -> String {
    ChatFormat::SMOLLM2.render(messages)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
        }
    }

    #[test]
    fn adds_default_system_prompt() {
        let p = chatml(&[msg("user", "hi")]);
        assert_eq!(
            p,
            "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n\
             <|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn keeps_explicit_system_prompt_and_history() {
        let p = chatml(&[
            msg("system", "be terse"),
            msg("user", "a"),
            msg("assistant", "b"),
            msg("user", "c"),
        ]);
        assert!(p.starts_with("<|im_start|>system\nbe terse<|im_end|>\n<|im_start|>user\na"));
        assert!(p.ends_with("<|im_start|>user\nc<|im_end|>\n<|im_start|>assistant\n"));
        assert_eq!(p.matches("system").count(), 1);
    }

    // Expected strings are Llama-3.2-1B-Instruct's own Jinja template, rendered
    // as HF's `apply_chat_template` does, minus the leading BOS and the
    // "Today Date" line.
    #[test]
    fn llama3_matches_its_template() {
        assert_eq!(
            ChatFormat::Llama3.render(&[msg("user", " hi\n")]),
            "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n\n\
             <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\
             hi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        assert_eq!(
            ChatFormat::Llama3.render(&[
                msg("system", "be terse\n"),
                msg("user", "a"),
                msg("assistant", "b"),
                msg("user", "c"),
            ]),
            "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n\n\
             be terse<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\
             a<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nb<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\nc<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    // Qwen2.5-0.5B-Instruct's own template, rendered as HF does.
    #[test]
    fn qwen2_5_matches_its_template() {
        assert_eq!(
            ChatFormat::QWEN2_5.render(&[msg("user", "hi")]),
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful \
             assistant.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    // Qwen3-0.6B's own template with `enable_thinking=False`, rendered as HF
    // does.
    #[test]
    fn qwen3_matches_its_template_without_thinking() {
        assert_eq!(
            ChatFormat::QWEN3.render(&[
                msg("system", "be terse"),
                msg("user", "a"),
                msg("assistant", "b"),
                msg("user", "c"),
            ]),
            "<|im_start|>system\nbe terse<|im_end|>\n<|im_start|>user\na<|im_end|>\n\
             <|im_start|>assistant\nb<|im_end|>\n<|im_start|>user\nc<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(
            ChatFormat::QWEN3.render(&[msg("user", "hi")]),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    // gemma-3-270m-it's own template, rendered as HF does, minus the leading
    // BOS.
    #[test]
    fn gemma_matches_its_template() {
        assert_eq!(
            ChatFormat::Gemma.render(&[
                msg("system", " be terse "),
                msg("user", "a"),
                msg("assistant", " b\n"),
                msg("user", "c"),
            ]),
            "<start_of_turn>user\n be terse \n\na<end_of_turn>\n<start_of_turn>model\nb<end_of_turn>\n\
             <start_of_turn>user\nc<end_of_turn>\n<start_of_turn>model\n"
        );
        assert_eq!(
            ChatFormat::Gemma.render(&[msg("user", "  padded \n")]),
            "<start_of_turn>user\npadded<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    // LFM2.5-1.2B-Instruct's own template, rendered as HF does, minus the
    // leading BOS.
    #[test]
    fn lfm2_matches_its_template() {
        assert_eq!(
            ChatFormat::LFM2.render(&[
                msg("system", "be terse"),
                msg("user", "a"),
                msg("assistant", "b"),
                msg("user", "c"),
            ]),
            "<|im_start|>system\nbe terse<|im_end|>\n<|im_start|>user\na<|im_end|>\n\
             <|im_start|>assistant\nb<|im_end|>\n<|im_start|>user\nc<|im_end|>\n\
             <|im_start|>assistant\n"
        );
        assert_eq!(
            ChatFormat::LFM2.render(&[msg("user", "hi")]),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn format_follows_the_tokenizer_family() {
        let format = |arch: Arch, pre: Option<&str>| {
            ChatFormat::for_model(&Config {
                arch,
                tokenizer_pre: pre.map(Into::into),
                ..Config::for_tests()
            })
        };
        assert_eq!(
            format(Arch::Llama, Some("smollm")),
            Some(ChatFormat::SMOLLM2)
        );
        assert_eq!(
            format(Arch::Llama, Some("llama-bpe")),
            Some(ChatFormat::Llama3)
        );
        assert_eq!(
            format(Arch::Qwen2, Some("qwen2")),
            Some(ChatFormat::QWEN2_5)
        );
        assert_eq!(format(Arch::Qwen3, Some("qwen2")), Some(ChatFormat::QWEN3));
        assert_eq!(
            format(Arch::Gemma3, Some("default")),
            Some(ChatFormat::Gemma)
        );
        assert_eq!(format(Arch::Lfm2, Some("lfm2")), Some(ChatFormat::LFM2));
        assert_eq!(format(Arch::Llama, Some("qwen2")), None);
        assert_eq!(format(Arch::Llama, Some("default")), None);
        assert_eq!(format(Arch::Llama, None), None);
    }

    #[test]
    fn stop_tokens_add_end_of_turn_markers_the_vocabulary_has() {
        // SmolLM2's tokenizer: 0 = <|endoftext|>, 2 = <|im_end|>.
        let base = Tokenizer::from_file("tokenizer.json", 0, 0).unwrap();
        let instruct = Tokenizer::from_file("tokenizer.json", 2, 1).unwrap();
        assert_eq!(stop_tokens(&base, Some(ChatFormat::SMOLLM2)), [0, 2]);
        assert_eq!(stop_tokens(&instruct, Some(ChatFormat::SMOLLM2)), [2]);
        assert_eq!(stop_tokens(&base, None), [0, 2]);
        // None of Llama 3's markers exist in this vocabulary.
        assert_eq!(stop_tokens(&base, Some(ChatFormat::Llama3)), [0]);
    }
}

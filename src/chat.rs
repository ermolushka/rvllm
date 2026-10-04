// Chat prompt formatting. SmolLM2-Instruct uses ChatML.

pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

const DEFAULT_SYSTEM: &str = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

// Terminates each turn; also the token generation should stop on.
pub const END_OF_TURN: &str = "<|im_end|>";

pub fn chatml(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    if messages.first().is_none_or(|m| m.role != "system") {
        out.push_str(&format!(
            "<|im_start|>system\n{DEFAULT_SYSTEM}{END_OF_TURN}\n"
        ));
    }
    for m in messages {
        out.push_str(&format!(
            "<|im_start|>{}\n{}{END_OF_TURN}\n",
            m.role, m.content
        ));
    }
    out.push_str("<|im_start|>assistant\n");
    out
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
}

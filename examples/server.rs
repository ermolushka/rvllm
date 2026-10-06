// OpenAI-compatible server built only on the library (`server` feature, no
// clap/ratatui):
//   cargo run --release --no-default-features --features server --example server -- <model.gguf> <tokenizer.json> [addr]

use rvllm::engine::BoxError;
use rvllm::server::{self, ServeOptions};

fn main() -> Result<(), BoxError> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .ok_or("usage: server <model.gguf> <tokenizer.json> [addr]")?;
    let tokenizer = args.next().ok_or("missing tokenizer path")?;
    let mut opts = ServeOptions::default();
    if let Some(addr) = args.next() {
        opts.addr = addr;
    }
    server::run(&model, &tokenizer, opts)
}

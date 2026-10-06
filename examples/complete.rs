use rvllm::generator::{GenerateOptions, Generator};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .ok_or("usage: complete <model.gguf> <tokenizer.json> [prompt]")?;
    let tok = args.next().ok_or("missing tokenizer path")?;
    let prompt = args
        .next()
        .unwrap_or_else(|| "The capital of France is".into());
    let generator = Generator::load(&model, &tok, "cpu")?;
    let opts = GenerateOptions {
        max_tokens: 20,
        ..Default::default()
    };
    println!("{}", generator.complete(&prompt, &opts)?);
    println!(
        "{:?}",
        generator.complete_batch(&["The sky is", "2 + 2 ="], &opts)?
    );
    Ok(())
}

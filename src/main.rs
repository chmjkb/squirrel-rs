use squirrel_rs::cli::{assets_dir, configure_thread_pool, parse_flag};
use squirrel_rs::file_parser::gguf_file::GGUFFile;
use squirrel_rs::models::llama::model::Llama3_2;
use squirrel_rs::samplers::greedy::GreedySampler;
use squirrel_rs::text_generator::TextTokenGenerator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    configure_thread_pool();

    let assets_path = assets_dir();
    let model_filename = parse_flag("--model", "Llama-3.2-1B-Instruct-Q8_0.gguf");
    let tokenizer_filename = "tokenizer.json";
    // `--prompt-file` keeps long benchmark prompts out of the command line.
    let prompt_file = parse_flag("--prompt-file", "");
    let prompt = if prompt_file.is_empty() {
        parse_flag("--prompt", "The capital of France is")
    } else {
        std::fs::read_to_string(&prompt_file)?
            .trim_end()
            .to_string()
    };
    let max_tokens: usize = parse_flag("--max-tokens", "512")
        .parse()
        .expect("--max-tokens expects a number");

    let model_path = format!("{}/{}", assets_path, model_filename);
    let tokenizer_path = format!("{}/{}", assets_path, tokenizer_filename);

    let gguf_file = GGUFFile::from_file(&model_path)?;
    let llama = Llama3_2::from_gguf(&gguf_file, &tokenizer_path)
        .expect("failed to build Llama3_2 from GGUF");

    let generator = TextTokenGenerator::new(&llama);
    let output = generator
        .generate::<GreedySampler>(&prompt, max_tokens)
        .expect("generation failed");
    println!("{prompt}{output}");
    Ok(())
}

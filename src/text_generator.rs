//! Generic autoregressive text generation: tokenize → forward → sample → repeat.

use crate::models::common::{error::LLMError, model::Model};
use crate::samplers::common::Sampler;
use std::io::Write;
use std::sync::OnceLock;
use std::time::Instant;

/// `SQUIRREL_NO_KV_CACHE=1` disables the KV cache
fn no_kv_cache() -> bool {
    static NO_KV: OnceLock<bool> = OnceLock::new();
    *NO_KV.get_or_init(|| std::env::var_os("SQUIRREL_NO_KV_CACHE").is_some_and(|v| v != "0"))
}

/// `SQUIRREL_IGNORE_EOS=1` keeps generating past the end-of-sequence token.
fn ignore_eos() -> bool {
    static IGNORE: OnceLock<bool> = OnceLock::new();
    *IGNORE.get_or_init(|| std::env::var_os("SQUIRREL_IGNORE_EOS").is_some_and(|v| v != "0"))
}

/// Greedy/single-sample autoregressive decoder.
///
/// Owns a reference to a model; generation is parameterized by the sampler
/// type. Streams each newly produced token to stdout as it's decoded, with
/// per-step latency on stderr so the timing log doesn't mix with the
/// generated text.
pub struct TextTokenGenerator<'a, M: Model> {
    model: &'a M,
}

impl<'a, M: Model> TextTokenGenerator<'a, M> {
    pub fn new(model: &'a M) -> Self {
        Self { model }
    }

    /// Encodes `prompt`, runs forward+sample until EOS or `max_tokens`
    /// generations, returns the decoded text
    pub fn generate<S>(&self, prompt: &str, max_tokens: usize) -> Result<String, LLMError>
    where
        S: Sampler<f32>,
    {
        let encoding = self
            .model
            .tokenizer()
            .encode(prompt, true)
            .map_err(|_| LLMError::InvalidTokenizer)?;
        let prompt_len = encoding.get_ids().len();
        // Reported so benchmarks can label runs by real token count rather
        // than by an estimate from the character length.
        eprintln!("[prompt: {prompt_len} tokens]");
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        let mut printed = String::new();

        // The context (KV cache + scratch buffers) persists across steps. the
        // first forward prefills the whole prompt, and each later step feeds
        // only the newly sampled token, attending against the cached history.
        // Reusing it means we don't reallocate intermediate buffers per step.
        let mut ctx = self.model.new_context();
        let mut input: Vec<u32> = tokens.clone();

        for step in 1..=max_tokens {
            let t0 = Instant::now();
            self.model.forward(&input, &mut ctx)?;
            let logits_slice = ctx.logits().map_err(LLMError::Kernel)?;
            let next = S::sample(logits_slice) as u32;
            let elapsed = t0.elapsed();

            if next == self.model.eos_token_id() && !ignore_eos() {
                eprintln!(
                    "[step {step:>3}: {:>7.1}ms]  <eos>",
                    elapsed.as_secs_f64() * 1000.0
                );
                break;
            }
            tokens.push(next);
            if no_kv_cache() {
                ctx.reset_cache();
                input = tokens.clone();
            } else {
                // Next step decodes only the new token against the cache.
                input = vec![next];
            }

            let full = self
                .model
                .tokenizer()
                .decode(&tokens[prompt_len..], true)
                .map_err(|_| LLMError::InvalidTokenizer)?;
            let new_text = &full[printed.len()..];
            eprintln!(
                "[step {step:>3}: {:>7.1}ms]  {new_text:?}",
                elapsed.as_secs_f64() * 1000.0
            );
            print!("{new_text}");
            std::io::stdout().flush().ok();
            printed = full;
        }
        println!();

        Ok(printed)
    }
}

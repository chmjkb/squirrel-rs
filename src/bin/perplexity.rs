//! Usage:
//!   cargo run --release --bin perplexity -- \
//!     --model Llama-3.2-1B-Instruct-Q8_0.gguf \
//!     --text assets/wikitext2-raw-test.txt \
//!     [--ctx 512] [--windows 0(=all)]

use squirrel_rs::cli::{assets_dir, configure_thread_pool, parse_flag};
use squirrel_rs::file_parser::gguf_file::GGUFFile;
use squirrel_rs::models::common::model::Model;
use squirrel_rs::models::llama::model::Llama3_2;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    configure_thread_pool();

    let assets_path = assets_dir();
    let model_filename = parse_flag("--model", "Llama-3.2-1B-Instruct-Q8_0.gguf");
    let text_path = parse_flag("--text", &format!("{assets_path}/wikitext2-raw-test.txt"));
    let n_ctx: usize = parse_flag("--ctx", "512").parse()?;
    let max_windows: usize = parse_flag("--windows", "0").parse()?;
    if n_ctx < 4 {
        return Err("--ctx must be at least 4".into());
    }

    let model_path = format!("{assets_path}/{model_filename}");
    let gguf_file = GGUFFile::from_file(&model_path)?;
    let llama = Llama3_2::from_gguf(&gguf_file, &format!("{assets_path}/tokenizer.json"))
        .expect("failed to build Llama3_2 from GGUF");

    let text = std::fs::read_to_string(&text_path)?;
    let encoding = llama
        .tokenizer()
        .encode(text.as_str(), false)
        .map_err(|e| format!("tokenize: {e}"))?;
    let ids = encoding.get_ids();

    let acts = if std::env::var_os("SQUIRREL_F32_ACTS").is_some_and(|v| v != "0") {
        "f32"
    } else {
        "int8(where applicable)"
    };
    let per_window = n_ctx - 1; // one slot reserved for BOS
    let n_windows_total = ids.len() / per_window;
    let n_windows = if max_windows == 0 {
        n_windows_total
    } else {
        max_windows.min(n_windows_total)
    };
    let burn = n_ctx / 2;
    eprintln!(
        "model={model_filename} acts={acts} corpus={} tokens ctx={n_ctx} burn-in={burn} \
         windows={n_windows}/{n_windows_total}",
        ids.len()
    );

    let bos = llama.bos_token_id();
    let t0 = Instant::now();
    let mut total_nll = 0.0f64;
    let mut total_scored = 0usize;

    for (w, chunk) in ids.chunks_exact(per_window).take(n_windows).enumerate() {
        let mut window = Vec::with_capacity(n_ctx);
        window.push(bos);
        window.extend_from_slice(chunk);

        // Fresh cache per window, we prefill the burn-in context in one pass.
        // Its logits predict window[burn], the first scored position.
        let mut ctx = llama.new_context();
        llama
            .forward(&window[..burn], &mut ctx)
            .map_err(|e| format!("prefill: {e:?}"))?;

        let mut window_nll = 0.0f64;
        for i in burn..window.len() {
            let logits = ctx.logits().map_err(|e| format!("logits: {e:?}"))?;
            window_nll -= log_softmax_at(logits, window[i] as usize);
            total_scored += 1;
            // Teacher-force the actual token; its logits predict window[i+1].
            if i + 1 < window.len() {
                llama
                    .forward(&window[i..=i], &mut ctx)
                    .map_err(|e| format!("decode: {e:?}"))?;
            }
        }
        total_nll += window_nll;

        let running_ppl = (total_nll / total_scored as f64).exp();
        eprintln!(
            "[window {:>4}/{n_windows}] ppl so far: {running_ppl:.4}  ({:.1}s)",
            w + 1,
            t0.elapsed().as_secs_f64()
        );
    }

    let ppl = (total_nll / total_scored as f64).exp();
    println!(
        "model={model_filename} acts={acts} ctx={n_ctx} windows={n_windows} \
         scored_tokens={total_scored} nll_per_token={:.6} ppl={ppl:.4} time_s={:.1}",
        total_nll / total_scored as f64,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// log p(idx) under softmax(logits), computed in f64 via max subtraction.
fn log_softmax_at(logits: &[f32], idx: usize) -> f64 {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let sum_exp: f64 = logits.iter().map(|&l| (l as f64 - max).exp()).sum();
    logits[idx] as f64 - max - sum_exp.ln()
}

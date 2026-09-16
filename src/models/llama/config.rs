use crate::file_parser::{gguf_file::GGUFFile, gguf_metadata_value::GGUFValue};

#[derive(Debug)]
pub enum ConfigError {
    MissingMetadata(String),
    WrongMetadataType { key: String, expected: &'static str },
}

/// Architectural constants read once from GGUF metadata (Llama 3.2)
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub n_blocks: usize,
    pub embed_dim: usize,
    pub ffn_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_base: f32,
    pub rms_eps: f32,
    pub vocab_size: usize,
    pub context_length: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    /// Precomputed RoPE inverse frequencies, length `head_dim / 2`.
    pub rope_inv_freq: Vec<f32>,
}

impl LlamaConfig {
    pub fn from_gguf(gguf: &GGUFFile) -> Result<Self, ConfigError> {
        let n_blocks = read_u32(gguf, "llama.block_count")? as usize;
        let embed_dim = read_u32(gguf, "llama.embedding_length")? as usize;
        let ffn_dim = read_u32(gguf, "llama.feed_forward_length")? as usize;
        let n_heads = read_u32(gguf, "llama.attention.head_count")? as usize;
        let n_kv_heads = read_u32(gguf, "llama.attention.head_count_kv")? as usize;
        let head_dim = read_u32(gguf, "llama.rope.dimension_count")
            .map(|v| v as usize)
            .unwrap_or(embed_dim / n_heads);
        let rope_base = read_f32(gguf, "llama.rope.freq_base")?;
        let rms_eps = read_f32(gguf, "llama.attention.layer_norm_rms_epsilon")?;
        let context_length = read_u32(gguf, "llama.context_length")? as usize;
        let bos_token_id = read_u32(gguf, "tokenizer.ggml.bos_token_id")?;
        let eos_token_id = read_u32(gguf, "tokenizer.ggml.eos_token_id")?;
        // `llama.vocab_size` isn't always present — derive from the token list when missing.
        let vocab_size = match read_u32(gguf, "llama.vocab_size") {
            Ok(v) => v as usize,
            Err(_) => match gguf.metadata.metadata.get("tokenizer.ggml.tokens") {
                Some(GGUFValue::Array(arr)) => arr.len(),
                _ => {
                    return Err(ConfigError::MissingMetadata(
                        "llama.vocab_size / tokenizer.ggml.tokens".to_string(),
                    ));
                }
            },
        };

        let rope_inv_freq = compute_rope_inv_freq(head_dim, rope_base, gguf);

        Ok(Self {
            n_blocks,
            embed_dim,
            ffn_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_base,
            rms_eps,
            vocab_size,
            context_length,
            bos_token_id,
            eos_token_id,
            rope_inv_freq,
        })
    }
}

/// Computes `inv_freq[j] = 1 / base^(2j / head_dim)` for `j in 0..head_dim/2`,
/// then applies `llama3` rope scaling if the GGUF declares it.
///
/// The llama3 scaling reshapes the frequency table itself (not just runtime
/// theta), so it must be applied at config-load time, before any RoPE call.
/// See the reference impl in HuggingFace `transformers`:
/// `modeling_rope_utils._compute_llama3_parameters`.
fn compute_rope_inv_freq(head_dim: usize, base: f32, gguf: &GGUFFile) -> Vec<f32> {
    let half = head_dim / 2;
    let inv_head_dim = 1.0 / head_dim as f32;
    let mut inv_freq: Vec<f32> = (0..half)
        .map(|j| base.powf(-2.0 * j as f32 * inv_head_dim))
        .collect();

    // Read the GGUF-declared scaling params if present, otherwise fall back
    // to the canonical Llama 3.2 values from the HF config.json. The unsloth
    // GGUF we target doesn't include scaling metadata, so the fallback is
    // load-bearing rather than defensive.
    const LLAMA3_DEFAULT_FACTOR: f32 = 32.0;
    const LLAMA3_DEFAULT_LOW: f32 = 1.0;
    const LLAMA3_DEFAULT_HIGH: f32 = 4.0;
    const LLAMA3_DEFAULT_ORIG_CTX: f32 = 8192.0;

    let factor = read_f32(gguf, "llama.rope.scaling.factor").unwrap_or(LLAMA3_DEFAULT_FACTOR);
    let low_freq_factor =
        read_f32(gguf, "llama.rope.scaling.low_freq_factor").unwrap_or(LLAMA3_DEFAULT_LOW);
    let high_freq_factor =
        read_f32(gguf, "llama.rope.scaling.high_freq_factor").unwrap_or(LLAMA3_DEFAULT_HIGH);
    let orig_ctx = read_u32(gguf, "llama.rope.scaling.original_context_length")
        .map(|v| v as f32)
        .unwrap_or(LLAMA3_DEFAULT_ORIG_CTX);

    let two_pi = std::f32::consts::TAU;
    let low_freq_wavelen = orig_ctx / low_freq_factor;
    let high_freq_wavelen = orig_ctx / high_freq_factor;

    for f in inv_freq.iter_mut() {
        let wavelen = two_pi / *f;
        if wavelen > low_freq_wavelen {
            // Low-frequency (long-wavelength) component: scale down inv_freq.
            *f /= factor;
        } else if wavelen >= high_freq_wavelen {
            // Mid-range: smoothly blend between scaled and unscaled.
            let smooth =
                (orig_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
            *f = (1.0 - smooth) * (*f / factor) + smooth * (*f);
        }
        // High-frequency component: leave as-is.
    }

    inv_freq
}

fn read_u32(gguf: &GGUFFile, key: &str) -> Result<u32, ConfigError> {
    match gguf.metadata.metadata.get(key) {
        None => Err(ConfigError::MissingMetadata(key.to_string())),
        Some(GGUFValue::U32(v)) => Ok(*v),
        Some(_) => Err(ConfigError::WrongMetadataType {
            key: key.to_string(),
            expected: "u32",
        }),
    }
}

fn read_f32(gguf: &GGUFFile, key: &str) -> Result<f32, ConfigError> {
    match gguf.metadata.metadata.get(key) {
        None => Err(ConfigError::MissingMetadata(key.to_string())),
        Some(GGUFValue::F32(v)) => Ok(*v),
        Some(_) => Err(ConfigError::WrongMetadataType {
            key: key.to_string(),
            expected: "f32",
        }),
    }
}

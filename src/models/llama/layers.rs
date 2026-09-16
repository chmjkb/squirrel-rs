use crate::{
    kernels::{kernel_error::KernelError, tensor::TensorView},
    models::{
        common::{kv_cache::KvCache, workspace::Workspace},
        llama::{
            config::LlamaConfig,
            weights::{LlamaAttnWeights, LlamaBlock, LlamaFfnWeights},
        },
    },
    ops::{
        elementwise_add::elementwise_add,
        elementwise_mul::elementwise_mul,
        linear::{linear, linear_shared},
        rms_norm::rms_norm,
        rope::rope_rows,
        sdpa::sdpa_cached,
        silu::silu,
    },
};

/// Attention sub-layer: reads `ws.normed`, writes `ws.attn_out`.
///
/// Expects `ws.normed` already normalized.
pub fn attention(
    ws: &mut Workspace,
    weights: &LlamaAttnWeights,
    cfg: &LlamaConfig,
    position: usize,
    cache: &mut KvCache,
    layer: usize,
) -> Result<(), KernelError> {
    // One shared activation quantization for all three projections.
    linear_shared(
        &ws.normed,
        &mut ws.act_q,
        &[&weights.attn_q, &weights.attn_k, &weights.attn_v],
        &mut [&mut ws.q, &mut ws.k, &mut ws.v],
    )?;

    // Per-row RoPE: token at row i is rotated by absolute position (position + i).
    // For decode (seq_len == 1) this collapses to a single rotation at `position`.
    rope_rows(&mut ws.q, position, &cfg.rope_inv_freq)?;
    rope_rows(&mut ws.k, position, &cfg.rope_inv_freq)?;

    // Store this step's K/V, then attend Q against the full cached history.
    cache.append(layer, ws.k.as_f32_slice()?, ws.v.as_f32_slice()?)?;
    sdpa_cached(
        &ws.q,
        cache.k(layer),
        cache.v(layer),
        &mut ws.attn,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
    )?;

    linear(&ws.attn, &weights.attn_output, &mut ws.attn_out)
}

/// SwiGLU FFN sub-layer: reads `ws.normed`, writes `ws.ffn_out`.
///
/// `down(silu(gate(x)) * up(x))`. Expects `ws.normed` already normalized (the
/// pre-FFN `rms_norm` lives in `block`). Does NOT add the residual.
pub fn ffn(ws: &mut Workspace, weights: &LlamaFfnWeights) -> Result<(), KernelError> {
    // Gate and up share the same normed input — quantize it once.
    linear_shared(
        &ws.normed,
        &mut ws.act_q,
        &[&weights.ffn_gate, &weights.ffn_up],
        &mut [&mut ws.gate, &mut ws.up],
    )?;

    silu(&mut ws.gate)?;
    elementwise_mul(&ws.gate, &ws.up, &mut ws.ffn_act)?;

    linear(&ws.ffn_act, &weights.ffn_down, &mut ws.ffn_out)
}

/// One full transformer block, updating `ws.hidden` in place:
/// `hidden += attention(norm(hidden))` then `hidden += ffn(norm(hidden))`.
pub fn block(
    ws: &mut Workspace,
    weights: &LlamaBlock,
    cfg: &LlamaConfig,
    position: usize,
    cache: &mut KvCache,
    layer: usize,
) -> Result<(), KernelError> {
    rms_norm(
        &ws.hidden,
        &weights.attn.attn_norm,
        &mut ws.normed,
        cfg.rms_eps,
    )?;
    attention(ws, &weights.attn, cfg, position, cache, layer)?;
    elementwise_add(&mut ws.hidden, &ws.attn_out)?;

    // FFN half: hidden += ffn(norm(hidden)).
    rms_norm(
        &ws.hidden,
        &weights.ffn.ffn_norm,
        &mut ws.normed,
        cfg.rms_eps,
    )?;
    ffn(ws, &weights.ffn)?;
    elementwise_add(&mut ws.hidden, &ws.ffn_out)?;

    Ok(())
}

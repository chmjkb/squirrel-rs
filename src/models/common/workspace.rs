//! Reusable scratch buffers for a transformer forward pass.
//!
//! Every intermediate activation (the residual stream, the per-block norm
//! output, Q/K/V, attention output, the FFN temporaries, and the final logits)
//! lives here as an `OwningTensor`.

use crate::kernels::tensor::{BlockQ8_0, OwningTensor};

#[derive(Debug)]
pub struct Workspace {
    embed_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    ffn_dim: usize,

    /// Residual stream, `[seq, embed_dim]`. Embedded into, then updated in place
    /// by each block's attention and FFN residual adds.
    pub hidden: OwningTensor,
    /// Pre-norm output, `[seq, embed_dim]`. Overwritten before attention and again
    /// before the FFN.
    pub normed: OwningTensor,

    /// Attention projections, `[seq, q_dim]` / `[seq, kv_dim]`.
    pub q: OwningTensor,
    pub k: OwningTensor,
    pub v: OwningTensor,
    /// SDPA output `[seq, q_dim]`.
    pub attn: OwningTensor,
    /// Output projection result, `[seq, embed_dim]`.
    pub attn_out: OwningTensor,

    /// FFN gate / up projections, `[seq, ffn_dim]`.
    pub gate: OwningTensor,
    pub up: OwningTensor,
    /// `silu(gate) * up`, `[seq, ffn_dim]`.
    pub ffn_act: OwningTensor,
    /// FFN down projection result, `[seq, embed_dim]`.
    pub ffn_out: OwningTensor,

    /// Last token's hidden row, `[1, embed_dim]`
    pub last_hidden: OwningTensor,
    /// LM-head output, `[1, vocab_size]`.
    pub logits: OwningTensor,

    /// Scratch for Q8_0-quantized activations, shared by projections that read
    /// the same input (Q/K/V, gate/up) so the input is quantized once. Only
    /// filled on the aarch64 integer path, so it stays empty for BF16 weights
    /// and under `SQUIRREL_F32_ACTS`. Capacity persists across passes.
    pub act_q: Vec<BlockQ8_0>,
}

impl Workspace {
    /// Allocate all buffers. Seq-dependent buffers start at `seq = 1` (decode
    /// size) and grow on the first `prepare(prompt_len)`; the fixed-size
    /// `last_hidden` / `logits` are sized once here.
    pub fn new(
        embed_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        ffn_dim: usize,
        vocab_size: usize,
    ) -> Self {
        let row = |dim: usize| OwningTensor::zeros_f32(vec![1, dim]);
        Self {
            embed_dim,
            q_dim,
            kv_dim,
            ffn_dim,
            hidden: row(embed_dim),
            normed: row(embed_dim),
            q: row(q_dim),
            k: row(kv_dim),
            v: row(kv_dim),
            attn: row(q_dim),
            attn_out: row(embed_dim),
            gate: row(ffn_dim),
            up: row(ffn_dim),
            ffn_act: row(ffn_dim),
            ffn_out: row(embed_dim),
            last_hidden: row(embed_dim),
            logits: row(vocab_size),
            act_q: Vec::new(),
        }
    }

    /// Resize the per-token buffers to `seq_len` rows, reusing their existing
    /// allocations. Call once at the start of every forward pass. `last_hidden`
    /// and `logits` are sequence-independent and left as-is.
    pub fn prepare(&mut self, seq_len: usize) {
        self.hidden.resize_f32(vec![seq_len, self.embed_dim]);
        self.normed.resize_f32(vec![seq_len, self.embed_dim]);
        self.q.resize_f32(vec![seq_len, self.q_dim]);
        self.k.resize_f32(vec![seq_len, self.kv_dim]);
        self.v.resize_f32(vec![seq_len, self.kv_dim]);
        self.attn.resize_f32(vec![seq_len, self.q_dim]);
        self.attn_out.resize_f32(vec![seq_len, self.embed_dim]);
        self.gate.resize_f32(vec![seq_len, self.ffn_dim]);
        self.up.resize_f32(vec![seq_len, self.ffn_dim]);
        self.ffn_act.resize_f32(vec![seq_len, self.ffn_dim]);
        self.ffn_out.resize_f32(vec![seq_len, self.embed_dim]);
    }
}

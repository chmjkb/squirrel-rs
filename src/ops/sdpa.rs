use crate::kernels::{attention::sdpa_f32, kernel_error::KernelError, tensor::TensorView};

/// SDPA where K/V come from a flat KV-cache slice rather than a tensor.
///
/// `q`/`out` are the current step's tensors (`seq_q` rows). `k`/`v` are the
/// full cached history, `[seq_k, n_kv_heads * head_dim]` flat — typically
/// borrowed straight from a `KvCache`. `seq_k` is derived from the K slice.
pub fn sdpa_cached(
    q: &impl TensorView,
    k: &[f32],
    v: &[f32],
    out: &mut impl TensorView,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<(), KernelError> {
    if n_kv_heads == 0 || head_dim == 0 {
        return Err(KernelError::ShapeMismatch);
    }
    let q_per_token = n_heads * head_dim;
    let kv_per_token = n_kv_heads * head_dim;
    if !q.numel().is_multiple_of(q_per_token)
        || !k.len().is_multiple_of(kv_per_token)
        || v.len() != k.len()
        || out.numel() != q.numel()
    {
        return Err(KernelError::ShapeMismatch);
    }
    let seq_q = q.numel() / q_per_token;
    let seq_k = k.len() / kv_per_token;

    let q_slice = q.as_f32_slice()?;
    let out_slice = out.as_f32_slice_mut()?;

    sdpa_f32(
        q_slice, k, v, out_slice, seq_q, seq_k, n_heads, n_kv_heads, head_dim,
    )
}

use crate::kernels::kernel_error::KernelError;

/// Rotary positional embedding, interleaved-pair convention (`GGML_ROPE_TYPE_NORM`).
///
/// Rotates adjacent pairs `(x[2j], x[2j+1])`, by a precomputed
/// `inv_freq` table of length `head_dim / 2`.
pub fn rope_f32_interleaved(
    x: &mut [f32],
    position: usize,
    inv_freq: &[f32],
) -> Result<(), KernelError> {
    if x.is_empty() {
        return Err(KernelError::EmptyInput);
    }
    let half = inv_freq.len();
    let head_dim = 2 * half;
    if head_dim == 0 || !x.len().is_multiple_of(head_dim) {
        return Err(KernelError::ShapeMismatch);
    }

    let pos = position as f32;

    for head in x.chunks_exact_mut(head_dim) {
        for j in 0..half {
            let theta = pos * inv_freq[j];
            let (sin_t, cos_t) = theta.sin_cos();

            let a = head[2 * j];
            let b = head[2 * j + 1];
            head[2 * j] = a * cos_t - b * sin_t;
            head[2 * j + 1] = a * sin_t + b * cos_t;
        }
    }

    Ok(())
}

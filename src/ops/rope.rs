use crate::kernels::{kernel_error::KernelError, rope::rope_f32_interleaved, tensor::TensorView};

/// Per-row RoPE: for a tensor shaped `[seq, ..]`, rotates row `i` using
/// position `base_position + i`. Used during prefill where each token has
/// its own absolute position.
pub fn rope_rows(
    t: &mut impl TensorView,
    base_position: usize,
    inv_freq: &[f32],
) -> Result<(), KernelError> {
    let head_dim = 2 * inv_freq.len();
    if head_dim == 0 {
        return Err(KernelError::ShapeMismatch);
    }
    let shape: Vec<usize> = t.shape().to_vec();
    if shape.is_empty() {
        return Err(KernelError::ShapeMismatch);
    }
    let seq = shape[0];
    let row_dim: usize = shape[1..].iter().product();
    if !row_dim.is_multiple_of(head_dim) {
        return Err(KernelError::ShapeMismatch);
    }
    let slice = t.as_f32_slice_mut()?;
    if slice.len() != seq * row_dim {
        return Err(KernelError::ShapeMismatch);
    }
    for (i, row) in slice.chunks_exact_mut(row_dim).enumerate() {
        rope_f32_interleaved(row, base_position + i, inv_freq)?;
    }
    Ok(())
}

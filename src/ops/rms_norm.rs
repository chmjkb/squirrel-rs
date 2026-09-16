use crate::kernels::{
    kernel_error::KernelError,
    rms_norm::{rms_norm_f32_inplace, rms_norm_f32_into},
    tensor::TensorView,
};

/// RMS normalization with a learned scale: `output = (input / rms(input)) * weight`.
///
/// Shapes:
/// - `input`:  `[*, dim]` F32 (any leading dims; normalized along the last axis)
/// - `weight`: `[dim]` F32 (broadcast across the leading dims)
/// - `output`: same shape as `input`, F32, pre-allocated
pub fn rms_norm(
    input: &impl TensorView,
    weight: &impl TensorView,
    output: &mut impl TensorView,
    eps: f32,
) -> Result<(), KernelError> {
    let in_shape = input.shape();
    let w_shape = weight.shape();
    let out_shape = output.shape();

    if in_shape != out_shape {
        return Err(KernelError::ShapeMismatch);
    }
    let dim = *in_shape.last().ok_or(KernelError::ShapeMismatch)?;
    if w_shape.len() != 1 || w_shape[0] != dim {
        return Err(KernelError::ShapeMismatch);
    }

    let in_slice = input.as_f32_slice()?;
    let w_slice = weight.as_f32_slice()?;
    let out_slice = output.as_f32_slice_mut()?;

    for (in_row, out_row) in in_slice
        .chunks_exact(dim)
        .zip(out_slice.chunks_exact_mut(dim))
    {
        rms_norm_f32_into(in_row, w_slice, out_row, eps)?;
    }

    Ok(())
}

/// In-place RMS normalization: `x = (x / rms(x)) * weight`, alloc-free.
///
/// Shapes:
/// - `x`:      `[*, dim]` F32 (any leading dims; normalized along the last axis)
/// - `weight`: `[dim]`    F32 (broadcast across leading dims)
pub fn rms_norm_inplace(
    x: &mut impl TensorView,
    weight: &impl TensorView,
    eps: f32,
) -> Result<(), KernelError> {
    let dim = *x.shape().last().ok_or(KernelError::ShapeMismatch)?;
    let w_shape = weight.shape();
    if w_shape.len() != 1 || w_shape[0] != dim {
        return Err(KernelError::ShapeMismatch);
    }
    let w_slice = weight.as_f32_slice()?;
    let x_slice = x.as_f32_slice_mut()?;
    for row in x_slice.chunks_exact_mut(dim) {
        rms_norm_f32_inplace(row, w_slice, eps)?;
    }
    Ok(())
}

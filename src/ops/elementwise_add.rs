use crate::kernels::{
    elementwise_add::elementwise_add_f32_inplace, kernel_error::KernelError, tensor::TensorView,
};

/// In-place elementwise add: `a += b`. Both tensors must have identical shapes.
///
/// Used for residual connections (`h = h + attention(h)`, `h = h + ffn(h)`).
pub fn elementwise_add(a: &mut impl TensorView, b: &impl TensorView) -> Result<(), KernelError> {
    if a.shape() != b.shape() {
        return Err(KernelError::ShapeMismatch);
    }
    let b_slice = b.as_f32_slice()?;
    let a_slice = a.as_f32_slice_mut()?;
    elementwise_add_f32_inplace(a_slice, b_slice)
}

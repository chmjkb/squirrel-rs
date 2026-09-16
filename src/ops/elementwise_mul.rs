use crate::kernels::{
    elementwise_mul::elementwise_mul_f32, kernel_error::KernelError, tensor::TensorView,
};

/// Elementwise multiplication: `output = a * b`.
///
/// All three tensors must have identical shapes.
pub fn elementwise_mul(
    a: &impl TensorView,
    b: &impl TensorView,
    output: &mut impl TensorView,
) -> Result<(), KernelError> {
    if a.shape() != b.shape() || a.shape() != output.shape() {
        return Err(KernelError::ShapeMismatch);
    }

    let a_slice = a.as_f32_slice()?;
    let b_slice = b.as_f32_slice()?;
    let out_slice = output.as_f32_slice_mut()?;

    elementwise_mul_f32(a_slice, b_slice, out_slice)
}

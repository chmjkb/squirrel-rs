use crate::kernels::{kernel_error::KernelError, silu::silu_f32, tensor::TensorView};

/// In-place SiLU (a.k.a. Swish) activation: `x = x * sigmoid(x)`.
pub fn silu(t: &mut impl TensorView) -> Result<(), KernelError> {
    let slice = t.as_f32_slice_mut()?;
    silu_f32(slice);
    Ok(())
}

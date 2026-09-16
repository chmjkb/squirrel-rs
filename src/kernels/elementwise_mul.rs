use crate::kernels::kernel_error::KernelError;

pub fn elementwise_mul_f32(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), KernelError> {
    if a.len() != b.len() || a.len() != out.len() {
        return Err(KernelError::ShapeMismatch);
    }
    #[cfg(all(feature = "use-optimized-ops", target_arch = "aarch64"))]
    {
        elementwise_mul_f32_simd(a, b, out);
        return Ok(());
    }
    #[allow(unreachable_code)]
    {
        elementwise_mul_f32_fallback(a, b, out);
        Ok(())
    }
}

#[cfg(all(feature = "use-optimized-ops", target_arch = "aarch64"))]
fn elementwise_mul_f32_simd(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::{vld1q_f32, vmulq_f32, vst1q_f32};
    let mut i = 0;
    unsafe {
        while i + 4 <= a.len() {
            let curr_a = vld1q_f32(a.as_ptr().add(i));
            let curr_b = vld1q_f32(b.as_ptr().add(i));
            let prod = vmulq_f32(curr_a, curr_b);
            vst1q_f32(out.as_mut_ptr().add(i), prod);
            i += 4;
        }
    }
    while i < a.len() {
        out[i] = a[i] * b[i];
        i += 1
    }
}

fn elementwise_mul_f32_fallback(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] * b[i]
    }
}

use crate::kernels::kernel_error::KernelError;

pub fn elementwise_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<(), KernelError> {
    if a.len() != b.len() || a.len() != out.len() {
        return Err(KernelError::ShapeMismatch);
    }
    #[cfg(all(feature = "use-optimized-ops", target_arch = "aarch64"))]
    {
        elementwise_add_f32_simd(a, b, out);
        return Ok(());
    }
    #[allow(unreachable_code)]
    {
        elementwise_add_f32_fallback(a, b, out);
        Ok(())
    }
}

fn elementwise_add_f32_fallback(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

#[cfg(all(feature = "use-optimized-ops", target_arch = "aarch64"))]
fn elementwise_add_f32_simd(a: &[f32], b: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::{vaddq_f32, vld1q_f32, vst1q_f32};

    let mut i = 0;
    unsafe {
        while i + 4 <= a.len() {
            let curr_a = vld1q_f32(a.as_ptr().add(i));
            let curr_b = vld1q_f32(b.as_ptr().add(i));
            let sum = vaddq_f32(curr_a, curr_b);
            vst1q_f32(out.as_mut_ptr().add(i), sum);
            i += 4;
        }
    }
    while i < a.len() {
        out[i] = a[i] + b[i];
        i += 1;
    }
}

/// In-place `a[i] += b[i]`. Used for residual connections where the running
/// hidden state is updated by a freshly computed delta (attention or FFN out).
pub fn elementwise_add_f32_inplace(a: &mut [f32], b: &[f32]) -> Result<(), KernelError> {
    if a.len() != b.len() {
        return Err(KernelError::ShapeMismatch);
    }
    #[cfg(all(feature = "use-optimized-ops", target_arch = "aarch64"))]
    {
        elementwise_add_f32_inplace_simd(a, b);
        return Ok(());
    }
    #[allow(unreachable_code)]
    {
        elementwise_add_f32_inplace_fallback(a, b);
        Ok(())
    }
}

fn elementwise_add_f32_inplace_fallback(a: &mut [f32], b: &[f32]) {
    for (av, &bv) in a.iter_mut().zip(b.iter()) {
        *av += bv;
    }
}

#[cfg(all(feature = "use-optimized-ops", target_arch = "aarch64"))]
fn elementwise_add_f32_inplace_simd(a: &mut [f32], b: &[f32]) {
    use std::arch::aarch64::{vaddq_f32, vld1q_f32, vst1q_f32};

    let mut i = 0;
    unsafe {
        while i + 4 <= a.len() {
            let curr_a = vld1q_f32(a.as_ptr().add(i));
            let curr_b = vld1q_f32(b.as_ptr().add(i));
            let sum = vaddq_f32(curr_a, curr_b);
            vst1q_f32(a.as_mut_ptr().add(i), sum);
            i += 4;
        }
    }
    while i < a.len() {
        a[i] += b[i];
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assign_matches_out_of_place() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [10.0, 20.0, 30.0, 40.0];
        let mut expected = [0.0; 4];
        elementwise_add_f32(&a, &b, &mut expected).unwrap();
        let mut x = a;
        elementwise_add_f32_inplace(&mut x, &b).unwrap();
        assert_eq!(x, expected);
    }

    #[test]
    fn assign_length_mismatch_errors() {
        let mut a = [0.0; 4];
        let r = elementwise_add_f32_inplace(&mut a, &[1.0; 3]);
        assert!(matches!(r, Err(KernelError::ShapeMismatch)));
    }
}

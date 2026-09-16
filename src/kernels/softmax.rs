use crate::kernels::kernel_error::KernelError;

/// Softmax over a 1-D slice, written into `out`.
pub fn softmax_f32(a: &[f32], out: &mut [f32]) -> Result<(), KernelError> {
    if a.len() != out.len() {
        return Err(KernelError::ShapeMismatch);
    }
    if a.is_empty() {
        return Err(KernelError::EmptyInput);
    }

    // f32 doesn't implement Ord, so `iter().max()` doesn't work — use fold.
    let max = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0f32;
    for (o, &v) in out.iter_mut().zip(a) {
        *o = (v - max).exp();
        sum += *o;
    }

    let inv = 1.0 / sum;
    for o in out.iter_mut() {
        *o *= inv;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() < tol, "expected {b}, got {a}");
    }

    #[test]
    fn sums_to_one() {
        let mut out = [0.0f32; 3];
        softmax_f32(&[1.0, 2.0, 3.0], &mut out).unwrap();
        approx_eq(out.iter().sum::<f32>(), 1.0, EPS);
        for &v in &out {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn known_values() {
        // input=[1,2,3], shifted=[-2,-1,0], exp=[e^-2, e^-1, 1], normalized
        let mut out = [0.0f32; 3];
        softmax_f32(&[1.0, 2.0, 3.0], &mut out).unwrap();
        approx_eq(out[0], 0.09003057, EPS);
        approx_eq(out[1], 0.24472847, EPS);
        approx_eq(out[2], 0.66524096, EPS);
    }

    #[test]
    fn survives_large_values() {
        // f32::exp overflows past ~88. Without max-subtract these become inf
        // and the result is NaN; with it, they shift to [-1, 0] and work.
        let mut out = [0.0f32; 2];
        softmax_f32(&[100.0, 101.0], &mut out).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
        approx_eq(out.iter().sum::<f32>(), 1.0, EPS);
        assert!(out[1] > out[0]);
    }

    #[test]
    fn preserves_argmax() {
        let mut out = [0.0f32; 4];
        softmax_f32(&[0.1, -5.0, 3.7, 2.0], &mut out).unwrap();
        let arg = out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(arg, 2);
    }

    #[test]
    fn empty_input_errors() {
        let mut out: [f32; 0] = [];
        assert!(matches!(
            softmax_f32(&[], &mut out),
            Err(KernelError::EmptyInput)
        ));
    }
}

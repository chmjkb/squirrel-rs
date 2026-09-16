use crate::kernels::kernel_error::KernelError;

static DEFAULT_RMS_EPSILON: f32 = 1e-5;

/// Computes the RMS normalization.
///
/// For reference on the math, see the following:
/// https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html
pub fn rms_norm_f32(
    input: &[f32],
    weight: &[f32],
    eps: Option<f32>,
) -> Result<Vec<f32>, KernelError> {
    if input.len() != weight.len() {
        return Err(KernelError::ShapeMismatch);
    }
    Ok(rms_norm_fallback(
        input,
        weight,
        eps.unwrap_or(DEFAULT_RMS_EPSILON),
    ))
}

fn rms_norm_fallback(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let squared_sum: f32 = input.iter().map(|e| e * e).sum();
    let mean = squared_sum / (input.len() as f32);
    let rms = (mean + eps).sqrt();

    let mut result = Vec::with_capacity(input.len());

    for (idx, act) in input.iter().enumerate() {
        let y = (act * weight[idx]) / rms;
        result.push(y)
    }
    result
}

/// Out-of-place RMS normalization writing into a caller buffer: no allocation.
/// `dst[i] = (src[i] * weight[i]) / rms(src)`.
pub fn rms_norm_f32_into(
    src: &[f32],
    weight: &[f32],
    dst: &mut [f32],
    eps: f32,
) -> Result<(), KernelError> {
    if src.len() != weight.len() || src.len() != dst.len() {
        return Err(KernelError::ShapeMismatch);
    }
    let squared_sum: f32 = src.iter().map(|v| v * v).sum();
    let mean = squared_sum / (src.len() as f32);
    let rms = (mean + eps).sqrt();
    for (d, (&s, &w)) in dst.iter_mut().zip(src.iter().zip(weight.iter())) {
        *d = (s * w) / rms;
    }
    Ok(())
}

/// In-place RMS normalization: `x[i] = (x[i] * weight[i]) / rms(x)`.
///
/// Safe to alias `x` as both input and output: the squared sum is computed
/// in a full pass before any writes, and the per-element update only reads
/// `x[i]` and the scalar `rms`.
pub fn rms_norm_f32_inplace(x: &mut [f32], weight: &[f32], eps: f32) -> Result<(), KernelError> {
    if x.len() != weight.len() {
        return Err(KernelError::ShapeMismatch);
    }
    let squared_sum: f32 = x.iter().map(|v| v * v).sum();
    let mean = squared_sum / (x.len() as f32);
    let rms = (mean + eps).sqrt();
    for (xi, &w) in x.iter_mut().zip(weight.iter()) {
        *xi = (*xi * w) / rms;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() < tol, "expected {b}, got {a}");
    }

    #[test]
    fn matches_hand_computed() {
        // input=[1,2,3,4], weight=[1,1,1,1], eps=0
        // ss=30, mean=7.5, rms=sqrt(7.5)
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = [1.0, 1.0, 1.0, 1.0];
        let rms = 7.5f32.sqrt();
        let out = rms_norm_f32(&input, &weight, Some(0.0)).unwrap();
        approx_eq(out[0], 1.0 / rms, EPS);
        approx_eq(out[1], 2.0 / rms, EPS);
        approx_eq(out[2], 3.0 / rms, EPS);
        approx_eq(out[3], 4.0 / rms, EPS);
    }

    #[test]
    fn zero_input_stays_finite() {
        // Without eps protection this would divide 0 by 0 and NaN.
        let out = rms_norm_f32(&[0.0; 4], &[1.0; 4], None).unwrap();
        for v in out {
            assert!(v.is_finite());
            approx_eq(v, 0.0, EPS);
        }
    }

    #[test]
    fn weight_scales_output() {
        // Same input, weight=[2,2,2,2] → output is 2x the unit-weight version.
        let input = [1.0, 2.0, 3.0, 4.0];
        let unit = rms_norm_f32(&input, &[1.0; 4], Some(0.0)).unwrap();
        let scaled = rms_norm_f32(&input, &[2.0; 4], Some(0.0)).unwrap();
        for (s, u) in scaled.iter().zip(unit.iter()) {
            approx_eq(*s, 2.0 * u, EPS);
        }
    }

    #[test]
    fn length_mismatch_errors() {
        let r = rms_norm_f32(&[1.0, 2.0], &[1.0; 4], None);
        assert!(matches!(r, Err(KernelError::ShapeMismatch)));
    }

    #[test]
    fn into_matches_allocating_version() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = [0.5, 1.0, 1.5, 2.0];
        let expected = rms_norm_f32(&input, &weight, Some(1e-5)).unwrap();
        let mut dst = [0.0f32; 4];
        rms_norm_f32_into(&input, &weight, &mut dst, 1e-5).unwrap();
        for (a, b) in dst.iter().zip(expected.iter()) {
            approx_eq(*a, *b, EPS);
        }
    }

    #[test]
    fn inplace_matches_allocating_version() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = [0.5, 1.0, 1.5, 2.0];
        let expected = rms_norm_f32(&input, &weight, Some(1e-5)).unwrap();
        let mut x = input;
        rms_norm_f32_inplace(&mut x, &weight, 1e-5).unwrap();
        for (a, b) in x.iter().zip(expected.iter()) {
            approx_eq(*a, *b, EPS);
        }
    }
}

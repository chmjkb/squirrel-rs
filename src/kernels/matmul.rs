use crate::kernels::{kernel_error::KernelError, tensor::WeightBlock};
use rayon::prelude::*;

#[cfg(target_arch = "aarch64")]
use crate::kernels::tensor::BlockQ8_0;

fn validate_shapes<B: WeightBlock>(
    a_len: usize,
    b_len: usize,
    out_len: usize,
    a_rows: usize,
    a_cols: usize,
    b_rows: usize,
    b_cols: usize,
) -> Result<usize, KernelError> {
    if a_cols != b_rows || !a_cols.is_multiple_of(B::ELEMS) {
        return Err(KernelError::ShapeMismatch);
    }
    let blocks_per_row = a_cols / B::ELEMS;
    if a_len != a_rows * a_cols || b_len != b_cols * blocks_per_row || out_len != a_rows * b_cols {
        return Err(KernelError::ShapeMismatch);
    }
    Ok(blocks_per_row)
}

/// Computes several projections of the same activation row in one parallel
/// pass. One output element is one weight row dotted with the activation row.
fn fused_bridge_rows<B: WeightBlock>(
    parts: &mut [(&[B], &mut [f32], usize)],
    a_rows: usize,
    bpr: usize,
    row_dot: impl Fn(usize, &[B]) -> f32 + Sync,
) {
    for r in 0..a_rows {
        let dot = |(o, b_row): (&mut f32, &[B])| *o = row_dot(r, b_row);

        let mut rows = parts
            .iter_mut()
            .map(|(w, out, bc)| (*w, &mut out[r * *bc..(r + 1) * *bc]))
            .collect::<Vec<_>>()
            .into_iter();
        match (rows.next(), rows.next(), rows.next(), rows.next()) {
            (Some((w0, o0)), None, _, _) => {
                o0.par_iter_mut().zip_eq(w0.par_chunks(bpr)).for_each(dot)
            }
            (Some((w0, o0)), Some((w1, o1)), None, _) => o0
                .par_iter_mut()
                .zip_eq(w0.par_chunks(bpr))
                .chain(o1.par_iter_mut().zip_eq(w1.par_chunks(bpr)))
                .for_each(dot),
            (Some((w0, o0)), Some((w1, o1)), Some((w2, o2)), None) => o0
                .par_iter_mut()
                .zip_eq(w0.par_chunks(bpr))
                .chain(o1.par_iter_mut().zip_eq(w1.par_chunks(bpr)))
                .chain(o2.par_iter_mut().zip_eq(w2.par_chunks(bpr)))
                .for_each(dot),
            (first, second, third, _) => {
                for (w, o) in [first, second, third].into_iter().flatten().chain(rows) {
                    o.par_iter_mut().zip_eq(w.par_chunks(bpr)).for_each(dot);
                }
            }
        }
    }
}

/// One projection: `out = mat_a @ mat_b`, f32 activations against quantized
/// weights stored as `b_cols` rows of blocks.
pub fn matmul_f32_q<B: WeightBlock>(
    mat_a: &[f32],
    mat_b: &[B],
    out: &mut [f32],
    a_rows: usize,
    a_cols: usize,
    b_rows: usize,
    b_cols: usize,
) -> Result<(), KernelError> {
    validate_shapes::<B>(
        mat_a.len(),
        mat_b.len(),
        out.len(),
        a_rows,
        a_cols,
        b_rows,
        b_cols,
    )?;

    // Convert the activations to 8-bit once, then multiply integers. Cheaper
    // than turning every weight back into an f32 as it is read.
    #[cfg(target_arch = "aarch64")]
    {
        let mut a_q = vec![
            BlockQ8_0 {
                scale: half::f16::from_f32(0.0),
                qs: [0; 32]
            };
            mat_a.len() / 32
        ];
        crate::kernels::dot_prod::quantize_row_q8_0(mat_a, &mut a_q);
        matmul_prequant_q(&a_q, mat_b, out, a_rows, a_cols, b_cols)
    }

    #[cfg(not(target_arch = "aarch64"))]
    matmul_f32_fused_q(mat_a, vec![(mat_b, out, b_cols)], a_rows, a_cols)
}

/// Several projections of the same activations, kept as f32. Used by BF16 and
/// by the `SQUIRREL_F32_ACTS` ablation.
pub fn matmul_f32_fused_q<B: WeightBlock>(
    mat_a: &[f32],
    mut parts: Vec<(&[B], &mut [f32], usize)>,
    a_rows: usize,
    a_cols: usize,
) -> Result<(), KernelError> {
    if !a_cols.is_multiple_of(B::ELEMS) || a_rows * a_cols != mat_a.len() {
        return Err(KernelError::ShapeMismatch);
    }
    let bpr = a_cols / B::ELEMS;
    for (mat_b, out, b_cols) in parts.iter() {
        if mat_b.len() != b_cols * bpr || out.len() != a_rows * b_cols {
            return Err(KernelError::ShapeMismatch);
        }
    }

    fused_bridge_rows(&mut parts, a_rows, bpr, |r, b_row| {
        B::dot_f32(&mat_a[r * a_cols..(r + 1) * a_cols], b_row)
    });
    Ok(())
}

/// One projection whose activations are already 8-bit. Activations are always
/// Q8_0 blocks of 32, whatever the weight format is.
#[cfg(target_arch = "aarch64")]
pub fn matmul_prequant_q<B: WeightBlock>(
    a_q: &[BlockQ8_0],
    mat_b: &[B],
    out: &mut [f32],
    a_rows: usize,
    a_cols: usize,
    b_cols: usize,
) -> Result<(), KernelError> {
    matmul_prequant_fused_q(a_q, vec![(mat_b, out, b_cols)], a_rows, a_cols)
}

/// Several projections of the same 8-bit activations. The normal decode path.
/// `parts` is one `(weight rows, output, output width)` per projection.
#[cfg(target_arch = "aarch64")]
pub fn matmul_prequant_fused_q<B: WeightBlock>(
    a_q: &[BlockQ8_0],
    mut parts: Vec<(&[B], &mut [f32], usize)>,
    a_rows: usize,
    a_cols: usize,
) -> Result<(), KernelError> {
    if !a_cols.is_multiple_of(B::ELEMS)
        || !a_cols.is_multiple_of(32)
        || a_q.len() != a_rows * (a_cols / 32)
    {
        return Err(KernelError::ShapeMismatch);
    }
    let bpr = a_cols / B::ELEMS;
    for (mat_b, out, b_cols) in parts.iter() {
        if mat_b.len() != b_cols * bpr || out.len() != a_rows * b_cols {
            return Err(KernelError::ShapeMismatch);
        }
    }

    let acts_per_row = a_cols / 32;
    fused_bridge_rows(&mut parts, a_rows, bpr, |r, b_row| {
        B::dot_q8(&a_q[r * acts_per_row..(r + 1) * acts_per_row], b_row)
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::tensor::{BlockBF16, BlockQ8_0};
    use half::bf16;

    /// `a_rows x a_cols` activations, deterministic but not degenerate.
    fn acts(a_rows: usize, a_cols: usize) -> Vec<f32> {
        (0..a_rows * a_cols)
            .map(|i| (i as f32 * 0.037).sin() * 2.0 - 0.5)
            .collect()
    }

    /// `b_cols` weight rows of `a_cols` bf16 values, row by row.
    fn bf16_weights(b_cols: usize, a_cols: usize) -> Vec<BlockBF16> {
        (0..b_cols * (a_cols / 32))
            .map(|blk| {
                let mut v = [bf16::from_f32(0.0); 32];
                for (i, x) in v.iter_mut().enumerate() {
                    *x = bf16::from_f32(((blk * 32 + i) as f32 * 0.011).cos() * 0.4);
                }
                BlockBF16 { v }
            })
            .collect()
    }

    /// The operation written the obvious way, using nothing from this module.
    fn naive(
        mat_a: &[f32],
        mat_b: &[BlockBF16],
        a_rows: usize,
        a_cols: usize,
        b_cols: usize,
    ) -> Vec<f32> {
        let bpr = a_cols / 32;
        let mut out = vec![0.0f32; a_rows * b_cols];
        for r in 0..a_rows {
            for c in 0..b_cols {
                let mut sum = 0.0f32;
                for blk in 0..bpr {
                    let w = &mat_b[c * bpr + blk];
                    for i in 0..32 {
                        sum += mat_a[r * a_cols + blk * 32 + i] * w.v[i].to_f32();
                    }
                }
                out[r * b_cols + c] = sum;
            }
        }
        out
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol * w.abs().max(1.0),
                "element {i}: {g} vs {w}"
            );
        }
    }

    #[test]
    fn matches_naive_single_row() {
        let (a_rows, a_cols, b_cols) = (1, 128, 7);
        let a = acts(a_rows, a_cols);
        let w = bf16_weights(b_cols, a_cols);
        let mut out = vec![0.0; a_rows * b_cols];
        matmul_f32_fused_q(&a, vec![(&w[..], &mut out[..], b_cols)], a_rows, a_cols).unwrap();
        assert_close(&out, &naive(&a, &w, a_rows, a_cols, b_cols), 1e-4);
    }

    /// The prefill shape: several activation rows through the same weights.
    #[test]
    fn matches_naive_multi_row() {
        let (a_rows, a_cols, b_cols) = (5, 96, 9);
        let a = acts(a_rows, a_cols);
        let w = bf16_weights(b_cols, a_cols);
        let mut out = vec![0.0; a_rows * b_cols];
        matmul_f32_fused_q(&a, vec![(&w[..], &mut out[..], b_cols)], a_rows, a_cols).unwrap();
        assert_close(&out, &naive(&a, &w, a_rows, a_cols, b_cols), 1e-4);
    }

    /// Projections run together must match the same projections run singly.
    #[test]
    fn fused_matches_separate_calls() {
        let (a_rows, a_cols) = (3, 64);
        let a = acts(a_rows, a_cols);
        let dims = [4usize, 6, 5];
        let ws: Vec<Vec<BlockBF16>> = dims.iter().map(|&c| bf16_weights(c, a_cols)).collect();

        let mut separate: Vec<Vec<f32>> = dims.iter().map(|&c| vec![0.0; a_rows * c]).collect();
        for ((w, out), &c) in ws.iter().zip(separate.iter_mut()).zip(dims.iter()) {
            matmul_f32_fused_q(&a, vec![(&w[..], &mut out[..], c)], a_rows, a_cols).unwrap();
        }

        let mut fused: Vec<Vec<f32>> = dims.iter().map(|&c| vec![0.0; a_rows * c]).collect();
        {
            let parts: Vec<_> = ws
                .iter()
                .zip(fused.iter_mut())
                .zip(dims.iter())
                .map(|((w, out), &c)| (&w[..], &mut out[..], c))
                .collect();
            matmul_f32_fused_q(&a, parts, a_rows, a_cols).unwrap();
        }
        assert_eq!(fused, separate);
    }

    /// The integer path rounds activations to 8 bits, so it only has to stay
    /// within that rounding error of the f32 path.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn prequant_path_close_to_f32_path() {
        let (a_rows, a_cols, b_cols) = (2, 128, 6);
        let a = acts(a_rows, a_cols);
        let w: Vec<BlockQ8_0> = (0..b_cols * (a_cols / 32))
            .map(|blk| {
                let mut qs = [0i8; 32];
                for (i, q) in qs.iter_mut().enumerate() {
                    *q = (((i as i32 * 7 + blk as i32 * 13) % 251) - 125) as i8;
                }
                BlockQ8_0 {
                    scale: half::f16::from_f32(0.001 * (blk as f32 + 1.0)),
                    qs,
                }
            })
            .collect();

        let mut want = vec![0.0; a_rows * b_cols];
        matmul_f32_fused_q(&a, vec![(&w[..], &mut want[..], b_cols)], a_rows, a_cols).unwrap();

        let mut a_q = vec![
            BlockQ8_0 {
                scale: half::f16::from_f32(0.0),
                qs: [0; 32]
            };
            a.len() / 32
        ];
        crate::kernels::dot_prod::quantize_row_q8_0(&a, &mut a_q);
        let mut got = vec![0.0; a_rows * b_cols];
        matmul_prequant_q(&a_q, &w, &mut got, a_rows, a_cols, b_cols).unwrap();

        // The error grows with how much was added up, not with how big the
        // answer is: one element here sums terms worth 95 and lands on -2.7.
        // So the tolerance is measured against the terms.
        let bpr = a_cols / 32;
        for r in 0..a_rows {
            for c in 0..b_cols {
                let mut mag = 0.0f32;
                for blk in 0..bpr {
                    let wb = &w[c * bpr + blk];
                    let ws = wb.scale.to_f32();
                    for i in 0..32 {
                        mag += (a[r * a_cols + blk * 32 + i] * wb.qs[i] as f32 * ws).abs();
                    }
                }
                let (g, e) = (got[r * b_cols + c], want[r * b_cols + c]);
                assert!(
                    (g - e).abs() <= 5e-3 * mag,
                    "r{r} c{c}: {g} vs {e}, terms worth {mag}"
                );
            }
        }
    }

    #[test]
    fn rejects_mismatched_shapes() {
        let (a_rows, a_cols, b_cols) = (1, 64, 4);
        let a = acts(a_rows, a_cols);
        let w = bf16_weights(b_cols, a_cols);
        let mut out = vec![0.0; a_rows * b_cols];
        // a_cols claimed larger than the activations actually hold.
        let r = matmul_f32_fused_q(
            &a,
            vec![(&w[..], &mut out[..], b_cols)],
            a_rows,
            a_cols + 32,
        );
        assert!(matches!(r, Err(KernelError::ShapeMismatch)));
    }

    #[test]
    fn rejects_non_block_multiple_cols() {
        let (a_rows, a_cols, b_cols) = (1, 40, 2);
        let a = acts(a_rows, a_cols);
        let w = bf16_weights(b_cols, 64);
        let mut out = vec![0.0; a_rows * b_cols];
        let r = matmul_f32_fused_q(&a, vec![(&w[..], &mut out[..], b_cols)], a_rows, a_cols);
        assert!(matches!(r, Err(KernelError::ShapeMismatch)));
    }
}

use crate::kernels::{dot_prod::dot_f32_f32, kernel_error::KernelError, softmax::softmax_f32};
use rayon::prelude::*;

/// Scaled dot-product attention with a causal mask and grouped-query support.
///
/// Inputs are flat f32 slices shaped as following:
/// - `q`:   `[seq_q, n_heads,    head_dim]`
/// - `k/v`: `[seq_k, n_kv_heads, head_dim]`
/// - `out`: `[seq_q, n_heads,    head_dim]`
#[allow(clippy::too_many_arguments)]
pub fn sdpa_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_q: usize,
    seq_k: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<(), KernelError> {
    if n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
        return Err(KernelError::ShapeMismatch);
    }
    if q.len() != seq_q * n_heads * head_dim
        || k.len() != seq_k * n_kv_heads * head_dim
        || v.len() != seq_k * n_kv_heads * head_dim
        || out.len() != seq_q * n_heads * head_dim
    {
        return Err(KernelError::ShapeMismatch);
    }
    // Decode against a KV cache: seq_q rows correspond to the last seq_q
    // positions in the cache, so each query t attends to keys 0..=offset+t.
    let offset = seq_k.checked_sub(seq_q).ok_or(KernelError::ShapeMismatch)?;

    let group_size = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Every (t, h) pair is independent, scores, softmax, and the weighted
    // sum touch only that pair's rows, so parallelize across the output's
    // head rows.
    out.par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(idx, out_row)| {
            let t = idx / n_heads;
            let h = idx % n_heads;
            let kv_h = h / group_size;
            let live = offset + t + 1; // attend to keys 0..live
            let q_row = &q[(t * n_heads + h) * head_dim..][..head_dim];

            let mut scores = vec![0.0f32; live];
            for (s, score) in scores.iter_mut().enumerate() {
                let k_row = &k[(s * n_kv_heads + kv_h) * head_dim..][..head_dim];
                *score = dot_f32_f32(q_row, k_row).expect("validated head_dim") * scale;
            }

            let mut probs = vec![0.0f32; live];
            softmax_f32(&scores, &mut probs).expect("live >= 1 rows");

            out_row.fill(0.0);
            for (s, &p) in probs.iter().enumerate() {
                let v_row = &v[(s * n_kv_heads + kv_h) * head_dim..][..head_dim];
                for (o, &vv) in out_row.iter_mut().zip(v_row.iter()) {
                    *o += p * vv;
                }
            }
        });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() < tol, "expected {b}, got {a}");
    }

    /// Single token, single head, k == v == q. Softmax over one element is
    /// [1.0], so the output must equal v. Sanity-checks indexing and scale.
    #[test]
    fn single_token_single_head_returns_v() {
        let q = vec![0.5, -1.0, 2.0, 0.25];
        let k = q.clone();
        let v = vec![7.0, 8.0, 9.0, 10.0];
        let mut out = vec![0.0; 4];
        sdpa_f32(&q, &k, &v, &mut out, 1, 1, 1, 1, 4).unwrap();
        for (o, vv) in out.iter().zip(v.iter()) {
            approx_eq(*o, *vv, EPS);
        }
    }

    /// Causal mask: at t=0 the model can only see s=0, so its output must
    /// equal v[0] regardless of what v[1] contains.
    #[test]
    fn causal_mask_hides_future() {
        let q = vec![1.0, 0.0, 0.5, 0.5];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v_a = vec![3.0, 4.0, 99.0, -99.0];
        let v_b = vec![3.0, 4.0, -42.0, 1234.0];
        let mut out_a = vec![0.0; 4];
        let mut out_b = vec![0.0; 4];
        sdpa_f32(&q, &k, &v_a, &mut out_a, 2, 2, 1, 1, 2).unwrap();
        sdpa_f32(&q, &k, &v_b, &mut out_b, 2, 2, 1, 1, 2).unwrap();
        // First query row (t=0) must be identical across the two runs.
        approx_eq(out_a[0], out_b[0], EPS);
        approx_eq(out_a[1], out_b[1], EPS);
        // And it must equal v[0] (the only key/value it can see).
        approx_eq(out_a[0], 3.0, EPS);
        approx_eq(out_a[1], 4.0, EPS);
    }

    #[test]
    fn gqa_shares_kv_heads() {
        let seq = 1;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 2;

        // Q: heads 0 and 1 identical; heads 2 and 3 identical.
        let q = vec![
            1.0, 0.0, // head 0
            1.0, 0.0, // head 1 (== head 0)
            0.0, 1.0, // head 2
            0.0, 1.0, // head 3 (== head 2)
        ];
        let k = vec![1.0, 1.0, 0.5, -0.5]; // [seq=1, n_kv_heads=2, head_dim=2]
        let v = vec![10.0, 20.0, 30.0, 40.0];
        let mut out = vec![0.0; seq * n_heads * head_dim];

        sdpa_f32(
            &q, &k, &v, &mut out, seq, seq, n_heads, n_kv_heads, head_dim,
        )
        .unwrap();

        // Head 0 and head 1 should produce identical output (both → kv_head 0).
        approx_eq(out[0], out[2], EPS);
        approx_eq(out[1], out[3], EPS);
        // Head 2 and head 3 likewise (both → kv_head 1).
        approx_eq(out[4], out[6], EPS);
        approx_eq(out[5], out[7], EPS);
    }

    /// At seq_q = seq_k = 1 the kernel reduces to one dot product per head,
    /// passed through softmax (trivially 1.0), and produces v exactly.
    #[test]
    fn decode_step_against_single_cached_token() {
        let q = vec![0.1, 0.2, 0.3, 0.4];
        let k = vec![1.0, 0.0, -1.0, 2.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0; 4];
        sdpa_f32(&q, &k, &v, &mut out, 1, 1, 2, 2, 2).unwrap();
        // Each head sees one key → softmax = [1.0] → out = v.
        for (o, vv) in out.iter().zip(v.iter()) {
            approx_eq(*o, *vv, EPS);
        }
    }

    #[test]
    fn rejects_bad_group_ratio() {
        let q = vec![0.0; 6];
        let k = vec![0.0; 6];
        let v = vec![0.0; 6];
        let mut out = vec![0.0; 6];
        // n_heads=3, n_kv_heads=2 — not evenly divisible.
        let r = sdpa_f32(&q, &k, &v, &mut out, 1, 1, 3, 2, 2);
        assert!(matches!(r, Err(KernelError::ShapeMismatch)));
    }
}

use crate::file_parser::ggml_type::GGMLType;
use crate::kernels::{
    kernel_error::KernelError,
    tensor::{BlockBF16, BlockQ4_0, BlockQ8_0, WeightBlock},
};
use half::f16;

/// Convert an f32 row into Q8_0 blocks: per 32 values, scale by the largest
/// magnitude and round to i8. `out` needs `src.len() / 32` blocks.
pub fn quantize_row_q8_0(src: &[f32], out: &mut [BlockQ8_0]) {
    #[cfg(all(target_arch = "aarch64", feature = "use-optimized-ops"))]
    {
        quantize_row_q8_0_neon(src, out);
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "use-optimized-ops")))]
    {
        quantize_row_q8_0_scalar(src, out);
    }
}

fn quantize_row_q8_0_scalar(src: &[f32], out: &mut [BlockQ8_0]) {
    for (chunk, block) in src.chunks_exact(32).zip(out.iter_mut()) {
        let amax = chunk.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        let scale = amax / 127.0;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        block.scale = f16::from_f32(scale);
        for (q, &x) in block.qs.iter_mut().zip(chunk.iter()) {
            *q = (x * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn quantize_row_q8_0_neon(src: &[f32], out: &mut [BlockQ8_0]) {
    use std::arch::aarch64::*;

    unsafe {
        for (chunk, block) in src.chunks_exact(32).zip(out.iter_mut()) {
            let p = chunk.as_ptr();
            let v: [float32x4_t; 8] = [
                vld1q_f32(p),
                vld1q_f32(p.add(4)),
                vld1q_f32(p.add(8)),
                vld1q_f32(p.add(12)),
                vld1q_f32(p.add(16)),
                vld1q_f32(p.add(20)),
                vld1q_f32(p.add(24)),
                vld1q_f32(p.add(28)),
            ];

            // Largest magnitude in the block
            let m01 = vmaxq_f32(vabsq_f32(v[0]), vabsq_f32(v[1]));
            let m23 = vmaxq_f32(vabsq_f32(v[2]), vabsq_f32(v[3]));
            let m45 = vmaxq_f32(vabsq_f32(v[4]), vabsq_f32(v[5]));
            let m67 = vmaxq_f32(vabsq_f32(v[6]), vabsq_f32(v[7]));
            let amax = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m01, m23), vmaxq_f32(m45, m67)));

            let scale = amax / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            block.scale = f16::from_f32(scale);

            let lo = vdupq_n_s32(-127);
            let hi = vdupq_n_s32(127);
            let mut q16 = [vdupq_n_s16(0); 4];
            for (pair, q) in q16.iter_mut().enumerate() {
                let a = vminq_s32(
                    vmaxq_s32(vcvtaq_s32_f32(vmulq_n_f32(v[2 * pair], inv)), lo),
                    hi,
                );
                let b = vminq_s32(
                    vmaxq_s32(vcvtaq_s32_f32(vmulq_n_f32(v[2 * pair + 1], inv)), lo),
                    hi,
                );
                *q = vcombine_s16(vmovn_s32(a), vmovn_s32(b));
            }
            let out_p = block.qs.as_mut_ptr();
            vst1q_s8(out_p, vcombine_s8(vmovn_s16(q16[0]), vmovn_s16(q16[1])));
            vst1q_s8(
                out_p.add(16),
                vcombine_s8(vmovn_s16(q16[2]), vmovn_s16(q16[3])),
            );
        }
    }
}

fn dot_f32_q8_0_fallback(a: &[f32], b: &[BlockQ8_0]) -> f32 {
    let mut acc = 0.0;
    for (chunk, block) in a.chunks_exact(32).zip(b.iter()) {
        let mut block_acc = 0.0;
        for (val_a, val_b) in chunk.iter().zip(block.qs) {
            block_acc += val_a * (val_b as f32)
        }
        acc += block_acc * block.scale.to_f32();
    }
    acc
}

/// f32 activations against Q8_0 weights. Vectorized form of dot_f32_q8_0_fallback
#[cfg(target_arch = "aarch64")]
fn dot_f32_q8_0_neon(a: &[f32], b: &[BlockQ8_0]) -> f32 {
    use std::arch::aarch64::*;

    unsafe {
        // running total
        let mut gacc = vdupq_n_f32(0.0);

        for (chunk, block) in a.chunks_exact(32).zip(b.iter()) {
            // Load 32 i8 weights as two 16-lane vectors.
            let q_lo = vld1q_s8(block.qs.as_ptr());
            let q_hi = vld1q_s8(block.qs.as_ptr().add(16));

            // Widen i8 -> i16 -> i32 -> f32: eight vectors of four.
            let q01 = vmovl_s8(vget_low_s8(q_lo));
            let q23 = vmovl_s8(vget_high_s8(q_lo));
            let q45 = vmovl_s8(vget_low_s8(q_hi));
            let q67 = vmovl_s8(vget_high_s8(q_hi));
            let qf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q01)));
            let qf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q01)));
            let qf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q23)));
            let qf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q23)));
            let qf4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q45)));
            let qf5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q45)));
            let qf6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q67)));
            let qf7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q67)));

            // Four separate chains, so none waits on another.
            let p = chunk.as_ptr();
            let mut acc0 = vmulq_f32(vld1q_f32(p), qf0);
            let mut acc1 = vmulq_f32(vld1q_f32(p.add(4)), qf1);
            let mut acc2 = vmulq_f32(vld1q_f32(p.add(8)), qf2);
            let mut acc3 = vmulq_f32(vld1q_f32(p.add(12)), qf3);
            acc0 = vfmaq_f32(acc0, vld1q_f32(p.add(16)), qf4);
            acc1 = vfmaq_f32(acc1, vld1q_f32(p.add(20)), qf5);
            acc2 = vfmaq_f32(acc2, vld1q_f32(p.add(24)), qf6);
            acc3 = vfmaq_f32(acc3, vld1q_f32(p.add(28)), qf7);

            // Collapse the four accumulators, apply the block's scale, add to the total.
            let acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            gacc = vfmaq_n_f32(gacc, acc, block.scale.to_f32());
        }

        vaddvq_f32(gacc)
    }
}

/// 16 i8 pairs multiplied and summed into four i32 lanes, in one instruction.
/// Rust's intrinsic for is unstable, so it is written as inline asm.
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
#[inline(always)]
unsafe fn sdot(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let mut r = acc;
    std::arch::asm!(
        "sdot {r:v}.4s, {a:v}.16b, {b:v}.16b",
        r = inout(vreg) r,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack)
    );
    r
}

/// One block: 32 i8 pairs -> four i32 lanes. Two SDOTs where the CPU has it,
/// otherwise widen to i16 and pairwise-add.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mac_block_i8(pa: *const i8, pb: *const i8) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;
    #[cfg(target_feature = "dotprod")]
    {
        let acc = sdot(vdupq_n_s32(0), vld1q_s8(pa), vld1q_s8(pb));
        sdot(acc, vld1q_s8(pa.add(16)), vld1q_s8(pb.add(16)))
    }
    #[cfg(not(target_feature = "dotprod"))]
    {
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        acc0 = vpadalq_s16(acc0, vmull_s8(vld1_s8(pa), vld1_s8(pb)));
        acc1 = vpadalq_s16(acc1, vmull_s8(vld1_s8(pa.add(8)), vld1_s8(pb.add(8))));
        acc0 = vpadalq_s16(acc0, vmull_s8(vld1_s8(pa.add(16)), vld1_s8(pb.add(16))));
        acc1 = vpadalq_s16(acc1, vmull_s8(vld1_s8(pa.add(24)), vld1_s8(pb.add(24))));
        vaddq_s32(acc0, acc1)
    }
}

/// Integer dot of two Q8_0 rows.
#[cfg(target_arch = "aarch64")]
pub fn dot_q8_0_q8_0_neon(a: &[BlockQ8_0], b: &[BlockQ8_0]) -> f32 {
    use std::arch::aarch64::*;

    unsafe {
        // Four running totals over four blocks at a time.
        let mut gacc0 = vdupq_n_f32(0.0);
        let mut gacc1 = vdupq_n_f32(0.0);
        let mut gacc2 = vdupq_n_f32(0.0);
        let mut gacc3 = vdupq_n_f32(0.0);

        let n = a.len().min(b.len());
        let mut i = 0;
        while i + 4 <= n {
            let (a0, b0) = (a.get_unchecked(i), b.get_unchecked(i));
            let (a1, b1) = (a.get_unchecked(i + 1), b.get_unchecked(i + 1));
            let (a2, b2) = (a.get_unchecked(i + 2), b.get_unchecked(i + 2));
            let (a3, b3) = (a.get_unchecked(i + 3), b.get_unchecked(i + 3));
            //  i8 pairs multiplied and summed into four i32 lanes. Nothing is converted to f32 yet.
            let s0 = mac_block_i8(a0.qs.as_ptr(), b0.qs.as_ptr());
            let s1 = mac_block_i8(a1.qs.as_ptr(), b1.qs.as_ptr());
            let s2 = mac_block_i8(a2.qs.as_ptr(), b2.qs.as_ptr());
            let s3 = mac_block_i8(a3.qs.as_ptr(), b3.qs.as_ptr());
            // apply scales once per
            // block: total += lanes_as_f32 * (activation scale * weight scale).
            gacc0 = vfmaq_n_f32(
                gacc0,
                vcvtq_f32_s32(s0),
                a0.scale.to_f32() * b0.scale.to_f32(),
            );
            gacc1 = vfmaq_n_f32(
                gacc1,
                vcvtq_f32_s32(s1),
                a1.scale.to_f32() * b1.scale.to_f32(),
            );
            gacc2 = vfmaq_n_f32(
                gacc2,
                vcvtq_f32_s32(s2),
                a2.scale.to_f32() * b2.scale.to_f32(),
            );
            gacc3 = vfmaq_n_f32(
                gacc3,
                vcvtq_f32_s32(s3),
                a3.scale.to_f32() * b3.scale.to_f32(),
            );
            i += 4;
        }
        // Whatever is left when the block count is not a multiple of four.
        while i < n {
            let (ba, bb) = (a.get_unchecked(i), b.get_unchecked(i));
            let isum4 = mac_block_i8(ba.qs.as_ptr(), bb.qs.as_ptr());
            gacc0 = vfmaq_n_f32(
                gacc0,
                vcvtq_f32_s32(isum4),
                ba.scale.to_f32() * bb.scale.to_f32(),
            );
            i += 1;
        }

        // Collapse the four totals, then sum the 4 lanes into one number.
        vaddvq_f32(vaddq_f32(vaddq_f32(gacc0, gacc1), vaddq_f32(gacc2, gacc3)))
    }
}

/// f32 activations against Q4_0 weights.
fn dot_f32_q4_0_fallback(a: &[f32], b: &[BlockQ4_0]) -> f32 {
    let mut acc = 0.0;
    for (chunk, block) in a.chunks_exact(32).zip(b.iter()) {
        let mut block_acc = 0.0;
        // 16 bytes, two weights each: the bottom 4 bits are weight j, the top 4
        // are weight j+16
        for (j, &byte) in block.qs.iter().enumerate() {
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = (byte >> 4) as i32 - 8;
            block_acc += chunk[j] * lo as f32 + chunk[j + 16] * hi as f32;
        }
        // The scale is the same for all 32, so it is applied once at the end.
        acc += block_acc * block.scale.to_f32();
    }
    acc
}

/// One Q4_0 block against 32 i8 activations. Unpacks the 16 packed bytes into
/// 32 signed weights — low nibbles are elements 0..16, high nibbles 16..32,
/// both biased by -8 — then multiplies as usual.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mac_block_q4(
    pa: *const i8,
    qs_packed: *const u8,
    mask: std::arch::aarch64::uint8x16_t,
    eight: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;
    let qs = vld1q_u8(qs_packed);
    // Byte j holds weight j in its bottom 4 bits and weight j+16 in its top 4.
    // Keep the bottom / shift the top down, then undo the -8 bias on both.
    let w_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(qs, mask)), eight);
    let w_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qs, 4)), eight);
    #[cfg(target_feature = "dotprod")]
    {
        let acc = sdot(vdupq_n_s32(0), vld1q_s8(pa), w_lo);
        sdot(acc, vld1q_s8(pa.add(16)), w_hi)
    }
    #[cfg(not(target_feature = "dotprod"))]
    {
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        acc0 = vpadalq_s16(acc0, vmull_s8(vget_low_s8(w_lo), vld1_s8(pa)));
        acc1 = vpadalq_s16(acc1, vmull_s8(vget_high_s8(w_lo), vld1_s8(pa.add(8))));
        acc0 = vpadalq_s16(acc0, vmull_s8(vget_low_s8(w_hi), vld1_s8(pa.add(16))));
        acc1 = vpadalq_s16(acc1, vmull_s8(vget_high_s8(w_hi), vld1_s8(pa.add(24))));
        vaddq_s32(acc0, acc1)
    }
}

#[cfg(target_arch = "aarch64")]
pub fn dot_q8_0_q4_0_neon(a: &[BlockQ8_0], b: &[BlockQ4_0]) -> f32 {
    use std::arch::aarch64::*;

    unsafe {
        // Q4_0 packs two weights per byte, 4 bits each. Four bits hold 0..15,
        // so the format stores `q` and means `q - 8`, giving -8..7. These two
        // constants do that: keep the bottom 4 bits, then subtract 8.
        let mask = vdupq_n_u8(0x0F);
        let eight = vdupq_n_s8(8);

        // Same structure as the Q8 dot above: four running totals, four blocks
        // per pass, one horizontal sum at the very end.
        let mut gacc0 = vdupq_n_f32(0.0);
        let mut gacc1 = vdupq_n_f32(0.0);
        let mut gacc2 = vdupq_n_f32(0.0);
        let mut gacc3 = vdupq_n_f32(0.0);

        let n = a.len().min(b.len());
        let mut i = 0;
        while i + 4 <= n {
            let (a0, b0) = (a.get_unchecked(i), b.get_unchecked(i));
            let (a1, b1) = (a.get_unchecked(i + 1), b.get_unchecked(i + 1));
            let (a2, b2) = (a.get_unchecked(i + 2), b.get_unchecked(i + 2));
            let (a3, b3) = (a.get_unchecked(i + 3), b.get_unchecked(i + 3));
            // Each call unpacks one block's 16 bytes into 32 weights and
            // multiplies them by the activations: four i32 lane sums back.
            let s0 = mac_block_q4(a0.qs.as_ptr(), b0.qs.as_ptr(), mask, eight);
            let s1 = mac_block_q4(a1.qs.as_ptr(), b1.qs.as_ptr(), mask, eight);
            let s2 = mac_block_q4(a2.qs.as_ptr(), b2.qs.as_ptr(), mask, eight);
            let s3 = mac_block_q4(a3.qs.as_ptr(), b3.qs.as_ptr(), mask, eight);
            gacc0 = vfmaq_n_f32(
                gacc0,
                vcvtq_f32_s32(s0),
                a0.scale.to_f32() * b0.scale.to_f32(),
            );
            gacc1 = vfmaq_n_f32(
                gacc1,
                vcvtq_f32_s32(s1),
                a1.scale.to_f32() * b1.scale.to_f32(),
            );
            gacc2 = vfmaq_n_f32(
                gacc2,
                vcvtq_f32_s32(s2),
                a2.scale.to_f32() * b2.scale.to_f32(),
            );
            gacc3 = vfmaq_n_f32(
                gacc3,
                vcvtq_f32_s32(s3),
                a3.scale.to_f32() * b3.scale.to_f32(),
            );
            i += 4;
        }
        // Whatever is left when the block count is not a multiple of four.
        while i < n {
            let (ba, bb) = (a.get_unchecked(i), b.get_unchecked(i));
            let isum4 = mac_block_q4(ba.qs.as_ptr(), bb.qs.as_ptr(), mask, eight);
            gacc0 = vfmaq_n_f32(
                gacc0,
                vcvtq_f32_s32(isum4),
                ba.scale.to_f32() * bb.scale.to_f32(),
            );
            i += 1;
        }

        // Collapse the four totals, then sum the 4 lanes into one number.
        vaddvq_f32(vaddq_f32(vaddq_f32(gacc0, gacc1), vaddq_f32(gacc2, gacc3)))
    }
}

/// Scalar twin of the kernel above. Run by the `use-optimized-ops` ablation,
/// and used as the reference the NEON version is tested against.
fn dot_q8_0_q8_0_scalar(a: &[BlockQ8_0], b: &[BlockQ8_0]) -> f32 {
    let mut total = 0.0f32;
    for (ba, bb) in a.iter().zip(b.iter()) {
        let mut isum = 0i32;
        for (&qa, &qb) in ba.qs.iter().zip(bb.qs.iter()) {
            isum += qa as i32 * qb as i32;
        }
        total += isum as f32 * ba.scale.to_f32() * bb.scale.to_f32();
    }
    total
}

/// Scalar twin of the Q4_0 kernel, unpacking each byte by hand.
fn dot_q8_0_q4_0_scalar(a: &[BlockQ8_0], b: &[BlockQ4_0]) -> f32 {
    let mut total = 0.0f32;
    for (ba, bb) in a.iter().zip(b.iter()) {
        let mut isum = 0i32;
        for (j, &byte) in bb.qs.iter().enumerate() {
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = (byte >> 4) as i32 - 8;
            isum += ba.qs[j] as i32 * lo + ba.qs[j + 16] as i32 * hi;
        }
        total += isum as f32 * ba.scale.to_f32() * bb.scale.to_f32();
    }
    total
}

impl WeightBlock for BlockQ8_0 {
    const ELEMS: usize = 32;
    const DTYPE: GGMLType = GGMLType::Q8_0;

    fn dequantize_into(&self, out: &mut [f32]) {
        let scale = self.scale.to_f32();
        for (o, &q) in out.iter_mut().zip(self.qs.iter()) {
            *o = scale * q as f32;
        }
    }

    fn dot_f32(acts: &[f32], w: &[Self]) -> f32 {
        #[cfg(all(target_arch = "aarch64", feature = "use-optimized-ops"))]
        {
            return dot_f32_q8_0_neon(acts, w);
        }
        #[allow(unreachable_code)]
        dot_f32_q8_0_fallback(acts, w)
    }

    #[cfg(target_arch = "aarch64")]
    fn dot_q8(acts: &[BlockQ8_0], w: &[Self]) -> f32 {
        #[cfg(feature = "use-optimized-ops")]
        {
            return dot_q8_0_q8_0_neon(acts, w);
        }
        #[allow(unreachable_code)]
        dot_q8_0_q8_0_scalar(acts, w)
    }
}

impl WeightBlock for BlockQ4_0 {
    const ELEMS: usize = 32;
    const DTYPE: GGMLType = GGMLType::Q4_0;

    fn dequantize_into(&self, out: &mut [f32]) {
        let scale = self.scale.to_f32();
        for (j, &byte) in self.qs.iter().enumerate() {
            out[j] = scale * ((byte & 0x0F) as i32 - 8) as f32;
            out[j + 16] = scale * ((byte >> 4) as i32 - 8) as f32;
        }
    }

    fn dot_f32(acts: &[f32], w: &[Self]) -> f32 {
        dot_f32_q4_0_fallback(acts, w)
    }

    #[cfg(target_arch = "aarch64")]
    fn dot_q8(acts: &[BlockQ8_0], w: &[Self]) -> f32 {
        #[cfg(feature = "use-optimized-ops")]
        {
            return dot_q8_0_q4_0_neon(acts, w);
        }
        #[allow(unreachable_code)]
        dot_q8_0_q4_0_scalar(acts, w)
    }
}

/// f32 activations against raw BF16 weights (scalar reference).
fn dot_f32_bf16_fallback(a: &[f32], b: &[BlockBF16]) -> f32 {
    let mut acc = 0.0;
    for (chunk, block) in a.chunks_exact(32).zip(b.iter()) {
        for (x, w) in chunk.iter().zip(block.v.iter()) {
            acc += x * w.to_f32();
        }
    }
    acc
}

/// f32 activations against BF16 weights. A bf16 is the top half of an f32, so
/// widening is a 16-bit shift and a reinterpret, with no conversion at all.
#[cfg(target_arch = "aarch64")]
fn dot_f32_bf16_neon(a: &[f32], b: &[BlockBF16]) -> f32 {
    use std::arch::aarch64::*;

    /// 4 bf16 -> 4 f32: shift the bits into the top half and relabel them.
    #[inline(always)]
    unsafe fn widen(half: uint16x4_t) -> float32x4_t {
        vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(half)))
    }

    unsafe {
        // Four running totals instead of one, so consecutive adds do not wait
        // on each other. Each holds 4 f32, because a vector register is 128 bits.
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        for (chunk, block) in a.chunks_exact(32).zip(b.iter()) {
            let pw = block.v.as_ptr() as *const u16;
            let pa = chunk.as_ptr();
            // The block's 32 weights, as raw bits: 8 per load.
            let u0 = vld1q_u16(pw);
            let u1 = vld1q_u16(pw.add(8));
            let u2 = vld1q_u16(pw.add(16));
            let u3 = vld1q_u16(pw.add(24));

            // 8 steps of 4 values = the whole block. Each step is
            // `acc += activations * weights` on 4 pairs at once.
            acc0 = vfmaq_f32(acc0, vld1q_f32(pa), widen(vget_low_u16(u0)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(pa.add(4)), widen(vget_high_u16(u0)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(pa.add(8)), widen(vget_low_u16(u1)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(pa.add(12)), widen(vget_high_u16(u1)));
            acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(16)), widen(vget_low_u16(u2)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(pa.add(20)), widen(vget_high_u16(u2)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(pa.add(24)), widen(vget_low_u16(u3)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(pa.add(28)), widen(vget_high_u16(u3)));
        }

        // Collapse the four totals, then sum the 4 lanes into one number.
        vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)))
    }
}

impl WeightBlock for BlockBF16 {
    const ELEMS: usize = 32;
    const DTYPE: GGMLType = GGMLType::BF16;

    fn dequantize_into(&self, out: &mut [f32]) {
        for (o, w) in out.iter_mut().zip(self.v.iter()) {
            *o = w.to_f32();
        }
    }

    fn dot_f32(acts: &[f32], w: &[Self]) -> f32 {
        #[cfg(all(target_arch = "aarch64", feature = "use-optimized-ops"))]
        {
            return dot_f32_bf16_neon(acts, w);
        }
        #[allow(unreachable_code)]
        dot_f32_bf16_fallback(acts, w)
    }
}

pub fn dot_f32_f32(a: &[f32], b: &[f32]) -> Result<f32, KernelError> {
    if a.len() != b.len() {
        return Err(KernelError::ShapeMismatch);
    }
    let mut acc: f32 = 0.0;
    for (val_a, val_b) in a.iter().zip(b.iter()) {
        acc += val_a * val_b
    }
    Ok(acc)
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use half::f16;

    /// NEON must match the scalar version, bar the order of the additions.
    #[test]
    fn neon_matches_fallback() {
        // A few blocks with varied scales, weights and activations.
        let blocks: Vec<BlockQ8_0> = (0..4)
            .map(|blk| {
                let mut qs = [0i8; 32];
                for (i, q) in qs.iter_mut().enumerate() {
                    // Spread across the full i8 range, sign-alternating.
                    *q = (((i as i32 * 7 + blk * 13) % 251) - 125) as i8;
                }
                BlockQ8_0 {
                    scale: f16::from_f32(0.001 * (blk as f32 + 1.0)),
                    qs,
                }
            })
            .collect();

        let a: Vec<f32> = (0..blocks.len() * 32)
            .map(|i| (i as f32 * 0.013).sin() * 3.0 - 1.0)
            .collect();

        let neon = dot_f32_q8_0_neon(&a, &blocks);
        let scalar = dot_f32_q8_0_fallback(&a, &blocks);
        assert!(
            (neon - scalar).abs() <= 1e-4 * scalar.abs().max(1.0),
            "neon {neon} vs scalar {scalar}"
        );
    }

    /// The integer path rounds activations to i8, so it only has to stay within
    /// that rounding error of the full-f32 dot.
    #[test]
    fn int_dot_close_to_f32() {
        let blocks: Vec<BlockQ8_0> = (0..8)
            .map(|blk| {
                let mut qs = [0i8; 32];
                for (i, q) in qs.iter_mut().enumerate() {
                    *q = (((i as i32 * 5 + blk * 11) % 251) - 125) as i8;
                }
                BlockQ8_0 {
                    scale: f16::from_f32(0.002 * (blk as f32 + 1.0)),
                    qs,
                }
            })
            .collect();

        let a: Vec<f32> = (0..blocks.len() * 32)
            .map(|i| (i as f32 * 0.017).cos() * 2.5)
            .collect();

        // Reference: f32 activations against dequantized weights.
        let reference = dot_f32_q8_0_fallback(&a, &blocks);

        // Integer path: quantize the row first.
        let mut a_q = vec![
            BlockQ8_0 {
                scale: f16::from_f32(0.0),
                qs: [0; 32]
            };
            blocks.len()
        ];
        quantize_row_q8_0(&a, &mut a_q);
        let int = dot_q8_0_q8_0_neon(&a_q, &blocks);

        assert!(
            (int - reference).abs() <= 1e-2 * reference.abs().max(1.0),
            "int {int} vs reference {reference}"
        );
    }

    /// The two quantizers must agree bit for bit. Covers positive and negative
    /// maxima, ties at .5, an all-zero block, and values that clamp.
    #[test]
    fn neon_quantize_matches_scalar() {
        let mut src: Vec<f32> = (0..32 * 6)
            .map(|i| ((i as f32 * 0.37).sin() * 4.0) - 0.5)
            .collect();
        src[32..64].fill(0.0); // all-zero block
        src[64] = -100.0; // negative absmax
        src[96] = 0.5; // rounding tie
        src[97] = -0.5;
        src[128] = f32::MAX / 2.0; // extreme scale

        let n = src.len() / 32;
        let zero = BlockQ8_0 {
            scale: f16::from_f32(0.0),
            qs: [0; 32],
        };
        let mut neon = vec![zero; n];
        let mut scalar = vec![zero; n];
        quantize_row_q8_0_neon(&src, &mut neon);
        quantize_row_q8_0_scalar(&src, &mut scalar);

        for (b, (bn, bs)) in neon.iter().zip(scalar.iter()).enumerate() {
            assert_eq!(bn.scale, bs.scale, "block {b} scale");
            assert_eq!(bn.qs, bs.qs, "block {b} quants");
        }
    }

    /// Quantized activation rows for the integer tests, spanning the i8 range.
    fn test_q8_acts(n: usize) -> Vec<BlockQ8_0> {
        (0..n)
            .map(|blk| {
                let mut qs = [0i8; 32];
                for (i, q) in qs.iter_mut().enumerate() {
                    *q = (((i as i32 * 11 + blk as i32 * 17) % 251) - 125) as i8;
                }
                BlockQ8_0 {
                    scale: f16::from_f32(0.002 * (blk as f32 + 1.0)),
                    qs,
                }
            })
            .collect()
    }

    /// These two must agree exactly, since both do the products in i32. That is
    /// what makes the SIMD ablation a measurement of speed alone: the two builds
    /// produce the same numbers.
    #[test]
    fn q8_int_dot_scalar_matches_neon() {
        let acts = test_q8_acts(4);
        let weights = test_q8_acts(4); // same layout serves as Q8_0 weights
        let neon = dot_q8_0_q8_0_neon(&acts, &weights);
        let scalar = dot_q8_0_q8_0_scalar(&acts, &weights);
        let tol = neon.abs() * 1e-5 + 1e-6;
        assert!(
            (neon - scalar).abs() <= tol,
            "neon {neon} vs scalar {scalar}"
        );
    }

    /// The same, for Q4_0, including the nibble unpacking.
    #[test]
    fn q4_int_dot_scalar_matches_neon() {
        let acts = test_q8_acts(4);
        let weights = test_q4_blocks(4);
        let neon = dot_q8_0_q4_0_neon(&acts, &weights);
        let scalar = dot_q8_0_q4_0_scalar(&acts, &weights);
        let tol = neon.abs() * 1e-5 + 1e-6;
        assert!(
            (neon - scalar).abs() <= tol,
            "neon {neon} vs scalar {scalar}"
        );
    }

    fn test_q4_blocks(n: usize) -> Vec<BlockQ4_0> {
        (0..n)
            .map(|blk| {
                let mut qs = [0u8; 16];
                for (i, q) in qs.iter_mut().enumerate() {
                    // Both nibbles spread across the full [0, 15] range.
                    let lo = ((i + blk * 3) % 16) as u8;
                    let hi = ((i * 7 + blk * 5) % 16) as u8;
                    *q = (hi << 4) | lo;
                }
                BlockQ4_0 {
                    scale: f16::from_f32(0.003 * (blk as f32 + 1.0)),
                    qs,
                }
            })
            .collect()
    }

    /// Dotting against a one-hot activation picks out a single weight, so this
    /// checks `dequantize_into` against what the dot product actually does.
    #[test]
    fn q4_dequant_matches_dot() {
        let blocks = test_q4_blocks(2);
        let n = blocks.len() * 32;

        let mut dequant = vec![0.0f32; n];
        for (block, chunk) in blocks.iter().zip(dequant.chunks_exact_mut(32)) {
            block.dequantize_into(chunk);
        }

        for pos in [0usize, 1, 15, 16, 17, 31, 32, 63] {
            let mut one_hot = vec![0.0f32; n];
            one_hot[pos] = 1.0;
            let extracted = dot_f32_q4_0_fallback(&one_hot, &blocks);
            assert!(
                (extracted - dequant[pos]).abs() < 1e-6,
                "pos {pos}: dot {extracted} vs dequant {}",
                dequant[pos]
            );
        }
    }

    /// Same rounding-error bound as the Q8xQ8 test, for the Q4_0 path.
    #[test]
    fn q4_int_dot_close_to_f32() {
        let blocks = test_q4_blocks(8);
        let a: Vec<f32> = (0..blocks.len() * 32)
            .map(|i| (i as f32 * 0.019).sin() * 2.0 + 0.3)
            .collect();

        // Reference: f32 activations against dequantized weights.
        let reference = dot_f32_q4_0_fallback(&a, &blocks);

        // Integer path: quantize the row first.
        let mut a_q = vec![
            BlockQ8_0 {
                scale: f16::from_f32(0.0),
                qs: [0; 32]
            };
            blocks.len()
        ];
        quantize_row_q8_0(&a, &mut a_q);
        let int = dot_q8_0_q4_0_neon(&a_q, &blocks);

        assert!(
            (int - reference).abs() <= 1e-2 * reference.abs().max(1.0),
            "int {int} vs reference {reference}"
        );
    }

    /// With both scales at 1.0 and activations already whole numbers, the
    /// result must be exactly the integer sum.
    #[test]
    fn q4_int_dot_exact_on_integers() {
        let mut qs = [0u8; 16];
        for (i, q) in qs.iter_mut().enumerate() {
            *q = (((i % 16) as u8) << 4) | ((15 - i % 16) as u8);
        }
        let w = [BlockQ4_0 {
            scale: f16::from_f32(1.0),
            qs,
        }];

        let mut a_qs = [0i8; 32];
        for (i, a) in a_qs.iter_mut().enumerate() {
            *a = (i as i8) - 16;
        }
        let a = [BlockQ8_0 {
            scale: f16::from_f32(1.0),
            qs: a_qs,
        }];

        let mut expected = 0i32;
        for j in 0..16 {
            let lo = (qs[j] & 0x0F) as i32 - 8;
            let hi = (qs[j] >> 4) as i32 - 8;
            expected += a_qs[j] as i32 * lo + a_qs[j + 16] as i32 * hi;
        }

        let got = dot_q8_0_q4_0_neon(&a, &w);
        assert_eq!(got, expected as f32);
    }

    fn test_bf16_blocks(n: usize) -> Vec<BlockBF16> {
        (0..n)
            .map(|blk| {
                let mut v = [half::bf16::from_f32(0.0); 32];
                for (i, w) in v.iter_mut().enumerate() {
                    *w = half::bf16::from_f32(((i as f32 * 0.11 + blk as f32).sin()) * 2.5);
                }
                BlockBF16 { v }
            })
            .collect()
    }

    /// The shift-and-reinterpret widen must agree with going through
    /// `bf16::to_f32`, bar the order of the additions.
    #[test]
    fn bf16_neon_matches_fallback() {
        let blocks = test_bf16_blocks(5);
        let a: Vec<f32> = (0..blocks.len() * 32)
            .map(|i| (i as f32 * 0.023).cos() * 1.5 - 0.2)
            .collect();

        let neon = dot_f32_bf16_neon(&a, &blocks);
        let scalar = dot_f32_bf16_fallback(&a, &blocks);
        assert!(
            (neon - scalar).abs() <= 1e-4 * scalar.abs().max(1.0),
            "neon {neon} vs scalar {scalar}"
        );
    }

    /// One-hot activation extracts exactly one dequantized weight, tying
    /// dequantize_into and the dot path together (mirrors the Q4 test).
    #[test]
    fn bf16_dequant_matches_dot() {
        let blocks = test_bf16_blocks(2);
        let n = blocks.len() * 32;

        let mut dequant = vec![0.0f32; n];
        for (block, chunk) in blocks.iter().zip(dequant.chunks_exact_mut(32)) {
            block.dequantize_into(chunk);
        }

        for pos in [0usize, 1, 15, 16, 31, 32, 63] {
            let mut one_hot = vec![0.0f32; n];
            one_hot[pos] = 1.0;
            let extracted = dot_f32_bf16_fallback(&one_hot, &blocks);
            assert!(
                (extracted - dequant[pos]).abs() < 1e-6,
                "pos {pos}: dot {extracted} vs dequant {}",
                dequant[pos]
            );
        }
    }
}

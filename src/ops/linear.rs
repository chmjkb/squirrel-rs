use std::sync::OnceLock;

use crate::{
    file_parser::ggml_type::GGMLType,
    kernels::{
        kernel_error::KernelError,
        matmul::{matmul_f32_fused_q, matmul_f32_q},
        tensor::{
            BlockBF16, BlockQ4_0, BlockQ8_0, NonOwningTensor, OwningTensor, TensorView, WeightBlock,
        },
    },
};

/// `SQUIRREL_F32_ACTS=1` keeps the activations in f32 for every format. BF16 ignores the flag: it is
/// the reference, so its activations are never quantized either way.
fn force_f32_acts() -> bool {
    static FORCE: OnceLock<bool> = OnceLock::new();
    *FORCE.get_or_init(|| std::env::var_os("SQUIRREL_F32_ACTS").is_some_and(|v| v != "0"))
}

/// One projection: `output = input @ weight`.
/// Shapes follow the GGUF convention:
/// - `input`:  `[m, in_dim]` f32
/// - `weight`: `[in_dim, out_dim]`, stored as `out_dim` rows of `in_dim` weights
/// - `output`: `[m, out_dim]` f32, allocated by the caller
pub fn linear(
    input: &impl TensorView,
    weight: &impl TensorView,
    output: &mut impl TensorView,
) -> Result<(), KernelError> {
    match weight.dtype() {
        GGMLType::Q8_0 => linear_blocks::<BlockQ8_0>(input, weight, output, !force_f32_acts()),
        GGMLType::Q4_0 => linear_blocks::<BlockQ4_0>(input, weight, output, !force_f32_acts()),
        GGMLType::BF16 => linear_blocks::<BlockBF16>(input, weight, output, false),
        _ => Err(KernelError::InvalidType),
    }
}

fn linear_blocks<B: WeightBlock>(
    input: &impl TensorView,
    weight: &impl TensorView,
    output: &mut impl TensorView,
    int_acts: bool,
) -> Result<(), KernelError> {
    let in_shape = input.shape();
    let w_shape = weight.shape();
    let out_shape = output.shape();

    if in_shape.len() != 2 || w_shape.len() != 2 || out_shape.len() != 2 {
        return Err(KernelError::ShapeMismatch);
    }

    let m = in_shape[0];
    let in_dim = in_shape[1];
    let out_dim = w_shape[1];

    if w_shape[0] != in_dim || out_shape[0] != m || out_shape[1] != out_dim {
        return Err(KernelError::ShapeMismatch);
    }

    let in_slice = input.as_f32_slice()?;
    let w_blocks = weight.as_blocks::<B>()?;
    let out_slice = output.as_f32_slice_mut()?;

    if int_acts {
        matmul_f32_q(in_slice, w_blocks, out_slice, m, in_dim, in_dim, out_dim)
    } else {
        matmul_f32_fused_q(in_slice, vec![(w_blocks, out_slice, out_dim)], m, in_dim)
    }
}

/// Several projections of the same input: `outputs[i] = input @ weights[i]`.
pub fn linear_shared(
    input: &impl TensorView,
    act_scratch: &mut Vec<BlockQ8_0>,
    weights: &[&NonOwningTensor],
    outputs: &mut [&mut OwningTensor],
) -> Result<(), KernelError> {
    if weights.len() != outputs.len() || weights.is_empty() {
        return Err(KernelError::ShapeMismatch);
    }

    #[cfg(target_arch = "aarch64")]
    {
        use crate::kernels::{dot_prod::quantize_row_q8_0, matmul::matmul_prequant_fused_q};

        let in_shape = input.shape();
        if in_shape.len() != 2 || !in_shape[1].is_multiple_of(32) {
            return Err(KernelError::ShapeMismatch);
        }
        let (m, in_dim) = (in_shape[0], in_shape[1]);

        for (w, out) in weights.iter().zip(outputs.iter()) {
            let w_shape = w.shape();
            if w_shape.len() != 2 || w_shape[0] != in_dim || out.shape() != [m, w_shape[1]] {
                return Err(KernelError::ShapeMismatch);
            }
        }

        let dtype = weights[0].dtype();
        if weights.iter().any(|w| w.dtype() != dtype) {
            return Err(KernelError::InvalidType);
        }

        fn parts<'a, B: WeightBlock>(
            weights: &[&'a NonOwningTensor],
            outputs: &'a mut [&mut OwningTensor],
        ) -> Result<Vec<(&'a [B], &'a mut [f32], usize)>, KernelError> {
            let mut parts = Vec::with_capacity(weights.len());
            for (w, out) in weights.iter().zip(outputs.iter_mut()) {
                let out_dim = w.shape()[1];
                parts.push((w.as_blocks::<B>()?, out.as_f32_slice_mut()?, out_dim));
            }
            Ok(parts)
        }

        let int_acts = matches!(dtype, GGMLType::Q8_0 | GGMLType::Q4_0) && !force_f32_acts();
        if int_acts {
            let in_slice = input.as_f32_slice()?;
            act_scratch.resize(
                in_slice.len() / 32,
                BlockQ8_0 {
                    scale: half::f16::from_f32(0.0),
                    qs: [0; 32],
                },
            );
            quantize_row_q8_0(in_slice, act_scratch);
            return match dtype {
                GGMLType::Q8_0 => matmul_prequant_fused_q(
                    act_scratch,
                    parts::<BlockQ8_0>(weights, outputs)?,
                    m,
                    in_dim,
                ),
                GGMLType::Q4_0 => matmul_prequant_fused_q(
                    act_scratch,
                    parts::<BlockQ4_0>(weights, outputs)?,
                    m,
                    in_dim,
                ),
                _ => unreachable!("int_acts implies Q8_0/Q4_0"),
            };
        }

        let in_slice = input.as_f32_slice()?;
        match dtype {
            GGMLType::Q8_0 => {
                matmul_f32_fused_q(in_slice, parts::<BlockQ8_0>(weights, outputs)?, m, in_dim)
            }
            GGMLType::Q4_0 => {
                matmul_f32_fused_q(in_slice, parts::<BlockQ4_0>(weights, outputs)?, m, in_dim)
            }
            GGMLType::BF16 => {
                matmul_f32_fused_q(in_slice, parts::<BlockBF16>(weights, outputs)?, m, in_dim)
            }
            _ => Err(KernelError::InvalidType),
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = act_scratch;
        for (w, out) in weights.iter().zip(outputs.iter_mut()) {
            linear(input, *w, *out)?;
        }
        Ok(())
    }
}

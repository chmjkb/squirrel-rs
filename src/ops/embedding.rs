use crate::{
    file_parser::ggml_type::GGMLType,
    kernels::{
        kernel_error::KernelError,
        tensor::{BlockBF16, BlockQ4_0, BlockQ8_0, TensorView, WeightBlock},
    },
};

/// Embeds tokens by dequantizing rows of a quantized embedding table.
pub fn embed(
    tokens: &[u32],
    weight_tensor: &impl TensorView,
    output: &mut impl TensorView,
) -> Result<(), KernelError> {
    match weight_tensor.dtype() {
        GGMLType::Q8_0 => embed_blocks::<BlockQ8_0>(tokens, weight_tensor, output),
        GGMLType::Q4_0 => embed_blocks::<BlockQ4_0>(tokens, weight_tensor, output),
        GGMLType::BF16 => embed_blocks::<BlockBF16>(tokens, weight_tensor, output),
        _ => Err(KernelError::InvalidType),
    }
}

fn embed_blocks<B: WeightBlock>(
    tokens: &[u32],
    weight_tensor: &impl TensorView,
    output: &mut impl TensorView,
) -> Result<(), KernelError> {
    let blocks = weight_tensor.as_blocks::<B>()?;
    let embed_dim = weight_tensor.shape()[0];
    if !embed_dim.is_multiple_of(B::ELEMS) {
        return Err(KernelError::ShapeMismatch);
    }
    let blocks_per_token = embed_dim / B::ELEMS;

    let output_slice = output.as_f32_slice_mut()?;
    let expected_output_size = tokens.len() * embed_dim;
    if output_slice.len() < expected_output_size {
        return Err(KernelError::InvalidType);
    }

    // Dequantize each token's row of blocks straight into the output buffer.
    for (t, &token) in tokens.iter().enumerate() {
        let start = token as usize * blocks_per_token;
        let row = &blocks[start..start + blocks_per_token];
        let out_row = &mut output_slice[t * embed_dim..(t + 1) * embed_dim];
        for (block, out_chunk) in row.iter().zip(out_row.chunks_exact_mut(B::ELEMS)) {
            block.dequantize_into(out_chunk);
        }
    }

    Ok(())
}

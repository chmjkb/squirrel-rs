use crate::kernels::sigmoid::sigmoid_f32;

/// Computes the SILU activation function, also known as swish.
pub fn silu_f32(input: &mut [f32]) {
    for item in input.iter_mut() {
        *item = *item * sigmoid_f32(*item);
    }
}

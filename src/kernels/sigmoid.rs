/// Computes the sigmoid function in-place: σ(x) = 1 / (1 + e^-x)
pub fn sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

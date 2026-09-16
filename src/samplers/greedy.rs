use crate::samplers::common::Sampler;
use std::cmp::Ordering;

pub struct GreedySampler;

impl<T: PartialOrd + Copy> Sampler<T> for GreedySampler {
    fn sample(logits: &[T]) -> usize {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .expect("sample called on empty logits")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_max_index() {
        let logits = [0.1f32, -2.3, 5.7, 4.0];
        assert_eq!(<GreedySampler as Sampler<f32>>::sample(&logits), 2);
    }

    #[test]
    fn ignores_nan() {
        let logits = [1.0f32, f32::NAN, 3.0, 2.0];
        assert_eq!(<GreedySampler as Sampler<f32>>::sample(&logits), 2);
    }
}

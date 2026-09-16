pub trait Sampler<T: PartialOrd + Copy> {
    /// Returns the index of the chosen token in the logits slice.
    fn sample(logits: &[T]) -> usize;
}

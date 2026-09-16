#[derive(Debug, Clone)]
pub enum KernelError {
    ShapeMismatch,
    InvalidType,
    EmptyInput,
    TensorAlignmentError,
}

use crate::kernels::kernel_error::KernelError;
use crate::models::llama::{config::ConfigError, weights::WeightsError};

#[derive(Debug)]
pub enum LLMError {
    InvalidTokenizer,
    InvalidConfig(ConfigError),
    InvalidWeights(WeightsError),
    Kernel(KernelError),
}

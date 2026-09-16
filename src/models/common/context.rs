//! Per-sequence inference context
//! Bundles the KV cache (history across decode steps) and the
//! scratch workspace (reused intermediate buffers)

use crate::kernels::{kernel_error::KernelError, tensor::TensorView};
use crate::models::common::{kv_cache::KvCache, workspace::Workspace};

#[derive(Debug)]
pub struct ModelContext {
    pub cache: KvCache,
    pub ws: Workspace,
}

impl ModelContext {
    pub fn new(cache: KvCache, ws: Workspace) -> Self {
        Self { cache, ws }
    }

    /// Logits computed by the most recent `forward`, shape `[1, vocab_size]`.
    pub fn logits(&self) -> Result<&[f32], KernelError> {
        self.ws.logits.as_f32_slice()
    }

    /// Forget the cached history, keeping every buffer's capacity.
    pub fn reset_cache(&mut self) {
        self.cache.clear();
    }
}

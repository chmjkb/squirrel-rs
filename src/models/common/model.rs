use crate::models::common::{context::ModelContext, error::LLMError};
use tokenizers::Tokenizer;

/// Anything that can take a sequence of token ids and produce next-token
/// logits. Also exposes the model-specific tokenizer and stop token.
pub trait Model {
    /// A fresh inference context (KV cache + scratch workspace).
    fn new_context(&self) -> ModelContext;

    /// Forward pass over `token_ids`, reading and extending `ctx`.
    ///
    /// The first call (prefill) passes the whole prompt; subsequent calls
    /// (decode) pass only the single newly sampled token, attending against
    /// the cached history. The last token's logits are written into the
    /// context, read them via [`ModelContext::logits`].
    fn forward(&self, token_ids: &[u32], ctx: &mut ModelContext) -> Result<(), LLMError>;

    /// The tokenizer this model was loaded with.
    fn tokenizer(&self) -> &Tokenizer;

    /// End-of-sequence token id.
    fn eos_token_id(&self) -> u32;

    /// Beginning-of-sequence token id.
    fn bos_token_id(&self) -> u32;
}

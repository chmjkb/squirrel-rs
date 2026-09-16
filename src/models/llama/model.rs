use crate::models::common::{
    context::ModelContext, error::LLMError, kv_cache::KvCache, model::Model, workspace::Workspace,
};
use crate::{
    file_parser::gguf_file::GGUFFile,
    kernels::tensor::TensorView,
    models::llama::{config::LlamaConfig, layers, weights::LlamaWeights},
    ops::{embedding::embed, linear::linear, rms_norm::rms_norm_inplace},
};
use tokenizers::Tokenizer;

#[derive(Debug)]
pub struct Llama3_2<'a> {
    tokenizer: Tokenizer,
    config: LlamaConfig,
    weights: LlamaWeights<'a>,
}

impl<'a> Llama3_2<'a> {
    pub fn from_gguf(gguf: &'a GGUFFile, tokenizer_path: &str) -> Result<Self, LLMError> {
        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|_| LLMError::InvalidTokenizer)?;
        let config = LlamaConfig::from_gguf(gguf).map_err(LLMError::InvalidConfig)?;
        let weights =
            LlamaWeights::from_gguf(gguf, config.n_blocks).map_err(LLMError::InvalidWeights)?;

        Ok(Self {
            tokenizer,
            config,
            weights,
        })
    }
}

impl<'a> Model for Llama3_2<'a> {
    fn new_context(&self) -> ModelContext {
        let cfg = &self.config;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let cache = KvCache::new(cfg.n_blocks, kv_dim);
        let ws = Workspace::new(cfg.embed_dim, q_dim, kv_dim, cfg.ffn_dim, cfg.vocab_size);
        ModelContext::new(cache, ws)
    }

    fn forward(&self, token_ids: &[u32], ctx: &mut ModelContext) -> Result<(), LLMError> {
        let cfg = &self.config;
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return Err(LLMError::Kernel(
                crate::kernels::kernel_error::KernelError::EmptyInput,
            ));
        }

        ctx.ws.prepare(seq_len);

        // 1. Embed → ws.hidden [seq_len, embed_dim].
        embed(
            token_ids,
            &self.weights.token_embeddings,
            &mut ctx.ws.hidden,
        )
        .map_err(LLMError::Kernel)?;

        let position = ctx.cache.seq_len();
        for (layer, blk) in self.weights.blocks.iter().enumerate() {
            layers::block(&mut ctx.ws, blk, cfg, position, &mut ctx.cache, layer)
                .map_err(LLMError::Kernel)?;
        }
        // Every layer has appended this pass's K/V; advance the position count.
        ctx.cache.advance(seq_len);

        rms_norm_inplace(&mut ctx.ws.hidden, &self.weights.output_norm, cfg.rms_eps)
            .map_err(LLMError::Kernel)?;

        // Extract the last token's hidden row before the LM head. The earlier
        // seq_len - 1 rows would be discarded by the sampler anyway, so
        // computing them in the [embed_dim → vocab_size] matmul is wasteful.
        {
            let src = ctx.ws.hidden.as_f32_slice().map_err(LLMError::Kernel)?;
            let dst = ctx
                .ws
                .last_hidden
                .as_f32_slice_mut()
                .map_err(LLMError::Kernel)?;
            let row_start = (seq_len - 1) * cfg.embed_dim;
            dst.copy_from_slice(&src[row_start..row_start + cfg.embed_dim]);
        }

        // LM head: [1, embed_dim] @ [embed_dim, vocab] = [1, vocab] → ws.logits.
        linear(
            &ctx.ws.last_hidden,
            &self.weights.output,
            &mut ctx.ws.logits,
        )
        .map_err(LLMError::Kernel)?;
        Ok(())
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn eos_token_id(&self) -> u32 {
        self.config.eos_token_id
    }

    fn bos_token_id(&self) -> u32 {
        self.config.bos_token_id
    }
}

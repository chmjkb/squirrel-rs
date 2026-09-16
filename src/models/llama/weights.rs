use crate::{
    file_parser::{
        gguf_file::GGUFFile,
        gguf_tensor_info::{self, GGUFTensorInfo},
    },
    kernels::tensor::NonOwningTensor,
};

#[derive(Debug)]
pub enum WeightsError {
    MissingTensor(String),
    TensorBuild {
        name: String,
        source: gguf_tensor_info::Error,
    },
}

#[derive(Debug)]
pub struct LlamaAttnWeights<'a> {
    pub attn_norm: NonOwningTensor<'a>,
    pub attn_q: NonOwningTensor<'a>,
    pub attn_k: NonOwningTensor<'a>,
    pub attn_v: NonOwningTensor<'a>,
    pub attn_output: NonOwningTensor<'a>,
}

#[derive(Debug)]
pub struct LlamaFfnWeights<'a> {
    pub ffn_norm: NonOwningTensor<'a>,
    pub ffn_gate: NonOwningTensor<'a>,
    pub ffn_up: NonOwningTensor<'a>,
    pub ffn_down: NonOwningTensor<'a>,
}

#[derive(Debug)]
pub struct LlamaBlock<'a> {
    pub attn: LlamaAttnWeights<'a>,
    pub ffn: LlamaFfnWeights<'a>,
}

#[derive(Debug)]
pub struct LlamaWeights<'a> {
    pub token_embeddings: NonOwningTensor<'a>,
    pub output_norm: NonOwningTensor<'a>,
    pub output: NonOwningTensor<'a>,
    pub blocks: Vec<LlamaBlock<'a>>,
}

impl<'a> LlamaWeights<'a> {
    pub fn from_gguf(gguf: &'a GGUFFile, n_blocks: usize) -> Result<Self, WeightsError> {
        let get = |name: &str| -> Result<NonOwningTensor<'a>, WeightsError> {
            let info: &GGUFTensorInfo = gguf
                .tensor_info
                .tensors_map
                .get(name)
                .ok_or_else(|| WeightsError::MissingTensor(name.to_string()))?;
            NonOwningTensor::from_info(&gguf.mmap, info, gguf.tensor_data_offset).map_err(
                |source| WeightsError::TensorBuild {
                    name: name.to_string(),
                    source,
                },
            )
        };

        let blocks = (0..n_blocks)
            .map(|i| {
                Ok(LlamaBlock {
                    attn: LlamaAttnWeights {
                        attn_norm: get(&format!("blk.{i}.attn_norm.weight"))?,
                        attn_q: get(&format!("blk.{i}.attn_q.weight"))?,
                        attn_k: get(&format!("blk.{i}.attn_k.weight"))?,
                        attn_v: get(&format!("blk.{i}.attn_v.weight"))?,
                        attn_output: get(&format!("blk.{i}.attn_output.weight"))?,
                    },
                    ffn: LlamaFfnWeights {
                        ffn_norm: get(&format!("blk.{i}.ffn_norm.weight"))?,
                        ffn_gate: get(&format!("blk.{i}.ffn_gate.weight"))?,
                        ffn_up: get(&format!("blk.{i}.ffn_up.weight"))?,
                        ffn_down: get(&format!("blk.{i}.ffn_down.weight"))?,
                    },
                })
            })
            .collect::<Result<Vec<_>, WeightsError>>()?;

        let token_embeddings = get("token_embd.weight")?;
        let output_norm = get("output_norm.weight")?;
        // Some llama variants tie the LM head to the input embeddings — if `output.weight`
        // is missing, reuse `token_embd.weight`.
        let output = get("output.weight").or_else(|_| get("token_embd.weight"))?;

        Ok(Self {
            token_embeddings,
            output_norm,
            output,
            blocks,
        })
    }
}

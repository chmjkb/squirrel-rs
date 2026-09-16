//! one growable `Vec<f32>` per layer for K and another for V, each laid
//! out `[position, n_kv_heads * head_dim]` row-major,  exactly the shape the
//! projection produces, so appending is a flat `extend_from_slice`.

use crate::kernels::kernel_error::KernelError;

#[derive(Debug)]
pub struct KvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    /// Row width: `n_kv_heads * head_dim`.
    kv_dim: usize,
    /// Positions cached before the current forward pass. This is the absolute
    /// RoPE position of the next token to be processed.
    seq_len: usize,
}

impl KvCache {
    /// Empty cache for `n_layers` layers, each row `kv_dim` floats wide.
    pub fn new(n_layers: usize, kv_dim: usize) -> Self {
        Self {
            k: vec![Vec::new(); n_layers],
            v: vec![Vec::new(); n_layers],
            kv_dim,
            seq_len: 0,
        }
    }

    /// Number of positions cached before this pass = absolute position of the
    /// first new token. RoPE and the causal mask key off this.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Appends one row per new position, `kv_dim` floats wide. RoPE is applied
    /// by the caller. Does not move `seq_len`; call `advance` after the last
    /// layer.
    pub fn append(
        &mut self,
        layer: usize,
        k_new: &[f32],
        v_new: &[f32],
    ) -> Result<(), KernelError> {
        if self.kv_dim == 0
            || !k_new.len().is_multiple_of(self.kv_dim)
            || v_new.len() != k_new.len()
        {
            return Err(KernelError::ShapeMismatch);
        }
        self.k[layer].extend_from_slice(k_new);
        self.v[layer].extend_from_slice(v_new);
        Ok(())
    }

    /// All cached K for `layer`, shaped `[cached_positions, kv_dim]`, flattened.
    pub fn k(&self, layer: usize) -> &[f32] {
        &self.k[layer]
    }

    /// All cached V for `layer`, shaped `[cached_positions, kv_dim]`, flattened.
    pub fn v(&self, layer: usize) -> &[f32] {
        &self.v[layer]
    }

    /// Advance the position counter by `n` once a full pass over every layer is
    /// done.
    pub fn advance(&mut self, n: usize) {
        self.seq_len += n;
    }

    /// Drop every cached position, keeping the allocated capacity.
    pub fn clear(&mut self) {
        for (k, v) in self.k.iter_mut().zip(self.v.iter_mut()) {
            k.clear();
            v.clear();
        }
        self.seq_len = 0;
    }
}

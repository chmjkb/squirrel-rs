use crate::{
    file_parser::{
        ggml_type::GGMLType,
        gguf_tensor_info::{Error, GGUFTensorInfo},
    },
    kernels::kernel_error::KernelError,
};
use half::f16;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct BlockQ8_0 {
    pub scale: f16,
    pub qs: [i8; 32],
}

/// GGUF Q4_0: 32 weights in 18 bytes — an f16 scale plus 16 bytes of packed
/// nibbles. Byte `j` holds weight `j` in its low nibble and weight `j + 16`
/// in its high nibble; each nibble `q ∈ [0, 15]` encodes `(q - 8) * scale`.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct BlockQ4_0 {
    pub scale: f16,
    pub qs: [u8; 16],
}

/// 32 raw bf16 weights (64 bytes) — not quantized at all. BF16 is the top
/// half of an f32's bit pattern, so "dequantization" is a 16-bit left shift.
/// This is the reference precision (Llama 3.2's training dtype) that the
/// quantized schemes are measured against.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct BlockBF16 {
    pub v: [half::bf16; 32],
}

/// A quantized weight-block layout the compute layer knows how to consume.
///
/// This is the extension point for new quant schemes: define the `#[repr(C)]`
/// block struct matching the GGUF layout, implement this trait (dequant + the
/// two dot kernels, which live in `kernels::dot_prod`), and add the dtype arm
/// to the dispatchers in `ops::linear` / `ops::embedding`. Everything between
/// — matmul structure, weight loading, size math — is generic over this trait.
pub trait WeightBlock: Copy + Send + Sync + 'static {
    /// Elements per block.
    const ELEMS: usize;
    /// The GGML dtype whose storage layout this block matches.
    const DTYPE: GGMLType;

    /// Dequantize the whole block into `out` (`out.len() == Self::ELEMS`).
    fn dequantize_into(&self, out: &mut [f32]);

    /// Dot of an f32 activation row against a quantized weight row. Reference
    /// semantics, and the live path for schemes that keep f32 activations
    /// (BF16, or Q8/Q4 under the `SQUIRREL_F32_ACTS` ablation toggle).
    fn dot_f32(acts: &[f32], w: &[Self]) -> f32;

    /// Integer dot against a Q8_0-quantized activation row (aarch64 fast
    /// path: the activation row is quantized once per matmul row, then
    /// streamed against the weights without widening them to f32).
    ///
    /// Schemes that must keep f32 activations (BF16 — quantizing activations
    /// would contaminate the reference) don't implement this; the dispatch in
    /// `ops::linear` never routes them here, so the default is unreachable.
    #[cfg(target_arch = "aarch64")]
    fn dot_q8(_acts: &[BlockQ8_0], _w: &[Self]) -> f32 {
        unreachable!("{:?} weights use the f32-activation path", Self::DTYPE)
    }
}

pub trait TensorView {
    fn data(&self) -> &[u8];
    fn data_mut(&mut self) -> Option<&mut [u8]>;
    fn dtype(&self) -> GGMLType;
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];

    fn dim(&self) -> usize {
        self.shape().len()
    }

    fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    fn size_bytes(&self) -> usize {
        self.data().len()
    }

    fn as_f32_slice(&self) -> Result<&[f32], KernelError> {
        if self.dtype() != GGMLType::F32 {
            return Err(KernelError::InvalidType);
        }

        let (prefix, floats, suffix) = unsafe { self.data().align_to::<f32>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            return Err(KernelError::TensorAlignmentError);
        }
        Ok(floats)
    }

    fn as_f32_slice_mut(&mut self) -> Result<&mut [f32], KernelError> {
        if self.dtype() != GGMLType::F32 {
            return Err(KernelError::InvalidType);
        }

        let data = self.data_mut().ok_or(KernelError::InvalidType)?;
        let (prefix, floats, suffix) = unsafe { data.align_to_mut::<f32>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            return Err(KernelError::TensorAlignmentError);
        }
        Ok(floats)
    }

    /// Zero-copy view of the raw bytes as quant blocks of type `B`, checked
    /// against the tensor's dtype.
    fn as_blocks<B: WeightBlock>(&self) -> Result<&[B], KernelError> {
        if self.dtype() != B::DTYPE {
            return Err(KernelError::InvalidType);
        }

        let (prefix, blocks, suffix) = unsafe { self.data().align_to::<B>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            return Err(KernelError::TensorAlignmentError);
        }
        Ok(blocks)
    }
}

/// Tensor backed by borrowed data (e.g., from memory-mapped file)
/// Used for model weights that are loaded from disk
#[derive(Debug, Clone)]
pub struct NonOwningTensor<'a> {
    // This should be a reference to a mmaped-file.
    pub data: &'a [u8],
    pub dtype: GGMLType,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

/// Tensor backed by owned data (heap-allocated)
/// Used for intermediate activations during inference
#[derive(Debug, Clone)]
pub struct OwningTensor {
    data: Vec<u8>,
    pub dtype: GGMLType,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl<'a> NonOwningTensor<'a> {
    pub fn from_info(
        mmap: &'a [u8],
        info: &GGUFTensorInfo,
        tensor_data_offset: u64,
    ) -> Result<Self, Error> {
        let size_in_bytes = info.tensor_bytes_size()?;
        let start = (tensor_data_offset + info.offset) as usize;
        let end = start + size_in_bytes as usize;
        let data = &mmap[start..end];
        let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
        let strides = Self::compute_default_strides(&shape);
        Ok(Self {
            data,
            dtype: info.ggml_type,
            shape,
            strides,
        })
    }

    fn compute_default_strides(shape: &[usize]) -> Vec<usize> {
        // EXAMPLE: [1024, 512] [r, c]
        // To get the next row element, we need to jump 1024 elements. To get the next column
        // element, we need to jump 1 element.
        let mut strides: Vec<usize> = vec![0; shape.len()];
        let mut stride = 1;
        for dim in (0..shape.len()).rev() {
            strides[dim] = stride;
            stride *= shape[dim];
        }
        strides
    }
}

impl<'a> TensorView for NonOwningTensor<'a> {
    fn data(&self) -> &[u8] {
        self.data
    }

    fn dim(&self) -> usize {
        self.strides.len()
    }

    fn data_mut(&mut self) -> Option<&mut [u8]> {
        // Borrowed tensors are immutable
        None
    }

    fn dtype(&self) -> GGMLType {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn strides(&self) -> &[usize] {
        &self.strides
    }
}

impl OwningTensor {
    pub fn new(dtype: GGMLType, shape: Vec<usize>) -> Self {
        let strides = Self::compute_default_strides(&shape);
        let numel: usize = shape.iter().product();
        let bytes_per_element = dtype.bytes_per_element();
        let size_bytes = numel * bytes_per_element;

        Self {
            data: vec![0u8; size_bytes],
            dtype,
            shape,
            strides,
        }
    }

    /// Create a new F32 tensor with zeros
    pub fn zeros_f32(shape: Vec<usize>) -> Self {
        Self::new(GGMLType::F32, shape)
    }

    fn compute_default_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides: Vec<usize> = vec![0; shape.len()];
        let mut stride = 1;
        for dim in (0..shape.len()).rev() {
            strides[dim] = stride;
            stride *= shape[dim];
        }
        strides
    }

    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Reshape to a new F32 shape, reusing the existing allocation. The backing
    /// `Vec` keeps its capacity, so growing reallocates only when the new size
    /// exceeds the high-water mark and shrinking never frees — letting a single
    /// buffer be reused across forward passes of different sequence lengths
    /// (large prefill once, then cheap one-token decodes) without reallocating.
    pub fn resize_f32(&mut self, shape: Vec<usize>) {
        let numel: usize = shape.iter().product();
        self.data
            .resize(numel * GGMLType::F32.bytes_per_element(), 0);
        self.strides = Self::compute_default_strides(&shape);
        self.shape = shape;
        self.dtype = GGMLType::F32;
    }
}

impl TensorView for OwningTensor {
    fn data(&self) -> &[u8] {
        &self.data
    }

    fn data_mut(&mut self) -> Option<&mut [u8]> {
        Some(&mut self.data)
    }

    fn dtype(&self) -> GGMLType {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn strides(&self) -> &[usize] {
        &self.strides
    }
}

use std::collections::HashMap;
use std::io;

use crate::file_parser::{
    binary_reader_common::{read_str, read_u32, read_u64},
    ggml_type::GGMLType,
};

#[derive(Debug)]
pub enum Error {
    UnsupportedQuantScheme(GGMLType),
}

#[derive(Debug)]
pub struct GGUFTensorInfo {
    pub name: String,
    pub n_dimensions: u32,
    pub dimensions: Vec<u64>,
    pub ggml_type: GGMLType,
    pub offset: u64,
    pub numel: u64,
}

impl GGUFTensorInfo {
    fn read_from<R: io::Read>(r: &mut R) -> io::Result<Self> {
        let name = read_str(r)?;
        let n_dimensions = read_u32(r)?;
        let mut dimensions = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            let dim = read_u64(r)?;
            dimensions.push(dim);
        }
        let raw = read_u32(r)?;
        let ggml_type = GGMLType::try_from(raw).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown GGML type id {raw} in tensor {name:?}"),
            )
        })?;
        let offset = read_u64(r)?;
        let numel = dimensions.iter().product();
        Ok(GGUFTensorInfo {
            name,
            n_dimensions,
            dimensions,
            offset,
            ggml_type,
            numel,
        })
    }

    pub fn tensor_bytes_size(&self) -> Result<u64, Error> {
        let block_size = self
            .ggml_type
            .block_size()
            .ok_or(Error::UnsupportedQuantScheme(self.ggml_type))? as u64;
        let block_bytes = self
            .ggml_type
            .block_bytes()
            .ok_or(Error::UnsupportedQuantScheme(self.ggml_type))? as u64;
        if !self.numel.is_multiple_of(block_size) {
            return Err(Error::UnsupportedQuantScheme(self.ggml_type));
        }
        Ok(self.numel / block_size * block_bytes)
    }
}

#[derive(Debug)]
pub struct GGUFTensorsInfo {
    pub tensors_map: HashMap<String, GGUFTensorInfo>,
}

impl GGUFTensorsInfo {
    pub fn read_from<R: io::Read>(r: &mut R, count: usize) -> io::Result<Self> {
        let mut tensors_map = HashMap::new();
        for _ in 0..count {
            let tensor = GGUFTensorInfo::read_from(r)?;
            tensors_map.insert(tensor.name.clone(), tensor);
        }
        Ok(Self { tensors_map })
    }
}

use crate::file_parser::binary_reader_common::{read_u32, read_u64};
use std::io;

#[derive(Debug)]
pub struct GGUFHeader {
    magic: u32,
    pub gguf_version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

impl GGUFHeader {
    pub fn read_from<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        let magic = read_u32(reader)?;
        let gguf_version = read_u32(reader)?;
        let tensor_count = read_u64(reader)?;
        let metadata_kv_count = read_u64(reader)?;
        Ok(Self {
            magic,
            gguf_version,
            tensor_count,
            metadata_kv_count,
        })
    }
}

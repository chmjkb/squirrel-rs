use std::fs::File;
use std::io::{Cursor, Read, Result, Seek};

use memmap2::Mmap;

use crate::file_parser::{
    gguf_header::GGUFHeader, gguf_metadata::GGUFMetadata, gguf_tensor_info::GGUFTensorsInfo,
};

const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug)]
pub struct GGUFFile {
    pub mmap: Mmap,
    pub header: GGUFHeader,
    pub metadata: GGUFMetadata,
    pub tensor_info: GGUFTensorsInfo,
    pub tensor_data_offset: u64,
}

impl GGUFFile {
    pub fn from_file(path: &str) -> Result<Self> {
        let file_handle = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file_handle) }?;
        let mut reader = Cursor::new(&mmap[..]);
        let (header, metadata, tensor_info, tensor_data_offset) = Self::read_from(&mut reader)?;

        Ok(Self {
            mmap,
            header,
            metadata,
            tensor_info,
            tensor_data_offset,
        })
    }

    pub(crate) fn read_from<R: Read + Seek>(
        r: &mut R,
    ) -> Result<(GGUFHeader, GGUFMetadata, GGUFTensorsInfo, u64)> {
        let header = GGUFHeader::read_from(r)?;
        let metadata = GGUFMetadata::read_from(r, header.metadata_kv_count as usize)?;
        let tensor_info = GGUFTensorsInfo::read_from(r, header.tensor_count as usize)?;

        let current_pos = r.stream_position()?;
        let tensor_data_offset =
            (current_pos + GGUF_DEFAULT_ALIGNMENT - 1) & !(GGUF_DEFAULT_ALIGNMENT - 1);

        Ok((header, metadata, tensor_info, tensor_data_offset))
    }
}

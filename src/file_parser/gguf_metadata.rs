use std::collections::HashMap;
use std::io;

use crate::file_parser::{
    binary_reader_common::{read_str, read_u32},
    gguf_metadata_value::{GGUFType, GGUFValue},
};

#[derive(Debug)]
pub struct GGUFMetadata {
    pub metadata: HashMap<String, GGUFValue>,
}

impl GGUFMetadata {
    pub fn read_from<R: io::Read>(reader: &mut R, count: usize) -> io::Result<Self> {
        let mut meta = HashMap::with_capacity(count);
        for _ in 0..count {
            let key = read_str(reader)?;
            let value_type = GGUFType::try_from(read_u32(reader)?)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid GGUF Type!"))?;
            let value = GGUFValue::read_from(reader, value_type)?;
            meta.insert(key, value);
        }
        Ok(Self { metadata: meta })
    }
}

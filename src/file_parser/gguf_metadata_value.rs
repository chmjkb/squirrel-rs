use crate::file_parser::binary_reader_common::{
    read_bool, read_f32, read_f64, read_i16, read_i32, read_i64, read_i8, read_str, read_u16,
    read_u32, read_u64, read_u8,
};
use num_enum::TryFromPrimitive;
use std::io::{Error, ErrorKind, Read, Result};

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, TryFromPrimitive)]
pub enum GGUFType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

#[derive(Debug, Clone)]
pub enum GGUFValue {
    Bool(bool),

    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),

    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),

    F32(f32),
    F64(f64),

    String(String),

    Array(Vec<GGUFValue>),
}

impl GGUFValue {
    pub fn read_from<R: Read>(r: &mut R, value_type: GGUFType) -> Result<Self> {
        match value_type {
            GGUFType::Bool => Ok(GGUFValue::Bool(read_bool(r)?)),
            GGUFType::I8 => Ok(GGUFValue::I8(read_i8(r)?)),
            GGUFType::I16 => Ok(GGUFValue::I16(read_i16(r)?)),
            GGUFType::I32 => Ok(GGUFValue::I32(read_i32(r)?)),
            GGUFType::I64 => Ok(GGUFValue::I64(read_i64(r)?)),

            GGUFType::U8 => Ok(GGUFValue::U8(read_u8(r)?)),
            GGUFType::U16 => Ok(GGUFValue::U16(read_u16(r)?)),
            GGUFType::U32 => Ok(GGUFValue::U32(read_u32(r)?)),
            GGUFType::U64 => Ok(GGUFValue::U64(read_u64(r)?)),

            GGUFType::F32 => Ok(GGUFValue::F32(read_f32(r)?)),
            GGUFType::F64 => Ok(GGUFValue::F64(read_f64(r)?)),

            GGUFType::String => Ok(GGUFValue::String(read_str(r)?)),
            GGUFType::Array => {
                let array_type = read_u32(r)?;
                let gguf_array_type = GGUFType::try_from(array_type)
                    .map_err(|_| Error::new(ErrorKind::InvalidData, "Invalid GGUF array type!"))?;
                let array_length = read_u64(r)?;

                let mut result: Vec<GGUFValue> = Vec::with_capacity(array_length as usize);
                for _ in 0..array_length {
                    let current_element = GGUFValue::read_from(r, gguf_array_type)?;
                    result.push(current_element);
                }
                Ok(GGUFValue::Array(result))
            }
        }
    }
}

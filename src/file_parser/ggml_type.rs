#![allow(non_camel_case_types, clippy::upper_case_acronyms)]

use num_enum::TryFromPrimitive;
use strum::Display;

#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive, Display)]
#[repr(u32)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
}

impl GGMLType {
    /// Elements per storage block, or `None` if unsupported.
    pub fn block_size(&self) -> Option<usize> {
        match self {
            GGMLType::F32 | GGMLType::BF16 => Some(1),
            GGMLType::Q8_0 | GGMLType::Q4_0 => Some(32),
            _ => None,
        }
    }

    /// Bytes per storage block, or `None` if unsupported.
    pub fn block_bytes(&self) -> Option<usize> {
        match self {
            GGMLType::F32 => Some(4),
            GGMLType::BF16 => Some(2),
            GGMLType::Q8_0 => Some(34), // f16 scale + 32 x i8
            GGMLType::Q4_0 => Some(18), // f16 scale + 16 bytes of packed nibbles
            _ => None,
        }
    }

    pub fn bytes_per_element(&self) -> usize {
        match (self.block_size(), self.block_bytes()) {
            (Some(1), Some(bytes)) => bytes,
            _ => panic!(
                "bytes_per_element only applies to scalar types, not {:?}",
                self
            ),
        }
    }
}

#![allow(dead_code)]

//! squirrel-rs: a small LLM GGUF inference engine

pub mod cli;
pub mod file_parser;
pub mod kernels;
pub mod models;
pub mod ops;
pub mod samplers;
pub mod text_generator;

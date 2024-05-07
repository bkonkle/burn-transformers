//! # Burn Transformers
#![forbid(unsafe_code)]

/// Models
pub mod models;

/// Pipelines
pub mod pipelines;

/// Datasets
pub mod datasets;

/// Utilities
pub mod utils;

/// CLI indexes and utilities
pub mod cli;

/// Error macros
#[macro_use]
extern crate anyhow;

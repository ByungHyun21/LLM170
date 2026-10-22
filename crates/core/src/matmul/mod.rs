
pub mod cpu;
pub mod weight;

pub use cpu::{greedy_from, matmul, matmul_batch, n_threads};
pub use weight::Weight;

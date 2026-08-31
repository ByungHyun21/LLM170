//! 스테이지 컨텍스트 — hc/gdn/qsa/moe/ple 스테이지의 공통 의존.
//!
//! 스테이지는 `Ctx`(모델 뷰 + 가속기)와 `&mut SeqState4`만 받는다 —
//! Engine4 내부 구조와 독립 (리팩토링 P1, 2026-09-01). CPU/GPU 동일 코드.

pub mod gdn;
pub mod hc;
pub mod moe;
pub mod ple;
pub mod qsa;

pub use gdn::gdn_layer;
pub use hc::{hc_mix, hc_mix_head};
pub use moe::moe_ffn;
pub use ple::{ple_block, ple_hash};
pub use qsa::qsa_layer;

use super::{Model4, Q4Error};
use crate::matmul::{matmul, matmul_batch, Accelerator, Weight};

/// 스테이지 실행 컨텍스트 — 모델 뷰(불변) + 가속기.
pub struct Ctx<'a> {
    pub model: &'a Model4,
    pub acc: Option<&'a dyn Accelerator>,
}

impl Ctx<'_> {
    pub fn mm(&self, x: &[f32], w: &Weight, out: &mut [f32]) -> Result<(), Q4Error> {
        match self.acc {
            Some(a) => a.matmul(x, w, out).map_err(Q4Error::Io),
            None => {
                matmul(x, w, out);
                Ok(())
            }
        }
    }

    /// 짝 디스패치 — 가중치마다 다른 1행 입력 (MoE down). 가속기 없으면 CPU 개별.
    pub fn mm_paired(
        &self,
        xs: &[Vec<f32>],
        ws: &[Weight],
        outs: &mut [Vec<f32>],
    ) -> Result<(), Q4Error> {
        match self.acc {
            Some(a) => a.matmul_paired(xs, ws, outs).map_err(Q4Error::Io),
            None => {
                for ((x, w), o) in xs.iter().zip(ws.iter()).zip(outs.iter_mut()) {
                    matmul(x, w, o);
                }
                Ok(())
            }
        }
    }

    /// 그룹 디스패치 — 동일 입력 복수 가중치. 가속기 없으면 CPU 개별.
    pub fn mm_group(
        &self,
        xs: &[Vec<f32>],
        ws: &[Weight],
        outs: &mut [Vec<Vec<f32>>],
    ) -> Result<(), Q4Error> {
        match self.acc {
            Some(a) => a.matmul_group(xs, ws, outs).map_err(Q4Error::Io),
            None => {
                for (w, o) in ws.iter().zip(outs.iter_mut()) {
                    matmul_batch(xs, w, o);
                }
                Ok(())
            }
        }
    }

    /// 배치 디스패치 — 가속기 없으면 CPU thread::scope 경로.
    pub fn mm_batch(
        &self,
        xs: &[Vec<f32>],
        w: &Weight,
        outs: &mut [Vec<f32>],
    ) -> Result<(), Q4Error> {
        match self.acc {
            Some(a) => a.matmul_batch(xs, w, outs).map_err(Q4Error::Io),
            None => {
                matmul_batch(xs, w, outs);
                Ok(())
            }
        }
    }
}

// SeqState4 재수출 — 스테이지 시그니처에서 super::SeqState4로 접근.

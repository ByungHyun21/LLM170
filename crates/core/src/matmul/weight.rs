use crate::wtype::WType;

/// mmap 상의 무게 텐서 참조.
#[derive(Clone, Copy)]
pub struct Weight<'a> {
    pub data: &'a [u8],
    /// W4A16 split 계열의 scale 버퍼(직접
    /// 로드). 무압축(F32/F16/Bf16)은 None. `W4a16G128Split`은 aux가 필수 계약이며
    /// Model::w가 보장한다(dequant_row 비경유 — cpu matmul 전용 arm).
    pub aux: Option<&'a [u8]>,
    pub ty: WType,
    pub n_in: u64,
    pub n_out: u64,
}

impl<'a> Weight<'a> {
    /// 텐서 전체를 f32 벡터로 펼침 (ne0-빠른 행 우선: 요소 (i, j) @ j*n_in+i).
    pub fn dequant_f32_vec(&self) -> Vec<f32> {
        let n = self.n_in * self.n_out;
        let rows = self.n_out;
        let mut v = vec![0.0f32; n as usize];
        for r in 0..rows {
            let s = r as usize * self.n_in as usize;
            crate::quant::dequant_row(
                self.ty,
                self.data,
                r,
                self.n_in,
                &mut v[s..s + self.n_in as usize],
            );
        }
        v
    }
}

use crate::wtype::WType;

#[derive(Clone, Copy)]
pub struct Weight<'a> {
    pub data: &'a [u8],
    pub aux: Option<&'a [u8]>,
    pub ty: WType,
    pub n_in: u64,
    pub n_out: u64,
    pub group: usize,
    pub scale_bf16: bool,
}

impl<'a> Weight<'a> {
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

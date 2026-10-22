use super::QwenEngine;

impl QwenEngine {
    pub fn spec_on(&self) -> bool {
        self.dec.spec_on()
    }

    pub fn enable_spec(&mut self) -> Result<(), String> {
        self.dec.enable_spec()
    }
    pub fn spec_verify(&mut self, slot: usize, rows: &[f32], t: usize) -> Result<Vec<u32>, String> {
        self.dec.spec_verify(slot, rows, t)
    }
    pub fn spec_verify_logits(
        &mut self,
        slot: usize,
        rows: &[f32],
        t: usize,
    ) -> Result<Vec<f32>, String> {
        let mut lg = self.dec.spec_verify_logits(slot, rows, t)?;
        self.softcap_row(&mut lg);
        Ok(lg)
    }

    pub fn spec_rollback(&mut self, slot: usize, keep: usize) -> Result<(), String> {
        self.dec.spec_rollback(slot, keep)
    }
    pub fn spec_rewind_pos(&mut self, slot: usize, n: u32) -> Result<(), String> {
        self.dec.spec_rewind_pos(slot, n)
    }
}

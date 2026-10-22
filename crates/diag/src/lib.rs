
pub mod dump;
pub mod fb;
pub mod flag;
pub mod fp;
pub mod span;
pub mod watchdog;

pub use fp::{fp_diff, fp_record};

pub const PROFILE_BUILD: bool = cfg!(feature = "profile");

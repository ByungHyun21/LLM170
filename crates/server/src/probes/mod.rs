use std::process::ExitCode;

pub(crate) fn arg_str(args: &[String], i: usize, d: &str) -> String {
    args.get(i).cloned().unwrap_or_else(|| d.into())
}


mod diag;
mod eval;
mod w4a16;

pub fn run(cmd: &str, args: &[String]) -> Option<ExitCode> {
    diag::try_run(cmd, args)
        .or_else(|| eval::try_run(cmd, args))
        .or_else(|| w4a16::try_run(cmd, args))
}

pub(crate) fn finish(r: Result<String, String>) -> ExitCode {
    match r {
        Ok(msg) => {
            println!("{msg}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

pub(crate) fn arg<'a, I: Iterator<Item = &'a String>>(
    it: &mut I,
    msg: &str,
) -> Result<&'a str, String> {
    it.next().map(|s| s.as_str()).ok_or_else(|| msg.to_string())
}

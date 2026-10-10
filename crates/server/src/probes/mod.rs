//! 원오프 프로브/체크 서브커맨드 — main.rs에서 이관.
//! 현재 w4a16·diag 로컬 프로브만 — 인자 파싱 + 본체 호출. 결론난 A/B
//! 하니스(batch-abtest·tree-test·q6k-abtest·exp-ab)는 2026-09-08 폐기.
use std::process::ExitCode;

/// 위치 인자 규약 헬퍼 — `args[i] | default` 파싱.
pub(crate) fn arg_str(args: &[String], i: usize, d: &str) -> String {
    args.get(i).cloned().unwrap_or_else(|| d.into())
}

// ## 프로브 하네스 저작 원칙 (A10 — 사고 4건+회귀루프 5건의 교훈)
//
// 검증 하네스 자체가 결함을 만든 클래스: ① 선행 단계의 공유 버퍼 오염
// (dah 행0 — "WMMA 행0 오염" 3일 오답의 진범) ② 하네스의 이중 상태 진입
// (sec9c 이중 frame_begin 스테일 판독) ③ 하네스 비정렬(hcmp 토큰/위치)
// ④ 하네스 형상 하드코딩(dsb 16행·ffn 부분적재 경계). 새 프로브 작성 시:
// 1. 선행 단계가 공유 버퍼(dah/dsb/dq2…)를 덮어쓰는지 먼저 점검 — 전용
//    버퍼(dah5 선례)로 분리.
// 2. 형상은 하드코딩 금지 — 전 선형 메타에서 자동 열거(t·krate·S0≠0
//    합성/실캡처). "모듈 완성" = 격리 검증 PASS + 전 형상 스윕 + 헤더
//    corr·속도 기입 + covered_by 등재(A22)까지. 미완 모듈을 다음 스텝에
//    끌고 가면 조립 단계에서 역행 루프(conv 오프셋·l2perm·nw127·KV ctx).
// 3. 상태 경로 버그는 S0=0 합성이 숨긴다 — 비영 초기 상태 필수.
// 4. 캡처-재생 3방향(커널 vs f32 미러 vs core 기준)이 국소화의 표준.
// 5. 종단 최종 상태가 버전 간 유일 불변량(형상이 다르면 중간값 비교 무의미).

mod diag;
mod w4a16;

/// 프로브 커맨드이면 실행해 Some(코드) 반환, 아니면 None.
pub fn run(cmd: &str, args: &[String]) -> Option<ExitCode> {
    diag::try_run(cmd, args).or_else(|| w4a16::try_run(cmd, args))
}

/// Result<String, String> → ExitCode 공통 변환(구 run() 테일).
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

/// [R4 2026-10-10] 플래그 값 요구 — 다음 토큰 또는 msg 오류(프로브 공용).
/// 종전 `it.next().ok_or("...")?` 반복을 한 곳으로.
pub(crate) fn arg<'a, I: Iterator<Item = &'a String>>(
    it: &mut I,
    msg: &str,
) -> Result<&'a str, String> {
    it.next().map(|s| s.as_str()).ok_or_else(|| msg.to_string())
}

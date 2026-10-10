use super::*;

impl CudaCtx {
    /// 그래프 캡처 개시(THREAD_LOCAL) — 이후 이 스트림의 발사가 그래프 노드로
    /// 기록된다(실행 아님). 캡처 중 동기 복사·alloc·sync는 금지.
    pub fn capture_begin(&self) -> Result<(), String> {
        self.prof.capturing.set(true);
        // SAFETY: stream은 create_stream이 설정한 유효 핸들.
        unsafe {
            let r = (self.drv.stream_begin_capture)(self.stream, 1);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuStreamBeginCapture: {}",
                    ffi::err_text(r)
                ));
            }
        }
        Ok(())
    }

    /// 캡처 종료 → 그래프 핸들. 실패 시에도 캡처 상태는 해제된다.
    pub fn capture_end(&self) -> Result<ffi::CUgraph, String> {
        self.prof.capturing.set(false);
        // SAFETY: 출력은 스택 로컬 — 그래프 핸들은 호출자 소유(graph_destroy).
        unsafe {
            let mut g: ffi::CUgraph = std::ptr::null_mut();
            let r = (self.drv.stream_end_capture)(self.stream, &mut g);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuStreamEndCapture: {}", ffi::err_text(r)));
            }
            Ok(g)
        }
    }

    /// 그래프 인스턴스화 — 실행 핸들(캡처 1회·replay N회).
    /// clippy allow — launch와 동일 판정(불투명 핸들).
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn graph_instantiate(&self, g: ffi::CUgraph) -> Result<ffi::CUgraphExec, String> {
        // SAFETY: g는 capture_end가 돌려준 유효 그래프.
        unsafe {
            let mut e: ffi::CUgraphExec = std::ptr::null_mut();
            let r = (self.drv.graph_instantiate)(&mut e, g, 0);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuGraphInstantiateWithFlags: {}",
                    ffi::err_text(r)
                ));
            }
            Ok(e)
        }
    }

    /// 그래프 실행 — 전 노드를 1회 launch로 제출.
    /// clippy allow — launch와 동일 판정(불투명 핸들).
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn graph_launch(&self, e: ffi::CUgraphExec) -> Result<(), String> {
        // SAFETY: e는 graph_instantiate가 돌려준 유효 실행 핸들.
        unsafe {
            let r = (self.drv.graph_launch)(e, self.stream);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuGraphLaunch: {}", ffi::err_text(r)));
            }
        }
        Ok(())
    }

    /// 그래프·실행 핸들 해제.
    /// clippy allow — launch와 동일 판정(불투명 핸들).
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn graph_destroy(&self, e: ffi::CUgraphExec, g: ffi::CUgraph) -> Result<(), String> {
        // SAFETY: 두 핸들 모두 본 모듈이 발급한 유효 핸들(중복 해제 금지 계약).
        unsafe {
            if !e.is_null() {
                let r = (self.drv.graph_exec_destroy)(e);
                if r != CUDA_SUCCESS {
                    return Err(format!("rawcuda: cuGraphExecDestroy: {}", ffi::err_text(r)));
                }
            }
            if !g.is_null() {
                let r = (self.drv.graph_destroy)(g);
                if r != CUDA_SUCCESS {
                    return Err(format!("rawcuda: cuGraphDestroy: {}", ffi::err_text(r)));
                }
            }
        }
        Ok(())
    }
}

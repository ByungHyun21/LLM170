use super::*;

impl CudaCtx {
    pub fn capture_begin(&self) -> Result<(), String> {
        self.prof.capturing.set(true);
        unsafe {
            let r = (self.drv.stream_begin_capture)(self.stream, 0);
            if r != CUDA_SUCCESS {
                return Err(format!(
                    "rawcuda: cuStreamBeginCapture: {}",
                    ffi::err_text(r)
                ));
            }
        }
        Ok(())
    }

    pub fn capture_end(&self) -> Result<ffi::CUgraph, String> {
        self.prof.capturing.set(false);
        unsafe {
            let mut g: ffi::CUgraph = std::ptr::null_mut();
            let r = (self.drv.stream_end_capture)(self.stream, &mut g);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuStreamEndCapture: {}", ffi::err_text(r)));
            }
            Ok(g)
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn graph_instantiate(&self, g: ffi::CUgraph) -> Result<ffi::CUgraphExec, String> {
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

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn graph_launch(&self, e: ffi::CUgraphExec) -> Result<(), String> {
        unsafe {
            let r = (self.drv.graph_launch)(e, self.stream);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuGraphLaunch: {}", ffi::err_text(r)));
            }
        }
        Ok(())
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn graph_upload(&self, e: ffi::CUgraphExec) -> Result<(), String> {
        unsafe {
            let r = (self.drv.graph_upload)(e, self.stream);
            if r != CUDA_SUCCESS {
                return Err(format!("rawcuda: cuGraphUpload: {}", ffi::err_text(r)));
            }
        }
        Ok(())
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn graph_destroy(&self, e: ffi::CUgraphExec, g: ffi::CUgraph) -> Result<(), String> {
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

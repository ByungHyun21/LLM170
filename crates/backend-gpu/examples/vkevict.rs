// vkevict — GTT 고립 메모리 회수 시도: 대량 할당→제출→해제 반복으로 TTM 축출/축수 유발.
use llm170_backend_gpu::rawvk::context::VkCtx;

fn main() {
    let gtt0 = std::fs::read_to_string("/sys/class/drm/card1/device/mem_info_gtt_used")
        .or_else(|_| std::fs::read_to_string("/sys/class/drm/card0/device/mem_info_gtt_used"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    println!("GTT before: {:?}", gtt0);
    let mut ctx = match VkCtx::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ctx: {e}");
            std::process::exit(1);
        }
    };
    // 4GB 청크로 12GB 확보 시도 (host-visible = GTT)
    let mut keep = Vec::new();
    for i in 0..3 {
        match ctx.alloc(4 << 30) {
            Ok(b) => {
                println!("alloc 4GB #{} ok", i + 1);
                keep.push(b);
            }
            Err(e) => {
                println!("alloc 4GB #{} failed: {}", i + 1, e);
                break;
            }
        }
    }
    drop(keep);
    std::thread::sleep(std::time::Duration::from_secs(2));
    let gtt1 = std::fs::read_to_string("/sys/class/drm/card1/device/mem_info_gtt_used")
        .or_else(|_| std::fs::read_to_string("/sys/class/drm/card0/device/mem_info_gtt_used"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    println!("GTT after: {:?}", gtt1);
}

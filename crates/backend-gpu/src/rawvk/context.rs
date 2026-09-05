//! Vulkan 컨텍스트 — ash 기반 순수 Rust 실행기 (plans/12).
//! HIP과 병립: LLM170_GPU_RUNTIME=vulkan일 때만 사용.

use ash::khr;
use ash::vk;

#[derive(Clone)]
pub struct VkBuf {
    pub buf: vk::Buffer,
    pub ptr: *mut u8,
    pub bytes: usize,
    pub mem: vk::DeviceMemory,
}

pub struct VkCtx {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub qf: u32,
    pub pool: vk::CommandPool,
    pub cmdbuf: vk::CommandBuffer,
    pub cmdbuf2: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub coop_matrix: bool,
    pub coop_f16_f32: bool,
    pub max_ssbo: usize,
    pub mem_ty: u32,
    /// GTT(캐시 host-visible) 타입 — 스크래치용.
    pub mem_ty_host: u32,
    /// 배치 모드 — run()은 녹화만 하고 end_batch에서 일괄 제출 (plans/19).
    pub batching: std::sync::atomic::AtomicBool,
    /// 배치용 per-run 디스크립터 세트 풀 (세트 재사용 하저드 회피 — RCA 2026-09-05:
    /// 녹화된 커맨드가 같은 세트를 참조해 마지막 바인딩으로 전부 덮어씀).
    pub batch_dsl: std::cell::Cell<Option<(vk::DescriptorSetLayout, vk::DescriptorPool)>>,
    pub batch_pool: std::cell::Cell<Option<(vk::DescriptorSetLayout, vk::DescriptorPool)>>,
    pub batch_sets: std::cell::RefCell<Vec<vk::DescriptorSet>>,
}

unsafe impl Send for VkCtx {}
unsafe impl Sync for VkCtx {}

impl VkCtx {
    pub fn new() -> Result<Self, String> {
        unsafe {
            let entry = ash::Entry::load().map_err(|e| format!("libvulkan 로드: {e}"))?;
            let apps = vk::ApplicationInfo::default()
                .application_name(c"llm170-vk")
                .api_version(vk::make_api_version(0, 1, 3, 0));
            let ci = vk::InstanceCreateInfo::default().application_info(&apps);
            let instance = entry
                .create_instance(&ci, None)
                .map_err(|e| format!("인스턴스: {e:?}"))?;

            let pds = instance
                .enumerate_physical_devices()
                .map_err(|e| format!("물리장치: {e:?}"))?;
            let (physical, props) = pds
                .into_iter()
                .map(|p| (p, instance.get_physical_device_properties(p)))
                .find(|(_, pr)| pr.vendor_id == 0x1002)
                .ok_or("AMD Vulkan 디바이스 없음")?;

            // coop matrix 역량 (f16×f16→f32, subgroup 스코프, M>=16, K=16)
            let mut coop_matrix = false;
            let mut coop_f16_f32 = false;
            let exts = instance
                .enumerate_device_extension_properties(physical)
                .map_err(|e| format!("확장: {e:?}"))?;
            if exts
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(khr::cooperative_matrix::NAME))
            {
                coop_matrix = true;
                let cmprops = khr::cooperative_matrix::Instance::new(&entry, &instance);
                let types = cmprops
                    .get_physical_device_cooperative_matrix_properties(physical)
                    .map_err(|e| format!("coop props: {e:?}"))?;
                coop_f16_f32 = types.iter().any(|t| {
                    t.scope == vk::ScopeKHR::SUBGROUP
                        && t.a_type == vk::ComponentTypeKHR::FLOAT16
                        && t.b_type == vk::ComponentTypeKHR::FLOAT16
                        && t.result_type == vk::ComponentTypeKHR::FLOAT32
                        && t.k_size == 16
                        && t.m_size >= 16
                });
            }

            let qfams = instance.get_physical_device_queue_family_properties(physical);
            let qf = qfams
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or("컴퓨트 큐 없음")? as u32;

            // f16/16bit-storage/subgroup-extended-types는 1.2 코어 승격 — Vulkan 1.3 디바이스에서 기본.
            let mut dev_ext: Vec<*const std::ffi::c_char> = Vec::new();
            if coop_matrix {
                dev_ext.push(khr::cooperative_matrix::NAME.as_ptr());
            }
            let mut v11 = vk::PhysicalDeviceVulkan11Features::default()
                .storage_buffer16_bit_access(true)
                .shader_draw_parameters(true);
            let mut coopfeat = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
                .cooperative_matrix(true);
            let mut feats = vk::PhysicalDeviceFeatures2::default().push_next(&mut v11);
            if coop_matrix {
                feats = feats.push_next(&mut coopfeat);
            }
            let qci = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(qf)
                .queue_priorities(&[1.0f32])];
            let dci = vk::DeviceCreateInfo::default()
                .queue_create_infos(&qci)
                .enabled_extension_names(&dev_ext)
                .push_next(&mut feats);
            let device = instance
                .create_device(physical, &dci, None)
                .map_err(|e| format!("디바이스: {e:?}"))?;
            let queue = device.get_device_queue(qf, 0);

            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(qf)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(|e| format!("커맨드 풀: {e:?}"))?;
            let cbs = device
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(2))
                .map_err(|e| format!("커맨드 버퍼: {e:?}"))?;
            let cmdbuf = cbs[0];
            let cmdbuf2 = cbs[1];
            let fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| format!("펜스: {e:?}"))?;

            // 메모리 타입: DEVICE_LOCAL|HOST_VISIBLE 우선 (APU 대형 캐브아웃 힙 — RADV
            // STRIX_HALO heap1 74GB). GTT 힙(heap0)은 커널 GTT 상한(15.5GB) 미만만 핀 가능해
            // 16GB+ 가중치는 submit 시 vm_validate 실패 — 캐브아웃 배치로 회피.
            let mprops = instance.get_physical_device_memory_properties(physical);
            let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let ty = (0..mprops.memory_type_count as usize)
                .find(|&i| {
                    mprops.memory_types[i].property_flags
                        .contains(hv | vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .or_else(|| {
                    (0..mprops.memory_type_count as usize).find(|&i| {
                        mprops.memory_types[i].property_flags.contains(hv)
                    })
                })
                .ok_or("host-visible coherent 메모리 타입 없음")? as u32;
            let ty_host = (0..mprops.memory_type_count as usize)
                .find(|&i| {
                    let f = mprops.memory_types[i].property_flags;
                    f.contains(hv)
                        && !f.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
                .unwrap_or(ty as usize) as u32;

            Ok(Self {
                entry,
                instance,
                physical,
                device,
                queue,
                qf,
                pool,
                cmdbuf,
                cmdbuf2,
                fence,
                coop_matrix,
                coop_f16_f32,
                max_ssbo: props.limits.max_storage_buffer_range as usize,
                mem_ty: ty,
                mem_ty_host: ty_host,
                batching: std::sync::atomic::AtomicBool::new(false),
                batch_dsl: std::cell::Cell::new(None),
                batch_pool: std::cell::Cell::new(None),
                batch_sets: std::cell::RefCell::new(Vec::new()),
            })
        }
    }

    /// 매핑 해제 (가중치 업로드 후) — WC 캐브아웃 매핑이 열려 있으면 펜스 대기가
    /// 전체 플러시되어 op당 동기화 비용 폭증 (tg 1.8 실측).
    pub fn unmap(&self, b: &mut VkBuf) -> Result<(), String> {
        unsafe {
            self.device.unmap_memory(b.mem);
            b.ptr = std::ptr::null_mut();
        }
        Ok(())
    }

    /// 배치용 fresh descriptor set — 전용 대형 풀 (op당 1세트, 제출 후 전량 해제).
    pub fn fresh_ds(&mut self, n_buf: u32) -> Result<vk::DescriptorSet, String> {
        unsafe {
            // 전용 배치 풀 지연 생성 (세트 256·버퍼 12×256)
            if self.batch_pool.get().is_none() {
                let dsl = self
                    .batch_dsl
                    .get()
                    .map(|(d, _)| d)
                    .ok_or("fresh_ds: 배치 컨텍스트 없음")?;
                let pool_sizes = [vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(12 * 4096)];
                let pool = self
                    .device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .max_sets(4096)
                            .pool_sizes(&pool_sizes)
                            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET),
                        None,
                    )
                    .map_err(|e| format!("배치 풀: {e:?}"))?;
                self.batch_pool.set(Some((dsl, pool)));
            }
            let (_, pool) = self
                .batch_pool
                .get()
                .ok_or("fresh_ds: 배치 풀 없음")?;
            let sets = self
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&[self.batch_dsl.get().unwrap().0]),
                )
                .map_err(|e| format!("세트 할당: {e:?}"))?;
            self.batch_sets.borrow_mut().push(sets[0]);
            Ok(sets[0])
        }
    }

    /// 배치 시작 — 이후 run()은 cmdbuf2에 녹화만.
    pub fn begin_batch(&mut self) -> Result<(), String> {
        unsafe {
            self.device
                .reset_command_buffer(
                    self.cmdbuf2,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .map_err(|e| format!("리셋2: {e:?}"))?;
            self.device
                .begin_command_buffer(
                    self.cmdbuf2,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| format!("시작2: {e:?}"))?;
        }
        self.batching.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// 비배치 모드 플러시 — cmdbuf2 즉시 제출·대기 (NOBATCH 경로).
    pub fn flush2(&mut self) -> Result<(), String> {
        unsafe {
            self.device.end_command_buffer(self.cmdbuf2).map_err(|e| format!("종료2: {e:?}"))?;
            self.device.reset_fences(&[self.fence]).map_err(|e| format!("펜스: {e:?}"))?;
            let cbs = [self.cmdbuf2];
            let si = vk::SubmitInfo::default().command_buffers(&cbs);
            self.device.queue_submit(self.queue, &[si], self.fence).map_err(|e| format!("제출2: {e:?}"))?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX).map_err(|e| format!("대기2: {e:?}"))?;
            if let Some((_, pool)) = self.batch_pool.get() {
                let sets = std::mem::take(&mut *self.batch_sets.borrow_mut());
                if !sets.is_empty() {
                    self.device.free_descriptor_sets(pool, &sets);
                }
            }
        }
        Ok(())
    }

    /// 배치 종료 — 일괄 제출·대기.
    pub fn end_batch_wait(&mut self) -> Result<(), String> {
        self.batching.store(false, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            self.device
                .end_command_buffer(self.cmdbuf2)
                .map_err(|e| format!("종료2: {e:?}"))?;
            self.device
                .reset_fences(&[self.fence])
                .map_err(|e| format!("펜스 리셋: {e:?}"))?;
            let cbs = [self.cmdbuf2];
            let si = vk::SubmitInfo::default().command_buffers(&cbs);
            self.device
                .queue_submit(self.queue, &[si], self.fence)
                .map_err(|e| format!("제출2: {e:?}"))?;
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| format!("대기2: {e:?}"))?;
            // 배치 세트 전량 해제 (풀 재사용)
            if let Some((_, pool)) = self.batch_pool.get() {
                let sets = std::mem::take(&mut *self.batch_sets.borrow_mut());
                if !sets.is_empty() {
                    self.device.free_descriptor_sets(pool, &sets);
                }
            }
        }
        Ok(())
    }

    /// 스크래치용 할당 — GTT(캐시 호스트 가시) 타입: CPU 쓰기 빠름. 캐브아웃 타입은
    /// 매핑이 WC여서 토큰별 활성화 왕복이 느림 (실측 tg 1.9 vs 10.4).
    pub fn alloc_host(&mut self, bytes: usize) -> Result<VkBuf, String> {
        let saved = self.mem_ty;
        self.mem_ty = self.mem_ty_host;
        let r = self.alloc(bytes);
        self.mem_ty = saved;
        r
    }

    /// 버퍼 할당 — 자체 디바이스 메모리 + 매핑 (호스트 포인터 동반).
    /// bytes는 max_ssbo 이하 권장 (초과 시 호출부에서 청크 분할).
    pub fn alloc(&mut self, bytes: usize) -> Result<VkBuf, String> {
        unsafe {
            let bci = vk::BufferCreateInfo::default()
                .size(bytes as u64)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buf = self
                .device
                .create_buffer(&bci, None)
                .map_err(|e| format!("버퍼: {e:?}"))?;
            let ai = vk::MemoryAllocateInfo::default()
                .allocation_size(bytes as u64)
                .memory_type_index(self.mem_ty);
            let mem = self
                .device
                .allocate_memory(&ai, None)
                .map_err(|e| format!("할당({bytes}): {e:?}"))?;
            self.device
                .bind_buffer_memory(buf, mem, 0)
                .map_err(|e| format!("바인드: {e:?}"))?;
            let ptr = self
                .device
                .map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("맵: {e:?}"))? as *mut u8;
            Ok(VkBuf { buf, ptr, bytes, mem })
        }
    }


    /// 파이프라인 생성 (push constant + N개 SSBO).
    /// 스펙상수 지원 파이프라인 (부록87 — llama mul_mm 로드용).
    /// spec[i] → constantID i (llama create_pipeline 규약).
    pub fn pipeline_spec(
        &self,
        spv: &[u8],
        n_buf: u32,
        push_bytes: u32,
        spec: &[u32],
    ) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::DescriptorPool, vk::DescriptorSet, vk::Pipeline), String> {
        self.pipeline_spec_fg(spv, n_buf, push_bytes, spec, false)
    }

    /// full_subgroups 플래그 판 (coopmat 파이프라인용 — 부록87).
    pub fn pipeline_spec_fg(
        &self,
        spv: &[u8],
        n_buf: u32,
        push_bytes: u32,
        spec: &[u32],
        full_subgroups: bool,
    ) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::DescriptorPool, vk::DescriptorSet, vk::Pipeline), String> {
        unsafe {
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n_buf)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let dsl = self
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|e| format!("DSL: {e:?}"))?;
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(push_bytes)];
            let pl = self
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&[dsl])
                        .push_constant_ranges(&ranges),
                    None,
                )
                .map_err(|e| format!("레이아웃: {e:?}"))?;
            let code: Vec<u32> = spv
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let smci = vk::ShaderModuleCreateInfo::default().code(&code);
            let sm = self
                .device
                .create_shader_module(&smci, None)
                .map_err(|e| format!("셰이더 모듈: {e:?}"))?;
            let entries: Vec<vk::SpecializationMapEntry> = spec
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    vk::SpecializationMapEntry::default()
                        .constant_id(i as u32)
                        .offset((i * 4) as u32)
                        .size(4)
                })
                .collect();
            let spec_bytes: Vec<u8> = spec.iter().flat_map(|v| v.to_le_bytes()).collect();
            let sinfo = vk::SpecializationInfo::default()
                .map_entries(&entries)
                .data(&spec_bytes);
            let mut stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(sm)
                .name(c"main")
                .specialization_info(&sinfo);
            if full_subgroups {
                stage = stage.flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS);
            }
            let pci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(pl);
            let pipe = self
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pci], None)
                .map_err(|(_, e)| format!("파이프라인(스펙): {e:?}"))?[0];
            self.device.destroy_shader_module(sm, None);
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(n_buf)];
            let dp = self
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&pool_sizes)
                        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET),
                    None,
                )
                .map_err(|e| format!("디스크립터 풀: {e:?}"))?;
            let ds = self
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(dp)
                        .set_layouts(&[dsl]),
                )
                .map_err(|e| format!("디스크립터 셋: {e:?}"))?[0];
            Ok((dsl, pl, dp, ds, pipe))
        }
    }

    pub fn pipeline(
        &self,
        spv: &[u8],
        n_buf: u32,
        push_bytes: u32,
    ) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::DescriptorPool, vk::DescriptorSet, vk::Pipeline), String> {
        unsafe {
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n_buf)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let dsl = self
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(|e| format!("DSL: {e:?}"))?;
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(push_bytes)];
            let pl = self
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&[dsl])
                        .push_constant_ranges(if push_bytes > 0 { &ranges } else { &[] }),
                    None,
                )
                .map_err(|e| format!("레이아웃: {e:?}"))?;
            let code: Vec<u32> = spv
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let smci = vk::ShaderModuleCreateInfo::default().code(&code);
            let sm = self
                .device
                .create_shader_module(&smci, None)
                .map_err(|e| format!("셰이더 모듈: {e:?}"))?;
            let pci = vk::ComputePipelineCreateInfo::default().stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(sm)
                    .name(c"main"),
            ).layout(pl);
            let pipe = self
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pci], None)
                .map_err(|(_, e)| format!("파이프라인: {e:?}"))?[0];
            self.device.destroy_shader_module(sm, None);
            let pool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(n_buf)];
            let dp = self
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&pool_sizes)
                        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET),
                    None,
                )
                .map_err(|e| format!("디스크립터 풀: {e:?}"))?;
            let ds = self
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(dp)
                        .set_layouts(&[dsl]),
                )
                .map_err(|e| format!("디스크립터 셋: {e:?}"))?[0];
            Ok((dsl, pl, dp, ds, pipe))
        }
    }

    /// SSBO 바인딩 갱신.
    pub fn bind_bufs(&self, ds: vk::DescriptorSet, bufs: &[vk::Buffer]) {
        unsafe {
            let infos: Vec<vk::DescriptorBufferInfo> = bufs
                .iter()
                .map(|&b| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(b)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();
            self.device.update_descriptor_sets(&writes, &[]);
        }
    }

    /// 커맨드 녹화·제출·동기 대기.
    pub fn run(
        &self,
        pl: vk::PipelineLayout,
        ds: vk::DescriptorSet,
        pipe: vk::Pipeline,
        push: &[u8],
        gx: u32,
        gy: u32,
        gz: u32,
    ) -> Result<(), String> {
        unsafe {
            if !self.batching.load(std::sync::atomic::Ordering::Relaxed) {
                self.device
                    .reset_command_buffer(
                        self.cmdbuf,
                        vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                    )
                    .map_err(|e| format!("리셋: {e:?}"))?;
                self.device
                    .begin_command_buffer(
                        self.cmdbuf,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .map_err(|e| format!("시작: {e:?}"))?;
            }
            let cb = if self.batching.load(std::sync::atomic::Ordering::Relaxed) {
                self.cmdbuf2
            } else {
                self.cmdbuf
            };
            self.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe);
            self.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                pl,
                0,
                &[ds],
                &[],
            );
            if !push.is_empty() {
                self.device.cmd_push_constants(
                    cb,
                    pl,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                );
            }
            self.device.cmd_dispatch(cb, gx, gy, gz);
            // 배치 내 write→read 가시성 배리어 (비배칭 submit+wait의 암시 동기 대체)
            if self.batching.load(std::sync::atomic::Ordering::Relaxed) {
                let bar = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ);
                self.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[bar],
                    &[],
                    &[],
                );
            }
            if !self.batching.load(std::sync::atomic::Ordering::Relaxed) {
                self.device
                    .end_command_buffer(self.cmdbuf)
                    .map_err(|e| format!("종료: {e:?}"))?;
            }
            self.device
                .reset_fences(&[self.fence])
                .map_err(|e| format!("펜스 리셋: {e:?}"))?;
            let cbs = [self.cmdbuf];
            let si = vk::SubmitInfo::default().command_buffers(&cbs);
            if self.batching.load(std::sync::atomic::Ordering::Relaxed) {
                // 배치 모드: 녹화만. 커맨드 버퍼는 ONE_TIME_SUBMIT이지만 end_batch에서
                // 제출 전까지 재사용 불가 — 배치 세션 동안 별도 2차 버퍼 사용.
                return Ok(());
            }
            self.device
                .queue_submit(self.queue, &[si], self.fence)
                .map_err(|e| format!("제출: {e:?}"))?;
            self.device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|e| format!("대기: {e:?}"))?;
            Ok(())
        }
    }
}

impl Drop for VkCtx {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

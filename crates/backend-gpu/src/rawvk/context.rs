//! Vulkan 컨텍스트 — ash 기반 순수 Rust 실행기 (plans/12).
//! HIP과 병립: LLM170_GPU_RUNTIME=vulkan일 때만 사용.

use ash::khr;
use ash::vk;

pub struct VkBuf {
    pub buf: vk::Buffer,
    pub ptr: *mut u8,
    pub bytes: usize,
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
    pub fence: vk::Fence,
    pub coop_matrix: bool,
    pub coop_f16_f32: bool,
    mem: vk::DeviceMemory,
    base: *mut u8,
    cap: usize,
    offset: usize,
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
            let (physical, _props) = pds
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
            let cmdbuf = device
                .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))
                .map_err(|e| format!("커맨드 버퍼: {e:?}"))?[0];
            let fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| format!("펜스: {e:?}"))?;

            // 단일 bump 힙 (host-visible coherent — APU 단일 메모리)
            let cap = 256usize << 20;
            let mprops = instance.get_physical_device_memory_properties(physical);
            let ty = (0..mprops.memory_type_count as usize)
                .find(|&i| {
                    mprops.memory_types[i]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
                })
                .ok_or("host-visible coherent 메모리 타입 없음")? as u32;
            let ai = vk::MemoryAllocateInfo::default()
                .allocation_size(cap as u64)
                .memory_type_index(ty);
            let mem = device
                .allocate_memory(&ai, None)
                .map_err(|e| format!("할당: {e:?}"))?;
            let base = device
                .map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("맵: {e:?}"))? as *mut u8;

            Ok(Self {
                entry,
                instance,
                physical,
                device,
                queue,
                qf,
                pool,
                cmdbuf,
                fence,
                coop_matrix,
                coop_f16_f32,
                mem,
                base,
                cap,
                offset: 0,
            })
        }
    }

    /// 버퍼 할당 (bump, host 포인터 동반 반환).
    pub fn alloc(&mut self, bytes: usize) -> Result<VkBuf, String> {
        unsafe {
            let off = (self.offset + 255) & !255;
            if off + bytes > self.cap {
                return Err(format!(
                    "VkCtx 풀 부족: need {bytes}, left {}",
                    self.cap - off
                ));
            }
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
            self.device
                .bind_buffer_memory(buf, self.mem, off as u64)
                .map_err(|e| format!("바인드: {e:?}"))?;
            self.offset = off + bytes;
            Ok(VkBuf {
                buf,
                ptr: self.base.add(off),
                bytes,
            })
        }
    }

    /// 힙 rewind — 버퍼는 파괴자 책임 하 별도 관리.
    pub fn rewind(&mut self) {
        self.offset = 0;
    }

    /// 파이프라인 생성 (push constant + N개 SSBO).
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
            self.device.cmd_bind_pipeline(self.cmdbuf, vk::PipelineBindPoint::COMPUTE, pipe);
            self.device.cmd_bind_descriptor_sets(
                self.cmdbuf,
                vk::PipelineBindPoint::COMPUTE,
                pl,
                0,
                &[ds],
                &[],
            );
            if !push.is_empty() {
                self.device.cmd_push_constants(
                    self.cmdbuf,
                    pl,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push,
                );
            }
            self.device.cmd_dispatch(self.cmdbuf, gx, gy, gz);
            self.device
                .end_command_buffer(self.cmdbuf)
                .map_err(|e| format!("종료: {e:?}"))?;
            self.device
                .reset_fences(&[self.fence])
                .map_err(|e| format!("펜스 리셋: {e:?}"))?;
            let cbs = [self.cmdbuf];
            let si = vk::SubmitInfo::default().command_buffers(&cbs);
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

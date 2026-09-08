// ─── np×spec (plans/18): 행별 포인터/위치 색인 변형 — seq-major 그룹 전제 ───




extern "C" __global__ void gdn_conv_t2_ms(const float* qkv, const float* cw, float* const* states,
                                          float* out, int ch, int k, int t,
                                          const int* row_seq, const int* seg_start) {
    // plans/28 RCA: 포인터 테이블은 "행 → 자기 seq 상태" (ms_conv_ptr가 행별로
    // 업로드) — states[row_seq[ti]]는 이중 색인(시퀀스 id를 행 인덱스로 오용)으로
    // 그룹>0가 전부 seq0의 링을 읽었다. states[ti]가 정답.
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = blockIdx.y;
    if (c >= ch || ti >= t) return;
    const float* row = qkv + (size_t)ti * ch;
    float* st = states[ti];
    int ss = seg_start[ti];
    float sum = cw[c * k + (k - 1)] * row[c];
    for (int j = 0; j < k - 1; j++) {
        int pos = ti - (k - 1) + j;
        float xv = pos >= ss ? qkv[(size_t)pos * ch + c]
                             : st[(pos + (k - 1) - ss) * ch + c];
        sum += cw[c * k + j] * xv;
    }
    float oc = sum / (1.0f + exp_cr(-sum));
    out[(size_t)ti * ch + c] = oc;
}

extern "C" __global__ void gdn_conv_t2_ms_f32(const float* qkv, const float* cw, float* const* states,
                                              float* out, int ch, int k, int t,
                                              const int* row_seq, const int* seg_start) {
    // gdn_conv_t2_ms의 f32-silu 판 (트렁크 gdn_conv_t2_f32와 동일 산술 — 기본 경로).
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = blockIdx.y;
    if (c >= ch || ti >= t) return;
    const float* row = qkv + (size_t)ti * ch;
    float* st = states[ti];
    int ss = seg_start[ti];
    float sum = cw[c * k + (k - 1)] * row[c];
    for (int j = 0; j < k - 1; j++) {
        int pos = ti - (k - 1) + j;
        float xv = pos >= ss ? qkv[(size_t)pos * ch + c]
                             : st[(pos + (k - 1) - ss) * ch + c];
        sum += cw[c * k + j] * xv;
    }
    float oc = sum / (1.0f + __expf(-sum));
    out[(size_t)ti * ch + c] = oc;
}

extern "C" __global__ void gdn_conv_state_ms(const float* qkv, float* const* states, int ch,
                                             int k, int t, const int* row_seq, const int* seg_end) {
    // 그리드 (t, ch/64): ti = blockIdx.x, c = blockIdx.y 열 블록 — 세그별 꼬리 기입
    // plans/28: states[ti] (행 색인) — 위와 동일 이중 색인 버그 수정.
    int ti = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= ch || ti >= t) return;
    int se = seg_end[ti];
    int first_tail = se - (k - 1);
    if (ti < first_tail || ti >= se) return;
    int j = ti - first_tail;
    states[ti][j * ch + c] = qkv[(size_t)ti * ch + c];
}



// ── [S3-1 2026-10-11] 임베딩 행 gather(GPU) — 프리필 행 조립 대체 ──
// 호스트 embed_row(행당 mmap 디코드+Vec 할당+h2d 20KB)를 토큰 id h2d(4B/행)
// + 본 커널로 대체한다. 계약: h2f/b2f = Rust half_to_f32/bf16_to_f32와
// 비트동일(decode_f32가 쓰는 동일 함수) — 값 불변(골든 재판정).
#include "cast_common.cuh"
#include "g4_common.cuh" // b2f

extern "C" __global__ void w4a16_embed_gather(
    const unsigned short* __restrict__ emb, // [n][k] f16/bf16 원본
    const unsigned* __restrict__ toks,      // [t] u32 토큰 id(< n — 호스트 계약)
    float* __restrict__ out,                // [t][k] f32
    int t, int k, int n, int bf16)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t * k) {
        return;
    }
    const int r = i / k;
    const int c = i - r * k;
    const unsigned tok = toks[r];
    if (tok >= (unsigned)n) {
        out[i] = 0.0f; // 방어 — 호스트가 사전 거부(계약)
        return;
    }
    const unsigned short v = emb[(size_t)tok * (size_t)k + (size_t)c];
    out[i] = bf16 ? b2f(v) : h2f(v);
}

// ── [MTP-CHAIN 2026-10-12 §3.3 레버 ②] MTP 장치 체인 부기 ──
// 초안 재귀(mtp_recur_step)의 호스트 왕복(argmax 4B readback → 임베딩 행
// 조립·h2d → dpp 갱신)을 없애는 장치측 연결부: 이번 스텝이 소비할 토큰
// (직전 argmax = tok[0])을 로그에 남기고 카운터를 전진한 뒤 슬롯 pos
// (dpp)를 +1 한다. 같은 그래프 안의 w4a16_embed_gather가 tok[0]의 임베딩
// 행을 ptr0로 조립하고, 어텐션 커널은 dpp[slot]에서 pos를 판독한다(커널
// 인자 아님 — 캡처 안전). 산술 없음(복사·정수) — fmad 정책 무관.
// 1블록 32스레드(thread 0만 동작). ctr·log는 리플레이 사이에 유지되는
// 장치 전역 상태(라운드 시작에 호스트가 ctr=0 h2d 1회).
extern "C" __global__ void w4a16_mtp_chain_step(
    const unsigned* __restrict__ tok, // [1] 직전 argmax(dbatch_am — 앵커가 시드)
    unsigned* __restrict__ log,       // [kmax] 초안 로그(소비 토큰 기록)
    unsigned* __restrict__ ctr,       // [1] 스텝 카운터
    unsigned* __restrict__ dpp,       // [n_slots] 슬롯 pos(+=1 — atomic)
    int slot, int kmax)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const unsigned c = *ctr;
    if (c < (unsigned)kmax) {
        log[c] = tok[0];
    }
    *ctr = c + 1u;
    atomicAdd(&dpp[slot], 1u);
}
